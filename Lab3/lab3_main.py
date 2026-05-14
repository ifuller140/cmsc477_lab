import cv2
import time
import os
import sys
import math
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pupil_apriltags
from ultralytics import YOLO
import robomaster
from robomaster import robot, camera

# ── constants ─────────────────────────────────────────────────────────────────

ROBOT_IP  = "192.168.50.116"
ROBOT_SN  = "3JKCH8800100VW"

ARENA_M   = 3.048    # 10 ft arena side length (m)
BOX_SIZE  = 0.26     # fabric box side (m)

# Plan-frame: origin = bottom-right corner, +x = west, +y = north
START_X   = 2.85
START_Y   = 0.15

SCAN_Y    = 0.6     # horizontal scan / transit line y
DOCK_X    = 0.4    # loading dock x (near west wall)

# Loading dock plot bounds (plan-frame)
DOCK_SW   = (0.25, 0.25)
DOCK_NE   = (1.22, 0.91)

# AprilTag IDs (tag36h11)
TAGS_SMALL_GOAL = frozenset({30, 41})
TAGS_LARGE_GOAL = frozenset({45, 19})
TAGS_RECHARGE   = frozenset({38, 34})
TAGS_KNOWN      = TAGS_SMALL_GOAL | TAGS_LARGE_GOAL | TAGS_RECHARGE

TAG_SIZE_M = 0.15

# Calibrated camera intrinsics (measured in Lab0)
CAM_FX   = 314.0
CAM_FY   = 314.0
CAM_CX   = 320.0
CAM_CY   = 180.0

FRAME_W  = 640
FRAME_H  = 360

# YOLO class indices  (lab3_data.yaml: nc=3, names=[small_lego, large_lego, box])
CLS_SMALL   = 0
CLS_LARGE   = 1
CLS_BOX     = 2
CONF_THRESH = 0.50

# Battery (software simulation per project spec)
BAT_START      = 60
BAT_COST_SMALL = 20
BAT_COST_LARGE = 40
BAT_WARN_SMALL = 25
BAT_WARN_LARGE = 45

# Chassis control
YAW_KP     = 0.15
YAW_MAX    = 25.0
STRAFE_KP  = 0.002
STRAFE_MAX = 0.12   # m/s cap for visual-servo strafe (drive_speed)
CRAB_SPEED = 0.4   # m/s for chassis-move strafe (crab_walk)
FWD_MAX    = 0.15
FWD_SPEED  = 0.5
SLOW_SPEED = 0.1
TURN_SPEED = 20.0

# Strafe scan
STRAFE_SCAN_SPD = 0.8   # m/s horizontal sweep speed
X_RECORD_TOL    = 15     # px: frame-center tolerance when logging a tag

# Lego approach thresholds
APPROACH_STOP_BOT = 250   # coarse stop: bbox bottom >= this
GRAB_TOP_SMALL    = 225   # fine grab: bbox top > this (small lego)
GRAB_TOP_LARGE    = 120   # fine grab: bbox top > this (large lego)

# Goal delivery
GOAL_APPROACH_M = 0.10   # stop this far from goal tag (m)

# Recharge
RECHARGE_CLOSE_M = 0.08
RECHARGE_HOLD_S  = 5.2

# Arm (x mm forward from body, y mm vertical)
ARM_HOME    = (185, 40)
ARM_CARRY   = (185, 40)
ARM_PICKUP  = (185, -50)
GRIPPER_POWER = 50
GRIPPER_HOLD  = 1.0

# ── global state ───────────────────────────────────────────────────────────────

current_theta = 0.0   # degrees from north: 0 = north, 180 = south
_arm_pos      = None  # last known arm (x, y)
_yolo_model   = None  # lazy-loaded only when picking up legos


# ── odometry tracker ──────────────────────────────────────────────────────────

class OdometryTracker:
    """
    Subscribes to chassis position + attitude callbacks.
    cs=0 gives displacement in the robot's starting ground frame:
      self.x = metres driven forward/north from start
      self.y = metres strafed left/west  from start
    """
    def __init__(self):
        self.x = self.y = self.yaw = 0.0
        self._lock = threading.Lock()

    def _pos_cb(self, pos_info):
        x, y, _ = pos_info
        with self._lock:
            self.x = x
            self.y = y

    def _att_cb(self, att_info):
        yaw, _, _ = att_info
        with self._lock:
            self.yaw = yaw

    def subscribe(self, ep_chassis):
        ep_chassis.sub_position(cs=0, freq=20, callback=self._pos_cb)
        ep_chassis.sub_attitude(freq=20, callback=self._att_cb)

    def unsubscribe(self, ep_chassis):
        for fn in (ep_chassis.unsub_position, ep_chassis.unsub_attitude):
            try:
                fn()
            except Exception:
                pass

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.yaw


# ── battery manager ───────────────────────────────────────────────────────────

class BatteryManager:
    def __init__(self):
        self.level = BAT_START
        print(f"[battery] start {self.level}%")

    def needs_recharge(self, brick_type='small'):
        warn = BAT_WARN_LARGE if brick_type == 'large' else BAT_WARN_SMALL
        return self.level < warn

    def consume(self, brick_type):
        cost = BAT_COST_LARGE if brick_type == 'large' else BAT_COST_SMALL
        self.level = max(0, self.level - cost)
        print(f"[battery] {brick_type} delivered: -{cost}% → {self.level}%")

    def recharge(self):
        self.level = 100
        print("[battery] recharged → 100%")


# ── perception helpers ────────────────────────────────────────────────────────

def get_frame(ep_camera, retries=3):
    for _ in range(retries):
        try:
            f = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if f is not None:
                return f
        except Exception as e:
            print(f"[cam] {e}")
        time.sleep(0.05)
    return None


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        print("Loading YOLO model (first lego pickup)...")
        w = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "runs/detect/lab3_gpu_train/weights/best.pt")
        if not os.path.exists(w):
            print(f"[error] weights not found: {w}")
            sys.exit(1)
        _yolo_model = YOLO(w)
        print("[yolo] model loaded")
    return _yolo_model


def run_yolo(frame):
    model = _load_yolo()
    out = {CLS_SMALL: [], CLS_LARGE: [], CLS_BOX: []}
    try:
        res = model.predict(source=frame, show=False, verbose=False)[0]
        if res.boxes is None:
            return out
        for b in res.boxes:
            if float(b.conf) < CONF_THRESH:
                continue
            cls = int(b.cls)
            if cls not in out:
                continue
            xyxy = b.xyxy.cpu().numpy().flatten()
            if len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = map(float, xyxy[:4])
            out[cls].append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
                'w': x2 - x1, 'h': y2 - y1,
                'area': (x2 - x1) * (y2 - y1),
            })
        for c in out:
            out[c].sort(key=lambda d: d['area'], reverse=True)
    except Exception as e:
        print(f"[yolo] {e}")
    return out


def detect_apriltags(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        raw = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[CAM_FX, CAM_FY, CAM_CX, CAM_CY],
            tag_size=TAG_SIZE_M,
        )
        results = []
        for tag in raw:
            t = tag.pose_t.flatten()
            results.append({
                'id':      tag.tag_id,
                'dist':    float(t[2]),
                'bearing': math.degrees(math.atan2(float(t[0]), float(t[2]))),
                'center':  tag.center,
                'corners': tag.corners,
                'pose_t':  t,
            })
        return results
    except Exception as e:
        print(f"[apriltag] {e}")
        return []


def draw_box(frame, b, color=(0, 255, 0), label=""):
    cv2.rectangle(frame, (int(b['x1']), int(b['y1'])),
                  (int(b['x2']), int(b['y2'])), color, 2)
    if label:
        cv2.putText(frame, label, (int(b['x1']), int(b['y1']) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


def draw_apriltag(frame, tag, color=(0, 0, 255), extra=""):
    """Draw tag corners polygon, X diagonals, ID and distance label."""
    corners = tag['corners']
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    c = corners.astype(np.int32)
    cv2.line(frame, tuple(c[0]), tuple(c[2]), color, 2)
    cv2.line(frame, tuple(c[1]), tuple(c[3]), color, 2)
    cx, cy = int(tag['center'][0]), int(tag['center'][1])
    label = f"id={tag['id']} {tag['dist']:.2f}m"
    if extra:
        label += f" {extra}"
    cv2.putText(frame, label, (cx - 50, cy - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def write_text(frame, text, row=30, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, row),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def show(frame, title="lab3"):
    cv2.imshow(title, frame)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


# ── chassis helpers ───────────────────────────────────────────────────────────

def drive(ep_chassis, x=0.0, y=0.0, z=0.0):
    ep_chassis.drive_speed(
        x=max(-FWD_MAX,    min(FWD_MAX,    x)),
        y=max(-STRAFE_MAX, min(STRAFE_MAX, y)),
        z=max(-YAW_MAX,    min(YAW_MAX,    z)),
    )


def stop(ep_chassis):
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)


def drive_straight(ep_chassis, dist_m, speed=None):
    if speed is None:
        speed = FWD_SPEED
    if abs(dist_m) < 0.005:
        return
    print(f"[drive] {dist_m:+.3f}m @ {speed:.2f}m/s")
    # Positive dist_m = north; move x-axis is southward so negate.
    ep_chassis.move(x=dist_m, y=0, z=0, xy_speed=abs(speed)).wait_for_completed()
    time.sleep(0.12)


def crab_walk(ep_chassis, dist_m, speed=None):
    """Strafe using chassis move command.
    Positive dist_m = west (increase plan_x); negative = east.
    drive_speed y<0 = west, so move y=-dist_m maps the same convention.
    Uses ep_chassis.move() for accuracy — no external odom needed."""
    if abs(dist_m) < 0.005:
        return
    max_spd = min(abs(speed), CRAB_SPEED) if speed is not None else CRAB_SPEED
    print(f"[crab] {dist_m:+.3f}m @ {max_spd:.2f}m/s")
    ep_chassis.move(x=0, y=dist_m, z=0, xy_speed=max_spd).wait_for_completed()
    time.sleep(0.12)


def rotate_by(ep_chassis, delta_deg):
    global current_theta
    if abs(delta_deg) < 0.3:
        return
    ep_chassis.move(x=0, y=0, z=delta_deg, z_speed=TURN_SPEED).wait_for_completed()
    current_theta = (current_theta + delta_deg) % 360
    time.sleep(0.15)


def rotate_to(ep_chassis, target_deg):
    delta = target_deg - current_theta
    while delta >  180: delta -= 360
    while delta < -180: delta += 360
    rotate_by(ep_chassis, delta)


# ── position helper ───────────────────────────────────────────────────────────

def get_plan_pos(odom):
    """
    Convert cs=0 odometry to plan-frame (plan_x, plan_y).
    Robot starts bottom-right, facing north (arm faces south toward loading dock).
    ep_chassis.move(x=+d) drives arm-first (south); north requires move(x=-d).
    Therefore odom.x is southward displacement: north travel → odom.x negative.

    plan_x = west  from start = START_X + odom.y   (drive_speed y<0 = west → odom.y positive)
    plan_y = north from start = START_Y - odom.x   (north → odom.x negative → -odom.x positive)
    """
    ox, oy, _ = odom.get_pose()
    return START_X + ox, START_Y - oy


# ── arm / gripper ─────────────────────────────────────────────────────────────

def arm_moveto(ep_arm, pos, label="", step=10):
    global _arm_pos
    x, y = pos
    print(f"[arm] {label} → ({x}, {y})")
    if _arm_pos is None:
        ep_arm.moveto(x=x, y=y).wait_for_completed()
        _arm_pos = (x, y)
        return
    cx, cy = _arm_pos
    dist = math.hypot(x - cx, y - cy)
    if dist < 1:
        return
    steps = max(1, int(dist / step))
    for i in range(1, steps + 1):
        ep_arm.moveto(
            x=int(cx + (x - cx) * i / steps),
            y=int(cy + (y - cy) * i / steps),
        ).wait_for_completed()
    _arm_pos = (x, y)


def gripper_close(ep_gripper):
    ep_gripper.close(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()


def gripper_open(ep_gripper):
    ep_gripper.open(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()


# ── robot setup / teardown ────────────────────────────────────────────────────

def setup_robot():
    print("Setting up AprilTag detector...")
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)

    print("Connecting to robot...")
    robomaster.config.ROBOT_IP_STR = ROBOT_IP
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn=ROBOT_SN)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    ep_chassis = ep_robot.chassis
    ep_arm     = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    ep_camera  = ep_robot.camera

    arm_moveto(ep_arm, ARM_HOME, "init")
    gripper_open(ep_gripper)

    try:
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    except Exception as e:
        print(f"[error] camera: {e}")
        ep_robot.close()
        sys.exit(1)

    odom = OdometryTracker()
    odom.subscribe(ep_chassis)

    print("Robot ready. (YOLO loads on first lego pickup)\n")
    return ep_robot, ep_chassis, ep_arm, ep_gripper, ep_camera, detector, odom


def shutdown_robot(ep_robot, ep_chassis, ep_camera, odom):
    for fn in (lambda: odom.unsubscribe(ep_chassis),
               lambda: stop(ep_chassis),
               lambda: ep_camera.stop_video_stream(),
               lambda: ep_robot.close()):
        try:
            fn()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("Shut down.")


# ── step 1: advance to scan line ──────────────────────────────────────────────

def step1_advance_to_scan_line(ep_chassis, odom):
    """Drive north from start until plan_y = SCAN_Y (0.75 m)."""
    print("\n--- Step 1: Advance to scan line ---")
    _, plan_y = get_plan_pos(odom)
    dist = SCAN_Y - plan_y
    if dist > 0.01:
        drive_straight(ep_chassis, dist)
    print(f"[step1] at y ≈ {SCAN_Y} m")


# ── step 2: strafe scan → build tag_map ──────────────────────────────────────

def step2_scan_and_map(ep_chassis, ep_camera, detector, odom):
    """
    Strafe west (positive plan_x) from east wall to west wall while facing north.
    South-facing april tags (goals, obstacles) are ahead; record each tag's
    plan-frame position when its centre aligns with the camera centre column.

    Returns tag_map: {tag_id: (plan_x, plan_y)}
    """
    print("\n--- Step 2: Strafe scan for april tags ---")
    tag_map  = {}
    seen_ids = set()

    drive(ep_chassis, x=0.0, y=-1*STRAFE_SCAN_SPD, z=0.0)  # strafe left = west
    deadline = time.time() + 10

    while time.time() < deadline:
        plan_x, plan_y = get_plan_pos(odom)
        if plan_x >= ARENA_M - 0.10:
            break

        frame = get_frame(ep_camera)
        if frame is None:
            continue

        tags = detect_apriltags(detector, frame)
        for tag in tags:
            tid = tag['id']
            cx  = float(tag['center'][0])
            err_x = cx - CAM_CX

            draw_apriltag(frame, tag, color=(255, 0, 255),
                          extra=f"err={err_x:.0f}")

            if tid in seen_ids:
                continue

            if abs(err_x) < X_RECORD_TOL:
                tx, ty = plan_x, plan_y + tag['dist']
                tag_map[tid] = (tx, ty)
                seen_ids.add(tid)
                cat = ("SMALL-GOAL" if tid in TAGS_SMALL_GOAL else
                       "LARGE-GOAL" if tid in TAGS_LARGE_GOAL else
                       "RECHARGE"   if tid in TAGS_RECHARGE   else "OBSTACLE")
                print(f"[scan] tag {tid} ({cat}) → plan ({tx:.2f}, {ty:.2f})")

        write_text(frame, f"scan: x={plan_x:.2f}  tags={len(tag_map)}")
        if show(frame):
            raise KeyboardInterrupt

    stop(ep_chassis)
    time.sleep(0.2)
    print(f"[step2] done. found: {list(tag_map.keys())}")
    return tag_map


# ── step 3: plot arena map ────────────────────────────────────────────────────

def step3_plot_map(tag_map):
    """
    Draw a 3.048 m × 3.048 m (10×10 ft) plan-frame map with:
      - hardcoded loading dock (purple rect)
      - robot start position
      - all discovered april tag boxes colour-coded by type
    Saves arena_map.png and displays non-blocking for 3 s.
    """
    print("\n--- Step 3: Plot arena map ---")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.2, ARENA_M + 0.2)
    ax.set_ylim(-0.2, ARENA_M + 0.2)
    ax.set_aspect('equal')
    ax.set_title("Lab 3 Arena Map")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)

    # Arena outline
    ax.add_patch(patches.Rectangle((0, 0), ARENA_M, ARENA_M, linewidth=2,
                                    edgecolor='navy', facecolor='whitesmoke', alpha=0.3))

    # Loading dock (hardcoded)
    dw = DOCK_NE[0] - DOCK_SW[0]
    dh = DOCK_NE[1] - DOCK_SW[1]
    ax.add_patch(patches.Rectangle(DOCK_SW, dw, dh, linewidth=2,
                                    edgecolor='purple', facecolor='lavender', alpha=0.6))
    ax.text((DOCK_SW[0] + DOCK_NE[0]) / 2, (DOCK_SW[1] + DOCK_NE[1]) / 2,
            "Loading Dock", ha='center', va='center', fontsize=9, color='purple')

    # Robot start
    ax.scatter([START_X], [START_Y], c='dodgerblue', s=180, marker='s',
               zorder=10, label='Start')

    style = {
        'small_goal': ('green',        '*', 220, 'Small-Goal'),
        'large_goal': ('purple',       '*', 220, 'Large-Goal'),
        'recharge':   ('gold',         'P', 180, 'Recharge'),
        'obstacle':   ('saddlebrown',  's', 120, 'Obstacle'),
    }
    first = {k: True for k in style}

    for tid, (tx, ty) in tag_map.items():
        if tid in TAGS_SMALL_GOAL:   cat = 'small_goal'
        elif tid in TAGS_LARGE_GOAL: cat = 'large_goal'
        elif tid in TAGS_RECHARGE:   cat = 'recharge'
        else:                        cat = 'obstacle'
        col, mrk, sz, lbl = style[cat]
        kw = dict(c=col, s=sz, marker=mrk, zorder=8)
        if first[cat]:
            kw['label'] = lbl
            first[cat] = False
        ax.add_patch(patches.Rectangle(
            (tx - BOX_SIZE / 2, ty - BOX_SIZE / 2), BOX_SIZE, BOX_SIZE,
            linewidth=1, edgecolor=col, facecolor='tan', alpha=0.35))
        ax.scatter([tx], [ty], **kw)
        ax.text(tx, ty + BOX_SIZE / 2 + 0.06, f"tag {tid}",
                ha='center', fontsize=8, color=col)

    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_map.png")
    plt.savefig(out)
    print(f"[step3] map saved → {out}")


# ── step 4: go to loading dock position ──────────────────────────────────────

def step4_go_to_dock(ep_chassis, odom):
    """
    Bring the robot to (DOCK_X, SCAN_Y) facing north.
    Corrects forward distance first, then strafes to dock x.
    """
    print(f"\n--- Step 4: Go to dock ({DOCK_X}, {SCAN_Y}) ---")
    plan_x, plan_y = get_plan_pos(odom)

    y_err = SCAN_Y - plan_y
    if abs(y_err) > 0.02:
        drive_straight(ep_chassis, y_err)

    plan_x, _ = get_plan_pos(odom)
    x_err = DOCK_X - plan_x
    if abs(x_err) > 0.02:
        crab_walk(ep_chassis, x_err)

    print(f"[step4] at plan {get_plan_pos(odom)}")


# ── step 5: face south, detect and grab lego ─────────────────────────────────

def step5_pick_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, odom):
    """
    Turn south, two-phase visual-servo approach to the nearest lego brick, grab.
    YOLO is loaded lazily on first call.
    """
    print("\n--- Step 5: Pick lego ---")
    rotate_to(ep_chassis, 180.0)
    arm_moveto(ep_arm, ARM_PICKUP, "pre-pick")
    gripper_open(ep_gripper)

    brick_type = None
    fwd_dist   = 0.0
    no_det     = 0
    MAX_NO_DET = 40
    t_prev     = time.time()

    # ── phase 1: coarse approach ──────────────────────────────────────────────
    print("[pick] phase 1 coarse")
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            t_prev = time.time()
            continue

        dets     = run_yolo(frame)
        all_lego = dets[CLS_SMALL] + dets[CLS_LARGE]

        if not all_lego:
            stop(ep_chassis)
            no_det += 1
            write_text(frame, f"pick1: no lego ({no_det})", color=(0, 0, 255))
            if show(frame): raise KeyboardInterrupt
            if no_det >= MAX_NO_DET:
                print("[pick] lego lost — phase 1")
                stop(ep_chassis)
                rotate_to(ep_chassis, 0.0)
                return None, 0.0
            t_prev = time.time()
            continue

        no_det     = 0
        target     = max(all_lego, key=lambda b: b['area'])
        brick_type = 'large' if target in dets[CLS_LARGE] else 'small'

        err_x = target['cx'] - CAM_CX
        yaw   = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if target['y2'] >= APPROACH_STOP_BOT:
            stop(ep_chassis)
            break

        t_now     = time.time()
        spd       = 0.06
        fwd_dist += spd * (t_now - t_prev)
        t_prev    = t_now

        drive(ep_chassis, x=spd, y=0.0, z=yaw)
        draw_box(frame, target, label=brick_type)
        write_text(frame, f"pick1 {brick_type} bot={target['y2']:.0f} dist={fwd_dist:.2f}")
        if show(frame): raise KeyboardInterrupt

    # ── phase 2: fine approach ────────────────────────────────────────────────
    grab_thresh = GRAB_TOP_SMALL if brick_type == 'small' else GRAB_TOP_LARGE
    print(f"[pick] phase 2 fine (thresh={grab_thresh})")
    no_det = 0
    t_prev = time.time()

    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            t_prev = time.time()
            continue

        dets     = run_yolo(frame)
        all_lego = dets[CLS_SMALL] + dets[CLS_LARGE]

        if not all_lego:
            stop(ep_chassis)
            no_det += 1
            write_text(frame, f"pick2: no lego ({no_det})", color=(0, 0, 255))
            if show(frame): raise KeyboardInterrupt
            if no_det >= MAX_NO_DET:
                print("[pick] lego lost — phase 2, grabbing anyway")
                stop(ep_chassis)
                arm_moveto(ep_arm, ARM_PICKUP, "lower")
                gripper_close(ep_gripper)
                arm_moveto(ep_arm, ARM_CARRY, "carry")
                rotate_to(ep_chassis, 0.0)
                return brick_type, fwd_dist
            t_prev = time.time()
            continue

        no_det     = 0
        target     = max(all_lego, key=lambda b: b['area'])
        brick_type = 'large' if target in dets[CLS_LARGE] else 'small'
        grab_thresh = GRAB_TOP_SMALL if brick_type == 'small' else GRAB_TOP_LARGE

        err_x = target['cx'] - CAM_CX
        yaw   = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if target['y1'] > grab_thresh:
            stop(ep_chassis)
            write_text(frame, f"pick2 GRAB {brick_type} top={target['y1']:.0f}",
                       color=(0, 255, 255))
            show(frame)
            print(f"[pick] grabbing {brick_type} (top={target['y1']:.0f}, dist={fwd_dist:.2f}m)")
            arm_moveto(ep_arm, ARM_PICKUP, "lower")
            gripper_close(ep_gripper)
            arm_moveto(ep_arm, ARM_CARRY, "carry")
            return brick_type, fwd_dist

        t_now     = time.time()
        spd       = 0.04
        fwd_dist += spd * (t_now - t_prev)
        t_prev    = t_now

        drive(ep_chassis, x=spd, y=0.0, z=yaw)
        draw_box(frame, target, color=(0, 255, 255), label=brick_type)
        write_text(frame, f"pick2 {brick_type} top={target['y1']:.0f}")
        if show(frame): raise KeyboardInterrupt


# ── step 6: back up, face north, re-align x with obstacle tag ────────────────

def step6_backup_and_realign(ep_chassis, ep_camera, detector, odom, tag_map, fwd_dist):
    """
    Back up fwd_dist metres (returns to scan-line y), turn north, then strafe
    to the nearest obstacle tag's x to correct lateral drift from the pickup.
    """
    print(f"\n--- Step 6: Back up {fwd_dist:.2f} m and realign ---")
    drive_straight(ep_chassis, fwd_dist, speed=SLOW_SPEED)   # positive = north (back to scan line)
    rotate_to(ep_chassis, 0.0)

    obs_tags = {tid: pos for tid, pos in tag_map.items() if tid not in TAGS_KNOWN}
    if not obs_tags:
        print("[realign] no obstacle tags — skipping x correction")
        return

    plan_x, _ = get_plan_pos(odom)
    nearest_tid = min(obs_tags, key=lambda t: abs(obs_tags[t][0] - plan_x))
    obs_x = obs_tags[nearest_tid][0]
    x_err = obs_x - plan_x
    print(f"[realign] obstacle tag {nearest_tid} at x={obs_x:.2f}, correcting {x_err:+.3f}m")
    if abs(x_err) > 0.03:
        crab_walk(ep_chassis, x_err)


# ── step 7: deliver lego to the correct goal zone ─────────────────────────────

def step7_deliver(ep_chassis, ep_arm, ep_gripper, ep_camera, detector, odom, tag_map, brick_type):
    """
    1. Crab-walk to the goal tag's x (from tag_map).
    2. Visual-servo forward toward the goal april tag (no rotation — crab for lateral).
    3. Stop GOAL_APPROACH_M from the tag.
    4. Drop lego, reverse back to SCAN_Y.
    """
    print(f"\n--- Step 7: Deliver {brick_type} lego ---")

    goal_ids  = TAGS_SMALL_GOAL if brick_type == 'small' else TAGS_LARGE_GOAL
    other_ids = TAGS_LARGE_GOAL if brick_type == 'small' else TAGS_SMALL_GOAL

    goal_pos = None
    for tid in goal_ids:
        if tid in tag_map:
            goal_pos = tag_map[tid]
            print(f"[deliver] target tag {tid} at plan {goal_pos}")
            break

    if goal_pos is None:
        for tid in other_ids:
            if tid in tag_map:
                kx, ky = tag_map[tid]
                goal_pos = (ARENA_M - kx, ky)
                print(f"[deliver] inferred goal at {goal_pos} (mirror of tag {tid})")
                break

    if goal_pos is None:
        print("[deliver] goal unknown — dropping in place")
        arm_moveto(ep_arm, ARM_PICKUP, "drop-fallback")
        gripper_open(ep_gripper)
        arm_moveto(ep_arm, ARM_HOME, "retract")
        return

    goal_x, _ = goal_pos

    # 1. Crab to goal x
    plan_x, _ = get_plan_pos(odom)
    x_err = goal_x - plan_x
    if abs(x_err) > 0.02:
        crab_walk(ep_chassis, x_err)

    # 2. Forward approach with lateral fine-tuning using april tag visual servo
    print("[deliver] approaching goal tag...")
    no_det = 0
    MAX_NO_DET = 30

    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        tags      = detect_apriltags(detector, frame)
        goal_tags = [t for t in tags if t['id'] in TAGS_SMALL_GOAL | TAGS_LARGE_GOAL]

        if not goal_tags:
            stop(ep_chassis)
            no_det += 1
            write_text(frame, f"deliver: no tag ({no_det})", color=(0, 0, 255))
            if show(frame): raise KeyboardInterrupt
            if no_det >= MAX_NO_DET:
                print("[deliver] tag lost — stopping approach")
                stop(ep_chassis)
                break
            continue

        no_det = 0
        gt     = min(goal_tags, key=lambda t: t['dist'])

        err_x    = float(gt['center'][0]) - CAM_CX
        strafe_v = max(-STRAFE_MAX, min(STRAFE_MAX, -err_x * STRAFE_KP))

        draw_apriltag(frame, gt, color=(0, 255, 0),
                      extra=f"err={err_x:.0f}")

        if gt['dist'] <= GOAL_APPROACH_M + 0.02:
            stop(ep_chassis)
            write_text(frame, f"deliver: ARRIVED dist={gt['dist']:.2f}", color=(0, 255, 255))
            show(frame)
            break

        fwd_v = max(0.0, min(0.08, (gt['dist'] - GOAL_APPROACH_M) * 0.4))
        drive(ep_chassis, x=fwd_v, y=strafe_v, z=0.0)

        write_text(frame, f"deliver dist={gt['dist']:.2f} err={err_x:.0f}")
        if show(frame): raise KeyboardInterrupt

    # 3. Drop lego
    print("[deliver] dropping...")
    arm_moveto(ep_arm, ARM_PICKUP, "lower-to-drop")
    time.sleep(0.3)
    gripper_open(ep_gripper)
    arm_moveto(ep_arm, ARM_HOME, "retract")

    # 4. Back up to scan line
    _, plan_y2 = get_plan_pos(odom)
    retreat = plan_y2 - SCAN_Y
    if retreat > 0.02:
        drive_straight(ep_chassis, -retreat)
    print("[deliver] done, back at scan line")


# ── step 8: turn south and find / map the charger ─────────────────────────────

def step8_find_charger(ep_chassis, ep_camera, detector, odom, tag_map):
    """
    Turn south at the current position. The recharge tag faces north — readable
    when the robot faces south and is north of the charger.
    Updates tag_map in-place. Returns (plan_x, plan_y) of charger, or None.
    """
    print("\n--- Step 8: Find charger ---")

    for tid in TAGS_RECHARGE:
        if tid in tag_map:
            print(f"[charger] already mapped: tag {tid} at {tag_map[tid]}")
            return tag_map[tid]

    obs_tags = {tid: pos for tid, pos in tag_map.items() if tid not in TAGS_KNOWN}
    if obs_tags:
        plan_x, _ = get_plan_pos(odom)
        nearest = min(obs_tags, key=lambda t: abs(obs_tags[t][0] - plan_x))
        obs_x   = obs_tags[nearest][0]
        if abs(obs_x - plan_x) > 0.03:
            crab_walk(ep_chassis, obs_x - plan_x)

    rotate_to(ep_chassis, 180.0)

    charger_pos = None
    for sweep in range(20):
        frame = get_frame(ep_camera)
        if frame is None:
            time.sleep(0.1)
            continue

        tags    = detect_apriltags(detector, frame)
        rc_tags = [t for t in tags if t['id'] in TAGS_RECHARGE]

        for rt in rc_tags:
            draw_apriltag(frame, rt, color=(0, 215, 255))
            plan_x, plan_y = get_plan_pos(odom)
            bear_rad = math.radians(rt['bearing'])
            tx = plan_x + rt['dist'] * math.sin(bear_rad)
            ty = plan_y - rt['dist'] * math.cos(bear_rad)
            tag_map[rt['id']] = (tx, ty)
            charger_pos = (tx, ty)
            print(f"[charger] tag {rt['id']} at plan ({tx:.2f}, {ty:.2f})")

        write_text(frame, f"charger search {sweep + 1}/20")
        if show(frame): raise KeyboardInterrupt

        if charger_pos:
            break

        if sweep < 5:
            drive(ep_chassis, x=0.0, y=-0.05, z=0.0)
        elif sweep < 10:
            drive(ep_chassis, x=0.0, y=0.05, z=0.0)
        time.sleep(0.2)
        stop(ep_chassis)

    stop(ep_chassis)
    rotate_to(ep_chassis, 0.0)
    if charger_pos is None:
        print("[charger] not found")
    return charger_pos


# ── step 9: approach charger and hold ────────────────────────────────────────

def step9_charge(ep_chassis, ep_camera, detector, battery, odom, tag_map):
    """
    Navigate to charger, align head-on facing south (tag faces north),
    hold RECHARGE_HOLD_S seconds, back up, face north.
    """
    print("\n--- Step 9: Charge robot ---")

    charger_pos = None
    for tid in TAGS_RECHARGE:
        if tid in tag_map:
            charger_pos = tag_map[tid]
            break

    if charger_pos is None:
        charger_pos = step8_find_charger(ep_chassis, ep_camera, detector, odom, tag_map)

    if charger_pos is None:
        print("[charge] charger not found — skipping")
        return False

    cx, _ = charger_pos

    plan_x, _ = get_plan_pos(odom)
    if abs(cx - plan_x) > 0.02:
        crab_walk(ep_chassis, cx - plan_x)

    rotate_to(ep_chassis, 180.0)

    print("[charge] approaching charger...")
    for _ in range(300):
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        tags    = detect_apriltags(detector, frame)
        rc_tags = [t for t in tags if t['id'] in TAGS_RECHARGE]

        if not rc_tags:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0)
            write_text(frame, "charge: searching...")
            if show(frame): raise KeyboardInterrupt
            continue

        rt   = rc_tags[0]
        draw_apriltag(frame, rt, color=(0, 215, 255))
        dist = rt['dist']
        bear = rt['bearing']

        if abs(bear) > 3.0:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=bear * 2.5)
        elif dist > RECHARGE_CLOSE_M + 0.03:
            ep_chassis.drive_speed(x=0.03, y=0.0, z=0.0)
        else:
            stop(ep_chassis)
            print(f"[charge] holding {RECHARGE_HOLD_S}s (dist={dist:.3f}m)")
            time.sleep(RECHARGE_HOLD_S)
            battery.recharge()
            drive_straight(ep_chassis, 0.35, speed=SLOW_SPEED)   # positive = north (away from charger)
            rotate_to(ep_chassis, 0.0)
            print("[charge] complete")
            return True

        write_text(frame, f"charge dist={dist:.2f}m bear={bear:.1f}")
        if show(frame): raise KeyboardInterrupt

    stop(ep_chassis)
    rotate_to(ep_chassis, 0.0)
    print("[charge] failed after max attempts")
    return False


# ── main state machine ────────────────────────────────────────────────────────

def main():
    (ep_robot, ep_chassis, ep_arm, ep_gripper,
     ep_camera, detector, odom) = setup_robot()

    battery        = BatteryManager()
    tag_map        = {}    # {tag_id: (plan_x, plan_y)}
    deliveries     = 0
    charger_mapped = False

    try:
        # ── map the arena ──────────────────────────────────────────────────
        step1_advance_to_scan_line(ep_chassis, odom)
        tag_map = step2_scan_and_map(ep_chassis, ep_camera, detector, odom)
        step3_plot_map(tag_map)

        # ── go to dock ─────────────────────────────────────────────────────
        step4_go_to_dock(ep_chassis, odom)

        # ── delivery loop ──────────────────────────────────────────────────
        while True:
            print(f"\n{'='*50}")
            print(f" Delivery {deliveries + 1}  |  Battery: {battery.level}%")
            print(f"{'='*50}")

            brick_type, fwd_dist = step5_pick_lego(
                ep_chassis, ep_arm, ep_gripper, ep_camera, odom)

            if brick_type is None:
                print("[main] no lego found — ending run")
                break

            step6_backup_and_realign(
                ep_chassis, ep_camera, detector, odom, tag_map, fwd_dist)

            step7_deliver(
                ep_chassis, ep_arm, ep_gripper, ep_camera, detector,
                odom, tag_map, brick_type)

            battery.consume(brick_type)
            deliveries += 1
            print(f"[main] delivery {deliveries} done | battery: {battery.level}%")

            # After first delivery: map the charger (one-time)
            if deliveries == 1 and not charger_mapped:
                step8_find_charger(ep_chassis, ep_camera, detector, odom, tag_map)
                charger_mapped = True

            # Recharge if battery below warn threshold
            if battery.needs_recharge(brick_type):
                print(f"[main] battery low ({battery.level}%) — charging")
                ok = step9_charge(ep_chassis, ep_camera, detector, battery, odom, tag_map)
                if ok:
                    charger_mapped = True

            # Return to dock for next lego
            step4_go_to_dock(ep_chassis, odom)

        print(f"\n{'='*50}")
        print(f" Session complete: {deliveries} deliveries | battery: {battery.level}%")
        print(f"{'='*50}\n")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt")
    except RuntimeError as e:
        print(f"\nRuntime error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_robot(ep_robot, ep_chassis, ep_camera, odom)


if __name__ == "__main__":
    main()
