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

# constants

ROBOT_IP  = "192.168.50.120"
ROBOT_SN  = "3JKCH8800100AB"

ARENA_M   = 3.048
BOX_SIZE  = 0.26

# world frame origin = bottom-right corner, +x = west, +y = north
START_X   = 2.85
START_Y   = 0.15

SCAN_Y    = 0.8     # y-line to crabwalk along during scan

# loading dock plot bounds (plan-frame) — hardcoded
DOCK_SW  = (0.25, 0.25)
DOCK_NE  = (1.22, 0.91)

# fallback recharge position if tag not found during scan
RECHARGE_FALLBACK = (1.994, 0.344)

# april tag sets
TAGS_SMALL_GOAL = frozenset({30, 41})
TAGS_LARGE_GOAL = frozenset({45, 19})
TAGS_RECHARGE   = frozenset({38, 34})
TAGS_KNOWN      = TAGS_SMALL_GOAL | TAGS_LARGE_GOAL | TAGS_RECHARGE

TAG_SIZE_M = 0.15

CAM_FX   = 314.0
CAM_FY   = 314.0
CAM_CX   = 320.0
CAM_CY   = 180.0
FRAME_W  = 640
FRAME_H  = 360

# YOLO class indices (nc=3: small_lego, large_lego, box)
CLS_SMALL   = 0
CLS_LARGE   = 1
CLS_BOX     = 2
CONF_THRESH = 0.50

# battery (software simulation)
BAT_START      = 60
BAT_COST_SMALL = 20
BAT_COST_LARGE = 40
BAT_WARN_SMALL = 25
BAT_WARN_LARGE = 45

# chassis control
YAW_KP     = 0.15
YAW_MAX    = 25.0
STRAFE_KP  = 0.002
STRAFE_MAX = 0.12
CRAB_SPEED = 0.4
FWD_MAX    = 0.15
FWD_SPEED  = 0.1
SLOW_SPEED = 0.1
TURN_SPEED = 20.0

# crab walk scan
CRAB_SCAN_SPD = 0.8
X_RECORD_TOL    = 15

# lego approach
LEGO_GRAB_Y2  = 300   # bbox bottom >= this then close enough to grab
GOAL_STOP_M   = 0.2   # stop this far from delivery tag

# simple delivery approach gains
FWD_KP  = 0.4
FWD_CAP = 0.10

# recharge
RECHARGE_CLOSE_M = 0.08
RECHARGE_HOLD_S  = 5.2

# arm
ARM_HOME    = (185, 40)
ARM_CARRY   = (20,  40)
ARM_PICKUP  = (185, -50)
GRIPPER_POWER = 50
GRIPPER_HOLD  = 1.0

# hardcoded initial positioning (from lab3_simple_delivery)
INIT_FWD_DIST  = 0.6    # drive forward at start
INIT_LEFT_DIST = 1.8    # total strafe left to reach first lego column


# global state

current_theta = 0.0
_odom_ref     = None
_arm_pos      = None
_yolo_model   = None


# dead-reckoning odometry

class DeadReckoningOdom:
    """self-tracked dead-reckoning, no SDK subscriptions."""
    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self._lock = threading.Lock()

    def update_move(self, dx=0.0, dy=0.0, dyaw=0.0):
        with self._lock:
            rad = math.radians(self.yaw)
            self.x   += dx * math.cos(rad) - dy * math.sin(rad)
            self.y   += dx * math.sin(rad) + dy * math.cos(rad)
            self.yaw += dyaw

    def update_drive_speed(self, vx=0.0, vy=0.0, dt=0.0):
        self.update_move(vx * dt, vy * dt, 0.0)

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.yaw


# battery manager

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
        print(f"[battery] {brick_type} delivered: -{cost}% -> {self.level}%")

    def recharge(self):
        self.level = 100
        print("[battery] recharged -> 100%")


# perception helpers

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
        print("loading YOLO model...")
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


# chassis helpers

def drive(ep_chassis, x=0.0, y=0.0, z=0.0):
    ep_chassis.drive_speed(
        x=max(-FWD_MAX,    min(FWD_MAX,    x)),
        y=max(-STRAFE_MAX, min(STRAFE_MAX, y)),
        z=max(-YAW_MAX,    min(YAW_MAX,    z)),
    )


def stop(ep_chassis):
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)


def drive_straight(ep_chassis, dist_m, speed=None):
    global _odom_ref
    if speed is None:
        speed = FWD_SPEED
    if abs(dist_m) < 0.005:
        return
    print(f"[drive] {dist_m:+.3f}m @ {speed:.2f}m/s")
    ep_chassis.move(x=dist_m, y=0, z=0, xy_speed=abs(speed)).wait_for_completed()
    if _odom_ref is not None:
        _odom_ref.update_move(dx=dist_m)
    time.sleep(0.12)


def crab_walk(ep_chassis, dist_m, speed=None):
    """strafe using move command; positive = west."""
    global _odom_ref
    if abs(dist_m) < 0.005:
        return
    max_spd = min(abs(speed), CRAB_SPEED) if speed is not None else CRAB_SPEED
    print(f"[crab] {dist_m:+.3f}m @ {max_spd:.2f}m/s")
    ep_chassis.move(x=0, y=dist_m, z=0, xy_speed=max_spd).wait_for_completed()
    if _odom_ref is not None:
        _odom_ref.update_move(dy=dist_m)
    time.sleep(0.12)


def rotate_by(ep_chassis, delta_deg):
    global current_theta, _odom_ref
    if abs(delta_deg) < 0.3:
        return
    ep_chassis.move(x=0, y=0, z=delta_deg, z_speed=TURN_SPEED).wait_for_completed()
    current_theta = (current_theta + delta_deg) % 360
    if _odom_ref is not None:
        _odom_ref.update_move(dyaw=delta_deg)
    time.sleep(0.15)


def rotate_to(ep_chassis, target_deg):
    delta = target_deg - current_theta
    while delta >  180: delta -= 360
    while delta < -180: delta += 360
    rotate_by(ep_chassis, delta)


# position helper

def get_plan_pos(odom):
    # convert dead-reckoning odom to plan-frame (plan_x, plan_y).
    ox, oy, _ = odom.get_pose()
    return START_X + oy, START_Y - ox


# arm / gripper

def arm_moveto(ep_arm, pos, label="", step=10):
    global _arm_pos
    x, y = pos
    print(f"[arm] {label} -> ({x}, {y})")
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


# robot setup and shutdown

def setup_robot():
    print("setting up AprilTag detector...")
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)

    print("connecting to robot...")
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

    odom = DeadReckoningOdom()
    global _odom_ref
    _odom_ref = odom

    print("robot ready.\n")
    return ep_robot, ep_chassis, ep_arm, ep_gripper, ep_camera, detector, odom


def shutdown_robot(ep_robot, ep_chassis, ep_camera):
    for fn in (lambda: stop(ep_chassis),
               lambda: ep_camera.stop_video_stream(),
               lambda: ep_robot.close()):
        try:
            fn()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("shut down.")


# step 1: advance to the scan line

def step1_advance_to_scan_line(ep_chassis, odom):
    # drive north until plan_y == SCAN_Y.
    print("step 1: advance to scan line")
    _, plan_y = get_plan_pos(odom)
    dist = SCAN_Y - plan_y
    if dist > 0.01:
        drive_straight(ep_chassis, dist)


# step 2: strafe scan and build tag_map

def step2_scan_and_map(ep_chassis, ep_camera, detector, odom):
    print("\n--- step 2: strafe scan for april tags ---")
    tag_map             = {}
    seen_ids            = set()
    tag_discovery_order = []

    vy_cmd   = -CRAB_SCAN_SPD
    drive(ep_chassis, x=0.0, y=vy_cmd, z=0.0)
    t_prev   = time.time()
    deadline = t_prev + 20

    while time.time() < deadline:
        t_now  = time.time()
        dt     = t_now - t_prev
        t_prev = t_now
        odom.update_drive_speed(vx=0.0, vy=vy_cmd, dt=dt)

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

            draw_apriltag(frame, tag, color=(255, 0, 255), extra=f"err={err_x:.0f}")

            if tid in seen_ids:
                continue

            if abs(err_x) < X_RECORD_TOL:
                tx, ty = plan_x, plan_y + tag['dist']
                tag_map[tid] = (tx, ty)
                seen_ids.add(tid)
                tag_discovery_order.append(tid)
                cat = ("SMALL-GOAL" if tid in TAGS_SMALL_GOAL else
                       "LARGE-GOAL" if tid in TAGS_LARGE_GOAL else
                       "RECHARGE"   if tid in TAGS_RECHARGE   else "OBSTACLE")

        write_text(frame, f"scan: x={plan_x:.2f}  tags={len(tag_map)}")
        if show(frame):
            raise KeyboardInterrupt

    stop(ep_chassis)
    time.sleep(0.2)
    print(f"Tags mapped: {list(tag_map.keys())}")
    return tag_map, tag_discovery_order


# step 3: plot arena map

def step3_plot_map(tag_map):
    print("plot arena map")

    DOCK_CX   = (DOCK_SW[0] + DOCK_NE[0]) / 2
    DOCK_CY   = (DOCK_SW[1] + DOCK_NE[1]) / 2
    DOCK_HALF = 0.22

    recharge_pos = RECHARGE_FALLBACK
    for tid in TAGS_RECHARGE:
        if tid in tag_map:
            recharge_pos = tag_map[tid]
            break

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.15, ARENA_M + 0.15)
    ax.set_ylim(-0.15, ARENA_M + 0.15)
    ax.set_aspect('equal')
    ax.set_title("Lab 3 Arena - Bird's Eye View", fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    minor = np.arange(0, ARENA_M + 0.001, 0.1)
    major = np.arange(0, ARENA_M + 0.001, 0.5)
    ax.set_xticks(minor, minor=True)
    ax.set_yticks(minor, minor=True)
    ax.set_xticks(major)
    ax.set_yticks(major)
    ax.grid(which='minor', color='gray', alpha=0.20, linewidth=0.4)
    ax.grid(which='major', color='gray', alpha=0.45, linewidth=0.7)

    # arena boundary
    ax.add_patch(patches.Rectangle(
        (0, 0), ARENA_M, ARENA_M,
        linewidth=3, edgecolor='black', facecolor='whitesmoke', alpha=0.25, zorder=1))

    # loading dock — yellow square
    ax.add_patch(patches.Rectangle(
        (DOCK_CX - DOCK_HALF, DOCK_CY - DOCK_HALF),
        2 * DOCK_HALF, 2 * DOCK_HALF,
        linewidth=2, edgecolor='goldenrod', facecolor='yellow', alpha=0.85, zorder=3))
    ax.text(DOCK_CX, DOCK_CY, "Loading\nDock",
            ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)

    # recharging station — black square
    rx, ry = recharge_pos
    RECHARGE_HALF = 0.15
    ax.add_patch(patches.Rectangle(
        (rx - RECHARGE_HALF, ry - RECHARGE_HALF),
        2 * RECHARGE_HALF, 2 * RECHARGE_HALF,
        linewidth=2, edgecolor='black', facecolor='black', alpha=0.85, zorder=3))
    ax.text(rx, ry, "Recharge",
            ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=4)

    # april tag objects from tag_map
    for tid, (tx, ty) in tag_map.items():
        if tid in TAGS_RECHARGE:
            pass   # already drawn above
        elif tid in TAGS_SMALL_GOAL:
            R = 0.15
            h = R * math.sqrt(3) / 2
            tri_x = [tx - h, tx + h, tx,     tx - h]
            tri_y = [ty - R/2, ty - R/2, ty + R, ty - R/2]
            ax.fill(tri_x, tri_y, color='blue', alpha=0.85, zorder=5)
            ax.plot(tri_x, tri_y, color='darkblue', linewidth=1.5, zorder=6)
            ax.text(tx, ty + R + 0.07, "Small Goal",
                    ha='center', fontsize=8, color='blue', fontweight='bold', zorder=7)
        elif tid in TAGS_LARGE_GOAL:
            R = 0.18
            h = R * math.sqrt(3) / 2
            tri_x = [tx - h, tx + h, tx,     tx - h]
            tri_y = [ty - R/2, ty - R/2, ty + R, ty - R/2]
            ax.fill(tri_x, tri_y, color='green', alpha=0.85, zorder=5)
            ax.plot(tri_x, tri_y, color='darkgreen', linewidth=1.5, zorder=6)
            ax.text(tx, ty + R + 0.07, "Large Goal",
                    ha='center', fontsize=8, color='green', fontweight='bold', zorder=7)
        else:
            # red circle — obstacles
            ax.add_patch(patches.Circle(
                (tx, ty), BOX_SIZE / 2,
                linewidth=2, edgecolor='darkred', facecolor='red', alpha=0.75, zorder=5))
            ax.text(tx, ty + BOX_SIZE / 2 + 0.07, "Obstacle",
                    ha='center', fontsize=7, color='darkred', zorder=6)

    legend_handles = [
        patches.Patch(facecolor='red',    edgecolor='darkred',   alpha=0.75, label='Obstacle (fabric box)'),
        patches.Patch(facecolor='blue',   edgecolor='darkblue',  alpha=0.85, label='Small Goal'),
        patches.Patch(facecolor='green',  edgecolor='darkgreen', alpha=0.85, label='Large Goal'),
        patches.Patch(facecolor='yellow', edgecolor='goldenrod', alpha=0.85, label='Loading Dock'),
        patches.Patch(facecolor='black',  edgecolor='black',     alpha=0.85, label='Recharging Station'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_map.png")
    plt.savefig(out, dpi=150)
    print(f"[step3] map saved -> {out}")
    plt.close(fig)


# determine lego target class from scan discovery order

def determine_target_cls(tag_map, tag_discovery_order):

    # figure out which target is closest to the loading dock
    target_cls = CLS_SMALL
    check_up_to = min(4, len(tag_discovery_order))

    for i in range(check_up_to - 1, -1, -1):
        tid = tag_discovery_order[i]
        if tid in TAGS_SMALL_GOAL:
            target_cls = CLS_SMALL
            break
        elif tid in TAGS_LARGE_GOAL:
            target_cls = CLS_LARGE
            break
    else:
        print(f"No known goal tag found")

    brick_type  = 'large' if target_cls == CLS_LARGE else 'small'
    goal_set    = TAGS_LARGE_GOAL if target_cls == CLS_LARGE else TAGS_SMALL_GOAL
    goal_tag_id = next((tid for tid in goal_set if tid in tag_map), next(iter(goal_set)))
    return target_cls, brick_type, goal_tag_id


# pick lego target to grab

def pick_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, target_cls):
    cls_name = "small_lego" if target_cls == CLS_SMALL else "large_lego"
    print(f"searching for {cls_name}")
    arm_moveto(ep_arm, ARM_PICKUP, "pre-pick")
    gripper_open(ep_gripper)

    no_det = 0
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        dets = run_yolo(frame)[target_cls]

        if not dets:
            stop(ep_chassis)
            no_det += 1
            if no_det > 60:
                return False
            time.sleep(0.05)
            continue

        no_det = 0
        target = dets[0]   # largest by area

        err_x = target['cx'] - CAM_CX
        yaw   = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if target['y2'] >= LEGO_GRAB_Y2:
            stop(ep_chassis)
            gripper_close(ep_gripper)
            arm_moveto(ep_arm, ARM_CARRY, "carry")
            return True

        ep_chassis.drive_speed(x=0.06, y=0.0, z=yaw)

        for i, d in enumerate(dets):
            color = (0, 255, 255) if i == 0 else (0, 180, 0)
            label = f"TARGET: {cls_name}" if i == 0 else cls_name
            draw_box(frame, d, color=color, label=label)
        write_text(frame, f"pick: err={err_x:.0f}  y2={target['y2']:.0f}  grab@{LEGO_GRAB_Y2}")
        if show(frame):
            raise KeyboardInterrupt


# approach april tag
def approach_april_tag(ep_chassis, ep_camera, detector, tag_id):
    print(f"Approaching tag {tag_id}...")
    no_det = 0
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        tags      = detect_apriltags(detector, frame)
        goal_tags = [t for t in tags if t['id'] == tag_id]

        if not goal_tags:
            stop(ep_chassis)
            no_det += 1
            print(f"[deliver] tag {tag_id} not visible ({no_det})")
            if no_det > 60:
                print("[deliver] gave up on approach")
                return
            time.sleep(0.05)
            continue

        no_det = 0
        gt     = goal_tags[0]

        err_x = float(gt['center'][0]) - CAM_CX
        yaw   = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if gt['dist'] <= GOAL_STOP_M:
            stop(ep_chassis)
            print(f"[deliver] arrived (dist={gt['dist']:.3f}m)")
            return

        fwd = min(FWD_CAP, (gt['dist'] - GOAL_STOP_M) * FWD_KP)
        ep_chassis.drive_speed(x=max(0.0, fwd), y=0.0, z=yaw)

        write_text(frame, f"tag {tag_id}  dist={gt['dist']:.2f}m  err={err_x:.0f}")
        draw_apriltag(frame, gt, color=(0, 255, 0))
        if show(frame):
            raise KeyboardInterrupt


# find charger tag
def find_charger(ep_chassis, ep_camera, detector, odom, tag_map):
    print("find charger")

    for tid in TAGS_RECHARGE:
        if tid in tag_map:
            return tag_map[tid]

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

        write_text(frame, f"charger search {sweep + 1}/20")
        if show(frame):
            raise KeyboardInterrupt

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
        print("charger not found")
    return charger_pos


# approach charger and hold

def charge(ep_chassis, ep_camera, detector, battery, odom, tag_map):
    print("charge robot")

    charger_pos = None
    for tid in TAGS_RECHARGE:
        if tid in tag_map:
            charger_pos = tag_map[tid]
            break

    if charger_pos is None:
        charger_pos = find_charger(ep_chassis, ep_camera, detector, odom, tag_map)

    if charger_pos is None:
        print("charger not found")
        return False

    cx, _ = charger_pos
    plan_x, _ = get_plan_pos(odom)
    if abs(cx - plan_x) > 0.02:
        crab_walk(ep_chassis, cx - plan_x)

    rotate_to(ep_chassis, 180.0)

    print("approaching charger")
    for _ in range(300):
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        tags    = detect_apriltags(detector, frame)
        rc_tags = [t for t in tags if t['id'] in TAGS_RECHARGE]

        if not rc_tags:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0)
            if show(frame):
                raise KeyboardInterrupt
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
            time.sleep(RECHARGE_HOLD_S)
            battery.recharge()
            drive_straight(ep_chassis, 0.35, speed=SLOW_SPEED)
            rotate_to(ep_chassis, 0.0)
            print("Charge completed")
            return True

        write_text(frame, f"charge dist={dist:.2f}m bear={bear:.1f}")
        if show(frame):
            raise KeyboardInterrupt

    stop(ep_chassis)
    rotate_to(ep_chassis, 0.0)
    return False


def main():
    (ep_robot, ep_chassis, ep_arm, ep_gripper,
     ep_camera, detector, odom) = setup_robot()

    battery = BatteryManager()
    tag_map = {}
    charger_mapped = False

    try:
        # map the arena
        step1_advance_to_scan_line(ep_chassis, odom)
        tag_map, tag_discovery_order = step2_scan_and_map(ep_chassis, ep_camera, detector, odom)
        step3_plot_map(tag_map)

        # determine lego type from last known goal tag in first four discovered
        target_cls, brick_type, goal_tag_id = determine_target_cls(tag_map, tag_discovery_order)

        delivery = 0

        while True:
            delivery += 1
            print(f" delivery {delivery}  |  battery: {battery.level}%  |  target: {brick_type}")

            # recharge before pickup if needed
            if battery.needs_recharge(brick_type):
                print(f"[main] battery low ({battery.level}%) — charging before pickup")
                ok = charge(ep_chassis, ep_camera, detector, battery, odom, tag_map)
                if ok:
                    charger_mapped = True

            # turn south to face lego side
            rotate_to(ep_chassis, 180.0)

            grabbed = pick_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, target_cls)
            if not grabbed:
                print("[main] could not grab lego — stopping")
                break

            # turn north to face goal
            rotate_to(ep_chassis, 0.0)

            approach_april_tag(ep_chassis, ep_camera, detector, goal_tag_id)

            # drop lego
            print("[main] dropping lego")
            arm_moveto(ep_arm, ARM_PICKUP, "drop")
            time.sleep(0.3)
            gripper_open(ep_gripper)
            arm_moveto(ep_arm, ARM_HOME, "retract")

            # back up to clear the delivery zone
            print("[main] backing up 0.4m")
            drive_straight(ep_chassis, -0.4, speed=SLOW_SPEED)

            battery.consume(brick_type)
            print(f"[main] delivery {delivery} done | battery: {battery.level}%")

            # map charger after first successful delivery
            if delivery == 1 and not charger_mapped:
                find_charger(ep_chassis, ep_camera, detector, odom, tag_map)
                charger_mapped = True

            # recharge after delivery if needed
            if battery.needs_recharge(brick_type):
                print(f"[main] battery low ({battery.level}%) — charging")
                ok = charge(ep_chassis, ep_camera, detector, battery, odom, tag_map)
                if ok:
                    charger_mapped = True

        print(f" session complete: {delivery} deliveries | battery: {battery.level}%")

    except KeyboardInterrupt:
        print("\nkeyboard interrupt")
    except RuntimeError as e:
        print(f"\nruntime error: {e}")
    except Exception as e:
        print(f"\nunexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_robot(ep_robot, ep_chassis, ep_camera)


if __name__ == "__main__":
    main()
