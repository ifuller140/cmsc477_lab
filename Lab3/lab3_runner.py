"""
logistics_runner.py — cmsc477 project 3: energy-aware logistics challenge

═══════════════════════════════════════════════════════════════════════════════
STRATEGY OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

LOCALIZATION (how the robot knows where it is):
  layer 1 — chassis odometry:
    ep_chassis.sub_position() gives (x, y) in meters relative to start.
    ep_chassis.sub_attitude() gives yaw in degrees relative to start.
    this is our primary positioning backbone. it drifts slowly over time
    (~5-10cm per meter traveled) but is perfectly adequate for a 10x10ft arena.

  layer 2 — apriltag corrections:
    when we see any goal/recharge apriltag, pupil_apriltags returns the tag's
    exact pose in the camera frame (distance + bearing + full 6DOF transform).
    if we've already stored that tag's world position, we can use:
      world_pos_of_tag = known
      camera_pose_relative_to_tag = from detection
      robot_world_pose = solve(world_pos_of_tag, camera_pose)
    this corrects any odometry drift each time a tag becomes visible.

  layer 3 — cone proximity (boundary awareness only):
    YOLO cone detections + pinhole distance estimation give rough cone positions.
    we don't do full localization from cones (hard data association problem).
    instead: if cones appear large in frame → we're near a wall → steer away.

MAPPING (the matplotlib live map):
  the map is a 3.05m × 3.05m grid (10ft × 10ft arena).
  all discovered objects are plotted in world frame:
    - orange triangles  → cones (perimeter boundary markers)
    - tan squares       → fabric box obstacles
    - green stars       → small lego goal zone (found via apriltag)
    - purple stars      → large lego goal zone (found via apriltag)
    - gold P            → recharge station (found via apriltag)
    - red diamond       → loading dock (first lego cluster found)
    - blue trail        → robot path history

NAVIGATION:
  the star method from project 2 is extended here:
    home (origin) = robot's starting position
    all "go to known location" moves = rotate_to(heading) + drive_straight(dist)
    this avoids cumulative drift because each leg is measured independently
    from a known starting point.

  obstacle avoidance (reactive):
    during any navigation leg, if YOLO detects cones or boxes within
    OBSTACLE_WARN meters, we add a lateral velocity component to steer around them.
    the force is proportional to (OBSTACLE_WARN - detected_dist) / OBSTACLE_WARN
    and directed away from the object's bearing in the camera frame.

TASK STATE MACHINE:
  the robot cycles through:
    EXPLORE  →  find all apriltags, map obstacles, locate loading dock
    FETCH    →  go to dock, visual-servo to a lego brick, grab it
    DELIVER  →  navigate to appropriate goal zone, place brick
    RECHARGE →  navigate to recharge station, align head-on, hold 5s
    DONE     →  session complete

BATTERY:
  tracked purely in software (starts 60%, small costs 20%, large costs 40%).
  before every FETCH we check: will we reach the goal and still have ≥0% after?
  if not, we recharge first. recharge restores to 100%.

YOLO TRAINING NOTES:
  you need to train four classes on your specific hardware:
    class 0 → cone        (orange traffic cone, ~28cm tall)
    class 1 → fabric_box  (beige 26cm cube)
    class 2 → lego_small  (small duplo brick)
    class 3 → lego_large  (large duplo brick / tall stack)
  pool datasets with other teams for better generalization.
  use the same camera (robomaster) for training images.

APRILTAG SETUP:
  print tags from the tag36h11 family.
  fill in TAG_SMALL_GOAL, TAG_LARGE_GOAL, TAG_RECHARGE below with your actual ids.
  measure your printed tag size and set TAG_SIZE_M.
═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import cv2
import numpy as np
import time
import math
import threading
import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ultralytics import YOLO
import pupil_apriltags
import robomaster
from robomaster import robot, camera

# ── constants — tune these for your specific setup ────────────────────────────

# arena dimensions
ARENA_M      = 3.048        # 10 ft in meters
ARENA_HALF   = ARENA_M / 2

# camera (calibrate focal length by measuring known object at known distance)
# formula: f = (pixel_height * real_distance) / real_height
FOCAL_PX     = 600          # pixels — estimate for 640x360 stream
FRAME_W      = 640
FRAME_H      = 360
FRAME_CX     = FRAME_W / 2.0
FRAME_CY     = FRAME_H / 2.0

# real heights for distance estimation via pinhole model (Z = f*H_real / H_px)
CONE_H_M          = 0.28   # measure your actual cones (traffic cones ~28cm)
BOX_H_M           = 0.26   # given: 26cm fabric cubes
LEGO_SMALL_H_M    = 0.048  # small duplo stack — measure yours
LEGO_LARGE_H_M    = 0.096  # large duplo stack — measure yours

# apriltag ids — each zone has two valid tags (either triggers zone discovery)
TAGS_SMALL_GOAL = {11, 41}
TAGS_LARGE_GOAL = {45, 19}
TAGS_RECHARGE   = {38, 34}
TAG_SIZE_M      = 0.15      # measure your printed tag size

# yolo class indices — matches lab3_data.yaml: nc=3, names=[small_lego, large_lego, box]
# NOTE: cones are NOT a yolo class; detected via HSV color segmentation (key CLS_CONE=3)
CLS_LEGO_SMALL = 0
CLS_LEGO_LARGE = 1
CLS_BOX        = 2
CLS_CONE       = 3   # sentinel; filled by detect_cones_hsv, not by YOLO
CONF_THRESH    = 0.50

# battery (software simulation per project spec)
BAT_START      = 60         # percent at session start
BAT_COST_SMALL = 20         # cost per small brick delivery
BAT_COST_LARGE = 40         # cost per large brick delivery
BAT_WARN_SMALL = 25         # recharge if below this before picking small
BAT_WARN_LARGE = 45         # recharge if below this before picking large

# navigation / chassis
YAW_KP         = 0.15       # deg/s per pixel of horizontal center error
YAW_MAX        = 25.0       # max yaw rate deg/s
FWD_SPEED      = 0.10       # m/s for straight drive legs
SLOW_SPEED     = 0.05       # m/s for careful close-in moves
TURN_SPEED     = 15.0       # deg/s for chassis.move() rotations
WAYPOINT_DIST  = 0.28       # m: waypoint counted as reached
OBSTACLE_WARN  = 0.65       # m: steer away if obstacle closer than this

# lego visual approach thresholds (same as project 2 values)
APPROACH_STOP_BOTTOM = 250  # px: coarse approach stops when bbox bottom >= this
GRAB_TOP_THRESH      = 95   # px: grab when bbox top edge > this

# recharge protocol
RECHARGE_HOLD_SECS   = 5.2  # hold 5+ seconds stationary
RECHARGE_CLOSE_M     = 0.05 # within 5cm of tag

# gripper + arm (same positions as projects 1 and 2)
GRIPPER_POWER = 50
GRIPPER_HOLD  = 1.0
ARM_HOME      = (185, -80)
ARM_CARRY     = (185, -40)
ARM_PICKUP    = (185, -50)

# exploration waypoint grid (inner 3×3 points, margins from walls)
# arena is 3.05m; stay 0.5m from edges to avoid cones
_MARGIN = 0.50
_STEP   = (ARENA_M - 2 * _MARGIN) / 2.0
EXPLORE_WAYPOINTS = [
    (_MARGIN + c * _STEP, _MARGIN + r * _STEP)
    for r in range(3)
    for c in range(3)
]

# ── global mutable state ──────────────────────────────────────────────────────

_arm_pos    = None          # last known arm (x, y) — updated by arm_moveto
_theta      = 0.0           # robot heading in degrees (0 = initial forward direction)

# ── odometry subscriber ───────────────────────────────────────────────────────

class OdometryTracker:
    """
    subscribes to the robomaster chassis position + attitude callbacks.
    gives us (x, y, yaw) in the world frame relative to where we powered on.

    convention (verify against your hardware):
      x — meters forward from start
      y — meters left from start
      yaw — degrees counterclockwise from start forward direction

    NOTE: if the robot's yaw convention is clockwise-positive on your unit,
    flip the sign of yaw wherever heading_error is computed in navigate_to().
    """

    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self._lock = threading.Lock()

    def _pos_cb(self, pos_info):
        # callback signature: (x, y, z) in meters
        x, y, z = pos_info
        with self._lock:
            self.x = x
            self.y = y

    def _att_cb(self, att_info):
        # callback signature: (yaw, pitch, roll) in degrees
        yaw, pitch, roll = att_info
        with self._lock:
            self.yaw = yaw

    def subscribe(self, ep_chassis):
        ep_chassis.sub_position(cs=0, freq=20, callback=self._pos_cb)
        ep_chassis.sub_attitude(freq=20, callback=self._att_cb)

    def unsubscribe(self, ep_chassis):
        try:
            ep_chassis.unsub_position()
        except Exception:
            pass
        try:
            ep_chassis.unsub_attitude()
        except Exception:
            pass

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.yaw

# ── arena map (matplotlib live display) ──────────────────────────────────────

class ArenaMap:
    """
    maintains a live matplotlib plot of the 10×10 ft arena.
    call refresh() each main loop tick to update the display.
    """

    def __init__(self):
        self.cones      = []     # list of (x, y) estimated world positions
        self.obstacles  = []     # fabric boxes (no apriltag): list of (x, y)
        self.goal_small = None   # (x, y) when found
        self.goal_large = None
        self.recharge   = None
        self.dock       = None   # loading dock location (where we first see lego)
        self.path_x     = [0.0]
        self.path_y     = [0.0]
        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0

        # setup matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._init_plot()

    def _init_plot(self):
        ax = self.ax
        ax.set_xlim(-0.3, ARENA_M + 0.3)
        ax.set_ylim(-0.3, ARENA_M + 0.3)
        ax.set_aspect('equal')
        ax.set_title("live arena map — logistics runner")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # gray arena floor
        rect = plt.Rectangle((0, 0), ARENA_M, ARENA_M,
                              linewidth=2, edgecolor='blue',
                              facecolor='whitesmoke', alpha=0.4)
        ax.add_patch(rect)

        # plot handles — updated by refresh()
        self.sc_cones    = ax.scatter([], [], c='darkorange', s=90,  marker='^', label='cone',      zorder=5)
        self.sc_obs      = ax.scatter([], [], c='tan',        s=220, marker='s', label='box',       zorder=4)
        self.sc_sg       = ax.scatter([], [], c='green',      s=220, marker='*', label='goal-small',zorder=6)
        self.sc_lg       = ax.scatter([], [], c='purple',     s=220, marker='*', label='goal-large',zorder=6)
        self.sc_rc       = ax.scatter([], [], c='gold',       s=220, marker='P', label='recharge',  zorder=6)
        self.sc_dock     = ax.scatter([], [], c='red',        s=220, marker='D', label='dock',      zorder=6)
        self.ln_path,    = ax.plot([], [], 'b-', linewidth=1, alpha=0.4, label='path')
        self.sc_robot    = ax.scatter([], [], c='dodgerblue', s=120, zorder=10,  label='robot')
        self._arrow      = None

        ax.legend(loc='upper right', fontsize=7)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_robot(self, x, y, yaw):
        self.rx = x; self.ry = y; self.ryaw = yaw
        self.path_x.append(x)
        self.path_y.append(y)

    def add_cone(self, wx, wy):
        for cx, cy in self.cones:
            if math.hypot(wx - cx, wy - cy) < 0.30:
                return   # already mapped
        self.cones.append((wx, wy))

    def add_obstacle(self, wx, wy):
        for ox, oy in self.obstacles:
            if math.hypot(wx - ox, wy - oy) < 0.35:
                return
        self.obstacles.append((wx, wy))

    def refresh(self):
        """update all plot elements and flush the figure."""
        if self.cones:
            xs, ys = zip(*self.cones)
            self.sc_cones.set_offsets(list(zip(xs, ys)))
        if self.obstacles:
            xs, ys = zip(*self.obstacles)
            self.sc_obs.set_offsets(list(zip(xs, ys)))
        if self.goal_small:
            self.sc_sg.set_offsets([self.goal_small])
        if self.goal_large:
            self.sc_lg.set_offsets([self.goal_large])
        if self.recharge:
            self.sc_rc.set_offsets([self.recharge])
        if self.dock:
            self.sc_dock.set_offsets([self.dock])

        self.ln_path.set_data(self.path_x, self.path_y)
        self.sc_robot.set_offsets([[self.rx, self.ry]])

        # heading arrow
        if self._arrow:
            self._arrow.remove()
        yaw_r = math.radians(self.ryaw)
        dx = 0.18 * math.cos(yaw_r)
        dy = 0.18 * math.sin(yaw_r)
        self._arrow = self.ax.annotate(
            "", xy=(self.rx + dx, self.ry + dy),
            xytext=(self.rx, self.ry),
            arrowprops=dict(arrowstyle="->", color='dodgerblue', lw=2)
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ── battery manager ───────────────────────────────────────────────────────────

class BatteryManager:
    """software simulation of the project's battery rules."""

    def __init__(self):
        self.level = BAT_START
        print(f"[battery] starting at {self.level}%")

    def needs_recharge_for(self, brick_size):
        """returns True if we should recharge before picking this brick."""
        cost = BAT_COST_LARGE if brick_size == 'large' else BAT_COST_SMALL
        warn = BAT_WARN_LARGE if brick_size == 'large' else BAT_WARN_SMALL
        return self.level < warn or self.level < cost

    def consume(self, brick_size):
        cost = BAT_COST_LARGE if brick_size == 'large' else BAT_COST_SMALL
        self.level = max(0, self.level - cost)
        print(f"[battery] delivered {brick_size} brick: -{cost}%  →  {self.level}%")

    def recharge(self):
        self.level = 100
        print("[battery] recharged → 100%")

# ── perception helpers ────────────────────────────────────────────────────────

def get_frame(ep_camera):
    for _ in range(3):
        try:
            f = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if f is not None:
                return f
        except Exception as e:
            print(f"[cam] {e}")
        time.sleep(0.05)
    return None


def estimate_dist(bbox_h_px, real_h_m):
    """pinhole model: Z = focal_length * real_height / pixel_height"""
    if bbox_h_px < 3:
        return None
    return (FOCAL_PX * real_h_m) / bbox_h_px


def bearing_deg(bbox_cx):
    """
    bearing in degrees from robot's forward axis to detected object.
    positive = object is to the robot's left.
    """
    err = bbox_cx - FRAME_CX
    return -math.degrees(math.atan2(err, FOCAL_PX))


def world_xy(rx, ry, ryaw_deg, dist_m, bearing_deg_val):
    """project a range + bearing observation into world (x, y) coordinates."""
    abs_angle = math.radians(ryaw_deg + bearing_deg_val)
    return rx + dist_m * math.cos(abs_angle), ry + dist_m * math.sin(abs_angle)


def run_yolo(model, frame):
    """
    run yolo inference and return a dict of detections grouped by class.
    each detection has: x1,y1,x2,y2, cx,cy, w,h, area, conf.
    lists are sorted by area descending (largest = closest = most important).
    CLS_CONE (key 3) is always returned empty here — filled by detect_cones_hsv.
    """
    out = {CLS_LEGO_SMALL: [], CLS_LEGO_LARGE: [], CLS_BOX: [], CLS_CONE: []}
    try:
        res = model.predict(source=frame, show=False, verbose=False)[0]
        if res.boxes is None:
            return out
        for b in res.boxes:
            conf = float(b.conf)
            if conf < CONF_THRESH:
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
                'conf': conf,
            })
        for cls in out:
            out[cls].sort(key=lambda d: d['area'], reverse=True)
    except Exception as e:
        print(f"[yolo] {e}")
    return out


def detect_cones_hsv(frame):
    """
    detect orange traffic cones via HSV color segmentation.
    returns list of detection dicts in the same format as run_yolo entries.
    cones are bright saturated orange-red: H 0-20 or 160-179 in OpenCV scale.
    minimum area 800px² filters out small orange lego pieces.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # orange-red band (wrap-around for deep red-orange hues)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   150, 80]), np.array([20,  255, 255])),
        cv2.inRange(hsv, np.array([160, 150, 80]), np.array([179, 255, 255])),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cones.append({
            'x1': float(x), 'y1': float(y),
            'x2': float(x + w), 'y2': float(y + h),
            'cx': float(x + w / 2), 'cy': float(y + h / 2),
            'w': float(w), 'h': float(h),
            'area': float(area),
            'conf': 1.0,
        })
    cones.sort(key=lambda d: d['area'], reverse=True)
    return cones


def run_full_perception(model, frame):
    """run YOLO + HSV cone detection and return merged detections dict."""
    dets = run_yolo(model, frame)
    dets[CLS_CONE] = detect_cones_hsv(frame)
    return dets


def detect_apriltags(detector, frame, known_world_positions=None):
    """
    detect apriltags in frame.
    returns list of dicts:
      id, dist_m, bearing_deg, center (px), pose_t (camera frame xyz)
    if known_world_positions dict {tag_id: (wx,wy)} provided, also returns
    the estimated robot world position from each tag.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    try:
        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[FOCAL_PX, FOCAL_PX, FRAME_CX, FRAME_CY],
            tag_size=TAG_SIZE_M,
        )
        for tag in tags:
            t = tag.pose_t.flatten()          # (x_cam, y_cam, z_cam) in meters
            dist  = float(t[2])               # forward distance from camera
            bear  = math.degrees(math.atan2(float(t[0]), float(t[2])))
            entry = {
                'id':      tag.tag_id,
                'dist':    dist,
                'bearing': bear,
                'center':  tag.center,
                'pose_t':  t,
                'pose_R':  tag.pose_R,
            }

            # if we know where this tag is, estimate robot world pose
            if known_world_positions and tag.tag_id in known_world_positions:
                twx, twy = known_world_positions[tag.tag_id]
                # the tag faces the robot, so robot is at tag + dist in opposite direction
                # bearing from tag to robot = -(bear from robot to tag)
                robot_angle = math.radians(bear + 180.0)
                entry['corrected_rx'] = twx + dist * math.cos(robot_angle)
                entry['corrected_ry'] = twy + dist * math.sin(robot_angle)

            results.append(entry)
    except Exception as e:
        print(f"[apriltag] {e}")
    return results


def annotate_frame(frame, dets, tags, status=""):
    """draw all yolo boxes and apriltag detections on the frame."""
    colors = {
        CLS_CONE:       (0, 120, 255),
        CLS_BOX:        (180, 160, 0),
        CLS_LEGO_SMALL: (0, 220, 80),
        CLS_LEGO_LARGE: (0, 180, 120),
    }
    names = {CLS_CONE: 'cone', CLS_BOX: 'box', CLS_LEGO_SMALL: 'S-lego', CLS_LEGO_LARGE: 'L-lego'}
    for cls, det_list in dets.items():
        c = colors.get(cls, (255, 255, 255))
        for d in det_list:
            cv2.rectangle(frame, (int(d['x1']), int(d['y1'])),
                          (int(d['x2']), int(d['y2'])), c, 2)
            cv2.putText(frame, f"{names[cls]} {d['conf']:.2f}",
                        (int(d['x1']), int(d['y1']) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
    for tag in tags:
        cx, cy = tag['center']
        cv2.circle(frame, (int(cx), int(cy)), 14, (255, 0, 255), 2)
        cv2.putText(frame, f"tag{tag['id']} {tag['dist']:.2f}m",
                    (int(cx) - 20, int(cy) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)
    if status:
        cv2.putText(frame, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow("logistics", frame)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


# ── map update helper ─────────────────────────────────────────────────────────

def update_map(arena_map, dets, tags, rx, ry, ryaw, known_tag_positions):
    """
    project all yolo + apriltag detections into world frame and add to map.
    also corrects robot position estimate if a known tag is visible.
    returns corrected (rx, ry) if correction available, else original values.
    """
    # cones — treat as boundary markers only, don't over-plot
    for d in dets[CLS_CONE]:
        dist = estimate_dist(d['h'], CONE_H_M)
        if dist and dist < 3.5:
            bear = bearing_deg(d['cx'])
            wx, wy = world_xy(rx, ry, ryaw, dist, bear)
            # only add if plausibly near the perimeter (outside inner 1m)
            if wx < 0.4 or wx > ARENA_M - 0.4 or wy < 0.4 or wy > ARENA_M - 0.4:
                arena_map.add_cone(wx, wy)

    # fabric box obstacles
    for d in dets[CLS_BOX]:
        dist = estimate_dist(d['h'], BOX_H_M)
        if dist and dist < 3.0:
            bear = bearing_deg(d['cx'])
            wx, wy = world_xy(rx, ry, ryaw, dist, bear)
            # only add if it doesn't collide with a known goal position
            is_goal = False
            for gpos in [arena_map.goal_small, arena_map.goal_large, arena_map.recharge]:
                if gpos and math.hypot(wx - gpos[0], wy - gpos[1]) < 0.4:
                    is_goal = True
                    break
            if not is_goal:
                arena_map.add_obstacle(wx, wy)

    # apriltags — discover goals and recharge; optionally correct odometry
    corrected_rx, corrected_ry = rx, ry
    for tag in tags:
        dist = tag['dist']
        bear = tag['bearing']
        wx, wy = world_xy(rx, ry, ryaw, dist, bear)

        if tag['id'] in TAGS_SMALL_GOAL:
            if arena_map.goal_small is None:
                arena_map.goal_small = (wx, wy)
                known_tag_positions[tag['id']] = (wx, wy)
                print(f"[map] small goal found at world ({wx:.2f}, {wy:.2f}) via tag {tag['id']}")
            if 'corrected_rx' in tag:
                corrected_rx = tag['corrected_rx']
                corrected_ry = tag['corrected_ry']

        elif tag['id'] in TAGS_LARGE_GOAL:
            if arena_map.goal_large is None:
                arena_map.goal_large = (wx, wy)
                known_tag_positions[tag['id']] = (wx, wy)
                print(f"[map] large goal found at world ({wx:.2f}, {wy:.2f}) via tag {tag['id']}")
            if 'corrected_rx' in tag:
                corrected_rx = tag['corrected_rx']
                corrected_ry = tag['corrected_ry']

        elif tag['id'] in TAGS_RECHARGE:
            if arena_map.recharge is None:
                arena_map.recharge = (wx, wy)
                known_tag_positions[tag['id']] = (wx, wy)
                print(f"[map] recharge found at world ({wx:.2f}, {wy:.2f}) via tag {tag['id']}")
            if 'corrected_rx' in tag:
                corrected_rx = tag['corrected_rx']
                corrected_ry = tag['corrected_ry']

    return corrected_rx, corrected_ry

# ── arm + gripper helpers (identical pattern to projects 1 & 2) ───────────────

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
        ep_arm.moveto(x=int(cx + (x - cx) * i / steps),
                      y=int(cy + (y - cy) * i / steps)).wait_for_completed()
    _arm_pos = (x, y)


def gripper_close(ep_gripper):
    ep_gripper.close(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()


def gripper_open(ep_gripper):
    ep_gripper.open(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()

# ── chassis helpers ───────────────────────────────────────────────────────────

def stop(ep_chassis):
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)


def rotate_by(ep_chassis, delta_deg):
    """rotate exactly delta_deg degrees in place. updates global _theta."""
    global _theta
    if abs(delta_deg) < 0.3:
        return
    ep_chassis.move(x=0, y=0, z=delta_deg, xy_speed=0.05, z_speed=TURN_SPEED).wait_for_completed()
    _theta += delta_deg
    time.sleep(0.15)


def rotate_to(ep_chassis, target_deg):
    global _theta
    delta = target_deg - _theta
    while delta >  180: delta -= 360
    while delta < -180: delta += 360
    rotate_by(ep_chassis, delta)


def drive_straight(ep_chassis, dist_m, speed=None):
    """drive forward (positive) or backward (negative) by exact distance."""
    if speed is None:
        speed = FWD_SPEED
    if abs(dist_m) < 0.005:
        return
    print(f"[drive] {dist_m:+.4f}m at {speed:.3f}m/s")
    ep_chassis.move(x=dist_m, y=0, z=0, xy_speed=abs(speed)).wait_for_completed()
    time.sleep(0.12)

# ── navigate to a world coordinate waypoint ───────────────────────────────────

def navigate_to(ep_chassis, ep_camera, model, detector,
                arena_map, odom, known_tags,
                target_wx, target_wy,
                stop_dist=WAYPOINT_DIST, timeout=90):
    """
    navigate toward (target_wx, target_wy) in world frame using live odometry.
    proportional control on heading error + forward speed.
    reactive obstacle avoidance via yolo bounding box sizes.
    updates map continuously as new objects are detected.
    returns True when reached, False on timeout/interrupt.

    NOTE: the robomaster chassis yaw convention may differ from math.atan2.
    if the robot always turns the wrong way, negate heading_err below.
    """
    print(f"[nav] → ({target_wx:.2f}, {target_wy:.2f})")
    t_start = time.time()

    while True:
        if time.time() - t_start > timeout:
            stop(ep_chassis)
            print("[nav] timeout")
            return False

        frame = get_frame(ep_camera)
        if frame is None:
            continue

        rx, ry, ryaw = odom.get_pose()
        dets = run_full_perception(model, frame)
        tags = detect_apriltags(detector, frame, known_tags)
        crx, cry = update_map(arena_map, dets, tags, rx, ry, ryaw, known_tags)
        arena_map.update_robot(crx, cry, ryaw)

        dx   = target_wx - crx
        dy   = target_wy - cry
        dist = math.hypot(dx, dy)

        if dist <= stop_dist:
            stop(ep_chassis)
            print(f"[nav] reached target, dist={dist:.3f}m")
            return True

        # heading error: angle to target minus current yaw
        target_yaw_deg = math.degrees(math.atan2(dy, dx))
        heading_err    = target_yaw_deg - ryaw
        while heading_err >  180: heading_err -= 360
        while heading_err < -180: heading_err += 360

        # slow down forward speed when heading error is large
        yaw_cmd = max(-YAW_MAX, min(YAW_MAX, heading_err * 0.8))
        fwd_cmd = max(0.0, min(FWD_SPEED, dist * 0.25))
        if abs(heading_err) > 30:
            fwd_cmd *= 0.2     # mostly turn, little forward

        # reactive avoidance: push laterally away from close obstacles
        avoid_y = 0.0
        for cls, h_real in [(CLS_CONE, CONE_H_M), (CLS_BOX, BOX_H_M)]:
            for d in dets[cls]:
                od = estimate_dist(d['h'], h_real)
                if od and od < OBSTACLE_WARN:
                    push = (OBSTACLE_WARN - od) / OBSTACLE_WARN
                    side = 1.0 if d['cx'] > FRAME_CX else -1.0  # push away from side it's on
                    avoid_y += push * 0.15 * (-side)

        ep_chassis.drive_speed(
            x=max(-FWD_SPEED, min(FWD_SPEED, fwd_cmd)),
            y=max(-0.30,      min(0.30,      avoid_y)),
            z=max(-YAW_MAX,   min(YAW_MAX,   yaw_cmd)),
        )

        status = f"nav dist={dist:.2f}m  hdg_err={heading_err:.1f}  bat={''}"
        if annotate_frame(frame, dets, tags, status):
            stop(ep_chassis)
            raise KeyboardInterrupt

        arena_map.refresh()

# ── spin scan at current position ─────────────────────────────────────────────

def spin_scan(ep_chassis, ep_camera, model, detector,
              arena_map, odom, known_tags, steps=10):
    """
    rotate 360° in steps, running full perception at each stop.
    populates the map with all visible cones, boxes, and apriltags.
    """
    step_deg = 360.0 / steps
    for _ in range(steps):
        rotate_by(ep_chassis, step_deg)
        time.sleep(0.25)
        frame = get_frame(ep_camera)
        if frame is None:
            continue
        rx, ry, ryaw = odom.get_pose()
        dets = run_full_perception(model, frame)
        tags = detect_apriltags(detector, frame, known_tags)
        update_map(arena_map, dets, tags, rx, ry, ryaw, known_tags)
        annotate_frame(frame, dets, tags, f"scanning... {_+1}/{steps}")
        arena_map.refresh()

# ── csv map export ────────────────────────────────────────────────────────────

def _save_map_csv(arena_map, filename="arena_map.csv"):
    """write all discovered objects to a CSV for inspection / debugging."""
    rows = []
    for label, pos in [('small_goal', arena_map.goal_small),
                       ('large_goal', arena_map.goal_large),
                       ('recharge',   arena_map.recharge),
                       ('dock',       arena_map.dock)]:
        if pos:
            rows.append({'type': label, 'x': round(pos[0], 3), 'y': round(pos[1], 3)})
    for cx, cy in arena_map.cones:
        rows.append({'type': 'cone', 'x': round(cx, 3), 'y': round(cy, 3)})
    for ox, oy in arena_map.obstacles:
        rows.append({'type': 'box_obstacle', 'x': round(ox, 3), 'y': round(oy, 3)})
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'x', 'y'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[map] saved {len(rows)} objects to {filename}")

# ── phase a: explore arena and discover all zones ────────────────────────────

def phase_explore(ep_chassis, ep_camera, model, detector,
                  arena_map, odom, known_tags):
    """
    navigate a 3x3 grid of interior waypoints, scanning at each stop.
    populates the map with all goals, recharge station, and obstacles.
    stops early once all three apriltag zones are found.
    also looks for lego bricks to mark the loading dock.
    """
    print("\n═══ phase: explore ═══")

    # initial 360 spin at start position to capture nearby landmarks
    spin_scan(ep_chassis, ep_camera, model, detector, arena_map, odom, known_tags, steps=8)

    for i, (wx, wy) in enumerate(EXPLORE_WAYPOINTS):
        # bail early if we've found all three apriltag zones
        if (arena_map.goal_small is not None and
                arena_map.goal_large is not None and
                arena_map.recharge  is not None):
            print("[explore] all zones discovered — ending exploration early")
            break

        print(f"[explore] waypoint {i+1}/{len(EXPLORE_WAYPOINTS)} → ({wx:.2f}, {wy:.2f})")
        navigate_to(ep_chassis, ep_camera, model, detector,
                    arena_map, odom, known_tags, wx, wy)

        # look for lego here to mark the dock
        _check_for_dock(ep_camera, model, arena_map, odom)

        # spin and scan at this waypoint
        spin_scan(ep_chassis, ep_camera, model, detector, arena_map, odom, known_tags, steps=6)
        arena_map.refresh()

    # navigate home after exploration
    navigate_to(ep_chassis, ep_camera, model, detector,
                arena_map, odom, known_tags, 0.0, 0.0)

    print(f"[explore] done. found: small_goal={arena_map.goal_small}, "
          f"large_goal={arena_map.goal_large}, recharge={arena_map.recharge}, "
          f"dock={arena_map.dock}, obstacles={len(arena_map.obstacles)}")
    _save_map_csv(arena_map)


def _check_for_dock(ep_camera, model, arena_map, odom):
    """look at current frame for lego bricks; if found, record as loading dock."""
    if arena_map.dock is not None:
        return  # already found
    frame = get_frame(ep_camera)
    if frame is None:
        return
    dets = run_yolo(model, frame)
    legoes = dets[CLS_LEGO_SMALL] + dets[CLS_LEGO_LARGE]
    if not legoes:
        return
    rx, ry, ryaw = odom.get_pose()
    biggest = max(legoes, key=lambda d: d['area'])
    h_real = LEGO_SMALL_H_M if biggest in dets[CLS_LEGO_SMALL] else LEGO_LARGE_H_M
    dist = estimate_dist(biggest['h'], h_real)
    if dist and dist < 2.5:
        bear = bearing_deg(biggest['cx'])
        wx, wy = world_xy(rx, ry, ryaw, dist, bear)
        arena_map.dock = (wx, wy)
        print(f"[dock] loading dock found at ({wx:.2f}, {wy:.2f})")

# ── phase b: approach and grab a lego brick ──────────────────────────────────

def phase_pick_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, model):
    """
    visual servo approach to the nearest lego brick.
    identical two-phase logic to projects 1 and 2.
    returns 'small', 'large', or None if failed.
    """
    print("\n═══ phase: pick lego ═══")
    arm_moveto(ep_arm, ARM_HOME, "pre-pick")
    gripper_open(ep_gripper)

    brick_type  = None
    no_det      = 0
    MAX_NO_DET  = 40

    # phase 1 — coarse approach: drive until bbox bottom fills lower frame
    print("[pick] coarse approach...")
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue
        dets = run_full_perception(model, frame)
        all_lego = dets[CLS_LEGO_SMALL] + dets[CLS_LEGO_LARGE]

        if not all_lego:
            stop(ep_chassis)
            no_det += 1
            if frame is not None:
                annotate_frame(frame, dets, [], f"pick-coarse: no lego ({no_det})")
            if no_det >= MAX_NO_DET:
                print("[pick] lego lost during coarse approach")
                return None
            continue

        no_det = 0
        target = max(all_lego, key=lambda d: d['area'])
        # figure out small vs large from which list it came from
        brick_type = 'large' if target in dets[CLS_LEGO_LARGE] else 'small'

        err_x = target['cx'] - FRAME_CX
        yaw   = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if target['y2'] >= APPROACH_STOP_BOTTOM:
            stop(ep_chassis)
            break

        ep_chassis.drive_speed(x=0.06, y=0.0, z=yaw)
        annotate_frame(frame, dets, [], f"pick-coarse bot={target['y2']:.0f}")

    # phase 2 — fine approach: creep until bbox top edge is low in frame
    print("[pick] fine approach...")
    no_det = 0
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue
        dets = run_full_perception(model, frame)
        all_lego = dets[CLS_LEGO_SMALL] + dets[CLS_LEGO_LARGE]

        if not all_lego:
            stop(ep_chassis)
            no_det += 1
            if no_det >= MAX_NO_DET:
                return None
            continue

        no_det = 0
        target = max(all_lego, key=lambda d: d['area'])
        brick_type = 'large' if target in dets[CLS_LEGO_LARGE] else 'small'

        err_x = target['cx'] - FRAME_CX
        yaw   = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if target['y1'] > GRAB_TOP_THRESH:
            stop(ep_chassis)
            print(f"[pick] grabbing {brick_type} brick...")
            arm_moveto(ep_arm, ARM_PICKUP, "lower")
            gripper_close(ep_gripper)
            arm_moveto(ep_arm, ARM_CARRY, "carry")
            print(f"[pick] grabbed: {brick_type}")
            return brick_type

        ep_chassis.drive_speed(x=0.04, y=0.0, z=yaw)
        annotate_frame(frame, dets, [], f"pick-fine top={target['y1']:.0f}")

# ── phase c: place lego at goal zone ─────────────────────────────────────────

def phase_place_lego(ep_arm, ep_gripper):
    """lower arm, open gripper to deposit brick."""
    print("[place] placing brick...")
    arm_moveto(ep_arm, ARM_PICKUP, "lower-to-place")
    time.sleep(0.3)
    gripper_open(ep_gripper)
    arm_moveto(ep_arm, ARM_HOME, "retract")
    print("[place] done")

# ── phase d: recharge sequence ────────────────────────────────────────────────

def phase_recharge(ep_chassis, ep_camera, model, detector, battery, arena_map, odom, known_tags):
    """
    navigate to the recharge station and perform the head-on approach protocol.
    the project requires:
      1. approach head-on (bearing ≈ 0)
      2. start from at least 1 ft (0.3m) away
      3. get within 5cm of the tag
      4. remain stationary for ≥ 5 seconds → battery resets to 100%
    """
    print("\n═══ phase: recharge ═══")

    if arena_map.recharge is None:
        print("[recharge] station not yet mapped — cannot recharge")
        return False

    # navigate to ~0.6m from the recharge station so we start the
    # protocol from the required ≥ 1 ft distance
    rx, ry, ryaw = odom.get_pose()
    dx = arena_map.recharge[0] - rx
    dy = arena_map.recharge[1] - ry
    dist_to_station = math.hypot(dx, dy)
    approach_angle  = math.degrees(math.atan2(dy, dx))

    # move to a point 0.6m in front of the tag
    stage_x = arena_map.recharge[0] - 0.6 * math.cos(math.radians(approach_angle))
    stage_y = arena_map.recharge[1] - 0.6 * math.sin(math.radians(approach_angle))

    navigate_to(ep_chassis, ep_camera, model, detector,
                arena_map, odom, known_tags, stage_x, stage_y, stop_dist=0.15)

    # fine alignment using live apriltag detection
    print("[recharge] starting fine approach...")
    MAX_TRIES = 200
    for i in range(MAX_TRIES):
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        tags = detect_apriltags(detector, frame, known_tags)
        rc_tags = [t for t in tags if t['id'] in TAGS_RECHARGE]

        if not rc_tags:
            # rotate slowly to find the tag
            ep_chassis.drive_speed(x=0.0, y=0.0, z=10.0)
            time.sleep(0.08)
            annotate_frame(frame, {CLS_CONE:[], CLS_BOX:[], CLS_LEGO_SMALL:[], CLS_LEGO_LARGE:[]},
                           [], "recharge: searching for tag...")
            continue

        tag  = rc_tags[0]
        dist = tag['dist']
        bear = tag['bearing']   # positive = tag is to our left

        # align heading first (zero bearing = facing head-on)
        if abs(bear) > 3.0:
            yaw = max(-18.0, min(18.0, bear * 3.5))
            ep_chassis.drive_speed(x=0.0, y=0.0, z=yaw)
        elif dist > RECHARGE_CLOSE_M + 0.03:
            # aligned but not close enough — creep forward
            ep_chassis.drive_speed(x=0.035, y=0.0, z=0.0)
        else:
            # within 5cm and aligned — hold stationary for protocol
            stop(ep_chassis)
            print(f"[recharge] holding for {RECHARGE_HOLD_SECS}s... dist={dist:.3f}m")
            time.sleep(RECHARGE_HOLD_SECS)
            battery.recharge()
            # back away so we don't collide when turning
            drive_straight(ep_chassis, -0.35, speed=SLOW_SPEED)
            print("[recharge] complete")
            return True

        annotate_frame(frame, {CLS_CONE:[], CLS_BOX:[], CLS_LEGO_SMALL:[], CLS_LEGO_LARGE:[]},
                       tags, f"recharge dist={dist:.3f}m bear={bear:.1f}°")

    stop(ep_chassis)
    print("[recharge] failed to reach station after max attempts")
    return False

# ── robot setup / teardown ────────────────────────────────────────────────────

def setup_robot():
    print("loading yolo weights...")
    weights = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs/detect/lab3_gpu_train/weights/best.pt"
    )
    if not os.path.exists(weights):
        print(f"[error] weights not found at:\n  {weights}")
        sys.exit(1)
    model = YOLO(weights)

    print("setting up apriltag detector...")
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)

    print("connecting to robot...")
    robomaster.config.ROBOT_IP_STR = "192.168.50.121"
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100UB")
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

    # start odometry subscription
    odom = OdometryTracker()
    odom.subscribe(ep_chassis)

    print("robot ready.\n")
    return ep_robot, ep_chassis, ep_arm, ep_gripper, ep_camera, model, detector, odom


def shutdown_robot(ep_robot, ep_chassis, ep_camera, odom):
    print("shutting down...")
    for fn in [
        lambda: odom.unsubscribe(ep_chassis),
        lambda: stop(ep_chassis),
        lambda: ep_camera.stop_video_stream(),
        lambda: ep_robot.close(),
    ]:
        try:
            fn()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("done.")

# ── main state machine ────────────────────────────────────────────────────────

def main():
    (ep_robot, ep_chassis, ep_arm, ep_gripper,
     ep_camera, model, detector, odom) = setup_robot()

    arena_map   = ArenaMap()
    battery     = BatteryManager()
    known_tags  = {}        # {tag_id: (world_x, world_y)} — populated as tags found
    deliveries  = 0
    TARGET_DELIVERIES = 5  # aim for 25pt task efficiency bonus

    try:
        # ── explore: discover goals, recharge, obstacles ──────────────────
        phase_explore(ep_chassis, ep_camera, model, detector,
                      arena_map, odom, known_tags)

        # return to home (origin) before starting delivery loop
        navigate_to(ep_chassis, ep_camera, model, detector,
                    arena_map, odom, known_tags, 0.0, 0.0)
        rotate_to(ep_chassis, 0.0)

        # ── delivery loop ─────────────────────────────────────────────────
        while deliveries < TARGET_DELIVERIES:
            print(f"\n{'═'*60}")
            print(f" delivery {deliveries + 1} / {TARGET_DELIVERIES}  |  battery: {battery.level}%")
            print(f"{'═'*60}")

            rx, ry, ryaw = odom.get_pose()
            arena_map.update_robot(rx, ry, ryaw)
            arena_map.refresh()

            # decide if we should recharge first (use conservative threshold)
            # we check for the worst case (large brick) to be safe
            if battery.needs_recharge_for('large'):
                print(f"[main] battery low ({battery.level}%) — recharging first")
                phase_recharge(ep_chassis, ep_camera, model, detector,
                               battery, arena_map, odom, known_tags)
                navigate_to(ep_chassis, ep_camera, model, detector,
                            arena_map, odom, known_tags, 0.0, 0.0)

            # go to loading dock
            if arena_map.dock is None:
                print("[main] dock not found — spinning to search")
                spin_scan(ep_chassis, ep_camera, model, detector, arena_map, odom, known_tags)
                _check_for_dock(ep_camera, model, arena_map, odom)

            if arena_map.dock:
                navigate_to(ep_chassis, ep_camera, model, detector,
                            arena_map, odom, known_tags,
                            arena_map.dock[0], arena_map.dock[1], stop_dist=0.55)

            # pick a lego brick
            brick_size = phase_pick_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
            if brick_size is None:
                print("[main] failed to pick lego — spinning to search and retry")
                stop(ep_chassis)
                spin_scan(ep_chassis, ep_camera, model, detector, arena_map, odom, known_tags, steps=4)
                brick_size = phase_pick_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
                if brick_size is None:
                    print("[main] still no lego found — ending run")
                    break

            # pick the correct goal zone for this brick
            goal_pos = arena_map.goal_large if brick_size == 'large' else arena_map.goal_small
            if goal_pos is None:
                # fall back to whatever zone we found (better than not delivering)
                goal_pos = arena_map.goal_small or arena_map.goal_large
                print(f"[main] correct zone not mapped, falling back to {goal_pos}")

            if goal_pos is None:
                print("[main] no goal zone known — dropping brick and re-exploring")
                phase_place_lego(ep_arm, ep_gripper)
                phase_explore(ep_chassis, ep_camera, model, detector,
                              arena_map, odom, known_tags)
                continue

            # navigate to the goal zone and deliver
            navigate_to(ep_chassis, ep_camera, model, detector,
                        arena_map, odom, known_tags,
                        goal_pos[0], goal_pos[1], stop_dist=0.30)

            phase_place_lego(ep_arm, ep_gripper)
            battery.consume(brick_size)
            deliveries += 1

            print(f"[main] ✓ delivery {deliveries} complete | battery: {battery.level}%")

            # return home between deliveries for clean next-leg odometry
            navigate_to(ep_chassis, ep_camera, model, detector,
                        arena_map, odom, known_tags, 0.0, 0.0)

        print(f"\n{'═'*60}")
        print(f" session complete: {deliveries} deliveries | battery: {battery.level}%")
        print(f"{'═'*60}\n")

    except KeyboardInterrupt:
        print("\nkeyboard interrupt")
    except RuntimeError as e:
        print(f"\nruntime error: {e}")
    except Exception as e:
        print(f"\nunexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_robot(ep_robot, ep_chassis, ep_camera, odom)
        # keep final map visible
        plt.ioff()
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
