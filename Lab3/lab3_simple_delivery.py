import cv2
import time
import os
import sys
import math
import pupil_apriltags
from ultralytics import YOLO
import robomaster
from robomaster import robot, camera

# ── Hardcoded parameters ──────────────────────────────────────────────────────
ROBOT_IP   = "192.168.50.120"
ROBOT_SN   = "3JKCH8800100AB"

FWD_DIST   = 0.6    # metres to drive forward at start
LEFT_DIST  = 1.8    # metres to strafe left to reach first lego column

GOAL_TAG_ID = 45    # april tag ID at the delivery zone (change as needed)

# Camera
CAM_CX     = 320.0
CAM_CY     = 180.0
CAM_FX     = 314.0
CAM_FY     = 314.0
TAG_SIZE_M = 0.15

# YOLO
CLS_SMALL   = 0
CONF_THRESH = 0.50

# Approach thresholds
LEGO_GRAB_Y2 = 300    # bbox bottom row >= this → close enough to grab
GOAL_STOP_M  = 0.2   # stop this many metres from the delivery april tag

# Proportional gains / speed caps
YAW_KP   = 0.15
YAW_MAX  = 25.0
FWD_KP   = 0.4
FWD_CAP  = 0.10

# Arm positions (x mm forward, y mm vertical relative to body)
ARM_HOME    = (185,  -60)
ARM_PICKUP  = (185, -50)
ARM_CARRY   = (20,  40)
GRIPPER_PWR = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_frame(ep_camera):
    for _ in range(3):
        try:
            f = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if f is not None:
                return f
        except Exception:
            pass
        time.sleep(0.05)
    return None


def stop(ep_chassis):
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)


def gripper_open(ep_gripper):
    ep_gripper.open(power=GRIPPER_PWR)
    time.sleep(1.0)
    ep_gripper.pause()


def gripper_close(ep_gripper):
    ep_gripper.close(power=GRIPPER_PWR)
    time.sleep(1.0)
    ep_gripper.pause()


def arm_moveto(ep_arm, pos):
    ep_arm.moveto(x=pos[0], y=pos[1]).wait_for_completed()


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
            })
        return results
    except Exception as e:
        print(f"[apriltag] {e}")
        return []


def load_yolo():
    weights = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs/detect/lab3_gpu_train/weights/best.pt",
    )
    if not os.path.exists(weights):
        print(f"[error] YOLO weights not found: {weights}")
        sys.exit(1)
    print("[yolo] loading model...")
    return YOLO(weights)


def run_yolo_small(model, frame):
    """Return list of small_lego detections sorted largest-first."""
    dets = []
    try:
        res = model.predict(source=frame, show=False, verbose=False)[0]
        if res.boxes is None:
            return dets
        for b in res.boxes:
            if int(b.cls) != CLS_SMALL or float(b.conf) < CONF_THRESH:
                continue
            xyxy = b.xyxy.cpu().numpy().flatten()
            if len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = map(float, xyxy[:4])
            dets.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) / 2,
                'area': (x2 - x1) * (y2 - y1),
            })
    except Exception as e:
        print(f"[yolo] {e}")
    dets.sort(key=lambda d: d['area'], reverse=True)
    return dets


# ── High-level actions ────────────────────────────────────────────────────────

def pick_small_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, model):
    """
    Visual-servo toward the largest small_lego in view.
    Proportional yaw to centre it; constant forward speed until
    the bbox bottom edge is low enough in the frame, then grab.
    """
    print("[pick] searching for small_lego...")
    arm_moveto(ep_arm, ARM_PICKUP)
    gripper_open(ep_gripper)

    no_det = 0
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        dets = run_yolo_small(model, frame)

        if not dets:
            stop(ep_chassis)
            no_det += 1
            print(f"[pick] no small_lego ({no_det})")
            if no_det > 60:
                print("[pick] gave up — no lego found")
                return False
            time.sleep(0.05)
            continue

        no_det = 0
        target = dets[0]

        err_x = target['cx'] - CAM_CX
        yaw   = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if target['y2'] >= LEGO_GRAB_Y2:
            stop(ep_chassis)
            print(f"[pick] in range (y2={target['y2']:.0f}) — grabbing")
            gripper_close(ep_gripper)
            arm_moveto(ep_arm, ARM_CARRY)
            return True

        ep_chassis.drive_speed(x=0.06, y=0.0, z=yaw)

        for i, d in enumerate(dets):
            is_target = (i == 0)
            color = (0, 255, 255) if is_target else (0, 180, 0)
            label = "TARGET: small_lego" if is_target else "small_lego"
            cv2.rectangle(frame, (int(d['x1']), int(d['y1'])),
                          (int(d['x2']), int(d['y2'])), color, 2)
            cv2.putText(frame, label, (int(d['x1']), int(d['y1']) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, f"err={err_x:.0f}  y2={target['y2']:.0f}  grab@{LEGO_GRAB_Y2}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("lab3_simple", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt


def approach_april_tag(ep_chassis, ep_camera, detector, tag_id):
    """
    Proportional controller: yaw to face the target tag, forward to close
    the gap.  Stops GOAL_STOP_M in front of it.
    """
    print(f"[deliver] approaching tag {tag_id}...")
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

        cv2.putText(frame, f"tag {tag_id}  dist={gt['dist']:.2f}m  err={err_x:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("lab3_simple", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)
    model    = load_yolo()

    print("[init] connecting to robot...")
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

    arm_moveto(ep_arm, ARM_HOME)
    gripper_open(ep_gripper)

    try:
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    except Exception as e:
        print(f"[error] camera: {e}")
        ep_robot.close()
        sys.exit(1)

    try:
        # ── Initial positioning ───────────────────────────────────────────────
        print(f"[init] drive forward {FWD_DIST}m")
        ep_chassis.move(x=FWD_DIST, y=0, z=0, xy_speed=0.5).wait_for_completed()

        print(f"[init] strafe left {LEFT_DIST}m")
        ep_chassis.move(x=0, y=-1*LEFT_DIST/3, z=0, xy_speed=1).wait_for_completed()
        ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.4).wait_for_completed()

        ep_chassis.move(x=0, y=-1*LEFT_DIST/3, z=0, xy_speed=1).wait_for_completed()
        ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.4).wait_for_completed()

        ep_chassis.move(x=0, y=-1*LEFT_DIST/3, z=0, xy_speed=1).wait_for_completed()
        ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.4).wait_for_completed()


        # ── Delivery loop ─────────────────────────────────────────────────────
        delivery = 0
        while True:
            delivery += 1
            print(f"\n=== Delivery {delivery} ===")

            # Turn to face lego side
            print("[loop] turning 180 to face legos")
            ep_chassis.move(x=0, y=0, z=180, z_speed=45).wait_for_completed()
            time.sleep(0.2)

            # Pick up small lego
            grabbed = pick_small_lego(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
            if not grabbed:
                print("[loop] could not grab lego — stopping")
                break

            # Turn back to face goal
            print("[loop] turning 180 to face goal tag")
            ep_chassis.move(x=0, y=0, z=180, z_speed=45).wait_for_completed()
            time.sleep(0.2)

            # Approach delivery april tag
            approach_april_tag(ep_chassis, ep_camera, detector, GOAL_TAG_ID)

            # Drop lego
            print("[loop] dropping lego")
            arm_moveto(ep_arm, ARM_PICKUP)
            time.sleep(0.3)
            gripper_open(ep_gripper)

            # Back up to clear the zone
            print("[loop] backing up 0.4m")
            ep_chassis.move(x=-0.4, y=0, z=0, xy_speed=0.3).wait_for_completed()
            time.sleep(0.2)

            # Next iteration: the loop top will turn 180 to face the lego side again

    except KeyboardInterrupt:
        print("\n[main] interrupted")
    except Exception as e:
        print(f"[main] error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)
        except Exception:
            pass
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[main] done")


if __name__ == "__main__":
    main()
