import cv2
import time
import os
import sys
import math
from ultralytics import YOLO
import robomaster
from robomaster import robot, camera

# Constants (From robot_runner.py for reliable hardware interaction)
YAW_KP   = 0.1              # deg per horizontal pixel error
YAW_MAX  = 20.0             # deg clamp
FRAME_CX = 320

FWD_KP   = 0.0015           # m/s per pixel error
FWD_MAX  = 0.15             # m/s clamp

APPROACH_STOP_BOTTOM = 200  # px: stop when bbox bottom reaches this row
GRAB_TOP_THRESHOLD  = 90    # px: grab when bbox TOP edge (xyxy[1]) > this value

GRIPPER_POWER       = 50
GRIPPER_HOLD        = 1.0  # seconds hold

# Arm positions (x = mm forward, y = mm vertical)
ARM_HOME      = (185, -80)  # retracted standby
ARM_CARRY     = (185, -40)  # slight lift while carrying
ARM_PICKUP    = (185, -50)  # drop height / grab height

# Variables & Constants for Lab 2 Logic (Theta/dist homing)
SCAN_STEP  = 10.0           # degrees per scan increment when searching
SCAN_SPEED = 10.0           # deg/s for chassis.move() rotation calls
CENTER_TOL = 10             # pixels: bbox cx must be within this to count as centered
MOVE_SPEED = 0.1           # m/s for chassis.move() straight legs
SLOW_BACKUP_SPEED = 0.05

TEMP_ANGLE    = 270.0       # degrees from home forward direction
TEMP_DISTANCE = 0.10        # meters from home
TEMP_ANGLE_RETURN = 100.0   # goes back and looks around here for temp

# Global state for returning home
current_theta = 0.0         # Robot heading in degrees (0 = home forward)
CURRENT_ARM_POS = None

# Helper Functions (From robot_runner.py)
def get_frame(ep_camera, retries=3): 
    for _ in range(retries):
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is not None:
                return frame
        except Exception as e:
            print(f"[CAM] read error: {e}")
        time.sleep(0.05)
    return None

def run_yolo(model, frame):
    try:
        result = model.predict(source=frame, show=False, verbose=False)[0]
        bboxes = []
        if result.boxes is None or len(result.boxes) == 0:
            return []
        for b in result.boxes:
            xyxy = b.xyxy.cpu().numpy().flatten()
            if len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bboxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'area': (x2 - x1) * (y2 - y1),
                'cx': x1 + (x2 - x1) / 2.0,
                'cy': y1 + (y2 - y1) / 2.0,
            })
        bboxes.sort(key=lambda b: b['area'], reverse=True)
        return bboxes
    except Exception as e:
        print(f"yolo error: {e}")
        return []

def draw_box(frame, b, color=(0, 255, 0), label=""):
    cv2.rectangle(frame, (int(b['x1']), int(b['y1'])), (int(b['x2']), int(b['y2'])), color, 2)
    if label:
        cv2.putText(frame, label, (int(b['x1']), int(b['y1']) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def write_text_on_video_feed(frame, text, row=30, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def show(frame, title="RoboMaster"):
    cv2.imshow("lab2_runner", frame)
    return (cv2.waitKey(1) & 0xFF) == ord('q')

def drive(ep_chassis, x=0.0, y=0.0, z=0.0):
    ep_chassis.drive_speed(
        x=max(-FWD_MAX, min(FWD_MAX, x)),
        y=max(-0.5,     min(0.5,     y)),
        z=max(-YAW_MAX, min(YAW_MAX, z)),
    )

def stop(ep_chassis):
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)

def arm_moveto(ep_arm, pos, label="", step_size=10):
    global CURRENT_ARM_POS
    x, y = pos
    print(f"[ARM] {label} -> moveto(x={x}, y={y})")
    
    if CURRENT_ARM_POS is None:
        ep_arm.moveto(x=x, y=y).wait_for_completed()
        CURRENT_ARM_POS = (x, y)
        return
        
    cx, cy = CURRENT_ARM_POS
    dist = math.hypot(x - cx, y - cy)
    if dist == 0:
        return
        
    steps = max(1, int(dist / step_size))
    dx = (x - cx) / steps
    dy = (y - cy) / steps
    
    for i in range(1, steps + 1):
        next_x = int(cx + dx * i)
        next_y = int(cy + dy * i)
        ep_arm.moveto(x=next_x, y=next_y).wait_for_completed()
        
    CURRENT_ARM_POS = (x, y)

def gripper_close(ep_gripper):
    print(f"[GRIPPER] close (power={GRIPPER_POWER})")
    ep_gripper.close(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()

def gripper_open(ep_gripper):
    print(f"[GRIPPER] open (power={GRIPPER_POWER})")
    ep_gripper.open(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()

def setup_robot():
    print("Loading YOLO...")
    w = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmsc477_yolo/runs/detect/lego_gpu_train3/weights/best.pt")
    if not os.path.exists(w):
        print(f"ERROR: weights not found at {w}")
        sys.exit(1)
    model = YOLO(w)

    print("Connecting to robot...")
    robomaster.config.ROBOT_IP_STR = "192.168.50.116"
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    ep_chassis = ep_robot.chassis
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    ep_camera = ep_robot.camera

    arm_moveto(ep_arm, ARM_HOME, "home")
    gripper_open(ep_gripper)

    try:
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    except Exception as e:
        print(f"ERROR: camera failed: {e}")
        ep_robot.close()
        sys.exit(1)

    print("Connected!\n")
    return ep_robot, ep_chassis, ep_arm, ep_gripper, ep_camera, model

def shutdown_robot(ep_robot, ep_chassis, ep_camera):
    print("Cleaning up...")
    for fn in [lambda: stop(ep_chassis),
               lambda: ep_camera.stop_video_stream(),
               lambda: ep_robot.close()]:
        try:
            fn()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("Shut down.")

# Odometry & Navigation Helpers (Tracking Theta)
def rotate_by(ep_chassis, delta_deg):    
    global current_theta
    if abs(delta_deg) < 0.3:
        return
    new_theta = current_theta + delta_deg
    print(f"[ROTATE] {delta_deg:+.1f} deg -> theta={new_theta:.1f} deg")
    ep_chassis.move(x=0, y=0, z=delta_deg, z_speed=SCAN_SPEED).wait_for_completed()
    current_theta = new_theta
    time.sleep(0.15)

def rotate_to(ep_chassis, target_deg):
    global current_theta
    delta = target_deg - current_theta
    while delta > 180: delta -= 360
    while delta < -180: delta += 360
    rotate_by(ep_chassis, delta)

def drive_straight(ep_chassis, dist_m, speed=None):
    if speed is None: speed = MOVE_SPEED
    if abs(dist_m) < 0.005: return
    print(f"[DRIVE] {dist_m:+.4f} m at {speed:.3f} m/s")
    ep_chassis.move(x=dist_m, y=0, z=0, xy_speed=abs(speed)).wait_for_completed()
    time.sleep(0.15)

# Core Actions (Using robot_runner.py's proportional methodology)
def scan_and_center(ep_chassis, ep_camera, model):
    """
    Search for tower 360 degrees and center.
    Track the exact theta we turn using rotate_by.
    """
    print("[SCAN] Searching for tower...")
    scanned_total = 0.0
    MAX_SCAN_DEG  = 360.0
    global current_theta

    t_prev = time.time()

    while scanned_total < MAX_SCAN_DEG:
        frame = get_frame(ep_camera)
        if frame is None:
            t_prev = time.time()
            continue

        bboxes = run_yolo(model, frame)
        if not bboxes:
            stop(ep_chassis)
            rotate_by(ep_chassis, abs(SCAN_STEP))
            scanned_total += abs(SCAN_STEP)
            write_text_on_video_feed(frame, f"Scanning: {scanned_total} deg", color=(0,0,255))
            if show(frame): raise KeyboardInterrupt
            t_prev = time.time()
            continue

        if len(bboxes) > 1:
            bboxes.sort(key=lambda b: b['cx'], reverse=True)
            
        target = bboxes[0]
        draw_box(frame, target, label="Target")
        err_x = target['cx'] - FRAME_CX

        if abs(err_x) <= CENTER_TOL:
            stop(ep_chassis)
            print(f"[SCAN] Centered! theta={current_theta:.1f} deg")
            write_text_on_video_feed(frame, "CENTERED!", color=(0,255,255))
            show(frame)
            time.sleep(0.3)
            return

        # Continuous Proportional Controller mapping exactly to robot_runner.py
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))
        
        # Drive continuously
        drive(ep_chassis, x=0.0, y=0.0, z=yaw_speed)
        
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now

        # Because drive_speed uses opposite coordinate limits mapping we adjust theta natively
        # Positive yaw_speed = Turn right (negative degrees mathematically depending on setting)
        # So integration tracks drift ensuring home alignment persists.
        current_theta -= yaw_speed * dt
        scanned_total += abs(yaw_speed * dt)

        write_text_on_video_feed(frame, f"Centering error: {err_x:.1f}px")
        if show(frame): 
            raise KeyboardInterrupt

    raise RuntimeError("[SCAN] No tower found after full 360 deg")


def approach_tower(ep_chassis, ep_camera, model):
    """
    Two-phase approach:
      1. Proportional coarse error response from robot_runner.py Step1
      2. Fine drive-in from robot_runner.py Step3
    """
    print("[APPROACH] Driving toward tower...")
    fwd_distance = 0.0
    t_prev = time.time()
    
    # Coarse Phase
    print("[APPROACH] Coarse phase...")
    while True:
        frame = get_frame(ep_camera)
        if frame is None: continue
        
        bboxes = run_yolo(model, frame)
        if not bboxes:
            stop(ep_chassis)
            t_prev = time.time()
            write_text_on_video_feed(frame, "Coarse - no detection", color=(0,0,255))
            if show(frame): raise KeyboardInterrupt
            continue
            
        t1 = bboxes[0]
        draw_box(frame, t1, label="Target")
        
        err_x = t1['cx'] - FRAME_CX
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))
        
        if t1['y2'] >= APPROACH_STOP_BOTTOM:
            stop(ep_chassis)
            write_text_on_video_feed(frame, f"Coarse Done bot={t1['y2']:.0f}", color=(0, 255, 255))
            show(frame)
            print(f"[APPROACH] Coarse done. bot={t1['y2']:.0f}")
            break
            
        err_y = APPROACH_STOP_BOTTOM - t1['y2']
        # Proportional fwd_speed based on vertical error distance (from robot_runner)
        fwd_speed = max(0.0, min(FWD_MAX, err_y * FWD_KP))
        
        t_now = time.time()
        fwd_distance += fwd_speed * (t_now - t_prev)
        t_prev = t_now
        
        drive(ep_chassis, x=fwd_speed, y=0.0, z=yaw_speed)
        
        write_text_on_video_feed(frame, f"Coarse FWD:{fwd_speed:.2f} DIST:{fwd_distance:.2f} BOT:{t1['y2']:.0f}")
        if show(frame): raise KeyboardInterrupt

    # Fine Phase
    print("[APPROACH] Fine phase...")
    t_prev = time.time()
    while True:
        frame = get_frame(ep_camera)
        if frame is None: continue
        
        bboxes = run_yolo(model, frame)
        if not bboxes:
            stop(ep_chassis)
            t_prev = time.time()
            write_text_on_video_feed(frame, "Fine - no detection", color=(0,0,255))
            if show(frame): raise KeyboardInterrupt
            continue
            
        t1 = bboxes[0]
        draw_box(frame, t1, color=(0, 255, 255), label="Target Fine")
        
        err_x = t1['cx'] - FRAME_CX
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))
        
        if t1['y1'] > GRAB_TOP_THRESHOLD:
            stop(ep_chassis)
            write_text_on_video_feed(frame, f"GRAB top={t1['y1']:.0f}", color=(0, 255, 255))
            show(frame)
            print(f"[APPROACH] Grab position reached. top={t1['y1']:.0f}. Total dist: {fwd_distance:.3f}m")
            return fwd_distance
            
        fwd_speed = 0.07  # from robot_runner.py step3
        t_now = time.time()
        fwd_distance += fwd_speed * (t_now - t_prev)
        t_prev = t_now
        
        drive(ep_chassis, x=fwd_speed, y=0.0, z=yaw_speed)
        
        write_text_on_video_feed(frame, f"Fine FWD:{fwd_speed:.2f} DIST:{fwd_distance:.2f} TOP:{t1['y1']:.0f}")
        if show(frame): raise KeyboardInterrupt


def grab_tower(ep_arm, ep_gripper):
    """Lower arm, close gripper, lift to carry height."""
    arm_moveto(ep_arm, ARM_PICKUP, "lower-to-grab")
    gripper_close(ep_gripper)
    arm_moveto(ep_arm, ARM_CARRY, "lift-to-carry")

def place_tower(ep_arm, ep_gripper):
    """Lower arm to place height, open gripper to release tower."""
    arm_moveto(ep_arm, ARM_PICKUP, "lower-to-place")
    time.sleep(0.3)
    gripper_open(ep_gripper)

def return_to_home(ep_chassis, dist_m):
    """
    Back straight up dist_m meters, then snap heading back to 0 degrees.
    """
    print(f"[HOME] Backing up {dist_m:.3f}m to return to origin")
    drive_straight(ep_chassis, -dist_m)
    #rotate_to(ep_chassis, 0.0)
    print(f"[HOME] At origin, theta={current_theta:.1f} deg")

# High Level Sequence corresponding to lab2 tracking approach
def step_a_find_and_grab_tower1(ep_chassis, ep_arm, ep_gripper, ep_camera, model):
    print("\n--- step a: find and grab tower 1 ---")
    arm_moveto(ep_arm, ARM_HOME, "prep-home")
    gripper_open(ep_gripper)

    scan_and_center(ep_chassis, ep_camera, model)
    theta1 = current_theta
    print(f"[step-a] tower 1 is at theta={theta1:.1f} deg")

    dist1 = approach_tower(ep_chassis, ep_camera, model)
    grab_tower(ep_arm, ep_gripper)

    print(f"[step-a] done: grabbed tower 1 at theta={theta1:.1f} deg  dist={dist1:.3f}m")
    return theta1, dist1

def step_b_place_tower1_at_temp(ep_chassis, ep_arm, ep_gripper):
    print("\n--- step b: place tower 1 at temp location ---")
    rotate_to(ep_chassis, TEMP_ANGLE)
    drive_straight(ep_chassis, TEMP_DISTANCE)

    place_tower(ep_arm, ep_gripper)

    drive_straight(ep_chassis, -(TEMP_DISTANCE))
    arm_moveto(ep_arm, ARM_HOME, "retract-after-temp-drop")

    rotate_to(ep_chassis, 0.0)

    print(f"[step-b] temp drop done at angle={TEMP_ANGLE} deg, dist={TEMP_DISTANCE}m. Back at home")

def step_c_find_and_grab_tower2(ep_chassis, ep_arm, ep_gripper, ep_camera, model):
    print("\n--- step c: find and grab tower 2 ---")
    arm_moveto(ep_arm, ARM_HOME, "prep-home")
    gripper_open(ep_gripper)

    scan_and_center(ep_chassis, ep_camera, model)
    theta2 = current_theta
    print(f"[step-c] tower 2 is at theta={theta2:.1f} deg")

    dist2 = approach_tower(ep_chassis, ep_camera, model)
    grab_tower(ep_arm, ep_gripper)

    print(f"[step-c] done: grabbed tower 2 at theta={theta2:.1f} deg  dist={dist2:.3f}m")
    return theta2, dist2

def step_d_place_tower2_at_t1_spot(ep_chassis, ep_arm, ep_gripper, theta1, dist1):
    print(f"\n--- step d: carry tower 2 to t1's original spot (theta={theta1:.1f} deg, dist={dist1:.3f}m) ---")
    rotate_to(ep_chassis, theta1)
    drive_straight(ep_chassis, dist1)

    place_tower(ep_arm, ep_gripper)

    clear = min(0.06, dist1)
    drive_straight(ep_chassis, -clear, speed=SLOW_BACKUP_SPEED)
    arm_moveto(ep_arm, ARM_HOME, "retract-after-t1-spot")

    drive_straight(ep_chassis, -(dist1 - clear))
    rotate_to(ep_chassis, 0.0)
    print("[step-d] tower 2 placed at t1 spot, back at home")

def step_e_find_and_grab_temp_tower1(ep_chassis, ep_arm, ep_gripper, ep_camera, model):
    print("\n--- step e: find and grab temp tower 1 ---")
    arm_moveto(ep_arm, ARM_HOME, "prep-home")
    gripper_open(ep_gripper)

    # Note: Lab2 approach does a rotate first to look at temp area, then scan
    print("[step-e] rotating toward temp region to start scan")
    rotate_to(ep_chassis, TEMP_ANGLE_RETURN)

    scan_and_center(ep_chassis, ep_camera, model)
    theta_found = current_theta
    print(f"[step-e] temp tower 1 found at theta={theta_found:.1f} deg")

    dist_found = approach_tower(ep_chassis, ep_camera, model)
    grab_tower(ep_arm, ep_gripper)

    print(f"[step-e] done: grabbed temp tower 1 at theta={theta_found:.1f} deg  dist={dist_found:.3f}m")
    return theta_found, dist_found

def step_f_place_tower1_at_t2_spot(ep_chassis, ep_arm, ep_gripper, theta2, dist2):
    print(f"\n--- step f: carry tower 1 to t2's original spot (theta={theta2:.1f} deg, dist={dist2:.3f}m) ---")
    rotate_to(ep_chassis, theta2)
    drive_straight(ep_chassis, dist2)

    place_tower(ep_arm, ep_gripper)

    clear = min(0.06, dist2)
    drive_straight(ep_chassis, -clear, speed=SLOW_BACKUP_SPEED)
    arm_moveto(ep_arm, ARM_HOME, "retract-final")

    drive_straight(ep_chassis, -(dist2 - clear))
    rotate_to(ep_chassis, 0.0)

    print("[step-f] tower 1 placed at t2 spot, back at home")
    print("[step-f] *** swap complete ***")

# Main Execution
def main():
    ep_robot, ep_chassis, ep_arm, ep_gripper, ep_camera, model = setup_robot()

    try:
        # a — find tower 1 (the closer/larger one), grab it
        theta1, dist1 = step_a_find_and_grab_tower1(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
        return_to_home(ep_chassis, dist1)

        # b — drop tower 1 behind us at the temp location
        step_b_place_tower1_at_temp(ep_chassis, ep_arm, ep_gripper)

        # c — find tower 2 and grab it
        theta2, dist2 = step_c_find_and_grab_tower2(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
        return_to_home(ep_chassis, dist2)

        # d — place tower 2 at tower 1's original position
        step_d_place_tower2_at_t1_spot(ep_chassis, ep_arm, ep_gripper, theta1, dist1)

        # e — find the temp tower 1
        theta_temp, dist_temp = step_e_find_and_grab_temp_tower1(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
        return_to_home(ep_chassis, dist_temp)

        # f — place tower 1 at tower 2's original position — swap done
        step_f_place_tower1_at_t2_spot(ep_chassis, ep_arm, ep_gripper, theta2, dist2)

        print("\n========================================")
        print("  Lego tower swap complete!")
        print("========================================\n")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except RuntimeError as e:
        print(f"\nRuntime error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_robot(ep_robot, ep_chassis, ep_camera)

if __name__ == "__main__":
    main()
