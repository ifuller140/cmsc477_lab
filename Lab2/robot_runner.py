import cv2
import time
import os
import sys
import math
from ultralytics import YOLO
import robomaster
from robomaster import robot, camera

# Constants
# Yaw / centering
YAW_KP   = 0.2              # deg per horizontal pixel error
YAW_MAX  = 40.0              # deg clamp
FRAME_CX = 320

# Forward proportional speed
FWD_KP   = 0.0015            # m/s per pixel error
FWD_KP_ORBIT   = 0.0005            # m/s per pixel error
FWD_MAX  = 0.20              # m/s clamp

# Step 1 - initial approach stop condition
APPROACH_STOP_BOTTOM = 250   # px: stop when bbox bottom reaches this row

# Step 2 - orbit
ORBIT_STRAFE_SPEED  = 0.05   # m/s; positive y = strafe LEFT = CCW orbit when facing tower
ORBIT_ALIGN_TOL     = 8      # px: T2 must be within this many px of frame centre
SECONDARY_MAX_BOTTOM = 150   # px: far-tower bbox bottom must be above this (peeking over T1)

# Step 3 - final approach and grab
GRAB_TOP_THRESHOLD  = 95     # px: grab when bbox TOP edge (xyxy[1]) > this value
GRIPPER_POWER       = 50
GRIPPER_HOLD        = 1.0    # s — hold after issuing open/close (must be ≤ 1 s)

# Step 4 - approach the second tower
# Stop when the LOWEST y1 (top edge) across all detected bboxes is greater than this.
T2_TOP_Y      = 128    # px: stop when every bbox top-edge is below this row
T2_FWD_MAX          = 0.15   # m/s cap while approaching T2
HELD_TOWER_MAX_AREA = 4000   # px²: bboxes smaller than this are ignored (held-scrap)

# Arm positions  (x = mm forward from body, y = mm vertical)
ARM_HOME      = (185, -80)   # retracted standby
ARM_PICKUP    = (185, -50)   # slight lift while carrying   
ARM_CARRY     = (185, -40)   # slight lift while carrying
ARM_HIGH      = (175,  140)  # highest — used during swap lift
ARM_LOW_TRAP_1  = (175,   50)  # low enough to trap / push T2
ARM_LOW_TRAP_2  = (200,   50)  # low enough to trap / push T2
ARM_LOW_TRAP_3  = (200,   -50)  # low enough to trap / push T2

ARM_PICKUP    = (185,  -50)  # drop height
ARM_INSIDE_TRAP = (165, -40)
# Step 5 - swap towers
SWAP_NUDGE_DIST  = 0.22      # m: creep forward to position T1 over T2
SWAP_PUSH_DIST   = 0.155      # m: reverse to push T2 out of its spot
SWAP_CLEAR_DIST  = 0.1      # m: back away after dropping T1
SWAP_SPEED       = 0.3      # m/s for all swap chassis.move calls

#Step 6 - pick up tower 2
step6_dist = 0.0

# Helper Functions

# Gets the current frame from the camera
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

# Runs Yolo on the current camera frame
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
        bboxes.sort(key=lambda b: b['area'], reverse=True) # make the largest one first
        return bboxes
    except Exception as e:
        print(f"yolo error: {e}")
        return []

# Draws a box around the detected object
def draw_box(frame, b, color=(0, 255, 0), label=""):
    cv2.rectangle(frame, (int(b['x1']), int(b['y1'])), (int(b['x2']), int(b['y2'])), color, 2)
    if label:
        cv2.putText(frame, label, (int(b['x1']), int(b['y1']) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Displays the text on the screen
def write_text_on_video_feed(frame, text, row=30, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

# Displays the live screen frame
def show(frame, title="RoboMaster"):
    cv2.imshow(title, frame)
    return (cv2.waitKey(1) & 0xFF) == ord('q')

# Driver control that drives the robot
def drive(ep_chassis, x=0.0, y=0.0, z=0.0):
    ep_chassis.drive_speed(
        x=max(-FWD_MAX, min(FWD_MAX, x)),
        y=max(-0.5,     min(0.5,     y)),
        z=max(-YAW_MAX, min(YAW_MAX, z)),
    )

# Stops the robot
def stop(ep_chassis):
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)

# Global to track the arm's position
CURRENT_ARM_POS = None

# Moves the arm to a specific position slowly
def arm_moveto(ep_arm, pos, label="", step_size=10):
    global CURRENT_ARM_POS
    x, y = pos
    print(f"Arm {label} → moveto(x={x}, y={y})")
    
    # If starting up and current position is unknown, just jump there
    if CURRENT_ARM_POS is None:
        ep_arm.moveto(x=x, y=y).wait_for_completed()
        CURRENT_ARM_POS = (x, y)
        return
        
    cx, cy = CURRENT_ARM_POS
    dist = math.hypot(x - cx, y - cy)
    if dist == 0:
        return
        
    # Interpolate from current position to target
    steps = max(1, int(dist / step_size))
    dx = (x - cx) / steps
    dy = (y - cy) / steps
    
    for i in range(1, steps + 1):
        next_x = int(cx + dx * i)
        next_y = int(cy + dy * i)
        ep_arm.moveto(x=next_x, y=next_y).wait_for_completed()
        
    CURRENT_ARM_POS = (x, y)

# Closes the gripper
def gripper_close(ep_gripper):
    print(f"Gripper close (power={GRIPPER_POWER})")
    ep_gripper.close(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()

# Opens the gripper
def gripper_open(ep_gripper):
    print(f"Gripper open (power={GRIPPER_POWER})")
    ep_gripper.open(power=GRIPPER_POWER)
    time.sleep(GRIPPER_HOLD)
    ep_gripper.pause()

# Main setup for the robot
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
    print("cleaning up...")
    for fn in [lambda: stop(ep_chassis),
               lambda: ep_camera.stop_video_stream(),
               lambda: ep_robot.close()]:
        try:
            fn()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("Shut down.")



#  Step 1: approach the first tower
def step1(ep_chassis, ep_camera, model):

    # Theory: find largest bbox which is the closest tower, center it and drive forward until
    # bbox bottom >= APPROACH_STOP_BOTTOM
    print("-------- Step 1 --------")
    print("Approaching the first tower\n")

    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        bboxes = run_yolo(model, frame)

        if not bboxes:
            stop(ep_chassis)
            write_text_on_video_feed(frame, "Step 1 - no detection", color=(0, 0, 255))
            if show(frame):
                raise KeyboardInterrupt
            continue

        t1 = bboxes[0]  # largest = closest
        draw_box(frame, t1, color=(0, 255, 0), label="Tower 1")

        err_x = t1['cx'] - FRAME_CX
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP)) # positive z = turn left and invert so we chase the target

        if t1['y2'] >= APPROACH_STOP_BOTTOM: # stop when the bottom of the bbox is at the bottom of the frame
            stop(ep_chassis)
            write_text_on_video_feed(frame, f"Step 1 - DONE  bot={t1['y2']:.0f}", color=(0, 255, 255))
            show(frame)
            print(f"[STEP 1] Done. bbox bottom={t1['y2']:.1f}")
            return

        err_y     = APPROACH_STOP_BOTTOM - t1['y2']
        print(f'{err_y}')
        fwd_speed = max(0.0, min(FWD_MAX, err_y * FWD_KP))
        drive(ep_chassis, x=fwd_speed, y=0.0, z=yaw_speed)

        write_text_on_video_feed(frame, f"Step 1 - fwd={fwd_speed:.2f}  yaw={yaw_speed:.1f}  bot={t1['y2']:.0f}")
        if show(frame):
            raise KeyboardInterrupt


#  Step 2: orbit until tower 2 is directly behind tower 1
def step2(ep_chassis, ep_camera, model):
    
    # Orbit counter clockwise and stop when a secondary bbox (tower 2) is centred within ORBIT_ALIGN_TOL px.

    print("Orbiting to align towers...")
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        bboxes = run_yolo(model, frame)

        if not bboxes:
            stop(ep_chassis)
            write_text_on_video_feed(frame, "Step 2 - waiting for tower 1", color=(0, 0, 255))
            if show(frame):
                raise KeyboardInterrupt
            continue

        t1 = bboxes[0]
        draw_box(frame, t1, color=(0, 255, 0), label="T1")

        err_x     = t1['cx'] - FRAME_CX
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        # Minor distance correction to stay ~0.2 m from T1 while orbiting
        err_y     = APPROACH_STOP_BOTTOM - t1['y2']
        fwd_corr  = max(-0.05, min(0.05, err_y * FWD_KP_ORBIT))

        # Check for second tower peeking over the first tower in the upper frame
        aligned = False
        for b in bboxes[1:]:
            if b['y2'] < SECONDARY_MAX_BOTTOM: # upper frame = far tower
                draw_box(frame, b, color=(255, 100, 0), label="T2?")
                dist = abs(b['cx'] - FRAME_CX)
                write_text_on_video_feed(frame, f"Step 2 - T2 cx={b['cx']:.0f}  err={dist:.0f}", row=60)
                if dist < ORBIT_ALIGN_TOL:
                    aligned = True
                    break

        if aligned:
            stop(ep_chassis)
            write_text_on_video_feed(frame, "Step 2 - ALIGNED!", color=(0, 255, 255))
            show(frame)
            print("[STEP 2] Towers aligned. Done.")
            return

        # Positive y = strafe left = CCW orbit (same as track_lego.py)
        drive(ep_chassis, x=fwd_corr, y=ORBIT_STRAFE_SPEED, z=yaw_speed)
        write_text_on_video_feed(frame, f"Step 2 - yaw={yaw_speed:.1f}  strafe={ORBIT_STRAFE_SPEED}  fwd={fwd_corr:.2f}")
        if show(frame):
            raise KeyboardInterrupt


#  Step 3: final approach and grab tower 1
def step3(ep_chassis, ep_arm, ep_gripper, ep_camera, model):
    """
    Centre on T1 and drive forward until bbox top-edge (y1) < GRAB_TOP_THRESHOLD
    (camera sees tower-top very close → time to grab).
    Then: close gripper, lift arm to ARM_CARRY.
    """
    print("[STEP 3] Final approach and grab T1...")
    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        bboxes = run_yolo(model, frame)

        if not bboxes:
            stop(ep_chassis)
            write_text_on_video_feed(frame, "Step 3 - no detection", color=(0, 0, 255))
            if show(frame):
                raise KeyboardInterrupt
            continue

        t1 = bboxes[0]
        draw_box(frame, t1, color=(0, 255, 0), label="T1-GRAB")

        err_x     = t1['cx'] - FRAME_CX
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if t1['y1'] > GRAB_TOP_THRESHOLD:
            stop(ep_chassis)
            write_text_on_video_feed(frame, f"Step 3 - GRAB  top={t1['y1']:.0f}", color=(0, 255, 255))
            show(frame)
            print(f"[STEP 3] Close enough (top={t1['y1']:.1f}). Grabbing...")
            arm_moveto(ep_arm, ARM_PICKUP, "pickup")
            gripper_close(ep_gripper)
            arm_moveto(ep_arm, ARM_CARRY, "carry")
            print("[STEP 3] Done.")
            return

        err_y     = t1['y1'] - GRAB_TOP_THRESHOLD    # how much further to go
        #fwd_speed = max(0.0, min(FWD_MAX * 0.6, err_y * FWD_KP))
        fwd_speed = 0.07
        print(f'{fwd_speed}')
        drive(ep_chassis, x=fwd_speed, y=0.0, z=yaw_speed)

        write_text_on_video_feed(frame, f"Step 3 - fwd={fwd_speed:.2f}  yaw={yaw_speed:.1f}  top={t1['y1']:.0f}")
        if show(frame):
            raise KeyboardInterrupt


#  Step 4: approach tower 2 and keep track of distance
def step4(ep_chassis, ep_camera, model):
    # Drive toward T2.  Bboxes smaller than HELD_TOWER_MAX_AREA are ignored
    # (tiny top of T1 visible in gripper).  Largest remaining bbox = T2.

    # Accumulates fwd_speed × dt every loop tick.
    # Returns total forward distance driven (metres).
    print("Step 4 - Approaching tower 2...")
    fwd_distance    = 0.0
    t_prev          = time.time()
    no_detect       = 0
    MAX_NO_DETECT   = 30

    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        bboxes = run_yolo(model, frame)

        if not bboxes and len(bboxes) >= 2:
            stop(ep_chassis)
            no_detect += 1
            write_text_on_video_feed(frame, f"Step 4 - no bboxes ({no_detect})", color=(0, 0, 255))
            if show(frame):
                raise KeyboardInterrupt
            if no_detect >= MAX_NO_DETECT:
                raise RuntimeError("Step 4 - T2 lost for too long")
            time.sleep(0.05)
            t_prev = time.time()  # don't accumulate while stopped
            continue

        no_detect = 0
        
        # Assign tower 2 to the bounding box with the highest top bar (minimum y1 value)
        
        t2 = min(bboxes, key=lambda b: b['y1']) # the highest bounding box in frame
        
        draw_box(frame, t2, color=(255, 120, 0), label="T2")

        err_x = t2['cx'] - FRAME_CX
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        # We move closer to T2 until its top reaches the threshold T2_TOP_Y
        # (greater y value = lower on screen = closer to robot)
        if t2['y1'] >= T2_TOP_Y and t2['y1'] < T2_TOP_Y + 10:
            stop(ep_chassis)
            write_text_on_video_feed(
                frame,
                f"Step 4 - DONE  t2_y1={t2['y1']:.0f}  dist={fwd_distance:.3f}m",
                color=(0, 255, 255)
            )
            show(frame)
            print(f"Step 4 - T2 reached. t2_y1={t2['y1']:.1f}  odometry={fwd_distance:.3f} m")
            return fwd_distance

        # Error style control: Drive faster when t2 is still high in the frame (small y1)
        err_y = T2_TOP_Y - t2['y1']  # positive while still far
        #fwd_speed = max(0.0, min(T2_FWD_MAX, err_y * FWD_KP))
        fwd_speed = 0.05

        # Dead-reckoning: accumulate speed * dt
        t_now = time.time()
        fwd_distance += fwd_speed * (t_now - t_prev)
        t_prev = t_now

        drive(ep_chassis, x=fwd_speed, y=0.0, z=yaw_speed)
        write_text_on_video_feed(
            frame,
            f"Step 4 - fwd={fwd_speed:.2f}  dist={fwd_distance:.3f}m  t2_y1={t2['y1']:.0f}"
        )
        if show(frame):
            raise KeyboardInterrupt

        


#  Step 5: swap towers (pure odometry)
def step5(ep_chassis, ep_arm, ep_gripper):
      # 1. Raise arm to ARM_HIGH        (T1 lifted above T2's height)
      # 2. Nudge forward SWAP_NUDGE_DIST (T1 in air is directly above T2)
      # 3. Lower arm to ARM_LOW_TRAP    (T1 descends, trapping T2)
      # 4. Reverse SWAP_PUSH_DIST       (T1 arm pushes T2 backward along floor)
      # 5. Open gripper                 (drop T1 — now in T2's old general area)
      # 6. Raise arm to ARM_HIGH        (arm lifts over T2)
      # 7. Reverse SWAP_CLEAR_DIST      (robot clears both towers)

    # Returns net displacement (metres) relative to the step-4 stop position.
    # (Always negative — robot ends up behind where it stopped at T2.)

    print("Step 5 - Swap sequence...")
    arm_moveto(ep_arm, ARM_HIGH, "raise-for-swap")
    time.sleep(1)

    print(f"Step 5 - Nudge forward {SWAP_NUDGE_DIST} m")
    ep_chassis.move(x=SWAP_NUDGE_DIST, y=0, z=0, xy_speed=SWAP_SPEED).wait_for_completed()
    time.sleep(1)

    arm_moveto(ep_arm, ARM_LOW_TRAP_1, "lower-trap-1")
    arm_moveto(ep_arm, ARM_LOW_TRAP_2, "lower-trap-2")
    arm_moveto(ep_arm, ARM_LOW_TRAP_3, "lower-trap-3")
    time.sleep(1)   # brief wait before we push the tower

    print(f"Step 5 - Push backward {SWAP_PUSH_DIST} m")
    ep_chassis.move(x=-SWAP_PUSH_DIST, y=0, z=0, xy_speed=SWAP_SPEED).wait_for_completed()
    
    gripper_open(ep_gripper)    # drop T1
    
    # arm_moveto(ep_arm, ARM_LOW_TRAP_1, "raise-over-T2")
    ep_chassis.move(x=0.01, y=0, z=0, xy_speed=0.1).wait_for_completed()

    arm_moveto(ep_arm, ARM_INSIDE_TRAP, "trap-inside")

    net = SWAP_NUDGE_DIST - SWAP_PUSH_DIST - SWAP_CLEAR_DIST   # e.g. 0.05-0.15-0.10 = -0.20
    print(f"Step 5 - Done. Net displacement from step-4 stop: {net:.3f} m")
    return net


#  Step 6: pick up tower 2 (accumulate memorizing motion distance)
def step6(ep_chassis, ep_arm, ep_gripper, ep_camera, model):
    # The idea: T2 is now near the robot (pushed during swap).  Home the arm, open gripper,
    # then approach and grab T2 using the same bbox logic as step 3
    # Returns forward distance driven in meters for memorizing its motion

    print("Step 6 - Picking up tower 2...")
    arm_moveto(ep_arm, ARM_HOME, "home-for-pickup")
    gripper_open(ep_gripper)
    time.sleep(0.2)

    fwd_distance  = 0.0
    t_prev        = time.time()
    no_detect     = 0
    MAX_NO_DETECT = 40

    while True:
        frame = get_frame(ep_camera)
        if frame is None:
            continue

        bboxes = run_yolo(model, frame)

        if len(bboxes) < 1:
            stop(ep_chassis)
            no_detect += 1
            write_text_on_video_feed(frame, f"STEP 6 | no detection ({no_detect})", color=(0, 0, 255))
            if show(frame):
                raise KeyboardInterrupt
            if no_detect >= MAX_NO_DETECT:
                raise RuntimeError("[STEP 6] lost T2 during pickup approach")
            t_prev = time.time()
            continue

        bboxes.sort(key=lambda b: b['area'], reverse=True)

        no_detect = 0
        t2 = bboxes[0]
        draw_box(frame, t2, color=(0, 200, 255), label="T2-PICK")

        err_x     = t2['cx'] - FRAME_CX
        yaw_speed = max(-YAW_MAX, min(YAW_MAX, err_x * YAW_KP))

        if t2['y1'] > GRAB_TOP_THRESHOLD:
            stop(ep_chassis)
            write_text_on_video_feed(frame, f"Step 6 - GRAB  top={t2['y1']:.0f}", color=(0, 255, 255))
            show(frame)
            print(f"Step 6 - Close enough. Grabbing T2...")
            gripper_close(ep_gripper)
            arm_moveto(ep_arm, ARM_CARRY, "carry-T2")
            print(f"Step 6 - Done. Step-6 odometry = {fwd_distance:.3f} m")
            return fwd_distance

        err_y     = t2['y1'] - GRAB_TOP_THRESHOLD
        fwd_speed = max(0.0, min(FWD_MAX * 0.6, err_y * FWD_KP))

        t_now         = time.time()
        fwd_distance += fwd_speed * (t_now - t_prev)
        t_prev        = t_now

        drive(ep_chassis, x=fwd_speed, y=0.0, z=yaw_speed)
        write_text_on_video_feed(frame, f"Step 6 - fwd={fwd_speed:.2f}  dist={fwd_distance:.3f}m  top={t2['y1']:.0f}")
        if show(frame):
            raise KeyboardInterrupt


#  Step 7: return and drop tower 2 at tower 1's original spot
def step7(ep_chassis, ep_arm, ep_gripper, step4_dist, swap_net, step6_dist):

    return_dist = step4_dist + swap_net + step6_dist
    return_dist = max(0.0, return_dist)   # safety: never drive forward here

    print(f"\nStep 7 - Returning to T1 original spot.")
    print(f"  step4={step4_dist:.3f}  swap_net={swap_net:.3f}  step6={step6_dist:.3f}")
    print(f"  → driving backward {return_dist:.3f} m")
    
    

    if return_dist > 0.01:
        ep_chassis.move(x=-return_dist, y=0, z=0, xy_speed=0.08).wait_for_completed()

    arm_moveto(ep_arm, ARM_LOW_TRAP_1, "drop-lower")
    time.sleep(1)

    # Back up a little to clear the droped tower
    ep_chassis.move(x=-0.08, y=0, z=0, xy_speed=SWAP_SPEED).wait_for_completed()

    gripper_open(ep_gripper)    # drop T2

    print("Step 7 - T2 droped at T1's original spot. Done.")



def main():
    ep_robot, ep_chassis, ep_arm, ep_gripper, ep_camera, model = setup_robot()

    try:
        
        # ── 1. Find and close on the nearest lego tower ───────────────────────
        step1(ep_chassis, ep_camera, model)

        # ── 2. Orbit CCW until tower 2 is directly behind tower 1 ────────────
        step2(ep_chassis, ep_camera, model)
        
        # ── 3. Final centred drive-in; grab tower 1 and lift ─────────────────
        step3(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
        
        # ── 4. Approach tower 2 while accumulating odometry ──────────────────
        step4_dist = step4(ep_chassis, ep_camera, model)

        # ── 5. Physical swap manoeuvre (pure odometry) ────────────────────────
        swap_net = step5(ep_chassis, ep_arm, ep_gripper)

        '''
        # ── 6. Pick up tower 2, accumulate return odometry ───────────────────
        step6_dist = step6(ep_chassis, ep_arm, ep_gripper, ep_camera, model)
        '''
        # ── 7. Drive back to tower 1's spot and drop tower 2 ────────
        step7(ep_chassis, ep_arm, ep_gripper, step4_dist, swap_net, step6_dist)
        
        print("\n Lego tower swap complete!")

    except KeyboardInterrupt:
        print("\n Keyboard interrupt.")
    except RuntimeError as e:
        print(f"\n Runtime error: {e}")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_robot(ep_robot, ep_chassis, ep_camera)


if __name__ == "__main__":
    main()
