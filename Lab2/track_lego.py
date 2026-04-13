import cv2
import time
import os
import sys
from ultralytics import YOLO
import robomaster
from robomaster import robot
from robomaster import camera

def main():
    # ── Load YOLO model ───────────────────────────────────────────────────────
    print('Loading YOLO model...')
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cmsc477_yolo/runs/detect/lego_gpu_train3/weights/best.pt"
    )
    if not os.path.exists(file_path):
        print(f"ERROR: YOLO weights not found at:\n  {file_path}")
        sys.exit(1)

    model = YOLO(file_path)

    # ── Connect to robot ──────────────────────────────────────────────────────
    print('Connecting to RoboMaster...')
    # FIX: correct config key is ROBOT_IP_STR, not ROBOT_IP_STRING
    robomaster.config.ROBOT_IP_STR = "192.168.50.116"
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        sys.exit(1)

    ep_camera  = ep_robot.camera
    ep_arm     = ep_robot.robotic_arm
    ep_chassis = ep_robot.chassis

    try:
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    except Exception as e:
        print(f"ERROR: Could not start video stream: {e}")
        ep_robot.close()
        sys.exit(1)

    print('Starting tracking loop. Press "q" to quit.')

    FRAME_W        = 640
    FRAME_CENTER_X = FRAME_W / 2.0
    # Suppress YOLO verbose output after first prediction
    _yolo_warmed_up = False

    # Check whether a display is available (headless Pi vs desktop)
    display_available = True

    try:
        while True:
            # ── Read frame ────────────────────────────────────────────────────
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Exception as e:
                print(f"WARNING: Frame read failed: {e} — retrying...")
                time.sleep(0.1)
                continue

            if frame is None:
                # Camera not ready yet; wait a tick
                time.sleep(0.05)
                continue

            # ── Run inference ─────────────────────────────────────────────────
            try:
                result = model.predict(source=frame, show=False, verbose=False)[0]
            except Exception as e:
                print(f"WARNING: YOLO inference error: {e}")
                time.sleep(0.1)
                continue

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                # Find the largest bounding box (likely the closest target)
                try:
                    best_box = max(
                        boxes,
                        key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
                    )
                    xyxy = best_box.xyxy.cpu().numpy().flatten()
                except Exception as e:
                    print(f"WARNING: Box extraction failed: {e}")
                    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)
                    continue

                # Validate box has 4 coordinates
                if len(xyxy) < 4:
                    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)
                    continue

                width  = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]
                cx     = xyxy[0] + width / 2.0

                # Draw detection box
                cv2.rectangle(
                    frame,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    color=(0, 255, 0), thickness=3
                )

                # ── Control logic ─────────────────────────────────────────────
                error_x = cx - FRAME_CENTER_X

                # Proportional yaw: positive z = turn left; invert so we chase the target
                yaw_speed = error_x * 0.15
                yaw_speed = max(-40.0, min(40.0, yaw_speed))

                # Forward speed: approach while the box top-edge (xyxy[1]) is high
                # in the frame (small value); stop when object fills lower portion
                if xyxy[1] < 140:  # TODO: tune this value
                    forward_speed = 0.10
                else:
                    forward_speed = 0.0

                ep_chassis.drive_speed(x=forward_speed, y=0.0, z=yaw_speed)

                status_text = (
                    f"Spd: {forward_speed:.2f}  Yaw: {yaw_speed:.2f}"
                    f"  Err: {error_x:.1f}  BoxTop: {xyxy[1]:.1f}"
                )
                cv2.putText(
                    frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            else:
                # No detection — stop chassis
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)
                cv2.putText(
                    frame, "No target detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

            # ── Display ───────────────────────────────────────────────────────
            if display_available:
                try:
                    cv2.imshow('RoboMaster YOLO Tracking', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception:
                    print("WARNING: Display unavailable, running headless.")
                    display_available = False
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    except Exception as e:
        print(f"ERROR: Unexpected exception in tracking loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print('Stopping chassis and shutting down...')
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
        print("Done.")


if __name__ == "__main__":
    main()
