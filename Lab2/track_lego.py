import cv2
import time
import os
from ultralytics import YOLO
import robomaster
from robomaster import robot
from robomaster import camera

def main():
    print('Loading YOLO model...')
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmsc477_yolo/runs/detect/lego_gpu_train3/weights/best.pt")
    model = YOLO(file_path)

    print('Connecting to RoboMaster...')
    robomaster.config.ROBOT_IP_STRING = "192.168.50.116"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
    
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    ep_arm = ep_robot.robotic_arm
    ep_chassis = ep_robot.chassis

    print('Starting tracking loop. Press "q" to quit.')
    
    FRAME_W = 640
    FRAME_CENTER_X = FRAME_W / 2.0
    TARGET_HEIGHT = 200 # Pixel height of bounding box when object is close to gripper
    
    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is not None:
                if hasattr(model, 'predictor') and model.predictor:
                    model.predictor.args.verbose = False
                    
                result = model.predict(source=frame, show=False)[0]
                boxes = result.boxes
                
                if len(boxes) > 0:
                    # Find the largest bounding box (assuming it's the target Lego piece)
                    best_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                    xyxy = best_box.xyxy.cpu().numpy().flatten()
                    
                    width = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    cx = xyxy[0] + width / 2.0
                    
                    # Draw rectangle on target
                    cv2.rectangle(frame,
                                  (int(xyxy[0]), int(xyxy[1])), 
                                  (int(xyxy[2]), int(xyxy[3])),
                                  color=(0, 255, 0), thickness=3)
                                  
                    # --- CONTROL LOGIC ---
                    error_x = cx - FRAME_CENTER_X
                    
                    # Proportional control for yaw (z-axis rotation)
                    # Robomaster: positive z is turn left. 
                    # If target is to the right (cx > center), error is positive, we want turn right (negative z)
                    yaw_speed = -error_x * 0.15 
                    
                    # Clamp yaw speed
                    yaw_speed = max(-40.0, min(40.0, yaw_speed)) 
                    
                    # Proportional control for forward (x-axis)
                    if height < TARGET_HEIGHT:
                        # Approach speed proportional to distance (inversely proportional to height)
                        # We use a constant slow speed instead for safety
                        forward_speed = 0.15 
                    else:
                        # Too close, stop
                        forward_speed = 0.0
                        
                    ep_chassis.drive_speed(x=forward_speed, y=0.0, z=yaw_speed)
                    
                    status_text = f"Spd: {forward_speed:.2f} Yaw: {yaw_speed:.2f} Err: {error_x:.1f} H: {height:.1f}"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                else:
                    # Stop if no object detected
                    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)
                
                cv2.imshow('RoboMaster YOLO Tracking', frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print('Stopping chassis and shutting down camera stream...')
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
