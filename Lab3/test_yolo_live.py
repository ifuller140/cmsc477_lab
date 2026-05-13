import cv2
import os
import json
from ultralytics import YOLO
import robomaster
from robomaster import robot
from robomaster import camera

CALIBRATION_FILE = "distance_calibration.json"
calibration_models = {}

def load_calibration():
    global calibration_models
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            calibration_models = json.load(f)
        print(f"Loaded distance calibration models for: {list(calibration_models.keys())}")
    except FileNotFoundError:
        print(f"Warning: Calibration file '{CALIBRATION_FILE}' not found. Run calibrate_distance.py first.")
    except Exception as e:
        print(f"Error loading calibration: {e}")

def main():
    load_calibration()
    
    print('Loading YOLO model for Lab 3...')
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/detect/lab3_gpu_train/weights/best.pt")
    model = YOLO(file_path)

    print('Connecting to RoboMaster...')
    # Reusing the IP configuration from Lab 2
    robomaster.config.ROBOT_IP_STR = "192.168.50.121"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100UB")
    
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    print('Starting live feed. Press "q" to quit.')
    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is not None:
                if hasattr(model, 'predictor') and model.predictor:
                    model.predictor.args.verbose = False
                    
                result = model.predict(source=frame, show=False)[0]
                
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = model.names[cls] if hasattr(model, 'names') else str(cls)
                    
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    # Calculate approximated distance if calibration exists
                    dist_text = ""
                    if class_name in calibration_models and y2 != 0:
                        A = calibration_models[class_name]['A']
                        B = calibration_models[class_name]['B']
                        distance = A * (1.0 / y2) + B
                        dist_text = f" | Dist: {distance:.1f}cm"
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    
                    # Add confidence text, class name, and estimated distance
                    label = f"{class_name} {conf:.2f}{dist_text}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Display the pixel coordinates of the opposite corners
                    coord_text_1 = f"({x1}, {y1})"
                    coord_text_2 = f"({x2}, {y2})"
                    
                    # Display near top-left corner
                    cv2.putText(frame, coord_text_1, (x1 + 5, y1 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    # Display near bottom-right corner
                    cv2.putText(frame, coord_text_2, (x2 - 60, y2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                
                cv2.imshow('RoboMaster YOLO Live - Lab 3', frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print('Shutting down camera stream...')
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
