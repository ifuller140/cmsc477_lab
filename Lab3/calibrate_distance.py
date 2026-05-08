import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
import robomaster
from robomaster import robot
from robomaster import camera

# Distances to calibrate at (in cm). We go up to 320 cm in steps.
# You can adjust these steps if you want to capture more or fewer points.
DISTANCES_CM = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 320]

CALIBRATION_FILE = "distance_calibration.json"

def fit_calibration_model(calibration_data):
    """
    Fits a mathematical model: Distance = A * (1 / y2) + B
    calibration_data: dict { 'class_name': [(distance, y2, width, height), ...] }
    """
    models = {}
    print("\n" + "="*50)
    print("CALIBRATION RESULTS & MATHEMATICAL MODELS")
    print("="*50)
    
    for cls, data in calibration_data.items():
        if len(data) == 0:
            continue
        elif len(data) < 2:
            print(f"Not enough data for '{cls}' to fit a model (needs at least 2 points).")
            continue
            
        distances = np.array([d[0] for d in data])
        y2_vals = np.array([d[1] for d in data])
        
        # Fit a linear model: Distance = A * (1/y2) + B
        inv_y2 = 1.0 / y2_vals
        
        # np.polyfit returns coefficients highest power first. 
        # For degree 1: A, B where y = A * x + B
        A, B = np.polyfit(inv_y2, distances, 1)
        
        models[cls] = {'A': float(A), 'B': float(B)}
        print(f"[{cls}]:")
        print(f"  -> Equation: Distance = {A:.2f} / y2 + {B:.2f}")
        print(f"  -> Datapoints used: {len(data)}")
        
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(models, f, indent=4)
        
    print(f"\nSuccessfully saved calibration models to {CALIBRATION_FILE}.")
    print("You can now use this file in your live test script to approximate distances.")
    return models

def main():
    print('Loading YOLO model for Lab 3 Calibration...')
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/detect/lab3_gpu_train/weights/best.pt")
    model = YOLO(file_path)

    # Automatically grab class names from your trained YOLO model
    if hasattr(model, 'names'):
        class_names = list(model.names.values())
    else:
        class_names = ['small_lego', 'large_lego', 'box'] # Fallback

    print('Connecting to RoboMaster...')
    robomaster.config.ROBOT_IP_STRING = "192.168.50.116"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
    
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    calibration_data = {cls: [] for cls in class_names}
    
    current_class_idx = 0
    current_dist_idx = 0
    
    print("\n" + "="*50)
    print("STARTING AUTOMATIC CALIBRATION")
    print("="*50)
    print("Controls (make sure the OpenCV window is selected):")
    print(" 'c' - Capture frame & record box for current distance")
    print(" 's' - Skip current distance (if you can't place it there)")
    print(" 'n' - Skip to next object class")
    print(" 'q' - Quit early and calculate models with current data")
    print("="*50)
    
    try:
        while True:
            # Check if we finished all classes
            if current_class_idx >= len(class_names):
                print("\nCompleted all classes!")
                break
                
            current_class = class_names[current_class_idx]
            
            if current_dist_idx >= len(DISTANCES_CM):
                current_class_idx += 1
                current_dist_idx = 0
                continue
                
            current_dist = DISTANCES_CM[current_dist_idx]
            
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None:
                continue

            if hasattr(model, 'predictor') and model.predictor:
                model.predictor.args.verbose = False
                
            result = model.predict(source=frame, show=False)[0]
            
            boxes = result.boxes
            
            # Find the target object in the frame
            target_box = None
            for box in boxes:
                cls_id = int(box.cls)
                name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
                if name == current_class:
                    target_box = box
                    break # Take the first detection of this class
            
            # Draw all detections
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                cls_id = int(box.cls)
                name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
                
                x1, y1, x2, y2 = map(int, xyxy[:4])
                
                is_target = (name == current_class)
                color = (0, 255, 0) if is_target else (100, 100, 100)
                thickness = 3 if is_target else 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
            # Display UI instructions
            msg1 = f"Target: {current_class}"
            msg2 = f"Distance: {current_dist} cm"
            msg3 = "[c]apture | [s]kip dist | [n]ext obj | [q]uit"
            
            # Text background for readability
            cv2.rectangle(frame, (5, 5), (400, 90), (0, 0, 0), -1)
            cv2.putText(frame, msg1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, msg2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, msg3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Calibration - Live feed', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_class_idx += 1
                current_dist_idx = 0
            elif key == ord('s'):
                current_dist_idx += 1
            elif key == ord('c'):
                if target_box is not None:
                    xyxy = target_box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = map(int, xyxy[:4])
                    width = x2 - x1
                    height = y2 - y1
                    
                    calibration_data[current_class].append((current_dist, y2, width, height))
                    print(f"[{current_class}] Captured at {current_dist}cm | y2: {y2}, w: {width}, h: {height}")
                    
                    # Flash green to indicate successful capture
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
                    cv2.imshow('Calibration - Live feed', frame)
                    cv2.waitKey(200)
                    
                    current_dist_idx += 1
                else:
                    print(f"Warning: Could not find '{current_class}' in frame. Move it into view or press 's' to skip.")
                    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print('\nShutting down camera stream...')
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()
        
        # Calculate models and save them
        fit_calibration_model(calibration_data)

if __name__ == "__main__":
    main()
