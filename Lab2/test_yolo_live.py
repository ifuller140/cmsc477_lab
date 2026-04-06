import cv2
import os
import time
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
                    
                    # Draw rectangle
                    cv2.rectangle(frame,
                                  (int(xyxy[0]), int(xyxy[1])), 
                                  (int(xyxy[2]), int(xyxy[3])),
                                  color=(0, 0, 255), thickness=2)
                    
                    # Add confidence text
                    cv2.putText(frame, f"Lego {conf:.2f}", (int(xyxy[0]), int(xyxy[1])-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.imshow('RoboMaster YOLO Live', frame)
                
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
