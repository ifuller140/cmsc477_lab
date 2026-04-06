import cv2
import os
import robomaster
from robomaster import robot, camera

def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "robomaster") # CHANGE THIS TO THE CORRECT FOLDER
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the robot's IP
    robomaster.config.ROBOT_IP_STR = "192.168.50.116"
    ep_robot = robot.Robot()
    
    print("Connecting to RoboMaster...")
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
        print("Connected.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    ep_camera = ep_robot.camera
    
    # Start the video stream
    print("Starting video stream...")
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    
    print(f"Images will be saved to: {output_dir}")
    print("Press 'c' to capture an image. Press 'q' to quit.")
    
    image_count = 1
    display_available = True # to help us catch errors i
    
    try:
        while True:
            img = ep_camera.read_cv2_image()
            
            if img is not None:
                if display_available:
                    try:
                        cv2.imshow("Robot", img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('c'):
                            filename = os.path.join(output_dir, f"image{image_count}.jpg")
                            cv2.imwrite(filename, img)
                            print(f"Captured image{image_count}.jpg")
                            image_count += 1
                    except Exception:
                        print("No display availible")
                        display_available = False
            else:
                import time
                time.sleep(0.01)
    
    
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        print("Disconnecting")
        ep_camera.stop_video_stream()
        cv2.destroyAllWindows()
        ep_robot.close()
        print("Disconnected")

if __name__ == '__main__':
    main()
