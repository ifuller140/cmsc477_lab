import pupil_apriltags
import cv2
import numpy as np
import time
import math
import traceback
from queue import Empty

import robomaster
from robomaster import robot
from robomaster import camera

# This mapping defines the location and orientation of the tags in the world frame
# 0 facing +X, 90 facing - Y, 180 facing -X, 270 facing +Y
# 26.6 cm per box side = .266 meters
TAG_MAP = {
    30: (0.532, 1.995, 180.0),     
    31: (0.798, 1.995, 0.0),
    32: (0.532, 1.463, 180.0),
    33: (0.798, 1.463, 0.0),
    34: (0.665, 1.064, 90.0),
    35: (1.463, 2.128, 90),
    36: (1.729, 2.128, 90),
    37: (1.463, 1.330, 270.0),
    38: (1.330, 0.931, 180.0),
    39: (1.596, 0.931, 0.0),
    40: (1.330, 0.399, 180.0),
    41: (1.596, 0.399, 0.0),
    42: (2.128, 1.995, 180),
    43: (2.394, 1.995, 0),
    44: (2.128, 1.463,180),
    45: (2.394, 1.463, 0),
    46: (2.261, 1.064, 90.0)
}

def get_transform_robot_camera():
    """
    Get the 4x4 transform mapping a point in the Camera frame to the Robot frame.
    Camera frame: Z forward, X right, Y down
    Robot frame: X forward, Y left, Z up
    """
    # Rotation from camera to robot
    rotation_robot_camera = np.array([
        [ 0,  0,  1],  # robot X is forward = camera Z
        [-1,  0,  0],  # robot Y is left = camera -X
        [ 0, -1,  0]   # robot Z is up = camera -Y
    ])

    
    # Translation from camera to robot
    transform_robot_camera = np.eye(4)
    transform_robot_camera[:3, :3] = rotation_robot_camera # assign the same robot rotation matrix to transform_robot_camera
    
    return transform_robot_camera

def get_transform_world_apriltag(x, y, yaw_deg):
    """
    Get the 4x4 transform of the Tag in the World frame.
    Using pupil_apriltags convention: Tag Z points INTO the tag.
    The normal yaw from TAG_MAP defines the outward face: 0 (+X), 90 (-Y), 180 (-X), 270 (+Y).
    """
    rad = math.radians(-yaw_deg)
    
    # Tag Z points INTO the tag (away from outward normal) -> Z = [-cos(-yaw), -sin(-yaw), 0]
    # Tag Y points DOWN -> Y = [0, 0, -1]
    # Tag X = Y x Z -> X = [-sin(-yaw), cos(-yaw), 0]
    rotation_world_apriltag = np.array([
        [-math.sin(rad),  0, -math.cos(rad)],
        [ math.cos(rad),  0, -math.sin(rad)],
        [             0, -1,              0]
    ])
    transform_world_apriltag = np.eye(4)
    transform_world_apriltag[:3, :3] = rotation_world_apriltag
    # Set the world x, y position. Z is half-height of the 26.6cm block = 0.133m
    transform_world_apriltag[0, 3] = x
    transform_world_apriltag[1, 3] = y
    transform_world_apriltag[2, 3] = 0.133 
    return transform_world_apriltag

def get_pose_apriltag_in_camera_frame(detection):
    rotation_camera_apriltag = detection.pose_R
    translation_camera_apriltag = detection.pose_t.flatten()
    
    transform_camera_apriltag = np.eye(4)
    transform_camera_apriltag[:3, :3] = rotation_camera_apriltag
    transform_camera_apriltag[:3, 3] = translation_camera_apriltag
    return transformation_camera_apriltag

def calculate_robot_world_pose(transformation_camera_apriltag, tag_id):
    if tag_id not in TAG_MAP:
        return None

    x, y, yaw_deg = TAG_MAP[tag_id]
    
    transform_world_apriltag = get_transform_world_apriltag(x, y, yaw_deg)
    transform_robot_camera = get_transform_robot_camera()
    transform_camera_robot = np.linalg.inv(transform_robot_camera)

    # transformation_world_robot = transform_world_apriltag * (transformation_camera_apriltag)^-1 * transform_camera_robot
    transform_apriltag_camera = np.linalg.inv(transformation_camera_apriltag)
    
    transformation_world_robot = transform_world_apriltag @ transform_apriltag_camera @ transform_camera_robot
    
    # get robot translation
    robot_x = transformation_world_robot[0, 3]
    robot_y = transformation_world_robot[1, 3]
    
    # get robot rotation
    robot_x_world = transformation_world_robot[:3, 0]
    robot_yaw_rad = math.atan2(robot_x_world[1], robot_x_world[0])
    robot_yaw = math.degrees(robot_yaw_rad)
    
    dist_to_tag = transformation_camera_apriltag[2, 3]

    return robot_x, robot_y, robot_yaw, dist_to_tag

# Draw the detections on the frame
def draw_detections(frame, detections):
    for detection in detections:
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  
        top_right = tuple(pts[1][0])  
        bottom_right = tuple(pts[2][0])  
        bottom_left = tuple(pts[3][0])  
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)
        
        # Draw tag ID
        center = tuple(detection.center.astype(int))
        cv2.putText(frame, str(detection.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def detect_tag_loop(ep_camera, detector):
    
    # main loop to detect tags and perform localization
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detections = detector.detect(gray, estimate_tag_pose=True,
            camera_params=[314.0, 314.0, 320.0, 180.0], tag_size=0.153)

        draw_detections(img, detections)
        
        poses = []        
        if len(detections) > 0:
            for detection in detections:
                transformation_camera_apriltag = get_pose_apriltag_in_camera_frame(detection)
                pose = calculate_robot_world_pose(transformation_camera_apriltag, detection.tag_id)
                if pose is not None:
                    poses.append(pose)
                    
                    # draw individual perceived location text near the tag
                    # this massively helps for debugging speciifc tags in the maze
                    info_str = f"R_W(X:{pose[0]:.2f}, Y:{pose[1]:.2f})"
                    center = tuple(detection.center.astype(int))
                    cv2.putText(img, info_str, (center[0] - 60, center[1] + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

         
        if poses: # if we get multiple april tags in view then we want to average the poses
            poses.sort(key=lambda p: p[3]) # sort by distance to tag since the closest tag is usually most accurate
            
            best_pose = poses[0]
            cluster = [best_pose]
            
            # only add other tags if they are within 0.3 meters of the closest tag's estimate
            for p in poses[1:]:
                if math.hypot(p[0] - best_pose[0], p[1] - best_pose[1]) < 0.3:
                    cluster.append(p)
            
            # average teh translation of the poses
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            avg_y = sum(p[1] for p in cluster) / len(cluster)
            
            # average the yaw of the poses which also has to account for jumps between -180 and 180 degrees
            sum_sin = sum(math.sin(math.radians(p[2])) for p in cluster)
            sum_cos = sum(math.cos(math.radians(p[2])) for p in cluster)
            avg_yaw = math.degrees(math.atan2(sum_sin, sum_cos))
            
            # print the localization data
            print(f"[Live Localization] X: {avg_x:7.3f} m | Y: {avg_y:7.3f} m | Yaw: {avg_yaw:7.2f}° | Tags used: {len(cluster)}/{len(poses)}           ", end='\r')
        else:
            # if there are tags found but none of them can be mapped, or no tags seen
            print(f"[Live Localization] Searching for valid tags...                                                       ", end='\r')

        cv2.imshow("Robot View", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    # connect to correct robot IP
    robomaster.config.ROBOT_IP_STRING = "192.168.50.116"
    ep_robot = robot.Robot()
    
    print("Connecting to robot...")
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
    print("Connected.")
    
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    # Initialize apriltag detector
    import pupil_apriltags
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)

    try:
        detect_tag_loop(ep_camera, detector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print('\nShutting down robot and camera stream...')
        ep_camera.stop_video_stream()
        ep_robot.close()