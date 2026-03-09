import pupil_apriltags
import cv2
import numpy as np
import time
import math
import traceback
from queue import Empty
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    35: (1.197, 2.128, 90),
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
    # Tag X = Y * Z -> X = [-sin(-yaw), cos(-yaw), 0]
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
    return transform_camera_apriltag

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

# draw the april tagdetections on the frame
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
        
        # add the tag ID
        center = tuple(detection.center.astype(int))
        cv2.putText(frame, str(detection.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# start the maze plot 
def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.2, 3.2) 
    ax.set_ylim(-0.2, 2.8)
    ax.set_aspect('equal')
    ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=1)
    
    ticks = np.arange(0, 3.2, 0.266)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Draw walls (Gray)
    wall1 = patches.Rectangle((0.532, 1.064), 0.266, 1.33, facecolor='gray', edgecolor='dimgray')
    wall2 = patches.Rectangle((0.532, 2.128), 1.862, 0.266, facecolor='gray', edgecolor='dimgray')
    wall3 = patches.Rectangle((2.128, 1.064), 0.266, 1.33, facecolor='gray', edgecolor='dimgray')
    wall4 = patches.Rectangle((1.330, 0.0), 0.266, 1.33, facecolor='gray', edgecolor='dimgray')
    ax.add_patch(wall1)
    ax.add_patch(wall2)
    ax.add_patch(wall3)
    ax.add_patch(wall4)
    
    # Draw Start and Goal
    start_box = patches.Rectangle((0.0, 1.330), 0.266, 0.266, facecolor='lawngreen', edgecolor='gray')
    goal_box = patches.Rectangle((2.66, 1.330), 0.266, 0.266, facecolor='orangered', edgecolor='gray')
    ax.add_patch(start_box)
    ax.add_patch(goal_box)
    
    # Draw Tags
    for tag_id, (x, y, yaw) in TAG_MAP.items():
        tw, th = 0.08, 0.08
        tx, ty = x, y
        if yaw == 0:   tx -= tw/2
        elif yaw == 180: tx += tw/2
        elif yaw == 90:  ty += th/2
        elif yaw == 270: ty -= th/2
        
        tag_rect = patches.Rectangle((tx - tw/2, ty - th/2), tw, th, facecolor='black')
        ax.add_patch(tag_rect)
        
        txt_x, txt_y = x, y
        if yaw == 0:   txt_x += 0.07
        elif yaw == 180: txt_x -= 0.07 
        elif yaw == 90:  txt_y -= 0.07
        elif yaw == 270: txt_y += 0.07
        
        ax.text(txt_x, txt_y, str(tag_id), color='black', fontsize=9, ha='center', va='center', fontweight='bold')
        
    robot_dot, = ax.plot([], [], 'C0o', markersize=14, zorder=5) # Robot position
    robot_line, = ax.plot([], [], 'C0-', linewidth=3, zorder=4) # Robot heading
    path_line, = ax.plot([], [], 'g--', linewidth=2, zorder=3)
    target_dot, = ax.plot([], [], 'mo', markersize=10, zorder=6)
    
    plt.title("Live AprilTag Maze Localization")
    plt.show(block=False)
    
    return fig, ax, robot_dot, robot_line, path_line, target_dot

def update_plot(fig, robot_dot, robot_line, x, y, yaw_deg):
    robot_dot.set_data([x], [y])
    
    rad = math.radians(yaw_deg)
    end_x = x + 0.15 * math.cos(rad)
    end_y = y + 0.15 * math.sin(rad)
    robot_line.set_data([x, end_x], [y, end_y])
    
    fig.canvas.flush_events()

def get_localization(ep_camera, detector, img_out=False):
    try:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
    except Empty:
        return (None, None) if img_out else None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detections = detector.detect(gray, estimate_tag_pose=True,
        camera_params=[314.0, 314.0, 320.0, 180.0], tag_size=0.153)

    draw_detections(img, detections)
    
    poses = []        
    if len(detections) > 0:
        for detection in detections:
            transform_camera_apriltag = get_pose_apriltag_in_camera_frame(detection)
            pose = calculate_robot_world_pose(transform_camera_apriltag, detection.tag_id)
            if pose is not None:
                poses.append(pose)
                info_str = f"R_W(X:{pose[0]:.2f}, Y:{pose[1]:.2f})"
                center = tuple(detection.center.astype(int))
                cv2.putText(img, info_str, (center[0] - 60, center[1] + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if poses:
        poses.sort(key=lambda p: p[3]) 
        best_pose = poses[0]
        cluster = [best_pose]
        for p in poses[1:]:
            if math.hypot(p[0] - best_pose[0], p[1] - best_pose[1]) < 0.3:
                cluster.append(p)
        avg_x = sum(p[0] for p in cluster) / len(cluster)
        avg_y = sum(p[1] for p in cluster) / len(cluster)
        sum_sin = sum(math.sin(math.radians(p[2])) for p in cluster)
        sum_cos = sum(math.cos(math.radians(p[2])) for p in cluster)
        avg_yaw = math.degrees(math.atan2(sum_sin, sum_cos))
        
        if img_out: return (avg_x, avg_y, avg_yaw), img
        return (avg_x, avg_y, avg_yaw)

    if img_out: return None, img
    return None

def detect_tag_loop(ep_camera, detector):
    fig, ax, robot_dot, robot_line, path_line, target_dot = init_plot()
    
    # main loop to detect tags and perform localization
    while True:
        pose, img = get_localization(ep_camera, detector, img_out=True)
        if img is None:
            time.sleep(0.001)
            continue
            
        if pose is not None:
            print(f"[Live Localization] X: {pose[0]:7.3f} m | Y: {pose[1]:7.3f} m | Yaw: {pose[2]:7.2f}°       ", end='\r')
            update_plot(fig, robot_dot, robot_line, pose[0], pose[1], pose[2])
        else:
            print(f"[Live Localization] Searching for valid tags...                                                       ", end='\r')

        cv2.imshow("Robot View", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    # connect to correct robot IP
    robomaster.config.ROBOT_IP_STR = "192.168.50.116"
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