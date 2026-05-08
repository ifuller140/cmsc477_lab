import pupil_apriltags
import cv2
import numpy as np
import time
import math
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

import robomaster
from robomaster import robot, camera

ARENA_M = 3.048 # 10 feet
ROBOT_L = 0.32
ROBOT_W = 0.24
BOX_SIZE = 0.26
TAG_SIZE_M = 0.15 

FOCAL_PX = 600
FRAME_CX = 320
FRAME_CY = 180

def get_transform_robot_camera():
    """
    Get the 4x4 transform mapping a point in the Camera frame to the Robot frame.
    Camera frame: Z forward, X right, Y down
    Robot frame: X forward, Y left, Z up
    """
    rotation_robot_camera = np.array([
        [ 0,  0,  1],  # robot X is forward = camera Z
        [-1,  0,  0],  # robot Y is left = camera -X
        [ 0, -1,  0]   # robot Z is up = camera -Y
    ])
    transform_robot_camera = np.eye(4)
    transform_robot_camera[:3, :3] = rotation_robot_camera
    return transform_robot_camera

def get_transform_world_robot(rx, ry, ryaw_deg):
    """
    Get the 4x4 transform mapping a point in the Robot frame to the World frame.
    """
    rad = math.radians(ryaw_deg)
    rotation_world_robot = np.array([
        [ math.cos(rad), -math.sin(rad), 0],
        [ math.sin(rad),  math.cos(rad), 0],
        [             0,              0, 1]
    ])
    transform_world_robot = np.eye(4)
    transform_world_robot[:3, :3] = rotation_world_robot
    transform_world_robot[0, 3] = rx
    transform_world_robot[1, 3] = ry
    return transform_world_robot

class OdometryTracker:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._lock = threading.Lock()

    def _pos_cb(self, pos_info):
        x, y, z = pos_info
        with self._lock:
            self.x = x
            self.y = y

    def _att_cb(self, att_info):
        yaw, pitch, roll = att_info
        with self._lock:
            self.yaw = yaw

    def subscribe(self, ep_chassis):
        ep_chassis.sub_position(cs=0, freq=20, callback=self._pos_cb)
        ep_chassis.sub_attitude(freq=20, callback=self._att_cb)

    def unsubscribe(self, ep_chassis):
        try: ep_chassis.unsub_position()
        except: pass
        try: ep_chassis.unsub_attitude()
        except: pass

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.yaw

class ArenaMap:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-0.2, ARENA_M + 0.2)
        self.ax.set_ylim(-0.2, ARENA_M + 0.2)
        self.ax.set_aspect('equal')
        self.ax.set_title("Live AprilTag Mapping")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(True)
        
        # Draw Arena boundary
        rect = patches.Rectangle((0, 0), ARENA_M, ARENA_M, linewidth=2, edgecolor='blue', facecolor='whitesmoke', alpha=0.4)
        self.ax.add_patch(rect)
        
        self.robot_patch = patches.Rectangle((-ROBOT_W/2, -ROBOT_L/2), ROBOT_W, ROBOT_L, facecolor='dodgerblue', edgecolor='black', zorder=5)
        self.ax.add_patch(self.robot_patch)
        self.path_x = []
        self.path_y = []
        self.ln_path, = self.ax.plot([], [], 'b--', linewidth=1, alpha=0.6)
        
        self.tags = {}
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_robot(self, rx, ry, ryaw):
        self.path_x.append(rx)
        self.path_y.append(ry)
        self.ln_path.set_data(self.path_x, self.path_y)
        
        self.robot_patch.set_xy((-ROBOT_W/2, -ROBOT_L/2))
        tr = transforms.Affine2D().rotate_deg(ryaw - 90).translate(rx, ry) + self.ax.transData
        self.robot_patch.set_transform(tr)

    def add_tag(self, tag_id, tag_wx, tag_wy, tag_yaw_deg):
        if tag_id not in self.tags:
            # Box is 26x26cm. Place it behind the tag
            box_patch = patches.Rectangle((-BOX_SIZE/2, 0), BOX_SIZE, BOX_SIZE, facecolor='tan', edgecolor='black', zorder=3)
            self.ax.add_patch(box_patch)
            
            # Draw the tag face line
            tag_line = patches.Rectangle((-TAG_SIZE_M/2, -0.01), TAG_SIZE_M, 0.02, facecolor='red', zorder=4)
            self.ax.add_patch(tag_line)
            
            # Transform to world coordinates based on the orientation
            tr_tag = transforms.Affine2D().rotate_deg(tag_yaw_deg - 90).translate(tag_wx, tag_wy) + self.ax.transData
            box_patch.set_transform(tr_tag)
            tag_line.set_transform(tr_tag)
            
            # Label the tag
            self.ax.text(tag_wx, tag_wy + 0.15, f"Tag {tag_id}", fontsize=9, ha='center', fontweight='bold', zorder=6)
            self.tags[tag_id] = True

    def refresh(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def get_world_pose(odom_x, odom_y, odom_yaw):
    """ Calculate world pose given starting in bottom right facing North """
    start_x = ARENA_M - 0.15
    start_y = 0.15
    start_yaw = 90.0
    start_yaw_rad = math.radians(start_yaw)
    
    world_x = start_x + odom_x * math.cos(start_yaw_rad) - odom_y * math.sin(start_yaw_rad)
    world_y = start_y + odom_x * math.sin(start_yaw_rad) + odom_y * math.cos(start_yaw_rad)
    world_yaw = start_yaw + odom_yaw
    return world_x, world_y, world_yaw

def main():
    print("Setting up AprilTag detector...")
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)

    print("Connecting to robot...")
    robomaster.config.ROBOT_IP_STR = "192.168.50.121"
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100UB")
    except Exception as e:
        print(f"Error connecting to robot: {e}")
        return

    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    
    try:
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    except Exception as e:
        print(f"Error starting camera stream: {e}")
        ep_robot.close()
        return

    odom = OdometryTracker()
    odom.subscribe(ep_chassis)
    
    arena_map = ArenaMap()
    transform_robot_camera = get_transform_robot_camera()

    print("Mapping started. Moving forward in small increments...")
    try:
        # Move forward 25 times in 10cm increments
        for step in range(25):
            print(f"Step {step+1}/25: Moving forward 0.1m")
            ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.1).wait_for_completed()
            time.sleep(0.5)
            
            # Flush camera buffer
            frame = None
            for _ in range(3):
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                time.sleep(0.05)
                
            if frame is None:
                continue
                
            odom_x, odom_y, odom_yaw = odom.get_pose()
            rx, ry, ryaw = get_world_pose(odom_x, odom_y, odom_yaw)
            arena_map.update_robot(rx, ry, ryaw)
            
            # Detect tags
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=[FOCAL_PX, FOCAL_PX, FRAME_CX, FRAME_CY],
                tag_size=TAG_SIZE_M
            )
            
            transform_world_robot = get_transform_world_robot(rx, ry, ryaw)
            
            for tag in tags:
                # Transform to get tag pose in world coordinates
                transform_camera_apriltag = np.eye(4)
                transform_camera_apriltag[:3, :3] = tag.pose_R
                transform_camera_apriltag[:3, 3] = tag.pose_t.flatten()
                
                transform_robot_apriltag = transform_robot_camera @ transform_camera_apriltag
                transform_world_apriltag = transform_world_robot @ transform_robot_apriltag
                
                tag_wx = transform_world_apriltag[0, 3]
                tag_wy = transform_world_apriltag[1, 3]
                
                # Z-axis of tag points into the tag. Calculate yaw in world frame.
                z_axis_world = transform_world_apriltag @ np.array([0, 0, 1, 0])
                tag_yaw_deg = math.degrees(math.atan2(z_axis_world[1], z_axis_world[0]))
                
                arena_map.add_tag(tag.tag_id, tag_wx, tag_wy, tag_yaw_deg)
                
                cv2.putText(frame, f"Tag {tag.tag_id}", (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            arena_map.refresh()
            cv2.imshow("Robot View", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        odom.unsubscribe(ep_chassis)
        ep_chassis.drive_speed(x=0, y=0, z=0)
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
