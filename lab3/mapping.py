import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pupil_apriltags
import robomaster
from robomaster import robot
from robomaster import camera

# Constants for a 10x10 foot mapping area
BOX_SIZE_FT = 10.0
ROBOT_START_X_FT = BOX_SIZE_FT / 2.0
ROBOT_START_Y_FT = BOX_SIZE_FT / 2.0
METERS_TO_FEET = 3.28084
CAMERA_PARAMS = [314.0, 314.0, 320.0, 180.0]
TAG_SIZE_M = 0.153
ROTATION_SPEED_DPS = 12.0  # degrees per second for small incremental rotations
ROTATION_STEP_S = 0.15     # seconds per update step


def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, BOX_SIZE_FT)
    ax.set_ylim(0, BOX_SIZE_FT)
    ax.set_aspect('equal')
    ax.set_title('Top-down AprilTag Localization (Robot at center, pointing north)')
    ax.set_xlabel('East (ft)')
    ax.set_ylabel('North (ft)')
    ax.grid(True, linestyle='--', color='lightgray')
    ax.add_patch(patches.Rectangle((0, 0), BOX_SIZE_FT, BOX_SIZE_FT, fill=False, edgecolor='black', linewidth=2))
    ax.plot(ROBOT_START_X_FT, ROBOT_START_Y_FT, 'bo', markersize=12, label='Robot')
    robot_line, = ax.plot([], [], 'b-', linewidth=3, label='Heading')
    tag_scatter = ax.scatter([], [], c='red', s=80, marker='s', label='AprilTag estimates')
    annotation_texts = []
    ax.legend(loc='upper right')
    fig.canvas.draw()
    return fig, ax, robot_line, tag_scatter, annotation_texts


def update_plot(fig, ax, robot_line, tag_scatter, annotation_texts, yaw_deg, tag_estimates):
    # Update robot heading from center
    yaw_rad = math.radians(yaw_deg)
    robot_x = ROBOT_START_X_FT
    robot_y = ROBOT_START_Y_FT
    heading_length = 1.0
    heading_x = robot_x + heading_length * math.sin(yaw_rad)
    heading_y = robot_y + heading_length * math.cos(yaw_rad)
    robot_line.set_data([robot_x, heading_x], [robot_y, heading_y])

    if annotation_texts:
        for txt in annotation_texts:
            txt.remove()
        annotation_texts.clear()

    if tag_estimates:
        xs = [pos[0] for pos in tag_estimates.values()]
        ys = [pos[1] for pos in tag_estimates.values()]
        tag_scatter.set_offsets(np.column_stack((xs, ys)))
        for tag_id, (x, y) in tag_estimates.items():
            annotation_texts.append(ax.text(x + 0.12, y + 0.12, str(tag_id), color='darkred', fontsize=10, weight='bold'))
    else:
        tag_scatter.set_offsets(np.empty((0, 2)))

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


def camera_to_world_feet(t_ca, yaw_deg):
    # Robot frame: x_right, z_forward from camera pose.
    x_robot = float(t_ca[0]) * METERS_TO_FEET
    z_robot = float(t_ca[2]) * METERS_TO_FEET
    yaw_rad = math.radians(yaw_deg)
    world_x = ROBOT_START_X_FT + x_robot * math.cos(yaw_rad) - z_robot * math.sin(yaw_rad)
    world_y = ROBOT_START_Y_FT + x_robot * math.sin(yaw_rad) + z_robot * math.cos(yaw_rad)
    return world_x, world_y


def detect_tags(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray, estimate_tag_pose=True,
                                 camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE_M)
    return detections


def draw_detection_overlay(frame, detections, tag_estimates, yaw_deg):
    for detection in detections:
        corners = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.line(frame, tuple(corners[0][0]), tuple(corners[2][0]), (0, 255, 0), 2)
        cv2.line(frame, tuple(corners[1][0]), tuple(corners[3][0]), (0, 255, 0), 2)
        center = tuple(detection.center.astype(int))
        cv2.putText(frame, f"ID:{detection.tag_id}", (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        t_ca = detection.pose_t.flatten()
        world_x, world_y = camera_to_world_feet(t_ca, yaw_deg)
        dist_ft = math.sqrt((t_ca[0] ** 2) + (t_ca[1] ** 2) + (t_ca[2] ** 2)) * METERS_TO_FEET
        cv2.putText(frame, f"{world_x:.1f}ft,{world_y:.1f}ft", (center[0] + 5, center[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"d:{dist_ft:.1f}ft", (center[0] + 5, center[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw world center marker and heading text
    cv2.putText(frame, f"Robot heading: {yaw_deg:.1f} deg (north=0)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for tag_id, (x, y) in tag_estimates.items():
        cv2.putText(frame, f"Tag {tag_id}: ({x:.1f}ft,{y:.1f}ft)", (10, 60 + 20 * tag_id % 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def main():
    print('Starting AprilTag mapping...')

    robomaster.config.ROBOT_IP_STRING = "192.168.50.116"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    detector = pupil_apriltags.Detector(families='tag36h11', nthreads=2)
    fig, ax, robot_line, tag_scatter, annotation_texts = init_plot()

    yaw_deg = 0.0
    tag_estimates = {}
    observation_history = {}

    try:
        while True:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=ROTATION_SPEED_DPS)
            time.sleep(ROTATION_STEP_S)
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)

            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None:
                continue

            detections = detect_tags(frame, detector)

            for detection in detections:
                tag_id = int(detection.tag_id)
                t_ca = detection.pose_t.flatten()
                world_pos = camera_to_world_feet(t_ca, yaw_deg)
                observation_history.setdefault(tag_id, []).append(world_pos)
                avg_x = float(np.mean([p[0] for p in observation_history[tag_id]]))
                avg_y = float(np.mean([p[1] for p in observation_history[tag_id]]))
                tag_estimates[tag_id] = (avg_x, avg_y)

            draw_detection_overlay(frame, detections, tag_estimates, yaw_deg)
            cv2.imshow('AprilTag Localization', frame)

            update_plot(fig, ax, robot_line, tag_scatter, annotation_texts, yaw_deg, tag_estimates)

            yaw_deg = (yaw_deg + ROTATION_SPEED_DPS * ROTATION_STEP_S) % 360.0

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print('Keyboard interrupt received, shutting down...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0)
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close(fig)


if __name__ == '__main__':
    main()
