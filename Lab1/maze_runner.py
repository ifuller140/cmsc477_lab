import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import robomaster
from robomaster import robot, camera

# import all our lab1 files
from dijkstra_search import define_grid, dijkstra, redefine_coords
import perception
import Controller

if __name__ == '__main__':
    # first find the path using Dijkstra
    print("Planning path with Dijkstra's algorithm...")
    
    try:
        import dijkstra_search
        dijkstra_search.grid = define_grid()
        grid_path = dijkstra()
        calculated_path = redefine_coords(dijkstra_search.grid, grid_path)
    except Exception as e:
        print(f"Failed to find graph path: {e}")
        exit(1)
        
    print(f"Found path with {len(calculated_path)} waypoints.")

    # connect to the RoboMaster chassis and camera
    robomaster.config.ROBOT_IP_STR = "192.168.50.116"
    ep_robot = robot.Robot()
    print("Connecting to RoboMaster...")
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")
    print("Connected.")
    
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    ep_chassis = ep_robot.chassis

    # initialize the perception system
    import pupil_apriltags
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)
    fig, ax, robot_dot, robot_line, path_line, target_dot = perception.init_plot()
    
    # plot the full generated Dijkstra trajectory onto the map
    px = [p[0] for p in calculated_path]
    py = [p[1] for p in calculated_path]
    path_line.set_data(px, py)
    fig.canvas.flush_events()

    # maze running loop
    current_target_index = 0
    print("Starting maze traversal...")
    last_seen_time = time.time()
    aborted = False
    
    try:
        while current_target_index < len(calculated_path):
            # grab the next target x,y location
            target = calculated_path[current_target_index]
            target_x, target_y = target
            target_dot.set_data([target_x], [target_y])

            # get visualization output from the Perception script
            pose, img = perception.get_localization(ep_camera, detector, img_out=True)
            
            # show live camera feed
            if img is not None:
                cv2.imshow("Robot View", img)
                if cv2.waitKey(1) == ord('q'):
                    print("User requested exit.")
                    break
                    
            # check localization status
            if pose is None:
                # if tags have been completely lost for over half a second, start rotating physically to search
                if time.time() - last_seen_time > 0.5:
                    print(f"Rotate to find AprilTags...         ", end='\r')
                    # Controller.search_for_tags(ep_chassis)
                    Controller.stop(ep_chassis) # disabled search for now
                else:
                    # minor dropouts, just coast
                    Controller.stop(ep_chassis)
                time.sleep(0.05)
                continue
                
            # successfully localized
            last_seen_time = time.time()
            robot_x, robot_y, robot_yaw = pose
            
            # update live map plot
            perception.update_plot(fig, robot_dot, robot_line, robot_x, robot_y, robot_yaw)
            
            # check if reached target
            dist_to_target = math.hypot(target_x - robot_x, target_y - robot_y)
            if dist_to_target < 0.3: #threshold
                print(f"Reached point {current_target_index}/{len(calculated_path)}, going to next point")
                current_target_index += 1 
                if current_target_index >= len(calculated_path):
                    print("Reached the Final Goal!")
                    break
                continue

            target_yaw = 0 # default yaw

            # add manual turn points for the robot to always see the april tags throughout the maze
            if current_target_index == 15:
                target_yaw = -90
            elif current_target_index == 25:
                target_yaw = 0
            elif current_target_index == 30:
                target_yaw = -90
            elif current_target_index == 35:
                target_yaw = 180

            # execute drive via custom Controller (calculates velocity and sends commands)
            Controller.move_towards_target(ep_chassis, [target_x, target_y], target_yaw, [robot_x, robot_y], robot_yaw)

            # keep loop slightly bottlenecked to allow robot processing
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nCTRL+C caught. Stopping...")
        aborted = True
    except Exception as e:
        print(f"\nError encountered: {e}")
        aborted = True
    finally:
        print("\nMaze run finished. Shutting down safely...")
        try:
            Controller.stop(ep_chassis)
        except:
            pass
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()
        
        if not aborted:
            # keep plot open at the end so we can see the path traveled
            plt.ioff()
            plt.show()
        else:
            plt.close('all')
            import sys
            sys.exit(0)