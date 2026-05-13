"""
tune_cone_hsv.py — interactive HSV slider tool for dialing in cone segmentation.

Run this with the robot camera live.  Adjust the six sliders until only the
orange cones are white in the mask window and everything else is black.
Write down the H_lo, H_hi, S_lo, S_hi, V_lo, V_hi values, then update the
corresponding np.array values in detect_cones_hsv() inside lab3_runner.py.

Controls:
  q — quit and print the final HSV bounds
"""

import cv2
import numpy as np
import robomaster
from robomaster import robot, camera

def nothing(_): pass

def main():
    robomaster.config.ROBOT_IP_STR = "192.168.50.121"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100UB")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    # defaults tuned from offline training images
    cv2.createTrackbar("H_lo",  "mask",   0, 179, nothing)
    cv2.createTrackbar("H_hi",  "mask",  20, 179, nothing)
    cv2.createTrackbar("S_lo",  "mask", 150, 255, nothing)
    cv2.createTrackbar("S_hi",  "mask", 255, 255, nothing)
    cv2.createTrackbar("V_lo",  "mask",  80, 255, nothing)
    cv2.createTrackbar("V_hi",  "mask", 255, 255, nothing)
    cv2.createTrackbar("H_lo2", "mask", 160, 179, nothing)  # red-wrap low
    cv2.createTrackbar("H_hi2", "mask", 179, 179, nothing)  # red-wrap high

    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            h_lo  = cv2.getTrackbarPos("H_lo",  "mask")
            h_hi  = cv2.getTrackbarPos("H_hi",  "mask")
            s_lo  = cv2.getTrackbarPos("S_lo",  "mask")
            s_hi  = cv2.getTrackbarPos("S_hi",  "mask")
            v_lo  = cv2.getTrackbarPos("V_lo",  "mask")
            v_hi  = cv2.getTrackbarPos("V_hi",  "mask")
            h_lo2 = cv2.getTrackbarPos("H_lo2", "mask")
            h_hi2 = cv2.getTrackbarPos("H_hi2", "mask")

            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([h_lo,  s_lo, v_lo]), np.array([h_hi,  s_hi, v_hi])),
                cv2.inRange(hsv, np.array([h_lo2, s_lo, v_lo]), np.array([h_hi2, s_hi, v_hi])),
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # draw contours on the live frame
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            display = frame.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 800:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("mask",  mask)
            cv2.imshow("frame", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n=== Final HSV bounds for detect_cones_hsv() ===")
                print(f"  lower1 = np.array([{h_lo},  {s_lo}, {v_lo}])")
                print(f"  upper1 = np.array([{h_hi},  {s_hi}, {v_hi}])")
                print(f"  lower2 = np.array([{h_lo2}, {s_lo}, {v_lo}])")
                print(f"  upper2 = np.array([{h_hi2}, {s_hi}, {v_hi}])")
                break
    except KeyboardInterrupt:
        pass
    finally:
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
