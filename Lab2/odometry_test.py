import time
import math
import random
from robomaster import robot

class OdometryTester:
    def __init__(self, mode=1, num_moves=5):
        self.mode = mode
        self.num_moves = num_moves
        
        # Mathematical global tracking
        self.math_x = 0.0
        self.math_y = 0.0
        self.math_yaw = 0.0  # degrees
        
        # SDK Internal Tracking
        self.sdk_x = 0.0
        self.sdk_y = 0.0
        self.sdk_yaw = 0.0
        
        print("Initializing RoboMaster...")
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type="ap")
        self.chassis = self.ep_robot.chassis
        
        if self.mode == 3:
            # Subscribe to exact hardware odometry in Mode 3
            print("Activating SDK Telemetry streams...")
            self.chassis.sub_position(freq=5, callback=self.position_cb)
            self.chassis.sub_attitude(freq=5, callback=self.attitude_cb)
            time.sleep(1) # wait for first callback to populate

    def position_cb(self, position_info):
        x, y, z = position_info
        self.sdk_x = x
        self.sdk_y = y

    def attitude_cb(self, attitude_info):
        yaw, pitch, roll = attitude_info
        self.sdk_yaw = yaw

    def run(self):
        print(f"--- Starting Odometry Test Mode {self.mode} ---")
        for i in range(self.num_moves):
            # Generate random movements, respecting 2x2m bounds (-1.0 to 1.0)
            valid_move = False
            dx, dy, dz = 0.0, 0.0, 0.0
            new_x, new_y, new_yaw = 0.0, 0.0, 0.0
            
            while not valid_move:
                dx = random.uniform(-0.5, 0.5)
                dy = random.uniform(-0.5, 0.5)
                
                if self.mode == 1:
                    dz = 0.0
                else:
                    dz = random.uniform(-30, 30)
                
                # Predict global position
                yaw_rad = math.radians(self.math_yaw)
                # Apply translation relative to current yaw
                # We assume a standard projection: x forward, y left/right
                temp_x = self.math_x + dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
                temp_y = self.math_y + dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
                
                if abs(temp_x) <= 1.0 and abs(temp_y) <= 1.0:
                    valid_move = True
                    new_x = temp_x
                    new_y = temp_y
                    new_yaw = self.math_yaw + dz
            
            print(f"Move {i+1}/{self.num_moves}: Cmd (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f})")
            
            if self.mode == 1 or self.mode == 2:
                # Basic combined movement, might arc slightly
                self.chassis.move(x=dx, y=dy, z=dz, xy_speed=0.5, z_speed=30).wait_for_completed()
            elif self.mode == 3:
                # Hyper-accurate mathematical movement: translate then rotate to preserve strict linear SE(2) vectors
                if dx != 0 or dy != 0:
                    self.chassis.move(x=dx, y=dy, z=0, xy_speed=0.5).wait_for_completed()
                if dz != 0:
                    self.chassis.move(x=0, y=0, z=dz, z_speed=30).wait_for_completed()
            
            # Update math model
            self.math_x, self.math_y, self.math_yaw = new_x, new_y, new_yaw
            
            # Print Live Output
            print(f"   => Math Expected: X={self.math_x:.2f}, Y={self.math_y:.2f}, Yaw={self.math_yaw:.2f}")
            if self.mode == 3:
                print(f"   => SDK Internal:  X={self.sdk_x:.2f}, Y={self.sdk_y:.2f}, Yaw={self.sdk_yaw:.2f}")
            time.sleep(0.5)
            
        self.return_to_start()
        
    def return_to_start(self):
        print("\n--- Returning to Start (0.0, 0.0, 0.0) ---")
        
        # Calculate distance home in global frame
        home_x, home_y = -self.math_x, -self.math_y
        
        # Calculate local move needed based on our current math_yaw
        yaw_rad = math.radians(self.math_yaw)
        # Transform global vector (home_x, home_y) back into local frame to drive
        # We solve:
        # home_x = local_dx * cos(yaw) - local_dy * sin(yaw)
        # home_y = local_dx * sin(yaw) + local_dy * cos(yaw)
        # By inverse rotation matrix:
        local_dx = home_x * math.cos(-yaw_rad) - home_y * math.sin(-yaw_rad)
        local_dy = home_x * math.sin(-yaw_rad) + home_y * math.cos(-yaw_rad)
        
        print(f"Returning vector: Local dx={local_dx:.2f}, Local dy={local_dy:.2f}")
        self.chassis.move(x=local_dx, y=local_dy, z=0, xy_speed=0.5).wait_for_completed()
        
        local_dz = -self.math_yaw
        print(f"Returning rotation: Local dz={local_dz:.2f}")
        self.chassis.move(x=0, y=0, z=local_dz, z_speed=30).wait_for_completed()
        
        self.math_x = 0.0
        self.math_y = 0.0
        self.math_yaw = 0.0
        
        print("\n--- Return Complete ---")
        print(f"Final Math Expected: X=0.00, Y=0.00, Yaw=0.00")
        if self.mode == 3:
            time.sleep(1.0) # Wait for internal systems to settle
            print(f"Final SDK Internal:  X={self.sdk_x:.2f}, Y={self.sdk_y:.2f}, Yaw={self.sdk_yaw:.2f}")
            
            self.chassis.unsub_position()
            self.chassis.unsub_attitude()
            
        self.ep_robot.close()

if __name__ == '__main__':
    print("Select Mode:")
    print("  1 - Translation Only")
    print("  2 - Translation + Rotation")
    print("  3 - Hyper-Accurate Mode (Isolated Vectors + Live Telemetry)")
    try:
        user_mode_str = input("Enter mode (1/2/3) [Default: 3]: ")
        user_mode = int(user_mode_str) if user_mode_str.strip() else 3
    except ValueError:
        user_mode = 3
        
    tester = OdometryTester(mode=user_mode, num_moves=4)
    tester.run()
