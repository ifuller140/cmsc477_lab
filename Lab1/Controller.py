from robomaster import robot
import numpy as np

#inputs: next target coordinate, robot x y orientation
#output: 



def calculateVelocity (targetCoord, robotPose, orientation):
    diffX = targetCoord[0] - robotPose[0]
    diffY = targetCoord[1] - robotPose[1]
    

#project desired global xy direction into robot xy components
def convertRobotFrame (x,y,orientation):


#returns a numpy array representing the unit vector of x direction of robot in global frame
def orientationVector (robotYaw):
    unitX = np.cos(robotYaw*np.pi/180)
    unitY = np.sin(robotYaw*np.pi/180)
    return np.array([unitX, unitY])



if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")

    speed = 0.5

    ep_chassis = ep_robot.chassis
