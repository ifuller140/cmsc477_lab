from robomaster import robot
import numpy as np
import time

#inputs: next target coordinate, robot x y orientation
#output: 
speed = 0.25
angSpeedThres = 90

def calculateVelocity (targetCoord, targetYaw, robotCoord, robotYaw):
    diffX = targetCoord[0] - robotCoord[0]
    diffY = targetCoord[1] - robotCoord[1]

    angularSpeed = calculateAngVel(targetYaw, robotYaw)

    if diffX != 0 and diffY != 0:

        direction = np.array([diffX, diffY])
        direction = direction / np.linalg.norm(direction) * speed

        XYspeeds = convertRobotFrame(direction, orientationVector(robotYaw))

        XYspeeds = (1 - angularSpeed/angSpeedThres) * XYspeeds
        return [XYspeeds[0], XYspeeds[1], angularSpeed]
    
    return [0,0,angularSpeed]

def calculateAngVel (targetYaw, robotYaw):
    kP = 0.5
    error = targetYaw - robotYaw
    if error > 180:
        error = error - 180
    return - kP * error
    

#project desired global vector into robot's xy components
def convertRobotFrame (desiredVector,orientationVector):
    robotX = orientationVector
    normX = np.linalg.norm(robotX)
    desiredX = desiredVector
    normDesX = np.linalg.norm(desiredX)

    robotY = np.array([orientationVector[1], -orientationVector[0]])
    normY = np.linalg.norm(robotY)
    desiredY = np.array([desiredVector[1], -desiredVector[0]])
    normDesY = np.linalg.norm(desiredY)

    #project desired vector onto robotX axis
    dotX = np.dot(desiredVector, robotX)
    speedX = np.linalg.norm((dotX / normX**2) * robotX)
    
    cosX = np.dot(robotX, desiredX) / (normX * normDesX)
    angleX = np.arccos(cosX)
    if angleX > np.pi/2:
        speedX = speedX * -1

    #project desired vector onto robotY axis
    dotY = np.dot(desiredVector, robotY)
    speedY = np.linalg.norm((dotY / normY**2) * robotY)

    cosY = np.dot(robotY, desiredVector) / (normY * np.linalg.norm(desiredVector))
    angleY = np.arccos(cosY)
    if angleY > np.pi/2:
        speedY = speedY * -1

    return np.array([speedX, speedY])


#returns a numpy array representing the unit vector of x direction of robot in global frame
def orientationVector (robotYaw):
    unitX = np.cos(robotYaw*np.pi/180)
    unitY = np.sin(robotYaw*np.pi/180)
    return np.array([unitX, unitY])



if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100VW")

    ep_chassis = ep_robot.chassis



    robotCoord = [1,0]
    robotYaw = 45
    targetCoord = [1,0]
    targetYaw = 90

    #desiredVector = np.array([-0.2,-0.4])
    wheelSpeeds = calculateVelocity(targetCoord, targetYaw, robotCoord, robotYaw)

    print(wheelSpeeds)
    ep_chassis.drive_speed(wheelSpeeds[0], wheelSpeeds[1], wheelSpeeds[2], timeout=5)
    #ep_chassis.drive_speed(0, 0.4, 0, timeout = 5)
    time.sleep(2)
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)

    ep_robot.close()

