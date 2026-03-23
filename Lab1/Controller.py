from robomaster import robot
import numpy as np
import time

#inputs: next target coordinate, robot x y orientation
#output: x, y, z speeds

speed = 0.1
angSpeedThres = 90

def calculateVelocity (targetCoord, targetYaw, robotCoord, robotYaw):
    diffX = targetCoord[0] - robotCoord[0]
    diffY = targetCoord[1] - robotCoord[1]

    angularSpeed = calculateAngVel(targetYaw, robotYaw)
    # angularSpeed = 0 # disabled for now

    if diffX != 0 or diffY != 0:

        direction = np.array([diffX, diffY])
        direction = direction / np.linalg.norm(direction) * speed 

        XYspeeds = convertRobotFrame(direction, orientationVector(robotYaw))

        #scale down the translation based on angular velocity
        XYspeeds = (1 - angularSpeed/angSpeedThres) * XYspeeds

        return [XYspeeds[0], XYspeeds[1], angularSpeed]
    
    return [0,0,angularSpeed]

def calculateAngVel (targetYaw, robotYaw):
    kP = 0.5
    error = targetYaw + robotYaw
    #print(f"target yaw: {targetYaw}")
    #print(f"robot yaw: {robotYaw}")

    if error > 180:
        error = error - 360
    elif error < -180:
        error = error + 360
    return kP * error
    

#project desired global vector into robot's xy components
def convertRobotFrame (desiredVector,orientationVector):
    robotX = orientationVector
    normX = np.linalg.norm(robotX)
    desiredX = desiredVector
    normDesX = np.linalg.norm(desiredX)

    robotY = np.array([-orientationVector[1], orientationVector[0]])
    normY = np.linalg.norm(robotY)
    desiredY = np.array([desiredVector[1], -desiredVector[0]])
    normDesY = np.linalg.norm(desiredY)

    #project desired vector onto robotX axis
    dotX = np.dot(desiredVector, robotX)
    speedX = np.linalg.norm((dotX / normX**2) * robotX)
    
    cosX = np.clip(np.dot(robotX, desiredX) / (normX * normDesX), -1.0, 1.0)
    angleX = np.arccos(cosX)
    if angleX > np.pi/2:
        speedX = speedX * -1

    #project desired vector onto robotY axis
    dotY = np.dot(desiredVector, robotY)
    speedY = np.linalg.norm((dotY / normY**2) * robotY)

    cosY = np.clip(np.dot(robotY, desiredVector) / (normY * np.linalg.norm(desiredVector)), -1.0, 1.0)
    angleY = np.arccos(cosY)
    if angleY > np.pi/2:
        speedY = speedY * -1

    return np.array([speedX, speedY])


def orientationVector (robotYaw):
    unitX = np.cos(robotYaw*np.pi/180)
    unitY = np.sin(robotYaw*np.pi/180)
    return np.array([unitX, unitY])

def search_for_tags(ep_chassis):
    ep_chassis.drive_speed(x=0, y=0, z=-30, timeout=0.5)

def stop(ep_chassis):
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)

def move_towards_target(ep_chassis, targetCoord, targetYaw, robotCoord, robotYaw):
    speeds = calculateVelocity(targetCoord, targetYaw, robotCoord, robotYaw)
    #print(f"speeds: {speeds}")
    #print(f"robot position: {robotCoord}")
    #print(f"robot yaw: {robotYaw:.2f}")
    #print("robot moving...")
    
    # execute drive_speed 
    ep_chassis.drive_speed(x=speeds[0], y=speeds[1] * -1, z=speeds[2], timeout=0.5) # I negated Y here!!

if __name__ == '__main__':
    pass
