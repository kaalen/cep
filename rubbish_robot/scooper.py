# Class For Performing Scoop Actions


import tkinter as tk
import threading, queue
from threading import Lock
import time
from numpy import datetime64
import cait.essentials
from datetime import datetime
from enum import Enum, auto



class Scooper:
    # Default Hub Name
    lego_hub_name = "Robot Inventor: A8:E2:C1:95:22:45"
    motors = {"wheels" : "motor_B",
    "scoop" : "motor_D",
    "cam" : "motor_F"}
   


    def __init__(self, hubName=lego_hub_name,
     dropAngle=-159, catchAngle=-116,
      driveDuration=2, drivePower=40,
      catchDuration=2, dropDuration=5, debug=False) -> None:

        self.hubName = hubName 
        self.dropAngle = dropAngle
        self.catchAngle = catchAngle
        self.driveDuration = driveDuration
        self.drivePower = drivePower
        self.catchDuration = catchDuration
        self.dropDuration = dropDuration
        self.debug = debug

        self.pAngle = 0

        self.startLock = Lock()

        self.hasStarted = False

        self.state = self.States.WAIT_TO_CATCH
        self.initHub()
    
    def initHub(self):
        succ, msg = cait.essentials.initialize_component('control', [self.hubName])
        if self.debug:
            print(str(succ) + " " + msg)


    def begin(self):
        self.startLock.acquire()
        self.hasStarted = True
        self.startLock.release()

        # Begin Catching
        self.state = self.States.CATCHING
        self.beginCatching()

        # Begin Drive To Dump
        self.state = self.States.DRIVING_TO_DUMP
        self.driveToDump()

        # Begin Dumping
        self.state = self.States.DUMPING
        self.dump()

        # Begin Drive Back
        self.state = self.States.DRIVING_TO_CATCH
        self.driveToCatch()

        self.reset()

    def reset(self):
        self.startLock.acquire()
        self.hasStarted = False
        self.state = self.States.WAIT_TO_CATCH
        self.startLock.release()

    def beginCatching(self):
        self.setCatcherAngle(self.catchAngle)
        time.sleep(self.catchDuration)

    def driveToDump(self):
        succ, msg = cait.essentials.set_motor_power(self.hubName, self.motors["wheels"], self.drivePower)
        time.sleep(self.driveDuration)
        succ, msg = cait.essentials.set_motor_power(self.hubName, self.motors["wheels"], 0)
    
    def driveToCatch(self):
        succ, msg = cait.essentials.set_motor_power(self.hubName, self.motors["wheels"], self.drivePower * -1)
        time.sleep(self.driveDuration)
        succ, msg = cait.essentials.set_motor_power(self.hubName, self.motors["wheels"], 0)


    
    def setCatcherAngle(self, angle):
        if angle != self.pAngle:
            csucc = cait.essentials.set_motor_position(self.hubName, self.motors["scoop"], angle)
            if self.debug:
                print(csucc)
            self.pAngle = angle

    def dump(self):
        # TODO: needs to be implemented
        self.setCatcherAngle(self.dropAngle)
        time.sleep(self.dropDuration)

    
    class States(Enum):
        WAIT_TO_CATCH = auto
        CATCHING = auto
        DRIVING_TO_DUMP = auto
        DUMPING = auto
        DRIVING_TO_CATCH = auto
