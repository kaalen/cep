from scooper import Scooper

import time

import cait.essentials

from queue import Queue
from threading import Thread, Lock

# Class To Drive Robot To Provided Position Or Percent Between Start And End Point
# Location, Start And End Are Defined As Duration * DrivePower Relative To Start 

class Sweeper(Scooper):


    def __init__(self, end=50, dump=50):
        super().__init__()

        self.location = 0
        self.start = 0
        self.end = end
        self.dumpLoc = dump
        self.messages = Queue()
        self.lock = Lock()
        self.completingAction = False
        print('init sweeper')

    # Run Thread Loop
    def run(self):
        while True:
            job = self.messages.get()
            print(f'Description: {job.description}')
            print(f'data: {job.data}')

            self.lock.acquire()
            self.completingAction = True
            self.lock.release()


            if job.description == "dtd":
                self.driveToDump()
            elif job.description == 'dts':
                self.driveToStart()
            elif job.description == 'gtl':
                self.goToLocation(job.data)
            elif job.description == 'gtp':
                self.goToPercent(job.data)
            elif job.description == 'ssu':
                self.setScoopUp()
            elif job.description == 'ssd':
                self.setScoopDown()
            elif job.description == 'sst':
                self.setScoopToggle()

            if self.messages.empty():
                self.lock.acquire()
                self.completingAction = False
                self.lock.release()


    # Function To Drive To Dump Location
    def driveToDump(self):
        self.goToLocation(self.dumpLoc)

    # Function To Drive To Start Location
    def driveToStart(self):
        self.goToLocation(self.start)


    # Go To Location Defined As Duration * self.drivePower
    def goToLocation(self, location):
        distance = self.location - location
        duration = abs(distance) / self.drivePower
        power = distance > 0 if self.drivePower else -1 * self.drivePower

        succ, msg = cait.essentials.set_motor_power(self.hubName, 
                                    self.motors["wheels"], power)
        time.sleep(duration)
        succ, msg = cait.essentials.set_motor_power(self.hubName,
                                     self.motors["wheels"], 0)

        self.location = location

    # Go To Percent Along A Path, Where Start Is self.start,
    # End Is self.end
    def goToPercent(self, percentAlongPath):
        destination = percentAlongPath * self.end
        self.goToLocation(destination)

    def setScoopUp(self):
        self.setCatcherAngle(self.catchAngle)

    def setScoopDown(self):
        self.setCatcherAngle(self.dropAngle)

    def setScoopToggle(self):
        angle = self.dropAngle if self.dropAngle == self.pAngle else self.catchAngle
        self.setCatcherAngle(angle)

class Job:

    def __init__(self, description, data=None):
        self.description = description
        self.data = data


class SweeperController:
    
    def __init__(self, end=50, dump=50):
        self.sweeper = Sweeper(end, dump)
        self.messages = self.sweeper.messages
        print('init controller')

    # Run Thread Loop
    def runSweeper(self):
        Thread(target = self.sweeper.run).start()

    def isBusy(self):
        self.sweeper.lock.acquire()
        isBusy = self.sweeper.completingAction
        self.sweeper.lock.release()
        return isBusy

    # Function To Drive To Dump Location
    def driveToDump(self):
        self.messages.put(Job("dtd"))

    # Function To Drive To Start Location
    def driveToStart(self):
        self.messages.put(Job("dts"))

    # Go To Location Defined As Duration * self.drivePower
    def goToLocation(self, location):
        self.messages.put(Job("gtl", location))

    # Go To Percent Along A Path, Where Start Is self.start,
    # End Is self.end
    def goToPercent(self, percentAlongPath):
        self.messages.put(Job('gtp', percentAlongPath))

    def setScoopUp(self):
        self.messages.put(Job("ssu"))


    def setScoopDown(self):
        self.messages.put(Job("ssd"))

    def setScoopToggle(self):
        self.messages.put(Job("sst"))

