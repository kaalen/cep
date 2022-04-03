import logging
from scooper import Scooper
import time
# import cait.essentials
from queue import Queue
from threading import Thread, Lock, Condition

# Class To Drive Robot To Provided Position Or Percent Between Start And End Point
# Location, Start And End Are Defined As Duration * DrivePower Relative To Start


class Sweeper(Scooper):
    log_msg_prefix = "Sweeper: "
    
    def __init__(self, end=50, dump=50):
        super().__init__()

        # Location Vars
        self.location = 0
        self.start = 0
        self.end = end
        self.dumpLoc = dump

        # Sync Vars
        self.messages = Queue()
        self.busyLock = Lock()
        # State Vars
        self.completingAction = False

        # Init Message
        logging.info(self.log_msg_prefix + "Init sweeper")

    # Run Thread Loop
    def run(self):
        self.shouldRun = True
        while self.shouldRun:
            job = self.messages.get()
            logging.debug(self.log_msg_prefix + f"Description: {job.description}")
            logging.debug(self.log_msg_prefix + f"data: {job.data}")

            self.busyLock.acquire()
            self.completingAction = True
            self.busyLock.release()

            if job.description == "dtd":
                self.driveToDump()
            elif job.description == "dts":
                self.driveToStart()
            elif job.description == "gtl":
                self.goToLocation(job.data)
            elif job.description == "gtp":
                self.goToPercent(job.data)
            elif job.description == "ssu":
                self.setScoopUp()
            elif job.description == "ssd":
                self.setScoopDown()
            elif job.description == "sst":
                self.setScoopToggle()
            elif job.description == "movedistance":
                self.moveDistance(job.data)

            if self.messages.empty():
                self.busyLock.acquire()
                self.completingAction = False
                self.busyLock.release()

            self.messages.task_done()

    def stop(self):
        self.shouldRun = False

    def getLocation(self):
        return self.location

    # Move Functions ------

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
        power = self.drivePower if distance > 0 else -1 * self.drivePower
        logging.info(f"Sweeper goToLocation: {str(location)}, distance {str(distance)}")

        self.set_motor_power(
            self.motors["wheels"], power
        )
        time.sleep(duration)
        self.set_motor_power(
            self.motors["wheels"], 0
        )

        self.location = location
        logging.info(f"Sweeper position: {str(self.location)}")

    def moveDistance(self, distance):
        duration = abs(distance) / self.drivePower
        power = self.drivePower if distance > 0 else -1 * self.drivePower
        logging.info(f"Sweeper move distance: {str(distance)}")

        # succ, msg = cait.essentials.set_motor_power(
        #     self.hubName, self.motors["wheels"], power
        # )
        self.set_motor_power(
            self.motors["wheels"], power
        )
        time.sleep(duration)
        self.set_motor_power(
            self.motors["wheels"], 0
        )

        self.location = self.location + distance
        logging.info(f"Sweeper position: {str(self.location)}")

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
    log_msg_prefix = "SweeperController: "

    def __init__(self, end=50, dump=50):

        # Child Variables
        self.sweeper = Sweeper(end, dump)
        self.messages = self.sweeper.messages

        # Init Message
        logging.info(self.log_msg_prefix + "init controller")

    # Run Thread Loop
    def runSweeper(self):
        Thread(target=self.sweeper.run).start()

    def stopSweeper(self):
        self.sweeper.stop()

    # Method To Clear Action Queue
    def clear(self):
        while not self.messages.empty():
            self.messages.get(block=False)

    def isBusy(self):
        self.sweeper.busyLock.acquire()
        isBusy = self.sweeper.completingAction
        self.sweeper.busyLock.release()
        return isBusy

    def dumpAndReturn(self):
        self.messages.put(Job("dtd"))
        self.messages.put(Job("ssd"))
        self.messages.put(Job("dts"))
        self.messages.put(Job("ssu"))
        time.sleep(5)

    # Function To Drive To Dump Location
    def driveToDump(self):
        self.messages.put(Job("dtd"))

    # Function To Drive To Start Location
    def driveToStart(self):
        self.messages.put(Job("dts"))

    # Go To Location Defined As Duration * self.drivePower
    def goToLocation(self, location, skipIfBusy):
        if skipIfBusy == False or self.isBusy() == False:
            self.messages.put(Job("gtl", location))


    # Go To Percent Along A Path, Where Start Is self.start,
    # End Is self.end
    def goToPercent(self, percentAlongPath):
        self.messages.put(Job("gtp", percentAlongPath))

    def moveDistance(self, distance, skipIfBusy):
        if skipIfBusy == False or self.isBusy() == False:
            self.messages.put(Job("movedistance", distance))

    def setScoopUp(self):
        self.messages.put(Job("ssu"))

    def setScoopDown(self):
        self.messages.put(Job("ssd"))

    def setScoopToggle(self):
        self.messages.put(Job("sst"))

    def getLocation(self):
        return self.sweeper.getLocation()
