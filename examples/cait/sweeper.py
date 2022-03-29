from scooper import Scooper

import time

import cait.essentials


# Class To Drive Robot To Provided Position Or Percent Between Start And End Point
# Location, Start And End Are Defined As Duration * DrivePower Relative To Start 

class Sweeper(Scooper):
    def __init__(self, end=50, dump=end):
        super().__init__()

        self.location = 0
        self.start = 0
        self.end = end
        self.dumpLoc = dump

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

