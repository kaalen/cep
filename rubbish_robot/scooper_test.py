from scooper import Scooper
import time

if __name__ == "__main__":
    scooper = Scooper(debug=True)
    scooper.begin()
    while True:
        scooper.setCatcherAngle(scooper.catchAngle)
        time.sleep(2)
        scooper.driveToDump()
        scooper.setCatcherAngle(scooper.dropAngle)
        time.sleep(2)
        scooper.driveToCatch()

