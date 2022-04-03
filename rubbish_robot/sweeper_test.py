import logging
from threading import Thread
import time
from sweeper import SweeperController

def main():
    logging.getLogger().setLevel(logging.INFO)

    sweeper = SweeperController()
    sweeper.runSweeper()
    time.sleep(4)

    location = 10
    # sweeper.goToLocation(60, True)

    t_end = time.time() + 60 * 1
    step = 1

    while time.time() < t_end and step <= 5:
        direction = 1
        if step % 2 == 0:
            direction = -1
        sweeper.moveDistance(direction * location, False)
        step += 1
        location += 10
        time.sleep(3)

    sweeper.goToLocation(0, False)



if __name__ == "__main__":
    main()

