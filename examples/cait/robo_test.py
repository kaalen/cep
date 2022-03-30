import logging
from threading import Thread
import time
from sweeper import SweeperController


def main():
    logging.getLogger().setLevel(logging.INFO)

    sweeper = SweeperController()
    logging.info("Run sweeper....")
    sweeper.runSweeper()
    logging.info("Go to location 300")
    sweeper.goToLocation(300)
    time.sleep(3)
    print(sweeper.isBusy())
    while True:
        continue

if __name__ == "__main__":
    main()
