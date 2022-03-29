from threading import Thread
import time
from sweeper import SweeperController


def main():
    sweeper = SweeperController()
    sweeper.runSweeper()
    sweeper.goToLocation(300)
    time.sleep(3)
    print(sweeper.isBusy())
    while True:
        continue

if __name__ == "__main__":
    main()