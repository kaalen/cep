import track_rubbish_dai
from scooper import Scooper
from threading import Thread
import time

scoop = Scooper()

def launch_scooper(dir, count):
    if dir == "up":
        scoop.startLock.acquire()
        if scoop.hasStarted:
            return
        scoop.startLock.release()
        Thread(target=scoop.begin).start()


def main():
    launch_scooper('up', 5)
    print('start sleep')
    time.sleep(50)
    print('stop_sleep')
    while True:
        pass
    pass

if __name__ == "__main__":
    main()