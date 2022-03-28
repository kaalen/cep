import track_rubbish_dai
from scooper import Scooper
from threading import Thread
scoop = Scooper()

def launch_scooper(dir, count):
    if dir == "up":
        scoop.startLock.acquire()
        if scoop.hasStarted:
            return
        scoop.startLock.release()
        Thread(target=scoop.begin).start()


def main():
    track_rubbish_dai.subscribeOnCount(launch_scooper)
    track_rubbish_dai.main()
    pass

if __name__ == "__main__":
    main()