import logging
from threading import Thread
import time
from curt.modules.control.base_control import BaseControl
from curt.modules.control.robot_inventor_control import RobotInventorControl
from cait.managers.device_manager import DeviceManager

def main():
    logging.getLogger().setLevel(logging.INFO)

    lego_hub_name = "Robot Inventor: A8:E2:C1:95:22:45"
    hub_address = "A8:E2:C1:95:22:45"
    device_manager = DeviceManager()
    robot = RobotInventorControl()
    success = robot.config_control_handler({
        "hub_address": hub_address
    })
    if success:
        print("Connection successfull")
        robot.display({
            "display_type": "image",
            "image": "Happy"
        })

        control_params = {
            "motor_arrangement": "individual",
            "motor": "B",
            "motion": "speed",
            "speed": 70,
        }
        robot.control_motor(control_params)
        time.sleep(3)
        control_params = {
            "motor_arrangement": "individual",
            "motor": "B",
            "motion": "speed",
            "speed": -70,
        }
        robot.control_motor(control_params)
        time.sleep(3)
        control_params = {
            "motor_arrangement": "individual",
            "motor": "B",
            "motion": "speed",
            "speed": 0,
        }
        robot.control_motor(control_params)
        time.sleep(3)

        # while True:
        #     robot.setPowerdownTimeout(18000)
        #     config = robot.getConfig()
        #     logging.info("Hub config: " + config)
        #     time.sleep(1)
        #     version = robot.getVersion()
        #     logging.info("Hub version: " + version)
        #     time.sleep(1)
        #     status = robot.getStatus()
        #     logging.info("Hub status: " + status)
        #     time.sleep(2)
    else:
        print("... no luck")


if __name__ == "__main__":
    main()
