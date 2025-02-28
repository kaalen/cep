# Python code generated by CAIT's Visual Programming Interface

from sqlalchemy import true
import cait.essentials
import threading
import time


object_coordinate = None
screen_center = None
rotate_power = None
x1 = None
object_of_interest = None
x2 = None
object_center = None
coordinates = None
object2 = None
power = None
robot_inventor_hub = 'Robot Inventor: A8:E2:C1:95:22:45'
motor_scoop = 'motor_D'
motor_wheels = 'motor_B'
motor_camera = 'motor_F'
target_object_name = "person"

preview_width = 640
preview_heigth = 360

"""Describe this function...
"""
def follow_object(object_coordinate):
    #TODO: instead of moving the camera to follow object we need to move the wheels
    global screen_center, rotate_power, x1, object_of_interest, x2, object_center, coordinates, object2, power
    screen_center = 640 / 2
    x1 = object_coordinate[0]

    x2 = object_coordinate[2]
    
    object_center = x1 + (x2 - x1) / 2
    rotate_power = cait.essentials.update_pid((object_center - screen_center))['value']
    cait.essentials.set_motor_power(robot_inventor_hub, motor_wheels, rotate_power)
    return rotate_power
    
def scoop():
    # TODO: Implement function to lower and raise the scoop
    #hooked up to motor D
    # Lower scoop
    cait.essentials.control_motor(robot_inventor_hub, motor_scoop, 1, '3s')
    # wait - assuming target object falls in
    time.sleep(3)
    # Raise scoop
    cait.essentials.control_motor(robot_inventor_hub, motor_scoop, -1, '3s')
    
    return true
    
# Auto generated dispatch function code
def dispatch_func_0():
    global object2, coordinates, object_of_interest
    power = follow_object(object2)
# End of auto generated dispatch function

nn_input_size = 300
nn_model_blob = "detect_rubbish_openvino_2021.4_5shave.blob"
coco_model = "ssdlite_mbv2_coco.blob"
face_detection_model = "face-detection-retail-0004_openvino_2021.2_6shave.blob"
vision_config1 = [
    ["add_rgb_cam_node", preview_width, preview_heigth], 
    ["add_rgb_cam_preview_node"],
    ["add_stereo_cam_node", False], 
    ["add_stereo_frame_node"],
    ["add_spatial_mobilenetSSD_node", "object_detection", nn_model_blob, nn_input_size, nn_input_size, 0.5]#
    ["add_nn_node", "object_detection", nn_model_blob, nn_input_size, nn_input_size],
]
vision_config2 = [
    ["add_rgb_cam_node", preview_width, preview_heigth], 
    ["add_rgb_cam_preview_node"],
    ["add_stereo_cam_node", False], 
    ["add_stereo_frame_node"],
    ["add_spatial_mobilenetSSD_node", "object_detection", nn_model_blob, nn_input_size, nn_input_size] 
]
# oakd_pipeline_config = [
#     ["add_rgb_cam_node", preview_width, preview_heigth],
#     ["add_rgb_cam_preview_node"],
#     ["add_nn_node", "palm_detection", "palm_detection_sh6.blob", palm_detection_nn_input_size, palm_detection_nn_input_size],
#     ["add_nn_node", "hand_landmarks", "hand_landmark_sh6.blob", hand_landmarks_nn_input_size, hand_landmarks_nn_input_size],
#     ["add_nn_node", "hand_asl", "hand_asl_6_shaves.blob", hand_asl_nn_input_size, hand_asl_nn_input_size],
# ]

def setup():
    # Using face detection model for now as it's easier to use for testing. Faces are more reliably recognised compared to other stuff
    cait.essentials.initialize_component('vision', processor='oakd', mode=vision_config1)
    # cait.essentials.initialize_component('vision', processor='oakd', mode=)
    cait.essentials.initialize_component('control', [robot_inventor_hub])
    cait.essentials.initialize_pid(0.015, 0, 0)
    
def main():
    global object_coordinate, screen_center, rotate_power, x1, object_of_interest, x2, object_center, coordinates, object2, power
    while True:
        object_of_interest = cait.essentials.detect_objects(processor='oakd', spatial=True)

        target_object_index=-1
        if target_object_name in object_of_interest["names"]:
            target_object_index = object_of_interest["names"].index(target_object_name)
        # object_of_interest['names'] //grab first index of the target object class
        # object_of_interest['coordinates']
       
        cait.essentials.draw_detected_objects(object_of_interest)
        coordinates = object_of_interest['coordinates']
        #if len(coordinates) > 0:
        if target_object_index > -1:
            object2 = coordinates[target_object_index]
            dispatch_thread_0 = threading.Thread(target=dispatch_func_0, daemon=True)
            dispatch_thread_0.start()

if __name__ == "__main__":
    setup()
    main()