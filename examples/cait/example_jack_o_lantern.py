# Python code generated by CAIT's Visual Programming Interface

import random
import cait.essentials
import threading

face_coordinate = None
screen_center = None
rotate_power = None
playing_audio = None
x1 = None
audio_list = None
x2 = None
face_center = None
person = None
coordinates = None
face = None
power = None
selected_index = None
vision_config = [
    ["add_rgb_cam_node", 640, 360], 
    ["add_rgb_cam_preview_node"],
    ["add_nn_node_pipeline", "face_detection", "face-detection-retail-0004_openvino_2021.2_6shave.blob", 300, 300]]

"""Describe this function...
"""
def follow_face(face_coordinate):
    global screen_center, rotate_power, playing_audio, x1, audio_list, x2, face_center, person, coordinates, face, power, selected_index
    screen_center = 640 / 2
    x1 = face_coordinate[0]
    x2 = face_coordinate[2]
    face_center = x1 + (x2 - x1) / 2
    rotate_power = cait.essentials.update_pid((face_center - screen_center))['value']
    cait.essentials.set_motor_power('Robot Inventor: A8:E2:C1:9B:10:36', 'motor_F', rotate_power)
    return rotate_power
    
    
# Auto generated dispatch function code
def dispatch_func_0():
    global face, coordinates, person, audio_list, playing_audio
    power = follow_face(face)
# End of auto generated dispatch function


# Auto generated dispatch function code
def dispatch_func_1():
    global selected_index, face, coordinates, person, audio_list, playing_audio
    cait.essentials.play_audio('/home/pi/Documents/halloween_musics/' + str((audio_list[int(selected_index - 1)])))
    cait.essentials.sleep(2)
    playing_audio = False
# End of auto generated dispatch function


def setup():
    cait.essentials.initialize_component('vision', processor='oakd', mode=vision_config)
    cait.essentials.initialize_component('control', ['Robot Inventor: A8:E2:C1:9B:10:36'])
    cait.essentials.initialize_pid(0.015, 0, 0)
    cait.essentials.initialize_component('voice', mode='on_devie')
    
def main():
    global face_coordinate, screen_center, rotate_power, playing_audio, x1, audio_list, x2, face_center, person, coordinates, face, power, selected_index
    playing_audio = False
    audio_list = cait.essentials.create_file_list('/home/pi/Documents/halloween_musics')['file_list']
    while True:
        person = cait.essentials.detect_face(processor='oakd')
        cait.essentials.draw_detected_face(person)
        coordinates = person['coordinates']
        if len(coordinates) > 0:
            face = coordinates[0]
            dispatch_thread_0 = threading.Thread(target=dispatch_func_0, daemon=True)
            dispatch_thread_0.start()
            if not playing_audio:
                selected_index = random.randint(1, len(audio_list))
                dispatch_thread_1 = threading.Thread(target=dispatch_func_1, daemon=True)
                dispatch_thread_1.start()
                playing_audio = True

if __name__ == "__main__":
    setup()
    main()
