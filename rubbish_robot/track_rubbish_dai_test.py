#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import trackable_object
from depthai_sdk.managers import *
from datetime import datetime

labelMap = ["", "biomaterial", "rubbish"]


nnPathDefault = "src/curt/models/modules/vision/platforms/oakd/rubbish_detection/rubbish-detection_openvino_2021.4_5shave.blob"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model', type=str, help='File path of .blob file.')
parser.add_argument('-roi', '--roi_position', type=float,
                    default=0.5, help='ROI Position (0-1)')
parser.add_argument('-a', '--axis', default=False, action='store_false',
                    help='Axis for cumulative counting (default=x axis)')
parser.add_argument('-sh', '--show', default=True,
                    action='store_false', help='Show output')
parser.add_argument('-sp', '--save_path', type=str, default='',
                    help='Path to save the output. If None output won\'t be saved')
parser.add_argument('-s', '--sync', action="store_true",
                    help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

fullFrameTracking = True

WIDTH, HEIGHT = 300, 300

# Create pipeline
pManager = PipelineManager()
nNetMan = NNetManager(inputSize=(WIDTH, HEIGHT), nnFamily="mobilenet")

# Define sources and outputs
# Inputs
camRgb = pManager.pipeline.create(dai.node.ColorCamera)
detectionNetwork = pManager.pipeline.create(dai.node.MobileNetDetectionNetwork)
objectTracker = pManager.pipeline.create(dai.node.ObjectTracker)


# Outputs
xlinkOut = pManager.pipeline.create(dai.node.XLinkOut)
trackerOut = pManager.pipeline.create(dai.node.XLinkOut)
detectionsOut = pManager.pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("preview")
trackerOut.setStreamName("tracklets")
detectionsOut.setStreamName('detections')

# Properties
camRgb.setPreviewSize(WIDTH, HEIGHT)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# testing MobileNet DetectionNetwork
pManager.addNn(detectionNetwork)
detectionNetwork.setBlobPath(nnPathDefault)
detectionNetwork.setConfidenceThreshold(0.2)
detectionNetwork.input.setBlocking(False)

#objectTracker.setDetectionLabelsToTrack([15])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Linking
camRgb.preview.link(detectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)




detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

detectionNetwork.out.link(objectTracker.inputDetections)
objectTracker.out.link(trackerOut.input)

# Connect to device and start pipeline
with dai.Device(pManager.pipeline) as device:

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)
    detections = device.getOutputQueue("detections", 4, False)


    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None
    trackableObjects = {}

    cv2.namedWindow("tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("tracker", 2*WIDTH, 2*HEIGHT)

    while(True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter+=1

        labelCounter = [0] * len(labelMap)

        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        color = (255, 0, 0)
        frame = imgFrame.getCvFrame()
        track = tracklets.tryGet()
        inDet = detections.tryGet()
        cDetections = None
        if inDet:
            cDetections = inDet.detections


        t0 = datetime.now()
        if frame is not None:
            height = frame.shape[0]
            width = frame.shape[1]
            if track:
                trackletsData = track.tracklets
                print(len(trackletsData))
                for t in trackletsData:
                    to = trackableObjects.get(t.id, None)
                    # calculate centroid
                    roi = t.roi.denormalize(width, height)
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)
                    centroid = (int((x2-x1)/2+x1), int((y2-y1)/2+y1))

                    tWidth = x2 - x1
                    tHeight = y1 - y2

                    # If new tracklet, save its centroid
                    if t.status == dai.Tracklet.TrackingStatus.NEW:
                        to = trackable_object.TrackableObject(t.id, centroid, t.label)
                    elif to is not None:
                        if args.axis and not to.counted:
                            x = [c[0] for c in to.centroids]
                            direction = centroid[0] - np.mean(x)
                            if centroid[0] > args.roi_position*width and direction > 0 and np.mean(x) < args.roi_position*width:
                                labelCounter[to.label] += 1
                                to.counted = True

                            elif centroid[0] < args.roi_position*width and direction < 0 and np.mean(x) > args.roi_position*width:
                                labelCounter[to.label] += 1
                                to.counted = True
                        elif not args.axis and not to.counted:
                            y = [c[1] for c in to.centroids]
                            direction = centroid[1] - np.mean(y)
                            if centroid[1] > args.roi_position*height and direction > 0 and np.mean(y) < args.roi_position*height:
                                labelCounter[to.label] += 1
                                to.counted = True
                            elif centroid[1] < args.roi_position*height and direction < 0 and np.mean(y) > args.roi_position*height:
                                labelCounter[to.label] += 1
                                to.counted = True
                        to.centroids.append(centroid)
                    trackableObjects[t.id] = to
                    if t.status != dai.Tracklet.TrackingStatus.LOST and t.status != dai.Tracklet.TrackingStatus.REMOVED:
                        text = "ID {}".format(t.id)
                        cv2.circle(
                            frame, (centroid[0], centroid[1]), tWidth, (255, 0, 0), -1)
                        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
        t1 = datetime.now()
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        frame = cv2.resize(frame, (600, 600))
        cv2.imshow("tracker", frame)
        print(cDetections)

        if cv2.waitKey(1) == ord('q'):
            break
