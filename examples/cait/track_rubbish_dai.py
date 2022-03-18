"""Here's another example that uses our rubbish detection model to count cummulative number of objects.
    The count increases when an object centroid passes over the boudary line either up or down.
"""
import blobconverter
import cv2
import argparse
import numpy as np
import time
from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager, NNetManager
import depthai as dai
import trackable_object

PREVIEW_WIDTH = 300#960
PREVIEW_HEIGHT = 300#540


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

if args.model is None:
    args.model = "src/curt/models/modules/vision/platforms/oakd/rubbish_detection/rubbish-detection_openvino_2021.4_5shave.blob"
    #blobconverter.from_zoo(name="mobilenet-ssd", shaves=7)

# Create pipeline
# pipeline = dai.Pipeline()

pm = PipelineManager()
# pm.createColorCam(xout=True)

nm = NNetManager(inputSize=(300, 300), nnFamily="mobilenet")

# Define a neural network that will make predictions based on the source frames
nn = pm.pipeline.create(dai.node.MobileNetDetectionNetwork)
# nn = nm.createNN(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
#                 blobPath=args.model)
pm.addNn(nn)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(args.model)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Create color camera node.
cam = pm.pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
cam.setInterleaved(False)
# Connect (link) the camera preview output to the neural network input
cam.preview.link(nn.input)

# Create XLinkOut object as conduit for passing camera frames to the host
xoutFrame = pm.pipeline.create(dai.node.XLinkOut)
xoutFrame.setStreamName("outFrame")
cam.preview.link(xoutFrame.input)

# Create neural network output (inference) stream
nnOut = pm.pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nnOut")
nn.out.link(nnOut.input)

# Create and configure the object tracker
objectTracker = pm.pipeline.create(dai.node.ObjectTracker)
# objectTracker.setDetectionLabelsToTrack([0])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
# objectTracker.setTrackerIdAssigmentPolicy(
#     dai.TrackerIdAssigmentPolicy.SMALLEST_ID)
# Link detection networks outputs to the object tracker
nn.passthrough.link(objectTracker.inputTrackerFrame)
nn.passthrough.link(objectTracker.inputDetectionFrame)
nn.out.link(objectTracker.inputDetections)

# Send tracklets to the host
trackerOut = pm.pipeline.create(dai.node.XLinkOut)
trackerOut.setStreamName("tracklets")
objectTracker.out.link(trackerOut.input)

# Pipeline defined, now the device is connected to
with dai.Device(pm.pipeline) as device:
    pv = PreviewManager(display=[Previews.color.name])
    # pv.createQueues(device)
    # nm.createQueues(device)

    # Output queue for retrieving camera frames from device
    qOut_Frame = device.getOutputQueue(name="outFrame", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nnOut", maxSize=4, blocking=False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    if args.save_path:
        width = 300
        height = 300
        fps = 30

        out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps, (width, height))

    def should_run():
        return True

    def get_frame():
        in_Frame = qOut_Frame.get()
        frame = in_Frame.getCvFrame()
        return True, frame

    startTime = time.monotonic()
    detections = []
    frame_count = 0
    counter = [0, 0, 0, 0]  # left, right, up, down

    trackableObjects = {}

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    while should_run():
        # Get image frames from camera or video file
        read_correctly, frame = get_frame()
        if not read_correctly:
            break

        in_Frame = qOut_Frame.tryGet()

        if in_Frame is not None:
            frame = in_Frame.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(frame_count / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        inDet = qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections
            frame_count += 1

        track = tracklets.tryGet()

        if frame is not None:
            height = frame.shape[0]
            width = frame.shape[1]

            if track:
                trackletsData = track.tracklets
                for t in trackletsData:
                    to = trackableObjects.get(t.id, None)

                    # calculate centroid
                    roi = t.roi.denormalize(width, height)
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)
                    centroid = (int((x2-x1)/2+x1), int((y2-y1)/2+y1))

                    # If new tracklet, save its centroid
                    if t.status == dai.Tracklet.TrackingStatus.NEW:
                        to = trackable_object.TrackableObject(t.id, centroid)
                    elif to is not None:
                        if args.axis and not to.counted:
                            x = [c[0] for c in to.centroids]
                            direction = centroid[0] - np.mean(x)

                            if centroid[0] > args.roi_position*width and direction > 0 and np.mean(x) < args.roi_position*width:
                                counter[1] += 1
                                to.counted = True
                            elif centroid[0] < args.roi_position*width and direction < 0 and np.mean(x) > args.roi_position*width:
                                counter[0] += 1
                                to.counted = True

                        elif not args.axis and not to.counted:
                            y = [c[1] for c in to.centroids]
                            direction = centroid[1] - np.mean(y)

                            if centroid[1] > args.roi_position*height and direction > 0 and np.mean(y) < args.roi_position*height:
                                counter[3] += 1
                                to.counted = True
                            elif centroid[1] < args.roi_position*height and direction < 0 and np.mean(y) > args.roi_position*height:
                                counter[2] += 1
                                to.counted = True

                        to.centroids.append(centroid)

                    trackableObjects[t.id] = to

                    if t.status != dai.Tracklet.TrackingStatus.LOST and t.status != dai.Tracklet.TrackingStatus.REMOVED:
                        text = "ID {}".format(t.id)
                        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.circle(
                            frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            # Draw ROI line
            if args.axis:
                cv2.line(frame, (int(args.roi_position*width), 0),
                        (int(args.roi_position*width), height), (0xFF, 0, 0), 5)
            else:
                cv2.line(frame, (0, int(args.roi_position*height)),
                        (width, int(args.roi_position*height)), (0xFF, 0, 0), 5)

            # display count and status
            font = cv2.FONT_HERSHEY_SIMPLEX
            if args.axis:
                cv2.putText(frame, f'Left: {counter[0]}; Right: {counter[1]}', (
                    10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
            else:
                cv2.putText(frame, f'Up: {counter[2]}; Down: {counter[3]}', (
                    10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

            if args.show:
                cv2.imshow('cumulative_object_counting', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            if args.save_path:
                out.write(frame)

        pv.showFrames()

        if cv2.waitKey(1) == ord('q'):
            break


    cv2.destroyAllWindows()

    if args.save_path:
        out.release()
