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
    args.model = "src/curt/models/modules/vision/platforms/oakd/rubbishdetection_fCNN_openvino_2021.4_6shave.blob"
    #blobconverter.from_zoo(name="mobilenet-ssd", shaves=7)

labelMap = ["", "biomaterial", "rubbish"]

# Create pipeline
pm = PipelineManager()
nm = NNetManager(inputSize=(PREVIEW_WIDTH, PREVIEW_HEIGHT), nnFamily="mobilenet")

# Define sources and outputs
cam = pm.pipeline.create(dai.node.ColorCamera)
nn = pm.pipeline.create(dai.node.MobileNetDetectionNetwork)

xoutVideo = pm.pipeline.create(dai.node.XLinkOut)
xoutPreview = pm.pipeline.create(dai.node.XLinkOut)
nnOut = pm.pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")
xoutPreview.setStreamName("preview")
nnOut.setStreamName("nn")

# Set color camera properties
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setPreviewKeepAspectRatio(False)

# Define a neural network that will make predictions based on the source frames
pm.addNn(nn)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(args.model)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Connect (link) the camera preview output to the neural network input
cam.video.link(xoutVideo.input)
cam.preview.link(xoutPreview.input)
cam.preview.link(nn.input)
nn.out.link(nnOut.input) # NN output (inference) stream

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

def display_label_count(videoFrame, labelMap, labelCounter):
    """Displays label counter for tracked object on the video frame

        Args:
        videoFrame: video frame on which to display count
        labelMap: array mapping numeric labels to string
        labelCounter: int array of counters for each label
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(labelCounter)):  
        if labelMap[i] != "":              
            cv2.putText(videoFrame, f'{labelMap[i]}: {labelCounter[i]}', (10, 60+(25*i)), font, 0.8, (0, 0, 255), 2, font)

onCountObserve = []
def alertOnCount(dir, newValue):
    for func in onCountObserve:
        func(dir, newValue)
def subscribeOnCount(callback):
    onCountObserve.append(callback)
    
def main():
    with dai.Device(pm.pipeline) as device:
        pv = PreviewManager(display=[Previews.color.name])

        # Output queues will be used to get the frames and nn data from the outputs defined above
        qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        qPreview = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
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
            inVideo = qVideo.get()
            frame = inVideo.getCvFrame()
            return True, frame

        startTime = time.monotonic()
        detections = []
        previewFrame = None
        videoFrame = None
    
        frame_count = 0
        counter = [0, 0, 0, 0]  # left, right, up, down
        labelCounter = [0] * len(labelMap)

        trackableObjects = {}

        def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
            return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        def displayFrame(name, frame):
            color = (255, 0, 0)
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # Show the frame
            cv2.imshow(name, frame)

        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("video", 1280, 720)
        print("Resize video window with mouse drag!")

        while should_run():
            # Get image frames from camera or video file
            read_correctly, frame = get_frame()
            if not read_correctly:
                break

            # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
            inVideo = qVideo.tryGet()
            inPreview = qPreview.tryGet()
            inDet = qDet.tryGet()

            if inVideo is not None:
                videoFrame = inVideo.getCvFrame()
                cv2.putText(videoFrame, "NN fps: {:.2f}".format(frame_count / (time.monotonic() - startTime)),
                            (2, videoFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            if inVideo is not None:
                videoFrame = inVideo.getCvFrame()

            if inPreview is not None:
                previewFrame = inPreview.getCvFrame()

            if inDet is not None:
                detections = inDet.detections
                frame_count += 1

            track = tracklets.tryGet()

            if videoFrame is not None:
                height = videoFrame.shape[0]
                width = videoFrame.shape[1]

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

                        # If new tracklet, save its centroid
                        if t.status == dai.Tracklet.TrackingStatus.NEW:
                            to = trackable_object.TrackableObject(t.id, centroid, t.label)
                        elif to is not None:
                            if args.axis and not to.counted:
                                x = [c[0] for c in to.centroids]
                                direction = centroid[0] - np.mean(x)

                                if centroid[0] > args.roi_position*width and direction > 0 and np.mean(x) < args.roi_position*width:
                                    counter[1] += 1 #left
                                    labelCounter[to.label] += 1
                                    to.counted = True
                                    alertOnCount("left", labelCounter[to.label])
    
                                elif centroid[0] < args.roi_position*width and direction < 0 and np.mean(x) > args.roi_position*width:
                                    counter[0] += 1 #right
                                    labelCounter[to.label] += 1
                                    to.counted = True
                                    alertOnCount("right", labelCounter[to.label])


                            elif not args.axis and not to.counted:
                                y = [c[1] for c in to.centroids]
                                direction = centroid[1] - np.mean(y)

                                if centroid[1] > args.roi_position*height and direction > 0 and np.mean(y) < args.roi_position*height:
                                    counter[3] += 1 #down
                                    labelCounter[to.label] += 1
                                    to.counted = True
                                    alertOnCount("down", labelCounter[to.label])

                                elif centroid[1] < args.roi_position*height and direction < 0 and np.mean(y) > args.roi_position*height:
                                    counter[2] += 1 #up
                                    labelCounter[to.label] += 1
                                    to.counted = True
                                    alertOnCount("up", labelCounter[to.label])


                            to.centroids.append(centroid)

                        trackableObjects[t.id] = to

                        if t.status != dai.Tracklet.TrackingStatus.LOST and t.status != dai.Tracklet.TrackingStatus.REMOVED:
                            text = "ID {}".format(t.id)
                            cv2.putText(videoFrame, text, (centroid[0] - 10, centroid[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.circle(
                                videoFrame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

                # Draw ROI line
                if args.axis:
                    cv2.line(videoFrame, (int(args.roi_position*width), 0),
                            (int(args.roi_position*width), height), (0xFF, 0, 0), 5)
                else:
                    cv2.line(videoFrame, (0, int(args.roi_position*height)),
                            (width, int(args.roi_position*height)), (0xFF, 0, 0), 5)

                # display count and status
                font = cv2.FONT_HERSHEY_SIMPLEX
                if args.axis:
                    cv2.putText(videoFrame, f'Left: {counter[0]}; Right: {counter[1]}', (
                        10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(videoFrame, f'Up: {counter[2]}; Down: {counter[3]}', (
                        10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

                display_label_count(videoFrame, labelMap, labelCounter)


                if args.show:
                    if videoFrame is not None:
                        displayFrame("video", videoFrame)

                    # if previewFrame is not None:
                    #     displayFrame("preview", previewFrame)

                if args.save_path:
                    out.write(videoFrame)

            if cv2.waitKey(1) == ord('q'):
                break


        cv2.destroyAllWindows()

        if args.save_path:
            out.release()

if __name__ == "__main__":
    main()
