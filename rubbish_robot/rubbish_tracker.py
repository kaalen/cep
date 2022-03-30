import blobconverter
import cv2
import argparse
import numpy as np
import logging
import time
from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager, NNetManager
import depthai as dai
import trackable_object
from sweeper import SweeperController

class RubbishTracker():
    """Encapsulates rubbish tracking functionality.
        * can count cummulative number of objects.
          The count increases when an object centroid passes over the boudary line either up or down.
        * robot movement
    """

    PREVIEW_WIDTH = 300#960
    PREVIEW_HEIGHT = 300#540
    DISTANCE_THRESHOLD = 60

    labelMap = ["", "biomaterial", "rubbish"]
    onCountObserve = []

    def __init__(self, model, show, axis, roi_position, save_path, activate_robot):
        self.model = model
        self.show = show
        self.axis = axis
        self.roi_position = roi_position
        self.save_path = save_path
        self.activate_robot = activate_robot

        self.sweeper = None

        # Create pipeline
        self.pm = PipelineManager()
        self.nm = NNetManager(inputSize=(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), nnFamily="mobilenet")

        # Define sources and outputs
        self.cam = self.configCamera()
        self.nn = self.pm.pipeline.create(dai.node.MobileNetDetectionNetwork)

        xoutVideo = self.pm.pipeline.create(dai.node.XLinkOut)
        xoutPreview = self.pm.pipeline.create(dai.node.XLinkOut)
        nnOut = self.pm.pipeline.create(dai.node.XLinkOut)

        xoutVideo.setStreamName("video")
        xoutPreview.setStreamName("preview")
        nnOut.setStreamName("nn")

        # Define a neural network that will make predictions based on the source frames
        self.pm.addNn(self.nn)
        self.nn.setConfidenceThreshold(0.5)
        self.nn.setBlobPath(self.model)
        self.nn.setNumInferenceThreads(2)
        self.nn.input.setBlocking(False)

        # Connect (link) the camera preview output to the neural network input
        self.cam.video.link(xoutVideo.input)
        self.cam.preview.link(xoutPreview.input)
        self.cam.preview.link(self.nn.input)
        self.nn.out.link(nnOut.input) # NN output (inference) stream

        # Create and configure the object tracker
        objectTracker = self.pm.pipeline.create(dai.node.ObjectTracker)
        # objectTracker.setDetectionLabelsToTrack([0])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        # objectTracker.setTrackerIdAssigmentPolicy(
        #     dai.TrackerIdAssigmentPolicy.SMALLEST_ID)
        # Link detection networks outputs to the object tracker
        self.nn.passthrough.link(objectTracker.inputTrackerFrame)
        self.nn.passthrough.link(objectTracker.inputDetectionFrame)
        self.nn.out.link(objectTracker.inputDetections)

        # Send tracklets to the host
        trackerOut = self.pm.pipeline.create(dai.node.XLinkOut)
        trackerOut.setStreamName("tracklets")
        objectTracker.out.link(trackerOut.input)

    def configCamera(self):
        cam = self.pm.pipeline.create(dai.node.ColorCamera)
        
        # Set color camera properties
        cam.setPreviewSize(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setPreviewKeepAspectRatio(False)
        cam.setFps(15)
        return cam

    def display_label_count(self, videoFrame, labelCounter):
        """Displays label counter for tracked object on the video frame

            Args:
            videoFrame: video frame on which to display count
            labelCounter: int array of counters for each label
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(labelCounter)):  
            if self.labelMap[i] != "":              
                cv2.putText(videoFrame, f'{self.labelMap[i]}: {labelCounter[i]}', (10, 60+(25*i)), font, 0.8, (0, 0, 255), 2, font)

    def move_robot(self, active_to, centroid):
        if self.sweeper is None:
            return

        if active_to is not None:
            cx = centroid[0]
            screen_cx = 1920/2
            diff_x = screen_cx - cx
            distance_factor = 20
            
            if abs(diff_x) > self.DISTANCE_THRESHOLD:
                logging.info(f"Tracklet {str(active_to.id)} centroid: {str(centroid[0])}, {str(centroid[1])}   diff_x: {str(diff_x)}")
                # only move robot if not close enough (prevents jittery motion)
                location = diff_x/distance_factor
                direction = "right"
                if cx < screen_cx:
                    # move right
                    location = -1 * location
                    direction = "left"

                if self.sweeper.isBusy() == False:
                    logging.info(f"move {direction}: {str(location)}")
                    self.sweeper.goToLocation(location, True)
                    self.sweeper.setScoopUp()


    def alertOnCount(self, dir, newValue):
        for func in self.onCountObserve:
            func(dir, newValue)

    def subscribeOnCount(self, callback):
        self.onCountObserve.append(callback)
    
    def run(self):

        with dai.Device(self.pm.pipeline) as device:
            pv = PreviewManager(display=[Previews.color.name])

            # Output queues will be used to get the frames and nn data from the outputs defined above
            qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
            qPreview = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
            qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            if self.activate_robot:
                self.sweeper = SweeperController()
                self.sweeper.runSweeper()

            if self.save_path:
                width = 300
                height = 300
                fps = 30

                out = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(
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
            labelCounter = [0] * len(self.labelMap)

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
                    cv2.putText(frame, self.labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                # Show the frame
                cv2.imshow(name, frame)

            cv2.namedWindow("video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("video", 1280, 720)
            print("Resize video window with mouse drag!")

            active_to = None # Active tracked object - the one we're trying to capture
            if self.sweeper is not None:
                self.sweeper.setScoopUp()

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

                        if len(trackletsData) == 0:
                            active_to = None

                        centroid = None
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
                                if self.axis and not to.counted:
                                    x = [c[0] for c in to.centroids]
                                    direction = centroid[0] - np.mean(x)

                                    if centroid[0] > self.roi_position*width and direction > 0 and np.mean(x) < self.roi_position*width:
                                        counter[1] += 1 #left
                                        labelCounter[to.label] += 1
                                        to.counted = True
                                        self.alertOnCount("left", labelCounter[to.label])
        
                                    elif centroid[0] < self.roi_position*width and direction < 0 and np.mean(x) > self.roi_position*width:
                                        counter[0] += 1 #right
                                        labelCounter[to.label] += 1
                                        to.counted = True
                                        self.alertOnCount("right", labelCounter[to.label])


                                elif not self.axis and not to.counted:
                                    y = [c[1] for c in to.centroids]
                                    direction = centroid[1] - np.mean(y)

                                    if centroid[1] > self.roi_position*height and direction > 0 and np.mean(y) < self.roi_position*height:
                                        counter[3] += 1 #down
                                        labelCounter[to.label] += 1
                                        to.counted = True
                                        self.alertOnCount("down", labelCounter[to.label])

                                    elif centroid[1] < self.roi_position*height and direction < 0 and np.mean(y) > self.roi_position*height:
                                        counter[2] += 1 #up
                                        labelCounter[to.label] += 1
                                        to.counted = True
                                        self.alertOnCount("up", labelCounter[to.label])


                                to.centroids.append(centroid)

                            trackableObjects[t.id] = to

                            if t.status != dai.Tracklet.TrackingStatus.LOST and t.status != dai.Tracklet.TrackingStatus.REMOVED:
                                text = "ID {}".format(t.id)
                                cv2.putText(videoFrame, text, (centroid[0] - 10, centroid[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.circle(
                                    videoFrame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

                            if active_to is None:
                                # Activate the first object
                                active_to = t
                            else:
                                # check if active object lost or removed
                                remove = False
                                matching = [x for x in trackletsData if x.id == active_to.id]
                                if len(matching) == 0:
                                    # if scoop is down, go and dump the object
                                    # goDumpObject(sweeper)
                                    logging.warning(f"**** no matching object for {str(active_to.id)}")
                                    remove = True
                                    # x.status == dai.Tracklet.TrackingStatus.LOST or 
                                lost_or_removed = [x for x in trackletsData if x.status == dai.Tracklet.TrackingStatus.REMOVED]
                                matching_lost_removed = [x for x in lost_or_removed if x.id == active_to.id]
                                if len(matching_lost_removed) > 0:
                                    for x in matching_lost_removed:
                                        status = "Lost" if x.status == dai.Tracklet.TrackingStatus.LOST else "Removed"
                                        logging.warning(f"              {status}: {str(x.id)}")
                                        remove = True
                                if remove:
                                    active_to = None
                                    logging.warning(f"**** Removed object")
                                    if self.sweeper is not None: 
                                        self.sweeper.dumpAndReturn()
                
                        # if active_to is not None and active_to.id == t.id:
                            
                        self.move_robot(active_to, centroid)

                    # Draw ROI line
                    if self.axis:
                        cv2.line(videoFrame, (int(self.roi_position*width), 0),
                                (int(self.roi_position*width), height), (0xFF, 0, 0), 5)
                    else:
                        cv2.line(videoFrame, (0, int(self.roi_position*height)),
                                (width, int(self.roi_position*height)), (0xFF, 0, 0), 5)

                    # display count and status
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if self.axis:
                        cv2.putText(videoFrame, f'Left: {counter[0]}; Right: {counter[1]}', (
                            10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                    else:
                        cv2.putText(videoFrame, f'Up: {counter[2]}; Down: {counter[3]}', (
                            10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

                    self.display_label_count(videoFrame, labelCounter)


                    if self.show:
                        if videoFrame is not None:
                            displayFrame("video", videoFrame)

                        # if previewFrame is not None:
                        #     displayFrame("preview", previewFrame)

                    if self.save_path:
                        out.write(videoFrame)

                if cv2.waitKey(1) == ord('q'):
                    break


            cv2.destroyAllWindows()

            if self.save_path:
                out.release()

def main():
    logging.getLogger().setLevel(logging.INFO)
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

    app = RubbishTracker(
        model=args.model,
        show=args.show,
        axis=args.axis,
        roi_position=args.roi_position,
        save_path=args.save_path,
        activate_robot=False
    )
    app.run()

if __name__ == "__main__":
    main()
