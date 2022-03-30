# from https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
class TrackableObject:
    def __init__(self, objectID, centroid, label):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.label = label

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False