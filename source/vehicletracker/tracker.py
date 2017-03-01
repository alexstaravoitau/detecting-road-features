import numpy as np
import cv2
from skimage.feature import hog
from skimage.transform import resize
from scipy.ndimage.measurements import label
from vehicletracker.features import FeatureExtractor
from collections import deque

class VehicleTracker(object):
    """
    Tracks surrounding vehicles in a series of consecutive frames.
    """

    def __init__(self, scaler, classifier, first_frame):
        """
        Initialises an instance.

        Parameters
        ----------
        scaler      : SciPy scaler to apply to X.
        classifier  : Trained SciPy classifier for detecting vehicles.
        first_frame : First video frame.
        """
        self.scaler = scaler
        self.classifier = classifier
        self.frame_shape = first_frame.shape
        self.detections_history = deque(maxlen=20)

    def process(self, frame, draw_detections=True):
        """
        Perform single frame processing and saves detected vehicles data.

        Parameters
        ----------
        frame           : Current video frame.
        draw_detections : Flag indicating if we need to highlight vehicles in the frame.

        Returns
        -------
        Video frame
        """
        self.detect_vehicles(frame)
        if draw_detections:
            for c in self.detections():
                cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)
        return frame

    def detections(self):
        """
        Approximates surrounding vehicles based on a heatmap of last N detections.

        Returns
        -------
        Boundaries of detected vehicles.
        """
        detections, _ = self.merge_detections(
            np.concatenate(np.array(self.detections_history)),
            self.frame_shape,
            threshold=min(len(self.detections_history), 15)
        )
        return detections

    def detect_vehicles(self, image):
        """
        Perform a full window passthrough in the specified frame.

        Parameters
        ----------
        image   : Current frame.
        """
        scales = np.array([.3, .5, .65, .8])
        y_top = np.array([.6, .57, .56, .55])
        frame_detections = np.empty([0, 4], dtype=np.int64)
        for scale, y in zip(scales, y_top):
            scale_detections = self.detections_for_scale(image, scale, y, 64)
            frame_detections = np.append(frame_detections, scale_detections, axis=0)
        detections, self.heatmap = self.merge_detections(frame_detections, image.shape, threshold=1)
        self.detections_history.append(detections)


    def detections_for_scale(self, image, scale, y, k):
        """
        Runs a classifier on all windows for specified frame scale.

        Parameters
        ----------
        image   : Current frame.
        scale   : Scale of the image.
        y       : Top Y coordinate of the windows.
        k       : Size of the window.

        Returns
        -------
        Boundaries of windows that got detections.
        """
        (h, w, d) = image.shape
        scaled = resize((image / 255.).astype(np.float64), (int(h * scale), int(w * scale), d), preserve_range=True).astype(np.float32)
        extractor = FeatureExtractor(scaled)
        (h, w, d) = scaled.shape
        detections = np.empty([0, 4], dtype=np.int)
        y = int(h*y)
        s = k // 3
        x_range = np.linspace(0, w - k, (w + s) // s)
        for x in x_range.astype(np.int):
            features = extractor.feature_vector(x, y, k)
            features = self.scaler.transform(np.array(features).reshape(1, -1))
            if self.classifier.predict(features)[0] == 1:
                detections = np.append(detections, [[x, y, x + k, y + k]], axis=0)
        return (detections / scale).astype(np.int)

    def add_heat(self, heatmap, coordinates):
        """
        Adds a 1 for pixels inside each detected region.

        Parameters
        ----------
        heatmap     : Array with a heatmap.
        coordinates : Detections to merge.

        Returns
        -------
        Updated heatmap.
        """
        for c in coordinates:
            # Assuming each set of coordinates takes the form (x1, y1, x2, y2)
            heatmap[c[1]:c[3], c[0]:c[2]] += 1
        return heatmap

    def merge_detections(self, detections, image_shape, threshold):
        """
        Merges specified detections based on a heatmap and threshold.
        Parameters
        ----------
        detections  : Array of detections to merge.
        image_shape : Shape of the image.
        threshold   : Heatmap threshold.

        Returns
        -------
        Tuple of merged regions and a heatmap.
        """
        heatmap = np.zeros((image_shape[0], image_shape[1])).astype(np.float)
        # Add heat to each box in box list
        heatmap = self.add_heat(heatmap, detections)
        # Apply threshold to help remove false positives
        heatmap[heatmap < threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        labels = label(heatmap)
        cars = np.empty([0, 4], dtype=np.int64)
        # Iterate through all detected cars
        for car in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car).nonzero()
            cars = np.append(
                cars,
                [[np.min(nonzero[1]), np.min(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[0])]],
                axis=0
            )
        # Return the image
        return (cars, heatmap)