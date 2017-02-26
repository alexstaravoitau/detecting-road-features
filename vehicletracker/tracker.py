import numpy as np
import cv2
from skimage.feature import hog
from skimage.transform import resize
from scipy.ndimage.measurements import label
from vehicletracker.features import FeatureExtractor

class VehicleTracker(object):

    def __init__(self, scaler, classifier):
        self.scaler = scaler
        self.classifier = classifier
        self.detections = np.empty([0, 4], dtype=np.int64)

    def process(self, frame, draw_detections=True):
        self.detect_vehicles(frame, highlight_detections=draw_detections)
        return frame

    def detect_vehicles(self, image, highlight_detections=False):
        scales = np.linspace(.3, .8, 8)
        y_start = np.linspace(.55, .65, 8)[::-1]
        found_coordinates = np.empty([0, 4], dtype=np.int64)
        for scale, y in zip(scales, y_start):
            regions_coordinates = self.get_regions(image, y, scale=scale, k=64)
            found_coordinates = np.append(found_coordinates, regions_coordinates, axis=0)

        self.detections, self.heatmap = self.merge_detections(found_coordinates, image.shape)
        if highlight_detections:
            for c in self.detections:
                cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)

    def get_regions(self, image, y, scale=1., k=64):
        (h, w, d) = image.shape
        scaled_image = resize((image / 255.).astype(np.float64), (int(h * scale), int(w * scale), d), preserve_range=True).astype(np.float32)

        extractor = FeatureExtractor(scaled_image)

        (h, w, d) = scaled_image.shape
        regions = np.empty([0, k, k, d], dtype=np.float32)
        regions_coordinates = np.empty([0, 4], dtype=np.int)
        y = int(h*y)
        s = k // 2
        x_range = np.linspace(0, w - k, (w + s) // s)
        for x in x_range.astype(np.int):
            features = extractor.feature_vector(x, y, k)
            features = self.scaler.transform(np.array(features).reshape(1, -1))

            if self.classifier.predict(features)[0] == 1:
                regions_coordinates = np.append(regions_coordinates, [[x, y, x + k, y + k]], axis=0)

        return (regions_coordinates / scale).astype(np.int)

    def add_heat(self, heatmap, coordinates):
        # Iterate through list of bboxes
        for c in coordinates:
            # Add += 1 for all pixels inside each detected region
            # Assuming each set of coordinates takes the form (x1, y1, x2, y2)
            heatmap[c[1]:c[3], c[0]:c[2]] += 1
        # Return updated heatmap
        return heatmap

    def merge_detections(self, detected_regions, image_shape, threshold=2):
        heatmap = np.zeros((image_shape[0], image_shape[1])).astype(np.float)
        # Add heat to each box in box list
        heatmap = self.add_heat(heatmap, detected_regions)
        # Apply threshold to help remove false positives
        heatmap[heatmap < threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        labels = label(heatmap)
        features = np.empty([0, 4], dtype=np.int64)
        # Iterate through all detected cars
        for car in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car).nonzero()
            features = np.append(
                features,
                [[np.min(nonzero[1]), np.min(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[0])]],
                axis=0
            )
        # Return the image
        return (features, heatmap)