import numpy as np
import cv2
from skimage.feature import hog
from skimage.transform import resize
from scipy.ndimage.measurements import label
from vehicletracker.feature_extraction import extract_features

class VehicleTracker(object):

    def __init__(self, scaler, classifier):
        self.scaler = scaler
        self.classifier = classifier
        self.detections = np.empty([0, 4], dtype=np.int64)

    def process(self, frame, draw_detections=True):
        self.detect_vehicles(frame, highlight_detections=draw_detections)
        return frame

    def detect_vehicles(self, image, highlight_detections=False):
        found_coordinates = np.empty([0, 4], dtype=np.int64)
        for scale in np.linspace(.2, 1., 5):
            regions, regions_coordinates = self.get_regions(image, scale=scale, k=64)
            predictions = self.classifier.predict(self.scaler.transform(extract_features(regions)))
            # print(regions.shape[0], 'regions,', int(predictions.sum()), 'with cars.')
            found_coordinates = np.append(found_coordinates, regions_coordinates[predictions == 1], axis=0)

        self.detections, self.heatmap = self.merge_detections(found_coordinates, image.shape)
        if highlight_detections:
            for c in self.detections:
                cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)

    def get_regions(self, image, scale=1., k=64):
        (h, w, d) = image.shape
        scaled_image = resize((image / 255.).astype(np.float64), (h * scale, w * scale, d), preserve_range=True).astype(np.float32)
        (h, w, d) = scaled_image.shape
        regions = np.empty([0, k, k, d], dtype=np.float32)
        regions_coordinates = np.empty([0, 4], dtype=np.int)
        s = k // 2
        y_range, y_s = np.linspace(h / 2, h - k, (h + s) // (2 * s), retstep=True)
        x_range, x_s = np.linspace(0, w - k, (w + s) // s, retstep=True)
        for i in y_range.astype(np.int):
            for j in x_range.astype(np.int):
                regions = np.append(regions, [scaled_image[i:i+k, j:j+k, :]], axis=0)
                regions_coordinates = np.append(regions_coordinates, [[j, i, j+k, i+k]], axis=0)

        return regions.astype(np.float32), (regions_coordinates / scale).astype(np.int)

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