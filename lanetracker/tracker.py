import numpy as np
import cv2
import matplotlib.pyplot as plt
from lanetracker.window import Window
from lanetracker.line import Line


class LaneTracker(object):
    def __init__(self, first_frame, n_windows=9, window_margin=100, min_points=50):
        (self.h, self.w) = first_frame.shape
        self.win_n = n_windows
        self.win_m = window_margin
        self.min_points = min_points
        self.left = None
        self.right = None
        self.l_windows = []
        self.r_windows = []
        self.initialize_lines(first_frame)

    def initialize_lines(self, frame):
        """
        Finds starting points for left and right lines.
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(frame[int(self.h / 2):, :], axis=0)

        nonzero = frame.nonzero()
        # Create empty lists to receive left and right lane pixel indices
        l_indices = np.empty([0], dtype=np.int)
        r_indices = np.empty([0], dtype=np.int)
        window_height = int(self.h / self.win_n)

        for i in range(self.win_n):
            l_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.l_windows[-1].x if len(self.l_windows) > 0 else np.argmax(histogram[:self.w // 2])
            )
            r_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.r_windows[-1].x if len(self.r_windows) > 0 else np.argmax(histogram[self.w // 2:]) + self.w // 2
            )
            # Append nonzero indices in the window boundary to the lists
            l_indices = np.append(l_indices, l_window.pixels_in(nonzero), axis=0)
            r_indices = np.append(r_indices, r_window.pixels_in(nonzero), axis=0)
            self.l_windows.append(l_window)
            self.r_windows.append(r_window)
        self.left = Line(
            x=nonzero[1][l_indices],
            y=nonzero[0][l_indices],
            h=self.h, w = self.w
        )
        self.right = Line(
            x=nonzero[1][r_indices],
            y=nonzero[0][r_indices],
            h=self.h, w = self.w
        )

    def scan_frame_with_windows(self, frame, windows):
        indices = np.empty([0], dtype=np.int)
        nonzero = frame.nonzero()
        for window in windows:
            indices = np.append(indices, window.pixels_in(nonzero), axis=0)
        return (nonzero[1][indices], nonzero[0][indices])

    def process(self, frame):
        (x, y) = self.scan_frame_with_windows(frame, self.l_windows)
        self.left.fit(x, y)

        (x, y) = self.scan_frame_with_windows(frame, self.r_windows)
        self.right.fit(x, y)

    def draw_statistics_overlay(self, binary, lines=True, windows=True):
        image = np.dstack((binary, binary, binary))
        if windows:
            for window in self.l_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
            for window in self.r_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
        if lines:
            cv2.polylines(image, [self.left.points], False, (1., 0, 0), 2)
            cv2.polylines(image, [self.right.points], False, (0, 0, 1.), 2)
        return image

    def draw_lane_overlay(self, image, unwarp_matrix):
        # Create an image to draw the lines on
        color_warp = np.zeros_like(image).astype(np.uint8)
        points = np.vstack((self.left.points, np.flipud(self.right.points)))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, [points], (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarped_lane = cv2.warpPerspective(color_warp, unwarp_matrix, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(image, 1, unwarped_lane, 0.3, 0)

    def radius_of_curvature(self):
        return np.average([self.left.radius_of_curvature(), self.right.radius_of_curvature()])