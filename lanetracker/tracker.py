import numpy as np
import cv2
import matplotlib.pyplot as plt
from lanetracker.window import Window
from lanetracker.line import Line


class LaneTracker(object):
    """
    Tracks the lane in a series of consecutive frames.
    """

    def __init__(self, first_frame, n_windows=9):
        """
        Initialises a tracker object.

        Parameters
        ----------
        first_frame     : First frame of the frame series. We use it to get dimensions and initialise values.
        n_windows       : Number of windows we use to track each lane edge.
        """
        (self.h, self.w) = first_frame.shape
        self.win_n = n_windows
        self.left = None
        self.right = None
        self.l_windows = []
        self.r_windows = []
        self.initialize_lines(first_frame)

    def initialize_lines(self, frame):
        """
        Finds starting points for left and right lines (e.g. lane edges) and initialises Window and Line objects.

        Parameters
        ----------
        frame   : Frame to scan for lane edges.
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
        """
        Scans a frame using initialised windows in an attempt to track the lane edges.

        Parameters
        ----------
        frame   : New frame
        windows : Array of windows to use for scanning the frame.

        Returns
        -------
        A tuple of arrays containing coordinates of points found in the specified windows.
        """
        indices = np.empty([0], dtype=np.int)
        nonzero = frame.nonzero()
        for window in windows:
            indices = np.append(indices, window.pixels_in(nonzero), axis=0)
        return (nonzero[1][indices], nonzero[0][indices])

    def process(self, frame):
        """
        Performs a full lane tracking pipeline on a frame.

        Parameters
        ----------
        frame   : New frame to process.
        """
        (x, y) = self.scan_frame_with_windows(frame, self.l_windows)
        self.left.fit(x, y)

        (x, y) = self.scan_frame_with_windows(frame, self.r_windows)
        self.right.fit(x, y)

    def draw_statistics_overlay(self, binary, lines=True, windows=True):
        """
        Draws an overlay with debugging information on a bird's-eye view of the road (e.g. after applying perspective
        transform).

        Parameters
        ----------
        binary  : Frame to overlay.
        lines   : Flag indicating if we need to draw lines.
        windows : Flag indicating if we need to draw windows.

        Returns
        -------
        Frame with an debug information overlay.
        """
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
        """
        Draws an overlay with tracked lane applying perspective unwarp to project it on the original frame.

        Parameters
        ----------
        image           : Original frame.
        unwarp_matrix   : Transformation matrix to unwarp the bird's eye view to initial frame.

        Returns
        -------
        Frame with a lane overlay.
        """
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
        """
        Calculates radius of the lane curvature by averaging curvature of the edge lines.

        Returns
        -------
        Radius of the lane curvature in meters.
        """
        return np.average([self.left.radius_of_curvature(), self.right.radius_of_curvature()])