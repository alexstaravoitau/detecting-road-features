import numpy as np
import cv2
from lanetracker.window import Window
from lanetracker.line import Line
from lanetracker.gradients import get_edges
from lanetracker.perspective import flatten_perspective


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
        (self.h, self.w, _) = first_frame.shape
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
        edges = get_edges(frame)
        (flat_edges, _) = flatten_perspective(edges)
        histogram = np.sum(flat_edges[int(self.h / 2):, :], axis=0)

        nonzero = flat_edges.nonzero()
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
        self.left = Line(x=nonzero[1][l_indices], y=nonzero[0][l_indices], h=self.h, w = self.w)
        self.right = Line(x=nonzero[1][r_indices], y=nonzero[0][r_indices], h=self.h, w = self.w)

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
        window_x = None
        for window in windows:
            indices = np.append(indices, window.pixels_in(nonzero, window_x), axis=0)
            window_x = window.mean_x
        return (nonzero[1][indices], nonzero[0][indices])

    def process(self, frame, draw_lane=True, draw_statistics=True):
        """
        Performs a full lane tracking pipeline on a frame.

        Parameters
        ----------
        frame               : New frame to process.
        draw_lane           : Flag indicating if we need to draw the lane on top of the frame.
        draw_statistics     : Flag indicating if we need to render the debug information on top of the frame.

        Returns
        -------
        Resulting frame.
        """
        edges = get_edges(frame)
        (flat_edges, unwarp_matrix) = flatten_perspective(edges)
        (l_x, l_y) = self.scan_frame_with_windows(flat_edges, self.l_windows)
        self.left.process_points(l_x, l_y)
        (r_x, r_y) = self.scan_frame_with_windows(flat_edges, self.r_windows)
        self.right.process_points(r_x, r_y)

        if draw_statistics:
            edges = get_edges(frame, separate_channels=True)
            debug_overlay = self.draw_debug_overlay(flatten_perspective(edges)[0])
            top_overlay = self.draw_lane_overlay(flatten_perspective(frame)[0])
            debug_overlay = cv2.resize(debug_overlay, (0, 0), fx=0.3, fy=0.3)
            top_overlay = cv2.resize(top_overlay, (0, 0), fx=0.3, fy=0.3)
            frame[:250, :, :] = frame[:250, :, :] * .4
            (h, w, _) = debug_overlay.shape
            frame[20:20 + h, 20:20 + w, :] = debug_overlay
            frame[20:20 + h, 20 + 20 + w:20 + 20 + w + w, :] = top_overlay
            text_x = 20 + 20 + w + w + 20
            self.draw_text(frame, 'Radius of curvature:  {} m'.format(self.radius_of_curvature()), text_x, 80)
            self.draw_text(frame, 'Distance (left):       {:.1f} m'.format(self.left.camera_distance()), text_x, 140)
            self.draw_text(frame, 'Distance (right):      {:.1f} m'.format(self.right.camera_distance()), text_x, 200)

        if draw_lane:
            frame = self.draw_lane_overlay(frame, unwarp_matrix)

        return frame

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def draw_debug_overlay(self, binary, lines=True, windows=True):
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
        if len(binary.shape) == 2:
            image = np.dstack((binary, binary, binary))
        else:
            image = binary
        if windows:
            for window in self.l_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
            for window in self.r_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
        if lines:
            cv2.polylines(image, [self.left.get_points()], False, (1., 0, 0), 2)
            cv2.polylines(image, [self.right.get_points()], False, (1., 0, 0), 2)
        return image * 255

    def draw_lane_overlay(self, image, unwarp_matrix=None):
        """
        Draws an overlay with tracked lane applying perspective unwarp to project it on the original frame.

        Parameters
        ----------
        image           : Original frame.
        unwarp_matrix   : Transformation matrix to unwarp the bird's eye view to initial frame. Defaults to `None` (in
        which case no unwarping is applied).

        Returns
        -------
        Frame with a lane overlay.
        """
        # Create an image to draw the lines on
        overlay = np.zeros_like(image).astype(np.uint8)
        points = np.vstack((self.left.get_points(), np.flipud(self.right.get_points())))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        if unwarp_matrix is not None:
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            overlay = cv2.warpPerspective(overlay, unwarp_matrix, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(image, 1, overlay, 0.3, 0)

    def radius_of_curvature(self):
        """
        Calculates radius of the lane curvature by averaging curvature of the edge lines.

        Returns
        -------
        Radius of the lane curvature in meters.
        """
        return int(np.average([self.left.radius_of_curvature(), self.right.radius_of_curvature()]))