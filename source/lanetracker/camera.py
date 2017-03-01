import numpy as np
import cv2
import matplotlib.image as mpimg


class CameraCalibration(object):
    """
    Prepares camera calibration pipeline based on a set of calibration images.
    """

    def __init__(self, calibration_images, pattern_size=(9, 6), retain_calibration_images=False):
        """
        Initialises camera calibration pipeline based on a set of calibration images.

        Parameters
        ----------
        calibration_images          : Calibration images.
        pattern_size                : Shape of the calibration pattern.
        retain_calibration_images   : Flag indicating if we need to preserve calibration images.
        """
        self.camera_matrix = None
        self.dist_coefficients = None
        self.calibration_images_success = []
        self.calibration_images_error = []
        self.calculate_calibration(calibration_images, pattern_size, retain_calibration_images)

    def __call__(self, image):
        """
        Calibrates an image based on saved settings.

        Parameters
        ----------
        image       : Image to calibrate.

        Returns
        -------
        Calibrated image.
        """
        if self.camera_matrix is not None and self.dist_coefficients is not None:
            return cv2.undistort(image, self.camera_matrix, self.dist_coefficients, None, self.camera_matrix)
        else:
            return image

    def calculate_calibration(self, images, pattern_size, retain_calibration_images):
        """
        Prepares calibration settings.

        Parameters
        ----------
        images                      : Set of calibration images.
        pattern_size                : Calibration pattern shape.
        retain_calibration_images   : Flag indicating if we need to preserve calibration images.
        """
        # Prepare object points: (0,0,0), (1,0,0), (2,0,0), ...
        pattern = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        pattern[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        pattern_points = []  # 3d points in real world space
        image_points = []  # 2d points in image plane.
        image_size = None

        # Step through the list and search for chessboard corners
        for i, path in enumerate(images):
            image = mpimg.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            # If found, add object points and image points
            if found:
                pattern_points.append(pattern)
                image_points.append(corners)
                image_size = (image.shape[1], image.shape[0])
                if retain_calibration_images:
                    cv2.drawChessboardCorners(image, pattern_size, corners, True)
                    self.calibration_images_success.append(image)
            else:
                if retain_calibration_images:
                    self.calibration_images_error.append(image)

        if pattern_points and image_points:
            _, self.camera_matrix, self.dist_coefficients, _, _ = cv2.calibrateCamera(
                pattern_points, image_points, image_size, None, None
            )