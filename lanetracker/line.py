import numpy as np


class Line(object):
    def __init__(self, x, y, h, w):
        # polynomial coefficients for the most recent fit
        self.h = h
        self.w = w
        self.current_fit = self.fit(x, y)
        self.points = self.get_points()

    def get_points(self):
        y = np.linspace(0, self.h - 1, self.h)
        return np.stack((
            self.current_fit[0] * y ** 2 + self.current_fit[1] * y + self.current_fit[2],
            y
        )).astype(np.int).T

    def fit(self, x, y):
        return np.polyfit(y, x, 2)

    def radius_of_curvature(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 24 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        y = self.points[:, 1]
        x = self.points[:, 0]
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        return ((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    def camera_distance(self):
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = self.points[np.max(self.points[:, 1])][0]
        return np.abs((self.w // 2 - x) * xm_per_pix)