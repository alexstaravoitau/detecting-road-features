import numpy as np


class Window(object):
    def __init__(self, y1, y2, x, m=100, tolerance=50):
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.m = m
        self.tolerance = tolerance

    def pixels_in(self, nonzero):
        win_indices = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) & (nonzero[1] >= self.x - self.m) & (
            nonzero[1] < self.x + self.m)
        ).nonzero()[0]
        if len(win_indices) > self.tolerance:
            self.x = np.int(np.mean(nonzero[1][win_indices]))
        return win_indices

    def coordinates(self):
        return ((self.x - self.m, self.y1), (self.x + self.m, self.y2))