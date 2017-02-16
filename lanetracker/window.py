import numpy as np


class Window(object):
    """
    Represents a scanning window used to detect points likely to represent lane edge lines.
    """

    def __init__(self, y1, y2, x, m=100, tolerance=50):
        """
        Initialises a window object.

        Parameters
        ----------
        y1          : Top y axis coordinate of the window rect.
        y2          : Bottom y axis coordinate of the window rect.
        x           : X coordinate of the center of the window rect
        m           : X axis span, e.g. window rect width would be m*2..
        tolerance   : Min number of pixels we need to detect within a window in order to adjust its x coordinate.
        """
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.m = m
        self.tolerance = tolerance

    def pixels_in(self, nonzero):
        """
        Returns indices of the pixels in `nonzero` that are located within this window.

        Notes
        -----
        Since this looks a bit tricky, I will go into a bit more detail. `nonzero` contains two arrays of coordinates
        non-zero pixels. Say, there were 50 non-zero pixels in the image and `nonzero` would contain two arrays of
        shape (50, ) with x and y coordinates of those pixels respectively. What we return here is a array of indices
        within those 50 that are located inside this window. Basically the result would be a 1-dimensional array of
        ints in the [0, 49] range with a size of less than 50.

        Parameters
        ----------
        nonzero : Coordinates of the non-zero pixels in the image.

        Returns
        -------
        Array of indices of the pixels within this window.
        """
        win_indices = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) &
            (nonzero[1] >= self.x - self.m) & (nonzero[1] < self.x + self.m)
        ).nonzero()[0]
        if len(win_indices) > self.tolerance:
            self.x = np.int(np.mean(nonzero[1][win_indices]))
        return win_indices

    def coordinates(self):
        return ((self.x - self.m, self.y1), (self.x + self.m, self.y2))