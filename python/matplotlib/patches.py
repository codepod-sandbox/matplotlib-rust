"""matplotlib.patches --- Patch, Rectangle, and Circle artists."""

from matplotlib.artist import Artist
from matplotlib.colors import to_rgba


class Patch(Artist):
    """Base class for 2-D filled shapes."""

    zorder = 1

    def __init__(self, facecolor=None, edgecolor=None, linewidth=None,
                 **kwargs):
        super().__init__()
        self._facecolor = facecolor if facecolor is not None else 'C0'
        self._edgecolor = edgecolor if edgecolor is not None else 'black'
        self._linewidth = linewidth if linewidth is not None else 1.0

        # Apply any remaining kwargs through the batch setter
        if kwargs:
            self.set(**kwargs)

    # --- facecolor ---
    def get_facecolor(self):
        """Return facecolor as an RGBA tuple, applying artist alpha."""
        fc = self._facecolor
        if isinstance(fc, str) and fc.lower() == 'none':
            return (0.0, 0.0, 0.0, 0.0)
        rgba = to_rgba(fc)
        if self._alpha is not None:
            rgba = (rgba[0], rgba[1], rgba[2], self._alpha)
        return rgba

    def set_facecolor(self, color):
        self._facecolor = color

    # --- edgecolor ---
    def get_edgecolor(self):
        """Return edgecolor as an RGBA tuple, applying artist alpha.

        If edgecolor is 'none', returns (0, 0, 0, 0) without applying alpha.
        """
        ec = self._edgecolor
        if isinstance(ec, str) and ec.lower() == 'none':
            return (0.0, 0.0, 0.0, 0.0)
        rgba = to_rgba(ec)
        if self._alpha is not None:
            rgba = (rgba[0], rgba[1], rgba[2], self._alpha)
        return rgba

    def set_edgecolor(self, color):
        self._edgecolor = color

    # --- linewidth ---
    def get_linewidth(self):
        return self._linewidth

    def set_linewidth(self, w):
        self._linewidth = w


class Rectangle(Patch):
    """A rectangle defined by an anchor point, width, and height."""

    def __init__(self, xy, width, height, **kwargs):
        self._xy = tuple(xy)
        self._width = width
        self._height = height
        super().__init__(**kwargs)

    def get_x(self):
        return self._xy[0]

    def get_y(self):
        return self._xy[1]

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_xy(self):
        return self._xy

    def set_xy(self, xy):
        self._xy = tuple(xy)

    def set_width(self, w):
        self._width = w

    def set_height(self, h):
        self._height = h

    def get_corners(self):
        """Return the four corners as a list of (x, y) tuples.

        Order: bottom-left, bottom-right, top-right, top-left.
        """
        x0, y0 = self._xy
        x1, y1 = x0 + self._width, y0 + self._height
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


class Circle(Patch):
    """A circle defined by a center point and radius."""

    def __init__(self, center=(0.0, 0.0), radius=0.5, **kwargs):
        self._center = tuple(center)
        self._radius = radius
        super().__init__(**kwargs)

    def get_center(self):
        return self._center

    def set_center(self, center):
        self._center = tuple(center)

    def get_radius(self):
        return self._radius

    def set_radius(self, radius):
        self._radius = radius
