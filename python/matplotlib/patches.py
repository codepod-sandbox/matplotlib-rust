"""matplotlib.patches --- Patch, Rectangle, and Circle artists."""

from matplotlib.artist import Artist
from matplotlib.colors import to_rgba, to_hex


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

    # --- linestyle ---
    def get_linestyle(self):
        return getattr(self, '_linestyle', 'solid')

    def set_linestyle(self, ls):
        self._linestyle = ls

    # --- antialiased ---
    def get_antialiased(self):
        return getattr(self, '_antialiased', True)

    def set_antialiased(self, aa):
        self._antialiased = aa

    def _resolved_facecolor_hex(self):
        fc = self._facecolor
        if isinstance(fc, str) and fc.lower() == 'none':
            return 'none'
        return to_hex(fc)

    def _resolved_edgecolor_hex(self):
        ec = self._edgecolor
        if isinstance(ec, str) and ec.lower() == 'none':
            return 'none'
        return to_hex(ec)


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

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        x0, y0 = self._xy
        x1 = x0 + self._width
        y1 = y0 + self._height
        px_left = layout.sx(x0)
        px_right = layout.sx(x1)
        px_top = layout.sy(y1)      # top of data = smaller pixel y
        px_bottom = layout.sy(y0)   # bottom of data = larger pixel y
        pw = px_right - px_left
        ph = px_bottom - px_top
        if pw <= 0 or ph <= 0:
            return
        renderer.draw_rect(px_left, px_top, pw, ph,
                           self._resolved_edgecolor_hex(),
                           self._resolved_facecolor_hex())


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

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        cx_px = layout.sx(self._center[0])
        cy_px = layout.sy(self._center[1])
        r_px = abs(layout.sx(self._center[0] + self._radius) - cx_px)
        if r_px <= 0:
            return
        renderer.draw_circle(cx_px, cy_px, r_px,
                             self._resolved_facecolor_hex())


class Polygon(Patch):
    """A polygon defined by a list of (x, y) vertices."""

    def __init__(self, xy, closed=True, **kwargs):
        self._xy = [tuple(pt) for pt in xy]
        self._closed = closed
        super().__init__(**kwargs)

    def get_xy(self):
        return list(self._xy)

    def set_xy(self, xy):
        self._xy = [tuple(pt) for pt in xy]

    def get_closed(self):
        return self._closed

    def set_closed(self, closed):
        self._closed = closed

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        if len(self._xy) < 3:
            return
        x_px = [layout.sx(pt[0]) for pt in self._xy]
        y_px = [layout.sy(pt[1]) for pt in self._xy]
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        points = list(zip(x_px, y_px))
        renderer.draw_polygon(points,
                              self._resolved_facecolor_hex(),
                              alpha)


class Ellipse(Patch):
    """An ellipse defined by center, width, height, and angle."""

    def __init__(self, xy, width, height, angle=0, **kwargs):
        self._center = tuple(xy)
        self._width = width
        self._height = height
        self._angle = angle
        super().__init__(**kwargs)

    def get_center(self):
        return self._center

    def set_center(self, center):
        self._center = tuple(center)

    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = width

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = height

    def get_angle(self):
        return self._angle

    def set_angle(self, angle):
        self._angle = angle


class Arc(Ellipse):
    """An elliptical arc, a segment of an ellipse."""

    def __init__(self, xy, width, height, angle=0, theta1=0.0, theta2=360.0,
                 **kwargs):
        super().__init__(xy, width, height, angle=angle, **kwargs)
        self._theta1 = theta1
        self._theta2 = theta2

    def get_theta1(self):
        return self._theta1

    def set_theta1(self, theta1):
        self._theta1 = theta1

    def get_theta2(self):
        return self._theta2

    def set_theta2(self, theta2):
        self._theta2 = theta2


class FancyBboxPatch(Patch):
    """A patch with a fancy bounding box."""

    def __init__(self, xy, width, height, boxstyle='round', **kwargs):
        self._xy = tuple(xy)
        self._width = width
        self._height = height
        self._boxstyle = boxstyle
        super().__init__(**kwargs)

    def get_x(self):
        return self._xy[0]

    def get_y(self):
        return self._xy[1]

    def get_xy(self):
        return self._xy

    def set_xy(self, xy):
        self._xy = tuple(xy)

    def get_width(self):
        return self._width

    def set_width(self, w):
        self._width = w

    def get_height(self):
        return self._height

    def set_height(self, h):
        self._height = h

    def get_boxstyle(self):
        return self._boxstyle

    def set_boxstyle(self, boxstyle):
        self._boxstyle = boxstyle


class FancyArrowPatch(Patch):
    """A fancy arrow patch."""

    def __init__(self, posA=None, posB=None, path=None,
                 arrowstyle='->', connectionstyle=None,
                 patchA=None, patchB=None,
                 shrinkA=2, shrinkB=2,
                 mutation_scale=1, mutation_aspect=None,
                 **kwargs):
        self._posA = posA
        self._posB = posB
        self._path = path
        self._arrowstyle = arrowstyle
        self._connectionstyle = connectionstyle
        self._patchA = patchA
        self._patchB = patchB
        self._shrinkA = shrinkA
        self._shrinkB = shrinkB
        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect
        super().__init__(**kwargs)

    def get_arrowstyle(self):
        return self._arrowstyle

    def set_arrowstyle(self, arrowstyle):
        self._arrowstyle = arrowstyle

    def get_connectionstyle(self):
        return self._connectionstyle

    def set_connectionstyle(self, connectionstyle):
        self._connectionstyle = connectionstyle

    def get_mutation_scale(self):
        return self._mutation_scale

    def set_mutation_scale(self, scale):
        self._mutation_scale = scale

    def set_positions(self, posA, posB):
        """Set the begin and end positions of the connecting path."""
        self._posA = posA
        self._posB = posB


class Arrow(Patch):
    """A simple arrow patch."""

    def __init__(self, x, y, dx, dy, width=1.0, **kwargs):
        self._x = x
        self._y = y
        self._dx = dx
        self._dy = dy
        self._arrow_width = width
        super().__init__(**kwargs)


class RegularPolygon(Patch):
    """A regular polygon with *numVertices* vertices, centered at *xy*."""

    def __init__(self, xy, numVertices, radius=5, orientation=0, **kwargs):
        self._xy_center = tuple(xy)
        self._numVertices = numVertices
        self._radius = radius
        self._orientation = orientation
        super().__init__(**kwargs)

    @property
    def numvertices(self):
        return self._numVertices

    @property
    def xy(self):
        return self._xy_center

    @property
    def orientation(self):
        return self._orientation


class PathPatch(Patch):
    """A patch defined by a path."""

    def __init__(self, path, **kwargs):
        self._path = path
        super().__init__(**kwargs)

    def get_path(self):
        return self._path

    def set_path(self, path):
        self._path = path


class ConnectionPatch(FancyArrowPatch):
    """A patch connecting two points, possibly in different Axes."""

    def __init__(self, xyA, xyB, coordsA, coordsB=None,
                 axesA=None, axesB=None, **kwargs):
        self._xyA = xyA
        self._xyB = xyB
        self._coordsA = coordsA
        self._coordsB = coordsB or coordsA
        super().__init__(posA=xyA, posB=xyB, **kwargs)


class Wedge(Patch):
    """A wedge (pie slice) defined by center, radius, and two angles."""

    def __init__(self, center, r, theta1, theta2, **kwargs):
        self._center = tuple(center)
        self._r = r
        self._theta1 = theta1  # start angle in degrees
        self._theta2 = theta2  # end angle in degrees
        super().__init__(**kwargs)

    def get_center(self):
        return self._center

    def set_center(self, center):
        self._center = tuple(center)

    def get_r(self):
        return self._r

    def set_r(self, r):
        self._r = r

    def set_radius(self, radius):
        self._r = radius

    def get_theta1(self):
        return self._theta1

    def set_theta1(self, theta1):
        self._theta1 = theta1

    def get_theta2(self):
        return self._theta2

    def set_theta2(self, theta2):
        self._theta2 = theta2

    def set_width(self, width):
        """Set the angular width (theta2 - theta1)."""
        self._theta2 = self._theta1 + width

    def get_width(self):
        """Get the angular width (theta2 - theta1)."""
        return self._theta2 - self._theta1

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        cx_px = layout.sx(self._center[0])
        cy_px = layout.sy(self._center[1])
        edge_px = layout.sx(self._center[0] + self._r)
        r_px = abs(edge_px - cx_px)
        if r_px <= 0:
            return
        renderer.draw_wedge(cx_px, cy_px, r_px,
                            self._theta1, self._theta2,
                            self._resolved_facecolor_hex())
