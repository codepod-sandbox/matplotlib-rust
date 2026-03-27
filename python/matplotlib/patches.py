"""matplotlib.patches --- Patch, Rectangle, and Circle artists."""

from matplotlib.artist import Artist
from matplotlib.colors import to_rgba, to_hex


def _rounded_rect_points(x0, y0, x1, y1, radius):
    """Generate polygon points approximating a rounded rectangle.

    Parameters
    ----------
    x0, y0 : float  Top-left corner (display coords, y0 < y1).
    x1, y1 : float  Bottom-right corner.
    radius : float  Corner radius in pixels.

    Returns
    -------
    list of (x, y) tuples
    """
    import math
    w = x1 - x0
    h = y1 - y0
    r = min(radius, w / 2, h / 2)
    pts = []
    # 4 corners, 8 arc points each (quarter circle)
    corners = [
        (x0 + r, y0 + r, math.pi,       1.5 * math.pi),  # top-left
        (x1 - r, y0 + r, 1.5 * math.pi, 2 * math.pi),    # top-right
        (x1 - r, y1 - r, 0,              0.5 * math.pi),  # bottom-right
        (x0 + r, y1 - r, 0.5 * math.pi, math.pi),        # bottom-left
    ]
    n_arc = 8
    for cx, cy, t_start, t_end in corners:
        for i in range(n_arc + 1):
            t = t_start + (t_end - t_start) * i / n_arc
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


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

    def set_color(self, color):
        """Set both facecolor and edgecolor to *color*."""
        self.set_facecolor(color)
        self.set_edgecolor(color)

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

    # --- hatch ---
    def set_hatch(self, hatch):
        self._hatch = hatch

    def get_hatch(self):
        return getattr(self, '_hatch', None)

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

    def get_bbox(self):
        from .transforms import Bbox
        x, y = self._xy
        return Bbox([[x, y], [x + self._width, y + self._height]])

    def get_path(self):
        """Return a path-like object representing the rectangle boundary."""
        x0, y0 = self._xy
        x1, y1 = x0 + self._width, y0 + self._height
        return _PolygonPath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)

    def get_patch_transform(self):
        """Return identity transform."""
        return _IdentityTransform()

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

    def get_path(self):
        """Return a path-like object for this polygon."""
        return _PolygonPath(self._xy, self._closed)

    def get_patch_transform(self):
        """Return identity transform (no-op for compatibility)."""
        return _IdentityTransform()

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

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        self._center = tuple(val)

    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = width

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        self._width = val

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = height

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, val):
        self._height = val

    def get_angle(self):
        return self._angle

    def set_angle(self, angle):
        self._angle = angle

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, val):
        self._angle = val

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        cx = layout.sx(self._center[0])
        cy = layout.sy(self._center[1])
        rx = abs(layout.sx(self._center[0] + self._width / 2) - cx)
        ry = abs(layout.sy(self._center[1] + self._height / 2) - cy)
        if rx <= 0 or ry <= 0:
            return
        fc = self._resolved_facecolor_hex()
        ec = self._resolved_edgecolor_hex()
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        renderer.draw_ellipse(cx, cy, rx, ry, self._angle,
                              fc if fc != 'none' else None,
                              ec if ec != 'none' else None,
                              alpha)


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

    @property
    def theta1(self):
        return self._theta1

    @theta1.setter
    def theta1(self, val):
        self._theta1 = val

    def get_theta2(self):
        return self._theta2

    def set_theta2(self, theta2):
        self._theta2 = theta2

    @property
    def theta2(self):
        return self._theta2

    @theta2.setter
    def theta2(self, val):
        self._theta2 = val

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        import math
        cx = layout.sx(self._center[0])
        cy = layout.sy(self._center[1])
        rx = abs(layout.sx(self._center[0] + self._width / 2) - cx)
        ry = abs(layout.sy(self._center[1] + self._height / 2) - cy)
        if rx <= 0 or ry <= 0:
            return
        t1 = math.radians(self._theta1)
        t2 = math.radians(self._theta2)
        sweep = abs(t2 - t1)
        n = max(32, int(sweep / (2 * math.pi) * 64))
        xdata = []
        ydata = []
        for i in range(n + 1):
            t = t1 + (t2 - t1) * i / n
            xdata.append(cx + rx * math.cos(t))
            ydata.append(cy - ry * math.sin(t))  # negate: screen y-down
        ec = self._resolved_edgecolor_hex()
        lw = self._linewidth if self._linewidth is not None else 1.0
        renderer.draw_line(xdata, ydata, ec if ec != 'none' else 'black', lw, '-', 1.0)


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

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        # Map data coords to display coords
        x0 = layout.sx(self._xy[0])
        y0 = layout.sy(self._xy[1])
        x1 = layout.sx(self._xy[0] + self._width)
        y1 = layout.sy(self._xy[1] + self._height)
        # Normalise: ensure top < bottom, left < right in screen coords
        x_left, x_right = min(x0, x1), max(x0, x1)
        y_top, y_bot = min(y0, y1), max(y0, y1)
        fc = self._resolved_facecolor_hex()
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0

        if 'round' in str(self._boxstyle).lower():
            pts = _rounded_rect_points(x_left, y_top, x_right, y_bot, radius=8)
            renderer.draw_polygon(pts, fc if fc != 'none' else '#ffffff', alpha)
        else:
            w = x_right - x_left
            h = y_bot - y_top
            renderer.draw_rect(x_left, y_top, w, h,
                               stroke=None, fill=fc if fc != 'none' else None)


class FancyArrowPatch(Patch):
    """A fancy arrow patch."""

    zorder = 2

    def __init__(self, posA=None, posB=None, path=None,
                 arrowstyle='->', connectionstyle=None,
                 patchA=None, patchB=None,
                 shrinkA=2, shrinkB=2,
                 mutation_scale=1, mutation_aspect=None,
                 color='black', linewidth=1.5,
                 **kwargs):
        self._posA = tuple(posA) if posA is not None else posA
        self._posB = tuple(posB) if posB is not None else posB
        self._path = path
        self._arrowstyle = arrowstyle
        self._connectionstyle = connectionstyle
        self._patchA = patchA
        self._patchB = patchB
        self._shrinkA = shrinkA
        self._shrinkB = shrinkB
        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect
        self._color = color
        self._linewidth = linewidth
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

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        if self._posA is None or self._posB is None:
            return
        import math
        x1, y1 = layout.sx(self._posA[0]), layout.sy(self._posA[1])
        x2, y2 = layout.sx(self._posB[0]), layout.sy(self._posB[1])

        # Apply shrink
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length > 1e-6 and (self._shrinkA or self._shrinkB):
            ux, uy = dx / length, dy / length
            x1 += ux * self._shrinkA
            y1 += uy * self._shrinkA
            x2 -= ux * self._shrinkB
            y2 -= uy * self._shrinkB

        color = to_hex(self._color)
        renderer.draw_arrow(x1, y1, x2, y2,
                            self._arrowstyle, color, self._linewidth)


class Arrow(Patch):
    """A simple arrow patch."""

    def __init__(self, x, y, dx, dy, width=1.0, **kwargs):
        self._x = x
        self._y = y
        self._dx = dx
        self._dy = dy
        self._arrow_width = width
        super().__init__(**kwargs)

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        import math
        x0 = layout.sx(self._x)
        y0 = layout.sy(self._y)
        x1 = layout.sx(self._x + self._dx)
        y1 = layout.sy(self._y + self._dy)
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            return
        # Unit vector along arrow direction
        ux = (x1 - x0) / length
        uy = (y1 - y0) / length
        # Perpendicular unit vector
        px = -uy
        py = ux
        # Arrow dimensions in display pixels
        shaft_half_w = self._arrow_width / 2 * length * 0.15
        head_half_w = shaft_half_w * 3.0
        head_len = min(length * 0.35, head_half_w * 2.5)
        shaft_end = length - head_len
        # 7-point polygon: shaft rectangle + arrowhead triangle
        pts = [
            (x0 + px * shaft_half_w,                   y0 + py * shaft_half_w),
            (x0 + ux * shaft_end + px * shaft_half_w,  y0 + uy * shaft_end + py * shaft_half_w),
            (x0 + ux * shaft_end + px * head_half_w,   y0 + uy * shaft_end + py * head_half_w),
            (x1,                                        y1),
            (x0 + ux * shaft_end - px * head_half_w,   y0 + uy * shaft_end - py * head_half_w),
            (x0 + ux * shaft_end - px * shaft_half_w,  y0 + uy * shaft_end - py * shaft_half_w),
            (x0 - px * shaft_half_w,                   y0 - py * shaft_half_w),
        ]
        fc = self._resolved_facecolor_hex()
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        renderer.draw_polygon(pts, fc if fc != 'none' else '#000000', alpha)


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

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        import math
        cx = layout.sx(self._xy_center[0])
        cy = layout.sy(self._xy_center[1])
        # Map radius to display pixels (use x-axis scale)
        r_px = abs(layout.sx(self._xy_center[0] + self._radius) - cx)
        if r_px <= 0:
            return
        n = self._numVertices
        pts = []
        for i in range(n):
            angle = self._orientation + 2 * math.pi * i / n
            px = cx + r_px * math.cos(angle)
            py = cy - r_px * math.sin(angle)  # negate: screen y-down
            pts.append((px, py))
        fc = self._resolved_facecolor_hex()
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        renderer.draw_polygon(pts, fc if fc != 'none' else '#ffffff', alpha)


class PathPatch(Patch):
    """A patch defined by a path."""

    def __init__(self, path, **kwargs):
        self._path = path
        super().__init__(**kwargs)

    def get_path(self):
        return self._path

    def set_path(self, path):
        self._path = path

    def draw(self, renderer, layout):
        if not self.get_visible() or self._path is None:
            return
        # Path code constants (match matplotlib.path.Path)
        MOVETO = 1
        LINETO = 2
        CURVE3 = 3
        CURVE4 = 4
        CLOSEPOLY = 79

        codes = self._path.codes
        verts = self._path.vertices
        fc = self._resolved_facecolor_hex()
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        fc_draw = fc if fc != 'none' else '#ffffff'

        current = []
        for code, vertex in zip(codes, verts):
            x, y = vertex
            sx = layout.sx(x)
            sy = layout.sy(y)
            if code == MOVETO:
                if len(current) >= 3:
                    renderer.draw_polygon(current, fc_draw, alpha)
                current = [(sx, sy)]
            elif code == LINETO:
                current.append((sx, sy))
            elif code == CLOSEPOLY:
                if len(current) >= 3:
                    renderer.draw_polygon(current, fc_draw, alpha)
                current = []
            elif code in (CURVE3, CURVE4):
                # Simple linearisation: treat control points as vertices
                current.append((sx, sy))
        if len(current) >= 3:
            renderer.draw_polygon(current, fc_draw, alpha)


class FancyArrow(Patch):
    """A fancy arrow patch used by ax.arrow()."""

    _valid_shapes = ('full', 'left', 'right')

    def __init__(self, x, y, dx, dy, width=0.001, length_includes_head=False,
                 head_width=None, head_length=None, shape='full', overhang=0,
                 head_starts_at_zero=False, **kwargs):
        if shape not in self._valid_shapes:
            raise ValueError(
                f"shape must be one of {self._valid_shapes!r}, got {shape!r}")
        self._x = x
        self._y = y
        self._dx = dx
        self._dy = dy
        self._arrow_width = width
        self._length_includes_head = length_includes_head
        self._head_width = head_width if head_width is not None else 3 * width
        self._head_length = head_length if head_length is not None else 1.5 * self._head_width
        self._shape = shape
        self._overhang = overhang
        self._head_starts_at_zero = head_starts_at_zero
        super().__init__(**kwargs)
        # Build the polygon vertices representing the arrow
        self._verts = self._compute_verts()

    def _compute_verts(self):
        """Compute arrow polygon vertices in data coordinates."""
        import math
        length = math.hypot(self._dx, self._dy)
        if length < 1e-10:
            return [(self._x, self._y)]
        ux, uy = self._dx / length, self._dy / length
        px, py = -uy, ux
        hw = self._head_width / 2
        hl = min(self._head_length, length)
        sw = self._arrow_width / 2
        x0, y0 = self._x, self._y
        shaft_end_x = x0 + ux * (length - hl)
        shaft_end_y = y0 + uy * (length - hl)
        tip_x, tip_y = x0 + ux * length, y0 + uy * length
        return [
            (x0 + px * sw, y0 + py * sw),
            (shaft_end_x + px * sw, shaft_end_y + py * sw),
            (shaft_end_x + px * hw, shaft_end_y + py * hw),
            (tip_x, tip_y),
            (shaft_end_x - px * hw, shaft_end_y - py * hw),
            (shaft_end_x - px * sw, shaft_end_y - py * sw),
            (x0 - px * sw, y0 - py * sw),
        ]

    def get_xy(self):
        """Return the arrow polygon vertices as a list of (x, y) tuples."""
        return list(self._verts)

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        import math
        x0 = layout.sx(self._x)
        y0 = layout.sy(self._y)
        x1 = layout.sx(self._x + self._dx)
        y1 = layout.sy(self._y + self._dy)
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            return
        ux = (x1 - x0) / length
        uy = (y1 - y0) / length
        px = -uy
        py = ux
        shaft_half_w = self._arrow_width / 2 * length * 0.15
        head_half_w = shaft_half_w * 3.0
        head_len = min(length * 0.35, head_half_w * 2.5)
        shaft_end = length - head_len
        pts = [
            (x0 + px * shaft_half_w,                   y0 + py * shaft_half_w),
            (x0 + ux * shaft_end + px * shaft_half_w,  y0 + uy * shaft_end + py * shaft_half_w),
            (x0 + ux * shaft_end + px * head_half_w,   y0 + uy * shaft_end + py * head_half_w),
            (x1,                                        y1),
            (x0 + ux * shaft_end - px * head_half_w,   y0 + uy * shaft_end - py * head_half_w),
            (x0 + ux * shaft_end - px * shaft_half_w,  y0 + uy * shaft_end - py * shaft_half_w),
            (x0 - px * shaft_half_w,                   y0 - py * shaft_half_w),
        ]
        fc = self._resolved_facecolor_hex()
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        renderer.draw_polygon(pts, fc if fc != 'none' else '#000000', alpha)


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

    @property
    def theta1(self):
        return self._theta1

    @theta1.setter
    def theta1(self, val):
        self._theta1 = val

    def get_theta2(self):
        return self._theta2

    def set_theta2(self, theta2):
        self._theta2 = theta2

    @property
    def theta2(self):
        return self._theta2

    @theta2.setter
    def theta2(self, val):
        self._theta2 = val

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


class _PolygonPath:
    """Lightweight path-like object for Polygon.get_path()."""

    def __init__(self, xy, closed=True):
        self.vertices = list(xy)
        self.closed = closed

    def contains_point(self, point, radius=0.0):
        """Ray casting algorithm to test if *point* is inside the polygon."""
        x, y = point
        verts = self.vertices
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
                inside = not inside
            j = i
        return inside


class _IdentityTransform:
    """Minimal identity transform for compatibility."""

    def transform(self, points):
        return points

    def transform_point(self, point):
        return point

    def transform_path(self, path):
        """Return path unchanged (identity transform)."""
        return path
