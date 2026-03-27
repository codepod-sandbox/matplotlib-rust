"""
matplotlib.transforms --- Bbox, Affine2D, and transform classes.

Provides minimal transform classes needed by many upstream tests.
"""

import math


class BboxBase:
    """Base class for bounding boxes."""

    def __init__(self):
        pass

    @property
    def bounds(self):
        return (self.x0, self.y0, self.width, self.height)

    @property
    def extents(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def min(self):
        return (self.x0, self.y0)

    @property
    def max(self):
        return (self.x1, self.y1)

    @property
    def intervalx(self):
        return (self.x0, self.x1)

    @property
    def intervaly(self):
        return (self.y0, self.y1)

    @property
    def p0(self):
        return (self.x0, self.y0)

    @property
    def p1(self):
        return (self.x1, self.y1)

    @property
    def size(self):
        return (self.width, self.height)

    def contains(self, x, y):
        """Return whether (x, y) is inside the bbox."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def containsx(self, x):
        return self.x0 <= x <= self.x1

    def containsy(self, y):
        return self.y0 <= y <= self.y1

    def overlaps(self, other):
        """Return whether this bbox overlaps *other*."""
        return (self.x0 < other.x1 and self.x1 > other.x0 and
                self.y0 < other.y1 and self.y1 > other.y0)

    def fully_containsx(self, x):
        return self.x0 < x < self.x1

    def fully_containsy(self, y):
        return self.y0 < y < self.y1

    def fully_contains(self, x, y):
        return self.fully_containsx(x) and self.fully_containsy(y)

    def frozen(self):
        """Return a frozen copy."""
        return Bbox.from_bounds(self.x0, self.y0, self.width, self.height)

    def __repr__(self):
        return (f"Bbox([[{self.x0}, {self.y0}], "
                f"[{self.x1}, {self.y1}]])")

    def __iter__(self):
        return iter(self.bounds)

    def __eq__(self, other):
        if not isinstance(other, BboxBase):
            return NotImplemented
        return (self.x0 == other.x0 and self.y0 == other.y0 and
                self.x1 == other.x1 and self.y1 == other.y1)


class Bbox(BboxBase):
    """A mutable bounding box.

    Parameters
    ----------
    points : array-like of shape (2, 2)
        [[x0, y0], [x1, y1]]
    """

    def __init__(self, points=None):
        super().__init__()
        if points is not None:
            self.x0 = float(points[0][0])
            self.y0 = float(points[0][1])
            self.x1 = float(points[1][0])
            self.y1 = float(points[1][1])
        else:
            self.x0 = 0.0
            self.y0 = 0.0
            self.x1 = 1.0
            self.y1 = 1.0

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @staticmethod
    def from_bounds(x0, y0, width, height):
        """Create a Bbox from (x0, y0, width, height)."""
        return Bbox([[x0, y0], [x0 + width, y0 + height]])

    @staticmethod
    def from_extents(x0, y0, x1, y1):
        """Create a Bbox from (x0, y0, x1, y1)."""
        return Bbox([[x0, y0], [x1, y1]])

    @staticmethod
    def unit():
        """Return the unit Bbox from (0, 0) to (1, 1)."""
        return Bbox([[0, 0], [1, 1]])

    @staticmethod
    def null():
        """Return a null (inverted, empty) Bbox."""
        return Bbox([[float('inf'), float('inf')],
                     [float('-inf'), float('-inf')]])

    def update_from_data_xy(self, xy, ignore=True):
        """Update bbox from data xy points."""
        if not xy:
            return
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        if ignore:
            self.x0 = min(xs)
            self.y0 = min(ys)
            self.x1 = max(xs)
            self.y1 = max(ys)
        else:
            self.x0 = min(self.x0, min(xs))
            self.y0 = min(self.y0, min(ys))
            self.x1 = max(self.x1, max(xs))
            self.y1 = max(self.y1, max(ys))

    def expanded(self, sw, sh):
        """Return a Bbox expanded by factors *sw* and *sh*."""
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        w = self.width * sw / 2
        h = self.height * sh / 2
        return Bbox.from_extents(cx - w, cy - h, cx + w, cy + h)

    def translated(self, tx, ty):
        """Return a Bbox translated by (tx, ty)."""
        return Bbox.from_extents(self.x0 + tx, self.y0 + ty,
                                 self.x1 + tx, self.y1 + ty)

    def padded(self, p):
        """Return a Bbox padded by *p* on all sides."""
        return Bbox.from_extents(self.x0 - p, self.y0 - p,
                                 self.x1 + p, self.y1 + p)

    def shrunk(self, mx, my):
        """Return a Bbox shrunk by factors *mx* and *my*."""
        w = self.width * mx
        h = self.height * my
        return Bbox.from_bounds(self.x0, self.y0, w, h)

    def shrunk_to_aspect(self, box_aspect, container=None, fig_aspect=1.0):
        """Return a Bbox shrunk to have a given aspect ratio."""
        if container is None:
            container = self
        w, h = container.width, container.height
        current_aspect = h / w if w != 0 else 1
        if box_aspect > current_aspect:
            # Height-limited
            new_w = h / box_aspect
            new_h = h
        else:
            new_w = w
            new_h = w * box_aspect
        return Bbox.from_bounds(self.x0, self.y0, new_w, new_h)

    def anchored(self, c, container=None):
        """Return a Bbox anchored to position *c* within *container*."""
        if container is None:
            container = self
        # c can be string like 'C', 'NE', etc. or (cx, cy) tuple
        if isinstance(c, str):
            anchor_map = {
                'C': (0.5, 0.5), 'SW': (0, 0), 'S': (0.5, 0),
                'SE': (1, 0), 'E': (1, 0.5), 'NE': (1, 1),
                'N': (0.5, 1), 'NW': (0, 1), 'W': (0, 0.5),
            }
            cx, cy = anchor_map.get(c, (0.5, 0.5))
        else:
            cx, cy = c
        x0 = container.x0 + cx * (container.width - self.width)
        y0 = container.y0 + cy * (container.height - self.height)
        return Bbox.from_bounds(x0, y0, self.width, self.height)

    @staticmethod
    def union(bboxes):
        """Return a Bbox that contains all *bboxes*."""
        if not bboxes:
            return Bbox.unit()
        x0 = min(b.x0 for b in bboxes)
        y0 = min(b.y0 for b in bboxes)
        x1 = max(b.x1 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)
        return Bbox.from_extents(x0, y0, x1, y1)

    @staticmethod
    def intersection(bbox1, bbox2):
        """Return the intersection of two Bboxes, or None if no overlap."""
        x0 = max(bbox1.x0, bbox2.x0)
        y0 = max(bbox1.y0, bbox2.y0)
        x1 = min(bbox1.x1, bbox2.x1)
        y1 = min(bbox1.y1, bbox2.y1)
        if x0 > x1 or y0 > y1:
            return None
        return Bbox.from_extents(x0, y0, x1, y1)

    def is_unit(self):
        """Return whether this is the unit bbox."""
        return (self.x0 == 0 and self.y0 == 0 and
                self.x1 == 1 and self.y1 == 1)

    def is_empty(self):
        """Return whether this bbox has zero area."""
        return self.width == 0 or self.height == 0

    @property
    def size(self):
        """Return (width, height)."""
        return (self.width, self.height)

    def count_contains(self, vertices):
        """Count how many of *vertices* are inside the bbox."""
        return sum(1 for v in vertices if self.contains(v[0], v[1]))

    def count_overlaps(self, bboxes):
        """Count how many *bboxes* overlap this bbox."""
        return sum(1 for b in bboxes if self.overlaps(b))

    def rotated(self, radians):
        """Return the bounding box of the rotated bbox."""
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        corners = [
            (self.x0, self.y0), (self.x1, self.y0),
            (self.x0, self.y1), (self.x1, self.y1),
        ]
        c = math.cos(radians)
        s = math.sin(radians)
        rotated = []
        for x, y in corners:
            dx, dy = x - cx, y - cy
            rotated.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        return Bbox.from_extents(min(xs), min(ys), max(xs), max(ys))

    def get_points(self):
        """Return [[x0, y0], [x1, y1]]."""
        return [[self.x0, self.y0], [self.x1, self.y1]]

    def set_points(self, points):
        """Set from [[x0, y0], [x1, y1]]."""
        self.x0 = float(points[0][0])
        self.y0 = float(points[0][1])
        self.x1 = float(points[1][0])
        self.y1 = float(points[1][1])


class TransformNode:
    """Base class for transform graph nodes."""

    INVALID_NON_AFFINE = 1
    INVALID_AFFINE = 2
    INVALID = INVALID_NON_AFFINE | INVALID_AFFINE

    is_affine = False
    is_bbox = False
    pass_through = False

    def __init__(self, shorthand_name=None):
        self._shorthand_name = shorthand_name

    def frozen(self):
        return self

    def __add__(self, other):
        """Compose two transforms."""
        return CompositeGenericTransform(self, other)

    def __sub__(self, other):
        return CompositeGenericTransform(other.inverted(), self)


class Transform(TransformNode):
    """Base class for transforms."""

    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = False

    def __init__(self, shorthand_name=None):
        super().__init__(shorthand_name)

    def transform(self, values):
        """Transform a set of values."""
        return values

    def transform_point(self, point):
        """Transform a single point."""
        return point

    def inverted(self):
        """Return the inverse transform."""
        return self

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return True


class IdentityTransform(Transform):
    """The identity transform."""

    is_affine = True
    is_separable = True
    has_inverse = True

    def transform(self, values):
        return values

    def inverted(self):
        return self

    def __repr__(self):
        return "IdentityTransform()"


class Affine2DBase(Transform):
    """Base class for 2D affine transforms."""

    is_affine = True
    is_separable = True
    has_inverse = True

    def get_matrix(self):
        """Return the 3x3 affine matrix."""
        raise NotImplementedError


class Affine2D(Affine2DBase):
    """A mutable 2D affine transform.

    The transform is represented by a 3x3 matrix::

        | a  b  tx |
        | c  d  ty |
        | 0  0  1  |
    """

    def __init__(self, matrix=None):
        super().__init__()
        if matrix is not None:
            self._mtx = [row[:] for row in matrix]
        else:
            # Identity
            self._mtx = [[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]]

    def get_matrix(self):
        return [row[:] for row in self._mtx]

    def set(self, other):
        """Copy matrix from *other*."""
        if isinstance(other, Affine2D):
            self._mtx = [row[:] for row in other._mtx]
        return self

    def clear(self):
        """Reset to identity."""
        self._mtx = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
        return self

    def rotate(self, theta):
        """Add a rotation (in radians) to this transform."""
        c = math.cos(theta)
        s = math.sin(theta)
        self._mtx = _mmul([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                           self._mtx)
        return self

    def rotate_deg(self, degrees):
        """Add a rotation (in degrees)."""
        return self.rotate(math.radians(degrees))

    def rotate_around(self, x, y, theta):
        """Rotate around point (x, y) by *theta* radians."""
        return self.translate(-x, -y).rotate(theta).translate(x, y)

    def rotate_deg_around(self, x, y, degrees):
        """Rotate around point (x, y) by *degrees*."""
        return self.rotate_around(x, y, math.radians(degrees))

    def translate(self, tx, ty):
        """Add a translation."""
        self._mtx = _mmul([[1, 0, tx], [0, 1, ty], [0, 0, 1]],
                           self._mtx)
        return self

    def scale(self, sx, sy=None):
        """Add a scaling."""
        if sy is None:
            sy = sx
        self._mtx = _mmul([[sx, 0, 0], [0, sy, 0], [0, 0, 1]],
                           self._mtx)
        return self

    def skew(self, xShear, yShear):
        """Add a skew (in radians)."""
        self._mtx = _mmul([[1, math.tan(xShear), 0],
                            [math.tan(yShear), 1, 0],
                            [0, 0, 1]],
                           self._mtx)
        return self

    def skew_deg(self, xShear, yShear):
        """Add a skew (in degrees)."""
        return self.skew(math.radians(xShear), math.radians(yShear))

    def transform(self, values):
        """Transform points."""
        if not values:
            return values
        # Handle single point
        if isinstance(values[0], (int, float)):
            x, y = values[0], values[1]
            nx = self._mtx[0][0] * x + self._mtx[0][1] * y + self._mtx[0][2]
            ny = self._mtx[1][0] * x + self._mtx[1][1] * y + self._mtx[1][2]
            return (nx, ny)
        # List of points
        result = []
        for pt in values:
            x, y = pt[0], pt[1]
            nx = self._mtx[0][0] * x + self._mtx[0][1] * y + self._mtx[0][2]
            ny = self._mtx[1][0] * x + self._mtx[1][1] * y + self._mtx[1][2]
            result.append((nx, ny))
        return result

    def transform_point(self, point):
        x, y = point[0], point[1]
        nx = self._mtx[0][0] * x + self._mtx[0][1] * y + self._mtx[0][2]
        ny = self._mtx[1][0] * x + self._mtx[1][1] * y + self._mtx[1][2]
        return (nx, ny)

    def inverted(self):
        """Return the inverse Affine2D."""
        m = self._mtx
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        if abs(det) < 1e-15:
            raise ValueError("Singular transform matrix")
        inv_det = 1.0 / det
        inv = [
            [m[1][1] * inv_det, -m[0][1] * inv_det, 0],
            [-m[1][0] * inv_det, m[0][0] * inv_det, 0],
            [0, 0, 1],
        ]
        inv[0][2] = -(inv[0][0] * m[0][2] + inv[0][1] * m[1][2])
        inv[1][2] = -(inv[1][0] * m[0][2] + inv[1][1] * m[1][2])
        return Affine2D(inv)

    def is_identity(self):
        """Return whether this is the identity transform."""
        m = self._mtx
        return (abs(m[0][0] - 1) < 1e-15 and abs(m[0][1]) < 1e-15 and
                abs(m[0][2]) < 1e-15 and abs(m[1][0]) < 1e-15 and
                abs(m[1][1] - 1) < 1e-15 and abs(m[1][2]) < 1e-15)

    def frozen(self):
        return Affine2D([row[:] for row in self._mtx])

    def __repr__(self):
        return f"Affine2D({self._mtx!r})"

    def __eq__(self, other):
        if not isinstance(other, Affine2D):
            return NotImplemented
        for i in range(3):
            for j in range(3):
                if abs(self._mtx[i][j] - other._mtx[i][j]) > 1e-12:
                    return False
        return True

    @staticmethod
    def identity():
        """Return a new identity Affine2D."""
        return Affine2D()


class BboxTransform(Affine2DBase):
    """Transform that maps one Bbox to another."""

    def __init__(self, boxin, boxout):
        super().__init__()
        self._boxin = boxin
        self._boxout = boxout

    def get_matrix(self):
        sx = self._boxout.width / self._boxin.width if self._boxin.width else 1
        sy = self._boxout.height / self._boxin.height if self._boxin.height else 1
        tx = self._boxout.x0 - self._boxin.x0 * sx
        ty = self._boxout.y0 - self._boxin.y0 * sy
        return [[sx, 0, tx], [0, sy, ty], [0, 0, 1]]


class BboxTransformTo(Affine2DBase):
    """Transform that maps the unit bbox to a given Bbox."""

    def __init__(self, boxout):
        super().__init__()
        self._boxout = boxout

    def get_matrix(self):
        return [[self._boxout.width, 0, self._boxout.x0],
                [0, self._boxout.height, self._boxout.y0],
                [0, 0, 1]]


class BboxTransformFrom(Affine2DBase):
    """Transform that maps a given Bbox to the unit bbox."""

    def __init__(self, boxin):
        super().__init__()
        self._boxin = boxin

    def get_matrix(self):
        w = self._boxin.width if self._boxin.width else 1
        h = self._boxin.height if self._boxin.height else 1
        return [[1.0 / w, 0, -self._boxin.x0 / w],
                [0, 1.0 / h, -self._boxin.y0 / h],
                [0, 0, 1]]


class ScaledTranslation(Affine2DBase):
    """A transform that translates by (xt, yt) scaled by *scale_trans*."""

    def __init__(self, xt, yt, scale_trans=None):
        super().__init__()
        self._xt = xt
        self._yt = yt
        self._scale_trans = scale_trans

    def get_matrix(self):
        return [[1, 0, self._xt], [0, 1, self._yt], [0, 0, 1]]


class CompositeGenericTransform(Transform):
    """Composite of two transforms: applies *a* then *b*."""

    def __init__(self, a, b):
        super().__init__()
        self._a = a
        self._b = b

    def transform(self, values):
        return self._b.transform(self._a.transform(values))

    def transform_point(self, point):
        return self._b.transform_point(self._a.transform_point(point))

    def inverted(self):
        return CompositeGenericTransform(self._b.inverted(),
                                         self._a.inverted())


class BlendedGenericTransform(Transform):
    """A blended transform that uses *x_transform* for x and *y_transform* for y."""

    def __init__(self, x_transform, y_transform):
        super().__init__()
        self._x = x_transform
        self._y = y_transform


class BlendedAffine2D(Affine2DBase):
    """A blended affine transform."""

    def __init__(self, x_transform, y_transform):
        super().__init__()
        self._x = x_transform
        self._y = y_transform


def blended_transform_factory(x_transform, y_transform):
    """Return a blended transform from two 1D transforms."""
    return BlendedGenericTransform(x_transform, y_transform)


def offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches'):
    """Return a new transform with an added offset.

    Parameters
    ----------
    trans : Transform
        Base transform.
    fig : Figure, optional
        Figure for converting units (required for 'inches', 'points').
    x, y : float
        Offset amounts.
    units : {'inches', 'points', 'dots'}
        Units for x and y offset.

    Returns
    -------
    A new transform equivalent to trans + offset(x, y).
    """
    if units == 'inches':
        dpi = fig.dpi if fig is not None else 72.0
        x_pts = x * dpi
        y_pts = y * dpi
    elif units == 'points':
        dpi = fig.dpi if fig is not None else 72.0
        x_pts = x * dpi / 72.0
        y_pts = y * dpi / 72.0
    elif units in ('dots', 'pixels'):
        x_pts = float(x)
        y_pts = float(y)
    else:
        raise ValueError(f"units must be 'inches', 'points', or 'dots', not {units!r}")

    class _OffsetTransform:
        def __init__(self, base, dx, dy):
            self._base = base
            self._dx = dx
            self._dy = dy

        def transform(self, points):
            import numpy as np
            pts = np.asarray(points)
            if hasattr(self._base, 'transform'):
                pts = self._base.transform(pts)
            pts = pts.copy()
            pts[..., 0] += self._dx
            pts[..., 1] += self._dy
            return pts

        def transform_point(self, point):
            if hasattr(self._base, 'transform_point'):
                point = self._base.transform_point(point)
            return (point[0] + self._dx, point[1] + self._dy)

        def transform_path(self, path):
            return path

    return _OffsetTransform(trans, x_pts, y_pts)


def _mmul(a, b):
    """Multiply two 3x3 matrices."""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


# Convenience: nonsingular helper
def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    """Modify endpoints to avoid singularity."""
    if vmin == vmax:
        if abs(vmin) < tiny:
            vmin = -expander
            vmax = expander
        else:
            vmin -= abs(vmin) * expander
            vmax += abs(vmax) * expander
    if increasing and vmin > vmax:
        vmin, vmax = vmax, vmin
    return vmin, vmax
