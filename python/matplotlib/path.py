"""matplotlib.path — Path class for defining arbitrary paths."""

import numpy as np


class Path:
    """A series of possibly disconnected, possibly closed, line and curve segments.

    Parameters
    ----------
    vertices : array-like of (x, y) pairs
        The vertices of the path.
    codes : list of int, optional
        Path codes (MOVETO, LINETO, etc.). If None, all are LINETO
        except the first which is MOVETO.
    """

    # Path code constants (match upstream matplotlib values)
    STOP = 0
    MOVETO = 1
    LINETO = 2
    CURVE3 = 3
    CURVE4 = 4
    CLOSEPOLY = 79

    def __init__(self, vertices, codes=None):
        self.vertices = np.asarray(vertices, dtype=float)
        if codes is None:
            n = len(self.vertices)
            codes = [self.MOVETO] + [self.LINETO] * (n - 1)
        self.codes = list(codes)

    def __len__(self):
        return len(self.vertices)

    def __repr__(self):
        return f"Path({self.vertices!r}, {self.codes!r})"

    @classmethod
    def unit_circle(cls):
        """Return a Path for the unit circle (Bezier approximation)."""
        # 4 cubic Bezier arcs; MAGIC = 4/3 * tan(pi/8)
        MAGIC = 0.2652031
        SQRT2 = 0.7071068  # not used but kept for reference
        vertices = np.array([
            [1.0,  0.0],
            [1.0,  MAGIC],    [MAGIC,  1.0],   [0.0,  1.0],
            [-MAGIC, 1.0],    [-1.0,  MAGIC],  [-1.0,  0.0],
            [-1.0, -MAGIC],   [-MAGIC, -1.0],  [0.0, -1.0],
            [MAGIC, -1.0],    [1.0, -MAGIC],   [1.0,  0.0],
            [0.0,  0.0],
        ])
        codes = [
            cls.MOVETO,
            cls.CURVE4, cls.CURVE4, cls.CURVE4,
            cls.CURVE4, cls.CURVE4, cls.CURVE4,
            cls.CURVE4, cls.CURVE4, cls.CURVE4,
            cls.CURVE4, cls.CURVE4, cls.CURVE4,
            cls.CLOSEPOLY,
        ]
        return cls(vertices, codes)
