"""matplotlib.path — Path class for defining arbitrary paths."""


class Path:
    """A series of possibly disconnected, possibly closed, line and curve segments.

    Parameters
    ----------
    vertices : list of (x, y) tuples
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
        self.vertices = list(vertices)
        if codes is None:
            codes = [self.MOVETO] + [self.LINETO] * (len(vertices) - 1)
        self.codes = list(codes)

    def __len__(self):
        return len(self.vertices)

    def __repr__(self):
        return f"Path({self.vertices!r}, {self.codes!r})"
