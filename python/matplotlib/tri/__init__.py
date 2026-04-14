"""Stub matplotlib.tri package."""


class Triangulation:
    """Stub Triangulation."""

    def __init__(self, x, y, triangles=None, mask=None):
        import numpy as np
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.triangles = triangles
        self.mask = mask


class TriContourSet:
    """Stub TriContourSet."""
    pass


class TriangulationInterpolator:
    """Stub TriangulationInterpolator."""
    pass


def tricontour(ax, *args, **kwargs):
    """Stub tricontour."""
    return TriContourSet()


def tricontourf(ax, *args, **kwargs):
    """Stub tricontourf."""
    return TriContourSet()


def tripcolor(ax, *args, **kwargs):
    """Stub tripcolor."""
    pass


def triplot(ax, *args, **kwargs):
    """Stub triplot."""
    return [], []
