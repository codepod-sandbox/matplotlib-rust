"""Stub matplotlib.quiver module."""
import numpy as np
from matplotlib.artist import Artist


class Quiver(Artist):
    """Stub Quiver artist for quiver plots."""

    def __init__(self, ax, *args, **kwargs):
        super().__init__()
        self.axes = ax
        self.N = 0
        self.scale = None

    def set_UVC(self, U, V, C=None):
        pass

    def get_datalim(self, transData):
        from matplotlib.transforms import Bbox
        return Bbox.null()


class QuiverKey(Artist):
    """Stub QuiverKey for labeling quiver plots."""

    def __init__(self, Q, X, Y, U, label, **kwargs):
        super().__init__()
        self.Q = Q
        self.X = X
        self.Y = Y
        self.U = U
        self.label = label


class Barbs(Artist):
    """Stub Barbs artist for barb plots."""

    def __init__(self, ax, *args, **kwargs):
        super().__init__()
        self.axes = ax
