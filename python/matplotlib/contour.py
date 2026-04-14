"""Stub matplotlib.contour module."""
import numpy as np
from matplotlib.artist import Artist


class ContourSet(Artist):
    """Stub ContourSet for contour/contourf."""

    def __init__(self, ax, *args, **kwargs):
        super().__init__()
        self.axes = ax
        self.levels = []
        self.collections = []
        self.labelTexts = []

    def clabel(self, levels=None, **kwargs):
        return []

    def legend_elements(self, **kwargs):
        return [], []


class QuadContourSet(ContourSet):
    """Stub QuadContourSet."""
    pass
