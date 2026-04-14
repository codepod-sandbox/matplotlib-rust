"""Stub matplotlib.streamplot module."""
import numpy as np
from matplotlib.artist import Artist


class StreamplotSet:
    """Stub StreamplotSet."""

    def __init__(self, lines, arrows):
        self.lines = lines
        self.arrows = arrows


def streamplot(axes, x, y, u, v, density=1, linewidth=None, color=None,
               cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
               minlength=0.1, transform=None, zorder=None, start_points=None,
               maxlength=4.0, integration_direction='both',
               broken_streamlines=True, **kwargs):
    """Stub streamplot function."""
    return StreamplotSet([], [])
