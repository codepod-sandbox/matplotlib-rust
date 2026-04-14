"""Stub matplotlib.markers module."""
import numpy as np


# Standard marker styles
MarkerStyle = None  # will be defined below


class MarkerStyle:
    """Stub MarkerStyle."""

    # Marker style constants
    markers = {
        '.': 'point', ',': 'pixel', 'o': 'circle',
        'v': 'triangle_down', '^': 'triangle_up',
        '<': 'triangle_left', '>': 'triangle_right',
        '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right',
        '8': 'octagon', 's': 'square', 'p': 'pentagon',
        '*': 'star', 'h': 'hexagon1', 'H': 'hexagon2',
        '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond',
        '|': 'vline', '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled',
        0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown',
        4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown',
        'None': 'nothing', 'none': 'nothing', ' ': 'nothing', '': 'nothing',
    }

    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*',
                      'h', 'H', 'D', 'd', 'P', 'X')

    def __init__(self, marker=None, fillstyle=None):
        self._marker = marker
        self._fillstyle = fillstyle
        self._transform = None
        self._joinstyle = None
        self._capstyle = None
        self._filled = True

    def get_marker(self):
        return self._marker

    def is_filled(self):
        return self._filled

    def get_fillstyle(self):
        return self._fillstyle or 'full'

    def get_path(self):
        from matplotlib.path import Path
        return Path([[0, 0]])

    def get_transform(self):
        from matplotlib.transforms import IdentityTransform
        return IdentityTransform()

    def get_joinstyle(self):
        return self._joinstyle or 'round'

    def get_capstyle(self):
        return self._capstyle or 'butt'

    def __eq__(self, other):
        if isinstance(other, MarkerStyle):
            return self._marker == other._marker
        return self._marker == other

    def __hash__(self):
        return hash(self._marker)
