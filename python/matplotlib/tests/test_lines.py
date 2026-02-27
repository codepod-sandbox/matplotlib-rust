"""Tests for matplotlib.lines module — Line2D artist."""

import pytest

from matplotlib.lines import Line2D
from matplotlib.colors import to_hex


class TestLine2DConstruction:
    def test_basic_construction(self):
        """Line2D stores x/y data."""
        line = Line2D([1, 2, 3], [4, 5, 6])
        assert line.get_xdata() == [1, 2, 3]
        assert line.get_ydata() == [4, 5, 6]

    def test_default_color(self):
        """Default color is C0."""
        line = Line2D([0], [0])
        assert line.get_color() == 'C0'

    def test_default_linewidth(self):
        """Default linewidth is 1.5."""
        line = Line2D([0], [0])
        assert line.get_linewidth() == 1.5

    def test_default_linestyle(self):
        """Default linestyle is '-'."""
        line = Line2D([0], [0])
        assert line.get_linestyle() == '-'

    def test_default_marker(self):
        """Default marker is 'None' (string)."""
        line = Line2D([0], [0])
        assert line.get_marker() == 'None'

    def test_default_markersize(self):
        """Default markersize is 6.0."""
        line = Line2D([0], [0])
        assert line.get_markersize() == 6.0

    def test_default_fillstyle(self):
        """Default fillstyle is 'full'."""
        line = Line2D([0], [0])
        assert line.get_fillstyle() == 'full'

    def test_default_drawstyle(self):
        """Default drawstyle is 'default'."""
        line = Line2D([0], [0])
        assert line.get_drawstyle() == 'default'

    def test_explicit_kwargs(self):
        """Explicit kwargs override defaults."""
        line = Line2D([0], [0], color='red', linewidth=3.0,
                       linestyle='--', marker='o')
        assert line.get_color() == 'red'
        assert line.get_linewidth() == 3.0
        assert line.get_linestyle() == '--'
        assert line.get_marker() == 'o'

    def test_label(self):
        """Label is set via kwarg."""
        line = Line2D([0], [0], label='test')
        assert line.get_label() == 'test'

    def test_no_label(self):
        """No label kwarg gives empty label."""
        line = Line2D([0], [0])
        assert line.get_label() == ''
