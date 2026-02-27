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


class TestLine2DSetters:
    def test_set_get_color(self):
        line = Line2D([0], [0])
        line.set_color('blue')
        assert line.get_color() == 'blue'

    def test_set_c_alias(self):
        """set_c is an alias for set_color."""
        line = Line2D([0], [0])
        line.set_c('green')
        assert line.get_color() == 'green'

    def test_set_get_linewidth(self):
        line = Line2D([0], [0])
        line.set_linewidth(5.0)
        assert line.get_linewidth() == 5.0

    def test_set_lw_alias(self):
        """set_lw is an alias for set_linewidth."""
        line = Line2D([0], [0])
        line.set_lw(2.5)
        assert line.get_linewidth() == 2.5

    def test_set_get_linestyle(self):
        line = Line2D([0], [0])
        line.set_linestyle(':')
        assert line.get_linestyle() == ':'

    def test_set_ls_alias(self):
        """set_ls is an alias for set_linestyle."""
        line = Line2D([0], [0])
        line.set_ls('-.')
        assert line.get_linestyle() == '-.'

    def test_set_get_marker(self):
        line = Line2D([0], [0])
        line.set_marker('s')
        assert line.get_marker() == 's'

    def test_set_get_markersize(self):
        line = Line2D([0], [0])
        line.set_markersize(12.0)
        assert line.get_markersize() == 12.0

    def test_set_ms_alias(self):
        """set_ms is an alias for set_markersize."""
        line = Line2D([0], [0])
        line.set_ms(8.0)
        assert line.get_markersize() == 8.0

    def test_set_get_fillstyle(self):
        line = Line2D([0], [0])
        line.set_fillstyle('none')
        assert line.get_fillstyle() == 'none'

    def test_set_get_drawstyle(self):
        line = Line2D([0], [0])
        line.set_drawstyle('steps')
        assert line.get_drawstyle() == 'steps'
