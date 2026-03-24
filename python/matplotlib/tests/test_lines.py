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


class TestLine2DData:
    def test_get_data(self):
        """get_data returns (xdata, ydata) tuple."""
        line = Line2D([1, 2], [3, 4])
        x, y = line.get_data()
        assert x == [1, 2]
        assert y == [3, 4]

    def test_set_data(self):
        """set_data replaces both x and y."""
        line = Line2D([1, 2], [3, 4])
        line.set_data([10, 20], [30, 40])
        assert line.get_xdata() == [10, 20]
        assert line.get_ydata() == [30, 40]

    def test_set_xdata(self):
        line = Line2D([1, 2], [3, 4])
        line.set_xdata([10, 20, 30])
        assert line.get_xdata() == [10, 20, 30]
        assert line.get_ydata() == [3, 4]  # unchanged

    def test_set_ydata(self):
        line = Line2D([1, 2], [3, 4])
        line.set_ydata([10, 20, 30])
        assert line.get_ydata() == [10, 20, 30]
        assert line.get_xdata() == [1, 2]  # unchanged

    def test_data_is_copy(self):
        """get_xdata/get_ydata return copies, not references."""
        line = Line2D([1, 2], [3, 4])
        xd = line.get_xdata()
        xd.append(999)
        assert line.get_xdata() == [1, 2]


class TestLine2DArtist:
    def test_zorder(self):
        """Line2D default zorder is 2."""
        line = Line2D([0], [0])
        assert line.get_zorder() == 2

    def test_visible(self):
        """Line2D defaults to visible."""
        line = Line2D([0], [0])
        assert line.get_visible() is True

    def test_set_visible(self):
        line = Line2D([0], [0])
        line.set_visible(False)
        assert line.get_visible() is False

    def test_set_alpha(self):
        line = Line2D([0], [0])
        line.set_alpha(0.5)
        assert line.get_alpha() == 0.5

    def test_batch_set(self):
        """Artist.set() applies multiple properties."""
        line = Line2D([0], [0])
        line.set(color='red', linewidth=3.0, visible=False)
        assert line.get_color() == 'red'
        assert line.get_linewidth() == 3.0
        assert line.get_visible() is False

    def test_remove_from_axes(self):
        """Line2D.remove() removes from axes' lines list."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4])
        line = lines[0]
        assert line in ax.lines
        line.remove()
        assert line not in ax.lines
        plt.close('all')


# ===================================================================
# Additional Line2D tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class TestLine2DParametric:
    """Parametric tests for Line2D properties."""

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', 'black', '#ff0000'])
    def test_line_colors(self, color):
        """Line2D accepts named and hex colors."""
        line = Line2D([0, 1], [0, 1], color=color)
        assert line.get_color() == color

    @pytest.mark.parametrize('lw', [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    def test_line_linewidths(self, lw):
        """Line2D linewidth is settable to various values."""
        line = Line2D([0, 1], [0, 1])
        line.set_linewidth(lw)
        assert abs(line.get_linewidth() - lw) < 1e-10

    @pytest.mark.parametrize('ls', ['-', '--', '-.', ':'])
    def test_line_linestyles(self, ls):
        """Line2D linestyle is settable."""
        line = Line2D([0, 1], [0, 1])
        line.set_linestyle(ls)
        assert line.get_linestyle() == ls

    @pytest.mark.parametrize('ms', [1, 3, 5, 6, 8, 10, 12])
    def test_marker_sizes(self, ms):
        """Line2D markersize is settable."""
        line = Line2D([0, 1], [0, 1], marker='o', markersize=ms)
        assert abs(line.get_markersize() - ms) < 1e-10

    @pytest.mark.parametrize('alpha', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_line_alpha(self, alpha):
        """Line2D alpha is settable."""
        line = Line2D([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('zorder', [0, 1, 2, 5, 10])
    def test_line_zorder(self, zorder):
        """Line2D zorder is settable."""
        line = Line2D([0, 1], [0, 1])
        line.set_zorder(zorder)
        assert line.get_zorder() == zorder


class TestLine2DInAxes:
    """Tests for Line2D behavior within axes."""

    def test_plot_line_in_ax_lines(self):
        """ax.plot adds line to ax.lines."""
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6])
        assert lines[0] in ax.lines
        plt.close('all')

    def test_multiple_plots_accumulate(self):
        """Multiple ax.plot calls accumulate in ax.lines."""
        fig, ax = plt.subplots()
        for i in range(5):
            ax.plot([i], [i])
        assert len(ax.lines) == 5
        plt.close('all')

    def test_line_label_in_legend(self):
        """Labeled line appears in legend handles."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='series_A')
        leg = ax.legend()
        texts = [t.get_text() for t in leg.get_texts()]
        assert 'series_A' in texts
        plt.close('all')

    def test_xdata_ydata_match_input(self):
        """Line2D stores x and y data from plot."""
        fig, ax = plt.subplots()
        xs = [1.0, 2.0, 3.0]
        ys = [4.0, 5.0, 6.0]
        lines = ax.plot(xs, ys)
        assert list(lines[0].get_xdata()) == xs
        assert list(lines[0].get_ydata()) == ys
        plt.close('all')

    def test_axhline_y_value(self):
        """axhline line has correct y value."""
        fig, ax = plt.subplots()
        line = ax.axhline(0.5)
        # axhline creates a horizontal line at y=0.5
        ydata = line.get_ydata()
        assert all(abs(y - 0.5) < 1e-10 for y in ydata)
        plt.close('all')

    def test_axvline_x_value(self):
        """axvline line has correct x value."""
        fig, ax = plt.subplots()
        line = ax.axvline(0.3)
        xdata = line.get_xdata()
        assert all(abs(x - 0.3) < 1e-10 for x in xdata)
        plt.close('all')
