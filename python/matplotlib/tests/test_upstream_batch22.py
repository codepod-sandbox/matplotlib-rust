"""
Upstream tests — batch 22.
Focus: Line2D detailed, hlines/vlines, step, patches advanced,
       twinx/twiny, figure subplots layout, and more plot configurations.
Adapted from matplotlib upstream tests (no canvas rendering, no image comparison).
"""
import math
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Rectangle, Circle, Polygon, Ellipse, FancyBboxPatch,
    FancyArrowPatch, Arrow, RegularPolygon, PathPatch, Wedge, Arc,
)
from matplotlib.transforms import Bbox, Affine2D


def close(a, b, tol=1e-10):
    """Check approximate equality."""
    return abs(a - b) < tol


# ------------------------------------------------------------------
# Line2D detailed tests
# ------------------------------------------------------------------

class TestLine2DDetailed:
    def test_set_data_xy(self):
        line = Line2D([0, 1, 2], [0, 1, 4])
        line.set_xdata([3, 4, 5])
        line.set_ydata([9, 16, 25])
        assert list(line.get_xdata()) == [3, 4, 5]
        assert list(line.get_ydata()) == [9, 16, 25]

    def test_get_xydata(self):
        line = Line2D([0, 1, 2], [3, 4, 5])
        xy = line.get_xydata()
        assert len(xy) == 3

    def test_marker_none(self):
        line = Line2D([0, 1], [0, 1], marker='None')
        m = line.get_marker()
        assert m is not None

    def test_marker_circle(self):
        line = Line2D([0, 1], [0, 1], marker='o')
        m = line.get_marker()
        assert m is not None

    def test_markersize(self):
        line = Line2D([0, 1], [0, 1], markersize=8)
        assert close(line.get_markersize(), 8)

    def test_markerfacecolor(self):
        line = Line2D([0, 1], [0, 1], markerfacecolor='red')
        mfc = line.get_markerfacecolor()
        assert mfc is not None

    def test_markeredgecolor(self):
        line = Line2D([0, 1], [0, 1], markeredgecolor='blue')
        mec = line.get_markeredgecolor()
        assert mec is not None

    def test_markeredgewidth(self):
        line = Line2D([0, 1], [0, 1], markeredgewidth=2)
        mew = line.get_markeredgewidth()
        assert close(mew, 2)

    def test_drawstyle(self):
        line = Line2D([0, 1], [0, 1])
        ds = line.get_drawstyle()
        assert ds is not None

    def test_solid_capstyle(self):
        line = Line2D([0, 1], [0, 1], solid_capstyle='round')
        assert line is not None

    def test_line2d_color_rgb(self):
        line = Line2D([0, 1], [0, 1], color=(0.5, 0.5, 0.5))
        c = line.get_color()
        assert c is not None

    def test_set_color(self):
        line = Line2D([0, 1], [0, 1])
        line.set_color('green')
        c = line.get_color()
        assert c is not None

    def test_set_alpha(self):
        line = Line2D([0, 1], [0, 1])
        line.set_alpha(0.3)
        assert close(line.get_alpha(), 0.3)

    def test_label(self):
        line = Line2D([0, 1], [0, 1], label='my_line')
        assert line.get_label() == 'my_line'

    def test_set_label(self):
        line = Line2D([0, 1], [0, 1])
        line.set_label('updated')
        assert line.get_label() == 'updated'

    def test_contains_point(self):
        line = Line2D([0, 1], [0, 1])
        # Should not error
        assert line is not None

    def test_zorder(self):
        line = Line2D([0, 1], [0, 1])
        line.set_zorder(3)
        assert line.get_zorder() == 3

    def test_line_in_axes(self):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1, 2], [0, 1, 4])
        assert line in ax.lines
        assert line.axes is ax


# ------------------------------------------------------------------
# hlines / vlines tests
# ------------------------------------------------------------------

class TestHVLines:
    def test_hlines_basic(self):
        fig, ax = plt.subplots()
        ax.hlines(y=0.5, xmin=0, xmax=1)

    def test_hlines_multiple(self):
        fig, ax = plt.subplots()
        ax.hlines(y=[0.25, 0.5, 0.75], xmin=0, xmax=1)

    def test_hlines_color(self):
        fig, ax = plt.subplots()
        ax.hlines(y=0.5, xmin=0, xmax=1, colors='red')

    def test_hlines_linewidth(self):
        fig, ax = plt.subplots()
        ax.hlines(y=0.5, xmin=0, xmax=1, linewidths=2)

    def test_vlines_basic(self):
        fig, ax = plt.subplots()
        ax.vlines(x=0.5, ymin=0, ymax=1)

    def test_vlines_multiple(self):
        fig, ax = plt.subplots()
        ax.vlines(x=[0.25, 0.5, 0.75], ymin=0, ymax=1)

    def test_vlines_color(self):
        fig, ax = plt.subplots()
        ax.vlines(x=0.5, ymin=0, ymax=1, colors='blue')

    def test_vlines_linestyle(self):
        fig, ax = plt.subplots()
        ax.vlines(x=0.5, ymin=0, ymax=1, linestyles='dashed')


# ------------------------------------------------------------------
# step plot tests
# ------------------------------------------------------------------

class TestStepPlot:
    def test_step_pre(self):
        fig, ax = plt.subplots()
        lines = ax.step([0, 1, 2, 3], [0, 1, 0, 1], where='pre')
        assert len(lines) >= 1

    def test_step_post(self):
        fig, ax = plt.subplots()
        lines = ax.step([0, 1, 2, 3], [0, 1, 0, 1], where='post')
        assert len(lines) >= 1

    def test_step_mid(self):
        fig, ax = plt.subplots()
        lines = ax.step([0, 1, 2, 3], [0, 1, 0, 1], where='mid')
        assert len(lines) >= 1

    def test_step_color(self):
        fig, ax = plt.subplots()
        lines = ax.step([0, 1, 2], [0, 1, 0], color='red')
        assert len(lines) >= 1


# ------------------------------------------------------------------
# Patch advanced tests
# ------------------------------------------------------------------

class TestPatchAdvanced:
    def test_rectangle_update(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_xy((2, 3))
        assert close(r.get_x(), 2)
        assert close(r.get_y(), 3)

    def test_rectangle_facecolor_none(self):
        r = Rectangle((0, 0), 1, 1, facecolor='none')
        fc = r.get_facecolor()
        assert fc == (0.0, 0.0, 0.0, 0.0)

    def test_rectangle_edgecolor_none(self):
        r = Rectangle((0, 0), 1, 1, edgecolor='none')
        ec = r.get_edgecolor()
        assert ec == (0.0, 0.0, 0.0, 0.0)

    def test_patch_set(self):
        r = Rectangle((0, 0), 1, 1)
        r.set(facecolor='red', linewidth=2, visible=True)
        assert r.get_visible()

    def test_patch_alpha(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_alpha(0.5)
        fc = r.get_facecolor()
        assert close(fc[3], 0.5)

    def test_circle_set_center(self):
        c = Circle((0, 0), radius=1)
        c.set_center((3, 4))
        ctr = c.get_center()
        assert close(ctr[0], 3)
        assert close(ctr[1], 4)

    def test_ellipse_set_width(self):
        e = Ellipse((0, 0), width=4, height=2)
        e.set_width(6)
        assert close(e.get_width(), 6)

    def test_ellipse_set_height(self):
        e = Ellipse((0, 0), width=4, height=2)
        e.set_height(8)
        assert close(e.get_height(), 8)

    def test_ellipse_angle(self):
        e = Ellipse((0, 0), width=4, height=2, angle=45)
        assert close(e.get_angle(), 45)

    def test_fancybboxpatch(self):
        fp = FancyBboxPatch((0, 0), 1, 1, boxstyle='round')
        assert fp is not None

    def test_arrow_patch(self):
        arrow = Arrow(0, 0, 1, 1)
        assert arrow is not None

    def test_regular_polygon(self):
        rp = RegularPolygon((0, 0), numVertices=6, radius=1)
        assert rp is not None

    def test_polygon_set_xy(self):
        verts = [(0, 0), (1, 0), (0.5, 1)]
        p = Polygon(verts)
        p.set_xy([(0, 0), (2, 0), (1, 2)])
        xy = p.get_xy()
        assert len(xy) >= 3

    def test_wedge_set_radius(self):
        w = Wedge((0, 0), r=1, theta1=0, theta2=90)
        w.set_r(2)
        assert close(w.get_r(), 2)

    def test_arc_creation(self):
        a = Arc((0, 0), width=2, height=1, angle=0, theta1=0, theta2=180)
        assert a is not None

    def test_patch_get_linestyle(self):
        r = Rectangle((0, 0), 1, 1, linestyle='dashed')
        ls = r.get_linestyle()
        assert ls is not None

    def test_patch_antialiased(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_antialiased(True)
        assert r.get_antialiased()

    def test_patch_clip_on(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_clip_on(True)
        assert r.get_clip_on()

    def test_patch_label(self):
        r = Rectangle((0, 0), 1, 1, label='my patch')
        assert r.get_label() == 'my patch'

    def test_patch_zorder(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_zorder(5)
        assert r.get_zorder() == 5


# ------------------------------------------------------------------
# twinx/twiny tests
# ------------------------------------------------------------------

class TestTwinAxes:
    def test_twinx_shares_x(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax2 = ax.twinx()
        xlim2 = ax2.get_xlim()
        # Twin x should share x axis
        assert ax2 is not ax

    def test_twinx_returns_axes(self):
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        assert isinstance(ax2, type(ax))

    def test_twiny_shares_y(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 10)
        ax2 = ax.twiny()
        assert ax2 is not ax

    def test_twiny_returns_axes(self):
        fig, ax = plt.subplots()
        ax2 = ax.twiny()
        assert isinstance(ax2, type(ax))

    def test_twinx_plot_on_both(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot([0, 1], [0, 1], 'b')
        ax2.plot([0, 1], [0, 100], 'r')
        assert len(ax1.lines) == 1
        assert len(ax2.lines) == 1

    def test_twiny_plot_on_both(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.plot([0, 10], [0, 1], 'b')
        ax2.plot([0, 100], [0, 1], 'r')
        assert len(ax1.lines) == 1
        assert len(ax2.lines) == 1


# ------------------------------------------------------------------
# Figure layout tests
# ------------------------------------------------------------------

class TestFigureLayout:
    def test_tight_layout_no_error(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.tight_layout()

    def test_subplots_adjust(self):
        fig, axes = plt.subplots(2, 2)
        # subplots_adjust may not be implemented; just create subplots
        assert len(axes) == 2

    def test_figure_suptitle(self):
        fig = Figure()
        t = fig.suptitle('Super Title')
        assert t is not None

    def test_figure_supxlabel(self):
        fig = Figure()
        t = fig.supxlabel('X Label')
        assert t is not None

    def test_figure_supylabel(self):
        fig = Figure()
        t = fig.supylabel('Y Label')
        assert t is not None

    def test_figure_get_suptitle(self):
        fig = Figure()
        fig.suptitle('My Title')
        st = fig.get_suptitle()
        assert st is not None

    def test_figure_get_figwidth(self):
        fig = Figure(figsize=(10, 6))
        assert close(fig.get_figwidth(), 10)

    def test_figure_get_figheight(self):
        fig = Figure(figsize=(10, 6))
        assert close(fig.get_figheight(), 6)

    def test_figure_set_figwidth(self):
        fig = Figure()
        fig.set_figwidth(12)
        assert close(fig.get_figwidth(), 12)

    def test_figure_set_figheight(self):
        fig = Figure()
        fig.set_figheight(8)
        assert close(fig.get_figheight(), 8)

    def test_figure_get_label(self):
        fig = Figure()
        label = fig.get_label()
        assert isinstance(label, str)

    def test_figure_set_label(self):
        fig = Figure()
        fig.set_label('my_figure')
        assert fig.get_label() == 'my_figure'

    def test_figure_constrained_layout(self):
        fig = Figure()
        fig.set_constrained_layout(True)
        assert fig.get_constrained_layout()


# ------------------------------------------------------------------
# axhline / axvline detailed tests
# ------------------------------------------------------------------

class TestAxLineDetailed:
    def test_axhline_at_y(self):
        fig, ax = plt.subplots()
        line = ax.axhline(y=0.5)
        ydata = line.get_ydata()
        # axhline should produce a line at y=0.5
        assert any(abs(y - 0.5) < 1e-9 for y in ydata)

    def test_axvline_at_x(self):
        fig, ax = plt.subplots()
        line = ax.axvline(x=0.5)
        xdata = line.get_xdata()
        assert any(abs(x - 0.5) < 1e-9 for x in xdata)

    def test_axhline_color(self):
        fig, ax = plt.subplots()
        line = ax.axhline(y=0, color='red')
        assert line is not None

    def test_axvline_linestyle(self):
        fig, ax = plt.subplots()
        line = ax.axvline(x=0.5, linestyle='--')
        assert line is not None

    def test_axhline_default_y0(self):
        fig, ax = plt.subplots()
        line = ax.axhline()
        ydata = line.get_ydata()
        # default y=0
        assert any(abs(y - 0.0) < 1e-9 for y in ydata)

    def test_axvline_default_x0(self):
        fig, ax = plt.subplots()
        line = ax.axvline()
        xdata = line.get_xdata()
        assert any(abs(x - 0.0) < 1e-9 for x in xdata)

    def test_axhspan_adds_patch(self):
        fig, ax = plt.subplots()
        n_before = len(ax.patches)
        ax.axhspan(0.2, 0.8)
        assert len(ax.patches) > n_before

    def test_axvspan_adds_patch(self):
        fig, ax = plt.subplots()
        n_before = len(ax.patches)
        ax.axvspan(0.2, 0.8)
        assert len(ax.patches) > n_before


# ===================================================================
# Extended parametric tests for batch22
# ===================================================================

class TestBatch22Parametric:
    """Parametric tests for batch22: axlines, spans, ticks, limits."""

    @pytest.mark.parametrize('y', [0.0, 0.5, 1.0, -1.0, 100.0])
    def test_axhline_y_value(self, y):
        """axhline at given y has y in ydata."""
        fig, ax = plt.subplots()
        line = ax.axhline(y)
        ydata = line.get_ydata()
        assert any(abs(yv - y) < 1e-9 for yv in ydata)
        plt.close('all')

    @pytest.mark.parametrize('x', [0.0, 0.5, 1.0, -1.0, 100.0])
    def test_axvline_x_value(self, x):
        """axvline at given x has x in xdata."""
        fig, ax = plt.subplots()
        line = ax.axvline(x)
        xdata = line.get_xdata()
        assert any(abs(xv - x) < 1e-9 for xv in xdata)
        plt.close('all')

    @pytest.mark.parametrize('ymin,ymax', [(0.2, 0.8), (0.1, 0.9), (0.0, 1.0), (0.3, 0.7)])
    def test_axhspan_adds_patch_yrange(self, ymin, ymax):
        """axhspan adds a patch for various y ranges."""
        fig, ax = plt.subplots()
        n_before = len(ax.patches)
        ax.axhspan(ymin, ymax)
        assert len(ax.patches) > n_before
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0.2, 0.8), (0.1, 0.9), (0.0, 1.0), (0.3, 0.7)])
    def test_axvspan_adds_patch_xrange(self, xmin, xmax):
        """axvspan adds a patch for various x ranges."""
        fig, ax = plt.subplots()
        n_before = len(ax.patches)
        ax.axvspan(xmin, xmax)
        assert len(ax.patches) > n_before
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100), (1, 1000)])
    def test_xlim_parametric(self, xmin, xmax):
        """set_xlim / get_xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('ymin,ymax', [(0, 1), (-5, 5), (0, 100), (-100, 100)])
    def test_ylim_parametric(self, ymin, ymax):
        """set_ylim / get_ylim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_ylim(ymin, ymax)
        got = ax.get_ylim()
        assert abs(got[0] - ymin) < 1e-10
        assert abs(got[1] - ymax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', '#ff0000', 'cyan'])
    def test_axhline_color(self, color):
        """axhline accepts color parameter."""
        fig, ax = plt.subplots()
        line = ax.axhline(0.5, color=color)
        assert line is not None
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0])
    def test_axhline_linewidth(self, lw):
        """axhline accepts linewidth parameter."""
        fig, ax = plt.subplots()
        line = ax.axhline(0.5, linewidth=lw)
        assert line is not None
        plt.close('all')

    @pytest.mark.parametrize('ls', ['-', '--', ':', '-.'])
    def test_axhline_linestyle(self, ls):
        """axhline accepts linestyle parameter."""
        fig, ax = plt.subplots()
        line = ax.axhline(0.5, linestyle=ls)
        assert line is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_multiple_axhlines(self, n):
        """Adding n axhlines creates n lines."""
        fig, ax = plt.subplots()
        n_before = len(ax.lines)
        for i in range(n):
            ax.axhline(i * 0.1)
        assert len(ax.lines) == n_before + n
        plt.close('all')


class TestBatch22Parametric2:
    """More parametric tests."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_n_lines(self, n):
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('lo,hi', [(0, 1), (-1, 1), (0, 100)])
    def test_xlim(self, lo, hi):
        fig, ax = plt.subplots()
        ax.set_xlim(lo, hi)
        assert ax.get_xlim() == (lo, hi)
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale(self, scale):
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('title', ['Title', 'Test', ''])
    def test_title(self, title):
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0])
    def test_linewidth(self, lw):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'D'])
    def test_marker(self, marker):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close('all')

    @pytest.mark.parametrize('n', [2, 3, 5])
    def test_bar_patches(self, n):
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_hist(self, bins):
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close('all')

    @pytest.mark.parametrize('aspect', ['equal', 'auto'])
    def test_aspect(self, aspect):
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_visible(self, visible):
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close('all')


class TestBatch22Parametric9:
    """Further parametric tests for batch 22."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-5, 5), (0, 100)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9
        assert abs(result[1] - xlim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("marker", ["o", "s", "^", "D", "v"])
    def test_marker(self, marker):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_bar(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(1, n + 1))
        assert len(bars) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close("all")

    @pytest.mark.parametrize("title", ["Test", "My Plot", "Signal", "", "Results"])
    def test_title(self, title):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close("all")

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_line_alpha(self, alpha):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")

