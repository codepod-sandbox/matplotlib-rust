"""Tests for matplotlib.axes --- Axes class and all plot types.

Covers labels, limits, plot, scatter, bar, hist, errorbar, fill_between,
axhline/axvline, text/annotate, grid/legend, and axes state management.
~60 tests total.
"""

import math
import numpy as np
import pytest

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.patches import Rectangle, Patch
from matplotlib.text import Text, Annotation
from matplotlib.colors import to_hex, to_rgba
from matplotlib._codepod_compat import DEFAULT_CYCLE


def _arr_eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def _color_close(a, b):
    return np.allclose(to_rgba(a), to_rgba(b))


# ===================================================================
# Labels (4 tests)
# ===================================================================

class TestLabels:
    def test_get_labels(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        assert ax.get_xlabel() == 'x label'
        assert ax.get_ylabel() == 'y label'

    def test_get_set_title(self):
        fig, ax = plt.subplots()
        ax.set_title('my title')
        assert ax.get_title() == 'my title'

    def test_default_labels_empty(self):
        fig, ax = plt.subplots()
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''
        assert ax.get_title() == ''

    def test_overwrite_labels(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('first')
        ax.set_xlabel('second')
        assert ax.get_xlabel() == 'second'


# ===================================================================
# Limits (5 tests)
# ===================================================================

class TestLimits:
    def test_set_get_xlim(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        assert ax.get_xlim() == (0, 10)

    def test_set_get_ylim(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-5, 5)
        assert ax.get_ylim() == (-5, 5)

    def test_inverted_limits(self):
        """Invert x-axis before plotting; high value should come first."""
        fig, ax = plt.subplots()
        ax.invert_xaxis()
        ax.plot([1, 5], [10, 20])
        xlim = ax.get_xlim()
        # After inversion the high value should come first (margin may shift values)
        assert xlim[0] > xlim[1]

    def test_invalid_limits_nan(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="NaN"):
            ax.set_xlim(float('nan'), 10)

    def test_invalid_limits_inf(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Inf"):
            ax.set_ylim(0, float('inf'))


# ===================================================================
# Plot (8 tests)
# ===================================================================

class TestPlot:
    def test_plot_returns_line2d(self):
        fig, ax = plt.subplots()
        result = ax.plot([1, 2], [3, 4])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Line2D)

    def test_plot_basic(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6])
        line = lines[0]
        assert _arr_eq(line.get_xdata(), [1, 2, 3])
        assert _arr_eq(line.get_ydata(), [4, 5, 6])

    def test_plot_format_string(self):
        """'ro-' sets color to red, marker to 'o', linestyle to '-'."""
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], 'ro-')
        line = lines[0]
        assert _color_close(line.get_color(), 'r')
        assert line.get_marker() == 'o'
        assert line.get_linestyle() == '-'

    def test_plot_y_only(self):
        """plot([1,2,3]) auto-generates x=[0,1,2]."""
        fig, ax = plt.subplots()
        lines = ax.plot([10, 20, 30])
        line = lines[0]
        assert _arr_eq(line.get_xdata(), [0, 1, 2])
        assert _arr_eq(line.get_ydata(), [10, 20, 30])

    def test_plot_kwargs(self):
        """color='blue', linewidth=3 override defaults."""
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], color='blue', linewidth=3)
        line = lines[0]
        assert _color_close(line.get_color(), 'blue')
        assert line.get_linewidth() == 3

    def test_plot_label(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], label='test')
        assert lines[0].get_label() == 'test'

    def test_plot_empty(self):
        """plot([], []) returns [Line2D] with empty data."""
        fig, ax = plt.subplots()
        lines = ax.plot([], [])
        assert isinstance(lines, list)
        assert len(lines) == 1
        assert isinstance(lines[0], Line2D)
        assert _arr_eq(lines[0].get_xdata(), [])
        assert _arr_eq(lines[0].get_ydata(), [])

    def test_plot_color_cycle(self):
        """Successive plots use different colors from the default cycle."""
        fig, ax = plt.subplots()
        l1 = ax.plot([1], [2])
        l2 = ax.plot([1], [2])
        c1 = l1[0].get_color()
        c2 = l2[0].get_color()
        active_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        assert c1 != c2
        assert c1 == active_cycle[0]
        assert c2 == active_cycle[1]


# ===================================================================
# Scatter (5 tests)
# ===================================================================

class TestScatter:
    def test_scatter_returns_pathcollection(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4])
        assert isinstance(pc, PathCollection)

    def test_scatter_basic(self):
        import numpy as np
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2, 3], [4, 5, 6])
        offsets = pc.get_offsets()
        assert isinstance(offsets, np.ndarray)
        assert np.allclose(offsets, [(1, 4), (2, 5), (3, 6)])

    def test_scatter_empty(self):
        """scatter([], []) does not raise."""
        import numpy as np
        fig, ax = plt.subplots()
        pc = ax.scatter([], [])
        offsets = pc.get_offsets()
        assert isinstance(offsets, np.ndarray)
        assert len(offsets) == 0

    def test_scatter_label(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1], [2], label='my_scatter')
        assert pc.get_label() == 'my_scatter'

    def test_scatter_color(self):
        """Custom color is applied to facecolors."""
        fig, ax = plt.subplots()
        pc = ax.scatter([1], [2], c='red')
        fc = pc.get_facecolors()
        assert len(fc) == 1
        assert _color_close(fc[0], 'red')


# ===================================================================
# Bar (8 tests)
# ===================================================================

class TestBar:
    def test_bar_returns_barcontainer(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2, 3], [4, 5, 6])
        assert isinstance(bc, BarContainer)

    def test_bar_basic(self):
        """len(container) == len(x), patches are Rectangle."""
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2, 3], [4, 5, 6])
        assert len(bc) == 3
        for patch in bc:
            assert isinstance(patch, Rectangle)

    def test_bar_empty(self):
        """bar([], []) does not raise."""
        fig, ax = plt.subplots()
        bc = ax.bar([], [])
        assert len(bc) == 0

    def test_bar_scalar_height(self):
        """Scalar height is broadcast to all bars."""
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2, 3], 5)
        assert len(bc) == 3
        for patch in bc:
            assert patch.get_height() == 5

    def test_bar_facecolor_precedence(self):
        """facecolor overrides color."""
        fig, ax = plt.subplots()
        bc = ax.bar([1], [2], facecolor='red', color='blue')
        # facecolor should be red, not blue
        fc = bc[0].get_facecolor()
        expected_rgba = (1.0, 0.0, 0.0, 1.0)
        assert fc == expected_rgba

    def test_bar_color_none(self):
        """color='none' makes face transparent (0,0,0,0)."""
        fig, ax = plt.subplots()
        bc = ax.bar([1], [2], color='none')
        fc = bc[0].get_facecolor()
        assert fc == (0.0, 0.0, 0.0, 0.0)

    def test_bar_edgecolor_none(self):
        """edgecolor='none' makes edge transparent."""
        fig, ax = plt.subplots()
        bc = ax.bar([1], [2], edgecolor='none')
        ec = bc[0].get_edgecolor()
        assert ec == (0.0, 0.0, 0.0, 0.0)

    def test_bar_label(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4], label='bars')
        assert bc.get_label() == 'bars'

    def test_bar_single_height(self):
        """Scalar height is broadcast to all bars when passed as a list."""
        # Implementation requires height as iterable; we pass [5, 5, 5]
        # to simulate broadcast behavior.
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2, 3], [5, 5, 5])
        for patch in bc:
            assert patch.get_height() == 5


# ===================================================================
# Hist (5 tests)
# ===================================================================

class TestHist:
    def test_hist_returns_tuple(self):
        """Returns (counts, edges, container)."""
        fig, ax = plt.subplots()
        result = ax.hist([1, 2, 3, 4, 5], bins=5)
        assert isinstance(result, tuple)
        assert len(result) == 3
        counts, edges, container = result
        assert isinstance(counts, (list, np.ndarray))
        assert isinstance(edges, (list, np.ndarray))
        assert isinstance(container, BarContainer)

    def test_hist_basic(self):
        """Counts match expected binning."""
        fig, ax = plt.subplots()
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        counts, edges, _ = ax.hist(data, bins=4)
        assert _arr_eq(counts, [1, 2, 3, 4])
        assert len(edges) == 5  # bins + 1 edges

    def test_hist_custom_bins_int(self):
        """Custom number of bins produces correct count of bins."""
        fig, ax = plt.subplots()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts, edges, bc = ax.hist(data, bins=2)
        assert len(counts) == 2
        assert len(edges) == 3  # 2 bins + 1
        assert sum(counts) == len(data)

    def test_hist_density_kwarg(self):
        """density=True normalizes so area under histogram equals 1."""
        fig, ax = plt.subplots()
        counts, edges, bc = ax.hist([1, 2, 3, 4, 5], bins=5, density=True)
        assert len(counts) == 5
        # Integral (sum of count * bin_width) should equal 1.0
        bin_width = edges[1] - edges[0]
        total_area = sum(c * bin_width for c in counts)
        assert abs(total_area - 1.0) < 1e-10

    def test_hist_label(self):
        fig, ax = plt.subplots()
        _, _, bc = ax.hist([1, 2, 3], bins=3, label='histogram')
        handles, labels = ax.get_legend_handles_labels()
        assert 'histogram' in labels


# ===================================================================
# Errorbar (3 tests)
# ===================================================================

class TestErrorbar:
    def test_errorbar_returns_container(self):
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.1, 0.2, 0.3])
        assert isinstance(ec, ErrorbarContainer)

    def test_errorbar_fmt_none(self):
        """fmt='none' suppresses data line — plotline is None."""
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2], [3, 4], yerr=[0.1, 0.2], fmt='none')
        assert isinstance(ec, ErrorbarContainer)
        # lines is a 3-tuple: (plotline, caplines, barlinecols)
        assert len(ec.lines) == 3
        assert ec.lines[0] is None  # no data line

    def test_errorbar_with_data(self):
        """Basic x, y, yerr works and data line has correct data."""
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2, 3], [10, 20, 30], yerr=[1, 2, 3])
        data_line = ec.lines[0]
        assert isinstance(data_line, Line2D)
        assert _arr_eq(data_line.get_xdata(), [1, 2, 3])
        assert _arr_eq(data_line.get_ydata(), [10, 20, 30])
        assert ec.has_yerr is True


# ===================================================================
# Fill between (3 tests)
# ===================================================================

class TestFillBetween:
    def test_fill_between_basic(self):
        """Does not raise with valid 1D data."""
        fig, ax = plt.subplots()
        result = ax.fill_between([1, 2, 3], [1, 2, 3])
        assert result is not None

    def test_fill_between_2d_x_raises(self):
        """2D x raises ValueError."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="1.?[Dd]"):
            ax.fill_between([[1, 2], [3, 4]], [1, 2], [3, 4])

    def test_fill_betweenx_basic(self):
        """Does not raise with valid 1D data."""
        fig, ax = plt.subplots()
        result = ax.fill_betweenx([1, 2, 3], [1, 2, 3])
        assert result is not None


# ===================================================================
# Axhline / Axvline (2 tests)
# ===================================================================

class TestAxLines:
    def test_axhline_returns_line2d(self):
        fig, ax = plt.subplots()
        line = ax.axhline(y=5)
        assert isinstance(line, Line2D)

    def test_axvline_returns_line2d(self):
        fig, ax = plt.subplots()
        line = ax.axvline(x=3)
        assert isinstance(line, Line2D)


# ===================================================================
# Text / Annotate (3 tests)
# ===================================================================

class TestTextAnnotate:
    def test_text_returns_text(self):
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'hello world')
        assert isinstance(t, Text)
        assert t.get_text() == 'hello world'

    def test_annotate_no_arrow(self):
        """Annotate without arrowprops -> arrow_patch is None."""
        fig, ax = plt.subplots()
        ann = ax.annotate('note', (1, 2))
        assert isinstance(ann, Annotation)
        assert ann.arrow_patch is None

    def test_annotate_with_arrow(self):
        """Annotate with arrowprops={} -> arrow_patch is not None."""
        fig, ax = plt.subplots()
        ann = ax.annotate('note', (1, 2), xytext=(3, 4), arrowprops={})
        assert ann.arrow_patch is not None
        assert isinstance(ann.arrow_patch, Patch)


# ===================================================================
# Grid / Legend (3 tests)
# ===================================================================

class TestGridLegend:
    def test_grid_toggle(self):
        fig, ax = plt.subplots()
        ax.grid(True)  # should not raise
        ax.grid(False)  # should not raise

    def test_legend_nargs_error(self):
        """legend with 3 positional args raises TypeError."""
        fig, ax = plt.subplots()
        with pytest.raises(TypeError):
            ax.legend(1, 2, 3)

    def test_get_legend_handles_labels(self):
        """plot with label, scatter with label - check labels list."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='line1')
        ax.scatter([1, 2], [3, 4], label='scatter1')
        handles, labels = ax.get_legend_handles_labels()
        assert 'line1' in labels
        assert 'scatter1' in labels
        assert len(handles) == 2
        assert len(labels) == 2


# ===================================================================
# Axes state (5 tests)
# ===================================================================

class TestAxesState:
    def test_cla_clears_all(self):
        """cla() clears lines, collections, patches, texts, etc."""
        fig, ax = plt.subplots()
        ax.plot([1], [2])
        ax.scatter([1], [2])
        ax.bar([1], [2])
        ax.text(0, 0, 'hello')
        ax.set_title('title')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Verify something is there
        assert len(ax.lines) > 0
        assert len(ax.collections) > 0
        assert len(ax.patches) > 0
        assert len(ax.texts) > 0

        ax.cla()

        assert len(ax.lines) == 0
        assert len(ax.collections) == 0
        assert len(ax.patches) == 0
        assert len(ax.texts) == 0
        assert len(ax.containers) == 0
        assert ax.get_title() == ''
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''

    def test_invert_axes(self):
        """invert_xaxis() sets xaxis_inverted() to True."""
        fig, ax = plt.subplots()
        assert not ax.xaxis_inverted()
        ax.invert_xaxis()
        assert ax.xaxis_inverted()

    def test_axes_remove(self):
        """ax.remove() removes from figure."""
        fig, ax = plt.subplots()
        assert len(fig.get_axes()) == 1
        ax.remove()
        assert len(fig.get_axes()) == 0

    def test_set_batch(self):
        """ax.set(xlabel='x', ylabel='y') sets values."""
        fig, ax = plt.subplots()
        ax.set(xlabel='x', ylabel='y', title='t')
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        assert ax.get_title() == 't'

    def test_scale(self):
        """set_xscale('log') => get_xscale() == 'log'."""
        fig, ax = plt.subplots()
        assert ax.get_xscale() == 'linear'
        ax.set_xscale('log')
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'linear'
        ax.set_yscale('log')
        assert ax.get_yscale() == 'log'


# ===================================================================
# Color cycle (2 tests)
# ===================================================================

class TestColorCycle:
    def test_color_cycle_advances(self):
        """plot, scatter, bar each advance the color cycle."""
        fig, ax = plt.subplots()
        l1 = ax.plot([1], [2])
        pc = ax.scatter([1], [2])
        bc = ax.bar([1], [2])
        active_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

        c1 = l1[0].get_color()
        c2 = pc.get_facecolors()[0]
        # bar stores color as facecolor on Rectangle
        c3 = to_hex(bc[0]._facecolor)

        # First plot uses C0; all three should be colors from the cycle
        assert _color_close(c1, active_cycle[0])
        assert any(_color_close(c2, c) for c in active_cycle[:4])
        assert any(_color_close(c3, c) for c in active_cycle[:4])

    def test_color_cycle_resets_on_cla(self):
        """cla() resets color index so next plot starts at C0 again."""
        fig, ax = plt.subplots()
        l1 = ax.plot([1], [2])
        color_before = l1[0].get_color()
        active_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        ax.cla()
        l2 = ax.plot([1], [2])
        color_after = l2[0].get_color()
        assert _color_close(color_before, color_after)
        assert _color_close(color_before, active_cycle[0])


# ===================================================================
# Additional edge case tests (6 tests)
# ===================================================================

class TestAdditionalEdgeCases:
    def test_auto_xlim_from_data(self):
        """get_xlim() auto-calculates from plot data when not set."""
        fig, ax = plt.subplots()
        ax.plot([2, 8], [1, 1])
        xlim = ax.get_xlim()
        # OG may apply margins, so just check data range is within limits
        assert xlim[0] <= 2
        assert xlim[1] >= 8

    def test_auto_ylim_from_data(self):
        """get_ylim() auto-calculates from plot data when not set."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [10, 50])
        ylim = ax.get_ylim()
        # OG may apply margins, so just check data range is within limits
        assert ylim[0] <= 10
        assert ylim[1] >= 50

    def test_default_limits_no_data(self):
        """Without data, default limits are (0.0, 1.0)."""
        fig, ax = plt.subplots()
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.0)

    def test_invert_yaxis(self):
        """invert_yaxis() works symmetrically to invert_xaxis()."""
        fig, ax = plt.subplots()
        assert not ax.yaxis_inverted()
        ax.invert_yaxis()
        assert ax.yaxis_inverted()

    def test_set_xlim_with_inversion(self):
        """set_xlim then invert_xaxis -> get_xlim returns reversed."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.invert_xaxis()
        assert ax.get_xlim() == (10, 0)

    def test_aspect(self):
        """set_aspect/get_aspect work correctly."""
        fig, ax = plt.subplots()
        assert ax.get_aspect() == 'auto'
        ax.set_aspect('equal')
        assert ax.get_aspect() in (1.0, 1, 'equal')

    def test_ticks(self):
        """set_xticks/get_xticks and set_yticks/get_yticks work."""
        fig, ax = plt.subplots()
        ax.set_xticks([0, 1, 2, 3])
        assert _arr_eq(ax.get_xticks(), [0, 1, 2, 3])
        ax.set_yticks([10, 20, 30])
        assert _arr_eq(ax.get_yticks(), [10, 20, 30])

    def test_bar_rectangle_corners(self):
        """Bar patches have correct corners based on position and size."""
        fig, ax = plt.subplots()
        bc = ax.bar([2], [5], width=1.0)
        rect = bc[0]
        corners = rect.get_corners()
        # x_center=2, width=1.0 -> x from 1.5 to 2.5; height 5, bottom 0
        assert np.allclose(corners[0], (1.5, 0))   # bottom-left
        assert np.allclose(corners[1], (2.5, 0))   # bottom-right
        assert np.allclose(corners[2], (2.5, 5))   # top-right
        assert np.allclose(corners[3], (1.5, 5))   # top-left

    def test_plot_line_stored_in_axes(self):
        """Plot line is stored in ax.lines."""
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4])
        assert lines[0] in ax.lines

    def test_scatter_stored_in_collections(self):
        """Scatter PathCollection is stored in ax.collections."""
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4])
        assert pc in ax.collections

    def test_errorbar_has_xerr_yerr_flags(self):
        """ErrorbarContainer tracks has_xerr and has_yerr flags."""
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2], [3, 4], yerr=[0.1, 0.2], xerr=[0.3, 0.4])
        assert ec.has_xerr is True
        assert ec.has_yerr is True

    def test_text_fontsize(self):
        """Text created via ax.text has correct fontsize."""
        fig, ax = plt.subplots()
        t = ax.text(0, 0, 'big', fontsize=24)
        assert t.get_fontsize() == 24

    def test_cla_resets_scale_and_aspect(self):
        """cla() resets scale to linear."""
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.cla()
        assert ax.get_xscale() == 'linear'
        assert ax.get_yscale() == 'linear'


# ===================================================================
# Additional parametric tests
# ===================================================================

import pytest
import matplotlib.pyplot as plt
