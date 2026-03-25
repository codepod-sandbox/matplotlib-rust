"""
Batch 3 of upstream-style tests.
Focus on integration tests, cross-feature tests, and edge cases
that match patterns from upstream matplotlib's test suite.
"""

import numpy as np
import pytest

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Patch, Rectangle, Circle, Polygon, Wedge,
    Ellipse, Arc, FancyBboxPatch, FancyArrowPatch,
)
from matplotlib.collections import (
    PathCollection, LineCollection, PolyCollection, EventCollection,
)
from matplotlib.text import Text, Annotation
from matplotlib.figure import Figure
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.colors import to_rgba, to_hex, Normalize


# ===========================================================================
# Upstream test_axes.py patterns
# ===========================================================================

class TestAxesPlotReturn:
    """Upstream pattern: verify return types and counts from plot methods."""

    def test_plot_returns_list(self):
        fig, ax = plt.subplots()
        result = ax.plot([1, 2, 3])
        assert isinstance(result, list)

    def test_plot_returns_line2d(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4])
        assert isinstance(lines[0], Line2D)

    def test_scatter_returns_pathcollection(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4])
        assert isinstance(pc, PathCollection)

    def test_bar_returns_barcontainer(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4])
        assert isinstance(bc, BarContainer)

    def test_barh_returns_barcontainer(self):
        fig, ax = plt.subplots()
        bc = ax.barh([1, 2], [3, 4])
        assert isinstance(bc, BarContainer)

    def test_hist_returns_tuple(self):
        fig, ax = plt.subplots()
        result = ax.hist([1, 2, 3, 4, 5])
        assert len(result) == 3

    def test_errorbar_returns_errorbarcontainer(self):
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2], [3, 4], yerr=0.1)
        assert isinstance(ec, ErrorbarContainer)

    def test_fill_between_returns_polygon(self):
        fig, ax = plt.subplots()
        poly = ax.fill_between([0, 1, 2], [0, 1, 0])
        assert isinstance(poly, Polygon)

    def test_stem_returns_stemcontainer(self):
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2, 3])
        assert isinstance(sc, StemContainer)

    def test_text_returns_text(self):
        fig, ax = plt.subplots()
        t = ax.text(0, 0, 'hello')
        assert isinstance(t, Text)

    def test_annotate_returns_annotation(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('test', (0, 0))
        assert isinstance(ann, Annotation)


class TestAxesLimitsIntegration:
    """Test auto limits with various plot types."""

    def test_plot_auto_xlim(self):
        fig, ax = plt.subplots()
        ax.plot([2, 8], [0, 0])
        xlim = ax.get_xlim()
        assert xlim == (2, 8)

    def test_plot_auto_ylim(self):
        fig, ax = plt.subplots()
        ax.plot([0, 0], [3, 7])
        ylim = ax.get_ylim()
        assert ylim == (3, 7)

    def test_scatter_auto_limits(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 5], [2, 8])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (1, 5)
        assert ylim == (2, 8)

    def test_manual_xlim_overrides(self):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.set_xlim(2, 8)
        assert ax.get_xlim() == (2, 8)

    def test_manual_ylim_overrides(self):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.set_ylim(-5, 15)
        assert ax.get_ylim() == (-5, 15)

    def test_inverted_xlim_after_plot(self):
        fig, ax = plt.subplots()
        ax.invert_xaxis()
        ax.plot([1, 5], [0, 0])
        xlim = ax.get_xlim()
        assert xlim[0] > xlim[1]

    def test_inverted_ylim_after_plot(self):
        fig, ax = plt.subplots()
        ax.invert_yaxis()
        ax.plot([0, 0], [1, 5])
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]

    def test_empty_axes_default_limits(self):
        fig, ax = plt.subplots()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (0.0, 1.0)
        assert ylim == (0.0, 1.0)


class TestAxesSharing:
    """Test shared axes behavior."""

    def test_sharex_propagates(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_xlim(0, 100)
        assert ax2.get_xlim() == (0, 100)

    def test_sharey_propagates(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.set_ylim(-10, 10)
        assert ax2.get_ylim() == (-10, 10)

    def test_twinx_shares_xlim(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlim(0, 50)
        assert ax2.get_xlim() == (0, 50)

    def test_twiny_shares_ylim(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.set_ylim(-20, 20)
        assert ax2.get_ylim() == (-20, 20)

    def test_cla_preserves_sharing(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.cla()
        ax2.set_xlim(0, 10)
        assert ax1.get_xlim() == (0, 10)


class TestAxesClear:
    """Test cla/clear behavior."""

    def test_cla_resets_title(self):
        fig, ax = plt.subplots()
        ax.set_title('Test')
        ax.cla()
        assert ax.get_title() == ''

    def test_cla_resets_labels(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.cla()
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''

    def test_cla_resets_limits(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.cla()
        # After clear, limits should be auto
        assert ax.get_xlim() == (0.0, 1.0)

    def test_cla_resets_color_cycle(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])  # uses C0
        ax.cla()
        ax.plot([1, 2], [3, 4])  # should use C0 again
        assert ax._color_idx == 1

    def test_cla_resets_grid(self):
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.cla()
        assert ax._grid is False

    def test_cla_resets_legend(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='test')
        ax.legend()
        ax.cla()
        assert ax._legend is False


class TestAxesLabelOuter:
    """Test label_outer() behavior."""

    def test_label_outer_bottom_left(self):
        fig, axes = plt.subplots(2, 2)
        axes[1][0].label_outer()
        # Bottom-left: both labels visible
        assert axes[1][0]._xticklabels_visible is True
        assert axes[1][0]._yticklabels_visible is True

    def test_label_outer_top_left(self):
        fig, axes = plt.subplots(2, 2)
        axes[0][0].label_outer()
        # Top-left: x hidden, y visible
        assert axes[0][0]._xticklabels_visible is False
        assert axes[0][0]._yticklabels_visible is True

    def test_label_outer_bottom_right(self):
        fig, axes = plt.subplots(2, 2)
        axes[1][1].label_outer()
        # Bottom-right: x visible, y hidden
        assert axes[1][1]._xticklabels_visible is True
        assert axes[1][1]._yticklabels_visible is False


class TestAxesBarLabel:
    """Test bar_label() behavior."""

    def test_bar_label_basic(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2, 3], [4, 5, 6])
        annotations = ax.bar_label(bc)
        assert len(annotations) == 3

    def test_bar_label_custom_labels(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4])
        annotations = ax.bar_label(bc, labels=['A', 'B'])
        assert annotations[0].get_text() == 'A'
        assert annotations[1].get_text() == 'B'

    def test_bar_label_center(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4])
        annotations = ax.bar_label(bc, label_type='center')
        assert len(annotations) == 2


# ===========================================================================
# Upstream test_lines.py patterns
# ===========================================================================

class TestLine2DUpstream:
    """Additional Line2D tests matching upstream patterns."""

    def test_line_contains_data(self):
        line = Line2D([1, 2, 3], [4, 5, 6])
        x, y = line.get_data()
        assert x == [1, 2, 3]
        assert y == [4, 5, 6]

    def test_set_data_replaces(self):
        line = Line2D([1, 2], [3, 4])
        line.set_data([10, 20], [30, 40])
        assert line.get_xdata() == [10, 20]
        assert line.get_ydata() == [30, 40]

    def test_line_color_validation(self):
        """Setting invalid color raises ValueError."""
        line = Line2D([], [])
        with pytest.raises(ValueError):
            line.set_color('notacolor')

    def test_line_set_zorder(self):
        line = Line2D([], [])
        line.set_zorder(10)
        assert line.get_zorder() == 10

    def test_line_set_visible(self):
        line = Line2D([], [])
        line.set_visible(False)
        assert line.get_visible() is False

    def test_line_set_alpha(self):
        line = Line2D([], [])
        line.set_alpha(0.3)
        assert line.get_alpha() == 0.3


# ===========================================================================
# Upstream test_patches.py patterns
# ===========================================================================

class TestPatchesUpstream:
    def test_rectangle_is_patch(self):
        r = Rectangle((0, 0), 1, 1)
        assert isinstance(r, Patch)

    def test_circle_is_patch(self):
        c = Circle((0, 0))
        assert isinstance(c, Patch)

    def test_polygon_is_patch(self):
        p = Polygon([(0, 0), (1, 0), (1, 1)])
        assert isinstance(p, Patch)

    def test_ellipse_is_patch(self):
        e = Ellipse((0, 0), 1, 1)
        assert isinstance(e, Patch)

    def test_wedge_is_patch(self):
        w = Wedge((0, 0), 1, 0, 90)
        assert isinstance(w, Patch)

    def test_fancy_bbox_is_patch(self):
        f = FancyBboxPatch((0, 0), 1, 1)
        assert isinstance(f, Patch)

    def test_fancy_arrow_is_patch(self):
        f = FancyArrowPatch()
        assert isinstance(f, Patch)

    def test_patch_default_linewidth(self):
        p = Patch()
        assert p.get_linewidth() == 1.0

    def test_patch_set_linewidth(self):
        p = Patch()
        p.set_linewidth(3.0)
        assert p.get_linewidth() == 3.0

    def test_patch_default_facecolor(self):
        p = Patch()
        fc = p.get_facecolor()
        # Default is C0
        assert len(fc) == 4

    def test_patch_facecolor_red(self):
        p = Patch(facecolor='red')
        fc = p.get_facecolor()
        assert fc[0] == 1.0
        assert fc[1] == 0.0
        assert fc[2] == 0.0

    def test_patch_edgecolor_blue(self):
        p = Patch(edgecolor='blue')
        ec = p.get_edgecolor()
        assert ec[0] == 0.0
        assert ec[1] == 0.0
        assert ec[2] == 1.0


# ===========================================================================
# Upstream test_text.py patterns
# ===========================================================================

class TestTextUpstream:
    def test_text_is_artist(self):
        t = Text()
        assert isinstance(t, Artist)

    def test_text_default_ha(self):
        t = Text()
        assert t.get_ha() == 'left'

    def test_text_default_va(self):
        t = Text()
        assert t.get_va() == 'baseline'

    def test_text_default_weight(self):
        t = Text()
        assert t.get_weight() == 'normal'

    def test_text_set_ha(self):
        t = Text()
        t.set_ha('center')
        assert t.get_ha() == 'center'

    def test_text_set_va(self):
        t = Text()
        t.set_va('top')
        assert t.get_va() == 'top'

    def test_annotation_is_text(self):
        ann = Annotation('test', (0, 0))
        assert isinstance(ann, Text)

    def test_annotation_text(self):
        ann = Annotation('hello', (0, 0))
        assert ann.get_text() == 'hello'


# ===========================================================================
# Upstream test_figure.py patterns
# ===========================================================================

class TestFigureUpstream:
    def test_figure_default_size(self):
        fig = Figure()
        assert fig.get_size_inches() == (6.4, 4.8)

    def test_figure_custom_size(self):
        fig = Figure(figsize=(10, 8))
        assert fig.get_size_inches() == (10, 8)

    def test_figure_add_axes(self):
        fig = Figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        assert len(fig.axes) == 1

    def test_figure_add_subplot(self):
        fig = Figure()
        ax = fig.add_subplot(2, 2, 1)
        assert len(fig.axes) == 1

    def test_figure_clear_removes_axes(self):
        fig = Figure()
        fig.add_subplot(1, 1, 1)
        fig.clear()
        assert len(fig.axes) == 0

    def test_figure_suptitle_text(self):
        fig = Figure()
        fig.suptitle('Hello')
        assert fig.get_suptitle() == 'Hello'

    def test_figure_no_suptitle(self):
        fig = Figure()
        assert fig.get_suptitle() == ''


# ===========================================================================
# Upstream test_collections.py patterns
# ===========================================================================

class TestCollectionsUpstream:
    def test_pathcollection_empty(self):
        pc = PathCollection()
        assert pc.get_offsets() == []

    def test_pathcollection_default_sizes(self):
        pc = PathCollection()
        assert pc.get_sizes() == [20.0]

    def test_pathcollection_set_offsets(self):
        pc = PathCollection()
        pc.set_offsets([(1, 2), (3, 4)])
        assert len(pc.get_offsets()) == 2

    def test_pathcollection_set_sizes(self):
        pc = PathCollection()
        pc.set_sizes([10, 20, 30])
        assert pc.get_sizes() == [10, 20, 30]

    def test_linecollection_empty(self):
        lc = LineCollection()
        assert lc.get_segments() == []

    def test_linecollection_segments(self):
        segs = [[(0, 0), (1, 1)], [(2, 2), (3, 3)]]
        lc = LineCollection(segs)
        assert len(lc.get_segments()) == 2

    def test_polycollection_empty(self):
        pc = PolyCollection()
        assert pc.get_verts() == []

    def test_eventcollection_positions(self):
        ec = EventCollection([1, 2, 3])
        assert ec.get_positions() == [1, 2, 3]


# ===========================================================================
# Integration: plot + modify + verify
# ===========================================================================

class TestIntegration:
    def test_plot_then_modify_line(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6])
        line = lines[0]
        line.set_color('red')
        line.set_linewidth(3.0)
        assert line.get_color() == 'red'
        assert line.get_linewidth() == 3.0

    def test_plot_then_modify_data(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6])
        line = lines[0]
        line.set_data([10, 20], [30, 40])
        assert line.get_xdata() == [10, 20]
        assert line.get_ydata() == [30, 40]

    def test_scatter_then_modify(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4])
        pc.set_sizes([50, 100])
        assert pc.get_sizes() == [50, 100]

    def test_bar_then_modify_rect(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4])
        bc[0].set_facecolor('red')
        fc = bc[0].get_facecolor()
        assert fc[0] == 1.0

    def test_multiple_plot_types(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.scatter([1, 2], [3, 4])
        ax.bar([5, 6], [7, 8])
        ax.text(0, 0, 'hello')
        assert len(ax.lines) == 1
        assert len(ax.collections) == 1
        assert len(ax.patches) == 2
        assert len(ax.texts) == 1

    def test_clear_then_replot(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.cla()
        ax.plot([5, 6], [7, 8])
        assert len(ax.lines) == 1
        assert ax.lines[0].get_xdata() == [5, 6]

    def test_remove_line(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4])
        line = lines[0]
        line.remove()
        assert len(ax.lines) == 0

    def test_remove_patch(self):
        fig, ax = plt.subplots()
        rect = Rectangle((0, 0), 1, 1)
        ax.add_patch(rect)
        assert rect in ax.patches
        rect.remove()
        assert rect not in ax.patches

    def test_remove_collection(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4])
        pc.remove()
        assert pc not in ax.collections

    def test_legend_with_labels(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='line1')
        ax.plot([1, 2], [5, 6], label='line2')
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ['line1', 'line2']

    def test_legend_skips_underscores(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='_hidden')
        ax.plot([1, 2], [5, 6], label='visible')
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ['visible']

    def test_subplots_grid_access(self):
        fig, axes = plt.subplots(2, 3)
        assert len(axes) == 2
        assert len(axes[0]) == 3
        axes[0][0].set_title('A')
        assert axes[0][0].get_title() == 'A'

    def test_figure_text(self):
        fig, ax = plt.subplots()
        txt = fig.text(0.5, 0.98, 'Hello')
        assert txt.get_text() == 'Hello'
        assert txt in fig.texts


# ===========================================================================
# Parametrized tests for broad coverage
# ===========================================================================

class TestParametrized:
    @pytest.mark.parametrize('fmt', [
        'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w',
        'r-', 'g--', 'b:', 'c-.',
        'ro', 'gs', 'b^', 'cv',
        'r-o', 'g--s', 'b:^',
    ])
    def test_plot_format_strings(self, fmt):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6], fmt)
        assert len(lines) == 1

    @pytest.mark.parametrize('marker', [
        'o', 's', '^', 'v', '<', '>', 'D', 'd',
        '+', 'x', '*', '.', 'p', 'h', 'H', '|', '_',
    ])
    def test_plot_markers(self, marker):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], marker=marker)
        assert lines[0].get_marker() == marker

    @pytest.mark.parametrize('ls', ['-', '--', '-.', ':', 'solid', 'dashed',
                                     'dashdot', 'dotted'])
    def test_plot_linestyles(self, ls):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], linestyle=ls)
        assert lines[0].get_linestyle() == ls

    @pytest.mark.parametrize('color', [
        'red', 'green', 'blue', 'cyan', 'magenta', 'yellow',
        'black', 'white', '#ff0000', '#00ff00', '#0000ff',
        'C0', 'C1', 'C2', 'tab:blue', 'tab:orange',
        (1.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5),
    ])
    def test_plot_colors(self, color):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], color=color)
        assert len(lines) == 1

    @pytest.mark.parametrize('where', ['pre', 'post', 'mid'])
    def test_step_where(self, where):
        fig, ax = plt.subplots()
        lines = ax.step([0, 1, 2, 3], [0, 1, 0, 1], where=where)
        assert lines is not None

    @pytest.mark.parametrize('baseline', ['zero', 'sym', 'wiggle', 'weighted_wiggle'])
    def test_stackplot_baselines(self, baseline):
        fig, ax = plt.subplots()
        polys = ax.stackplot([0, 1, 2], [1, 2, 3], [3, 2, 1],
                             baseline=baseline)
        assert len(polys) == 2

    @pytest.mark.parametrize('orientation', ['vertical', 'horizontal'])
    def test_stem_orientation(self, orientation):
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2, 3], orientation=orientation)
        assert sc is not None

    @pytest.mark.parametrize('nbins', [5, 10, 20, 50])
    def test_hist_bins(self, nbins):
        fig, ax = plt.subplots()
        data = list(range(100))
        counts, edges, bc = ax.hist(data, bins=nbins)
        assert len(counts) == nbins
        assert len(edges) == nbins + 1

    @pytest.mark.parametrize('vert', [True, False])
    def test_boxplot_vert(self, vert):
        fig, ax = plt.subplots()
        result = ax.boxplot([1, 2, 3, 4, 5], vert=vert)
        assert 'boxes' in result
        assert len(result['boxes']) == 1

    @pytest.mark.parametrize('vert', [True, False])
    def test_violinplot_vert(self, vert):
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5], vert=vert)
        assert 'bodies' in result


# ===========================================================================
# More upstream patterns - batch accessor tests
# ===========================================================================

class TestAccessorPatterns:
    """Upstream pattern: test get/set roundtrip for all properties."""

    def test_axes_title_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_title('Test')
        assert ax.get_title() == 'Test'

    def test_axes_xlabel_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X')
        assert ax.get_xlabel() == 'X'

    def test_axes_ylabel_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Y')
        assert ax.get_ylabel() == 'Y'

    def test_axes_xlim_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1, 10)
        assert ax.get_xlim() == (1, 10)

    def test_axes_ylim_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-5, 5)
        assert ax.get_ylim() == (-5, 5)

    def test_axes_xticks_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_xticks([1, 2, 3])
        assert ax.get_xticks() == [1, 2, 3]

    def test_axes_yticks_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_yticks([10, 20, 30])
        assert ax.get_yticks() == [10, 20, 30]

    def test_axes_xticklabels_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['a', 'b', 'c'])
        assert ax.get_xticklabels() == ['a', 'b', 'c']

    def test_axes_yticklabels_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['x', 'y', 'z'])
        assert ax.get_yticklabels() == ['x', 'y', 'z']

    def test_axes_xscale_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        assert ax.get_xscale() == 'log'

    def test_axes_yscale_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        assert ax.get_yscale() == 'log'

    def test_axes_aspect_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        assert ax.get_aspect() == 'equal'

    def test_axes_adjustable_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_adjustable('datalim')
        assert ax.get_adjustable() == 'datalim'

    def test_axes_anchor_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_anchor('NW')
        assert ax.get_anchor() == 'NW'

    def test_axes_facecolor_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_facecolor('lightblue')
        fc = ax.get_facecolor()
        assert fc[2] > 0.5  # blue-ish

    def test_axes_navigate_roundtrip(self):
        fig, ax = plt.subplots()
        ax.set_navigate(False)
        assert ax.get_navigate() is False

    def test_figure_size_roundtrip(self):
        fig = Figure(figsize=(8, 6))
        assert fig.get_size_inches() == (8, 6)
        fig.set_size_inches(10, 8)
        assert fig.get_size_inches() == (10, 8)

    def test_figure_dpi_roundtrip(self):
        fig = Figure(dpi=150)
        assert fig.get_dpi() == 150
        fig.set_dpi(200)
        assert fig.get_dpi() == 200

    def test_figure_label_roundtrip(self):
        fig = Figure()
        fig.set_label('test')
        assert fig.get_label() == 'test'

    def test_line_xdata_roundtrip(self):
        line = Line2D([1, 2], [3, 4])
        line.set_xdata([5, 6])
        assert line.get_xdata() == [5, 6]

    def test_line_ydata_roundtrip(self):
        line = Line2D([1, 2], [3, 4])
        line.set_ydata([5, 6])
        assert line.get_ydata() == [5, 6]

    def test_text_text_roundtrip(self):
        t = Text(text='hello')
        t.set_text('world')
        assert t.get_text() == 'world'

    def test_text_fontsize_roundtrip(self):
        t = Text(fontsize=12)
        t.set_fontsize(20)
        assert t.get_fontsize() == 20

    def test_text_rotation_roundtrip(self):
        t = Text(rotation=0)
        t.set_rotation(45)
        assert t.get_rotation() == 45

    def test_rectangle_width_roundtrip(self):
        r = Rectangle((0, 0), 5, 10)
        r.set_width(20)
        assert r.get_width() == 20

    def test_rectangle_height_roundtrip(self):
        r = Rectangle((0, 0), 5, 10)
        r.set_height(30)
        assert r.get_height() == 30

    def test_circle_radius_roundtrip(self):
        c = Circle((0, 0), radius=5)
        c.set_radius(10)
        assert c.get_radius() == 10

    def test_ellipse_width_roundtrip(self):
        e = Ellipse((0, 0), 5, 10)
        e.set_width(20)
        assert e.get_width() == 20

    def test_ellipse_height_roundtrip(self):
        e = Ellipse((0, 0), 5, 10)
        e.set_height(30)
        assert e.get_height() == 30

    def test_ellipse_angle_roundtrip(self):
        e = Ellipse((0, 0), 5, 10, angle=45)
        e.set_angle(90)
        assert e.get_angle() == 90


class TestBatch3Parametric14:
    """Yet more parametric tests for batch 3."""

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



class TestBatch3Parametric19:
    """Yet more parametric tests for batch 3."""

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

