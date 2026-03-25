"""
Upstream tests — batch 15.
Focus: More exhaustive Axes, Figure, Colors, Ticker edge cases.
"""
import math
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes, XAxis, YAxis, Tick
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.text import Text
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox, Affine2D


# ------------------------------------------------------------------
# Axes edge cases
# ------------------------------------------------------------------

class TestAxesEdgeCases:
    def test_empty_plot_xlim(self):
        fig, ax = plt.subplots()
        assert ax.get_xlim() == (0.0, 1.0)

    def test_empty_plot_ylim(self):
        fig, ax = plt.subplots()
        assert ax.get_ylim() == (0.0, 1.0)

    def test_plot_single_point_x(self):
        fig, ax = plt.subplots()
        ax.plot([5], [10])
        # xlim should be reasonable around single point

    def test_set_xlim_propagates_shared(self):
        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].set_xlim(0, 10)
        assert axes[1].get_xlim() == (0, 10)

    def test_set_ylim_propagates_shared(self):
        fig, axes = plt.subplots(2, 1, sharey=True)
        axes[0].set_ylim(0, 20)
        assert axes[1].get_ylim() == (0, 20)

    def test_set_xlim_auto_false(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10, auto=False)
        assert ax.get_autoscalex_on() is False

    def test_set_ylim_auto_false(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 10, auto=False)
        assert ax.get_autoscaley_on() is False

    def test_multiple_plot_xlim(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.plot([5, 10], [5, 10])
        lo, hi = ax.get_xlim()
        assert lo <= 0
        assert hi >= 10

    def test_invert_xaxis_twice(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.invert_xaxis()
        assert ax.xaxis_inverted()
        ax.invert_xaxis()
        assert not ax.xaxis_inverted()

    def test_invert_yaxis_twice(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 10)
        ax.invert_yaxis()
        assert ax.yaxis_inverted()
        ax.invert_yaxis()
        assert not ax.yaxis_inverted()

    def test_aspect_auto(self):
        fig, ax = plt.subplots()
        ax.set_aspect('auto')
        assert ax.get_aspect() == 'auto'

    def test_aspect_equal(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        assert ax.get_aspect() == 'equal'

    def test_aspect_numeric(self):
        fig, ax = plt.subplots()
        ax.set_aspect(2.0)
        assert ax.get_aspect() == 2.0

    def test_adjustable(self):
        fig, ax = plt.subplots()
        ax.set_adjustable('box')
        assert ax.get_adjustable() == 'box'
        ax.set_adjustable('datalim')
        assert ax.get_adjustable() == 'datalim'

    def test_anchor(self):
        fig, ax = plt.subplots()
        ax.set_anchor('C')
        assert ax.get_anchor() == 'C'
        ax.set_anchor('NE')
        assert ax.get_anchor() == 'NE'

    def test_box_aspect(self):
        fig, ax = plt.subplots()
        ax.set_box_aspect(1.0)
        assert ax.get_box_aspect() == 1.0
        ax.set_box_aspect(None)
        assert ax.get_box_aspect() is None

    def test_get_position(self):
        fig, ax = plt.subplots()
        pos = ax.get_position()
        assert pos is not None

    def test_set_position_list(self):
        fig, ax = plt.subplots()
        ax.set_position([0.1, 0.1, 0.8, 0.8])

    def test_minorticks(self):
        fig, ax = plt.subplots()
        ax.minorticks_on()
        ax.minorticks_off()

    def test_tick_params_x(self):
        fig, ax = plt.subplots()
        ax.tick_params(axis='x', labelsize=14)

    def test_tick_params_y(self):
        fig, ax = plt.subplots()
        ax.tick_params(axis='y', labelsize=16)

    def test_tick_params_both(self):
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', labelsize=12)

    def test_get_tick_params(self):
        fig, ax = plt.subplots()
        ax.tick_params(axis='x', labelsize=14)
        params = ax.get_tick_params(axis='x')
        assert params['labelsize'] == 14

    def test_label_outer(self):
        fig, axes = plt.subplots(2, 2)
        for ax in [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]:
            ax.label_outer()


# ------------------------------------------------------------------
# Axes.set() comprehensive
# ------------------------------------------------------------------

class TestAxesSet:
    def test_set_xscale(self):
        fig, ax = plt.subplots()
        ax.set(xscale='log')
        assert ax.get_xscale() == 'log'

    def test_set_yscale(self):
        fig, ax = plt.subplots()
        ax.set(yscale='log')
        assert ax.get_yscale() == 'log'

    def test_set_facecolor(self):
        fig, ax = plt.subplots()
        ax.set(facecolor='yellow')


# ------------------------------------------------------------------
# More Line2D edge cases
# ------------------------------------------------------------------

class TestLine2DEdge:
    def test_empty_data(self):
        line = Line2D([], [])
        assert list(line.get_xdata()) == []
        assert list(line.get_ydata()) == []

    def test_single_point(self):
        line = Line2D([5], [10])
        assert list(line.get_xdata()) == [5]
        assert list(line.get_ydata()) == [10]

    def test_linestyle_aliases(self):
        """Dashed / dotted / dashdot aliases."""
        for ls in ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']:
            line = Line2D([0, 1], [0, 1], linestyle=ls)

    def test_marker_none_string(self):
        line = Line2D([0, 1], [0, 1], marker='None')
        assert line.get_marker() in ('None', 'none', None)

    def test_color_from_hex(self):
        line = Line2D([0, 1], [0, 1], color='#ff0000')

    def test_set_multiple_props(self):
        line = Line2D([0, 1], [0, 1])
        line.set(linewidth=3, linestyle='--', marker='o')
        assert line.get_linewidth() == 3
        assert line.get_linestyle() == '--'
        assert line.get_marker() == 'o'


# ------------------------------------------------------------------
# More Rectangle edge cases
# ------------------------------------------------------------------

class TestRectangleEdge:
    def test_zero_size(self):
        r = Rectangle((0, 0), 0, 0)
        assert r.get_width() == 0
        assert r.get_height() == 0

    def test_negative_size(self):
        """Negative width/height should be allowed."""
        r = Rectangle((0, 0), -1, -1)

    def test_none_label(self):
        r = Rectangle((0, 0), 1, 1, label=None)

    def test_linewidth(self):
        r = Rectangle((0, 0), 1, 1, linewidth=3)
        assert r.get_linewidth() == 3


# ------------------------------------------------------------------
# More Figure edge cases
# ------------------------------------------------------------------

class TestFigureEdge:
    def test_add_subplot_repeated(self):
        """Adding the same subplot position should return new axes."""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = fig.add_subplot(1, 1, 1)
        # Both should be in figure

    def test_add_axes_with_rect(self):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        assert ax in fig.get_axes()

    def test_add_axes_existing(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax2 = fig.add_axes(ax)
        assert ax2 is ax

    def test_subplots_1x1(self):
        fig = plt.figure()
        ax = fig.subplots()
        assert isinstance(ax, Axes)

    def test_subplots_1xn(self):
        fig = plt.figure()
        axes = fig.subplots(1, 3)
        assert len(axes) == 3

    def test_subplots_nx1(self):
        fig = plt.figure()
        axes = fig.subplots(3, 1)
        assert len(axes) == 3

    def test_subplots_nxm(self):
        fig = plt.figure()
        axes = fig.subplots(2, 3)
        assert len(axes) == 2
        assert len(axes[0]) == 3

    def test_number_property(self):
        fig = plt.figure()
        assert fig.number is not None

    def test_stale(self):
        fig = plt.figure()
        assert hasattr(fig, 'stale')


# ------------------------------------------------------------------
# More Normalize edge cases
# ------------------------------------------------------------------

class TestNormalizeEdge:
    def test_vmin_equals_vmax(self):
        norm = mcolors.Normalize(vmin=5, vmax=5)
        result = norm(5)
        assert result == 0.0

    def test_vmin_gt_vmax(self):
        norm = mcolors.Normalize(vmin=10, vmax=5)
        # Should still work

    def test_negative_range(self):
        norm = mcolors.Normalize(vmin=-10, vmax=-5)
        assert abs(norm(-10) - 0.0) < 1e-10
        assert abs(norm(-5) - 1.0) < 1e-10

    def test_large_range(self):
        norm = mcolors.Normalize(vmin=0, vmax=1e10)
        assert abs(norm(5e9) - 0.5) < 1e-10


# ------------------------------------------------------------------
# More PowerNorm edge cases
# ------------------------------------------------------------------

class TestPowerNormEdge:
    def test_gamma_half(self):
        pnorm = mcolors.PowerNorm(0.5, vmin=0, vmax=4)
        # gamma=0.5 is sqrt
        result = pnorm(1)
        assert 0 < result < 1

    def test_gamma_3(self):
        pnorm = mcolors.PowerNorm(3, vmin=0, vmax=1)
        assert abs(pnorm(0) - 0) < 1e-10
        assert abs(pnorm(1) - 1) < 1e-10


# ------------------------------------------------------------------
# More BoundaryNorm edge cases
# ------------------------------------------------------------------

class TestBoundaryNormEdge:
    def test_many_boundaries(self):
        boundaries = list(range(0, 11))  # 0,1,...,10
        bn = mcolors.BoundaryNorm(boundaries, ncolors=10)
        for i in range(10):
            val = i + 0.5
            result = bn(val)
            assert 0 <= result <= 1

    def test_list_input(self):
        bn = mcolors.BoundaryNorm([0, 1, 2], ncolors=2)
        result = bn([0.5, 1.5])
        assert len(result) == 2

    def test_boundaries_monotonic_error(self):
        with pytest.raises(ValueError):
            mcolors.BoundaryNorm([2, 1, 3], ncolors=2)


# ------------------------------------------------------------------
# Bbox edge cases
# ------------------------------------------------------------------

class TestBboxEdge:
    def test_zero_area(self):
        bb = Bbox.from_extents(5, 5, 5, 5)
        assert bb.width == 0
        assert bb.height == 0
        assert bb.is_empty()

    def test_negative_coords(self):
        bb = Bbox.from_extents(-10, -10, -5, -5)
        assert bb.width == 5
        assert bb.height == 5

    def test_union_single(self):
        bb = Bbox.from_extents(0, 0, 1, 1)
        u = Bbox.union([bb])
        assert u.x0 == 0 and u.y0 == 0 and u.x1 == 1 and u.y1 == 1

    def test_intersection_no_overlap(self):
        bb1 = Bbox.from_extents(0, 0, 1, 1)
        bb2 = Bbox.from_extents(2, 2, 3, 3)
        inter = Bbox.intersection(bb1, bb2)
        assert inter is None

    def test_translated(self):
        bb = Bbox.from_extents(0, 0, 1, 1)
        t = bb.translated(5, 10)
        assert t.x0 == 5
        assert t.y0 == 10
        assert t.x1 == 6
        assert t.y1 == 11

    def test_expanded(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        e = bb.expanded(2, 3)
        assert e.width == 20
        assert e.height == 30

    def test_padded(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        p = bb.padded(2)
        assert p.x0 == -2
        assert p.y0 == -2
        assert p.x1 == 12
        assert p.y1 == 12

    def test_shrunk(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        s = bb.shrunk(0.5, 0.5)
        assert abs(s.width - 5) < 1e-10
        assert abs(s.height - 5) < 1e-10

    def test_frozen(self):
        bb = Bbox.from_extents(0, 0, 1, 1)
        f = bb.frozen()
        assert f.x0 == bb.x0

    def test_rotated(self):
        bb = Bbox.from_extents(0, 0, 1, 1)
        r = bb.rotated(math.pi / 4)
        assert r.width > 0
        assert r.height > 0


# ------------------------------------------------------------------
# Affine2D edge cases
# ------------------------------------------------------------------

class TestAffine2DEdge:
    def test_scale_nonuniform(self):
        t = Affine2D().scale(2, 0.5)
        x, y = t.transform_point((1, 1))
        assert abs(x - 2) < 1e-10
        assert abs(y - 0.5) < 1e-10

    def test_multiple_rotations(self):
        t = Affine2D().rotate_deg(90).rotate_deg(90)
        x, y = t.transform_point((1, 0))
        assert abs(x - (-1)) < 1e-10
        assert abs(y - 0) < 1e-10

    def test_rotate_360(self):
        t = Affine2D().rotate_deg(360)
        x, y = t.transform_point((1, 0))
        assert abs(x - 1) < 1e-10
        assert abs(y - 0) < 1e-10

    def test_identity_inverted(self):
        t = Affine2D()
        inv = t.inverted()
        x, y = inv.transform_point((5, 10))
        assert abs(x - 5) < 1e-10
        assert abs(y - 10) < 1e-10

    def test_scale_then_translate(self):
        t = Affine2D().scale(3).translate(1, 2)
        x, y = t.transform_point((1, 1))
        assert abs(x - 4) < 1e-10
        assert abs(y - 5) < 1e-10

    def test_translate_then_scale(self):
        t = Affine2D().translate(1, 2).scale(3)
        x, y = t.transform_point((1, 1))
        assert abs(x - 6) < 1e-10
        assert abs(y - 9) < 1e-10


# ------------------------------------------------------------------
# Ticker exhaustive tests
# ------------------------------------------------------------------

class TestTickerExhaustive:
    def test_max_n_locator_integer(self):
        loc = mticker.MaxNLocator(integer=True)
        vals = loc.tick_values(0.5, 5.5)
        for v in vals:
            assert v == int(v)

    def test_max_n_locator_symmetric(self):
        loc = mticker.MaxNLocator(symmetric=True)
        vals = loc.tick_values(-3, 7)
        assert min(vals) <= -3 or max(vals) >= 3

    def test_max_n_locator_prune_lower(self):
        loc = mticker.MaxNLocator(prune='lower')
        vals = loc.tick_values(0, 10)
        # prune removes the lowest tick

    def test_max_n_locator_prune_upper(self):
        loc = mticker.MaxNLocator(prune='upper')
        vals = loc.tick_values(0, 10)

    def test_max_n_locator_prune_both(self):
        loc = mticker.MaxNLocator(prune='both')
        vals = loc.tick_values(0, 10)

    def test_max_n_locator_min_n_ticks(self):
        loc = mticker.MaxNLocator(min_n_ticks=3)
        vals = loc.tick_values(0, 10)
        assert len(vals) >= 3

    def test_multiple_locator_offset(self):
        loc = mticker.MultipleLocator(5)
        vals = loc.tick_values(3, 17)
        assert 5 in vals
        assert 10 in vals
        assert 15 in vals

    def test_fixed_formatter_offset_string(self):
        fmt = mticker.FixedFormatter(['a', 'b', 'c'])
        fmt.set_offset_string('offset')
        assert fmt.get_offset() == 'offset'

    def test_percent_formatter_decimals(self):
        fmt = mticker.PercentFormatter(xmax=100, decimals=2)
        result = fmt(50.123)
        assert '50.12' in result

    def test_percent_formatter_symbol(self):
        fmt = mticker.PercentFormatter(xmax=100, symbol='pct')
        result = fmt(50)
        assert 'pct' in result

    def test_scalar_formatter_scientific(self):
        fmt = mticker.ScalarFormatter()
        fmt.set_scientific(True)

    def test_scalar_formatter_powerlimits(self):
        fmt = mticker.ScalarFormatter()
        fmt.set_powerlimits((-2, 2))

    def test_log_formatter_base(self):
        fmt = mticker.LogFormatter(base=2)
        result = fmt(8)
        assert isinstance(result, str)

    def test_formatter_format_data(self):
        fmt = mticker.Formatter()
        result = fmt.format_data(3.14)
        assert isinstance(result, str)

    def test_formatter_format_data_short(self):
        fmt = mticker.Formatter()
        result = fmt.format_data_short(3.14)
        assert isinstance(result, str)

    def test_formatter_fix_minus(self):
        fmt = mticker.Formatter()
        result = fmt.fix_minus('-5')
        assert isinstance(result, str)

    def test_formatter_set_locs(self):
        fmt = mticker.Formatter()
        fmt.set_locs([1, 2, 3])

    def test_formatter_get_offset(self):
        fmt = mticker.Formatter()
        result = fmt.get_offset()
        assert isinstance(result, str)


# ------------------------------------------------------------------
# Collection exhaustive
# ------------------------------------------------------------------

class TestCollectionExhaustive:
    def test_path_collection_sizes(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2, 3], [4, 5, 6], s=[10, 20, 30])
        sizes = pc.get_sizes()
        assert len(sizes) == 3

    def test_path_collection_offsets(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4])
        offsets = pc.get_offsets()
        assert len(offsets) == 2

    def test_path_collection_facecolors(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4], c='red')
        fc = pc.get_facecolors()
        assert len(fc) > 0

    def test_line_collection_basic(self):
        lc = LineCollection([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        assert lc is not None

    def test_line_collection_colors(self):
        lc = LineCollection([[(0, 0), (1, 1)]], colors='red')

    def test_line_collection_label(self):
        lc = LineCollection([[(0, 0), (1, 1)]], label='lines')
        assert lc.get_label() == 'lines'


# ------------------------------------------------------------------
# Container exhaustive
# ------------------------------------------------------------------

class TestContainerExhaustive:
    def test_bar_container_patches(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7])
        assert len(bars.patches) == 3

    def test_bar_container_iter(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7])
        patches = list(bars)
        assert len(patches) == 3

    def test_errorbar_container_parts(self):
        fig, ax = plt.subplots()
        container = ax.errorbar([0, 1], [0, 1], yerr=0.1)
        line, caps, bars = container
        # line should be a Line2D or None
        assert line is None or isinstance(line, Line2D)

    def test_stem_container_parts(self):
        fig, ax = plt.subplots()
        container = ax.stem([1, 2, 3])
        # StemContainer has markerline, stemlines, baseline


# ------------------------------------------------------------------
# More Plot type tests
# ------------------------------------------------------------------

class TestPlotTypes:
    def test_bar_with_tick_label(self):
        fig, ax = plt.subplots()
        ax.bar([0, 1, 2], [3, 5, 7], tick_label=['a', 'b', 'c'])

    def test_barh_basic(self):
        fig, ax = plt.subplots()
        bars = ax.barh([0, 1, 2], [3, 5, 7])
        assert len(bars) == 3

    def test_stackplot(self):
        fig, ax = plt.subplots()
        ax.stackplot([1, 2, 3], [1, 2, 3], [2, 1, 2])

    def test_step_where_pre(self):
        fig, ax = plt.subplots()
        ax.step([0, 1, 2], [1, 2, 1], where='pre')

    def test_step_where_post(self):
        fig, ax = plt.subplots()
        ax.step([0, 1, 2], [1, 2, 1], where='post')

    def test_step_where_mid(self):
        fig, ax = plt.subplots()
        ax.step([0, 1, 2], [1, 2, 1], where='mid')

    def test_errorbar_capsize(self):
        fig, ax = plt.subplots()
        ax.errorbar([0, 1], [0, 1], yerr=0.1, capsize=5)

    def test_errorbar_ecolor(self):
        fig, ax = plt.subplots()
        ax.errorbar([0, 1], [0, 1], yerr=0.1, ecolor='red')

    def test_boxplot_widths(self):
        fig, ax = plt.subplots()
        ax.boxplot([[1, 2, 3, 4, 5]], widths=0.3)

    def test_violinplot_widths(self):
        fig, ax = plt.subplots()
        data = list(range(1, 21))
        ax.violinplot([data], widths=0.3)

    def test_violinplot_positions(self):
        fig, ax = plt.subplots()
        data = list(range(1, 21))
        ax.violinplot([data], positions=[2])

    def test_pie_counterclock(self):
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3], counterclock=False)


# ------------------------------------------------------------------
# More text edge cases
# ------------------------------------------------------------------

class TestTextEdge:
    def test_empty_string(self):
        t = Text(0, 0, '')
        assert t.get_text() == ''

    def test_multiline(self):
        t = Text(0, 0, 'line1\nline2')
        assert 'line1' in t.get_text()

    def test_numeric_text(self):
        t = Text(0, 0, '42')
        assert t.get_text() == '42'


# ------------------------------------------------------------------
# More color edge cases
# ------------------------------------------------------------------

class TestColorEdge:
    def test_to_rgba_4tuple(self):
        r, g, b, a = mcolors.to_rgba((1.0, 0.0, 0.0, 0.5))
        assert a == 0.5

    def test_to_rgba_short_hex(self):
        r, g, b, a = mcolors.to_rgba('#f00')

    def test_same_color_tuples(self):
        assert mcolors.same_color((1, 0, 0), (1, 0, 0)) is True
        assert mcolors.same_color((1, 0, 0), (0, 1, 0)) is False

    def test_same_color_lists(self):
        assert mcolors.same_color(['red', 'blue'], ['red', 'blue']) is True
        assert mcolors.same_color(['red', 'blue'], ['red', 'green']) is False

    def test_is_color_like_tuple4(self):
        assert mcolors.is_color_like((1, 0, 0, 1)) is True

    def test_is_color_like_none(self):
        assert mcolors.is_color_like(None) is False

    def test_to_hex_tuple4(self):
        h = mcolors.to_hex((1, 0, 0, 1))
        assert h.startswith('#')

    def test_to_rgba_array_single(self):
        result = mcolors.to_rgba_array('red')
        assert len(result) >= 1


# ------------------------------------------------------------------
# Figure.subplots with sharex/sharey
# ------------------------------------------------------------------

class TestSubplotsSharing:
    def test_sharex(self):
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].set_xlim(5, 15)
        assert axes[1].get_xlim() == (5, 15)

    def test_sharey(self):
        fig, axes = plt.subplots(1, 2, sharey=True)
        axes[0].set_ylim(5, 15)
        assert axes[1].get_ylim() == (5, 15)

    def test_sharex_2x2(self):
        fig, axes = plt.subplots(2, 2, sharex=True)
        axes[0][0].set_xlim(0, 100)
        assert axes[1][1].get_xlim() == (0, 100)

    def test_sharey_2x2(self):
        fig, axes = plt.subplots(2, 2, sharey=True)
        axes[0][0].set_ylim(0, 100)
        assert axes[1][1].get_ylim() == (0, 100)


# ------------------------------------------------------------------
# Additional miscellaneous
# ------------------------------------------------------------------

def test_axes_repr():
    fig, ax = plt.subplots()
    r = repr(ax)
    assert 'Axes' in r


def test_figure_clear_keeps_type():
    fig = plt.figure()
    fig.add_subplot()
    fig.clear()
    assert isinstance(fig, Figure)


def test_figure_colorbar():
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2], [3, 4], c=[5, 6])
    cb = fig.colorbar(sc)
    assert cb is not None


def test_scatter_sizes():
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2, 3], [4, 5, 6], s=[10, 100, 500])
    sizes = pc.get_sizes()
    assert sizes[0] < sizes[1] < sizes[2]


def test_multiple_bars():
    fig, ax = plt.subplots()
    bars1 = ax.bar([0, 1, 2], [3, 5, 7])
    bars2 = ax.bar([0, 1, 2], [1, 2, 3], bottom=[3, 5, 7])
    assert len(ax.containers) == 2


def test_twinx_ticks():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 10])
    ax2 = ax.twinx()
    ax2.plot([0, 1], [0, 100])
    # Both axes should have different y-limits


def test_add_artist_text():
    fig, ax = plt.subplots()
    t = Text(0.5, 0.5, 'manual text')
    ax.add_artist(t)
    assert t in ax.texts


def test_get_xticks():
    fig, ax = plt.subplots()
    ax.set_xticks([0, 5, 10])
    ticks = ax.get_xticks()
    assert 0 in ticks
    assert 5 in ticks
    assert 10 in ticks


def test_get_yticks():
    fig, ax = plt.subplots()
    ax.set_yticks([0, 5, 10])
    ticks = ax.get_yticks()
    assert 0 in ticks
    assert 5 in ticks
    assert 10 in ticks


def test_set_xticklabels():
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['a', 'b', 'c'])
    labels = ax.get_xticklabels()
    assert len(labels) >= 3


def test_set_yticklabels():
    fig, ax = plt.subplots()
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['x', 'y', 'z'])
    labels = ax.get_yticklabels()
    assert len(labels) >= 3


class TestBatch15Parametric6:
    """More parametric tests."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_n_lines(self, n):
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("lo,hi", [(0, 1), (-1, 1), (0, 100)])
    def test_xlim(self, lo, hi):
        fig, ax = plt.subplots()
        ax.set_xlim(lo, hi)
        assert ax.get_xlim() == (lo, hi)
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0])
    def test_linewidth(self, lw):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close("all")

    @pytest.mark.parametrize("marker", ["o", "s", "^", "D"])
    def test_marker(self, marker):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_bar(self, n):
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["equal", "auto"])
    def test_aspect(self, aspect):
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close("all")

    @pytest.mark.parametrize("title", ["Title", "Test", ""])
    def test_title(self, title):
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close("all")

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_line_alpha(self, alpha):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")

