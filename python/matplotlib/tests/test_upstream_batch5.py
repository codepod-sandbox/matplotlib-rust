"""Upstream-style test batch 5: comprehensive edge cases and additional coverage."""

import math
import pytest
from matplotlib.tests._approx import approx

from matplotlib.figure import Figure
from matplotlib.colors import (
    Normalize, LogNorm, TwoSlopeNorm, BoundaryNorm,
    PowerNorm, SymLogNorm, NoNorm,
    to_rgba, to_hex, to_rgb, is_color_like, same_color,
    to_rgba_array, _has_alpha_channel, parse_fmt,
    CSS4_COLORS, TABLEAU_COLORS, BASE_COLORS, DEFAULT_CYCLE,
)
from matplotlib.cm import (
    ListedColormap, LinearSegmentedColormap, get_cmap,
    ScalarMappable, ColormapRegistry, _colormaps,
)
from matplotlib.ticker import (
    NullLocator, FixedLocator, MaxNLocator, AutoLocator,
    MultipleLocator, LogLocator, LinearLocator, IndexLocator,
    AutoMinorLocator, SymmetricalLogLocator,
    NullFormatter, FixedFormatter, FuncFormatter, FormatStrFormatter,
    StrMethodFormatter, ScalarFormatter, PercentFormatter, LogFormatter,
)
from matplotlib.text import Text, Annotation
from matplotlib.cycler import cycler, Cycler


# ===================================================================
# Color conversion edge cases
# ===================================================================

class TestColorConversionEdgeCases:
    def test_to_rgba_short_hex(self):
        assert to_rgba('#f00') == approx((1.0, 0.0, 0.0, 1.0))

    def test_to_rgba_with_alpha_hex(self):
        r, g, b, a = to_rgba('#ff000080')
        assert r == approx(1.0)
        assert a == approx(128/255, abs=0.01)

    def test_to_rgba_grayscale_0(self):
        assert to_rgba('0') == (0.0, 0.0, 0.0, 1.0)

    def test_to_rgba_grayscale_1(self):
        assert to_rgba('1') == (1.0, 1.0, 1.0, 1.0)

    def test_to_rgba_grayscale_half(self):
        r, g, b, a = to_rgba('0.5')
        assert r == approx(0.5)
        assert g == approx(0.5)

    def test_to_rgba_cn_cycle(self):
        for i in range(10):
            rgba = to_rgba(f'C{i}')
            assert len(rgba) == 4

    def test_to_rgba_none_transparent(self):
        assert to_rgba('none') == (0.0, 0.0, 0.0, 0.0)

    def test_to_rgba_alpha_override(self):
        r, g, b, a = to_rgba('red', alpha=0.5)
        assert a == 0.5

    def test_to_rgba_invalid_raises(self):
        with pytest.raises(ValueError):
            to_rgba('not_a_color')

    def test_to_hex_basic(self):
        assert to_hex('red') == '#ff0000'

    def test_to_hex_keep_alpha(self):
        h = to_hex((1, 0, 0, 0.5), keep_alpha=True)
        assert len(h) == 9  # #rrggbbaa

    def test_to_rgb_basic(self):
        assert to_rgb('blue') == approx((0, 0, 1))

    def test_is_color_like_valid(self):
        assert is_color_like('red')
        assert is_color_like('#ff0000')
        assert is_color_like((1, 0, 0))
        assert is_color_like((1, 0, 0, 0.5))
        assert is_color_like('C0')

    def test_is_color_like_invalid(self):
        assert not is_color_like('not_color_xyz')
        assert not is_color_like((1, 2, 3, 4, 5))

    def test_same_color_basic(self):
        assert same_color('red', '#ff0000')
        assert same_color('r', 'red')

    def test_same_color_lists(self):
        assert same_color(['red', 'blue'], ['red', 'blue'])

    def test_to_rgba_array_single(self):
        result = to_rgba_array('red')
        assert len(result) == 1

    def test_to_rgba_array_list(self):
        result = to_rgba_array(['red', 'blue', 'green'])
        assert len(result) == 3

    def test_to_rgba_array_with_alpha_list(self):
        result = to_rgba_array(['red', 'blue'], alpha=[0.5, 0.8])
        assert result[0][3] == approx(0.5)
        assert result[1][3] == approx(0.8)

    def test_has_alpha_hex8(self):
        assert _has_alpha_channel('#ff000080')

    def test_has_alpha_hex6(self):
        assert not _has_alpha_channel('#ff0000')

    def test_has_alpha_rgba_tuple(self):
        assert _has_alpha_channel((1, 0, 0, 0.5))

    def test_has_alpha_rgb_tuple(self):
        assert not _has_alpha_channel((1, 0, 0))


# ===================================================================
# Parse format string tests
# ===================================================================

class TestParseFmt:
    def test_empty(self):
        c, m, ls = parse_fmt('')
        assert c is None
        assert m is None
        assert ls is None

    def test_color_only(self):
        c, m, ls = parse_fmt('r')
        assert c == 'r'

    def test_marker_only(self):
        c, m, ls = parse_fmt('o')
        assert m == 'o'

    def test_line_only(self):
        c, m, ls = parse_fmt('-')
        assert ls == '-'

    def test_dashed(self):
        c, m, ls = parse_fmt('--')
        assert ls == '--'

    def test_dashdot(self):
        c, m, ls = parse_fmt('-.')
        assert ls == '-.'

    def test_dotted(self):
        c, m, ls = parse_fmt(':')
        assert ls == ':'

    def test_full_fmt(self):
        c, m, ls = parse_fmt('ro-')
        assert c == 'r'
        assert m == 'o'
        assert ls == '-'

    def test_color_linestyle(self):
        c, m, ls = parse_fmt('b--')
        assert c == 'b'
        assert ls == '--'

    def test_color_marker(self):
        c, m, ls = parse_fmt('gs')
        assert c == 'g'
        assert m == 's'


# ===================================================================
# Colormap detailed tests
# ===================================================================

class TestColormapDetailed:
    def test_viridis_is_not_grayscale(self):
        cmap = get_cmap('viridis')
        rgba = cmap(0.5)
        # viridis at 0.5 should not have equal RGB
        assert not (rgba[0] == rgba[1] == rgba[2])

    def test_gray_is_grayscale(self):
        cmap = get_cmap('gray')
        rgba = cmap(0.5)
        assert rgba[0] == approx(rgba[1], abs=0.05)
        assert rgba[1] == approx(rgba[2], abs=0.05)

    def test_listed_cmap_wraps_at_boundaries(self):
        cmap = ListedColormap(['red', 'blue'])
        # Values exactly at 0 and 1 should work
        assert cmap(0.0) is not None
        assert cmap(1.0) is not None

    def test_linear_segmented_from_list_many_colors(self):
        import random
        colors = [(random.random(), random.random(), random.random())
                  for _ in range(20)]
        cmap = LinearSegmentedColormap.from_list('random', colors)
        for x in [0, 0.25, 0.5, 0.75, 1.0]:
            rgba = cmap(x)
            assert len(rgba) == 4
            for c in rgba:
                assert 0 <= c <= 1

    def test_cmap_resampled_preserves_endpoints(self):
        cmap = get_cmap('viridis')
        resampled = cmap.resampled(10)
        # Endpoints should be similar
        orig_0 = cmap(0.0)
        resamp_0 = resampled(0.0)
        assert orig_0[0] == approx(resamp_0[0], abs=0.15)

    def test_all_builtin_cmaps_have_reversed(self):
        for name in ['viridis', 'jet', 'hot', 'cool', 'gray',
                      'plasma', 'inferno', 'magma', 'cividis']:
            assert name + '_r' in _colormaps

    def test_listed_cmap_copy(self):
        cmap = ListedColormap(['red', 'blue'], name='copytest')
        copied = cmap.copy()
        assert copied.name == 'copytest'


# ===================================================================
# Norm edge cases
# ===================================================================

class TestNormEdgeCases:
    def test_normalize_same_vmin_vmax(self):
        norm = Normalize(5, 5)
        assert norm(5) == 0.0

    def test_normalize_callbacks(self):
        norm = Normalize()
        called = []
        norm.callbacks.connect('changed', lambda: called.append(True))
        norm.callbacks.process('changed')
        assert len(called) == 1

    def test_normalize_callback_disconnect(self):
        norm = Normalize()
        called = []
        cid = norm.callbacks.connect('changed', lambda: called.append(True))
        norm.callbacks.disconnect(cid)
        norm.callbacks.process('changed')
        assert len(called) == 0

    def test_lognorm_zero_returns_0(self):
        norm = LogNorm(1, 100)
        assert norm(0) == 0.0

    def test_lognorm_vmin_zero_raises(self):
        norm = LogNorm(0, 100)
        with pytest.raises(ValueError, match='vmin > 0'):
            norm(50)

    def test_twoslope_asymmetric_ranges(self):
        # vmin much closer to vcenter than vmax
        norm = TwoSlopeNorm(0, vmin=-1, vmax=100)
        assert norm(0) == approx(0.5)
        assert norm(-1) == approx(0.0)
        assert norm(100) == approx(1.0)
        # -0.5 should map to 0.25
        assert norm(-0.5) == approx(0.25)

    def test_boundary_norm_exact_boundary(self):
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
        # Values at exact boundaries
        result0 = norm(0)
        result1 = norm(1)
        result2 = norm(2)
        assert isinstance(result0, float)
        assert isinstance(result1, float)
        assert isinstance(result2, float)

    def test_power_norm_gamma_small(self):
        norm = PowerNorm(gamma=0.1, vmin=0, vmax=1)
        result = norm(0.01)
        # With small gamma, values should be mapped higher
        assert result > 0.5

    def test_symlog_zero_maps_to_midpoint(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        assert norm(0) == approx(0.5, abs=0.01)

    def test_nonorm_large_values(self):
        norm = NoNorm()
        assert norm(1e10) == approx(1e10)
        assert norm(-1e10) == approx(-1e10)


# ===================================================================
# Ticker detailed tests
# ===================================================================

class TestTickerDetailed:
    def test_maxnlocator_negative_range(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(-100, -10)
        assert len(ticks) >= 2
        assert ticks[0] <= -100
        assert ticks[-1] >= -10

    def test_maxnlocator_crossing_zero(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(-50, 50)
        assert 0 in ticks or any(abs(t) < 1 for t in ticks)

    def test_maxnlocator_large_range(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0, 1e9)
        assert len(ticks) >= 2

    def test_maxnlocator_tiny_range(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0, 1e-9)
        assert len(ticks) >= 2

    def test_multiplelocator_negative(self):
        loc = MultipleLocator(base=5)
        ticks = loc.tick_values(-20, -5)
        assert -20 in ticks
        assert -15 in ticks
        assert -10 in ticks
        assert -5 in ticks

    def test_loglocator_base_2(self):
        loc = LogLocator(base=2.0)
        ticks = loc.tick_values(1, 64)
        assert len(ticks) >= 2

    def test_loglocator_base_e(self):
        loc = LogLocator(base=math.e)
        ticks = loc.tick_values(1, 100)
        assert len(ticks) >= 2

    def test_indexlocator_negative_range(self):
        loc = IndexLocator(base=5, offset=0)
        ticks = loc.tick_values(-20, -5)
        assert len(ticks) >= 1

    def test_linearlocator_two_ticks(self):
        loc = LinearLocator(numticks=2)
        ticks = loc.tick_values(0, 100)
        assert len(ticks) == 2
        assert ticks[0] == approx(0)
        assert ticks[1] == approx(100)

    def test_fixedlocator_empty_range(self):
        loc = FixedLocator([1, 2, 3, 4, 5])
        ticks = loc.tick_values(10, 20)
        assert len(ticks) == 0

    def test_formatstr_scientific(self):
        fmt = FormatStrFormatter('%.2e')
        result = fmt(0.001)
        assert 'e' in result.lower()

    def test_strmethod_complex(self):
        fmt = StrMethodFormatter('{x:+.1f}')
        assert fmt(3.14) == '+3.1'
        assert fmt(-3.14) == '-3.1'

    def test_percent_100(self):
        fmt = PercentFormatter()
        assert fmt(100) == '100%'

    def test_percent_0(self):
        fmt = PercentFormatter()
        assert fmt(0) == '0%'

    def test_logformatter_1(self):
        fmt = LogFormatter()
        result = fmt(1)
        assert '0' in result  # 10^0

    def test_logformatter_1000(self):
        fmt = LogFormatter()
        result = fmt(1000)
        assert '3' in result  # 10^3


# ===================================================================
# Figure/Axes with new features
# ===================================================================

class TestFigureAxesNewFeatures:
    def test_axes_default_color_cycle(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = [ax._next_color() for _ in range(10)]
        assert len(set(colors)) == 10  # all different

    def test_axes_color_cycle_wraps(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        # Default cycle has 10 colors
        colors = [ax._next_color() for _ in range(15)]
        # After 10, should wrap
        assert colors[0] == colors[10]

    def test_axes_custom_cycle_wraps(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_prop_cycle(['red', 'blue'])
        colors = [ax._next_color() for _ in range(5)]
        assert colors[0] == colors[2]
        assert colors[1] == colors[3]

    def test_axes_reset_cycle(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_prop_cycle(['red', 'blue'])
        ax._next_color()  # 'red'
        ax.set_prop_cycle(None)
        c = ax._next_color()
        assert c.startswith('#')

    def test_plot_uses_cycle(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        l1 = ax.plot([1, 2], [3, 4])[0]
        l2 = ax.plot([1, 2], [5, 6])[0]
        assert l1.get_color() != l2.get_color()

    def test_scatter_uses_cycle(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        s1 = ax.scatter([1, 2], [3, 4])
        s2 = ax.scatter([1, 2], [5, 6])
        # Each scatter should use a different color from cycle
        assert s1.get_facecolors() != s2.get_facecolors()

    def test_text_artist_properties(self):
        t = Text(0, 0, 'test')
        props = t.properties()
        assert 'fontsize' in props
        assert 'text' in props
        assert 'visible' in props


# ===================================================================
# Additional cycler tests
# ===================================================================

class TestCyclerDetailed:
    def test_cycler_outer_product(self):
        c1 = Cycler('color', ['r', 'b'])
        c2 = Cycler('linestyle', ['-', '--'])
        product = c1 * c2
        assert len(product) == 4

    def test_cycler_concat(self):
        c1 = Cycler('color', ['r'])
        c2 = Cycler('color', ['b'])
        result = c1 + c2
        assert len(result) == 2

    def test_cycler_repeat(self):
        c = Cycler('color', ['r', 'b'])
        result = c * 3
        assert len(result) == 6

    def test_cycler_by_key_multiple(self):
        c = Cycler('color', ['r', 'b'])
        keys = c.by_key()
        assert 'color' in keys
        assert len(keys['color']) == 2

    def test_cycler_function_keywords(self):
        c = cycler(color=['r', 'g', 'b'])
        assert len(c) == 3

    def test_cycler_iteration_order(self):
        c = Cycler('color', ['first', 'second', 'third'])
        values = [d['color'] for d in c]
        assert values == ['first', 'second', 'third']


# ===================================================================
# Extended parametric tests for batch5
# ===================================================================

class TestBatch5Parametric:
    """Parametric tests for batch5: cycler, axes, plots."""

    @pytest.mark.parametrize('colors', [
        ['r', 'g', 'b'],
        ['red', 'blue', 'green', 'orange'],
        ['cyan', 'magenta'],
    ])
    def test_cycler_colors_parametric(self, colors):
        """Cycler length matches colors."""
        from matplotlib.cycler import cycler as cyc
        c = cyc(color=colors)
        assert len(c) == len(colors)

    @pytest.mark.parametrize('n', [2, 3, 4, 5, 6])
    def test_cycler_n_items(self, n):
        """Cycler with n items has len n."""
        from matplotlib.cycler import cycler as cyc
        c = cyc(color=range(n))
        assert len(c) == n

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100)])
    def test_xlim_set_get(self, xmin, xmax):
        """set_xlim/get_xlim roundtrip."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10

    @pytest.mark.parametrize('ymin,ymax', [(0, 1), (-10, 10), (0, 1000)])
    def test_ylim_set_get(self, ymin, ymax):
        """set_ylim/get_ylim roundtrip."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim(ymin, ymax)
        got = ax.get_ylim()
        assert abs(got[0] - ymin) < 1e-10
        assert abs(got[1] - ymax) < 1e-10

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale_parametric(self, scale):
        """set_xscale/get_xscale roundtrip."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_n_plots_n_lines(self, n):
        """n plot calls creates n lines."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_hist_n_bins(self, bins):
        """hist with bins parameter returns correct count."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        n_counts, edges, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n_counts) == bins

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 5.0])
    def test_line_linewidth_param(self, lw):
        """Line created with specific linewidth."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10

    @pytest.mark.parametrize('color', ['r', 'g', 'b', 'red', '#ff0000'])
    def test_line_color_param(self, color):
        """Line created without error for various colors."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line is not None

    @pytest.mark.parametrize('title', ['My Title', '', 'Test 123'])
    def test_title_set_get(self, title):
        """set_title/get_title roundtrip."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        assert ax.get_title() == title
