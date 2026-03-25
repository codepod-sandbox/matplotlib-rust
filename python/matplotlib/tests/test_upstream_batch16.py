"""
Upstream tests — batch 16.
Focus: Colors, Norms, Colormaps, Ticker formatters/locators deep tests.
Adapted from matplotlib upstream tests (no canvas rendering, no image comparison).
Note: Using math.isclose() instead of pytest.approx (RustPython np.bool_ compat).
"""
import math
import pytest
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.colors import (
    Normalize, LogNorm, TwoSlopeNorm, BoundaryNorm, PowerNorm, SymLogNorm,
    to_rgba, to_rgb, to_hex, is_color_like, same_color,
)
from matplotlib.ticker import (
    NullFormatter, FixedFormatter, FuncFormatter, FormatStrFormatter,
    StrMethodFormatter, ScalarFormatter, LogFormatter, PercentFormatter,
    NullLocator, FixedLocator, IndexLocator, LinearLocator, MultipleLocator,
    MaxNLocator, AutoLocator, AutoMinorLocator, LogLocator,
)


def approx(a, b, rel=1e-6):
    """Check approximate equality without using pytest.approx."""
    if b == 0:
        return abs(a) < 1e-10
    return abs(a - b) / abs(b) < rel


# ------------------------------------------------------------------
# Normalize tests
# ------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        norm = Normalize(vmin=0, vmax=10)
        assert approx(norm(5), 0.5)
        assert approx(norm(0), 0.0)
        assert approx(norm(10), 1.0)

    def test_list_input(self):
        norm = Normalize(vmin=0, vmax=10)
        result = norm([0, 5, 10])
        assert approx(result[0], 0.0)
        assert approx(result[1], 0.5)
        assert approx(result[2], 1.0)

    def test_clip_true(self):
        norm = Normalize(vmin=0, vmax=10, clip=True)
        assert approx(norm(15), 1.0)
        assert approx(norm(-5), 0.0)

    def test_clip_false(self):
        norm = Normalize(vmin=0, vmax=10, clip=False)
        assert approx(norm(15), 1.5)
        assert approx(norm(-5), -0.5)

    def test_inverse(self):
        norm = Normalize(vmin=0, vmax=10)
        assert approx(norm.inverse(0.5), 5.0)
        assert approx(norm.inverse(0.0), 0.0)
        assert approx(norm.inverse(1.0), 10.0)

    def test_inverse_list(self):
        norm = Normalize(vmin=0, vmax=10)
        result = norm.inverse([0.0, 0.5, 1.0])
        assert approx(result[0], 0.0)
        assert approx(result[1], 5.0)
        assert approx(result[2], 10.0)

    def test_autoscale(self):
        norm = Normalize()
        norm.autoscale([1, 2, 3, 4, 5])
        assert approx(norm.vmin, 1.0)
        assert approx(norm.vmax, 5.0)

    def test_scaled_property(self):
        norm = Normalize()
        assert not norm.scaled
        norm.vmin = 0
        norm.vmax = 1
        assert norm.scaled

    def test_autoscale_none_keeps_existing(self):
        norm = Normalize(vmin=2)
        norm.autoscale_None([1, 5])
        # vmin was already set, should remain 2
        assert approx(norm.vmin, 2.0)
        # vmax was not set, should be updated to 5
        assert approx(norm.vmax, 5.0)

    def test_repr(self):
        norm = Normalize(vmin=0, vmax=1)
        r = repr(norm)
        assert 'Normalize' in r
        assert 'vmin=0' in r
        assert 'vmax=1' in r

    def test_eq(self):
        n1 = Normalize(vmin=0, vmax=1)
        n2 = Normalize(vmin=0, vmax=1)
        n3 = Normalize(vmin=0, vmax=2)
        assert n1 == n2
        assert n1 != n3

    def test_hash(self):
        n1 = Normalize(vmin=0, vmax=1)
        n2 = Normalize(vmin=0, vmax=1)
        assert hash(n1) == hash(n2)

    def test_equal_vmin_vmax(self):
        # When vmin == vmax should return 0
        norm = Normalize(vmin=5, vmax=5)
        assert approx(norm(5), 0.0)

    def test_no_args_raises_on_call(self):
        norm = Normalize()
        with pytest.raises((ValueError, Exception)):
            norm(5.0)


class TestLogNorm:
    def test_basic(self):
        norm = LogNorm(vmin=1, vmax=100)
        assert approx(norm(10), 0.5)
        assert approx(norm(1), 0.0)
        assert approx(norm(100), 1.0)

    def test_inverse(self):
        norm = LogNorm(vmin=1, vmax=100)
        assert approx(norm.inverse(0.0), 1.0)
        assert approx(norm.inverse(1.0), 100.0)
        assert approx(norm.inverse(0.5), 10.0)

    def test_list_input(self):
        norm = LogNorm(vmin=1, vmax=100)
        result = norm([1, 10, 100])
        assert approx(result[0], 0.0)
        assert approx(result[1], 0.5)
        assert approx(result[2], 1.0)

    def test_repr(self):
        norm = LogNorm(vmin=1, vmax=10)
        r = repr(norm)
        assert 'LogNorm' in r


class TestTwoSlopeNorm:
    def test_basic(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        assert approx(norm(0), 0.5)
        assert approx(norm(-10), 0.0)
        assert approx(norm(10), 1.0)

    def test_asymmetric(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-5, vmax=20)
        assert approx(norm(0), 0.5)
        assert approx(norm(-5), 0.0)
        assert approx(norm(20), 1.0)

    def test_repr(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        r = repr(norm)
        assert 'TwoSlopeNorm' in r


class TestBoundaryNorm:
    def test_basic(self):
        bounds = [0, 1, 2, 3]
        norm = BoundaryNorm(bounds, ncolors=3)
        # Values in the first bin
        result = norm(0.5)
        assert result is not None

    def test_repr(self):
        norm = BoundaryNorm([0, 1, 2], ncolors=2)
        r = repr(norm)
        assert 'BoundaryNorm' in r


class TestPowerNorm:
    def test_basic(self):
        norm = PowerNorm(gamma=2, vmin=0, vmax=1)
        assert approx(norm(0), 0.0)
        assert approx(norm(1), 1.0)
        # midpoint: 0.5^2 = 0.25
        assert approx(norm(0.5), 0.25, rel=1e-3)

    def test_inverse(self):
        norm = PowerNorm(gamma=2, vmin=0, vmax=1)
        assert approx(norm.inverse(0.25), 0.5)

    def test_repr(self):
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
        r = repr(norm)
        assert 'PowerNorm' in r


# ------------------------------------------------------------------
# Color utility tests
# ------------------------------------------------------------------

class TestColorUtils:
    def test_to_rgba_named(self):
        r, g, b, a = to_rgba('red')
        assert approx(r, 1.0)
        assert approx(g, 0.0)
        assert approx(b, 0.0)
        assert approx(a, 1.0)

    def test_to_rgba_hex(self):
        r, g, b, a = to_rgba('#ff0000')
        assert approx(r, 1.0)
        assert approx(g, 0.0)
        assert approx(b, 0.0)
        assert approx(a, 1.0)

    def test_to_rgba_tuple(self):
        r, g, b, a = to_rgba((0.5, 0.5, 0.5, 1.0))
        assert approx(r, 0.5)
        assert approx(g, 0.5)
        assert approx(b, 0.5)

    def test_to_rgba_with_alpha(self):
        r, g, b, a = to_rgba('red', alpha=0.5)
        assert approx(a, 0.5)

    def test_to_rgb_named(self):
        r, g, b = to_rgb('blue')
        assert approx(r, 0.0)
        assert approx(g, 0.0)
        assert approx(b, 1.0)

    def test_to_hex_red(self):
        h = to_hex((1.0, 0.0, 0.0))
        assert h.lower() == '#ff0000'

    def test_to_hex_green(self):
        h = to_hex((0.0, 1.0, 0.0))
        assert h.lower() == '#00ff00'

    def test_to_hex_blue(self):
        h = to_hex((0.0, 0.0, 1.0))
        assert h.lower() == '#0000ff'

    def test_is_color_like_named(self):
        assert is_color_like('red')
        assert is_color_like('blue')
        assert is_color_like('green')

    def test_is_color_like_hex(self):
        assert is_color_like('#ff0000')
        assert is_color_like('#000')

    def test_is_color_like_tuple(self):
        assert is_color_like((1.0, 0.0, 0.0))
        assert is_color_like((0.5, 0.5, 0.5, 1.0))

    def test_is_color_like_invalid(self):
        assert not is_color_like('notacolor')

    def test_same_color_true(self):
        assert same_color('red', '#ff0000')

    def test_same_color_false(self):
        assert not same_color('red', 'blue')

    def test_white_and_black(self):
        assert is_color_like('white')
        assert is_color_like('black')
        assert is_color_like('k')
        assert is_color_like('w')

    def test_shorthand_colors(self):
        assert is_color_like('r')
        assert is_color_like('g')
        assert is_color_like('b')
        assert is_color_like('c')
        assert is_color_like('m')
        assert is_color_like('y')

    def test_to_rgba_blue(self):
        r, g, b, a = to_rgba('blue')
        assert approx(r, 0.0)
        assert approx(g, 0.0)
        assert approx(b, 1.0)

    def test_to_rgba_green(self):
        r, g, b, a = to_rgba('green')
        assert r >= 0.0
        assert b >= 0.0
        assert a == 1.0


# ------------------------------------------------------------------
# Ticker Formatter tests
# ------------------------------------------------------------------

class TestFormatters:
    def test_null_formatter(self):
        f = NullFormatter()
        assert f(1.0) == ''
        assert f(0) == ''

    def test_fixed_formatter(self):
        f = FixedFormatter(['a', 'b', 'c'])
        # FixedFormatter uses pos (second arg) as the index
        assert f(0, 0) == 'a'
        assert f(0, 2) == 'c'
        assert f(0, 5) == ''  # out of range

    def test_func_formatter(self):
        f = FuncFormatter(lambda x, pos: f'{x:.1f}')
        assert f(1.5, 0) == '1.5'

    def test_format_str_formatter(self):
        f = FormatStrFormatter('%.2f')
        assert f(3.14159) == '3.14'
        assert f(1.0) == '1.00'

    def test_str_method_formatter(self):
        f = StrMethodFormatter('{x:.2f}')
        assert f(3.14159) == '3.14'

    def test_scalar_formatter_basic(self):
        f = ScalarFormatter()
        result = f(1234.5, 0)
        assert isinstance(result, str)

    def test_log_formatter(self):
        f = LogFormatter(base=10.0)
        result = f(100.0, 0)
        assert result is not None

    def test_percent_formatter(self):
        f = PercentFormatter(xmax=1.0, decimals=1)
        result = f(0.5, 0)
        assert '50' in result
        assert '%' in result

    def test_percent_formatter_100(self):
        f = PercentFormatter(xmax=100, decimals=0)
        result = f(50, 0)
        assert '50' in result

    def test_scalar_formatter_offset(self):
        f = ScalarFormatter()
        offset = f.get_offset()
        assert isinstance(offset, str)

    def test_fixed_formatter_get_offset(self):
        f = FixedFormatter(['a', 'b'])
        f.set_offset_string(' x10^3')
        assert f.get_offset() == ' x10^3'

    def test_formatter_set_locs(self):
        f = ScalarFormatter()
        f.set_locs([0, 1, 2, 3])  # Should not error

    def test_func_formatter_with_pos(self):
        f = FuncFormatter(lambda x, pos: f'{int(x)}-{pos}')
        assert f(5, 2) == '5-2'


# ------------------------------------------------------------------
# Ticker Locator tests
# ------------------------------------------------------------------

class TestLocators:
    def test_null_locator(self):
        loc = NullLocator()
        assert loc() == []
        assert loc.tick_values(0, 10) == []

    def test_fixed_locator(self):
        loc = FixedLocator([0, 1, 2, 3, 4])
        ticks = loc.tick_values(0, 10)
        assert list(ticks) == [0, 1, 2, 3, 4]

    def test_index_locator(self):
        loc = IndexLocator(base=2, offset=0)
        ticks = loc.tick_values(0, 10)
        assert all(t % 2 == 0 for t in ticks)

    def test_linear_locator(self):
        loc = LinearLocator(numticks=5)
        ticks = loc.tick_values(0, 1)
        assert len(ticks) == 5
        assert abs(ticks[0] - 0.0) < 1e-10
        assert abs(ticks[-1] - 1.0) < 1e-10

    def test_multiple_locator(self):
        loc = MultipleLocator(base=0.5)
        ticks = loc.tick_values(0, 2)
        for t in ticks:
            # each tick should be a multiple of 0.5
            remainder = abs(t % 0.5)
            assert remainder < 1e-10 or abs(remainder - 0.5) < 1e-10

    def test_multiple_locator_set_params(self):
        loc = MultipleLocator(base=1.0)
        loc.set_params(base=2.0)
        assert abs(loc._base - 2.0) < 1e-10

    def test_max_n_locator(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) >= 2
        assert ticks[0] >= 0
        assert ticks[-1] <= 10

    def test_auto_locator(self):
        loc = AutoLocator()
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2

    def test_auto_minor_locator(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.minorticks_on()
        ticks = ax.xaxis.get_ticklocs(minor=True)
        assert isinstance(ticks, list)
        assert len(ticks) > 0

    def test_log_locator(self):
        loc = LogLocator(base=10.0)
        ticks = loc.tick_values(1, 1000)
        assert any(abs(t - 10.0) < 1e-10 for t in ticks)
        assert any(abs(t - 100.0) < 1e-10 for t in ticks)

    def test_fixed_locator_set_params(self):
        loc = FixedLocator([1, 2, 3])
        loc.set_params(nbins=2)
        assert loc.nbins == 2

    def test_locator_raise_if_exceeds(self):
        loc = AutoLocator()
        # Should work without error for reasonable number
        loc.raise_if_exceeds(list(range(100)))

    def test_log_locator_subs(self):
        loc = LogLocator(base=10.0, subs=[1, 2, 5])
        ticks = loc.tick_values(1, 100)
        assert len(ticks) >= 3

    def test_fixed_locator_len(self):
        loc = FixedLocator([1, 2, 3, 4, 5])
        ticks = loc.tick_values(0, 100)
        assert len(ticks) == 5

    def test_max_n_locator_set_params(self):
        loc = MaxNLocator(nbins=5)
        loc.set_params(nbins=10)
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2

    def test_linear_locator_3_ticks(self):
        loc = LinearLocator(numticks=3)
        ticks = loc.tick_values(0, 100)
        assert len(ticks) == 3
        assert abs(ticks[0] - 0.0) < 1e-10
        assert abs(ticks[-1] - 100.0) < 1e-10


# ------------------------------------------------------------------
# Axes scale tests
# ------------------------------------------------------------------

class TestAxesScale:
    def test_set_xscale_log(self):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        assert ax.get_xscale() == 'log'

    def test_set_yscale_log(self):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        assert ax.get_yscale() == 'log'

    def test_set_xscale_linear(self):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xscale('linear')
        assert ax.get_xscale() == 'linear'

    def test_set_yscale_linear(self):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_yscale('linear')
        assert ax.get_yscale() == 'linear'

    def test_set_xscale_symlog(self):
        fig, ax = plt.subplots()
        ax.set_xscale('symlog')
        assert ax.get_xscale() == 'symlog'

    def test_default_scales(self):
        fig, ax = plt.subplots()
        assert ax.get_xscale() == 'linear'
        assert ax.get_yscale() == 'linear'

    def test_loglog(self):
        fig, ax = plt.subplots()
        ax.loglog([1, 10, 100], [1, 10, 100])
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'

    def test_semilogx(self):
        fig, ax = plt.subplots()
        ax.semilogx([1, 10, 100], [1, 2, 3])
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'linear'

    def test_semilogy(self):
        fig, ax = plt.subplots()
        ax.semilogy([1, 2, 3], [1, 10, 100])
        assert ax.get_xscale() == 'linear'
        assert ax.get_yscale() == 'log'


# ===================================================================
# Extended parametric tests for batch16
# ===================================================================

class TestBatch16Parametric:
    """Parametric tests for batch16: log/linear scale, plot types."""

    @pytest.mark.parametrize('data', [
        ([1, 10, 100], [1, 10, 100]),
        ([1, 100, 10000], [1, 2, 3]),
        ([0.1, 1, 10], [0.1, 1, 10]),
    ])
    def test_loglog_data(self, data):
        """loglog sets both scales to log."""
        x, y = data
        fig, ax = plt.subplots()
        ax.loglog(x, y)
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        plt.close('all')

    @pytest.mark.parametrize('data', [
        ([1, 10, 100], [1, 2, 3]),
        ([0.1, 1, 10, 100], [0, 1, 2, 3]),
        ([1, 1000], [0, 1]),
    ])
    def test_semilogx_data(self, data):
        """semilogx sets xscale=log, yscale=linear."""
        x, y = data
        fig, ax = plt.subplots()
        ax.semilogx(x, y)
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'linear'
        plt.close('all')

    @pytest.mark.parametrize('data', [
        ([1, 2, 3], [1, 10, 100]),
        ([0, 1, 2, 3], [0.1, 1, 10, 100]),
        ([0, 1], [1, 1000]),
    ])
    def test_semilogy_data(self, data):
        """semilogy sets xscale=linear, yscale=log."""
        x, y = data
        fig, ax = plt.subplots()
        ax.semilogy(x, y)
        assert ax.get_xscale() == 'linear'
        assert ax.get_yscale() == 'log'
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100), (-10, 10)])
    def test_linear_xlim(self, xmin, xmax):
        """Linear scale xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_n_lines_various(self, n):
        """n plot calls creates n lines."""
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('title', ['Title A', '', 'Test'])
    def test_title_roundtrip(self, title):
        """set_title/get_title roundtrip."""
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale_roundtrip(self, scale):
        """set_xscale/get_xscale roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_hist_bins(self, bins):
        """hist returns correct bin count."""
        fig, ax = plt.subplots()
        n_counts, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n_counts) == bins
        plt.close('all')


class TestBatch16Parametric2:
    """More parametric tests for batch16."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_plot_n_lines2(self, n):
        """ax.plot n times gives n lines."""
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('lo,hi', [(0, 1), (-1, 1), (0, 100), (-5, 5)])
    def test_xlim2(self, lo, hi):
        """xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(lo, hi)
        assert ax.get_xlim() == (lo, hi)
        plt.close('all')

    @pytest.mark.parametrize('lo,hi', [(0, 1), (-1, 1), (0, 100), (-5, 5)])
    def test_ylim2(self, lo, hi):
        """ylim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_ylim(lo, hi)
        assert ax.get_ylim() == (lo, hi)
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale2(self, scale):
        """xscale roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('title', ['Title', 'Test', ''])
    def test_title2(self, title):
        """title roundtrip."""
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', 'black'])
    def test_line_color(self, color):
        """Line color is stored (may be returned as hex)."""
        from matplotlib.colors import to_hex
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert to_hex(line.get_color()) == to_hex(color)
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0])
    def test_line_linewidth(self, lw):
        """Line linewidth is stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('label', ['line1', 'series_a', ''])
    def test_line_label(self, label):
        """Line label is stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], label=label)
        assert line.get_label() == label
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'D', 'x'])
    def test_line_marker(self, marker):
        """Line marker is stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.5, 0.8, 1.0])
    def test_line_alpha(self, alpha):
        """Line alpha is stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close('all')


class TestBatch16Parametric8:
    """Further parametric tests for batch 16."""

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



class TestBatch16Parametric12:
    """Yet more parametric tests for batch 16."""

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

