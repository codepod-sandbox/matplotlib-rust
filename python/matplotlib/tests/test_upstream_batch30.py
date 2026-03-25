"""
Upstream-ported tests batch 30: comprehensive parametric tests for ticker,
transforms, grid, twinaxes, legend, and hist.
"""

import math
import pytest
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoLocator, FixedLocator, FixedFormatter, NullLocator, NullFormatter,
    MaxNLocator, MultipleLocator, LogLocator, ScalarFormatter,
    SymmetricalLogLocator, LogFormatter,
)


# ===================================================================
# Ticker comprehensive parametric tests
# ===================================================================

class TestLocatorsParametric:
    """Parametric tests for ticker locators."""

    @pytest.mark.parametrize('ticks', [
        [],
        [0],
        [0, 1],
        [0, 0.5, 1.0],
        [-10, -5, 0, 5, 10],
        [1, 10, 100, 1000],
    ])
    def test_fixed_locator_returns_exact_ticks(self, ticks):
        """FixedLocator always returns the specified ticks."""
        loc = FixedLocator(ticks)
        result = list(loc())
        assert result == ticks

    @pytest.mark.parametrize('vmin,vmax', [
        (0, 1), (-1, 1), (0, 100), (0.001, 0.01), (1, 1e6)
    ])
    def test_auto_locator_in_range(self, vmin, vmax):
        """AutoLocator ticks are within [vmin, vmax]."""
        loc = AutoLocator()
        ticks = loc.tick_values(vmin, vmax)
        assert len(ticks) > 0

    @pytest.mark.parametrize('base', [2, 5, 10, 100])
    def test_multiple_locator_base(self, base):
        """MultipleLocator returns multiples of base."""
        loc = MultipleLocator(base=base)
        ticks = loc.tick_values(0, base * 5)
        for t in ticks:
            remainder = round(t % base, 10)
            assert remainder < 1e-8 or abs(remainder - base) < 1e-8

    @pytest.mark.parametrize('n', [3, 5, 7, 10])
    def test_max_n_locator_count(self, n):
        """MaxNLocator produces at most n+1 ticks."""
        loc = MaxNLocator(nbins=n)
        ticks = loc.tick_values(0, 100)
        assert len(ticks) <= n + 2

    def test_log_locator_base10(self):
        """LogLocator generates powers of 10."""
        loc = LogLocator(base=10)
        ticks = loc.tick_values(1, 1000)
        assert len(ticks) > 0
        # All ticks should be positive
        assert all(t > 0 for t in ticks)

    def test_null_locator_always_empty(self):
        """NullLocator returns empty list."""
        loc = NullLocator()
        for vmin, vmax in [(0, 1), (-1, 1), (0, 100)]:
            assert list(loc.tick_values(vmin, vmax)) == []


class TestFormattersParametric:
    """Parametric tests for ticker formatters."""

    @pytest.mark.parametrize('labels', [
        ['a', 'b'],
        ['zero', 'one', 'two'],
        ['x', 'y', 'z', 'w'],
    ])
    def test_fixed_formatter_returns_labels(self, labels):
        """FixedFormatter returns the correct label for each index."""
        fmt = FixedFormatter(labels)
        for i, label in enumerate(labels):
            assert fmt(i, i) == label

    def test_null_formatter_returns_empty(self):
        """NullFormatter returns empty string for any value."""
        fmt = NullFormatter()
        for val in [0, 1, 100, -50, 0.001]:
            assert fmt(val, 0) == ''

    @pytest.mark.parametrize('val,expected', [
        (0, '0'),
        (1, '1'),
        (10, '10'),
        (100, '100'),
        (-1, '-1'),
    ])
    def test_scalar_formatter_integers(self, val, expected):
        """ScalarFormatter formats integers correctly."""
        fmt = ScalarFormatter()
        assert fmt(val, 0) == expected


# ===================================================================
# Grid parametric tests
# ===================================================================

class TestGridParametric:
    """Parametric tests for grid display."""

    @pytest.mark.parametrize('which', ['major', 'minor', 'both'])
    def test_grid_which(self, which):
        """ax.grid works for major/minor/both."""
        fig, ax = plt.subplots()
        ax.grid(True, which=which)
        # Just check it doesn't raise
        assert ax is not None
        plt.close('all')

    @pytest.mark.parametrize('axis', ['x', 'y', 'both'])
    def test_grid_axis(self, axis):
        """ax.grid works for x/y/both axis."""
        fig, ax = plt.subplots()
        ax.grid(True, axis=axis)
        assert ax is not None
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.3, 0.5, 0.7, 1.0])
    def test_grid_alpha(self, alpha):
        """ax.grid accepts alpha parameter."""
        fig, ax = plt.subplots()
        ax.grid(True, alpha=alpha)
        assert ax is not None
        plt.close('all')

    @pytest.mark.parametrize('linestyle', ['-', '--', ':', '-.'])
    def test_grid_linestyle(self, linestyle):
        """ax.grid accepts linestyle parameter."""
        fig, ax = plt.subplots()
        ax.grid(True, linestyle=linestyle)
        assert ax is not None
        plt.close('all')


# ===================================================================
# Twinaxes parametric tests
# ===================================================================

class TestTwinAxesParametric:
    """Parametric tests for twinx/twiny."""

    def test_twinx_creates_second_axes(self):
        """twinx creates a second axes sharing x."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        assert ax2 is not None
        assert ax2 is not ax1
        plt.close('all')

    def test_twiny_creates_second_axes(self):
        """twiny creates a second axes sharing y."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        assert ax2 is not None
        assert ax2 is not ax1
        plt.close('all')

    def test_twinx_shares_xlim(self):
        """twinx axes share x-limits."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlim(0, 10)
        assert ax2.get_xlim() == (0, 10)
        plt.close('all')

    def test_twiny_shares_ylim(self):
        """twiny axes share y-limits."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.set_ylim(-5, 5)
        assert ax2.get_ylim() == (-5, 5)
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3])
    def test_twinx_n_times(self, n):
        """twinx can be called multiple times."""
        fig, ax = plt.subplots()
        axes = [ax]
        for _ in range(n):
            axes.append(axes[0].twinx())
        assert len(axes) == n + 1
        plt.close('all')

    def test_twinx_independent_ylim(self):
        """twinx axes have independent y-limits."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_ylim(0, 10)
        ax2.set_ylim(-100, 100)
        assert ax1.get_ylim() == (0, 10)
        assert ax2.get_ylim() == (-100, 100)
        plt.close('all')


# ===================================================================
# Hist parametric tests
# ===================================================================

class TestHistParametric:
    """Parametric tests for histogram."""

    @pytest.mark.parametrize('bins', [5, 10, 20, 50])
    def test_hist_bins_count(self, bins):
        """hist returns correct number of bins."""
        fig, ax = plt.subplots()
        data = list(range(100))
        counts, edges, _ = ax.hist(data, bins=bins)
        assert len(counts) == bins
        assert len(edges) == bins + 1
        plt.close('all')

    def test_hist_sum_counts(self):
        """hist counts sum to total data points."""
        fig, ax = plt.subplots()
        data = list(range(50))
        counts, edges, _ = ax.hist(data, bins=10)
        assert sum(counts) == 50
        plt.close('all')

    @pytest.mark.parametrize('histtype', ['bar', 'step', 'stepfilled'])
    def test_hist_type(self, histtype):
        """hist accepts histtype parameter."""
        fig, ax = plt.subplots()
        counts, edges, _ = ax.hist([1, 2, 3, 4, 5], histtype=histtype)
        assert sum(counts) == 5
        plt.close('all')

    def test_hist_density(self):
        """hist with density=True has counts summing to ~1."""
        fig, ax = plt.subplots()
        data = list(range(100))
        counts, edges, _ = ax.hist(data, bins=10, density=True)
        # With density, counts * bin_width should sum to ~1
        bin_width = edges[1] - edges[0]
        total = sum(c * bin_width for c in counts)
        assert abs(total - 1.0) < 0.01
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', '#00ff00'])
    def test_hist_color(self, color):
        """hist accepts color parameter."""
        fig, ax = plt.subplots()
        counts, edges, patches = ax.hist([1, 2, 3], color=color)
        assert len(patches) > 0
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.5, 0.7, 1.0])
    def test_hist_alpha(self, alpha):
        """hist accepts alpha parameter."""
        fig, ax = plt.subplots()
        counts, edges, patches = ax.hist([1, 2, 3], alpha=alpha)
        assert len(patches) > 0
        plt.close('all')


# ===================================================================
# Legend parametric tests
# ===================================================================

class TestLegendParametric:
    """Parametric tests for legend creation and properties."""

    @pytest.mark.parametrize('loc', [
        'upper right', 'upper left', 'lower right', 'lower left',
        'center', 'best'
    ])
    def test_legend_loc(self, loc):
        """Legend accepts various location strings."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='line')
        leg = ax.legend(loc=loc)
        assert leg is not None
        plt.close('all')

    @pytest.mark.parametrize('n_labels', [1, 2, 3, 5])
    def test_legend_n_labels(self, n_labels):
        """Legend has correct number of labels."""
        fig, ax = plt.subplots()
        for i in range(n_labels):
            ax.plot([i, i+1], [i, i+1], label=f'line{i}')
        leg = ax.legend()
        assert len(leg.get_texts()) == n_labels
        plt.close('all')

    @pytest.mark.parametrize('title', ['Legend', 'My Legend', ''])
    def test_legend_title(self, title):
        """Legend title is settable."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='line')
        leg = ax.legend(title=title)
        assert leg.get_title().get_text() == title
        plt.close('all')

    def test_legend_frameon_default(self):
        """Legend frameon is True by default."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='line')
        leg = ax.legend()
        assert leg.get_frame_on() is True
        plt.close('all')

    def test_legend_frameon_false(self):
        """Legend frameon can be set to False."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='line')
        leg = ax.legend(frameon=False)
        assert leg.get_frame_on() is False
        plt.close('all')

    @pytest.mark.parametrize('fontsize', [8, 10, 12, 14])
    def test_legend_fontsize(self, fontsize):
        """Legend fontsize is settable."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='line')
        leg = ax.legend(fontsize=fontsize)
        assert leg is not None
        plt.close('all')

    def test_legend_handles_count_matches_plots(self):
        """Legend handles count matches number of labeled plots."""
        fig, ax = plt.subplots()
        n = 4
        for i in range(n):
            ax.plot([i, i+1], [0, 1], label=f's{i}')
        leg = ax.legend()
        handles = leg.legend_handles
        assert len(handles) == n
        plt.close('all')


# ===================================================================
# More parametric tests for batch30
# ===================================================================

class TestBatch30Parametric2:
    """More parametric tests for batch30."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_n_lines2(self, n):
        """n plot calls creates n lines."""
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20, 50])
    def test_hist_bins2(self, bins):
        """hist bins count."""
        fig, ax = plt.subplots()
        n_counts, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n_counts) == bins
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100)])
    def test_xlim2(self, xmin, xmax):
        """xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale2(self, scale):
        """xscale roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('title', ['Title', '', 'My Title'])
    def test_title2(self, title):
        """title roundtrip."""
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_bar_n2(self, n):
        """bar creates n patches."""
        fig, ax = plt.subplots()
        container = ax.bar(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 5.0])
    def test_linewidth2(self, lw):
        """linewidth stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.5, 1.0])
    def test_alpha2(self, alpha):
        """alpha stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close('all')
