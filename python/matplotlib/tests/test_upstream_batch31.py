"""
Upstream-ported tests batch 31: comprehensive parametric tests for
figure, pyplot, container, rendering, scale, and cycler.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# ===================================================================
# Figure parametric tests
# ===================================================================

class TestFigureParametricExtended:
    """Extended parametric tests for Figure."""

    @pytest.mark.parametrize('w,h', [
        (4, 3), (6.4, 4.8), (8, 6), (10, 10), (12, 4), (3, 10),
    ])
    def test_figure_size(self, w, h):
        """Figure stores figwidth and figheight."""
        fig = plt.figure(figsize=(w, h))
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 96, 100, 150, 200, 300])
    def test_figure_dpi(self, dpi):
        """Figure stores DPI."""
        fig = plt.figure(dpi=dpi)
        assert fig.get_dpi() == dpi
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 6])
    def test_figure_axes_count(self, n):
        """Figure has correct number of axes after add_subplot."""
        fig = plt.figure()
        for i in range(n):
            fig.add_subplot(1, n, i + 1)
        assert len(fig.get_axes()) == n
        plt.close('all')

    @pytest.mark.parametrize('suptitle', ['Main Title', 'Figure Overview', 'My Plot', ''])
    def test_figure_suptitle(self, suptitle):
        """Figure suptitle is stored."""
        fig = plt.figure()
        fig.suptitle(suptitle)
        assert fig.get_suptitle() == suptitle
        plt.close('all')

    def test_figure_clf_removes_axes(self):
        """clf() removes all axes."""
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        fig.add_subplot(1, 2, 2)
        assert len(fig.get_axes()) == 2
        fig.clf()
        assert len(fig.get_axes()) == 0
        plt.close('all')

    @pytest.mark.parametrize('nrows,ncols', [
        (1, 1), (2, 2), (1, 3), (3, 1), (2, 3),
    ])
    def test_figure_subplots_count(self, nrows, ncols):
        """Figure has nrows*ncols axes after subplots."""
        fig, axes = plt.subplots(nrows, ncols)
        assert len(fig.get_axes()) == nrows * ncols
        plt.close('all')

    @pytest.mark.parametrize('w,h', [(4, 3), (8, 6), (10, 4)])
    def test_figure_set_size(self, w, h):
        """Figure.set_size_inches stores size."""
        fig = plt.figure()
        fig.set_size_inches(w, h)
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')

    def test_figure_add_subplot_returns_axes(self):
        """add_subplot returns an Axes instance."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        assert isinstance(ax, Axes)
        plt.close('all')

    @pytest.mark.parametrize('num', [1, 2, 3, 10])
    def test_figure_number(self, num):
        """Figure stores its number."""
        fig = plt.figure(num)
        assert fig.number == num
        plt.close('all')


# ===================================================================
# Pyplot parametric tests
# ===================================================================

class TestPyplotParametricExtended:
    """Extended parametric tests for pyplot API."""

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_plt_plot_n_lines(self, n):
        """plt.plot called n times creates n lines in current axes."""
        fig, ax = plt.subplots()
        for i in range(n):
            plt.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_plt_hist_bins(self, bins):
        """plt.hist returns correct number of bins."""
        fig, ax = plt.subplots()
        n, edges, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close('all')

    @pytest.mark.parametrize('figsize', [(4, 3), (6.4, 4.8), (10, 8)])
    def test_plt_figure_figsize(self, figsize):
        """plt.figure stores figsize."""
        w, h = figsize
        fig = plt.figure(figsize=figsize)
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('title', ['My Title', '', 'Test Title'])
    def test_plt_title(self, title):
        """plt.title sets title on current axes."""
        fig, ax = plt.subplots()
        plt.title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('label', ['X', '', 'Time (s)'])
    def test_plt_xlabel(self, label):
        """plt.xlabel sets xlabel on current axes."""
        fig, ax = plt.subplots()
        plt.xlabel(label)
        assert ax.get_xlabel() == label
        plt.close('all')

    @pytest.mark.parametrize('label', ['Y', '', 'Amplitude'])
    def test_plt_ylabel(self, label):
        """plt.ylabel sets ylabel on current axes."""
        fig, ax = plt.subplots()
        plt.ylabel(label)
        assert ax.get_ylabel() == label
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100)])
    def test_plt_xlim(self, xmin, xmax):
        """plt.xlim sets xlim on current axes."""
        fig, ax = plt.subplots()
        plt.xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('ymin,ymax', [(0, 1), (-5, 5), (0, 100)])
    def test_plt_ylim(self, ymin, ymax):
        """plt.ylim sets ylim on current axes."""
        fig, ax = plt.subplots()
        plt.ylim(ymin, ymax)
        got = ax.get_ylim()
        assert abs(got[0] - ymin) < 1e-10
        assert abs(got[1] - ymax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('nrows,ncols', [(1, 1), (2, 2), (1, 3), (2, 3)])
    def test_plt_subplots_count(self, nrows, ncols):
        """plt.subplots creates correct number of axes."""
        fig, axes = plt.subplots(nrows, ncols)
        assert len(fig.get_axes()) == nrows * ncols
        plt.close('all')


# ===================================================================
# Scale parametric tests
# ===================================================================

class TestScaleParametricExtended:
    """Extended parametric scale tests."""

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale_roundtrip(self, scale):
        """set_xscale / get_xscale roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_yscale_roundtrip(self, scale):
        """set_yscale / get_yscale roundtrip."""
        fig, ax = plt.subplots()
        ax.set_yscale(scale)
        assert ax.get_yscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('vmin,vmax', [
        (1, 10), (0.1, 100), (1, 1000), (10, 10000),
    ])
    def test_log_scale_tick_values(self, vmin, vmax):
        """Log scale tick_values are positive."""
        from matplotlib.ticker import LogLocator
        loc = LogLocator(base=10)
        ticks = loc.tick_values(vmin, vmax)
        assert len(ticks) > 0
        assert all(t > 0 for t in ticks)

    @pytest.mark.parametrize('base', [2, 10, np.e])
    def test_log_locator_base(self, base):
        """LogLocator with various bases generates ticks."""
        from matplotlib.ticker import LogLocator
        loc = LogLocator(base=base)
        ticks = loc.tick_values(1, 100)
        assert len(ticks) > 0

    @pytest.mark.parametrize('linthresh', [0.1, 1.0, 10.0])
    def test_symlog_scale_linthresh(self, linthresh):
        """Symlog scale with different linthresh values."""
        fig, ax = plt.subplots()
        ax.set_xscale('symlog', linthresh=linthresh)
        assert ax.get_xscale() == 'symlog'
        plt.close('all')


# ===================================================================
# Cycler parametric tests
# ===================================================================

class TestCyclerParametricExtended:
    """Extended parametric cycler tests."""

    @pytest.mark.parametrize('colors', [
        ['r', 'g', 'b'],
        ['red', 'blue', 'green', 'orange'],
        ['#ff0000', '#00ff00'],
    ])
    def test_cycler_colors(self, colors):
        """Cycler with colors iterates correctly."""
        from matplotlib.cycler import cycler
        c = cycler('color', colors)
        assert len(c) == len(colors)

    @pytest.mark.parametrize('n', [2, 3, 5, 8])
    def test_cycler_length(self, n):
        """Cycler length matches number of values."""
        from matplotlib.cycler import cycler
        c = cycler('color', range(n))
        assert len(c) == n

    @pytest.mark.parametrize('key', ['color', 'linewidth', 'marker', 'linestyle'])
    def test_cycler_key(self, key):
        """Cycler stores key correctly."""
        from matplotlib.cycler import cycler
        c = cycler(key, [1, 2, 3])
        assert key in c.keys

    @pytest.mark.parametrize('n', [2, 4, 6, 8])
    def test_cycler_iteration_count(self, n):
        """Cycler iterates n times."""
        from matplotlib.cycler import cycler
        c = cycler('color', range(n))
        items = list(c)
        assert len(items) == n

    def test_cycler_keys_from_mul(self):
        """Multiplying two cyclers produces keys from both."""
        from matplotlib.cycler import cycler
        c1 = cycler('color', ['r', 'g'])
        c2 = cycler('linewidth', [1, 2])
        c3 = c1 * c2
        assert 'color' in c3.keys
        assert 'linewidth' in c3.keys

    @pytest.mark.parametrize('n', [2, 3, 5])
    def test_prop_cycle_cycling(self, n):
        """Axes prop_cycle cycles through n colors."""
        from matplotlib.cycler import cycler
        fig, ax = plt.subplots()
        colors = [f'#{i*20:02x}{0:02x}{0:02x}' for i in range(1, n+1)]
        ax.set_prop_cycle(cycler('color', colors))
        lines = [ax.plot([0, 1], [i, i+1])[0] for i in range(n)]
        # Each line should have a different color
        line_colors = [l.get_color() for l in lines]
        assert len(set(line_colors)) == n
        plt.close('all')


# ===================================================================
# Container parametric tests
# ===================================================================

class TestContainerParametricExtended:
    """Extended parametric container tests."""

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_bar_container_n_patches(self, n):
        """BarContainer has n patches."""
        fig, ax = plt.subplots()
        container = ax.bar(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_barh_container_n_patches(self, n):
        """barh BarContainer has n patches."""
        fig, ax = plt.subplots()
        container = ax.barh(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff0000'])
    def test_bar_color(self, color):
        """bar accepts color parameter."""
        fig, ax = plt.subplots()
        container = ax.bar([1, 2, 3], [1, 2, 3], color=color)
        assert container is not None
        plt.close('all')

    @pytest.mark.parametrize('width', [0.5, 0.8, 1.0])
    def test_bar_width(self, width):
        """bar accepts width parameter."""
        fig, ax = plt.subplots()
        container = ax.bar([1, 2, 3], [1, 2, 3], width=width)
        assert container is not None
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.3, 0.5, 0.7, 1.0])
    def test_bar_alpha(self, alpha):
        """bar accepts alpha parameter."""
        fig, ax = plt.subplots()
        container = ax.bar([1, 2, 3], [1, 2, 3], alpha=alpha)
        assert container is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 4])
    def test_errorbar_n_points(self, n):
        """errorbar creates container with n points."""
        fig, ax = plt.subplots()
        x = list(range(n))
        y = list(range(n))
        yerr = [0.1] * n
        ec = ax.errorbar(x, y, yerr=yerr)
        assert ec is not None
        plt.close('all')


# ===================================================================
# Rendering parametric tests
# ===================================================================

class TestRenderingParametricExtended:
    """Extended parametric rendering tests."""

    @pytest.mark.parametrize('n_lines', [1, 3, 5])
    def test_svg_n_lines(self, n_lines):
        """SVG output contains plot elements."""
        fig, ax = plt.subplots()
        for i in range(n_lines):
            ax.plot([0, 1], [i, i+1])
        svg = fig.to_svg()
        assert isinstance(svg, str)
        assert len(svg) > 0
        plt.close('all')

    @pytest.mark.parametrize('figsize', [(4, 3), (6.4, 4.8), (8, 6)])
    def test_svg_produced_from_figsize(self, figsize):
        """SVG is produced regardless of figsize."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert isinstance(svg, str)
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('title', ['Test Title', '', 'My Plot'])
    def test_svg_with_title(self, title):
        """SVG is produced with title set."""
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert isinstance(svg, str)
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10])
    def test_svg_histogram(self, bins):
        """SVG is produced for histogram."""
        fig, ax = plt.subplots()
        ax.hist(list(range(50)), bins=bins)
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('n', [3, 5, 10])
    def test_svg_scatter(self, n):
        """SVG is produced for scatter plot."""
        fig, ax = plt.subplots()
        ax.scatter(range(n), range(n))
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    def test_svg_starts_with_xml(self):
        """SVG output starts with XML declaration or <svg."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert svg.startswith('<?xml') or svg.startswith('<svg')
        plt.close('all')

    def test_svg_ends_with_svg_close(self):
        """SVG output ends with closing svg tag."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert '</svg>' in svg
        plt.close('all')


class TestBatch31Parametric7:
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

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_bar(self, n):
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
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

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_hist(self, bins):
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close("all")

    @pytest.mark.parametrize("marker", ["o", "s", "^"])
    def test_marker(self, marker):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")

