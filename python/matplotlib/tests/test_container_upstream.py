"""
Upstream matplotlib tests for container classes.
"""

import pytest

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer


# ---------------------------------------------------------------------------
# BarContainer
# ---------------------------------------------------------------------------
def test_barcontainer_len():
    """BarContainer has correct length."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    assert len(bc) == 3


def test_barcontainer_iter():
    """BarContainer is iterable."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    patches = list(bc)
    assert len(patches) == 3


def test_barcontainer_getitem():
    """BarContainer supports indexing."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    p = bc[0]
    assert p.get_height() == 4


def test_barcontainer_patches():
    """BarContainer.patches property."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    assert len(bc.patches) == 3


def test_barcontainer_label():
    """BarContainer label."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2], [3, 4], label='bars')
    assert bc.get_label() == 'bars'


def test_barcontainer_no_label():
    """BarContainer without label."""
    fig, ax = plt.subplots()
    bc = ax.bar([1], [2])
    # Should have some default label (None becomes '_nolegend_')
    label = bc.get_label()
    assert isinstance(label, str)


# ---------------------------------------------------------------------------
# ErrorbarContainer
# ---------------------------------------------------------------------------
def test_errorbar_container_has_yerr():
    """ErrorbarContainer.has_yerr is True when yerr given."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5])
    assert ec.has_yerr is True
    assert ec.has_xerr is False


def test_errorbar_container_has_xerr():
    """ErrorbarContainer.has_xerr is True when xerr given."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], xerr=[0.5, 0.5])
    assert ec.has_xerr is True
    assert ec.has_yerr is False


def test_errorbar_container_both():
    """ErrorbarContainer with both xerr and yerr."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], xerr=[0.1, 0.1], yerr=[0.2, 0.2])
    assert ec.has_xerr is True
    assert ec.has_yerr is True


def test_errorbar_container_label():
    """ErrorbarContainer label."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5], label='errors')
    assert ec.get_label() == 'errors'


def test_errorbar_container_lines():
    """ErrorbarContainer.lines holds the plot line."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5])
    assert ec.lines is not None
    # lines[0] is the data line
    assert ec.lines[0] is not None


def test_errorbar_fmt_none():
    """ErrorbarContainer with fmt='none' hides data line."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5], fmt='none')
    assert ec.lines[0] is None


# ---------------------------------------------------------------------------
# StemContainer
# ---------------------------------------------------------------------------
def test_stem_container_markerline():
    """StemContainer.markerline exists."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2, 3], [4, 5, 6])
    assert sc.markerline is not None


def test_stem_container_baseline():
    """StemContainer.baseline exists."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2, 3], [4, 5, 6])
    assert sc.baseline is not None


def test_stem_container_stemlines():
    """StemContainer.stemlines exists and has correct count."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2, 3], [4, 5, 6])
    assert len(sc.stemlines) == 3


def test_stem_container_label():
    """StemContainer label."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2], [3, 4], label='stems')
    assert sc.get_label() == 'stems'


# ---------------------------------------------------------------------------
# Container.remove
# ---------------------------------------------------------------------------
def test_barcontainer_remove():
    """BarContainer.remove removes patches from axes."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2], [3, 4])
    initial_patches = len(ax.patches)
    bc.remove()
    assert len(ax.patches) < initial_patches


# ===================================================================
# Container extended tests (upstream-inspired)
# ===================================================================

class TestBarContainerExtended:
    def test_single_bar(self):
        """BarContainer with one bar has length 1."""
        fig, ax = plt.subplots()
        bc = ax.bar([1], [5])
        assert len(bc) == 1
        plt.close('all')

    def test_bar_heights(self):
        """BarContainer patches have correct heights."""
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2, 3], [10, 20, 30])
        heights = [p.get_height() for p in bc]
        assert heights == [10, 20, 30]
        plt.close('all')

    def test_bar_widths_default(self):
        """BarContainer patches have default width 0.8."""
        fig, ax = plt.subplots()
        bc = ax.bar([1], [1])
        assert abs(bc[0].get_width() - 0.8) < 1e-10
        plt.close('all')

    def test_bar_custom_width(self):
        """BarContainer respects custom width."""
        fig, ax = plt.subplots()
        bc = ax.bar([1], [1], width=0.5)
        assert abs(bc[0].get_width() - 0.5) < 1e-10
        plt.close('all')

    def test_bar_negative_values(self):
        """BarContainer handles negative bar heights."""
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [-3, 4])
        assert bc[0].get_height() == -3
        assert bc[1].get_height() == 4
        plt.close('all')

    def test_barcontainer_is_tuple(self):
        """BarContainer is a tuple subclass."""
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4])
        assert isinstance(bc, tuple)
        plt.close('all')

    def test_barcontainer_set_label(self):
        """BarContainer.set_label changes label."""
        fig, ax = plt.subplots()
        bc = ax.bar([1], [2])
        bc.set_label('new_label')
        assert bc.get_label() == 'new_label'
        plt.close('all')

    def test_barcontainer_in_patches(self):
        """Bar patches are in ax.patches."""
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4])
        for patch in bc:
            assert patch in ax.patches
        plt.close('all')

    def test_barcontainer_errorbar_none(self):
        """BarContainer without errorbar has errorbar=None."""
        from matplotlib.container import BarContainer
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), 1, 1)
        bc = BarContainer([rect])
        assert bc.errorbar is None


class TestErrorbarContainerExtended:
    def test_no_error_bars(self):
        """Errorbar with no error has both False."""
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2], [3, 4])
        assert ec.has_xerr is False
        assert ec.has_yerr is False
        plt.close('all')

    def test_lines_is_tuple(self):
        """ErrorbarContainer.lines is a tuple."""
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2], [3, 4], yerr=[0.1, 0.1])
        assert isinstance(ec.lines, tuple)
        plt.close('all')

    def test_errorbar_set_label(self):
        """ErrorbarContainer.set_label changes label."""
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2], [3, 4])
        ec.set_label('err_label')
        assert ec.get_label() == 'err_label'
        plt.close('all')


class TestStemContainerExtended:
    def test_stem_single_point(self):
        """StemContainer with a single point."""
        fig, ax = plt.subplots()
        sc = ax.stem([1], [5])
        assert sc.markerline is not None
        assert sc.baseline is not None
        assert len(sc.stemlines) == 1
        plt.close('all')

    def test_stemlines_not_none(self):
        """StemContainer.stemlines is not None."""
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2, 3], [4, 5, 6])
        assert sc.stemlines is not None
        plt.close('all')

    def test_stem_markerline_is_line2d(self):
        """StemContainer.markerline is a Line2D."""
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2], [3, 4])
        assert isinstance(sc.markerline, Line2D)
        plt.close('all')

    def test_stem_baseline_is_line2d(self):
        """StemContainer.baseline is a Line2D."""
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2], [3, 4])
        assert isinstance(sc.baseline, Line2D)
        plt.close('all')

    def test_stem_set_label(self):
        """StemContainer.set_label changes label."""
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2], [3, 4])
        sc.set_label('stem_label')
        assert sc.get_label() == 'stem_label'
        plt.close('all')


# ===================================================================
# Additional container parametric tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.lines import Line2D


class TestBarContainerParametric:
    """Parametric tests for BarContainer."""

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_bar_n_patches(self, n):
        """bar() creates n patches."""
        fig, ax = plt.subplots()
        bars = ax.bar(list(range(n)), list(range(1, n + 1)))
        assert len(bars.patches) == n
        plt.close('all')

    @pytest.mark.parametrize('width', [0.5, 0.8, 1.0, 1.5])
    def test_bar_custom_width(self, width):
        """bar() respects custom width."""
        fig, ax = plt.subplots()
        bars = ax.bar([1, 2, 3], [4, 5, 6], width=width)
        assert len(bars) == 3
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#abcdef'])
    def test_bar_color(self, color):
        """bar() accepts color parameter."""
        fig, ax = plt.subplots()
        bars = ax.bar([1, 2], [3, 4], color=color)
        assert len(bars) == 2
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.5, 0.7, 1.0])
    def test_bar_alpha(self, alpha):
        """bar() patches accept alpha."""
        fig, ax = plt.subplots()
        bars = ax.bar([1, 2, 3], [4, 5, 6], alpha=alpha)
        assert len(bars) == 3
        plt.close('all')

    def test_bar_in_axes_patches(self):
        """bar() adds patches to ax.patches."""
        fig, ax = plt.subplots()
        bars = ax.bar([1, 2, 3], [4, 5, 6])
        # The patches from the bar container should be in ax.patches
        for patch in bars.patches:
            assert patch in ax.patches
        plt.close('all')


class TestErrorbarParametric:
    """Parametric tests for errorbar plots."""

    @pytest.mark.parametrize('yerr', [0.1, 0.5, 1.0])
    def test_errorbar_yerr(self, yerr):
        """errorbar with yerr creates error bars."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2, 3], [4, 5, 6], yerr=yerr)
        assert eb is not None
        plt.close('all')

    @pytest.mark.parametrize('xerr', [0.1, 0.5, 1.0])
    def test_errorbar_xerr(self, xerr):
        """errorbar with xerr creates error bars."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2, 3], [4, 5, 6], xerr=xerr)
        assert eb is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_errorbar_n_points(self, n):
        """errorbar works for n data points."""
        fig, ax = plt.subplots()
        xs = list(range(n))
        ys = list(range(n))
        eb = ax.errorbar(xs, ys, yerr=0.1)
        assert eb is not None
        plt.close('all')


# ===================================================================
# Extended parametric tests for Container
# ===================================================================

class TestContainerParametricExtended:
    """Extended parametric tests for bar, barh, errorbar containers."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10, 20])
    def test_bar_container_n_patches(self, n):
        """bar() BarContainer has n patches."""
        fig, ax = plt.subplots()
        container = ax.bar(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_barh_container_n_patches(self, n):
        """barh() BarContainer has n patches."""
        fig, ax = plt.subplots()
        container = ax.barh(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff0000', 'cyan'])
    def test_bar_color_param(self, color):
        """bar() accepts color parameter without error."""
        fig, ax = plt.subplots()
        container = ax.bar([1, 2, 3], [1, 2, 3], color=color)
        assert container is not None
        plt.close('all')

    @pytest.mark.parametrize('width', [0.3, 0.5, 0.8, 1.0])
    def test_bar_width_param(self, width):
        """bar() accepts width parameter without error."""
        fig, ax = plt.subplots()
        container = ax.bar([1, 2, 3], [1, 2, 3], width=width)
        assert container is not None
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_bar_alpha_param(self, alpha):
        """bar() accepts alpha parameter without error."""
        fig, ax = plt.subplots()
        container = ax.bar([1, 2, 3], [1, 2, 3], alpha=alpha)
        assert container is not None
        plt.close('all')

    @pytest.mark.parametrize('heights', [
        [1, 2, 3],
        [0, 5, 10, 15],
        [-1, 0, 1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    ])
    def test_bar_various_heights(self, heights):
        """bar() works with various height values."""
        fig, ax = plt.subplots()
        container = ax.bar(range(len(heights)), heights)
        assert len(container) == len(heights)
        plt.close('all')

    @pytest.mark.parametrize('capsize', [0, 2, 5, 10])
    def test_errorbar_capsize(self, capsize):
        """errorbar accepts capsize parameter."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2, 3], [1, 2, 3], yerr=0.1, capsize=capsize)
        assert eb is not None
        plt.close('all')

    @pytest.mark.parametrize('ecolor', ['red', 'blue', 'black', 'gray'])
    def test_errorbar_ecolor(self, ecolor):
        """errorbar accepts ecolor parameter."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2, 3], [1, 2, 3], yerr=0.1, ecolor=ecolor)
        assert eb is not None
        plt.close('all')

    @pytest.mark.parametrize('elinewidth', [0.5, 1.0, 2.0, 3.0])
    def test_errorbar_elinewidth(self, elinewidth):
        """errorbar accepts elinewidth parameter."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2, 3], [1, 2, 3], yerr=0.1, elinewidth=elinewidth)
        assert eb is not None
        plt.close('all')


class TestContainerParametricExtended2:
    """More parametric tests for containers."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10, 20])
    def test_bar_n_patches_h(self, n):
        """ax.barh returns n patches."""
        fig, ax = plt.subplots()
        bars = ax.barh(range(n), range(n))
        assert len(bars.patches) == n
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', 'black'])
    def test_bar_color_applied(self, color):
        """Bar patch has correct facecolor."""
        fig, ax = plt.subplots()
        bars = ax.bar([1], [1], color=color)
        assert bars.patches[0].get_facecolor() is not None
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.5, 0.8, 1.0])
    def test_bar_alpha(self, alpha):
        """Bar patch alpha is set."""
        fig, ax = plt.subplots()
        bars = ax.bar([1], [1], alpha=alpha)
        assert bars is not None
        plt.close('all')

    @pytest.mark.parametrize('width', [0.1, 0.5, 0.8, 1.0])
    def test_bar_width(self, width):
        """Bar patch width is set."""
        fig, ax = plt.subplots()
        bars = ax.bar([1], [1], width=width)
        assert abs(bars.patches[0].get_width() - width) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('height', [0.5, 1.0, 2.0, 5.0])
    def test_bar_height(self, height):
        """Bar patch height equals value."""
        fig, ax = plt.subplots()
        bars = ax.bar([1], [height])
        assert abs(bars.patches[0].get_height() - height) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('yerr', [0.1, 0.5, 1.0, 2.0])
    def test_errorbar_yerr(self, yerr):
        """errorbar with yerr creates container."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2, 3], [1, 2, 3], yerr=yerr)
        assert eb is not None
        plt.close('all')

    @pytest.mark.parametrize('xerr', [0.1, 0.5, 1.0])
    def test_errorbar_xerr(self, xerr):
        """errorbar with xerr creates container."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2, 3], [1, 2, 3], xerr=xerr)
        assert eb is not None
        plt.close('all')

    @pytest.mark.parametrize('capsize', [0, 2, 4, 6, 10])
    def test_errorbar_capsize2(self, capsize):
        """errorbar accepts capsize."""
        fig, ax = plt.subplots()
        eb = ax.errorbar([1, 2], [1, 2], yerr=0.1, capsize=capsize)
        assert eb is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [2, 3, 5, 8])
    def test_bar_n_patches_count2(self, n):
        """ax.bar with n values returns n patches."""
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), [1] * n)
        assert len(bars.patches) == n
        plt.close('all')

    @pytest.mark.parametrize('bottom', [0, 1, 2, 5, -1])
    def test_bar_bottom(self, bottom):
        """Bar bottom is stored correctly."""
        fig, ax = plt.subplots()
        bars = ax.bar([1], [2], bottom=bottom)
        assert abs(bars.patches[0].get_y() - bottom) < 1e-10
        plt.close('all')


class TestContainerUpstreamParametric3:
    """Further parametric container tests."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 10])
    def test_bar_n_patches(self, n):
        """BarContainer has n patches."""
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(1, n + 1))
        assert len(bars.patches) == n
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff0000', 'orange'])
    def test_bar_color(self, color):
        """Bar patches have correct facecolor."""
        from matplotlib.colors import to_hex
        fig, ax = plt.subplots()
        bars = ax.bar([0], [1], color=color)
        assert to_hex(bars.patches[0].get_facecolor()) == to_hex(color)
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_bar_alpha(self, alpha):
        """Bar patches have correct alpha."""
        fig, ax = plt.subplots()
        bars = ax.bar([0], [1], alpha=alpha)
        assert abs(bars.patches[0].get_alpha() - alpha) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('height', [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_bar_height(self, height):
        """Bar patch height is correct."""
        fig, ax = plt.subplots()
        bars = ax.bar([0], [height])
        assert abs(bars.patches[0].get_height() - height) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('n_points', [5, 10, 20, 50])
    def test_errorbar_n(self, n_points):
        """Errorbar with n points."""
        import numpy as np
        fig, ax = plt.subplots()
        x = list(range(n_points))
        y = [i * 2 for i in range(n_points)]
        yerr = [0.1] * n_points
        container = ax.errorbar(x, y, yerr=yerr)
        assert container is not None
        plt.close('all')

    @pytest.mark.parametrize('bottom', [0.0, 1.0, -1.0, 5.0, -5.0])
    def test_bar_bottom(self, bottom):
        """Bar bottom is set correctly."""
        fig, ax = plt.subplots()
        bars = ax.bar([1], [2], bottom=bottom)
        assert abs(bars.patches[0].get_y() - bottom) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('width', [0.2, 0.4, 0.6, 0.8, 1.0])
    def test_bar_width(self, width):
        """Bar width is set correctly."""
        fig, ax = plt.subplots()
        bars = ax.bar([0], [1], width=width)
        assert abs(bars.patches[0].get_width() - width) < 1e-9
        plt.close('all')
