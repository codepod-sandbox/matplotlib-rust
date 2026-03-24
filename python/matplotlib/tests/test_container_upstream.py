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
