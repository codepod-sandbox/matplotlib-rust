"""Tests for matplotlib.figure.Figure class.

Covers figure creation, numbering, suptitle, sizing, axes management,
DPI, clear/clf, repr, and savefig.
"""

import os

import numpy as np
import pytest

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@pytest.fixture(autouse=True)
def _reset_pyplot_state():
    """Reset all pyplot global state before each test."""
    plt.close('all')
    plt._next_num = 1
    matplotlib._interactive = False
    yield
    plt.close('all')
    plt._next_num = 1
    matplotlib._interactive = False


# ===================================================================
# Figure Creation (3 tests)
# ===================================================================

class TestFigureCreation:
    def test_figure_default_size(self):
        """Default figsize is (6.4, 4.8)."""
        fig = plt.figure()
        assert np.allclose(fig.get_size_inches(), (6.4, 4.8))

    def test_figure_custom_size(self):
        """figure(figsize=(10, 8)) sets correct size."""
        fig = plt.figure(figsize=(10, 8))
        assert np.allclose(fig.get_size_inches(), (10.0, 8.0))

    def test_figure_dpi(self):
        """figure(dpi=200) sets correct dpi."""
        fig = plt.figure(dpi=200)
        assert fig.get_dpi() == 200


# ===================================================================
# Figure Numbering (4 tests)
# ===================================================================

class TestFigureNumbering:
    def test_figure_label(self):
        """Figures created with string labels appear in get_figlabels."""
        plt.figure('a')
        plt.figure('b')
        assert plt.get_figlabels() == ['a', 'b']

    def test_fignum_exists(self):
        """fignum_exists returns True after creation, False after close."""
        fig = plt.figure()
        num = fig.number
        assert plt.fignum_exists(num) is True
        plt.close(num)
        assert plt.fignum_exists(num) is False

    def test_auto_numbering(self):
        """Figures get sequential numbers starting from 1."""
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        assert fig1.number == 1
        assert fig2.number == 2
        assert fig3.number == 3

    def test_figure_reactivation(self):
        """plt.figure(num=existing) returns the same figure, not a new one."""
        fig1 = plt.figure()
        fig2 = plt.figure()
        reactivated = plt.figure(num=fig1.number)
        assert reactivated is fig1
        # No new figure was created
        assert plt.get_fignums() == [1, 2]


# ===================================================================
# Suptitle (3 tests)
# ===================================================================

class TestSuptitle:
    def test_suptitle(self):
        """fig.suptitle('Hello') is retrievable via get_suptitle()."""
        fig = plt.figure()
        fig.suptitle('Hello')
        assert fig.get_suptitle() == 'Hello'

    def test_suptitle_default_empty(self):
        """get_suptitle() on a new figure returns ''."""
        fig = plt.figure()
        assert fig.get_suptitle() == ''

    def test_suptitle_cleared_on_clear(self):
        """suptitle is reset to '' after clear()."""
        fig = plt.figure()
        fig.suptitle('Title')
        assert fig.get_suptitle() == 'Title'
        fig.clear()
        assert fig.get_suptitle() == ''


# ===================================================================
# Sizing (4 tests)
# ===================================================================

class TestSizing:
    def test_set_figwidth(self):
        """set_figwidth(10) makes get_figwidth() return 10."""
        fig = plt.figure()
        fig.set_figwidth(10)
        assert fig.get_figwidth() == 10.0

    def test_set_figheight(self):
        """set_figheight(8) makes get_figheight() return 8."""
        fig = plt.figure()
        fig.set_figheight(8)
        assert fig.get_figheight() == 8.0

    def test_set_size_inches_two_args(self):
        """set_size_inches(12, 6) sets both width and height."""
        fig = plt.figure()
        fig.set_size_inches(12, 6)
        assert np.allclose(fig.get_size_inches(), (12.0, 6.0))

    def test_set_size_inches_tuple(self):
        """set_size_inches((5, 4)) accepts a tuple."""
        fig = plt.figure()
        fig.set_size_inches((5, 4))
        assert np.allclose(fig.get_size_inches(), (5.0, 4.0))


# ===================================================================
# Axes Management (5 tests)
# ===================================================================

class TestAxesManagement:
    def test_add_subplot(self):
        """add_subplot adds an Axes to the figure's axes list."""
        fig = plt.figure()
        assert len(fig.get_axes()) == 0
        ax = fig.add_subplot(1, 1, 1)
        assert len(fig.get_axes()) == 1
        assert fig.get_axes()[0] is ax

    def test_add_axes(self):
        """add_axes adds an Axes to the figure's axes list."""
        fig = plt.figure()
        assert len(fig.get_axes()) == 0
        ax = fig.add_axes([0, 0, 1, 1])
        assert len(fig.get_axes()) == 1
        assert fig.get_axes()[0] is ax

    def test_axes_property(self):
        """The axes property returns a copy of the internal list."""
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        axes_list = fig.axes
        assert isinstance(axes_list, list)
        assert len(axes_list) == 1
        # Modifying the returned list must not affect the figure
        axes_list.clear()
        assert len(fig.axes) == 1

    def test_delaxes(self):
        """delaxes removes the specified Axes from the figure."""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        assert len(fig.get_axes()) == 2
        fig.delaxes(ax1)
        assert len(fig.get_axes()) == 1
        assert ax1 not in fig.get_axes()
        assert ax2 in fig.get_axes()

    def test_axes_remove(self):
        """ax.remove() removes the Axes from its parent figure."""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        assert len(fig.get_axes()) == 2
        ax1.remove()
        assert len(fig.get_axes()) == 1
        assert ax1 not in fig.get_axes()


# ===================================================================
# DPI (2 tests)
# ===================================================================

class TestDPI:
    def test_get_set_dpi(self):
        """set_dpi(150) makes get_dpi() return 150."""
        fig = plt.figure()
        fig.set_dpi(150)
        assert fig.get_dpi() == 150

    def test_dpi_in_repr(self):
        """repr shows correct pixel dimensions based on DPI."""
        fig = plt.figure(figsize=(10, 5), dpi=100)
        r = repr(fig)
        # 10*100=1000, 5*100=500
        assert '1000x500' in r
        fig.set_dpi(200)
        r = repr(fig)
        # 10*200=2000, 5*200=1000
        assert '2000x1000' in r


# ===================================================================
# Clear (3 tests)
# ===================================================================

class TestClear:
    def test_clear(self):
        """clear() removes all axes from the figure."""
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        assert len(fig.get_axes()) == 1
        fig.clear()
        assert fig.get_axes() == []

    def test_clf_alias(self):
        """clf() behaves the same as clear()."""
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        assert len(fig.get_axes()) == 1
        fig.clf()
        assert fig.get_axes() == []

    def test_clear_resets_suptitle(self):
        """clear() resets the suptitle to ''."""
        fig = plt.figure()
        fig.suptitle('test')
        assert fig.get_suptitle() == 'test'
        fig.clear()
        assert fig.get_suptitle() == ''


# ===================================================================
# Repr (1 test)
# ===================================================================

class TestRepr:
    def test_repr(self):
        """repr follows '<Figure size WxH with N Axes>' format."""
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        assert repr(fig) == '<Figure size 640x480 with 0 Axes>'
        fig.add_subplot(1, 1, 1)
        assert repr(fig) == '<Figure size 640x480 with 1 Axes>'
        fig.add_subplot(1, 2, 2)
        assert repr(fig) == '<Figure size 640x480 with 2 Axes>'


# ===================================================================
# Savefig (2 tests)
# ===================================================================

class TestSavefig:
    def test_savefig_svg(self, tmp_path):
        """savefig creates a valid SVG file."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3], [4, 5, 6])
        path = tmp_path / 'test.svg'
        fig.savefig(str(path))
        assert path.exists()
        content = path.read_text()
        # OG SVG backend emits <?xml ... ?> declaration before <svg
        assert '<svg' in content
        assert '</svg>' in content

    def test_savefig_format_detection(self, tmp_path):
        """savefig infers SVG format from .svg extension."""
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        svg_path = tmp_path / 'output.svg'
        fig.savefig(str(svg_path))
        assert svg_path.exists()
        content = svg_path.read_text()
        # Confirm it wrote SVG content (not PNG binary)
        assert '<svg' in content


# ===================================================================
# Additional figure tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
