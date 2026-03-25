"""Tests for matplotlib.pyplot state-machine API.

Covers interactive mode, figure management, close, subplot/axes,
clear, figure properties, labels, limits, and auto-numbering.
"""

import pytest

import matplotlib
import matplotlib.pyplot as plt


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
# Interactive mode (5 tests)
# ===================================================================

class TestInteractiveMode:
    def test_ioff(self):
        """ion() then ioff() makes interactive False."""
        plt.ion()
        assert plt.isinteractive() is True
        plt.ioff()
        assert plt.isinteractive() is False

    def test_ion(self):
        """ioff() then ion() makes interactive True."""
        plt.ioff()
        assert plt.isinteractive() is False
        plt.ion()
        assert plt.isinteractive() is True

    def test_ioff_context(self):
        """ioff() as context manager restores interactive state on exit."""
        plt.ion()
        assert plt.isinteractive() is True
        with plt.ioff():
            assert plt.isinteractive() is False
        assert plt.isinteractive() is True

    def test_ion_context(self):
        """ion() as context manager restores interactive state on exit."""
        plt.ioff()
        assert plt.isinteractive() is False
        with plt.ion():
            assert plt.isinteractive() is True
        assert plt.isinteractive() is False

    def test_nested_ion_ioff(self):
        """Nested context managers preserve and restore state correctly."""
        plt.ioff()
        assert plt.isinteractive() is False
        with plt.ion():
            assert plt.isinteractive() is True
            with plt.ioff():
                assert plt.isinteractive() is False
            # After inner context exits, back to True
            assert plt.isinteractive() is True
        # After outer context exits, back to False
        assert plt.isinteractive() is False


# ===================================================================
# Close (3 tests)
# ===================================================================

class TestClose:
    def test_close_all(self):
        """close('all') removes all figures."""
        plt.figure()
        plt.figure()
        assert len(plt.get_fignums()) == 2
        plt.close('all')
        assert plt.get_fignums() == []

    def test_close_by_num(self):
        """close(num) removes only the specified figure."""
        plt.figure()
        plt.figure()
        assert plt.get_fignums() == [1, 2]
        plt.close(1)
        assert plt.get_fignums() == [2]

    def test_close_float_raises(self):
        """close(1.0) raises TypeError."""
        with pytest.raises(TypeError, match="float"):
            plt.close(1.0)


# ===================================================================
# Figure management (4 tests)
# ===================================================================

class TestFigureManagement:
    def test_figure_label(self):
        """Figures created with string labels appear in get_figlabels."""
        plt.figure('a')
        plt.figure('b')
        assert plt.get_figlabels() == ['a', 'b']

    def test_fignum_exists(self):
        """fignum_exists returns True/False correctly."""
        fig = plt.figure()
        num = fig.number
        assert plt.fignum_exists(num) is True
        plt.close(num)
        assert plt.fignum_exists(num) is False

    def test_gca(self):
        """fig.gca() returns the same axes each call (no duplicates)."""
        fig = plt.figure()
        ax1 = fig.gca()
        ax2 = fig.gca()
        assert ax1 is ax2
        assert len(fig.get_axes()) == 1

    def test_gcf(self):
        """plt.gcf() returns the most recently created/activated figure."""
        fig1 = plt.figure()
        fig2 = plt.figure()
        assert plt.gcf() is fig2


# ===================================================================
# Subplot and Axes (2 tests)
# ===================================================================

class TestSubplotAxes:
    def test_subplot_reuse(self):
        """subplot(1,1,1) called twice returns the same axes."""
        plt.figure()
        ax1 = plt.subplot(1, 1, 1)
        ax2 = plt.subplot(1, 1, 1)
        assert ax1 is ax2

    def test_axes_always_new(self):
        """axes() called twice always returns different axes."""
        plt.figure()
        ax1 = plt.axes()
        ax2 = plt.axes()
        assert ax1 is not ax2


# ===================================================================
# Clear (2 tests)
# ===================================================================

class TestClear:
    def test_clf(self):
        """clf() removes all axes from the current figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        assert len(fig.get_axes()) > 0
        plt.clf()
        assert fig.get_axes() == []

    def test_cla(self):
        """cla() clears the current axes' elements."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        assert len(ax.lines) > 0 or len(ax.patches) > 0 or len(ax.collections) > 0
        plt.cla()
        assert ax.lines == [] and ax.patches == [] and ax.collections == [] and ax.texts == []
        assert ax.get_title() == ''
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''


# ===================================================================
# Figure properties (5 tests)
# ===================================================================

class TestFigureProperties:
    def test_set_fig_size(self):
        """set_figwidth, set_figheight, set_size_inches work correctly."""
        fig = plt.figure()
        # set_figwidth / set_figheight
        fig.set_figwidth(10)
        assert fig.get_figwidth() == 10.0
        fig.set_figheight(8)
        assert fig.get_figheight() == 8.0
        assert fig.get_size_inches() == (10.0, 8.0)
        # set_size_inches with two args
        fig.set_size_inches(5, 4)
        assert fig.get_size_inches() == (5.0, 4.0)
        # set_size_inches with tuple
        fig.set_size_inches((7, 3))
        assert fig.get_size_inches() == (7.0, 3.0)

    def test_suptitle(self):
        """fig.suptitle() sets and get_suptitle() retrieves the suptitle."""
        fig = plt.figure()
        fig.suptitle('Hello')
        assert fig.get_suptitle() == 'Hello'

    def test_figure_repr(self):
        """Figure repr contains size and axes count."""
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        r = repr(fig)
        assert '640x480' in r
        assert '0 Axes' in r
        fig.add_subplot(1, 1, 1)
        r = repr(fig)
        assert '1 Axes' in r

    def test_axes_remove(self):
        """delaxes removes one axes, leaving the others."""
        fig, axes_list = plt.subplots(2, 2)
        assert len(fig.get_axes()) == 4
        # axes_list is a list of lists for 2x2
        ax_to_remove = fig.get_axes()[0]
        fig.delaxes(ax_to_remove)
        assert len(fig.get_axes()) == 3
        assert ax_to_remove not in fig.get_axes()

    def test_figure_clear(self):
        """figure.clear() removes axes and resets suptitle."""
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        fig.suptitle('Title')
        assert len(fig.get_axes()) == 1
        assert fig.get_suptitle() == 'Title'
        fig.clear()
        assert fig.get_axes() == []
        assert fig.get_suptitle() == ''


# ===================================================================
# Additional tests (9 tests)
# ===================================================================

class TestAdditional:
    def test_get_set_labels(self):
        """set_xlabel/set_ylabel reflected by get_xlabel/get_ylabel."""
        fig, ax = plt.subplots()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        assert ax.get_xlabel() == 'X Label'
        assert ax.get_ylabel() == 'Y Label'

    def test_xlim_ylim(self):
        """plt.xlim() and plt.ylim() work as both getters and setters."""
        fig, ax = plt.subplots()
        # Set limits
        plt.xlim(0, 10)
        plt.ylim(-5, 5)
        # Get limits
        assert plt.xlim() == (0, 10)
        assert plt.ylim() == (-5, 5)

    def test_figure_number_auto(self):
        """Figures get auto-incrementing numbers starting from 1."""
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        assert fig1.number == 1
        assert fig2.number == 2
        assert fig3.number == 3

    def test_close_by_figure_instance(self):
        """close(fig) removes that specific figure from tracking."""
        fig1 = plt.figure()
        fig2 = plt.figure()
        assert plt.get_fignums() == [1, 2]
        plt.close(fig1)
        assert plt.get_fignums() == [2]
        assert not plt.fignum_exists(fig1.number)

    def test_reactivate_figure(self):
        """figure(num) with existing num reactivates it without creating new."""
        fig1 = plt.figure()
        fig2 = plt.figure()
        assert plt.gcf() is fig2
        reactivated = plt.figure(fig1.number)
        assert reactivated is fig1
        assert plt.gcf() is fig1
        # No new figure was created
        assert plt.get_fignums() == [1, 2]

    def test_suptitle_via_pyplot(self):
        """plt.suptitle() delegates to current figure."""
        fig = plt.figure()
        plt.suptitle('Test Title')
        assert fig.get_suptitle() == 'Test Title'

    def test_subplot_3digit(self):
        """subplot(211) is equivalent to subplot(2, 1, 1)."""
        fig = plt.figure()
        ax1 = plt.subplot(211)
        assert ax1._position == (2, 1, 1)

    def test_subplots_grid(self):
        """subplots(2, 2) returns a list of 4 axes in correct shape."""
        fig, axes_list = plt.subplots(2, 2)
        # 2x2 returns list of lists
        assert len(axes_list) == 2
        assert len(axes_list[0]) == 2
        assert len(fig.get_axes()) == 4

    def test_figure_with_clear(self):
        """figure(num, clear=True) clears an existing figure."""
        fig = plt.figure(1)
        fig.add_subplot(1, 1, 1)
        fig.suptitle('Old Title')
        assert len(fig.get_axes()) == 1
        fig2 = plt.figure(1, clear=True)
        assert fig2 is fig
        assert fig2.get_axes() == []
        assert fig2.get_suptitle() == ''


# ===================================================================
# Additional pyplot tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib
import matplotlib.pyplot as plt


class TestPyplotPlottingAPI:
    """Tests for pyplot plotting functions."""

    def test_plot_returns_list(self):
        """plt.plot returns a list of Line2D."""
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6])
        assert isinstance(lines, list)
        assert len(lines) == 1
        assert isinstance(lines[0], Line2D)
        plt.close('all')

    def test_scatter_returns_collection(self):
        """ax.scatter returns a PathCollection."""
        from matplotlib.collections import PathCollection
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [4, 5, 6])
        assert isinstance(sc, PathCollection)
        plt.close('all')

    def test_bar_returns_container(self):
        """ax.bar returns a BarContainer."""
        from matplotlib.container import BarContainer
        fig, ax = plt.subplots()
        bars = ax.bar([1, 2, 3], [4, 5, 6])
        assert isinstance(bars, BarContainer)
        plt.close('all')

    def test_axhline_returns_line(self):
        """ax.axhline returns a Line2D."""
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        line = ax.axhline(0.5)
        assert isinstance(line, Line2D)
        plt.close('all')

    def test_axvline_returns_line(self):
        """ax.axvline returns a Line2D."""
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        line = ax.axvline(0.5)
        assert isinstance(line, Line2D)
        plt.close('all')

    def test_fill_between_returns_artist(self):
        """ax.fill_between returns an artist."""
        from matplotlib.artist import Artist
        fig, ax = plt.subplots()
        pc = ax.fill_between([0, 1, 2], [0, 1, 0], [1, 2, 1])
        assert isinstance(pc, Artist)
        plt.close('all')

    def test_title_set_and_get(self):
        """ax.set_title / ax.get_title roundtrip."""
        fig, ax = plt.subplots()
        ax.set_title('My Plot')
        assert ax.get_title() == 'My Plot'
        plt.close('all')

    def test_xlabel_set_and_get(self):
        """ax.set_xlabel / ax.get_xlabel roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlabel('X axis')
        assert ax.get_xlabel() == 'X axis'
        plt.close('all')

    def test_ylabel_set_and_get(self):
        """ax.set_ylabel / ax.get_ylabel roundtrip."""
        fig, ax = plt.subplots()
        ax.set_ylabel('Y axis')
        assert ax.get_ylabel() == 'Y axis'
        plt.close('all')

    def test_multiple_lines_same_axes(self):
        """Multiple plot calls on same axes accumulate lines."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.plot([5, 6], [7, 8])
        assert len(ax.lines) == 2
        plt.close('all')

    def test_legend_after_labeled_plot(self):
        """Legend can be created after labeled plot."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='series')
        leg = ax.legend()
        assert leg is not None
        plt.close('all')


class TestPyplotParametric:
    """Parametrized pyplot tests."""

    @pytest.mark.parametrize('nrows,ncols', [
        (1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (1, 4)
    ])
    def test_subplots_shape(self, nrows, ncols):
        """plt.subplots returns correct number of axes."""
        fig, axes = plt.subplots(nrows, ncols)
        if nrows == 1 and ncols == 1:
            # Single axes — not a list
            assert hasattr(axes, 'plot')
        elif nrows == 1 or ncols == 1:
            total = nrows * ncols
            assert len(axes) == total
        else:
            assert len(axes) == nrows
            assert len(axes[0]) == ncols
        plt.close('all')

    @pytest.mark.parametrize('figsize', [
        (6.4, 4.8), (10, 8), (4, 3), (12, 6)
    ])
    def test_figure_figsize(self, figsize):
        """plt.figure accepts figsize tuple."""
        w, h = figsize
        fig = plt.figure(figsize=figsize)
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 96, 100, 150, 200])
    def test_figure_dpi(self, dpi):
        """plt.figure accepts dpi argument."""
        fig = plt.figure(dpi=dpi)
        assert fig.get_dpi() == dpi
        plt.close('all')


# ===================================================================
# Extended parametric tests for pyplot
# ===================================================================

class TestPyplotParametric:
    """Parametric tests for pyplot API."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_plot_n_lines_in_axes(self, n):
        """Calling plt.plot n times creates n lines."""
        fig, ax = plt.subplots()
        for i in range(n):
            plt.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20, 50])
    def test_hist_returns_correct_n_bins(self, bins):
        """plt.hist returns n bin counts."""
        fig, ax = plt.subplots()
        n, edges, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close('all')

    @pytest.mark.parametrize('figsize', [(4, 3), (6.4, 4.8), (10, 8), (12, 5)])
    def test_plt_figure_stores_figsize(self, figsize):
        """plt.figure stores figsize."""
        w, h = figsize
        fig = plt.figure(figsize=figsize)
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('title', ['My Title', '', 'Test Title', 'Plot 1'])
    def test_plt_title_on_current_axes(self, title):
        """plt.title sets title on current axes."""
        fig, ax = plt.subplots()
        plt.title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('label', ['X', '', 'Time (s)', 'Distance'])
    def test_plt_xlabel_on_current_axes(self, label):
        """plt.xlabel sets xlabel on current axes."""
        fig, ax = plt.subplots()
        plt.xlabel(label)
        assert ax.get_xlabel() == label
        plt.close('all')

    @pytest.mark.parametrize('label', ['Y', '', 'Amplitude', 'Value'])
    def test_plt_ylabel_on_current_axes(self, label):
        """plt.ylabel sets ylabel on current axes."""
        fig, ax = plt.subplots()
        plt.ylabel(label)
        assert ax.get_ylabel() == label
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100), (-10, 10)])
    def test_plt_xlim_stored(self, xmin, xmax):
        """plt.xlim sets xlim on current axes."""
        fig, ax = plt.subplots()
        plt.xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('ymin,ymax', [(0, 1), (-5, 5), (0, 100), (-100, 0)])
    def test_plt_ylim_stored(self, ymin, ymax):
        """plt.ylim sets ylim on current axes."""
        fig, ax = plt.subplots()
        plt.ylim(ymin, ymax)
        got = ax.get_ylim()
        assert abs(got[0] - ymin) < 1e-10
        assert abs(got[1] - ymax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('nrows,ncols', [(1, 1), (2, 2), (1, 3), (2, 3), (3, 3)])
    def test_plt_subplots_axes_count(self, nrows, ncols):
        """plt.subplots creates correct number of axes."""
        fig, axes = plt.subplots(nrows, ncols)
        assert len(fig.get_axes()) == nrows * ncols
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 96, 100, 150, 200])
    def test_plt_figure_dpi(self, dpi):
        """plt.figure(dpi=...) stores dpi."""
        fig = plt.figure(dpi=dpi)
        assert fig.get_dpi() == dpi
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_plt_xscale_roundtrip(self, scale):
        """plt.xscale roundtrip via ax.get_xscale."""
        fig, ax = plt.subplots()
        plt.xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_plt_yscale_roundtrip(self, scale):
        """plt.yscale roundtrip via ax.get_yscale."""
        fig, ax = plt.subplots()
        plt.yscale(scale)
        assert ax.get_yscale() == scale
        plt.close('all')


class TestPyplotParametric2:
    """More parametric tests for pyplot."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_plt_plot_n_lines(self, n):
        """plt.plot n times gives n lines."""
        fig, ax = plt.subplots()
        for i in range(n):
            plt.plot([i, i+1], [0, 1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('w,h', [(4, 3), (6, 4), (8, 6), (10, 8)])
    def test_plt_figsize(self, w, h):
        """plt.figure stores figsize."""
        fig = plt.figure(figsize=(w, h))
        size = fig.get_size_inches()
        assert abs(size[0] - w) < 1e-10 and abs(size[1] - h) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('title', ['Title', 'My Plot', 'Test Figure', ''])
    def test_plt_title_stored(self, title):
        """plt.title is stored on current axes."""
        fig, ax = plt.subplots()
        plt.title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('label', ['X Axis', 'Time (s)', 'Value', ''])
    def test_plt_xlabel_stored(self, label):
        """plt.xlabel is stored on current axes."""
        fig, ax = plt.subplots()
        plt.xlabel(label)
        assert ax.get_xlabel() == label
        plt.close('all')

    @pytest.mark.parametrize('label', ['Y Axis', 'Amplitude', 'Count', ''])
    def test_plt_ylabel_stored(self, label):
        """plt.ylabel is stored on current axes."""
        fig, ax = plt.subplots()
        plt.ylabel(label)
        assert ax.get_ylabel() == label
        plt.close('all')

    @pytest.mark.parametrize('lo,hi', [(0, 1), (-1, 1), (0, 100), (-5, 5)])
    def test_plt_xlim_stored(self, lo, hi):
        """plt.xlim is stored on current axes."""
        fig, ax = plt.subplots()
        plt.xlim(lo, hi)
        xlim = ax.get_xlim()
        assert abs(xlim[0] - lo) < 1e-10 and abs(xlim[1] - hi) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('lo,hi', [(0, 1), (-1, 1), (0, 100), (-5, 5)])
    def test_plt_ylim_stored(self, lo, hi):
        """plt.ylim is stored on current axes."""
        fig, ax = plt.subplots()
        plt.ylim(lo, hi)
        ylim = ax.get_ylim()
        assert abs(ylim[0] - lo) < 1e-10 and abs(ylim[1] - hi) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_plt_xscale_stored(self, scale):
        """plt.xscale roundtrip."""
        fig, ax = plt.subplots()
        plt.xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_plt_yscale_stored(self, scale):
        """plt.yscale roundtrip."""
        fig, ax = plt.subplots()
        plt.yscale(scale)
        assert ax.get_yscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 96, 100, 150])
    def test_plt_figure_dpi_stored(self, dpi):
        """plt.figure dpi is stored."""
        fig = plt.figure(dpi=dpi)
        assert fig.get_dpi() == dpi
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 15, 20])
    def test_plt_hist_bins(self, bins):
        """plt.hist bins count."""
        import numpy as np
        fig, ax = plt.subplots()
        n, edges, patches = plt.hist(np.random.randn(100), bins=bins)
        assert len(n) == bins
        plt.close('all')


class TestPyplotParametric3:
    """Further parametric pyplot tests."""

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8])
    def test_plot_n_lines(self, n):
        """pyplot.plot n times creates n lines."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff0000', 'cyan', 'magenta'])
    def test_plot_color(self, color):
        """pyplot.plot line has correct color."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert to_hex(line.get_color()) == to_hex(color)
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    def test_plot_linewidth(self, lw):
        """pyplot.plot line has correct linewidth."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'D', 'v', '+', 'x', '*'])
    def test_plot_marker(self, marker):
        """pyplot.plot line has correct marker."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale_yscale(self, scale):
        """pyplot xscale and yscale set without error."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 10, 100], [1, 10, 100])
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        assert ax.get_xscale() == scale
        assert ax.get_yscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('n_bars', [1, 2, 3, 5, 8, 10])
    def test_bar_n(self, n_bars):
        """pyplot.bar creates n patches."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bars = ax.bar(range(n_bars), range(1, n_bars + 1))
        assert len(bars) == n_bars
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 15, 20, 30, 50])
    def test_hist_bins(self, bins):
        """pyplot.hist creates correct number of bins."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        n, edges, patches = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close('all')


class TestPyplotParametric7:
    """Yet more parametric tests."""

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

