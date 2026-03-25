"""Tests for subplot layouts --- sharex, sharey, GridSpec, twinx, label_outer."""

import pytest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec


class TestSharexSharey:
    def test_sharex_true(self):
        """sharex=True links x-limits across all subplots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_xlim(0, 10)
        assert ax2.get_xlim() == (0, 10)
        plt.close('all')

    def test_sharey_true(self):
        """sharey=True links y-limits across all subplots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.set_ylim(-5, 5)
        assert ax2.get_ylim() == (-5, 5)
        plt.close('all')

    def test_sharex_bidirectional(self):
        """Setting limits on either shared axes updates the other."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax2.set_xlim(100, 200)
        assert ax1.get_xlim() == (100, 200)
        plt.close('all')

    def test_sharey_bidirectional(self):
        """Setting limits on either shared axes updates the other."""
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax2.set_ylim(10, 20)
        assert ax1.get_ylim() == (10, 20)
        plt.close('all')

    def test_sharex_grid(self):
        """sharex=True on 2x2 grid links all x-limits."""
        fig, axes = plt.subplots(2, 2, sharex=True)
        axes[0][0].set_xlim(0, 100)
        for row in axes:
            for ax in row:
                assert ax.get_xlim() == (0, 100)
        plt.close('all')

    def test_no_share_independent(self):
        """Without share, axes have independent limits."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_xlim(0, 10)
        ax2.set_xlim(100, 200)
        assert ax1.get_xlim() == (0, 10)
        assert ax2.get_xlim() == (100, 200)
        plt.close('all')


class TestTwinAxes:
    def test_twinx_creates_axes(self):
        """twinx() creates a new Axes on the same Figure."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        assert isinstance(ax2, Axes)
        assert ax2.figure is fig
        plt.close('all')

    def test_twinx_shares_x(self):
        """twinx() shares x-axis with parent."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlim(0, 10)
        assert ax2.get_xlim() == (0, 10)
        plt.close('all')

    def test_twinx_independent_y(self):
        """twinx() has independent y-axis."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_ylim(0, 10)
        ax2.set_ylim(100, 200)
        assert ax1.get_ylim() == (0, 10)
        assert ax2.get_ylim() == (100, 200)
        plt.close('all')

    def test_twiny_creates_axes(self):
        """twiny() creates a new Axes on the same Figure."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        assert isinstance(ax2, Axes)
        assert ax2.figure is fig
        plt.close('all')

    def test_twiny_shares_y(self):
        """twiny() shares y-axis with parent."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.set_ylim(0, 10)
        assert ax2.get_ylim() == (0, 10)
        plt.close('all')

    def test_twiny_independent_x(self):
        """twiny() has independent x-axis."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.set_xlim(0, 10)
        ax2.set_xlim(100, 200)
        assert ax1.get_xlim() == (0, 10)
        assert ax2.get_xlim() == (100, 200)
        plt.close('all')


class TestGridSpec:
    def test_gridspec_creation(self):
        """GridSpec(nrows, ncols) creates a grid specification."""
        gs = GridSpec(2, 3)
        assert gs.nrows == 2
        assert gs.ncols == 3

    def test_gridspec_indexing(self):
        """gs[row, col] returns a SubplotSpec."""
        gs = GridSpec(2, 2)
        ss = gs[0, 0]
        assert isinstance(ss, SubplotSpec)

    def test_gridspec_row_slice(self):
        """gs[0, :] spans the full first row."""
        gs = GridSpec(2, 3)
        ss = gs[0, :]
        assert ss.rowspan == (0, 1)
        assert ss.colspan == (0, 3)

    def test_gridspec_col_slice(self):
        """gs[:, 0] spans the full first column."""
        gs = GridSpec(2, 3)
        ss = gs[:, 0]
        assert ss.rowspan == (0, 2)
        assert ss.colspan == (0, 1)

    def test_gridspec_block(self):
        """gs[0:2, 0:2] spans a 2x2 block."""
        gs = GridSpec(3, 3)
        ss = gs[0:2, 0:2]
        assert ss.rowspan == (0, 2)
        assert ss.colspan == (0, 2)

    def test_add_subplot_with_subplotspec(self):
        """Figure.add_subplot(SubplotSpec) creates an axes."""
        fig = plt.figure()
        gs = GridSpec(2, 2)
        ax = fig.add_subplot(gs[0, 0])
        assert isinstance(ax, Axes)
        assert ax.figure is fig
        plt.close('all')

    def test_gridspec_different_spans(self):
        """GridSpec supports axes with different spans."""
        fig = plt.figure()
        gs = GridSpec(2, 2)
        ax_top = fig.add_subplot(gs[0, :])    # top row, full width
        ax_bl = fig.add_subplot(gs[1, 0])     # bottom-left
        ax_br = fig.add_subplot(gs[1, 1])     # bottom-right
        assert len(fig.axes) == 3
        plt.close('all')


class TestLabelOuter:
    def test_label_outer_hides_inner_xlabels(self):
        """label_outer() hides x-tick labels on non-bottom subplots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.label_outer()
        assert ax1._xticklabels_visible is False
        assert ax1._xlabel_visible is False
        plt.close('all')

    def test_label_outer_keeps_bottom_xlabels(self):
        """label_outer() keeps x-tick labels on bottom subplots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax2.label_outer()
        assert ax2._xticklabels_visible is True
        assert ax2._xlabel_visible is True
        plt.close('all')

    def test_label_outer_hides_inner_ylabels(self):
        """label_outer() hides y-tick labels on non-left subplots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax2.label_outer()
        assert ax2._yticklabels_visible is False
        assert ax2._ylabel_visible is False
        plt.close('all')

    def test_label_outer_keeps_left_ylabels(self):
        """label_outer() keeps y-tick labels on left subplots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.label_outer()
        assert ax1._yticklabels_visible is True
        assert ax1._ylabel_visible is True
        plt.close('all')

    def test_label_outer_2x2(self):
        """label_outer() on a 2x2 grid hides inner labels correctly."""
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        for row in axes:
            for ax in row:
                ax.label_outer()

        # Top-left (row=0, col=0): keep y-labels, hide x-labels
        assert axes[0][0]._yticklabels_visible is True
        assert axes[0][0]._xticklabels_visible is False

        # Top-right (row=0, col=1): hide both y and x labels
        assert axes[0][1]._yticklabels_visible is False
        assert axes[0][1]._xticklabels_visible is False

        # Bottom-left (row=1, col=0): keep both
        assert axes[1][0]._yticklabels_visible is True
        assert axes[1][0]._xticklabels_visible is True

        # Bottom-right (row=1, col=1): keep x, hide y
        assert axes[1][1]._yticklabels_visible is False
        assert axes[1][1]._xticklabels_visible is True
        plt.close('all')


class TestSubplotLayout:
    def test_subplots_1x1(self):
        """subplots(1,1) returns single Axes, not list."""
        fig, ax = plt.subplots(1, 1)
        assert isinstance(ax, Axes)
        plt.close('all')

    def test_subplots_1xn_flat(self):
        """subplots(1, n) returns flat list, not nested."""
        fig, axes = plt.subplots(1, 3)
        assert isinstance(axes, list)
        assert len(axes) == 3
        assert all(isinstance(a, Axes) for a in axes)
        plt.close('all')

    def test_subplots_nx1_flat(self):
        """subplots(n, 1) returns flat list, not nested."""
        fig, axes = plt.subplots(3, 1)
        assert isinstance(axes, list)
        assert len(axes) == 3
        assert all(isinstance(a, Axes) for a in axes)
        plt.close('all')

    def test_subplots_nxm_nested(self):
        """subplots(n, m) with n>1 and m>1 returns nested list."""
        fig, axes = plt.subplots(2, 3)
        assert isinstance(axes, list)
        assert len(axes) == 2
        assert all(isinstance(row, list) for row in axes)
        assert all(len(row) == 3 for row in axes)
        plt.close('all')

    def test_subplots_figure_axes_count(self):
        """Figure has correct number of axes after subplots()."""
        fig, axes = plt.subplots(2, 3)
        assert len(fig.axes) == 6
        plt.close('all')

    def test_subplot_3digit(self):
        """subplot(211) creates subplot at position (2,1,1)."""
        fig = plt.figure()
        ax = plt.subplot(211)
        assert ax._position == (2, 1, 1)
        plt.close('all')

    def test_subplot_reuse(self):
        """subplot() reuses existing axes at same position."""
        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(211)
        assert ax1 is ax2
        plt.close('all')

    def test_add_subplot_with_gridspec(self):
        """add_subplot with GridSpec positions axes correctly."""
        from matplotlib.gridspec import GridSpec
        fig = plt.figure()
        gs = GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        assert len(fig.axes) == 3
        plt.close('all')


# ===================================================================
# Additional subplot tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec


class TestSubplotsParametric:
    """Parametric tests for subplots creation."""

    @pytest.mark.parametrize('nrows,ncols,expected_total', [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 3, 6),
        (3, 3, 9),
    ])
    def test_subplots_total_count(self, nrows, ncols, expected_total):
        """plt.subplots creates the correct total number of axes."""
        fig, axes = plt.subplots(nrows, ncols)
        # Count via fig.get_axes()
        all_axes = fig.get_axes()
        assert len(all_axes) == expected_total
        plt.close('all')

    @pytest.mark.parametrize('nrows,ncols', [
        (2, 2), (3, 2), (2, 3), (4, 4)
    ])
    def test_subplots_all_axes_instances(self, nrows, ncols):
        """All created axes are Axes instances."""
        fig, axes = plt.subplots(nrows, ncols)
        for ax in fig.get_axes():
            assert isinstance(ax, Axes)
        plt.close('all')

    @pytest.mark.parametrize('figsize', [
        (6, 4), (10, 8), (4, 3)
    ])
    def test_subplots_figsize(self, figsize):
        """plt.subplots passes figsize to figure."""
        w, h = figsize
        fig, axes = plt.subplots(figsize=figsize)
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 100, 150])
    def test_subplots_dpi(self, dpi):
        """plt.subplots passes dpi to figure."""
        fig, axes = plt.subplots(dpi=dpi)
        assert fig.get_dpi() == dpi
        plt.close('all')


class TestAxesProperties:
    """Tests for axes object properties."""

    def test_axes_get_xlim_default(self):
        """Default xlim is (0, 1)."""
        fig, ax = plt.subplots()
        xmin, xmax = ax.get_xlim()
        assert xmin == 0.0
        assert xmax == 1.0
        plt.close('all')

    def test_axes_get_ylim_default(self):
        """Default ylim is (0, 1)."""
        fig, ax = plt.subplots()
        ymin, ymax = ax.get_ylim()
        assert ymin == 0.0
        assert ymax == 1.0
        plt.close('all')

    def test_axes_set_xlim_returns_tuple(self):
        """set_xlim returns the new limits."""
        fig, ax = plt.subplots()
        result = ax.set_xlim(0, 10)
        assert result == (0, 10)
        plt.close('all')

    def test_axes_set_ylim_returns_tuple(self):
        """set_ylim returns the new limits."""
        fig, ax = plt.subplots()
        result = ax.set_ylim(-1, 1)
        assert result == (-1, 1)
        plt.close('all')

    def test_axes_title_default_empty(self):
        """Axes title is empty by default."""
        fig, ax = plt.subplots()
        assert ax.get_title() == ''
        plt.close('all')

    def test_axes_xlabel_default_empty(self):
        """Axes xlabel is empty by default."""
        fig, ax = plt.subplots()
        assert ax.get_xlabel() == ''
        plt.close('all')

    def test_axes_ylabel_default_empty(self):
        """Axes ylabel is empty by default."""
        fig, ax = plt.subplots()
        assert ax.get_ylabel() == ''
        plt.close('all')

    def test_axes_lines_empty_default(self):
        """Axes lines list is empty by default."""
        fig, ax = plt.subplots()
        assert ax.lines == []
        plt.close('all')

    def test_axes_patches_list(self):
        """Axes patches is a list."""
        fig, ax = plt.subplots()
        assert isinstance(ax.patches, list)
        plt.close('all')

    def test_axes_texts_empty_default(self):
        """Axes texts list is empty by default."""
        fig, ax = plt.subplots()
        assert ax.texts == []
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_axes_set_xscale(self, scale):
        """Axes.set_xscale works for standard scales."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_axes_set_yscale(self, scale):
        """Axes.set_yscale works for standard scales."""
        fig, ax = plt.subplots()
        ax.set_yscale(scale)
        assert ax.get_yscale() == scale
        plt.close('all')


# ===================================================================
# Extended parametric tests for subplots
# ===================================================================

class TestSubplotsParametric2:
    """Extended parametric tests for subplots."""

    @pytest.mark.parametrize('nrows,ncols', [
        (1, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 3), (4, 1), (1, 4),
    ])
    def test_subplots_axes_count(self, nrows, ncols):
        """plt.subplots returns nrows*ncols axes."""
        fig, axes = plt.subplots(nrows, ncols)
        assert len(fig.get_axes()) == nrows * ncols
        plt.close('all')

    @pytest.mark.parametrize('figsize', [(4, 3), (6.4, 4.8), (8, 6), (10, 10), (12, 4)])
    def test_subplots_figsize(self, figsize):
        """plt.subplots accepts figsize."""
        w, h = figsize
        fig, ax = plt.subplots(figsize=figsize)
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 96, 100, 150, 200])
    def test_subplots_dpi(self, dpi):
        """plt.subplots accepts dpi."""
        fig, ax = plt.subplots(dpi=dpi)
        assert fig.get_dpi() == dpi
        plt.close('all')

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_subplots_sharex(self, n):
        """sharex=True shares x axis."""
        fig, axes = plt.subplots(n, 1, sharex=True)
        # Check that xlim on first is shared with others
        axes[0].set_xlim(0, 100)
        for ax in axes:
            got = ax.get_xlim()
            assert abs(got[0] - 0) < 1e-10
            assert abs(got[1] - 100) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_subplots_sharey(self, n):
        """sharey=True shares y axis."""
        fig, axes = plt.subplots(1, n, sharey=True)
        axes[0].set_ylim(-10, 10)
        for ax in axes:
            got = ax.get_ylim()
            assert abs(got[0] - (-10)) < 1e-10
            assert abs(got[1] - 10) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100)])
    def test_subplot_xlim(self, xmin, xmax):
        """Subplot xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('ymin,ymax', [(0, 1), (-5, 5), (0, 100)])
    def test_subplot_ylim(self, ymin, ymax):
        """Subplot ylim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_ylim(ymin, ymax)
        got = ax.get_ylim()
        assert abs(got[0] - ymin) < 1e-10
        assert abs(got[1] - ymax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_subplot_xscale(self, scale):
        """Subplot xscale roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_subplot_n_lines(self, n):
        """Plotting n lines in a subplot."""
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('title', ['My Title', '', 'Test 123'])
    def test_subplot_title(self, title):
        """Subplot title roundtrip."""
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')
