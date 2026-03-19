"""Upstream-ported tests for subplots, GridSpec, and label_outer."""

import pytest
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec


# ===================================================================
# GridSpec creation
# ===================================================================

class TestGridSpecCreation:
    def test_basic(self):
        gs = GridSpec(2, 3)
        assert gs.nrows == 2
        assert gs.ncols == 3

    def test_get_geometry(self):
        gs = GridSpec(3, 4)
        assert gs.get_geometry() == (3, 4)

    def test_repr(self):
        gs = GridSpec(2, 3)
        assert 'GridSpec' in repr(gs)
        assert '2' in repr(gs)
        assert '3' in repr(gs)

    def test_hspace_wspace(self):
        gs = GridSpec(2, 2, hspace=0.5, wspace=0.3)
        assert gs._hspace == 0.5
        assert gs._wspace == 0.3

    def test_width_ratios(self):
        gs = GridSpec(1, 3, width_ratios=[1, 2, 3])
        assert gs.get_width_ratios() == [1, 2, 3]

    def test_height_ratios(self):
        gs = GridSpec(3, 1, height_ratios=[1, 2, 3])
        assert gs.get_height_ratios() == [1, 2, 3]

    def test_no_ratios(self):
        gs = GridSpec(2, 2)
        assert gs.get_width_ratios() is None
        assert gs.get_height_ratios() is None

    def test_figure(self):
        fig = Figure()
        gs = GridSpec(2, 2, figure=fig)
        assert gs.figure is fig


# ===================================================================
# GridSpec indexing
# ===================================================================

class TestGridSpecIndexing:
    def test_single_cell(self):
        gs = GridSpec(2, 3)
        ss = gs[0, 0]
        assert isinstance(ss, SubplotSpec)
        assert ss.rowspan == (0, 1)
        assert ss.colspan == (0, 1)

    def test_single_cell_last(self):
        gs = GridSpec(2, 3)
        ss = gs[1, 2]
        assert ss.rowspan == (1, 2)
        assert ss.colspan == (2, 3)

    def test_row_slice(self):
        gs = GridSpec(2, 3)
        ss = gs[0, :]
        assert ss.rowspan == (0, 1)
        assert ss.colspan == (0, 3)

    def test_col_slice(self):
        gs = GridSpec(2, 3)
        ss = gs[:, 0]
        assert ss.rowspan == (0, 2)
        assert ss.colspan == (0, 1)

    def test_block_slice(self):
        gs = GridSpec(3, 3)
        ss = gs[0:2, 0:2]
        assert ss.rowspan == (0, 2)
        assert ss.colspan == (0, 2)

    def test_flat_integer(self):
        gs = GridSpec(2, 3)
        ss = gs[0]
        assert ss.rowspan == (0, 1)
        assert ss.colspan == (0, 1)

    def test_flat_integer_1(self):
        gs = GridSpec(2, 3)
        ss = gs[1]
        assert ss.rowspan == (0, 1)
        assert ss.colspan == (1, 2)

    def test_flat_integer_5(self):
        gs = GridSpec(2, 3)
        ss = gs[5]
        assert ss.rowspan == (1, 2)
        assert ss.colspan == (2, 3)

    def test_negative_index(self):
        gs = GridSpec(2, 3)
        ss = gs[-1, -1]
        assert ss.rowspan == (1, 2)
        assert ss.colspan == (2, 3)

    def test_invalid_index(self):
        gs = GridSpec(2, 3)
        with pytest.raises(IndexError):
            gs[0, 0, 0]


# ===================================================================
# SubplotSpec
# ===================================================================

class TestSubplotSpec:
    def test_get_gridspec(self):
        gs = GridSpec(2, 3)
        ss = gs[0, 0]
        assert ss.get_gridspec() is gs

    def test_num1(self):
        gs = GridSpec(2, 3)
        ss = gs[0, 0]
        assert ss.num1 == 0

    def test_num2(self):
        gs = GridSpec(2, 3)
        ss = gs[0, 0]
        assert ss.num2 == 0

    def test_num1_block(self):
        gs = GridSpec(2, 3)
        ss = gs[0:2, 0:2]
        assert ss.num1 == 0
        assert ss.num2 == 4  # (1, 1)

    def test_get_position(self):
        gs = GridSpec(2, 2)
        ss = gs[0, 0]
        x0, y0, w, h = ss.get_position()
        assert w == 0.5
        assert h == 0.5

    def test_repr(self):
        gs = GridSpec(2, 3)
        ss = gs[0, 0]
        r = repr(ss)
        assert 'SubplotSpec' in r


# ===================================================================
# GridSpecFromSubplotSpec
# ===================================================================

class TestGridSpecFromSubplotSpec:
    def test_basic(self):
        gs = GridSpec(2, 2)
        ss = gs[0, 0]
        inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=ss)
        assert inner.nrows == 2
        assert inner.ncols == 2
        assert inner._subplot_spec is ss


# ===================================================================
# Figure.add_subplot with GridSpec
# ===================================================================

class TestFigureGridSpec:
    def test_add_subplot_gs(self):
        fig = Figure()
        gs = GridSpec(2, 2)
        ax = fig.add_subplot(gs[0, 0])
        assert ax is not None
        assert ax.figure is fig

    def test_add_subplot_gs_block(self):
        fig = Figure()
        gs = GridSpec(2, 2)
        ax = fig.add_subplot(gs[0, :])
        assert ax is not None

    def test_subplots_method(self):
        fig = Figure()
        gs = GridSpec(2, 2, figure=fig)
        axes = gs.subplots()
        # 2x2 -> list of lists
        assert len(axes) == 2
        assert len(axes[0]) == 2

    def test_subplots_single(self):
        fig = Figure()
        gs = GridSpec(1, 1, figure=fig)
        ax = gs.subplots()
        assert ax is not None

    def test_subplots_row(self):
        fig = Figure()
        gs = GridSpec(1, 3, figure=fig)
        axes = gs.subplots()
        assert len(axes) == 3

    def test_subplots_col(self):
        fig = Figure()
        gs = GridSpec(3, 1, figure=fig)
        axes = gs.subplots()
        assert len(axes) == 3


# ===================================================================
# GridSpec update
# ===================================================================

class TestGridSpecUpdate:
    def test_update(self):
        gs = GridSpec(2, 2)
        gs.update(hspace=0.5)
        assert gs._hspace == 0.5

    def test_update_multiple(self):
        gs = GridSpec(2, 2)
        gs.update(hspace=0.5, wspace=0.3, left=0.1)
        assert gs._hspace == 0.5
        assert gs._wspace == 0.3
        assert gs._left == 0.1

    def test_tight_layout(self):
        gs = GridSpec(2, 2)
        # Should not raise
        gs.tight_layout()


# ===================================================================
# Figure subplots
# ===================================================================

class TestFigureSubplots:
    def test_single_subplot(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        assert ax is not None

    def test_two_subplots(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        assert ax1 is not ax2

    def test_three_digit(self):
        fig = Figure()
        ax = fig.add_subplot(221)
        assert ax is not None

    def test_subplots(self):
        fig = Figure()
        axes = fig.subplots(2, 2)
        assert len(axes) == 2
        assert len(axes[0]) == 2

    def test_subplots_single(self):
        fig = Figure()
        ax = fig.subplots(1, 1)
        assert ax is not None

    def test_subplots_row(self):
        fig = Figure()
        axes = fig.subplots(1, 3)
        assert len(axes) == 3

    def test_subplots_col(self):
        fig = Figure()
        axes = fig.subplots(3, 1)
        assert len(axes) == 3


# ===================================================================
# label_outer
# ===================================================================

class TestLabelOuter:
    def test_label_outer_grid(self):
        fig = Figure()
        axes = fig.subplots(2, 2)
        for row in axes:
            for ax in row:
                ax.label_outer()

        # Top-left: hide x labels, keep y labels
        assert axes[0][0]._xticklabels_visible is False
        assert axes[0][0]._yticklabels_visible is True

        # Top-right: hide x and y labels
        assert axes[0][1]._xticklabels_visible is False
        assert axes[0][1]._yticklabels_visible is False

        # Bottom-left: keep both
        assert axes[1][0]._xticklabels_visible is True
        assert axes[1][0]._yticklabels_visible is True

        # Bottom-right: keep x, hide y
        assert axes[1][1]._xticklabels_visible is True
        assert axes[1][1]._yticklabels_visible is False

    def test_label_outer_xlabel(self):
        fig = Figure()
        axes = fig.subplots(2, 1)
        for ax in axes:
            ax.set_xlabel('X')
            ax.label_outer()

        assert axes[0]._xlabel_visible is False
        assert axes[1]._xlabel_visible is True

    def test_label_outer_ylabel(self):
        fig = Figure()
        axes = fig.subplots(1, 2)
        for ax in axes:
            ax.set_ylabel('Y')
            ax.label_outer()

        assert axes[0]._ylabel_visible is True
        assert axes[1]._ylabel_visible is False


# ===================================================================
# Shared axes
# ===================================================================

class TestSharedAxes:
    def test_sharex(self):
        fig = Figure()
        axes = fig.subplots(2, 1, sharex=True)
        axes[0].set_xlim(0, 10)
        xlim = axes[1].get_xlim()
        assert xlim[0] == 0
        assert xlim[1] == 10

    def test_sharey(self):
        fig = Figure()
        axes = fig.subplots(1, 2, sharey=True)
        axes[0].set_ylim(0, 10)
        ylim = axes[1].get_ylim()
        assert ylim[0] == 0
        assert ylim[1] == 10

    def test_sharex_grid(self):
        fig = Figure()
        axes = fig.subplots(2, 2, sharex=True)
        axes[0][0].set_xlim(5, 15)
        # All should share
        for row in axes:
            for ax in row:
                xlim = ax.get_xlim()
                assert xlim[0] == 5
                assert xlim[1] == 15

    def test_sharey_grid(self):
        fig = Figure()
        axes = fig.subplots(2, 2, sharey=True)
        axes[0][0].set_ylim(5, 15)
        for row in axes:
            for ax in row:
                ylim = ax.get_ylim()
                assert ylim[0] == 5
                assert ylim[1] == 15


# ===================================================================
# Twin axes
# ===================================================================

class TestTwinAxes:
    def test_twinx(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        assert ax2 is not ax1
        assert ax2.figure is fig

    def test_twiny(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twiny()
        assert ax2 is not ax1
        assert ax2.figure is fig

    def test_twinx_shares_x(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlim(0, 100)
        ax2 = ax1.twinx()
        xlim = ax2.get_xlim()
        assert xlim[0] == 0
        assert xlim[1] == 100

    def test_twiny_shares_y(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylim(0, 100)
        ax2 = ax1.twiny()
        ylim = ax2.get_ylim()
        assert ylim[0] == 0
        assert ylim[1] == 100


# ===================================================================
# Figure axes management
# ===================================================================

class TestFigureAxesManagement:
    def test_get_axes(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        axes = fig.get_axes()
        assert len(axes) == 2

    def test_delaxes(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        fig.delaxes(ax1)
        assert len(fig.get_axes()) == 1

    def test_clear(self):
        fig = Figure()
        fig.add_subplot(1, 1, 1)
        fig.clf()
        assert len(fig.get_axes()) == 0

    def test_gca(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        assert fig.gca() is ax


# ===================================================================
# GridSpec get_subplot_params
# ===================================================================

class TestGridSpecSubplotParams:
    def test_default_params(self):
        gs = GridSpec(2, 2)
        params = gs.get_subplot_params()
        assert hasattr(params, 'left')
        assert hasattr(params, 'right')
        assert hasattr(params, 'top')
        assert hasattr(params, 'bottom')
        assert hasattr(params, 'hspace')
        assert hasattr(params, 'wspace')

    def test_custom_params(self):
        gs = GridSpec(2, 2, left=0.2, right=0.8, top=0.9, bottom=0.1)
        params = gs.get_subplot_params()
        assert params.left == 0.2
        assert params.right == 0.8
        assert params.top == 0.9
        assert params.bottom == 0.1
