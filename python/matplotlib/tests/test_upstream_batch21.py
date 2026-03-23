"""
Upstream tests — batch 21.
Focus: Gridspec, table, rcsetup, legend advanced, text advanced.
Adapted from matplotlib upstream tests (no canvas rendering, no image comparison).
"""
import math
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def close(a, b, tol=1e-10):
    """Check approximate equality."""
    return abs(a - b) < tol


# ------------------------------------------------------------------
# Gridspec tests
# ------------------------------------------------------------------

class TestGridspec:
    def test_basic_subplots_gridspec(self):
        from matplotlib.gridspec import GridSpec
        fig = Figure()
        gs = GridSpec(2, 3, figure=fig)
        assert gs is not None

    def test_gridspec_nrows_ncols(self):
        from matplotlib.gridspec import GridSpec
        fig = Figure()
        gs = GridSpec(3, 4, figure=fig)
        assert gs.nrows == 3
        assert gs.ncols == 4

    def test_subgridspec(self):
        from matplotlib.gridspec import GridSpec
        fig = Figure()
        gs = GridSpec(2, 2, figure=fig)
        # SubplotSpec indexing
        sp = gs[0, 0]
        assert sp is not None

    def test_gridspec_shape(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 4)
        assert gs.nrows == 3
        assert gs.ncols == 4

    def test_gridspec_add_subplot(self):
        from matplotlib.gridspec import GridSpec
        fig = Figure()
        gs = GridSpec(2, 2, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        assert ax is not None

    def test_subplots_with_gridspec(self):
        fig, axes = plt.subplots(2, 3)
        assert len(axes) == 2
        assert len(axes[0]) == 3

    def test_subplot_mosaic_like(self):
        from matplotlib.gridspec import GridSpec
        fig = Figure()
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None


# ------------------------------------------------------------------
# Legend advanced tests
# ------------------------------------------------------------------

class TestLegendAdvanced:
    def test_legend_handles_labels(self):
        fig, ax = plt.subplots()
        line1, = ax.plot([0, 1], [0, 1], label='line1')
        line2, = ax.plot([0, 1], [1, 0], label='line2')
        leg = ax.legend()
        texts = leg.get_texts()
        assert len(texts) == 2

    def test_legend_explicit_handles_labels(self):
        fig, ax = plt.subplots()
        line1, = ax.plot([0, 1], [0, 1])
        leg = ax.legend(handles=[line1], labels=['my line'])
        texts = leg.get_texts()
        assert len(texts) == 1

    def test_legend_title_text(self):
        fig, ax = plt.subplots()
        leg = ax.legend(title='Legend Title')
        title_obj = leg.get_title()
        assert title_obj is not None

    def test_legend_loc_str(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='line')
        leg = ax.legend(loc='upper right')
        assert leg._loc == 'upper right'

    def test_legend_fontsize(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='line')
        leg = ax.legend(fontsize=12)
        assert leg._fontsize == 12

    def test_legend_ncol(self):
        fig, ax = plt.subplots()
        for i in range(4):
            ax.plot([0, 1], label=f'line {i}')
        leg = ax.legend(ncol=2)
        assert leg._ncol == 2

    def test_legend_shadow(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='line')
        leg = ax.legend(shadow=True)
        assert leg._shadow

    def test_legend_fancybox(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='line')
        leg = ax.legend(fancybox=True)
        assert leg._fancybox

    def test_legend_alpha_frame(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='line')
        leg = ax.legend(framealpha=0.8)
        assert close(leg._framealpha, 0.8)

    def test_legend_draggable(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='line')
        leg = ax.legend()
        leg.set_draggable(True)
        assert leg._draggable

    def test_figure_legend(self):
        fig, axes = plt.subplots(1, 2)
        axes[0].plot([0, 1], label='line A')
        axes[1].plot([0, 1], label='line B')
        leg = fig.legend()
        assert leg is not None

    def test_pyplot_legend(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='data')
        leg = plt.legend()
        assert leg is not None


# ------------------------------------------------------------------
# Text advanced tests
# ------------------------------------------------------------------

class TestTextAdvanced:
    def test_text_ha(self):
        t = Text(0, 0, 'test', ha='center')
        ha = t.get_ha()
        assert ha == 'center'

    def test_text_va(self):
        t = Text(0, 0, 'test', va='top')
        va = t.get_va()
        assert va == 'top'

    def test_text_rotation(self):
        t = Text(0, 0, 'test', rotation=45)
        r = t.get_rotation()
        assert close(r, 45)

    def test_text_fontsize_string(self):
        t = Text(0, 0, 'test', fontsize='large')
        # Should not error
        assert t is not None

    def test_text_family(self):
        t = Text(0, 0, 'test', family='serif')
        assert t is not None

    def test_text_style(self):
        t = Text(0, 0, 'test', style='italic')
        assert t is not None

    def test_text_weight(self):
        t = Text(0, 0, 'test', weight='bold')
        assert t is not None

    def test_text_set_x(self):
        t = Text(0, 0, 'test')
        # set_position changes both x and y
        t.set_position((5.0, 0.0))
        pos = t.get_position()
        assert close(pos[0], 5.0)

    def test_text_set_y(self):
        t = Text(0, 0, 'test')
        # set_position changes both x and y
        t.set_position((0.0, 3.0))
        pos = t.get_position()
        assert close(pos[1], 3.0)

    def test_axes_text_multiple(self):
        fig, ax = plt.subplots()
        t1 = ax.text(0.1, 0.1, 'A')
        t2 = ax.text(0.5, 0.5, 'B')
        t3 = ax.text(0.9, 0.9, 'C')
        assert t1.get_text() == 'A'
        assert t2.get_text() == 'B'
        assert t3.get_text() == 'C'

    def test_text_in_texts_list(self):
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'hello')
        assert t in ax.texts

    def test_set_xlabel_fontsize(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X', fontsize=14)
        assert ax.get_xlabel() == 'X'

    def test_set_ylabel_fontsize(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Y', fontsize=14)
        assert ax.get_ylabel() == 'Y'

    def test_set_title_fontsize(self):
        fig, ax = plt.subplots()
        ax.set_title('Title', fontsize=16)
        assert ax.get_title() == 'Title'


# ------------------------------------------------------------------
# Table tests
# ------------------------------------------------------------------

class TestTable:
    def test_table_basic(self):
        fig, ax = plt.subplots()
        data = [[1, 2], [3, 4]]
        t = ax.table(cellText=data)
        assert t is not None

    def test_table_with_headers(self):
        fig, ax = plt.subplots()
        data = [[1, 2], [3, 4]]
        t = ax.table(cellText=data, colLabels=['A', 'B'])
        assert t is not None

    def test_table_with_row_labels(self):
        fig, ax = plt.subplots()
        data = [[1, 2], [3, 4]]
        t = ax.table(cellText=data, rowLabels=['row1', 'row2'])
        assert t is not None

    def test_table_location(self):
        fig, ax = plt.subplots()
        data = [[1, 2]]
        t = ax.table(cellText=data, loc='center')
        assert t is not None

    def test_table_cell_access(self):
        fig, ax = plt.subplots()
        data = [[1, 2], [3, 4]]
        t = ax.table(cellText=data)
        # Access a cell
        cell = t[0, 0]
        assert cell is not None

    def test_table_get_celld(self):
        fig, ax = plt.subplots()
        data = [[1, 2]]
        t = ax.table(cellText=data)
        celld = t.get_celld()
        assert isinstance(celld, dict)


# ------------------------------------------------------------------
# Axis advanced tests
# ------------------------------------------------------------------

class TestAxisAdvanced:
    def test_xaxis_get_major_ticks(self):
        fig, ax = plt.subplots()
        # set_ticks on xaxis directly updates majorTicks
        ax.xaxis.set_ticks([0, 1, 2, 3])
        ticks = ax.xaxis.get_major_ticks()
        assert len(ticks) == 4

    def test_yaxis_get_major_ticks(self):
        fig, ax = plt.subplots()
        ax.yaxis.set_ticks([0, 0.5, 1.0])
        ticks = ax.yaxis.get_major_ticks()
        assert len(ticks) == 3

    def test_get_ticklocs_x(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([0, 1, 2])
        locs = ax.xaxis.get_ticklocs()
        assert list(locs) == [0, 1, 2]

    def test_get_ticklocs_y(self):
        fig, ax = plt.subplots()
        ax.yaxis.set_ticks([0, 5, 10])
        locs = ax.yaxis.get_ticklocs()
        assert list(locs) == [0, 5, 10]

    def test_xaxis_label(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X Label')
        label = ax.xaxis.get_label()
        assert label is not None

    def test_yaxis_label(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Y Label')
        label = ax.yaxis.get_label()
        assert label is not None

    def test_set_minor_locator(self):
        fig, ax = plt.subplots()
        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    def test_get_major_locator(self):
        fig, ax = plt.subplots()
        from matplotlib.ticker import AutoLocator
        ax.xaxis.set_major_locator(AutoLocator())
        loc = ax.xaxis.get_major_locator()
        assert loc is not None

    def test_get_major_formatter(self):
        fig, ax = plt.subplots()
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        fmt = ax.xaxis.get_major_formatter()
        assert fmt is not None

    def test_tick_label_visibility(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_tick_params(labelbottom=False)

    def test_axis_inverted_x(self):
        fig, ax = plt.subplots()
        # Use set_inverted directly on xaxis object
        ax.xaxis.set_inverted(True)
        assert ax.xaxis.get_inverted()

    def test_axis_not_inverted_default(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        assert not ax.xaxis.get_inverted()

    def test_get_pickradius(self):
        fig, ax = plt.subplots()
        pr = ax.xaxis.get_pickradius()
        assert pr >= 0

    def test_set_pickradius(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_pickradius(10)
        assert ax.xaxis.get_pickradius() == 10


# ------------------------------------------------------------------
# Misc / edge cases
# ------------------------------------------------------------------

class TestMisc:
    def test_figure_add_axes_rect(self):
        fig = Figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        assert ax is not None

    def test_figure_get_axes_after_add(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        axes = fig.get_axes()
        assert ax1 in axes
        assert ax2 in axes

    def test_plt_close_by_number(self):
        fig = plt.figure(num=999)
        plt.close(999)

    def test_multiple_figures(self):
        fig1 = plt.figure()
        fig2 = plt.figure()
        assert fig1 is not fig2
        plt.close('all')

    def test_current_figure_tracking(self):
        fig = plt.figure()
        assert plt.gcf() is fig
        plt.close('all')

    def test_axes_get_figure(self):
        fig, ax = plt.subplots()
        assert ax.figure is fig

    def test_axes_xaxis_axes(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.axes is ax

    def test_axes_yaxis_axes(self):
        fig, ax = plt.subplots()
        assert ax.yaxis.axes is ax

    def test_artist_stale(self):
        from matplotlib.artist import Artist
        a = Artist()
        assert a._stale

    def test_line_set_data(self):
        line = Line2D([0, 1, 2], [0, 1, 4])
        line.set_data([3, 4, 5], [9, 16, 25])
        assert list(line.get_xdata()) == [3, 4, 5]
        assert list(line.get_ydata()) == [9, 16, 25]

    def test_rectangle_get_bounds(self):
        r = Rectangle((1, 2), 3, 4)
        assert close(r.get_xy()[0], 1)
        assert close(r.get_xy()[1], 2)
        assert close(r.get_width(), 3)
        assert close(r.get_height(), 4)

    def test_axes_repr(self):
        fig, ax = plt.subplots()
        r = repr(ax)
        assert 'Axes' in r

    def test_figure_repr(self):
        fig = Figure()
        r = repr(fig)
        assert 'Figure' in r

    def test_line_repr(self):
        line = Line2D([0, 1], [0, 1])
        r = repr(line)
        assert r is not None
