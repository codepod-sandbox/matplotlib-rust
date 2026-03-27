"""Upstream-ported tests for matplotlib.table."""

import pytest
from matplotlib.figure import Figure
from matplotlib.table import Table, Cell, table
from matplotlib.transforms import Bbox


# ===================================================================
# Cell tests
# ===================================================================

class TestCell:
    def test_creation(self):
        c = Cell((0, 0), 1.0, 0.5, text='hello')
        assert c.get_text().get_text() == 'hello'

    def test_width_height(self):
        c = Cell((0, 0), 2.0, 3.0)
        assert c.get_width() == 2.0
        assert c.get_height() == 3.0

    def test_set_width(self):
        c = Cell((0, 0), 1.0, 1.0)
        c.set_width(5.0)
        assert c.get_width() == 5.0

    def test_set_height(self):
        c = Cell((0, 0), 1.0, 1.0)
        c.set_height(5.0)
        assert c.get_height() == 5.0

    def test_facecolor(self):
        c = Cell((0, 0), 1.0, 1.0, facecolor='red')
        assert c.get_facecolor() == 'red'

    def test_set_facecolor(self):
        c = Cell((0, 0), 1.0, 1.0)
        c.set_facecolor('blue')
        assert c.get_facecolor() == 'blue'

    def test_edgecolor(self):
        c = Cell((0, 0), 1.0, 1.0, edgecolor='green')
        assert c.get_edgecolor() == 'green'

    def test_set_edgecolor(self):
        c = Cell((0, 0), 1.0, 1.0)
        c.set_edgecolor('purple')
        assert c.get_edgecolor() == 'purple'

    def test_loc(self):
        c = Cell((0, 0), 1.0, 1.0, loc='center')
        assert c.get_loc() == 'center'

    def test_set_loc(self):
        c = Cell((0, 0), 1.0, 1.0)
        c.set_loc('left')
        assert c.get_loc() == 'left'

    def test_visible(self):
        c = Cell((0, 0), 1.0, 1.0)
        assert c.get_visible() is True
        c.set_visible(False)
        assert c.get_visible() is False

    def test_PAD(self):
        c = Cell((0, 0), 1.0, 1.0)
        assert c.PAD == 0.1

    def test_repr(self):
        c = Cell((0, 0), 1.0, 1.0, text='hello')
        r = repr(c)
        assert 'Cell' in r
        assert 'hello' in r

    def test_set_text_props(self):
        c = Cell((0, 0), 1.0, 1.0, text='hello')
        c.set_text_props(fontsize=14)
        assert c.get_text().get_fontsize() == 14

    def test_default_loc(self):
        c = Cell((0, 0), 1.0, 1.0)
        assert c.get_loc() == 'right'


# ===================================================================
# Table tests
# ===================================================================

class TestTable:
    def _make_table(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        tbl = Table(ax)
        return tbl, ax

    def test_creation(self):
        tbl, ax = self._make_table()
        assert tbl.axes is ax

    def test_add_cell(self):
        tbl, ax = self._make_table()
        cell = tbl.add_cell(0, 0, text='test')
        assert (0, 0) in tbl
        assert tbl[0, 0] is cell

    def test_add_multiple_cells(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0, text='a')
        tbl.add_cell(0, 1, text='b')
        tbl.add_cell(1, 0, text='c')
        assert len(tbl.get_celld()) == 3

    def test_getitem(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0, text='hello')
        assert tbl[0, 0].get_text().get_text() == 'hello'

    def test_setitem(self):
        tbl, ax = self._make_table()
        cell = Cell((0, 0), 1, 1, text='direct')
        tbl[0, 0] = cell
        assert tbl[0, 0] is cell

    def test_contains(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0)
        assert (0, 0) in tbl
        assert (1, 1) not in tbl

    def test_get_celld(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0, text='a')
        tbl.add_cell(0, 1, text='b')
        d = tbl.get_celld()
        assert isinstance(d, dict)
        assert len(d) == 2

    def test_get_children(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0)
        tbl.add_cell(0, 1)
        children = tbl.get_children()
        assert len(children) == 2

    def test_fontsize_default(self):
        tbl, ax = self._make_table()
        assert tbl.get_fontsize() == Table.FONTSIZE

    def test_set_fontsize(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0, text='test')
        tbl.set_fontsize(16)
        assert tbl.get_fontsize() == 16

    def test_edges_property(self):
        tbl, ax = self._make_table()
        assert tbl.edges == 'closed'
        tbl.edges = 'open'
        assert tbl.edges == 'open'

    def test_scale(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0, width=1.0, height=0.5)
        tbl.scale(2.0, 3.0)
        assert tbl[0, 0].get_width() == 2.0
        assert tbl[0, 0].get_height() == 1.5

    def test_auto_set_font_size(self):
        tbl, ax = self._make_table()
        # Should not raise
        tbl.auto_set_font_size(True)

    def test_auto_set_column_width(self):
        tbl, ax = self._make_table()
        # Should not raise
        tbl.auto_set_column_width(0)

    def test_loc(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        tbl = Table(ax, loc='top')
        assert tbl._loc == 'top'

    def test_repr(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0)
        tbl.add_cell(0, 1)
        r = repr(tbl)
        assert 'Table' in r
        assert '2' in r

    def test_FONTSIZE(self):
        assert Table.FONTSIZE == 10

    def test_AXESPAD(self):
        assert Table.AXESPAD == 0.02

    def test_codes(self):
        assert Table.codes['best'] == 0
        assert Table.codes['upper right'] == 1
        assert Table.codes['bottom'] == 17
        assert Table.codes['top'] == 16

    def test_get_window_extent_empty(self):
        tbl, ax = self._make_table()
        ext = tbl.get_window_extent()
        assert ext.is_unit()

    def test_get_window_extent(self):
        tbl, ax = self._make_table()
        tbl.add_cell(0, 0, width=2.0, height=1.0)
        ext = tbl.get_window_extent()
        assert ext is not None


# ===================================================================
# table() function tests
# ===================================================================

class TestTableFunction:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_basic(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[['a', 'b'], ['c', 'd']])
        assert isinstance(tbl, Table)
        assert len(tbl.get_celld()) == 4

    def test_with_col_labels(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[['1', '2']], colLabels=['X', 'Y'])
        cells = tbl.get_celld()
        # 2 col labels + 2 data cells
        assert len(cells) == 4

    def test_with_row_labels(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[['1', '2'], ['3', '4']],
                    rowLabels=['r1', 'r2'])
        cells = tbl.get_celld()
        # 4 data + 2 row labels
        assert len(cells) == 6

    def test_empty_celltext(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[])
        assert isinstance(tbl, Table)

    def test_edges(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[['a']], edges='open')
        assert tbl.edges == 'open'

    def test_loc(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[['a']], loc='top')
        assert tbl._loc == 'top'

    def test_cell_colours(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[['a', 'b']],
                    cellColours=[['red', 'blue']])
        cells = tbl.get_celld()
        assert cells[(0, 0)].get_facecolor() == 'red'
        assert cells[(0, 1)].get_facecolor() == 'blue'

    def test_col_colours(self):
        ax = self._make_ax()
        tbl = table(ax, cellText=[['a']], colLabels=['X'],
                    colColours=['yellow'])
        cells = tbl.get_celld()
        assert cells[(0, 0)].get_facecolor() == 'yellow'

    def test_axes_table_method(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        tbl = ax.table(cellText=[['a', 'b'], ['c', 'd']])
        assert isinstance(tbl, Table)


# ===================================================================
# Table via Axes.table()
# ===================================================================

class TestAxesTable:
    def test_axes_table(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        tbl = ax.table(cellText=[['1', '2'], ['3', '4']])
        assert isinstance(tbl, Table)
        assert len(tbl.get_celld()) == 4

    def test_axes_table_with_labels(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        tbl = ax.table(cellText=[['1']], colLabels=['Col1'],
                       rowLabels=['Row1'])
        cells = tbl.get_celld()
        assert len(cells) >= 3


# ===================================================================
# Parametric tests
# ===================================================================
