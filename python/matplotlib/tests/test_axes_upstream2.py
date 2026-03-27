"""Additional upstream-ported axes tests — batch 2."""

import math
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text


# ===================================================================
# Axes properties
# ===================================================================

class TestAxesProperties:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_set_get_title(self):
        ax = self._make_ax()
        ax.set_title('Hello')
        assert ax.get_title() == 'Hello'

    def test_set_get_xlabel(self):
        ax = self._make_ax()
        ax.set_xlabel('X axis')
        assert ax.get_xlabel() == 'X axis'

    def test_set_get_ylabel(self):
        ax = self._make_ax()
        ax.set_ylabel('Y axis')
        assert ax.get_ylabel() == 'Y axis'

    def test_set_get_xlim(self):
        ax = self._make_ax()
        ax.set_xlim(5, 15)
        assert ax.get_xlim() == (5, 15)

    def test_set_get_ylim(self):
        ax = self._make_ax()
        ax.set_ylim(-10, 10)
        assert ax.get_ylim() == (-10, 10)

    def test_set_get_xscale(self):
        ax = self._make_ax()
        ax.set_xscale('log')
        assert ax.get_xscale() == 'log'

    def test_set_get_yscale(self):
        ax = self._make_ax()
        ax.set_yscale('log')
        assert ax.get_yscale() == 'log'

    def test_set_get_aspect(self):
        ax = self._make_ax()
        ax.set_aspect('equal')
        assert ax.get_aspect() == 'equal'

    def test_set_get_adjustable(self):
        ax = self._make_ax()
        ax.set_adjustable('datalim')
        assert ax.get_adjustable() == 'datalim'

    def test_set_get_anchor(self):
        ax = self._make_ax()
        ax.set_anchor('NE')
        assert ax.get_anchor() == 'NE'

    def test_set_get_box_aspect(self):
        ax = self._make_ax()
        ax.set_box_aspect(1.5)
        assert ax.get_box_aspect() == 1.5

    def test_set_get_facecolor(self):
        ax = self._make_ax()
        ax.set_facecolor('red')
        fc = ax.get_facecolor()
        assert fc[0] > 0.9  # red channel

    def test_set_get_frame_on(self):
        ax = self._make_ax()
        ax.set_frame_on(False)
        assert ax.get_frame_on() is False

    def test_set_get_axisbelow(self):
        ax = self._make_ax()
        ax.set_axisbelow(False)
        assert ax.get_axisbelow() is False

    def test_set_get_visible(self):
        ax = self._make_ax()
        ax.set_visible(False)
        assert ax.get_visible() is False

    def test_set_get_navigate(self):
        ax = self._make_ax()
        ax.set_navigate(False)
        assert ax.get_navigate() is False

    def test_navigate_mode(self):
        ax = self._make_ax()
        ax.set_navigate_mode('PAN')
        assert ax.get_navigate_mode() == 'PAN'

    def test_format_coord(self):
        ax = self._make_ax()
        s = ax.format_coord(1.5, 2.5)
        assert '1.5' in s
        assert '2.5' in s


# ===================================================================
# Axes inversion
# ===================================================================

class TestAxesInversion:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_invert_xaxis(self):
        ax = self._make_ax()
        assert not ax.xaxis_inverted()
        ax.invert_xaxis()
        assert ax.xaxis_inverted()

    def test_invert_xaxis_double(self):
        ax = self._make_ax()
        ax.invert_xaxis()
        ax.invert_xaxis()
        assert not ax.xaxis_inverted()

    def test_invert_yaxis(self):
        ax = self._make_ax()
        assert not ax.yaxis_inverted()
        ax.invert_yaxis()
        assert ax.yaxis_inverted()

    def test_invert_yaxis_double(self):
        ax = self._make_ax()
        ax.invert_yaxis()
        ax.invert_yaxis()
        assert not ax.yaxis_inverted()

    def test_inverted_xlim(self):
        ax = self._make_ax()
        ax.set_xlim(0, 10)
        ax.invert_xaxis()
        xlim = ax.get_xlim()
        assert xlim[0] > xlim[1]

    def test_inverted_ylim(self):
        ax = self._make_ax()
        ax.set_ylim(0, 10)
        ax.invert_yaxis()
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]


# ===================================================================
# Axes clear
# ===================================================================

class TestAxesClear:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_cla(self):
        ax = self._make_ax()
        ax.plot([1, 2, 3])
        ax.set_title('test')
        ax.cla()
        assert ax.get_title() == ''
        assert len(ax.lines) == 0

    def test_clear(self):
        ax = self._make_ax()
        ax.plot([1, 2, 3])
        ax.clear()
        assert len(ax.lines) == 0

    def test_cla_preserves_sharing(self):
        fig = Figure()
        axes = fig.subplots(2, 1, sharex=True)
        axes[0].cla()
        # Sharing should persist
        assert len(axes[0]._shared_x) > 0

    def test_cla_resets_scale(self):
        ax = self._make_ax()
        ax.set_xscale('log')
        ax.cla()
        assert ax.get_xscale() == 'linear'

    def test_cla_resets_aspect(self):
        ax = self._make_ax()
        ax.set_aspect('equal')
        ax.cla()
        assert ax.get_aspect() == 'auto'


# ===================================================================
# Axes batch set
# ===================================================================

class TestAxesSet:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_set_title(self):
        ax = self._make_ax()
        ax.set(title='Test')
        assert ax.get_title() == 'Test'

    def test_set_xlabel(self):
        ax = self._make_ax()
        ax.set(xlabel='X')
        assert ax.get_xlabel() == 'X'

    def test_set_ylabel(self):
        ax = self._make_ax()
        ax.set(ylabel='Y')
        assert ax.get_ylabel() == 'Y'

    def test_set_xlim(self):
        ax = self._make_ax()
        ax.set(xlim=(0, 10))
        assert ax.get_xlim() == (0, 10)

    def test_set_ylim(self):
        ax = self._make_ax()
        ax.set(ylim=(0, 10))
        assert ax.get_ylim() == (0, 10)

    def test_set_multiple(self):
        ax = self._make_ax()
        ax.set(title='T', xlabel='X', ylabel='Y')
        assert ax.get_title() == 'T'
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'


# ===================================================================
# Axes tick params
# ===================================================================

class TestAxesTickParams:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_tick_params(self):
        ax = self._make_ax()
        ax.tick_params(axis='x', direction='in')
        params = ax.get_tick_params('x')
        assert params['direction'] == 'in'

    def test_tick_params_both(self):
        ax = self._make_ax()
        ax.tick_params(axis='both', labelsize=14)
        assert ax.get_tick_params('x')['labelsize'] == 14
        assert ax.get_tick_params('y')['labelsize'] == 14

    def test_tick_params_y(self):
        ax = self._make_ax()
        ax.tick_params(axis='y', direction='out')
        params = ax.get_tick_params('y')
        assert params['direction'] == 'out'

    def test_set_xticks(self):
        ax = self._make_ax()
        ax.set_xticks([0, 1, 2, 3])
        assert ax.get_xticks() == [0, 1, 2, 3]

    def test_set_yticks(self):
        ax = self._make_ax()
        ax.set_yticks([0, 5, 10])
        assert ax.get_yticks() == [0, 5, 10]

    def test_set_xticks_with_labels(self):
        ax = self._make_ax()
        ax.set_xticks([0, 1, 2], labels=['a', 'b', 'c'])
        assert ax.get_xticklabels() == ['a', 'b', 'c']

    def test_set_yticks_with_labels(self):
        ax = self._make_ax()
        ax.set_yticks([0, 1], labels=['lo', 'hi'])
        assert ax.get_yticklabels() == ['lo', 'hi']


# ===================================================================
# Axes axis()
# ===================================================================

class TestAxesAxis:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_axis_none(self):
        ax = self._make_ax()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        result = ax.axis()
        assert result == (0, 10, 0, 20)

    def test_axis_equal(self):
        ax = self._make_ax()
        ax.axis('equal')
        assert ax.get_aspect() == 'equal'

    def test_axis_square(self):
        ax = self._make_ax()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('square')
        assert ax.get_aspect() == 'equal'


# ===================================================================
# Axes repr
# ===================================================================

class TestAxesRepr:
    def test_basic_repr(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        r = repr(ax)
        assert 'Axes' in r

    def test_repr_with_title(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('My Plot')
        r = repr(ax)
        assert 'My Plot' in r

    def test_repr_with_label(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_label('subplot_label')
        r = repr(ax)
        assert 'subplot_label' in r


# ===================================================================
# Axes position
# ===================================================================

class TestAxesPosition:
    def test_get_position(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        pos = ax.get_position()
        assert hasattr(pos, 'x0')
        assert hasattr(pos, 'y0')
        assert hasattr(pos, 'width')
        assert hasattr(pos, 'height')

    def test_set_position(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_position([0.1, 0.1, 0.8, 0.8])
        pos = ax.get_position()
        assert pos.x0 == 0.1
        assert pos.y0 == 0.1

    def test_subplot_position(self):
        fig = Figure()
        ax = fig.add_subplot(2, 2, 1)
        pos = ax.get_position()
        assert pos.width == 0.5
        assert pos.height == 0.5


# ===================================================================
# Axes get_children / findobj
# ===================================================================

class TestAxesChildren:
    def test_get_children_empty(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        children = ax.get_children()
        assert isinstance(children, list)
        assert len(children) == 0

    def test_get_children_with_line(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3])
        children = ax.get_children()
        assert len(children) >= 1

    def test_findobj_all(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3])
        result = ax.findobj()
        assert len(result) >= 2  # self + line

    def test_findobj_by_type(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3])
        ax.plot([3, 2, 1])
        result = ax.findobj(Line2D)
        assert len(result) == 2

    def test_findobj_no_match(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3])
        result = ax.findobj(Rectangle)
        assert len(result) == 0

    def test_findobj_with_callable(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3], label='foo')
        ax.plot([3, 2, 1], label='bar')
        result = ax.findobj(lambda a: hasattr(a, 'get_label') and a.get_label() == 'foo')
        assert len(result) >= 1


# ===================================================================
# Axes grid
# ===================================================================

class TestAxesGrid:
    def test_grid_on(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        assert ax._grid is True

    def test_grid_off(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        ax.grid(False)
        assert ax._grid is False


# ===================================================================
# Axes remove
# ===================================================================

class TestAxesRemove:
    def test_remove(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.remove()
        assert len(fig.get_axes()) == 1
        assert fig.get_axes()[0] is ax2


# ===================================================================
# Axes span methods
# ===================================================================

class TestAxesSpan:
    def test_axhspan(self):
        from matplotlib.patches import Polygon
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        poly = ax.axhspan(0.5, 1.5)
        assert isinstance(poly, Polygon)
        assert len(ax.patches) == 1

    def test_axvspan(self):
        from matplotlib.patches import Polygon
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        poly = ax.axvspan(2.0, 4.0)
        assert isinstance(poly, Polygon)
        assert len(ax.patches) == 1

    def test_axhspan_kwargs(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        rect = ax.axhspan(0, 1, facecolor='red', alpha=0.3)
        assert rect is not None

    def test_axvspan_kwargs(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        rect = ax.axvspan(0, 1, facecolor='blue', alpha=0.3)
        assert rect is not None


# ===================================================================
# Axes NaN/Inf limit validation
# ===================================================================

class TestAxesLimitValidation:
    def _make_ax(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_xlim_nan(self):
        ax = self._make_ax()
        with pytest.raises(ValueError):
            ax.set_xlim(float('nan'), 10)

    def test_xlim_inf(self):
        ax = self._make_ax()
        with pytest.raises(ValueError):
            ax.set_xlim(float('inf'), 10)

    def test_ylim_nan(self):
        ax = self._make_ax()
        with pytest.raises(ValueError):
            ax.set_ylim(0, float('nan'))

    def test_ylim_inf(self):
        ax = self._make_ax()
        with pytest.raises(ValueError):
            ax.set_ylim(0, float('inf'))


# ===================================================================
# Axes minorticks
# ===================================================================

class TestAxesMinorticks:
    def test_minorticks_on(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.minorticks_on()  # Should not raise

    def test_minorticks_off(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.minorticks_off()  # Should not raise


# ===================================================================
# Axes contour stubs
# ===================================================================

class TestAxesContour:
    def test_contour(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        result = ax.contour()
        assert result is not None

    def test_contourf(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        result = ax.contourf()
        assert result is not None


# ===================================================================
# Axes imshow
# ===================================================================

class TestAxesImshow:
    def test_imshow_basic(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        data = [[1, 2], [3, 4]]
        im = ax.imshow(data)
        assert im is not None
        assert len(ax.images) == 1

    def test_imshow_sets_equal_aspect(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow([[1, 2], [3, 4]])
        assert ax.get_aspect() == 'equal'

    def test_imshow_custom_aspect(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow([[1, 2], [3, 4]], aspect='auto')
        assert ax.get_aspect() == 'auto'


# ===================================================================
# Extended parametric tests for axes
# ===================================================================

import pytest as _pytest
