"""Upstream-ported tests for matplotlib.legend."""

import pytest
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text


# ===================================================================
# Legend creation
# ===================================================================

class TestLegendCreation:
    def _make_fig_ax(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax

    def test_legend_from_plot(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1, 2, 3], label='line1')
        leg = ax.legend()
        assert leg is not None

    def test_legend_labels(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1, 2, 3], label='line1')
        ax.plot([3, 2, 1], label='line2')
        leg = ax.legend()
        texts = leg.get_texts()
        assert len(texts) == 2
        assert texts[0].get_text() == 'line1'
        assert texts[1].get_text() == 'line2'

    def test_legend_explicit_labels(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1, 2, 3])
        ax.plot([3, 2, 1])
        leg = ax.legend(['a', 'b'])
        texts = leg.get_texts()
        assert len(texts) == 2
        assert texts[0].get_text() == 'a'

    def test_legend_handles_labels(self):
        fig, ax = self._make_fig_ax()
        l1, = ax.plot([1, 2, 3])
        l2, = ax.plot([3, 2, 1])
        leg = ax.legend([l1, l2], ['first', 'second'])
        texts = leg.get_texts()
        assert len(texts) == 2
        assert texts[0].get_text() == 'first'
        assert texts[1].get_text() == 'second'

    def test_legend_loc(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1, 2, 3], label='line')
        leg = ax.legend(loc='upper left')
        assert leg.get_loc() == 'upper left'

    def test_legend_title(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1, 2, 3], label='line')
        leg = ax.legend(title='My Title')
        assert leg.get_title().get_text() == 'My Title'

    def test_legend_set_title(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1, 2, 3], label='line')
        leg = ax.legend()
        leg.set_title('New Title')
        assert leg.get_title().get_text() == 'New Title'

    def test_legend_empty(self):
        fig, ax = self._make_fig_ax()
        leg = ax.legend()
        texts = leg.get_texts()
        assert len(texts) == 0

    def test_legend_ncol(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1], label='a')
        leg = ax.legend(ncol=2)
        assert leg._ncol == 2

    def test_legend_ncols(self):
        fig, ax = self._make_fig_ax()
        ax.plot([1], label='a')
        leg = ax.legend(ncols=3)
        assert leg._ncol == 3


# ===================================================================
# Legend properties
# ===================================================================

class TestLegendProperties:
    def _make_legend(self, **kwargs):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3], label='test')
        return ax.legend(**kwargs)

    def test_get_loc_default(self):
        leg = self._make_legend()
        assert leg.get_loc() == 'best'

    def test_set_loc(self):
        leg = self._make_legend()
        leg.set_loc('upper right')
        assert leg.get_loc() == 'upper right'

    def test_get_frame_on_default(self):
        leg = self._make_legend()
        assert leg.get_frame_on() is True

    def test_set_frame_on(self):
        leg = self._make_legend()
        leg.set_frame_on(False)
        assert leg.get_frame_on() is False

    def test_get_visible_default(self):
        leg = self._make_legend()
        assert leg.get_visible() is True

    def test_set_visible(self):
        leg = self._make_legend()
        leg.set_visible(False)
        assert leg.get_visible() is False

    def test_get_shadow_default(self):
        leg = self._make_legend()
        assert leg.get_shadow() is False

    def test_set_shadow(self):
        leg = self._make_legend()
        leg.set_shadow(True)
        assert leg.get_shadow() is True

    def test_frameon_kwarg(self):
        leg = self._make_legend(frameon=False)
        assert leg.get_frame_on() is False

    def test_shadow_kwarg(self):
        leg = self._make_legend(shadow=True)
        assert leg.get_shadow() is True

    def test_fancybox_kwarg(self):
        leg = self._make_legend(fancybox=True)
        assert leg.get_fancybox() is True

    def test_framealpha_kwarg(self):
        leg = self._make_legend(framealpha=0.5)
        assert leg.get_framealpha() == 0.5

    def test_edgecolor_kwarg(self):
        leg = self._make_legend(edgecolor='red')
        assert leg.get_edgecolor() == 'red'

    def test_facecolor_kwarg(self):
        leg = self._make_legend(facecolor='blue')
        assert leg.get_facecolor() == 'blue'


# ===================================================================
# Legend handles
# ===================================================================

class TestLegendHandles:
    def test_legend_handles_property(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        l1, = ax.plot([1, 2, 3], label='line1')
        leg = ax.legend()
        assert len(leg.legend_handles) == 1

    def test_legendHandles_deprecated(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3], label='line1')
        leg = ax.legend()
        # Should work (deprecated but functional)
        assert len(leg.legendHandles) == 1

    def test_get_lines(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        l1, = ax.plot([1, 2, 3], label='line1')
        leg = ax.legend()
        lines = leg.get_lines()
        assert len(lines) == 1

    def test_get_frame(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3], label='line1')
        leg = ax.legend()
        frame = leg.get_frame()
        assert isinstance(frame, Rectangle)


# ===================================================================
# Legend draggable
# ===================================================================

class TestLegendDraggable:
    def test_default_not_draggable(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2], label='x')
        leg = ax.legend()
        assert leg.get_draggable() is False

    def test_set_draggable(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2], label='x')
        leg = ax.legend()
        leg.set_draggable(True)
        assert leg.get_draggable() is True

    def test_unset_draggable(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2], label='x')
        leg = ax.legend()
        leg.set_draggable(True)
        leg.set_draggable(False)
        assert leg.get_draggable() is False


# ===================================================================
# Legend remove
# ===================================================================

class TestLegendRemove:
    def test_remove(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2], label='x')
        leg = ax.legend()
        assert ax._legend is True
        leg.remove()
        assert ax._legend is False


# ===================================================================
# Legend codes
# ===================================================================

class TestLegendCodes:
    def test_codes_dict(self):
        assert Legend.codes['best'] == 0
        assert Legend.codes['upper right'] == 1
        assert Legend.codes['upper left'] == 2
        assert Legend.codes['lower left'] == 3
        assert Legend.codes['lower right'] == 4
        assert Legend.codes['right'] == 5
        assert Legend.codes['center left'] == 6
        assert Legend.codes['center right'] == 7
        assert Legend.codes['lower center'] == 8
        assert Legend.codes['upper center'] == 9
        assert Legend.codes['center'] == 10


# ===================================================================
# Legend label
# ===================================================================

class TestLegendLabel:
    def test_set_get_label(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1], label='x')
        leg = ax.legend()
        leg.set_label('my_legend')
        assert leg.get_label() == 'my_legend'

    def test_default_label(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1], label='x')
        leg = ax.legend()
        assert leg.get_label() == ''

    def test_set_label_none(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1], label='x')
        leg = ax.legend()
        leg.set_label(None)
        assert leg.get_label() == ''


# ===================================================================
# Legend repr
# ===================================================================

class TestLegendRepr:
    def test_repr(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1], label='x')
        leg = ax.legend()
        r = repr(leg)
        assert '<Legend>' in r


# ===================================================================
# Legend _ncols property
# ===================================================================

class TestLegendNcols:
    def test_ncols_property(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1], label='x')
        leg = ax.legend(ncols=4)
        assert leg._ncols == 4

    def test_ncol_default(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1], label='x')
        leg = ax.legend()
        assert leg._ncol == 1


# ===================================================================
# Legend with no-legend lines
# ===================================================================

class TestLegendFiltering:
    def test_nolegend_label_skipped(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2], label='visible')
        ax.plot([3, 4], label='_nolegend_')
        h, l = ax.get_legend_handles_labels()
        assert len(l) == 1
        assert l[0] == 'visible'

    def test_underscore_label_skipped(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2], label='visible')
        ax.plot([3, 4], label='_hidden')
        h, l = ax.get_legend_handles_labels()
        assert len(l) == 1
        assert l[0] == 'visible'

    def test_no_labels(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2])
        h, l = ax.get_legend_handles_labels()
        assert len(l) == 0
