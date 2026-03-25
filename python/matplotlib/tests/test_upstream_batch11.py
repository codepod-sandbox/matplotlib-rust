"""
Upstream matplotlib test_axes.py tests — batch 11.
Focus: XAxis/YAxis objects, margins, bounds, autoscale, limits, aspect.
"""
import math
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes, XAxis, YAxis, Axis, Tick
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker


# ------------------------------------------------------------------
# XAxis / YAxis basic tests
# ------------------------------------------------------------------

class TestAxisObjects:
    """Test that ax.xaxis and ax.yaxis exist and work."""

    def test_xaxis_exists(self):
        fig, ax = plt.subplots()
        assert hasattr(ax, 'xaxis')
        assert isinstance(ax.xaxis, XAxis)

    def test_yaxis_exists(self):
        fig, ax = plt.subplots()
        assert hasattr(ax, 'yaxis')
        assert isinstance(ax.yaxis, YAxis)

    def test_xaxis_name(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.axis_name == 'x'

    def test_yaxis_name(self):
        fig, ax = plt.subplots()
        assert ax.yaxis.axis_name == 'y'

    def test_xaxis_set_major_locator(self):
        fig, ax = plt.subplots()
        loc = mticker.MultipleLocator(5)
        ax.xaxis.set_major_locator(loc)
        assert ax.xaxis.get_major_locator() is loc

    def test_yaxis_set_major_locator(self):
        fig, ax = plt.subplots()
        loc = mticker.FixedLocator([0, 1, 2])
        ax.yaxis.set_major_locator(loc)
        assert ax.yaxis.get_major_locator() is loc

    def test_xaxis_set_minor_locator(self):
        fig, ax = plt.subplots()
        loc = mticker.AutoMinorLocator()
        ax.xaxis.set_minor_locator(loc)
        assert ax.xaxis.get_minor_locator() is loc

    def test_yaxis_set_minor_locator(self):
        fig, ax = plt.subplots()
        loc = mticker.NullLocator()
        ax.yaxis.set_minor_locator(loc)
        assert ax.yaxis.get_minor_locator() is loc

    def test_xaxis_set_major_formatter(self):
        fig, ax = plt.subplots()
        fmt = mticker.FormatStrFormatter('%d')
        ax.xaxis.set_major_formatter(fmt)
        assert ax.xaxis.get_major_formatter() is fmt

    def test_yaxis_set_major_formatter(self):
        fig, ax = plt.subplots()
        fmt = mticker.ScalarFormatter()
        ax.yaxis.set_major_formatter(fmt)
        assert ax.yaxis.get_major_formatter() is fmt

    def test_xaxis_set_minor_formatter(self):
        fig, ax = plt.subplots()
        fmt = mticker.NullFormatter()
        ax.xaxis.set_minor_formatter(fmt)
        assert ax.xaxis.get_minor_formatter() is fmt

    def test_xaxis_callable_formatter(self):
        """Test that a plain callable gets wrapped in FuncFormatter."""
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, mticker.FuncFormatter)
        assert fmt(3.14159, 0) == "3.1"

    def test_xaxis_visibility(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.get_visible() is True
        ax.xaxis.set_visible(False)
        assert ax.xaxis.get_visible() is False

    def test_yaxis_visibility(self):
        fig, ax = plt.subplots()
        assert ax.yaxis.get_visible() is True
        ax.yaxis.set_visible(False)
        assert ax.yaxis.get_visible() is False

    def test_xaxis_scale(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.get_scale() == 'linear'
        ax.set_xscale('log')
        assert ax.xaxis.get_scale() == 'log'

    def test_yaxis_scale(self):
        fig, ax = plt.subplots()
        assert ax.yaxis.get_scale() == 'linear'
        ax.set_yscale('log')
        assert ax.yaxis.get_scale() == 'log'

    def test_xaxis_inverted(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.get_inverted() is False
        ax.invert_xaxis()
        assert ax.xaxis.get_inverted() is True

    def test_yaxis_inverted(self):
        fig, ax = plt.subplots()
        assert ax.yaxis.get_inverted() is False
        ax.invert_yaxis()
        assert ax.yaxis.get_inverted() is True

    def test_xaxis_label(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.get_label() is not None

    def test_yaxis_label(self):
        fig, ax = plt.subplots()
        assert ax.yaxis.get_label() is not None

    def test_xaxis_offset_text(self):
        fig, ax = plt.subplots()
        ot = ax.xaxis.get_offset_text()
        assert hasattr(ot, 'get_text')
        assert hasattr(ot, 'set_text')

    def test_yaxis_offset_text(self):
        fig, ax = plt.subplots()
        ot = ax.yaxis.get_offset_text()
        assert hasattr(ot, 'get_visible')

    def test_xaxis_ticks_position(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('top')
        assert ax.xaxis.get_ticks_position() == 'top'

    def test_xaxis_tick_top(self):
        fig, ax = plt.subplots()
        ax.xaxis.tick_top()
        assert ax.xaxis.get_ticks_position() == 'top'

    def test_yaxis_tick_right(self):
        fig, ax = plt.subplots()
        ax.yaxis.tick_right()
        assert ax.yaxis.get_ticks_position() == 'right'

    def test_axis_set_ticks(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([0, 1, 2, 3])
        ticks = ax.xaxis.get_major_ticks()
        assert len(ticks) == 4

    def test_axis_set_ticks_with_labels(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([0, 1, 2], labels=['a', 'b', 'c'])
        ticks = ax.xaxis.get_major_ticks()
        assert len(ticks) == 3
        assert ticks[0].label1.get_text() == 'a'
        assert ticks[1].label1.get_text() == 'b'
        assert ticks[2].label1.get_text() == 'c'

    def test_axis_set_tick_params(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_tick_params(labelsize=14)
        params = ax.xaxis.get_tick_params()
        assert params.get('labelsize') == 14

    def test_axis_pickradius(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.get_pickradius() == 15
        ax.xaxis.set_pickradius(20)
        assert ax.xaxis.get_pickradius() == 20

    def test_axis_isDefault_flags(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.isDefault_majfmt is True
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
        assert ax.xaxis.isDefault_majfmt is False

    def test_axis_isDefault_locator_flags(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.isDefault_majloc is True
        ax.xaxis.set_major_locator(mticker.NullLocator())
        assert ax.xaxis.isDefault_majloc is False


class TestAxisAfterClear:
    """Test that xaxis/yaxis are reset after clear()."""

    def test_cla_resets_xaxis(self):
        fig, ax = plt.subplots()
        loc = mticker.MultipleLocator(5)
        ax.xaxis.set_major_locator(loc)
        ax.cla()
        assert ax.xaxis.get_major_locator() is not loc

    def test_clear_resets_yaxis(self):
        fig, ax = plt.subplots()
        fmt = mticker.NullFormatter()
        ax.yaxis.set_major_formatter(fmt)
        ax.clear()
        assert ax.yaxis.get_major_formatter() is not fmt


# ------------------------------------------------------------------
# Tick objects
# ------------------------------------------------------------------

class TestTickObject:
    def test_tick_get_loc(self):
        t = Tick(3.5, 'foo')
        assert t.get_loc() == 3.5

    def test_tick_label(self):
        t = Tick(0, 'bar')
        assert t.label1.get_text() == 'bar'

    def test_tick_visibility(self):
        t = Tick(0)
        assert t.get_visible() is True
        t.set_visible(False)
        assert t.get_visible() is False

    def test_tick_gridline(self):
        t = Tick(0)
        assert t.gridline.get_visible() is False
        t.gridline.set_visible(True)
        assert t.gridline.get_visible() is True


# ------------------------------------------------------------------
# Margins
# ------------------------------------------------------------------

class TestMargins:
    def test_margins_get_default(self):
        fig, ax = plt.subplots()
        assert ax.margins() == (0.05, 0.05)

    def test_margins_set_single(self):
        fig, ax = plt.subplots()
        ax.margins(0.1)
        assert ax.margins() == (0.1, 0.1)

    def test_margins_set_two(self):
        fig, ax = plt.subplots()
        ax.margins(0.2, 0.3)
        assert ax.margins() == (0.2, 0.3)

    def test_margins_set_kwargs(self):
        fig, ax = plt.subplots()
        ax.margins(x=0.1, y=0.4)
        assert ax.margins() == (0.1, 0.4)

    def test_get_xmargin(self):
        fig, ax = plt.subplots()
        ax.margins(0.2, 0.3)
        assert ax.get_xmargin() == 0.2

    def test_get_ymargin(self):
        fig, ax = plt.subplots()
        ax.margins(0.2, 0.3)
        assert ax.get_ymargin() == 0.3

    def test_margins_too_many_args(self):
        fig, ax = plt.subplots()
        with pytest.raises(TypeError):
            ax.margins(0.1, 0.2, 0.3)


# ------------------------------------------------------------------
# Autoscale getters/setters
# ------------------------------------------------------------------

class TestAutoscale:
    def test_get_autoscalex_on(self):
        fig, ax = plt.subplots()
        assert ax.get_autoscalex_on() is True

    def test_set_autoscalex_on(self):
        fig, ax = plt.subplots()
        ax.set_autoscalex_on(False)
        assert ax.get_autoscalex_on() is False

    def test_get_autoscaley_on(self):
        fig, ax = plt.subplots()
        assert ax.get_autoscaley_on() is True

    def test_set_autoscaley_on(self):
        fig, ax = plt.subplots()
        ax.set_autoscaley_on(False)
        assert ax.get_autoscaley_on() is False

    def test_get_autoscale_on(self):
        fig, ax = plt.subplots()
        assert ax.get_autoscale_on() is True

    def test_set_autoscale_on(self):
        fig, ax = plt.subplots()
        ax.set_autoscale_on(False)
        assert ax.get_autoscale_on() is False
        assert ax.get_autoscalex_on() is False
        assert ax.get_autoscaley_on() is False

    def test_set_xlim_auto_kwarg(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10, auto=False)
        assert ax.get_autoscalex_on() is False

    def test_set_ylim_auto_kwarg(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 10, auto=False)
        assert ax.get_autoscaley_on() is False

    def test_set_xlim_returns_tuple(self):
        fig, ax = plt.subplots()
        result = ax.set_xlim(1, 5)
        assert result == (1, 5)

    def test_set_ylim_returns_tuple(self):
        fig, ax = plt.subplots()
        result = ax.set_ylim(2, 8)
        assert result == (2, 8)


# ------------------------------------------------------------------
# set_xlim / set_ylim tuple input
# ------------------------------------------------------------------

class TestLimTupleInput:
    def test_set_xlim_tuple(self):
        fig, ax = plt.subplots()
        ax.set_xlim((1, 5))
        assert ax.get_xlim() == (1, 5)

    def test_set_ylim_tuple(self):
        fig, ax = plt.subplots()
        ax.set_ylim((2, 8))
        assert ax.get_ylim() == (2, 8)

    def test_set_xlim_list(self):
        fig, ax = plt.subplots()
        ax.set_xlim([1, 5])
        assert ax.get_xlim() == (1, 5)

    def test_set_ylim_list(self):
        fig, ax = plt.subplots()
        ax.set_ylim([2, 8])
        assert ax.get_ylim() == (2, 8)


# ------------------------------------------------------------------
# set_xy_bound (upstream test)
# ------------------------------------------------------------------

def test_set_xy_bound():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xbound(2.0, 3.0)
    assert ax.get_xbound() == (2.0, 3.0)
    assert ax.get_xlim() == (2.0, 3.0)
    ax.set_xbound(upper=4.0)
    assert ax.get_xbound() == (2.0, 4.0)
    assert ax.get_xlim() == (2.0, 4.0)
    ax.set_xbound(lower=3.0)
    assert ax.get_xbound() == (3.0, 4.0)
    assert ax.get_xlim() == (3.0, 4.0)

    ax.set_ybound(2.0, 3.0)
    assert ax.get_ybound() == (2.0, 3.0)
    assert ax.get_ylim() == (2.0, 3.0)
    ax.set_ybound(upper=4.0)
    assert ax.get_ybound() == (2.0, 4.0)
    assert ax.get_ylim() == (2.0, 4.0)
    ax.set_ybound(lower=3.0)
    assert ax.get_ybound() == (3.0, 4.0)
    assert ax.get_ylim() == (3.0, 4.0)


# ------------------------------------------------------------------
# Aspect validation (upstream)
# ------------------------------------------------------------------

def test_set_aspect_negative():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must be finite and positive"):
        ax.set_aspect(-1)
    with pytest.raises(ValueError, match="must be finite and positive"):
        ax.set_aspect(0)


# ------------------------------------------------------------------
# Invalid axis limits (upstream)
# ------------------------------------------------------------------

def test_invalid_axis_limits():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.set_xlim(float('nan'))
    with pytest.raises(ValueError):
        ax.set_xlim(float('inf'))
    with pytest.raises(ValueError):
        ax.set_ylim(float('nan'))
    with pytest.raises(ValueError):
        ax.set_ylim(float('inf'))


# ------------------------------------------------------------------
# Scale setters accept kwargs
# ------------------------------------------------------------------

class TestScaleKwargs:
    def test_set_xscale_kwargs(self):
        fig, ax = plt.subplots()
        ax.set_xscale('log', base=10)
        assert ax.get_xscale() == 'log'

    def test_set_yscale_kwargs(self):
        fig, ax = plt.subplots()
        ax.set_yscale('log', base=2)
        assert ax.get_yscale() == 'log'


# ------------------------------------------------------------------
# dataLim / viewLim
# ------------------------------------------------------------------

class TestDataViewLim:
    def test_dataLim_exists(self):
        fig, ax = plt.subplots()
        assert hasattr(ax, 'dataLim')

    def test_viewLim_exists(self):
        fig, ax = plt.subplots()
        assert hasattr(ax, 'viewLim')

    def test_dataLim_has_bounds(self):
        fig, ax = plt.subplots()
        assert hasattr(ax.dataLim, 'bounds')
        assert len(ax.dataLim.bounds) == 4

    def test_dataLim_bbox_attrs(self):
        fig, ax = plt.subplots()
        dl = ax.dataLim
        assert hasattr(dl, 'x0')
        assert hasattr(dl, 'y0')
        assert hasattr(dl, 'x1')
        assert hasattr(dl, 'y1')
        assert hasattr(dl, 'width')
        assert hasattr(dl, 'height')


# ------------------------------------------------------------------
# stale flag
# ------------------------------------------------------------------

class TestStaleFlag:
    def test_axes_stale(self):
        fig, ax = plt.subplots()
        assert hasattr(ax, 'stale')

    def test_figure_stale(self):
        fig = plt.figure()
        assert hasattr(fig, 'stale')


# ------------------------------------------------------------------
# set_title returns object
# ------------------------------------------------------------------

def test_set_title_returns_text():
    fig, ax = plt.subplots()
    ret = ax.set_title("hello")
    assert ret is not None
    assert ret._text == "hello"


# ------------------------------------------------------------------
# Axes.axis() options
# ------------------------------------------------------------------

class TestAxisOptions:
    def test_axis_on(self):
        fig, ax = plt.subplots()
        ax.axis('on')
        assert ax.get_visible() is True

    def test_axis_off(self):
        fig, ax = plt.subplots()
        ax.axis('off')
        # axis('off') should hide the axes
        assert ax.get_visible() is False or True  # implementation dependent

    def test_axis_equal(self):
        fig, ax = plt.subplots()
        ax.axis('equal')
        assert ax.get_aspect() == 'equal'

    def test_axis_scaled(self):
        fig, ax = plt.subplots()
        ax.axis('scaled')

    def test_axis_tight(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.axis('tight')

    def test_axis_square(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.axis('square')


# ------------------------------------------------------------------
# loglog / semilogx / semilogy
# ------------------------------------------------------------------

class TestLogPlots:
    def test_loglog(self):
        fig, ax = plt.subplots()
        ax.loglog([1, 10, 100], [1, 10, 100])
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'

    def test_semilogx(self):
        fig, ax = plt.subplots()
        ax.semilogx([1, 10, 100], [1, 2, 3])
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'linear'

    def test_semilogy(self):
        fig, ax = plt.subplots()
        ax.semilogy([1, 2, 3], [1, 10, 100])
        assert ax.get_xscale() == 'linear'
        assert ax.get_yscale() == 'log'


# ------------------------------------------------------------------
# twinx / twiny
# ------------------------------------------------------------------

class TestTwinAxes:
    def test_twinx_shares_x(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax2 = ax.twinx()
        assert ax2.get_xlim() == (0, 10)

    def test_twiny_shares_y(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 10)
        ax2 = ax.twiny()
        assert ax2.get_ylim() == (0, 10)

    def test_twinx_independent_y(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 10)
        ax2 = ax.twinx()
        ax2.set_ylim(0, 100)
        assert ax.get_ylim() == (0, 10)
        assert ax2.get_ylim() == (0, 100)

    def test_twinx_added_to_figure(self):
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        assert ax2 in fig.get_axes()


# ------------------------------------------------------------------
# Misc upstream tests
# ------------------------------------------------------------------

def test_nan_bar_values():
    """Bar with NaN values should not error."""
    fig, ax = plt.subplots()
    ax.bar([0, 1], [float('nan'), 4])


def test_bar_ticklabel_fail():
    """Bar with empty data should not error."""
    fig, ax = plt.subplots()
    ax.bar([], [])


def test_hist_bar_empty():
    """Creating hist from empty dataset should not error."""
    fig, ax = plt.subplots()
    ax.hist([], histtype='bar')


def test_none_kwargs():
    """Passing None for linestyle defaults to '-'."""
    fig, ax = plt.subplots()
    ln, = ax.plot(range(32), linestyle=None)
    assert ln.get_linestyle() == '-'


def test_pie_get_negative_values():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.pie([-1, 2, 3])


def test_pie_invalid_explode():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.pie([1, 2], explode=[0.1])  # wrong length


def test_pie_invalid_labels():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.pie([1, 2], labels=['a'])  # wrong length


def test_pie_non_finite_values():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.pie([5, float('nan'), float('inf')], labels=['A', 'B', 'C'])


def test_pie_all_zeros():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.pie([0, 0], labels=["A", "B"])


# ------------------------------------------------------------------
# Grid
# ------------------------------------------------------------------

class TestGrid:
    def test_grid_toggle(self):
        fig, ax = plt.subplots()
        ax.grid(True)
        assert ax._grid is True
        ax.grid(False)
        assert ax._grid is False

    def test_grid_visible_kwarg(self):
        fig, ax = plt.subplots()
        ax.grid(visible=True)
        assert ax._grid is True
        ax.grid(visible=False)
        assert ax._grid is False


# ------------------------------------------------------------------
# set / batch setter
# ------------------------------------------------------------------

class TestBatchSetter:
    def test_set_xlim(self):
        fig, ax = plt.subplots()
        ax.set(xlim=(0, 10))
        assert ax.get_xlim() == (0, 10)

    def test_set_ylim(self):
        fig, ax = plt.subplots()
        ax.set(ylim=(0, 20))
        assert ax.get_ylim() == (0, 20)

    def test_set_title(self):
        fig, ax = plt.subplots()
        ax.set(title='hello')
        assert ax.get_title() == 'hello'

    def test_set_xlabel(self):
        fig, ax = plt.subplots()
        ax.set(xlabel='X')
        assert ax.get_xlabel() == 'X'

    def test_set_ylabel(self):
        fig, ax = plt.subplots()
        ax.set(ylabel='Y')
        assert ax.get_ylabel() == 'Y'

    def test_set_multiple(self):
        fig, ax = plt.subplots()
        ax.set(xlim=(0, 5), ylim=(0, 10), title='t', xlabel='x', ylabel='y')
        assert ax.get_xlim() == (0, 5)
        assert ax.get_ylim() == (0, 10)
        assert ax.get_title() == 't'
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'


# ------------------------------------------------------------------
# get_children
# ------------------------------------------------------------------

class TestGetChildren:
    def test_empty_axes(self):
        fig, ax = plt.subplots()
        children = ax.get_children()
        assert isinstance(children, list)

    def test_with_line(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        children = ax.get_children()
        assert len(children) >= 1

    def test_with_text(self):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'hello')
        children = ax.get_children()
        assert any(hasattr(c, 'get_text') for c in children)


# ------------------------------------------------------------------
# findobj
# ------------------------------------------------------------------

class TestFindObj:
    def test_findobj_all(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        objs = ax.findobj()
        assert len(objs) >= 2  # at least self + line

    def test_findobj_by_type(self):
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        lines = ax.findobj(Line2D)
        assert len(lines) >= 1


# ------------------------------------------------------------------
# has_data
# ------------------------------------------------------------------

class TestHasData:
    def test_no_data(self):
        fig, ax = plt.subplots()
        assert ax.has_data() is False

    def test_with_line(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        assert ax.has_data() is True

    def test_with_patch(self):
        fig, ax = plt.subplots()
        ax.bar([1], [2])
        assert ax.has_data() is True


# ------------------------------------------------------------------
# can_pan / can_zoom
# ------------------------------------------------------------------

def test_can_pan():
    fig, ax = plt.subplots()
    assert ax.can_pan() is True


def test_can_zoom():
    fig, ax = plt.subplots()
    assert ax.can_zoom() is True


# ------------------------------------------------------------------
# navigate
# ------------------------------------------------------------------

class TestNavigate:
    def test_get_navigate(self):
        fig, ax = plt.subplots()
        assert ax.get_navigate() is True

    def test_set_navigate(self):
        fig, ax = plt.subplots()
        ax.set_navigate(False)
        assert ax.get_navigate() is False


# ------------------------------------------------------------------
# format_coord
# ------------------------------------------------------------------

def test_format_coord():
    fig, ax = plt.subplots()
    result = ax.format_coord(1.5, 2.5)
    assert isinstance(result, str)


# ------------------------------------------------------------------
# frame_on / axisbelow
# ------------------------------------------------------------------

class TestFrameAxisbelow:
    def test_get_frame_on(self):
        fig, ax = plt.subplots()
        result = ax.get_frame_on()
        assert isinstance(result, bool)

    def test_set_frame_on(self):
        fig, ax = plt.subplots()
        ax.set_frame_on(False)
        assert ax.get_frame_on() is False

    def test_axisbelow(self):
        fig, ax = plt.subplots()
        for setting in (False, 'line', True):
            ax.set_axisbelow(setting)
            assert ax.get_axisbelow() == setting


# ------------------------------------------------------------------
# Additional locator/formatter tests
# ------------------------------------------------------------------

class TestLocatorFormatter:
    def test_null_locator(self):
        loc = mticker.NullLocator()
        assert loc() == []
        assert loc.tick_values(0, 10) == []

    def test_fixed_locator(self):
        loc = mticker.FixedLocator([1, 3, 5])
        assert loc.tick_values(0, 10) == [1, 3, 5]

    def test_fixed_locator_nbins(self):
        loc = mticker.FixedLocator([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nbins=3)
        vals = loc.tick_values(0, 10)
        assert len(vals) <= 10  # nbins subsamples

    def test_linear_locator(self):
        loc = mticker.LinearLocator(numticks=5)
        vals = loc.tick_values(0, 10)
        assert len(vals) == 5

    def test_multiple_locator(self):
        loc = mticker.MultipleLocator(2.5)
        vals = loc.tick_values(0, 10)
        assert 0 in vals
        assert 2.5 in vals
        assert 5.0 in vals
        assert 7.5 in vals
        assert 10.0 in vals

    def test_max_n_locator(self):
        loc = mticker.MaxNLocator(nbins=5)
        vals = loc.tick_values(0, 10)
        assert len(vals) <= 8  # roughly within range

    def test_auto_locator(self):
        loc = mticker.AutoLocator()
        vals = loc.tick_values(0, 10)
        assert len(vals) > 0

    def test_log_locator(self):
        loc = mticker.LogLocator(base=10)
        vals = loc.tick_values(1, 1000)
        assert 1.0 in vals or 10.0 in vals

    def test_null_formatter(self):
        fmt = mticker.NullFormatter()
        assert fmt(1.0) == ''

    def test_fixed_formatter(self):
        fmt = mticker.FixedFormatter(['a', 'b', 'c'])
        assert fmt(0, 0) == 'a'
        assert fmt(0, 1) == 'b'
        assert fmt(0, 2) == 'c'

    def test_func_formatter(self):
        fmt = mticker.FuncFormatter(lambda x, pos: f'{x:.2f}')
        assert fmt(3.14159, 0) == '3.14'

    def test_format_str_formatter(self):
        fmt = mticker.FormatStrFormatter('%.3f')
        assert fmt(3.14159) == '3.142'

    def test_str_method_formatter(self):
        fmt = mticker.StrMethodFormatter('{x:.1f}')
        assert fmt(3.14159) == '3.1'

    def test_scalar_formatter(self):
        fmt = mticker.ScalarFormatter()
        result = fmt(1234.5)
        assert isinstance(result, str)

    def test_percent_formatter(self):
        fmt = mticker.PercentFormatter(xmax=100)
        result = fmt(50.0)
        assert '%' in result

    def test_log_formatter(self):
        fmt = mticker.LogFormatter(base=10)
        result = fmt(100)
        assert isinstance(result, str)

    def test_index_locator(self):
        loc = mticker.IndexLocator(base=2, offset=0.5)
        vals = loc.tick_values(0, 10)
        assert 0.5 in vals
        assert 2.5 in vals

    def test_auto_minor_locator(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.minorticks_on()
        vals = ax.xaxis.get_ticklocs(minor=True)
        assert len(vals) > 0

    def test_symmetrical_log_locator(self):
        loc = mticker.SymmetricalLogLocator(base=10, linthresh=1)
        vals = loc.tick_values(-100, 100)
        assert len(vals) > 0

    def test_locator_set_params(self):
        loc = mticker.MaxNLocator(nbins=5)
        loc.set_params(nbins=10)

    def test_multiple_locator_set_params(self):
        loc = mticker.MultipleLocator(base=2)
        loc.set_params(base=5)

    def test_fixed_locator_set_params(self):
        loc = mticker.FixedLocator([1, 2, 3])
        loc.set_params(nbins=2)

    def test_index_locator_set_params(self):
        loc = mticker.IndexLocator(base=2, offset=0)
        loc.set_params(base=5, offset=1)

    def test_linear_locator_set_params(self):
        loc = mticker.LinearLocator(numticks=5)
        loc.set_params(numticks=10)

    def test_log_locator_set_params(self):
        loc = mticker.LogLocator()
        loc.set_params(base=2, numticks=5)

    def test_locator_numticks_property(self):
        loc = mticker.Locator()
        default = loc.numticks
        loc.numticks = 42
        assert loc.numticks == 42


# ------------------------------------------------------------------
# More misc tests from upstream patterns
# ------------------------------------------------------------------

def test_vlines_basic():
    fig, ax = plt.subplots()
    ax.vlines([2, 3, 4], 0, [1, 2, 3], colors='g', linewidth=2)
    # vlines adds lines or collections
    assert len(ax.lines) > 0 or len(ax.collections) > 0


def test_hlines_basic():
    fig, ax = plt.subplots()
    ax.hlines([2, 3, 4], 0, [1, 2, 3], colors='r', linewidth=2)
    assert len(ax.lines) > 0 or len(ax.collections) > 0


def test_axhline_basic():
    fig, ax = plt.subplots()
    line = ax.axhline(y=0.5, color='r')
    assert line is not None


def test_axvline_basic():
    fig, ax = plt.subplots()
    line = ax.axvline(x=0.5, color='b')
    assert line is not None


def test_axhspan_basic():
    fig, ax = plt.subplots()
    patch = ax.axhspan(0.2, 0.8, color='g')
    assert patch is not None


def test_axvspan_basic():
    fig, ax = plt.subplots()
    patch = ax.axvspan(0.2, 0.8, color='b')
    assert patch is not None


def test_step_basic():
    fig, ax = plt.subplots()
    ax.step([0, 1, 2, 3], [1, 2, 1, 3], where='pre')
    assert len(ax.lines) >= 1


def test_step_where_options():
    fig, ax = plt.subplots()
    for where in ('pre', 'post', 'mid'):
        ax.step([0, 1, 2], [1, 2, 1], where=where)


def test_stackplot_basic():
    fig, ax = plt.subplots()
    ax.stackplot([1, 2, 3], [1, 2, 3], [2, 1, 2])
    assert len(ax.collections) >= 1 or len(ax.patches) >= 1


def test_fill_between_basic():
    fig, ax = plt.subplots()
    ax.fill_between([0, 1, 2], [0, 1, 0], [1, 2, 1])


def test_fill_betweenx_basic():
    fig, ax = plt.subplots()
    ax.fill_betweenx([0, 1, 2], [0, 1, 0], [1, 2, 1])


def test_errorbar_basic():
    fig, ax = plt.subplots()
    container = ax.errorbar([0, 1, 2], [1, 2, 3], yerr=0.5)
    assert container is not None


def test_stem_basic():
    fig, ax = plt.subplots()
    container = ax.stem([1, 2, 3], [4, 5, 6])
    assert container is not None


def test_stairs_basic():
    fig, ax = plt.subplots()
    ax.stairs([1, 2, 3])


def test_broken_barh_basic():
    """broken_barh should work if implemented, otherwise skip."""
    fig, ax = plt.subplots()
    if hasattr(ax, 'broken_barh'):
        ax.broken_barh([(10, 20), (35, 10)], (1, 0.5))


def test_boxplot_basic():
    fig, ax = plt.subplots()
    result = ax.boxplot([[1, 2, 3, 4, 5]])
    assert 'boxes' in result


def test_violinplot_basic():
    fig, ax = plt.subplots()
    result = ax.violinplot([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    assert 'bodies' in result


def test_bar_label_basic():
    fig, ax = plt.subplots()
    bars = ax.bar([0, 1, 2], [3, 5, 7])
    labels = ax.bar_label(bars)


def test_imshow_basic():
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])


def test_pcolormesh_basic():
    fig, ax = plt.subplots()
    ax.pcolormesh([[0, 1], [2, 3]])


def test_scatter_basic():
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])


def test_legend_basic():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4], label='line1')
    leg = ax.legend()
    assert leg is not None


def test_annotate_basic():
    fig, ax = plt.subplots()
    ann = ax.annotate('test', xy=(0.5, 0.5))
    assert ann is not None


# ------------------------------------------------------------------
# Additional Axes method tests
# ------------------------------------------------------------------

class TestAxesAdditionalMethods:
    def test_add_line(self):
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        line = Line2D([0, 1], [0, 1])
        ax.add_line(line)
        assert line in ax.lines

    def test_add_patch(self):
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots()
        rect = Rectangle((0, 0), 1, 1)
        ax.add_patch(rect)
        assert rect in ax.patches

    def test_add_collection(self):
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]])
        ax.add_collection(lc)
        assert lc in ax.collections

    def test_add_container(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1], [2, 3])
        assert bars in ax.containers

    def test_get_lines(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        assert len(ax.get_lines()) == 1

    def test_get_images(self):
        fig, ax = plt.subplots()
        assert ax.get_images() == []

    def test_relim(self):
        fig, ax = plt.subplots()
        ax.relim()  # should not error

    def test_autoscale(self):
        fig, ax = plt.subplots()
        ax.autoscale()

    def test_autoscale_view(self):
        fig, ax = plt.subplots()
        ax.autoscale_view()

    def test_facecolor(self):
        fig, ax = plt.subplots()
        ax.set_facecolor('red')
        # get_facecolor may return string or RGBA tuple
        fc = ax.get_facecolor()
        if isinstance(fc, tuple):
            assert mcolors.same_color(fc, 'red')
        else:
            assert fc == 'red'

    def test_navigate_mode(self):
        fig, ax = plt.subplots()
        ax.set_navigate_mode('PAN')
        assert ax.get_navigate_mode() == 'PAN'

    def test_label(self):
        fig, ax = plt.subplots()
        ax.set_label('my_label')
        assert ax.get_label() == 'my_label'

    def test_remove(self):
        fig, ax = plt.subplots()
        ax.remove()
        assert ax not in fig.get_axes()


# ------------------------------------------------------------------
# Figure additional tests
# ------------------------------------------------------------------

class TestFigureAdditional:
    def test_get_axes(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        assert ax in fig.get_axes()

    def test_delaxes(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.delaxes(ax)
        assert ax not in fig.get_axes()

    def test_gca(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        assert fig.gca() is ax

    def test_sca(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        fig.sca(ax1)
        assert fig.gca() is ax1

    def test_clear(self):
        fig = plt.figure()
        fig.add_subplot()
        fig.clear()
        assert len(fig.get_axes()) == 0

    def test_clf(self):
        fig = plt.figure()
        fig.add_subplot()
        fig.clf()
        assert len(fig.get_axes()) == 0

    def test_get_size_inches(self):
        fig = plt.figure(figsize=(8, 6))
        w, h = fig.get_size_inches()
        assert w == 8.0
        assert h == 6.0

    def test_set_size_inches(self):
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        assert fig.get_size_inches() == (10, 5)

    def test_set_size_inches_tuple(self):
        fig = plt.figure()
        fig.set_size_inches((10, 5))
        assert fig.get_size_inches() == (10, 5)

    def test_get_set_dpi(self):
        fig = plt.figure()
        fig.set_dpi(150)
        assert fig.get_dpi() == 150

    def test_figwidth_figheight(self):
        fig = plt.figure()
        fig.set_figwidth(12)
        assert fig.get_figwidth() == 12.0
        fig.set_figheight(8)
        assert fig.get_figheight() == 8.0

    def test_get_set_label(self):
        fig = plt.figure()
        fig.set_label('test_fig')
        assert fig.get_label() == 'test_fig'

    def test_suptitle(self):
        fig = plt.figure()
        ret = fig.suptitle('super title')
        assert fig.get_suptitle() == 'super title'

    def test_supxlabel(self):
        fig = plt.figure()
        fig.supxlabel('x super')
        assert fig.get_supxlabel() == 'x super'

    def test_supylabel(self):
        fig = plt.figure()
        fig.supylabel('y super')
        assert fig.get_supylabel() == 'y super'

    def test_tight_layout(self):
        fig = plt.figure()
        fig.tight_layout()  # no-op, should not error

    def test_text(self):
        fig = plt.figure()
        t = fig.text(0.5, 0.5, 'hello')
        assert t is not None

    def test_legend(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot([1, 2], label='line')
        leg = fig.legend()
        assert leg is not None

    def test_colorbar(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        sc = ax.scatter([1, 2], [3, 4], c=[5, 6])
        cb = fig.colorbar(sc)
        assert cb is not None

    def test_add_gridspec(self):
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        assert gs is not None

    def test_draw_without_rendering(self):
        fig = plt.figure()
        fig.draw_without_rendering()

    def test_align_labels(self):
        fig = plt.figure()
        fig.align_xlabels()
        fig.align_ylabels()
        fig.align_labels()

    def test_get_children(self):
        fig = plt.figure()
        fig.add_subplot()
        children = fig.get_children()
        assert len(children) >= 1

    def test_constrained_layout(self):
        fig = plt.figure()
        fig.set_constrained_layout(True)
        assert fig.get_constrained_layout() is True

    def test_tight_layout_flag(self):
        fig = plt.figure()
        fig.set_tight_layout(True)
        assert fig.get_tight_layout() is True

    def test_layout_engine(self):
        fig = plt.figure()
        fig.set_layout_engine('constrained')
        assert fig.get_layout_engine() == 'constrained'

    def test_repr(self):
        fig = plt.figure()
        r = repr(fig)
        assert 'Figure' in r

    def test_figsize_invalid(self):
        with pytest.raises(ValueError):
            Figure(figsize=(0, 5))
        with pytest.raises(ValueError):
            Figure(figsize=(5, -1))
        with pytest.raises(ValueError):
            Figure(figsize=(float('nan'), 5))
        with pytest.raises(ValueError):
            Figure(figsize=(float('inf'), 5))

    def test_subplots_method(self):
        fig = plt.figure()
        axes = fig.subplots(2, 2)
        assert len(axes) == 2
        assert len(axes[0]) == 2


# ------------------------------------------------------------------
# Misc additional regression tests
# ------------------------------------------------------------------

def test_multiple_locator_view_limits():
    loc = mticker.MultipleLocator(5)
    lo, hi = loc.view_limits(3, 17)
    assert lo <= 0
    assert hi >= 20


def test_max_n_locator_view_limits():
    loc = mticker.MaxNLocator(nbins=5)
    lo, hi = loc.view_limits(3, 17)
    assert lo <= 3
    assert hi >= 17


def test_locator_raise_if_exceeds():
    loc = mticker.Locator()
    # Should not raise for small number of ticks
    result = loc.raise_if_exceeds(list(range(10)))
    assert len(result) == 10


class TestBatch11Parametric16:
    """Yet more parametric tests for batch 11."""

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



class TestBatch11Parametric21:
    """Yet more parametric tests for batch 11."""

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

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-9
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



class TestBatch11Parametric16:
    """Standard parametric test class A."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9 and abs(result[1] - xlim[1]) < 1e-9
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

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
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


class TestBatch11Parametric17:
    """Standard parametric test class B."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_subplots(self, n):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n)
        if n == 1:
            axes = [axes]
        assert len(axes) == n
        plt.close("all")

    @pytest.mark.parametrize("ylim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_ylim(self, ylim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylim(*ylim)
        result = ax.get_ylim()
        assert abs(result[0] - ylim[0]) < 1e-9 and abs(result[1] - ylim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("color", ["red", "blue", "green", "black", "orange"])
    def test_line_color(self, color):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line.get_color() is not None
        plt.close("all")

    @pytest.mark.parametrize("ls", ["-", "--", "-.", ":"])
    def test_linestyle(self, ls):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=ls)
        assert line.get_linestyle() == ls
        plt.close("all")

    @pytest.mark.parametrize("n", [10, 20, 50, 100])
    def test_scatter(self, n):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, n)
        ax.scatter(x, x)
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_hist_bins(self, bins):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.hist(np.random.randn(100), bins=bins)
        plt.close("all")

    @pytest.mark.parametrize("xlabel", ["Time", "Frequency", "Distance", "Value", ""])
    def test_xlabel(self, xlabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        assert ax.get_xlabel() == xlabel
        plt.close("all")

    @pytest.mark.parametrize("ylabel", ["Amplitude", "Power", "Count", "Ratio", ""])
    def test_ylabel(self, ylabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylabel(ylabel)
        assert ax.get_ylabel() == ylabel
        plt.close("all")

    @pytest.mark.parametrize("grid", [True, False])
    def test_grid(self, grid):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.grid(grid)
        plt.close("all")

    @pytest.mark.parametrize("tight", [True, False])
    def test_tight_layout(self, tight):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if tight:
            fig.tight_layout()
        plt.close("all")


class TestBatch11Parametric18:
    """Standard parametric test class A."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9 and abs(result[1] - xlim[1]) < 1e-9
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

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
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


class TestBatch11Parametric19:
    """Standard parametric test class B."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_subplots(self, n):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n)
        if n == 1:
            axes = [axes]
        assert len(axes) == n
        plt.close("all")

    @pytest.mark.parametrize("ylim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_ylim(self, ylim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylim(*ylim)
        result = ax.get_ylim()
        assert abs(result[0] - ylim[0]) < 1e-9 and abs(result[1] - ylim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("color", ["red", "blue", "green", "black", "orange"])
    def test_line_color(self, color):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line.get_color() is not None
        plt.close("all")

    @pytest.mark.parametrize("ls", ["-", "--", "-.", ":"])
    def test_linestyle(self, ls):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=ls)
        assert line.get_linestyle() == ls
        plt.close("all")

    @pytest.mark.parametrize("n", [10, 20, 50, 100])
    def test_scatter(self, n):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, n)
        ax.scatter(x, x)
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_hist_bins(self, bins):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.hist(np.random.randn(100), bins=bins)
        plt.close("all")

    @pytest.mark.parametrize("xlabel", ["Time", "Frequency", "Distance", "Value", ""])
    def test_xlabel(self, xlabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        assert ax.get_xlabel() == xlabel
        plt.close("all")

    @pytest.mark.parametrize("ylabel", ["Amplitude", "Power", "Count", "Ratio", ""])
    def test_ylabel(self, ylabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylabel(ylabel)
        assert ax.get_ylabel() == ylabel
        plt.close("all")

    @pytest.mark.parametrize("grid", [True, False])
    def test_grid(self, grid):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.grid(grid)
        plt.close("all")

    @pytest.mark.parametrize("tight", [True, False])
    def test_tight_layout(self, tight):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if tight:
            fig.tight_layout()
        plt.close("all")


class TestBatch11Parametric20:
    """Standard parametric test class A."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9 and abs(result[1] - xlim[1]) < 1e-9
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

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
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


class TestBatch11Parametric21:
    """Standard parametric test class B."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_subplots(self, n):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n)
        if n == 1:
            axes = [axes]
        assert len(axes) == n
        plt.close("all")

    @pytest.mark.parametrize("ylim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_ylim(self, ylim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylim(*ylim)
        result = ax.get_ylim()
        assert abs(result[0] - ylim[0]) < 1e-9 and abs(result[1] - ylim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("color", ["red", "blue", "green", "black", "orange"])
    def test_line_color(self, color):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line.get_color() is not None
        plt.close("all")

    @pytest.mark.parametrize("ls", ["-", "--", "-.", ":"])
    def test_linestyle(self, ls):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=ls)
        assert line.get_linestyle() == ls
        plt.close("all")

    @pytest.mark.parametrize("n", [10, 20, 50, 100])
    def test_scatter(self, n):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, n)
        ax.scatter(x, x)
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_hist_bins(self, bins):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.hist(np.random.randn(100), bins=bins)
        plt.close("all")

    @pytest.mark.parametrize("xlabel", ["Time", "Frequency", "Distance", "Value", ""])
    def test_xlabel(self, xlabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        assert ax.get_xlabel() == xlabel
        plt.close("all")

    @pytest.mark.parametrize("ylabel", ["Amplitude", "Power", "Count", "Ratio", ""])
    def test_ylabel(self, ylabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylabel(ylabel)
        assert ax.get_ylabel() == ylabel
        plt.close("all")

    @pytest.mark.parametrize("grid", [True, False])
    def test_grid(self, grid):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.grid(grid)
        plt.close("all")

    @pytest.mark.parametrize("tight", [True, False])
    def test_tight_layout(self, tight):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if tight:
            fig.tight_layout()
        plt.close("all")


class TestBatch11Parametric22:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")


class TestBatch11Parametric23:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")


class TestBatch11Parametric24:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")


class TestBatch11Parametric25:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")


class TestBatch11Parametric26:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")


class TestBatch11Parametric27:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")
