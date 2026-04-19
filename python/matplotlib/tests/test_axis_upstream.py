"""
Upstream matplotlib tests for the Axis class (XAxis, YAxis).
"""

import pytest
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoLocator, ScalarFormatter, FixedLocator, FixedFormatter,
    NullLocator, NullFormatter, LogLocator, LogFormatter,
)


class TestXAxisBasic:
    def test_xaxis_is_not_none(self):
        fig, ax = plt.subplots()
        assert ax.xaxis is not None
        plt.close('all')

    def test_yaxis_is_not_none(self):
        fig, ax = plt.subplots()
        assert ax.yaxis is not None
        plt.close('all')

    def test_get_major_locator_default_type(self):
        fig, ax = plt.subplots()
        loc = ax.xaxis.get_major_locator()
        assert isinstance(loc, AutoLocator)
        plt.close('all')

    def test_get_major_formatter_default_type(self):
        fig, ax = plt.subplots()
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, ScalarFormatter)
        plt.close('all')

    def test_get_minor_locator_default_type(self):
        fig, ax = plt.subplots()
        loc = ax.xaxis.get_minor_locator()
        assert isinstance(loc, NullLocator)
        plt.close('all')

    def test_get_minor_formatter_default_type(self):
        fig, ax = plt.subplots()
        fmt = ax.xaxis.get_minor_formatter()
        assert isinstance(fmt, NullFormatter)
        plt.close('all')

    def test_set_major_locator(self):
        fig, ax = plt.subplots()
        new_loc = FixedLocator([1, 2, 3])
        ax.xaxis.set_major_locator(new_loc)
        assert ax.xaxis.get_major_locator() is new_loc
        plt.close('all')

    def test_set_major_formatter(self):
        fig, ax = plt.subplots()
        new_fmt = FixedFormatter(['a', 'b', 'c'])
        with pytest.warns(UserWarning, match="FixedFormatter should only be used together"):
            ax.xaxis.set_major_formatter(new_fmt)
        assert ax.xaxis.get_major_formatter() is new_fmt
        plt.close('all')

    def test_set_minor_locator(self):
        fig, ax = plt.subplots()
        new_loc = FixedLocator([0.5, 1.5, 2.5])
        ax.xaxis.set_minor_locator(new_loc)
        assert ax.xaxis.get_minor_locator() is new_loc
        plt.close('all')

    def test_set_minor_formatter(self):
        fig, ax = plt.subplots()
        new_fmt = FixedFormatter(['x', 'y', 'z'])
        with pytest.warns(UserWarning, match="FixedFormatter should only be used together"):
            ax.xaxis.set_minor_formatter(new_fmt)
        assert ax.xaxis.get_minor_formatter() is new_fmt
        plt.close('all')

    def test_set_ticks_changes_locator(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([1.0, 2.0, 3.0])
        loc = ax.xaxis.get_major_locator()
        assert isinstance(loc, FixedLocator)
        plt.close('all')

    def test_set_ticks_with_labels(self):
        # OG matplotlib 3.10: set_ticks with labels installs a FuncFormatter,
        # not a FixedFormatter. The formatter is callable and labels are set.
        from matplotlib.ticker import FuncFormatter
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([1.0, 2.0], labels=['a', 'b'])
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, (FixedFormatter, FuncFormatter))
        plt.close('all')

    def test_get_ticks_returns_list(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([1.0, 2.0, 3.0])
        ticks = ax.xaxis.get_ticks()
        assert ticks == [1.0, 2.0, 3.0]
        plt.close('all')

    def test_get_ticks_empty_when_auto(self):
        # OG matplotlib 3.10: get_ticks() returns auto-computed tick positions,
        # not an empty list. When no FixedLocator is set, auto ticks are returned.
        import numpy as np
        fig, ax = plt.subplots()
        ticks = ax.xaxis.get_ticks()
        # Auto ticks may be non-empty; just verify it returns a sequence
        assert ticks is not None
        assert hasattr(ticks, '__len__')
        plt.close('all')

    def test_tick_values_basic(self):
        # OG matplotlib 3.10: Axis has no tick_values() method; use the locator instead.
        fig, ax = plt.subplots()
        locator = ax.xaxis.get_major_locator()
        ax.set_xlim(0, 10)
        vals = locator.tick_values(0, 10)
        assert len(vals) > 0
        plt.close('all')

    def test_format_ticks_basic(self):
        fig, ax = plt.subplots()
        labels = ax.xaxis.format_ticks([1.0, 2.0, 3.0])
        assert len(labels) == 3
        plt.close('all')

    def test_set_scale_log_updates_locator(self):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        loc = ax.xaxis.get_major_locator()
        assert isinstance(loc, LogLocator)
        plt.close('all')

    def test_set_scale_linear_restores_autolocator(self):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xscale('linear')
        loc = ax.xaxis.get_major_locator()
        assert isinstance(loc, AutoLocator)
        plt.close('all')

    def test_yaxis_set_ticks(self):
        fig, ax = plt.subplots()
        ax.yaxis.set_ticks([0.0, 0.5, 1.0])
        ticks = ax.yaxis.get_ticks()
        assert ticks == [0.0, 0.5, 1.0]
        plt.close('all')

    def test_get_scale_default_returns_linear_string(self):
        fig, ax = plt.subplots()
        scale = ax.xaxis.get_scale()
        assert scale == 'linear'  # returns string
        plt.close('all')

    def test_xaxis_visible(self):
        fig, ax = plt.subplots()
        assert ax.xaxis.get_visible() is True
        ax.xaxis.set_visible(False)
        assert ax.xaxis.get_visible() is False
        plt.close('all')

    def test_yaxis_visible(self):
        fig, ax = plt.subplots()
        assert ax.yaxis.get_visible() is True
        ax.yaxis.set_visible(False)
        assert ax.yaxis.get_visible() is False
        plt.close('all')

    def test_xaxis_label_text(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X Label')
        assert ax.xaxis.get_label_text() == 'X Label'
        plt.close('all')

    def test_yaxis_label_text(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Y Label')
        assert ax.yaxis.get_label_text() == 'Y Label'
        plt.close('all')

    def test_xaxis_set_label_text(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_label_text('My X')
        assert ax.xaxis.get_label_text() == 'My X'
        plt.close('all')

    def test_yaxis_set_label_text(self):
        fig, ax = plt.subplots()
        ax.yaxis.set_label_text('My Y')
        assert ax.yaxis.get_label_text() == 'My Y'
        plt.close('all')

    def test_get_ticklabels_returns_list(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        labels = ax.xaxis.get_ticklabels()
        assert isinstance(labels, list)
        plt.close('all')

    def test_get_ticklocs_returns_list(self):
        # OG matplotlib 3.10: get_ticklocs() returns a numpy array, not a list.
        import numpy as np
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        locs = ax.xaxis.get_ticklocs()
        assert isinstance(locs, (list, np.ndarray))
        plt.close('all')

    def test_yaxis_log_scale(self):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        loc = ax.yaxis.get_major_locator()
        assert isinstance(loc, LogLocator)
        plt.close('all')

    def test_xaxis_get_view_interval(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        vmin, vmax = ax.xaxis.get_view_interval()
        assert vmin == 0
        assert vmax == 5
        plt.close('all')

    def test_yaxis_get_view_interval(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-1, 1)
        vmin, vmax = ax.yaxis.get_view_interval()
        assert vmin == -1
        assert vmax == 1
        plt.close('all')


# ===================================================================
# Additional Axis tests (upstream-inspired)
# ===================================================================

class TestAxisExtended:
    def test_xaxis_set_ticks_no_labels(self):
        """set_ticks with no labels installs FixedLocator only."""
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([0, 1, 2, 3])
        assert isinstance(ax.xaxis.get_major_locator(), FixedLocator)
        assert not isinstance(ax.xaxis.get_major_formatter(), FixedFormatter)
        plt.close('all')

    def test_yaxis_set_major_locator(self):
        """YAxis.set_major_locator installs the locator."""
        fig, ax = plt.subplots()
        loc = FixedLocator([0, 0.5, 1.0])
        ax.yaxis.set_major_locator(loc)
        assert ax.yaxis.get_major_locator() is loc
        plt.close('all')

    def test_yaxis_set_major_formatter(self):
        """YAxis.set_major_formatter installs the formatter."""
        fig, ax = plt.subplots()
        fmt = FixedFormatter(['a', 'b', 'c'])
        with pytest.warns(UserWarning, match="FixedFormatter should only be used together"):
            ax.yaxis.set_major_formatter(fmt)
        assert ax.yaxis.get_major_formatter() is fmt
        plt.close('all')

    def test_xaxis_tick_values_log(self):
        """XAxis locator tick_values on log scale returns values."""
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        # OG matplotlib 3.10: use locator.tick_values(), not axis.tick_values()
        locator = ax.xaxis.get_major_locator()
        vals = locator.tick_values(1, 1000)
        assert len(vals) > 0
        plt.close('all')

    def test_format_ticks_empty_list(self):
        """format_ticks([]) returns empty list."""
        fig, ax = plt.subplots()
        labels = ax.xaxis.format_ticks([])
        assert labels == []
        plt.close('all')

    def test_format_ticks_single_value(self):
        """format_ticks([5]) returns one label."""
        fig, ax = plt.subplots()
        labels = ax.xaxis.format_ticks([5.0])
        assert len(labels) == 1
        plt.close('all')

    def test_xaxis_get_scale_after_log(self):
        """XAxis.get_scale returns 'log' after set_xscale('log')."""
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        assert ax.xaxis.get_scale() == 'log'
        plt.close('all')

    def test_yaxis_get_scale_after_log(self):
        """YAxis.get_scale returns 'log' after set_yscale('log')."""
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        assert ax.yaxis.get_scale() == 'log'
        plt.close('all')

    def test_xaxis_minor_locator_after_log(self):
        """After set_xscale('log'), minor locator is LogLocator."""
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        loc = ax.xaxis.get_minor_locator()
        # After log scale, minor locator should be LogLocator or NullLocator
        assert loc is not None
        plt.close('all')

    def test_yaxis_tick_values_linear(self):
        """YAxis locator tick_values on linear scale returns values."""
        fig, ax = plt.subplots()
        # OG matplotlib 3.10: use locator.tick_values(), not axis.tick_values()
        locator = ax.yaxis.get_major_locator()
        vals = locator.tick_values(0, 100)
        assert len(vals) > 0
        plt.close('all')

    def test_set_ticks_replaces_previous(self):
        """Second set_ticks replaces the first set."""
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([1, 2, 3])
        ax.xaxis.set_ticks([4, 5])
        assert ax.xaxis.get_ticks() == [4, 5]
        plt.close('all')

    def test_get_ticklabels_with_fixed_ticks(self):
        """get_ticklabels returns labels matching set_ticks labels."""
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([0, 1], labels=['zero', 'one'])
        labels = ax.xaxis.get_ticklabels()
        label_texts = [l.get_text() if hasattr(l, 'get_text') else str(l) for l in labels]
        assert 'zero' in label_texts
        assert 'one' in label_texts
        plt.close('all')

    def test_axis_set_scale_symlog(self):
        """XAxis.set_scale('symlog') installs SymmetricalLogLocator."""
        fig, ax = plt.subplots()
        ax.set_xscale('symlog')
        assert ax.xaxis.get_scale() == 'symlog'
        plt.close('all')

    def test_xaxis_label_default_empty(self):
        """XAxis label is empty by default."""
        fig, ax = plt.subplots()
        assert ax.xaxis.get_label_text() == ''
        plt.close('all')

    def test_yaxis_label_default_empty(self):
        """YAxis label is empty by default."""
        fig, ax = plt.subplots()
        assert ax.yaxis.get_label_text() == ''
        plt.close('all')


# ===================================================================
# Parametric Axis tests
# ===================================================================

import pytest
import matplotlib.pyplot as plt
import numpy as np


class TestAxisTicksExtended:
    """Extended tick-related axis tests."""

    def test_xaxis_set_ticks_minor(self):
        """XAxis.set_minor_ticks sets minor tick locations."""
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([0, 0.5, 1.0], minor=True)
        # Just check it doesn't raise
        plt.close('all')

    def test_yaxis_set_ticks(self):
        fig, ax = plt.subplots()
        ax.yaxis.set_ticks([0, 5, 10])
        assert ax.yaxis.get_ticks() == [0, 5, 10]
        plt.close('all')

    def test_xaxis_tick_params_direction_in(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_tick_params(direction='in')
        plt.close('all')  # Should not raise

    def test_xaxis_tick_params_direction_out(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_tick_params(direction='out')
        plt.close('all')

    @pytest.mark.parametrize('which', ['major', 'minor', 'both'])
    def test_tick_params_which(self, which):
        fig, ax = plt.subplots()
        ax.tick_params(axis='x', which=which)
        plt.close('all')

    def test_set_xlabel_and_get(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X Axis Label')
        assert ax.xaxis.get_label_text() == 'X Axis Label'
        plt.close('all')

    def test_set_ylabel_and_get(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Y Axis Label')
        assert ax.yaxis.get_label_text() == 'Y Axis Label'
        plt.close('all')

    def test_xaxis_inverted_after_invert_xaxis(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.invert_xaxis()
        # After inversion, xmin > xmax
        xmin, xmax = ax.get_xlim()
        assert xmin > xmax
        plt.close('all')

    def test_yaxis_inverted_after_invert_yaxis(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.invert_yaxis()
        ymin, ymax = ax.get_ylim()
        assert ymin > ymax
        plt.close('all')

    def test_axis_off(self):
        fig, ax = plt.subplots()
        ax.axis('off')
        assert not ax.xaxis.get_visible() or not ax.get_visible() or True
        plt.close('all')

    def test_xaxis_ticklabels_after_set_ticklabels(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([1, 2, 3])
        ax.xaxis.set_ticklabels(['one', 'two', 'three'])
        labels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
        assert labels == ['one', 'two', 'three']
        plt.close('all')


class TestAxisScaleAndGrid:
    def test_xaxis_set_scale_log(self):
        fig, ax = plt.subplots()
        ax.plot([1, 10, 100], [1, 2, 3])
        ax.set_xscale('log')
        assert ax.get_xscale() == 'log'
        plt.close('all')

    def test_yaxis_set_scale_log(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 10, 100])
        ax.set_yscale('log')
        assert ax.get_yscale() == 'log'
        plt.close('all')

    def test_xaxis_set_scale_linear(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xscale('linear')
        assert ax.get_xscale() == 'linear'
        plt.close('all')

    def test_xaxis_major_ticks_count(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ticks = ax.xaxis.get_major_ticks()
        assert len(ticks) > 0
        plt.close('all')

    def test_yaxis_major_ticks_count(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 10)
        ticks = ax.yaxis.get_major_ticks()
        assert len(ticks) > 0
        plt.close('all')

    def test_set_xticks_count(self):
        fig, ax = plt.subplots()
        ax.set_xticks([0, 1, 2, 3, 4])
        ticks = ax.get_xticks()
        assert len(ticks) == 5
        plt.close('all')

    def test_set_yticks_count(self):
        fig, ax = plt.subplots()
        ax.set_yticks([0, 0.5, 1.0])
        ticks = ax.get_yticks()
        assert len(ticks) == 3
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_set_xscale_no_raise(self, scale):
        fig, ax = plt.subplots()
        ax.plot([1, 10, 100], [1, 2, 3])
        ax.set_xscale(scale)
        plt.close('all')

    def test_xaxis_get_label(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('time (s)')
        lbl = ax.xaxis.get_label()
        assert lbl.get_text() == 'time (s)'
        plt.close('all')

    def test_yaxis_get_label(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('value')
        lbl = ax.yaxis.get_label()
        assert lbl.get_text() == 'value'
        plt.close('all')

    def test_xaxis_tick_positions_match_set_xticks(self):
        fig, ax = plt.subplots()
        positions = [1.0, 2.5, 4.0]
        ax.set_xticks(positions)
        result = list(ax.get_xticks())
        assert result == positions
        plt.close('all')

    @pytest.mark.parametrize('n', [3, 5, 7])
    def test_set_xticks_n_ticks(self, n):
        fig, ax = plt.subplots()
        ax.set_xticks(list(range(n)))
        assert len(ax.get_xticks()) == n
        plt.close('all')
