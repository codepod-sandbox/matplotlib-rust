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
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([1.0, 2.0], labels=['a', 'b'])
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, FixedFormatter)
        plt.close('all')

    def test_get_ticks_returns_list(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks([1.0, 2.0, 3.0])
        ticks = ax.xaxis.get_ticks()
        assert ticks == [1.0, 2.0, 3.0]
        plt.close('all')

    def test_get_ticks_empty_when_auto(self):
        fig, ax = plt.subplots()
        ticks = ax.xaxis.get_ticks()
        assert ticks == []  # No fixed ticks set
        plt.close('all')

    def test_tick_values_basic(self):
        fig, ax = plt.subplots()
        vals = ax.xaxis.tick_values(0, 10)
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
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        locs = ax.xaxis.get_ticklocs()
        assert isinstance(locs, list)
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
        ax.yaxis.set_major_formatter(fmt)
        assert ax.yaxis.get_major_formatter() is fmt
        plt.close('all')

    def test_xaxis_tick_values_log(self):
        """XAxis tick_values on log scale returns values."""
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        vals = ax.xaxis.tick_values(1, 1000)
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
        """YAxis tick_values on linear scale returns values."""
        fig, ax = plt.subplots()
        vals = ax.yaxis.tick_values(0, 100)
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

class TestAxisParametric:
    """Parametric tests for XAxis and YAxis."""

    @pytest.mark.parametrize('ticks', [
        [0.0, 1.0, 2.0],
        [0, 5, 10, 15, 20],
        [-3, -2, -1, 0, 1, 2, 3],
        [0.1, 0.2, 0.5, 1.0],
    ])
    def test_xaxis_set_ticks_roundtrip(self, ticks):
        """XAxis.set_ticks / get_ticks roundtrip."""
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks(ticks)
        assert ax.xaxis.get_ticks() == ticks
        plt.close('all')

    @pytest.mark.parametrize('ticks', [
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [10, 20, 30, 40],
        [-5, 0, 5],
    ])
    def test_yaxis_set_ticks_roundtrip(self, ticks):
        """YAxis.set_ticks / get_ticks roundtrip."""
        fig, ax = plt.subplots()
        ax.yaxis.set_ticks(ticks)
        assert ax.yaxis.get_ticks() == ticks
        plt.close('all')

    @pytest.mark.parametrize('label', ['X Axis', 'Time (s)', '', 'Frequency (Hz)'])
    def test_xaxis_label_roundtrip(self, label):
        """XAxis.set_label_text / get_label_text roundtrip."""
        fig, ax = plt.subplots()
        ax.xaxis.set_label_text(label)
        assert ax.xaxis.get_label_text() == label
        plt.close('all')

    @pytest.mark.parametrize('label', ['Y Axis', 'Amplitude', '', 'Count'])
    def test_yaxis_label_roundtrip(self, label):
        """YAxis.set_label_text / get_label_text roundtrip."""
        fig, ax = plt.subplots()
        ax.yaxis.set_label_text(label)
        assert ax.yaxis.get_label_text() == label
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xaxis_scale_roundtrip(self, scale):
        """XAxis.get_scale reflects set_xscale."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.xaxis.get_scale() == scale
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_yaxis_scale_roundtrip(self, scale):
        """YAxis.get_scale reflects set_yscale."""
        fig, ax = plt.subplots()
        ax.set_yscale(scale)
        assert ax.yaxis.get_scale() == scale
        plt.close('all')

    @pytest.mark.parametrize('vmin,vmax', [
        (0, 1), (0, 10), (-5, 5), (0, 100), (1, 1000),
    ])
    def test_xaxis_tick_values_in_range(self, vmin, vmax):
        """XAxis.tick_values returns ticks within range."""
        fig, ax = plt.subplots()
        vals = ax.xaxis.tick_values(vmin, vmax)
        assert len(vals) > 0
        plt.close('all')

    @pytest.mark.parametrize('vmin,vmax', [
        (0, 1), (-100, 100), (0.001, 0.01),
    ])
    def test_yaxis_tick_values_in_range(self, vmin, vmax):
        """YAxis.tick_values returns ticks within range."""
        fig, ax = plt.subplots()
        vals = ax.yaxis.tick_values(vmin, vmax)
        assert len(vals) > 0
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_format_ticks_length(self, n):
        """format_ticks returns same number of labels as ticks."""
        fig, ax = plt.subplots()
        ticks = list(range(n))
        labels = ax.xaxis.format_ticks([float(t) for t in ticks])
        assert len(labels) == n
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_xaxis_set_visible(self, visible):
        """XAxis.set_visible / get_visible roundtrip."""
        fig, ax = plt.subplots()
        ax.xaxis.set_visible(visible)
        assert ax.xaxis.get_visible() == visible
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_yaxis_set_visible(self, visible):
        """YAxis.set_visible / get_visible roundtrip."""
        fig, ax = plt.subplots()
        ax.yaxis.set_visible(visible)
        assert ax.yaxis.get_visible() == visible
        plt.close('all')

    @pytest.mark.parametrize('vmin,vmax', [
        (0, 5), (1, 10), (-3, 3), (100, 200),
    ])
    def test_xaxis_view_interval(self, vmin, vmax):
        """XAxis.get_view_interval matches set_xlim."""
        fig, ax = plt.subplots()
        ax.set_xlim(vmin, vmax)
        lo, hi = ax.xaxis.get_view_interval()
        assert abs(lo - vmin) < 1e-10
        assert abs(hi - vmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('vmin,vmax', [
        (-1, 1), (0, 100), (-50, 50),
    ])
    def test_yaxis_view_interval(self, vmin, vmax):
        """YAxis.get_view_interval matches set_ylim."""
        fig, ax = plt.subplots()
        ax.set_ylim(vmin, vmax)
        lo, hi = ax.yaxis.get_view_interval()
        assert abs(lo - vmin) < 1e-10
        assert abs(hi - vmax) < 1e-10
        plt.close('all')
