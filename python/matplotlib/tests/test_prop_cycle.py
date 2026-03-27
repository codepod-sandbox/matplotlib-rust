"""Tests for Axes.set_prop_cycle and color cycling."""

import itertools

import pytest

from matplotlib.figure import Figure
from matplotlib.cycler import cycler, Cycler


class TestPropCycle:
    def _make_axes(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        return ax

    def test_default_cycle(self):
        ax = self._make_axes()
        c1 = ax._next_color()
        c2 = ax._next_color()
        assert c1 != c2

    def test_set_prop_cycle_list(self):
        ax = self._make_axes()
        ax.set_prop_cycle(['red', 'green', 'blue'])
        assert ax._next_color() == 'red'
        assert ax._next_color() == 'green'
        assert ax._next_color() == 'blue'
        # Wraps around
        assert ax._next_color() == 'red'

    def test_set_prop_cycle_color_kwarg(self):
        ax = self._make_axes()
        ax.set_prop_cycle(color=['cyan', 'magenta'])
        assert ax._next_color() == 'cyan'
        assert ax._next_color() == 'magenta'

    def test_set_prop_cycle_key_values(self):
        ax = self._make_axes()
        ax.set_prop_cycle('color', ['#ff0000', '#00ff00'])
        assert ax._next_color() == '#ff0000'
        assert ax._next_color() == '#00ff00'

    def test_set_prop_cycle_none_resets(self):
        ax = self._make_axes()
        ax.set_prop_cycle(['red', 'green'])
        ax._next_color()  # consume one
        ax.set_prop_cycle(None)
        # After reset, should use default cycle
        c = ax._next_color()
        assert c.startswith('#')  # default cycle is hex

    def test_set_prop_cycle_cycler_object(self):
        ax = self._make_axes()
        c = Cycler('color', ['orange', 'purple'])
        ax.set_prop_cycle(c)
        assert ax._next_color() == 'orange'
        assert ax._next_color() == 'purple'

    def test_prop_cycle_affects_plot(self):
        ax = self._make_axes()
        ax.set_prop_cycle(color=['red', 'blue'])
        lines = ax.plot([1, 2], [3, 4])
        # The line should use the first color from the cycle
        assert lines[0].get_color() == '#ff0000'

    def test_prop_cycle_sequential_plots(self):
        ax = self._make_axes()
        ax.set_prop_cycle(color=['red', 'blue'])
        l1 = ax.plot([1, 2], [3, 4])
        l2 = ax.plot([1, 2], [5, 6])
        # First should be red, second blue
        assert l1[0].get_color() == '#ff0000'
        assert l2[0].get_color() == '#0000ff'


class TestPropCycleEdgeCases:
    def _make_axes(self):
        fig = Figure()
        return fig.add_subplot(1, 1, 1)

    def test_single_color_cycle(self):
        ax = self._make_axes()
        ax.set_prop_cycle(['red'])
        assert ax._next_color() == 'red'
        assert ax._next_color() == 'red'

    def test_many_colors(self):
        ax = self._make_axes()
        colors = [f'C{i}' for i in range(10)]
        ax.set_prop_cycle(colors)
        for c in colors:
            assert ax._next_color() == c


# ===================================================================
# Additional prop cycle tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cycler import cycler, Cycler
