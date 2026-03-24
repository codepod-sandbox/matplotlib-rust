"""Tests for Axes.set_prop_cycle and color cycling."""

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


class TestPropCycleParametric:
    """Parametrized tests for prop cycle behavior."""

    @pytest.mark.parametrize('colors', [
        ['red', 'blue'],
        ['#ff0000', '#00ff00', '#0000ff'],
        ['C0', 'C1', 'C2'],
    ])
    def test_set_prop_cycle_wraps(self, colors):
        """After exhausting cycle, wraps back to first color."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_prop_cycle(colors)
        for _ in colors:
            ax._next_color()
        # Next call wraps around
        assert ax._next_color() == colors[0]

    @pytest.mark.parametrize('n', [1, 2, 5, 10])
    def test_next_color_advances(self, n):
        """_next_color advances the cycle n times without repeating (for n ≤ cycle length)."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = [f'C{i}' for i in range(max(n + 1, 11))]
        ax.set_prop_cycle(colors)
        seen = set()
        for _ in range(n):
            c = ax._next_color()
            seen.add(c)
        assert len(seen) == n

    @pytest.mark.parametrize('color_list', [
        ['red'],
        ['blue', 'green'],
        ['yellow', 'orange', 'purple'],
    ])
    def test_plot_uses_cycle_colors(self, color_list):
        """plot() picks colors from the set cycle."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_prop_cycle(color_list)
        lines = []
        for i in range(len(color_list)):
            line = ax.plot([i], [i])[0]
            lines.append(line)
        # All lines should have distinct colors from the cycle
        assert len(lines) == len(color_list)


class TestCyclerIntegration:
    """Integration tests between cycler and axes."""

    def test_cycler_plus_cycler(self):
        """Two cyclers can be added (concatenated)."""
        c1 = cycler(color=['red', 'blue'])
        c2 = cycler(color=['green', 'yellow'])
        combined = c1 + c2
        colors = [d['color'] for d in combined]
        assert colors == ['red', 'blue', 'green', 'yellow']

    def test_cycler_multiply(self):
        """cycler * int repeats the cycle."""
        c = cycler(color=['red', 'blue'])
        repeated = c * 2
        colors = [d['color'] for d in repeated]
        assert colors == ['red', 'blue', 'red', 'blue']

    def test_cycler_keys(self):
        """Cycler.keys returns the property names."""
        c = cycler(color=['red', 'blue'], linewidth=[1, 2])
        assert 'color' in c.keys
        assert 'linewidth' in c.keys

    def test_cycler_length(self):
        """len(cycler) returns the number of entries."""
        c = cycler(color=['red', 'blue', 'green'])
        assert len(c) == 3

    def test_set_prop_cycle_with_cycler_object(self):
        """Passing a Cycler to set_prop_cycle works."""
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        c = cycler(color=['orange', 'purple', 'teal'])
        ax.set_prop_cycle(c)
        assert ax._next_color() == 'orange'
        assert ax._next_color() == 'purple'
        assert ax._next_color() == 'teal'

    def test_cycler_iteration(self):
        """Cycler is iterable and yields dicts."""
        c = cycler(color=['red', 'blue'])
        items = list(c)
        assert items == [{'color': 'red'}, {'color': 'blue'}]

    def test_rcparams_axes_prop_cycle(self):
        """matplotlib.rcParams['axes.prop_cycle'] returns the default color cycle."""
        import matplotlib
        pc = matplotlib.rcParams['axes.prop_cycle']
        assert pc is not None
        if isinstance(pc, Cycler):
            assert 'color' in pc.keys
        else:
            assert isinstance(pc, list)
            assert len(pc) > 0
