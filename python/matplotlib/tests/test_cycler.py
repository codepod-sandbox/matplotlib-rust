"""Tests for matplotlib.cycler --- property cycling."""

import pytest

from matplotlib.cycler import Cycler, cycler


class TestCycler:
    def test_basic_creation(self):
        c = Cycler('color', ['red', 'green', 'blue'])
        assert len(c) == 3

    def test_iteration(self):
        c = Cycler('color', ['red', 'green', 'blue'])
        items = list(c)
        assert items[0] == {'color': 'red'}
        assert items[1] == {'color': 'green'}
        assert items[2] == {'color': 'blue'}

    def test_getitem(self):
        c = Cycler('color', ['red', 'green', 'blue'])
        assert c[0] == {'color': 'red'}
        assert c[2] == {'color': 'blue'}

    def test_by_key(self):
        c = Cycler('color', ['red', 'green', 'blue'])
        keys = c.by_key()
        assert keys['color'] == ['red', 'green', 'blue']

    def test_keys_property(self):
        c = Cycler('color', ['red', 'green'])
        assert 'color' in c.keys

    def test_add(self):
        c1 = Cycler('color', ['red', 'green'])
        c2 = Cycler('color', ['blue'])
        c3 = c1 + c2
        assert len(c3) == 3

    def test_mul_int(self):
        c = Cycler('color', ['red', 'green'])
        c2 = c * 3
        assert len(c2) == 6

    def test_mul_cycler(self):
        c1 = Cycler('color', ['red', 'green'])
        c2 = Cycler('linewidth', [1, 2])
        c3 = c1 * c2
        assert len(c3) == 4

    def test_repr(self):
        c = Cycler('color', ['red'])
        r = repr(c)
        assert 'cycler' in r

    def test_empty(self):
        c = Cycler()
        assert len(c) == 0


class TestCyclerFunction:
    def test_basic(self):
        c = cycler('color', ['red', 'green', 'blue'])
        assert len(c) == 3

    def test_kwargs(self):
        c = cycler(color=['red', 'green', 'blue'])
        items = list(c)
        assert items[0] == {'color': 'red'}

    def test_empty(self):
        c = cycler()
        assert len(c) == 0

    def test_dict_input(self):
        c = cycler({'color': ['red', 'green']})
        assert len(c) == 2

    def test_by_key_from_func(self):
        c = cycler('color', ['r', 'g', 'b'])
        assert c.by_key()['color'] == ['r', 'g', 'b']


# ===================================================================
# Extended Cycler tests (upstream-inspired)
# ===================================================================

class TestCyclerExtended:
    def test_single_key_single_value(self):
        """Cycler with one key and one value has length 1."""
        c = Cycler('color', ['red'])
        assert len(c) == 1
        assert list(c) == [{'color': 'red'}]

    def test_multiple_keys_via_mul(self):
        """Multiplying two Cyclers gives Cartesian product."""
        c1 = Cycler('color', ['red', 'blue'])
        c2 = Cycler('lw', [1, 2])
        c = c1 * c2
        assert len(c) == 4
        items = list(c)
        assert {'color': 'red', 'lw': 1} in items
        assert {'color': 'blue', 'lw': 2} in items

    def test_keys_of_product(self):
        """Cycler product has both keys."""
        c1 = Cycler('color', ['red'])
        c2 = Cycler('lw', [1])
        c = c1 * c2
        assert 'color' in c.keys
        assert 'lw' in c.keys

    def test_mul_int_repeats(self):
        """Cycler * n repeats the cycle n times."""
        c = Cycler('color', ['red', 'blue'])
        c2 = c * 2
        assert len(c2) == 4
        items = list(c2)
        assert items[0] == {'color': 'red'}
        assert items[2] == {'color': 'red'}

    def test_add_concatenates(self):
        """Adding two Cyclers concatenates their values."""
        c1 = Cycler('color', ['red', 'green'])
        c2 = Cycler('color', ['blue'])
        c = c1 + c2
        assert len(c) == 3
        colors = [item['color'] for item in c]
        assert 'red' in colors
        assert 'green' in colors
        assert 'blue' in colors

    def test_by_key_multi(self):
        """by_key() on product cycler returns lists for each key."""
        c1 = Cycler('color', ['red', 'blue'])
        c2 = Cycler('lw', [1, 2])
        c = c1 * c2
        d = c.by_key()
        assert 'color' in d
        assert 'lw' in d

    def test_getitem_negative(self):
        """Cycler[-1] accesses last item."""
        c = Cycler('color', ['red', 'green', 'blue'])
        assert c[-1] == {'color': 'blue'}

    def test_getitem_slice(self):
        """Cycler slicing works."""
        c = Cycler('color', ['a', 'b', 'c', 'd'])
        items = c[1:3]
        assert items == [{'color': 'b'}, {'color': 'c'}]

    def test_empty_keys_set(self):
        """Empty Cycler has empty keys set."""
        c = Cycler()
        assert c.keys == set()

    def test_repr_contains_cycler(self):
        """Cycler repr contains 'cycler'."""
        c = Cycler('color', ['red', 'blue'])
        assert 'cycler' in repr(c)

    def test_iteration_exhausts(self):
        """Cycler iteration produces exactly len(c) items."""
        c = Cycler('x', [1, 2, 3, 4, 5])
        items = list(c)
        assert len(items) == 5

    def test_two_key_cycler_values(self):
        """Cycler with two keys via product has merged dicts."""
        c = Cycler('a', [1, 2]) * Cycler('b', [3, 4])
        for item in c:
            assert 'a' in item
            assert 'b' in item


class TestCyclerFunctionExtended:
    def test_single_kwarg(self):
        """cycler(color=...) creates single-key cycler."""
        c = cycler(color=['r', 'g', 'b'])
        assert len(c) == 3
        assert list(c)[0] == {'color': 'r'}

    def test_two_kwargs_same_length(self):
        """cycler(color=..., lw=...) creates product cycler."""
        c = cycler(color=['r', 'g'], lw=[1, 2])
        assert len(c) > 0

    def test_dict_multi_key(self):
        """cycler({'color': ..., 'lw': ...}) creates cycler."""
        c = cycler({'color': ['r', 'g'], 'lw': [1, 2]})
        assert len(c) > 0

    def test_returns_cycler_instance(self):
        """cycler() always returns a Cycler instance."""
        c = cycler('color', ['r'])
        assert isinstance(c, Cycler)

    def test_empty_call(self):
        """cycler() with no args returns empty Cycler."""
        c = cycler()
        assert len(c) == 0

    def test_string_values(self):
        """cycler with string values iterates correctly."""
        c = cycler('linestyle', ['-', '--', '-.'])
        styles = [item['linestyle'] for item in c]
        assert styles == ['-', '--', '-.']

    def test_numeric_values(self):
        """cycler with numeric values iterates correctly."""
        c = cycler('linewidth', [0.5, 1.0, 2.0])
        widths = [item['linewidth'] for item in c]
        assert widths == [0.5, 1.0, 2.0]
