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
