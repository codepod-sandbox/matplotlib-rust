"""
Upstream matplotlib tests for collections module.
"""

import numpy as np
import pytest

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection


# ---------------------------------------------------------------------------
# PathCollection from scatter
# ---------------------------------------------------------------------------
def test_scatter_returns_pathcollection():
    """scatter() returns a PathCollection."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2, 3], [4, 5, 6])
    assert isinstance(pc, PathCollection)


def test_scatter_offsets():
    """scatter offsets match input data."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4])
    offsets = pc.get_offsets()
    assert len(offsets) == 2
    assert offsets[0] == (1, 3)
    assert offsets[1] == (2, 4)


def test_scatter_sizes():
    """scatter sizes are stored."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], s=50)
    sizes = pc.get_sizes()
    assert sizes == [50]


def test_scatter_sizes_array():
    """scatter with per-point sizes."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], s=[10, 20])
    sizes = pc.get_sizes()
    assert sizes == [10, 20]


def test_scatter_color():
    """scatter facecolors are stored."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], c='red')
    fc = pc.get_facecolors()
    assert len(fc) > 0


def test_scatter_label():
    """scatter label is stored."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], label='points')
    assert pc.get_label() == 'points'


def test_scatter_in_collections():
    """scatter result is in ax.collections."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4])
    assert pc in ax.collections


def test_scatter_string_s_raises():
    """scatter with string s raises ValueError."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.scatter([1, 2], [3, 4], s='large')


def test_scatter_mismatched_s_raises():
    """scatter with wrong-length s raises ValueError."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.scatter([1, 2], [3, 4], s=[10, 20, 30])


# ---------------------------------------------------------------------------
# PathCollection set/get
# ---------------------------------------------------------------------------
def test_pathcollection_set_offsets():
    """PathCollection.set_offsets changes offsets."""
    pc = PathCollection(offsets=[(0, 0)])
    pc.set_offsets([(1, 1), (2, 2)])
    assert len(pc.get_offsets()) == 2


def test_pathcollection_set_sizes():
    """PathCollection.set_sizes changes sizes."""
    pc = PathCollection(sizes=[10])
    pc.set_sizes([20, 30])
    assert pc.get_sizes() == [20, 30]


def test_pathcollection_set_facecolors():
    """PathCollection.set_facecolors changes colors."""
    pc = PathCollection(facecolors=['red'])
    pc.set_facecolors(['blue', 'green'])
    assert pc.get_facecolors() == ['blue', 'green']


def test_pathcollection_set_edgecolors():
    """PathCollection.set_edgecolors changes colors."""
    pc = PathCollection(edgecolors=['red'])
    pc.set_edgecolors(['blue'])
    assert pc.get_edgecolors() == ['blue']


def test_pathcollection_visible():
    """PathCollection visibility."""
    pc = PathCollection()
    assert pc.get_visible() is True
    pc.set_visible(False)
    assert pc.get_visible() is False


def test_pathcollection_alpha():
    """PathCollection alpha."""
    pc = PathCollection()
    assert pc.get_alpha() is None
    pc.set_alpha(0.5)
    assert pc.get_alpha() == 0.5


def test_pathcollection_label():
    """PathCollection label."""
    pc = PathCollection(label='test')
    assert pc.get_label() == 'test'


def test_pathcollection_zorder():
    """PathCollection default zorder is 1."""
    pc = PathCollection()
    assert pc.get_zorder() == 1


def test_pathcollection_remove():
    """PathCollection.remove removes from axes."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4])
    assert pc in ax.collections
    pc.remove()
    assert pc not in ax.collections


# ---------------------------------------------------------------------------
# PathCollection empty
# ---------------------------------------------------------------------------
def test_pathcollection_empty():
    """PathCollection with no offsets."""
    pc = PathCollection()
    assert pc.get_offsets() == []


def test_pathcollection_default_sizes():
    """PathCollection default sizes is [20.0]."""
    pc = PathCollection()
    assert pc.get_sizes() == [20.0]
