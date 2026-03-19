"""
Upstream matplotlib tests for the Artist base class.
"""

import pytest

from matplotlib.artist import Artist


def test_artist_default_visible():
    """Artist is visible by default."""
    a = Artist()
    assert a.get_visible() is True


def test_artist_set_visible():
    """set_visible changes visibility."""
    a = Artist()
    a.set_visible(False)
    assert a.get_visible() is False
    a.set_visible(True)
    assert a.get_visible() is True


def test_artist_default_alpha():
    """Artist alpha is None by default."""
    a = Artist()
    assert a.get_alpha() is None


def test_artist_set_alpha():
    """set_alpha changes alpha."""
    a = Artist()
    a.set_alpha(0.5)
    assert a.get_alpha() == 0.5


def test_artist_default_label():
    """Artist label is empty by default."""
    a = Artist()
    assert a.get_label() == ''


def test_artist_set_label():
    """set_label changes label."""
    a = Artist()
    a.set_label('test')
    assert a.get_label() == 'test'


def test_artist_set_label_none():
    """set_label(None) sets to '_nolegend_'."""
    a = Artist()
    a.set_label(None)
    assert a.get_label() == '_nolegend_'


def test_artist_default_zorder():
    """Artist default zorder is 0."""
    a = Artist()
    assert a.get_zorder() == 0


def test_artist_set_zorder():
    """set_zorder changes zorder."""
    a = Artist()
    a.set_zorder(5)
    assert a.get_zorder() == 5


def test_artist_stale():
    """Artist starts stale."""
    a = Artist()
    assert a._stale is True


def test_artist_figure_none():
    """Artist figure is None by default."""
    a = Artist()
    assert a.figure is None


def test_artist_axes_none():
    """Artist axes is None by default."""
    a = Artist()
    assert a.axes is None


def test_artist_set_batch():
    """Artist.set() batch setter."""
    a = Artist()
    a.set(visible=False, alpha=0.3, label='batch')
    assert a.get_visible() is False
    assert a.get_alpha() == 0.3
    assert a.get_label() == 'batch'


def test_artist_remove_from_axes():
    """Artist.remove removes from axes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    ax.add_artist(r)
    assert r in ax.patches
    r.remove()
    assert r not in ax.patches
    assert r.axes is None
    assert r.figure is None
