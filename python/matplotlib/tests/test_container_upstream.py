"""
Upstream matplotlib tests for container classes.
"""

import pytest

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer


# ---------------------------------------------------------------------------
# BarContainer
# ---------------------------------------------------------------------------
def test_barcontainer_len():
    """BarContainer has correct length."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    assert len(bc) == 3


def test_barcontainer_iter():
    """BarContainer is iterable."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    patches = list(bc)
    assert len(patches) == 3


def test_barcontainer_getitem():
    """BarContainer supports indexing."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    p = bc[0]
    assert p.get_height() == 4


def test_barcontainer_patches():
    """BarContainer.patches property."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2, 3], [4, 5, 6])
    assert len(bc.patches) == 3


def test_barcontainer_label():
    """BarContainer label."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2], [3, 4], label='bars')
    assert bc.get_label() == 'bars'


def test_barcontainer_no_label():
    """BarContainer without label."""
    fig, ax = plt.subplots()
    bc = ax.bar([1], [2])
    # Should have some default label (None becomes '_nolegend_')
    label = bc.get_label()
    assert isinstance(label, str)


# ---------------------------------------------------------------------------
# ErrorbarContainer
# ---------------------------------------------------------------------------
def test_errorbar_container_has_yerr():
    """ErrorbarContainer.has_yerr is True when yerr given."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5])
    assert ec.has_yerr is True
    assert ec.has_xerr is False


def test_errorbar_container_has_xerr():
    """ErrorbarContainer.has_xerr is True when xerr given."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], xerr=[0.5, 0.5])
    assert ec.has_xerr is True
    assert ec.has_yerr is False


def test_errorbar_container_both():
    """ErrorbarContainer with both xerr and yerr."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], xerr=[0.1, 0.1], yerr=[0.2, 0.2])
    assert ec.has_xerr is True
    assert ec.has_yerr is True


def test_errorbar_container_label():
    """ErrorbarContainer label."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5], label='errors')
    assert ec.get_label() == 'errors'


def test_errorbar_container_lines():
    """ErrorbarContainer.lines holds the plot line."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5])
    assert ec.lines is not None
    # lines[0] is the data line
    assert ec.lines[0] is not None


def test_errorbar_fmt_none():
    """ErrorbarContainer with fmt='none' hides data line."""
    fig, ax = plt.subplots()
    ec = ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5], fmt='none')
    assert ec.lines[0] is None


# ---------------------------------------------------------------------------
# StemContainer
# ---------------------------------------------------------------------------
def test_stem_container_markerline():
    """StemContainer.markerline exists."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2, 3], [4, 5, 6])
    assert sc.markerline is not None


def test_stem_container_baseline():
    """StemContainer.baseline exists."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2, 3], [4, 5, 6])
    assert sc.baseline is not None


def test_stem_container_stemlines():
    """StemContainer.stemlines exists and has correct count."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2, 3], [4, 5, 6])
    assert len(sc.stemlines) == 3


def test_stem_container_label():
    """StemContainer label."""
    fig, ax = plt.subplots()
    sc = ax.stem([1, 2], [3, 4], label='stems')
    assert sc.get_label() == 'stems'


# ---------------------------------------------------------------------------
# Container.remove
# ---------------------------------------------------------------------------
def test_barcontainer_remove():
    """BarContainer.remove removes patches from axes."""
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2], [3, 4])
    initial_patches = len(ax.patches)
    bc.remove()
    assert len(ax.patches) < initial_patches
