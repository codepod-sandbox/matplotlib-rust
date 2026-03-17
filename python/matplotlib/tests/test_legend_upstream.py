# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_legend.py
import pytest
import matplotlib.pyplot as plt


def test_legend_auto_labels():
    """ax.legend() picks up handles/labels from plotted artists."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='line1')
    ax.plot([1, 2], [2, 1], label='line2')
    leg = ax.legend()
    assert leg is not None
    texts = leg.get_texts()
    assert len(texts) == 2
    assert texts[0].get_text() == 'line1'
    assert texts[1].get_text() == 'line2'
    plt.close('all')


def test_legend_no_handles_labels():
    """ax.legend() with nothing labeled returns empty legend (no error)."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])  # no label
    leg = ax.legend()
    assert len(leg.get_texts()) == 0
    plt.close('all')


def test_legend_explicit_labels():
    """ax.legend(['a', 'b']) uses explicit labels."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    ax.plot([1, 2], [2, 1])
    leg = ax.legend(['a', 'b'])
    texts = leg.get_texts()
    assert texts[0].get_text() == 'a'
    assert texts[1].get_text() == 'b'
    plt.close('all')


def test_legend_loc():
    """ax.legend(loc=...) stores the location."""
    fig, ax = plt.subplots()
    ax.plot([1], [1], label='x')
    leg = ax.legend(loc='lower left')
    assert leg.get_loc() == 'lower left'
    plt.close('all')


def test_legend_ncol():
    """ax.legend(ncol=2) stores column count."""
    fig, ax = plt.subplots()
    for i in range(4):
        ax.plot([1], [i], label=f'line{i}')
    leg = ax.legend(ncol=2)
    assert leg.get_ncol() == 2
    plt.close('all')


def test_legend_title():
    """ax.legend(title='T') stores the title."""
    fig, ax = plt.subplots()
    ax.plot([1], [1], label='x')
    leg = ax.legend(title='Legend Title')
    assert leg.get_title().get_text() == 'Legend Title'
    plt.close('all')


def test_legend_renders_without_error():
    """Calling fig.savefig (or to_svg) with legend must not raise."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='x')
    ax.legend()
    svg = fig.to_svg()
    assert len(svg) > 0
    plt.close('all')
