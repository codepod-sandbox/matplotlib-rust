import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axis import XTick


def test_tick_labelcolor_array():
    ax = plt.axes()
    XTick(ax, 0, labelcolor=np.array([1, 0, 0, 1]))


def test_axis_not_in_layout():
    fig1, (ax1_left, ax1_right) = plt.subplots(ncols=2, layout="constrained")
    fig2, (ax2_left, ax2_right) = plt.subplots(ncols=2, layout="constrained")

    ax1_left.set_xlim([0, 100])
    ax2_left.set_xlim([0, 120])

    for ax in ax1_left, ax2_left:
        ax.set_xticks([0, 100])
        ax.xaxis.set_in_layout(False)

    for fig in fig1, fig2:
        fig.draw_without_rendering()

    assert ax1_left.get_position().bounds == ax2_left.get_position().bounds
    assert ax1_right.get_position().bounds == ax2_right.get_position().bounds


def test_translate_tick_params_reverse():
    fig, ax = plt.subplots()
    kw = {"label1On": "a", "label2On": "b", "tick1On": "c", "tick2On": "d"}
    assert ax.xaxis._translate_tick_params(kw, reverse=True) == {
        "labelbottom": "a",
        "labeltop": "b",
        "bottom": "c",
        "top": "d",
    }
    assert ax.yaxis._translate_tick_params(kw, reverse=True) == {
        "labelleft": "a",
        "labelright": "b",
        "left": "c",
        "right": "d",
    }
