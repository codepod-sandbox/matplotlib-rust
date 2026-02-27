"""
Upstream matplotlib tests imported from lib/matplotlib/tests/test_figure.py.

These tests are copied (or minimally adapted) from the real matplotlib test
suite to validate compatibility of our Figure implementation.
"""

import matplotlib.pyplot as plt


# ===================================================================
# Figure sizing (1 test — direct import)
# ===================================================================

def test_set_fig_size():
    fig = plt.figure()

    # check figwidth
    fig.set_figwidth(5)
    assert fig.get_figwidth() == 5

    # check figheight
    fig.set_figheight(1)
    assert fig.get_figheight() == 1

    # check using set_size_inches
    fig.set_size_inches(2, 4)
    assert fig.get_figwidth() == 2
    assert fig.get_figheight() == 4

    # check using tuple to first argument
    fig.set_size_inches((1, 3))
    assert fig.get_figwidth() == 1
    assert fig.get_figheight() == 3


# ===================================================================
# Figure repr (1 test — direct import)
# ===================================================================

def test_figure_repr():
    fig = plt.figure(figsize=(10, 20), dpi=10)
    assert repr(fig) == "<Figure size 100x200 with 0 Axes>"
