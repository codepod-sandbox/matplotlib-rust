"""
Upstream matplotlib tests imported from lib/matplotlib/tests/test_axes.py.

These tests are copied or minimally adapted from the real matplotlib test
suite to validate compatibility of our Axes implementation.
"""

import numpy as np
import pytest

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. test_get_labels (upstream ~line 4200)
# ---------------------------------------------------------------------------
def test_get_labels():
    fig, ax = plt.subplots()
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    assert ax.get_xlabel() == 'x label'
    assert ax.get_ylabel() == 'y label'


# ---------------------------------------------------------------------------
# 2. test_inverted_limits (upstream ~line 2260)  -- first two stanzas
# ---------------------------------------------------------------------------
def test_inverted_limits():
    # Invert x-axis, then plot: x-limits should be reversed
    fig, ax = plt.subplots()
    ax.invert_xaxis()
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])
    assert ax.get_xlim() == (4, -5)
    assert ax.get_ylim() == (-3, 5)

    # Invert y-axis, then plot: y-limits should be reversed
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])
    assert ax.get_xlim() == (-5, 4)
    assert ax.get_ylim() == (5, -3)


# ---------------------------------------------------------------------------
# 3. test_fill_between_input (upstream ~line 5700)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'x, y1, y2', [
        (np.zeros((2, 2)), 3, 3),
        (np.arange(0.0, 2, 0.02), np.zeros((2, 2)), 3),
        (np.arange(0.0, 2, 0.02), 3, np.zeros((2, 2))),
    ], ids=['2d_x_input', '2d_y1_input', '2d_y2_input']
)
def test_fill_between_input(x, y1, y2):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.fill_between(x, y1, y2)


# ---------------------------------------------------------------------------
# 4. test_fill_betweenx_input (upstream ~line 5720)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'y, x1, x2', [
        (np.zeros((2, 2)), 3, 3),
        (np.arange(0.0, 2, 0.02), np.zeros((2, 2)), 3),
        (np.arange(0.0, 2, 0.02), 3, np.zeros((2, 2))),
    ], ids=['2d_y_input', '2d_x1_input', '2d_x2_input']
)
def test_fill_betweenx_input(y, x1, x2):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.fill_betweenx(y, x1, x2)


# ---------------------------------------------------------------------------
# 5. test_bar_ticklabel_fail (upstream ~line 3040)  -- smoke test
# ---------------------------------------------------------------------------
def test_bar_ticklabel_fail():
    fig, ax = plt.subplots()
    ax.bar([], [])


# ---------------------------------------------------------------------------
# 6. test_bar_color_none_alpha (upstream ~line 3120)
# ---------------------------------------------------------------------------
def test_bar_color_none_alpha():
    fig, ax = plt.subplots()
    rects = ax.bar([1, 2], [2, 4], alpha=0.3, color='none', edgecolor='r')
    for rect in rects:
        assert rect.get_facecolor() == (0, 0, 0, 0)
        assert rect.get_edgecolor() == (1, 0, 0, 0.3)


# ---------------------------------------------------------------------------
# 7. test_bar_edgecolor_none_alpha (upstream ~line 3135)
# ---------------------------------------------------------------------------
def test_bar_edgecolor_none_alpha():
    fig, ax = plt.subplots()
    rects = ax.bar([1, 2], [2, 4], alpha=0.3, color='r', edgecolor='none')
    for rect in rects:
        assert rect.get_facecolor() == (1, 0, 0, 0.3)
        assert rect.get_edgecolor() == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# 8. test_nan_bar_values (upstream ~line 3020)  -- smoke test
# ---------------------------------------------------------------------------
def test_nan_bar_values():
    fig, ax = plt.subplots()
    ax.bar([0, 1], [np.nan, 4])


# ---------------------------------------------------------------------------
# 9. test_scatter_empty_data (upstream ~line 4650)
# ---------------------------------------------------------------------------
def test_scatter_empty_data():
    fig, ax = plt.subplots()
    ax.scatter([], [])


# ---------------------------------------------------------------------------
# 10. test_annotate_default_arrow (upstream ~line 4400)
# ---------------------------------------------------------------------------
def test_annotate_default_arrow():
    fig, ax = plt.subplots()
    ann = ax.annotate("foo", (0, 1), xytext=(2, 3))
    assert ann.arrow_patch is None
    ann = ax.annotate("foo", (0, 1), xytext=(2, 3), arrowprops={})
    assert ann.arrow_patch is not None


# ---------------------------------------------------------------------------
# 11. test_color_None (upstream ~line 7600)
# ---------------------------------------------------------------------------
def test_color_None():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], color=None)


# ---------------------------------------------------------------------------
# 12. test_zero_linewidth (upstream ~line 7610)
# ---------------------------------------------------------------------------
def test_zero_linewidth():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], ls='--', lw=0)


# ---------------------------------------------------------------------------
# 13. test_empty_line_plots (upstream ~line 7580) -- second stanza only
# ---------------------------------------------------------------------------
def test_empty_line_plots():
    fig, ax = plt.subplots()
    line = ax.plot([], [])
    assert len(line) == 1


# ---------------------------------------------------------------------------
# 14. test_errorbar_nonefmt (upstream ~line 3700)
# ---------------------------------------------------------------------------
def test_errorbar_nonefmt():
    x = list(range(5))
    y = list(range(5))
    fig, ax = plt.subplots()
    ec = ax.errorbar(x, y, xerr=1, yerr=1, fmt='none')
    plotline, _, barlines = ec.lines
    assert plotline is None
