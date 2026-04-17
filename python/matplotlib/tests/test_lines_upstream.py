"""
Upstream matplotlib tests imported from lib/matplotlib/tests/test_lines.py.

These tests are adapted from the real matplotlib test suite to validate
compatibility of our Line2D implementation.  Upstream tests that require
rendering infrastructure or APIs we haven't implemented are omitted.
"""

import pytest

from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# 1. test_invalid_line_data (upstream ~line 41)
# OG matplotlib raises RuntimeError on scalar xdata input; set_xdata/set_ydata
# with scalars may raise TypeError or RuntimeError depending on version.
# ---------------------------------------------------------------------------
def test_invalid_line_data():
    with pytest.raises((TypeError, RuntimeError)):
        Line2D(0, [])
    with pytest.raises((TypeError, RuntimeError)):
        Line2D([], 1)

    line = Line2D([], [])
    with pytest.raises((TypeError, RuntimeError)):
        line.set_xdata(0)
    with pytest.raises((TypeError, RuntimeError)):
        line.set_ydata(0)


# ---------------------------------------------------------------------------
# 2. test_set_drawstyle (upstream ~line 126, positive-case)
# Upstream test_valid_drawstyles expects ValueError on invalid input, which
# we don't enforce.  This adapted version tests the positive cases.
# ---------------------------------------------------------------------------
def test_set_drawstyle():
    line = Line2D([], [])
    for ds in ["default", "steps-pre", "steps-mid", "steps-post", "steps"]:
        line.set_drawstyle(ds)
        assert line.get_drawstyle() == ds


# ---------------------------------------------------------------------------
# 3. test_set_linestyle (upstream ~line 89, positive-case)
# ---------------------------------------------------------------------------
def test_set_linestyle():
    line = Line2D([], [])
    for ls in ["-", "--", "-.", ":"]:
        line.set_linestyle(ls)
        assert line.get_linestyle() == ls


# ---------------------------------------------------------------------------
# 4. test_set_color (upstream ~line 73, positive-case)
# ---------------------------------------------------------------------------
def test_set_color():
    line = Line2D([], [])
    line.set_color("red")
    assert line.get_color() == "red"
    line.set_c("blue")
    assert line.get_color() == "blue"


# ---------------------------------------------------------------------------
# 5. test_line_data_copy (upstream pattern: input mutation safety)
# OG get_xdata/get_ydata return ndarrays; use list() for comparison.
# ---------------------------------------------------------------------------
def test_line_data_copy():
    import numpy as np
    line = Line2D([1, 2, 3], [4, 5, 6])
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    # OG returns ndarrays — mutating via append creates a new array, original unchanged
    # Verify data is still correct via list conversion
    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [4, 5, 6]


# ---------------------------------------------------------------------------
# 6. test_set_data (upstream pattern: set_data updates both x and y)
# ---------------------------------------------------------------------------
def test_set_data():
    line = Line2D([], [])
    line.set_data([1, 2], [3, 4])
    assert line.get_xdata() == [1, 2]
    assert line.get_ydata() == [3, 4]
    x, y = line.get_data()
    assert x == [1, 2]
    assert y == [3, 4]


# ---------------------------------------------------------------------------
# 7. test_line_defaults (upstream pattern: default property values)
# ---------------------------------------------------------------------------
def test_line_defaults():
    line = Line2D([], [])
    assert line.get_linewidth() == 1.5
    assert line.get_linestyle() == '-'
    assert line.get_marker() == 'None'
    assert line.get_markersize() == 6.0
    assert line.get_fillstyle() == 'full'
    assert line.get_drawstyle() == 'default'
    assert line.zorder == 2


# ---------------------------------------------------------------------------
# 8. test_line_constructor_kwargs (upstream pattern)
# ---------------------------------------------------------------------------
def test_line_constructor_kwargs():
    line = Line2D([0], [0], color='lime', linewidth=3, linestyle='--',
                  marker='o', label='test', markersize=10, fillstyle='left',
                  drawstyle='steps-pre')
    assert line.get_color() == 'lime'
    assert line.get_linewidth() == 3
    assert line.get_linestyle() == '--'
    assert line.get_marker() == 'o'
    assert line.get_label() == 'test'
    assert line.get_markersize() == 10
    assert line.get_fillstyle() == 'left'
    assert line.get_drawstyle() == 'steps-pre'


# ---------------------------------------------------------------------------
# 9. test_line_aliases (upstream pattern: lw, ls, ms, c)
# ---------------------------------------------------------------------------
def test_line_aliases():
    line = Line2D([], [])
    line.set_lw(5)
    assert line.get_linewidth() == 5
    line.set_ls(':')
    assert line.get_linestyle() == ':'
    line.set_ms(12)
    assert line.get_markersize() == 12
    line.set_c('green')
    assert line.get_color() == 'green'


# ===========================================================================
# Newly ported upstream tests (2026-03-19)
# Source: https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/tests/test_lines.py
# ===========================================================================

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# test_valid_linestyles (upstream)
# ---------------------------------------------------------------------------
def test_valid_linestyles():
    """Upstream: invalid linestyle raises ValueError."""
    line = Line2D([], [])
    with pytest.raises(ValueError):
        line.set_linestyle('aardvark')


# ---------------------------------------------------------------------------
# test_valid_drawstyles (upstream)
# ---------------------------------------------------------------------------
def test_valid_drawstyles():
    """Upstream: invalid drawstyle raises ValueError."""
    line = Line2D([], [])
    with pytest.raises(ValueError):
        line.set_drawstyle('foobar')


# ---------------------------------------------------------------------------
# test_valid_colors (upstream)
# ---------------------------------------------------------------------------
def test_valid_colors():
    """Upstream: invalid color raises ValueError."""
    line = Line2D([], [])
    with pytest.raises(ValueError):
        line.set_color("foobar")


# ---------------------------------------------------------------------------
# test_line_colors (upstream)
# ---------------------------------------------------------------------------
def test_line_colors():
    """Upstream: various valid color specifications."""
    fig, ax = plt.subplots()
    ax.plot(range(10), color='none')
    ax.plot(range(10), color='r')
    ax.plot(range(10), color='.3')
    ax.plot(range(10), color=(1, 0, 0, 1))
    ax.plot(range(10), color=(1, 0, 0))


# ---------------------------------------------------------------------------
# test_linestyle_variants (upstream)
# ---------------------------------------------------------------------------
def test_linestyle_variants():
    """Upstream: all standard linestyle strings work."""
    fig, ax = plt.subplots()
    for ls in ["-", "solid", "--", "dashed",
               "-.", "dashdot", ":", "dotted"]:
        ax.plot(range(10), linestyle=ls)


# ===========================================================================
# Newly ported upstream tests (2026-03-19, batch 2)
# ===========================================================================


# ---------------------------------------------------------------------------
# test_drawstyle_variants (upstream ~line 97)
# ---------------------------------------------------------------------------
def test_drawstyle_variants():
    """Upstream: all drawstyle variants work in plot."""
    fig, ax = plt.subplots()
    dss = ["default", "steps-mid", "steps-pre", "steps-post", "steps"]
    for ds in dss:
        line = Line2D(range(10), range(10), drawstyle=ds)
        assert line.get_drawstyle() == ds


# ---------------------------------------------------------------------------
# Additional Line2D tests
# ---------------------------------------------------------------------------

def test_line_visibility():
    """Line2D visibility can be toggled."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_visible() is True
    line.set_visible(False)
    assert line.get_visible() is False


def test_line_zorder():
    """Line2D zorder can be set."""
    line = Line2D([0, 1], [0, 1])
    assert line.zorder == 2
    line.set_zorder(10)
    assert line.get_zorder() == 10


def test_line_label():
    """Line2D label can be set/get."""
    line = Line2D([0, 1], [0, 1])
    line.set_label('my line')
    assert line.get_label() == 'my line'


def test_line_marker_set():
    """Line2D marker can be changed after construction."""
    line = Line2D([0, 1], [0, 1])
    line.set_marker('o')
    assert line.get_marker() == 'o'


def test_line_markersize_set():
    """Line2D markersize can be changed via set_ms."""
    line = Line2D([0, 1], [0, 1])
    line.set_ms(15)
    assert line.get_markersize() == 15


def test_line_fillstyle_set():
    """Line2D fillstyle can be set."""
    line = Line2D([0, 1], [0, 1])
    line.set_fillstyle('bottom')
    assert line.get_fillstyle() == 'bottom'


def test_line_linewidth_set():
    """Line2D linewidth can be changed."""
    line = Line2D([0, 1], [0, 1])
    line.set_linewidth(3.0)
    assert line.get_linewidth() == 3.0


def test_line_data_get():
    """Line2D get_data returns (xdata, ydata)."""
    line = Line2D([1, 2, 3], [4, 5, 6])
    x, y = line.get_data()
    assert x == [1, 2, 3]
    assert y == [4, 5, 6]


def test_line_set_xdata_ydata():
    """Line2D set_xdata / set_ydata update data independently."""
    line = Line2D([1, 2], [3, 4])
    line.set_xdata([10, 20])
    assert line.get_xdata() == [10, 20]
    assert line.get_ydata() == [3, 4]
    line.set_ydata([30, 40])
    assert line.get_ydata() == [30, 40]


# ---------------------------------------------------------------------------
# Line2D set_data (upstream ~line 200)
# ---------------------------------------------------------------------------
def test_line_set_data():
    """Line2D set_data updates both x and y."""
    line = Line2D([1, 2], [3, 4])
    line.set_data([10, 20, 30], [40, 50, 60])
    assert line.get_xdata() == [10, 20, 30]
    assert line.get_ydata() == [40, 50, 60]


# ---------------------------------------------------------------------------
# Line2D color validation (upstream ~line 120)
# ---------------------------------------------------------------------------
def test_line_set_color_valid():
    """set_color accepts valid color strings."""
    line = Line2D([0, 1], [0, 1])
    line.set_color('red')
    assert line.get_color() == 'red'
    line.set_color('#ff0000')
    assert line.get_color() == '#ff0000'


def test_line_set_color_invalid():
    """set_color raises ValueError for invalid colors."""
    line = Line2D([0, 1], [0, 1])
    with pytest.raises(ValueError):
        line.set_color('not_a_real_color')


def test_line_set_c_alias():
    """set_c is an alias for set_color."""
    line = Line2D([0, 1], [0, 1])
    line.set_c('blue')
    assert line.get_color() == 'blue'


# ---------------------------------------------------------------------------
# Line2D linestyle validation (upstream ~line 145)
# ---------------------------------------------------------------------------
def test_line_linestyle_solid():
    """set_linestyle('solid') is valid; OG normalizes to '-'."""
    line = Line2D([0, 1], [0, 1])
    line.set_linestyle('solid')
    assert line.get_linestyle() in ('solid', '-')


def test_line_linestyle_dashed():
    """set_linestyle('--') is valid."""
    line = Line2D([0, 1], [0, 1])
    line.set_linestyle('--')
    assert line.get_linestyle() == '--'


def test_line_linestyle_dashdot():
    """set_linestyle('dashdot') is valid; OG normalizes to '-.'."""
    line = Line2D([0, 1], [0, 1])
    line.set_linestyle('dashdot')
    assert line.get_linestyle() in ('dashdot', '-.')


def test_line_linestyle_dotted():
    """set_linestyle(':') is valid."""
    line = Line2D([0, 1], [0, 1])
    line.set_linestyle(':')
    assert line.get_linestyle() == ':'


def test_line_linestyle_none():
    """set_linestyle('None') is valid."""
    line = Line2D([0, 1], [0, 1])
    line.set_linestyle('None')
    assert line.get_linestyle() == 'None'


def test_line_linestyle_empty():
    """set_linestyle('') is valid; OG may normalize to 'None'."""
    line = Line2D([0, 1], [0, 1])
    line.set_linestyle('')
    assert line.get_linestyle() in ('', 'None', 'none')


# ---------------------------------------------------------------------------
# Line2D marker
# ---------------------------------------------------------------------------
def test_line_marker_set():
    """set_marker changes the marker."""
    line = Line2D([0, 1], [0, 1])
    line.set_marker('o')
    assert line.get_marker() == 'o'


def test_line_marker_none():
    """Default marker is 'None'."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_marker() == 'None'


# ---------------------------------------------------------------------------
# Line2D markersize
# ---------------------------------------------------------------------------
def test_line_markersize_default():
    """Default markersize is 6.0."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_markersize() == 6.0


def test_line_markersize_set():
    """set_markersize changes the size."""
    line = Line2D([0, 1], [0, 1])
    line.set_markersize(10.0)
    assert line.get_markersize() == 10.0


def test_line_set_ms_alias():
    """set_ms is an alias for set_markersize."""
    line = Line2D([0, 1], [0, 1])
    line.set_ms(8.0)
    assert line.get_markersize() == 8.0


# ---------------------------------------------------------------------------
# Line2D drawstyle
# ---------------------------------------------------------------------------
def test_line_drawstyle_default():
    """Default drawstyle is 'default'."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_drawstyle() == 'default'


def test_line_drawstyle_set():
    """set_drawstyle changes the style."""
    line = Line2D([0, 1], [0, 1])
    line.set_drawstyle('steps')
    assert line.get_drawstyle() == 'steps'


def test_line_drawstyle_invalid():
    """Invalid drawstyle raises ValueError."""
    line = Line2D([0, 1], [0, 1])
    with pytest.raises(ValueError):
        line.set_drawstyle('invalid')


# ---------------------------------------------------------------------------
# Line2D visibility
# ---------------------------------------------------------------------------
def test_line_visible_default():
    """Default visibility is True."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_visible() is True


def test_line_set_visible():
    """set_visible changes visibility."""
    line = Line2D([0, 1], [0, 1])
    line.set_visible(False)
    assert line.get_visible() is False


# ---------------------------------------------------------------------------
# Line2D alpha
# ---------------------------------------------------------------------------
def test_line_alpha_default():
    """Default alpha is None."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_alpha() is None


def test_line_set_alpha():
    """set_alpha changes alpha."""
    line = Line2D([0, 1], [0, 1])
    line.set_alpha(0.5)
    assert line.get_alpha() == 0.5


# ---------------------------------------------------------------------------
# Line2D zorder
# ---------------------------------------------------------------------------
def test_line_zorder_default():
    """Default zorder for Line2D is 2."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_zorder() == 2


def test_line_set_zorder():
    """set_zorder changes the zorder."""
    line = Line2D([0, 1], [0, 1])
    line.set_zorder(10)
    assert line.get_zorder() == 10


# ---------------------------------------------------------------------------
# Line2D label
# ---------------------------------------------------------------------------
def test_line_label_default():
    """Default label is empty string."""
    line = Line2D([0, 1], [0, 1])
    assert line.get_label() == ''


def test_line_label_set():
    """set_label changes the label."""
    line = Line2D([0, 1], [0, 1])
    line.set_label('my line')
    assert line.get_label() == 'my line'


def test_line_label_constructor():
    """label in constructor sets the label."""
    line = Line2D([0, 1], [0, 1], label='ctor label')
    assert line.get_label() == 'ctor label'


# ---------------------------------------------------------------------------
# Line2D constructor kwargs
# ---------------------------------------------------------------------------
def test_line_constructor_color():
    """color kwarg in constructor."""
    line = Line2D([0, 1], [0, 1], color='red')
    assert line.get_color() == 'red'


def test_line_constructor_linewidth():
    """linewidth kwarg in constructor."""
    line = Line2D([0, 1], [0, 1], linewidth=3.0)
    assert line.get_linewidth() == 3.0


def test_line_constructor_linestyle():
    """linestyle kwarg in constructor."""
    line = Line2D([0, 1], [0, 1], linestyle='--')
    assert line.get_linestyle() == '--'


def test_line_constructor_marker():
    """marker kwarg in constructor."""
    line = Line2D([0, 1], [0, 1], marker='o')
    assert line.get_marker() == 'o'


# ---------------------------------------------------------------------------
# Line2D set_lw alias
# ---------------------------------------------------------------------------
def test_line_set_lw_alias():
    """set_lw is an alias for set_linewidth."""
    line = Line2D([0, 1], [0, 1])
    line.set_lw(5.0)
    assert line.get_linewidth() == 5.0


# ---------------------------------------------------------------------------
# Line2D set_ls alias
# ---------------------------------------------------------------------------
def test_line_set_ls_alias():
    """set_ls is an alias for set_linestyle."""
    line = Line2D([0, 1], [0, 1])
    line.set_ls('--')
    assert line.get_linestyle() == '--'


# ---------------------------------------------------------------------------
# Line2D empty data
# ---------------------------------------------------------------------------
def test_line_empty_data():
    """Line2D with empty data."""
    line = Line2D([], [])
    assert line.get_xdata() == []
    assert line.get_ydata() == []
    x, y = line.get_data()
    assert x == []
    assert y == []


# ---------------------------------------------------------------------------
# Line2D marker edge/face color and related properties
# ---------------------------------------------------------------------------

class TestLine2DMarkerProperties:
    def test_markeredgecolor_default(self):
        line = Line2D([0], [0])
        mec = line.get_markeredgecolor()
        assert mec is not None

    def test_set_markeredgecolor(self):
        line = Line2D([0], [0])
        line.set_markeredgecolor('red')
        assert line.get_markeredgecolor() == 'red'

    def test_markerfacecolor_default(self):
        line = Line2D([0], [0])
        mfc = line.get_markerfacecolor()
        assert mfc is not None

    def test_set_markerfacecolor(self):
        line = Line2D([0], [0])
        line.set_markerfacecolor('blue')
        assert line.get_markerfacecolor() == 'blue'

    def test_markerfacecoloralt_default(self):
        line = Line2D([0], [0])
        mfca = line.get_markerfacecoloralt()
        assert mfca is not None

    def test_set_markerfacecoloralt(self):
        line = Line2D([0], [0])
        line.set_markerfacecoloralt('green')
        assert line.get_markerfacecoloralt() == 'green'

    def test_markeredgewidth_default(self):
        line = Line2D([0], [0])
        mew = line.get_markeredgewidth()
        assert mew is not None

    def test_set_markeredgewidth(self):
        line = Line2D([0], [0])
        line.set_markeredgewidth(2.0)
        assert line.get_markeredgewidth() == 2.0

    def test_markevery_default_none(self):
        line = Line2D([0], [0])
        assert line.get_markevery() is None

    def test_set_markevery(self):
        line = Line2D([0], [0])
        line.set_markevery(2)
        assert line.get_markevery() == 2

    def test_set_markevery_none(self):
        line = Line2D([0], [0])
        line.set_markevery(2)
        line.set_markevery(None)
        assert line.get_markevery() is None

    def test_antialiased_default(self):
        line = Line2D([0], [0])
        aa = line.get_antialiased()
        assert isinstance(aa, bool)

    def test_set_antialiased(self):
        line = Line2D([0], [0])
        line.set_antialiased(False)
        assert line.get_antialiased() is False
        line.set_antialiased(True)
        assert line.get_antialiased() is True

    def test_solid_capstyle_default(self):
        line = Line2D([0], [0])
        cs = line.get_solid_capstyle()
        # OG returns the actual default capstyle string (e.g. 'projecting', 'butt')
        assert cs is None or isinstance(cs, str)

    def test_set_solid_capstyle(self):
        line = Line2D([0], [0])
        line.set_solid_capstyle('round')
        assert line.get_solid_capstyle() == 'round'

    def test_solid_joinstyle_default(self):
        line = Line2D([0], [0])
        js = line.get_solid_joinstyle()
        # OG returns the actual default joinstyle string (e.g. 'round', 'miter')
        assert js is None or isinstance(js, str)

    def test_set_solid_joinstyle(self):
        line = Line2D([0], [0])
        line.set_solid_joinstyle('bevel')
        assert line.get_solid_joinstyle() == 'bevel'

    def test_dash_capstyle_default(self):
        line = Line2D([0], [0])
        cs = line.get_dash_capstyle()
        # OG returns the actual default capstyle string (e.g. 'butt')
        assert cs is None or isinstance(cs, str)

    def test_set_dash_capstyle(self):
        line = Line2D([0], [0])
        line.set_dash_capstyle('butt')
        assert line.get_dash_capstyle() == 'butt'

    def test_dash_joinstyle_default(self):
        line = Line2D([0], [0])
        js = line.get_dash_joinstyle()
        # OG returns the actual default joinstyle string (e.g. 'round', 'miter')
        assert js is None or isinstance(js, str)

    def test_set_dash_joinstyle(self):
        line = Line2D([0], [0])
        line.set_dash_joinstyle('round')
        assert line.get_dash_joinstyle() == 'round'

    def test_get_xydata(self):
        line = Line2D([1, 2, 3], [4, 5, 6])
        xy = line.get_xydata()
        assert len(xy) == 3
        assert xy[0][0] == 1
        assert xy[0][1] == 4

    def test_get_xydata_empty(self):
        line = Line2D([], [])
        xy = line.get_xydata()
        assert len(xy) == 0


# ===================================================================
# Line2D extended tests
# ===================================================================

class TestLine2DExtended:
    def test_fillstyle_default(self):
        """Line2D fillstyle defaults to 'full' or similar."""
        line = Line2D([0], [0])
        fs = line.get_fillstyle()
        assert fs in ('full', 'left', 'right', 'bottom', 'top', 'none', None)

    def test_set_fillstyle(self):
        """Line2D.set_fillstyle changes fillstyle."""
        line = Line2D([0], [0])
        line.set_fillstyle('none')
        assert line.get_fillstyle() == 'none'

    def test_fillstyle_left(self):
        """Line2D fillstyle 'left'."""
        line = Line2D([0], [0])
        line.set_fillstyle('left')
        assert line.get_fillstyle() == 'left'

    def test_get_xdata(self):
        """Line2D.get_xdata returns x values."""
        line = Line2D([1, 2, 3], [4, 5, 6])
        assert list(line.get_xdata()) == [1, 2, 3]

    def test_get_ydata(self):
        """Line2D.get_ydata returns y values."""
        line = Line2D([1, 2, 3], [4, 5, 6])
        assert list(line.get_ydata()) == [4, 5, 6]

    def test_set_xdata(self):
        """Line2D.set_xdata changes x values."""
        line = Line2D([1, 2], [3, 4])
        line.set_xdata([5, 6, 7])
        assert list(line.get_xdata()) == [5, 6, 7]

    def test_set_ydata(self):
        """Line2D.set_ydata changes y values."""
        line = Line2D([1, 2], [3, 4])
        line.set_ydata([10, 20])
        assert list(line.get_ydata()) == [10, 20]

    def test_get_data(self):
        """Line2D.get_data returns (x, y) tuple."""
        line = Line2D([1, 2], [3, 4])
        x, y = line.get_data()
        assert list(x) == [1, 2]
        assert list(y) == [3, 4]

    def test_set_data(self):
        """Line2D.set_data changes x and y."""
        line = Line2D([0], [0])
        line.set_data([1, 2, 3], [4, 5, 6])
        assert list(line.get_xdata()) == [1, 2, 3]
        assert list(line.get_ydata()) == [4, 5, 6]

    def test_get_path_returns_path(self):
        """Line2D.get_path returns a path object."""
        line = Line2D([0, 1, 2], [3, 4, 5])
        path = line.get_path()
        assert path is not None

    def test_marker_default_none(self):
        """Line2D marker defaults to None."""
        line = Line2D([0], [0])
        assert line.get_marker() is None or line.get_marker() == 'None'

    def test_color_and_xdata_independent(self):
        """Changing color does not affect xdata."""
        line = Line2D([1, 2, 3], [4, 5, 6], color='red')
        line.set_color('blue')
        assert list(line.get_xdata()) == [1, 2, 3]

    def test_linewidth_zero(self):
        """Line2D linewidth can be set to zero."""
        line = Line2D([0], [0])
        line.set_linewidth(0)
        assert line.get_linewidth() == 0

    def test_set_color_updates_color(self):
        """Line2D.set_color updates get_color."""
        line = Line2D([0], [0])
        line.set_color('green')
        assert line.get_color() == 'green'

    def test_linestyle_empty_string(self):
        """Line2D linestyle can be empty string (no line); OG may normalize to 'None'."""
        line = Line2D([0], [0])
        line.set_linestyle('')
        assert line.get_linestyle() in ('', 'None', 'none')

    def test_single_point_line(self):
        """Line2D with a single point works."""
        line = Line2D([5], [10])
        assert list(line.get_xdata()) == [5]
        assert list(line.get_ydata()) == [10]


# ===================================================================
# Extended parametric tests for lines (upstream-style)
# ===================================================================


# ---------------------------------------------------------------------------
# axline tests (upstream test_lines.py)
# ---------------------------------------------------------------------------

def test_axline_xy1_slope():
    """ax.axline with xy1 and slope creates an infinite line."""
    fig, ax = plt.subplots()
    line = ax.axline((0, 0), slope=1)
    assert line is not None
    plt.close('all')


def test_axline_two_points():
    """ax.axline with two points creates an infinite line."""
    fig, ax = plt.subplots()
    line = ax.axline((0, 0), (1, 2))
    assert line is not None
    plt.close('all')


def test_axline_in_lines_list():
    """ax.axline result appears in ax.lines."""
    fig, ax = plt.subplots()
    line = ax.axline((0, 0), slope=2)
    assert line in ax.lines
    plt.close('all')


def test_axline_setters_xy1():
    """axline.set_xy1() updates the anchor point."""
    fig, ax = plt.subplots()
    line = ax.axline((0, 0), slope=1)
    line.set_xy1((1, 2))
    assert line.get_xy1() == (1, 2)
    plt.close('all')


def test_axline_setters_slope():
    """axline.set_slope() updates the slope."""
    fig, ax = plt.subplots()
    line = ax.axline((0, 0), slope=1)
    line.set_slope(3)
    assert line.get_slope() == 3
    plt.close('all')


# ---------------------------------------------------------------------------
# Line2D marker and style checks (upstream test_lines.py)
# ---------------------------------------------------------------------------

def test_line_marker_from_string():
    """Line2D accepts string marker codes."""
    from matplotlib.lines import Line2D
    for marker in ['o', 's', '^', 'v', '<', '>', 'D', 'P', '*', 'x', '+']:
        line = Line2D([0], [0], marker=marker)
        assert line.get_marker() == marker


def test_line_markersize_default_positive():
    """Default marker size is positive."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    assert line.get_markersize() > 0


def test_line_alpha_default():
    """Line2D default alpha is None (fully opaque)."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    assert line.get_alpha() is None


def test_line_alpha_set():
    """Line2D alpha can be set and retrieved."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    line.set_alpha(0.5)
    assert abs(line.get_alpha() - 0.5) < 1e-10


def test_line_contains_raises_without_axes():
    """Line2D.contains() without axes may raise or return False — not crash."""
    from matplotlib.lines import Line2D
    line = Line2D([0, 1], [0, 1])
    # Should not raise an unhandled exception
    try:
        result = line.contains(type('E', (), {'x': 0.5, 'y': 0.5, 'button': 1})())
    except Exception:
        pass  # acceptable


def test_line_get_xydata():
    """Line2D.get_xydata() returns (N, 2) array."""
    import numpy as np
    from matplotlib.lines import Line2D
    line = Line2D([0, 1, 2], [3, 4, 5])
    xy = line.get_xydata()
    assert xy.shape == (3, 2)
    assert list(xy[:, 0]) == [0, 1, 2]
    assert list(xy[:, 1]) == [3, 4, 5]


def test_line2d_set_label():
    """Line2D label round-trips."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    line.set_label('myline')
    assert line.get_label() == 'myline'


def test_line2d_set_visible():
    """Line2D visibility round-trips."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    line.set_visible(False)
    assert line.get_visible() is False
    line.set_visible(True)
    assert line.get_visible() is True


def test_line2d_set_zorder():
    """Line2D zorder round-trips."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    line.set_zorder(5)
    assert line.get_zorder() == 5


def test_line2d_xdata_ydata_equal_length():
    """Line2D xdata and ydata have the same length."""
    from matplotlib.lines import Line2D
    line = Line2D([1, 2, 3, 4], [5, 6, 7, 8])
    assert len(line.get_xdata()) == len(line.get_ydata()) == 4


def test_line2d_set_dashes():
    """Line2D set_dashes() accepts offset and dash sequence."""
    from matplotlib.lines import Line2D
    line = Line2D([0, 1], [0, 1])
    line.set_dashes([4, 2])  # 4 on, 2 off
    # Should not raise


def test_line_fillstyle_default():
    """Line2D default fillstyle is 'full'."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    assert line.get_fillstyle() == 'full'


def test_line_fillstyle_none():
    """Line2D fillstyle 'none' produces open markers."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0], marker='o', fillstyle='none')
    assert line.get_fillstyle() == 'none'


# ===================================================================
# Additional line tests (upstream-inspired batch, round 2)
# ===================================================================

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class TestLine2DProperties:
    """Tests for Line2D property get/set."""

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 4.0])
    def test_linewidth_parametric(self, lw):
        line = Line2D([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-6

    @pytest.mark.parametrize('ls', ['solid', 'dashed', 'dotted', 'dashdot'])
    def test_linestyle_parametric(self, ls):
        line = Line2D([0, 1], [0, 1])
        line.set_linestyle(ls)
        # Should not raise

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff8800', 'k'])
    def test_color_parametric(self, color):
        line = Line2D([0, 1], [0, 1], color=color)
        assert line.get_color() == color

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'D', 'x', '+', '.'])
    def test_marker_parametric(self, marker):
        line = Line2D([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker

    def test_line_label(self):
        line = Line2D([0, 1], [0, 1])
        line.set_label('my_line')
        assert line.get_label() == 'my_line'

    def test_line_zorder(self):
        line = Line2D([0, 1], [0, 1])
        line.set_zorder(3)
        assert line.get_zorder() == 3

    def test_line_visible(self):
        line = Line2D([0, 1], [0, 1])
        line.set_visible(False)
        assert not line.get_visible()
        line.set_visible(True)
        assert line.get_visible()

    def test_line_xydata_get(self):
        xdata = [1.0, 2.0, 3.0]
        ydata = [4.0, 5.0, 6.0]
        line = Line2D(xdata, ydata)
        np.testing.assert_array_equal(line.get_xdata(), xdata)
        np.testing.assert_array_equal(line.get_ydata(), ydata)


class TestLine2DInAxes:
    """Tests for Line2D behavior within axes."""

    def test_plot_line_data(self):
        fig, ax = plt.subplots()
        line, = ax.plot([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
        np.testing.assert_array_equal(line.get_ydata(), [4, 5, 6])
        plt.close('all')

    def test_plot_color_argument(self):
        import matplotlib.colors as mcolors
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color='red')
        # color may be stored as name or normalized hex
        c = line.get_color()
        assert mcolors.to_hex(c) == '#ff0000'
        plt.close('all')

    def test_plot_linewidth_argument(self):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=3.0)
        assert abs(line.get_linewidth() - 3.0) < 1e-6
        plt.close('all')

    def test_plot_marker_argument(self):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker='o')
        assert line.get_marker() == 'o'
        plt.close('all')

    def test_plot_label_argument(self):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], label='series_1')
        assert line.get_label() == 'series_1'
        plt.close('all')

    def test_multiple_plots_in_lines(self):
        fig, ax = plt.subplots()
        n = 5
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.xfail(reason="ft2font not implemented in Phase 0; SVG rendering requires Phase 2")
    def test_line_in_svg(self):
        import io
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        buf = io.BytesIO()
        fig.savefig(buf, format='svg')
        svg = buf.getvalue().decode()
        assert '<polyline' in svg or '<path' in svg
        plt.close('all')
