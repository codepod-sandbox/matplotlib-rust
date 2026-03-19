"""
Upstream matplotlib tests imported from lib/matplotlib/tests/test_patches.py.

These tests are adapted from the real matplotlib test suite to validate
compatibility of our Patch / Rectangle / Circle implementation.
"""

from matplotlib.patches import Circle, Patch, Rectangle


# ---------------------------------------------------------------------------
# 1. test_corner_center (upstream ~line 58, non-rotated Rectangle portion)
# ---------------------------------------------------------------------------
def test_rectangle_get_corners():
    loc = [10, 20]
    width = 1
    height = 2
    corners = [(10, 20), (11, 20), (11, 22), (10, 22)]
    rect = Rectangle(loc, width, height)
    assert rect.get_corners() == corners


# ---------------------------------------------------------------------------
# 2. test_patch_color_none (upstream ~line 336)
# Make sure facecolor='none' returns transparent regardless of alpha.
# ---------------------------------------------------------------------------
def test_patch_color_none():
    c = Circle((0, 0), 1, facecolor='none', alpha=1)
    assert c.get_facecolor() == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# 3. test_patch_base_properties (upstream ~line various)
# ---------------------------------------------------------------------------
def test_patch_base_properties():
    p = Patch()
    # Defaults
    assert p.get_linewidth() == 1.0
    assert p.get_visible() is True
    assert p.zorder == 1

    # facecolor
    p.set_facecolor('red')
    assert p.get_facecolor() == (1.0, 0.0, 0.0, 1.0)

    # edgecolor
    p.set_edgecolor('blue')
    assert p.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)

    # linewidth
    p.set_linewidth(3.5)
    assert p.get_linewidth() == 3.5

    # alpha applies to facecolor
    p.set_alpha(0.5)
    fc = p.get_facecolor()
    assert fc == (1.0, 0.0, 0.0, 0.5)

    # alpha applies to edgecolor
    ec = p.get_edgecolor()
    assert ec == (0.0, 0.0, 1.0, 0.5)


# ---------------------------------------------------------------------------
# 4. test_patch_edgecolor_none (upstream ~line various)
# ---------------------------------------------------------------------------
def test_patch_edgecolor_none():
    p = Patch(edgecolor='none')
    assert p.get_edgecolor() == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 5. test_rectangle_properties (upstream ~line various)
# ---------------------------------------------------------------------------
def test_rectangle_properties():
    rect = Rectangle((1, 2), 3, 4)
    assert rect.get_x() == 1
    assert rect.get_y() == 2
    assert rect.get_xy() == (1, 2)
    assert rect.get_width() == 3
    assert rect.get_height() == 4

    rect.set_xy((5, 6))
    assert rect.get_xy() == (5, 6)
    assert rect.get_x() == 5
    assert rect.get_y() == 6

    rect.set_width(10)
    assert rect.get_width() == 10

    rect.set_height(20)
    assert rect.get_height() == 20


# ---------------------------------------------------------------------------
# 6. test_rectangle_kwargs_forwarding (upstream pattern)
# ---------------------------------------------------------------------------
def test_rectangle_kwargs_forwarding():
    rect = Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='blue',
                      linewidth=3, alpha=0.5)
    assert rect.get_facecolor() == (1.0, 0.0, 0.0, 0.5)
    assert rect.get_edgecolor() == (0.0, 0.0, 1.0, 0.5)
    assert rect.get_linewidth() == 3
    assert rect.get_alpha() == 0.5


# ---------------------------------------------------------------------------
# 7. test_circle_properties (upstream ~line 477, test_patch_str extract)
# ---------------------------------------------------------------------------
def test_circle_properties():
    c = Circle((1, 2), radius=3)
    assert c.get_center() == (1, 2)
    assert c.get_radius() == 3

    c.set_center((4, 5))
    assert c.get_center() == (4, 5)

    c.set_radius(10)
    assert c.get_radius() == 10


# ---------------------------------------------------------------------------
# 8. test_circle_kwargs_forwarding (upstream pattern)
# ---------------------------------------------------------------------------
def test_circle_kwargs_forwarding():
    c = Circle((0, 0), 1, facecolor='none', alpha=1)
    # facecolor='none' should produce transparent regardless of alpha
    assert c.get_facecolor() == (0.0, 0.0, 0.0, 0.0)

    c2 = Circle((1, 2), radius=5, edgecolor='blue', linewidth=2.5)
    assert c2.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)
    assert c2.get_linewidth() == 2.5


# ---------------------------------------------------------------------------
# 9. test_rectangle_negative_dims (upstream ~line 182, test_negative_rect)
# Our implementation allows negative width/height (it stores them as-is).
# ---------------------------------------------------------------------------
def test_rectangle_negative_dims():
    rect = Rectangle((0, 0), -1, -2)
    assert rect.get_width() == -1
    assert rect.get_height() == -2
    # Corners reflect the negative dimensions
    corners = rect.get_corners()
    assert corners[0] == (0, 0)      # anchor
    assert corners[1] == (-1, 0)     # anchor + width
    assert corners[2] == (-1, -2)    # anchor + (width, height)
    assert corners[3] == (0, -2)     # anchor + height


# ---------------------------------------------------------------------------
# 10. test_patch_visible_alpha (upstream pattern)
# ---------------------------------------------------------------------------
def test_patch_visible_alpha():
    p = Patch()
    assert p.get_visible() is True
    p.set_visible(False)
    assert p.get_visible() is False

    assert p.get_alpha() is None
    p.set_alpha(0.7)
    assert p.get_alpha() == 0.7


# ===========================================================================
# Newly ported upstream tests (2026-03-19)
# Source: https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/tests/test_patches.py
# ===========================================================================

from datetime import datetime, timedelta
from matplotlib.patches import Wedge


# ---------------------------------------------------------------------------
# test_datetime_rectangle (upstream)
# ---------------------------------------------------------------------------
def test_datetime_rectangle():
    """Upstream: Rectangle accepts datetime/timedelta dimensions."""
    start = datetime(2017, 1, 1, 0, 0, 0)
    delta = timedelta(seconds=16)
    patch = Rectangle((start, 0), delta, 1)
    assert patch.get_width() == delta
    assert patch.get_height() == 1


# ---------------------------------------------------------------------------
# test_wedge_basic (upstream-inspired)
# ---------------------------------------------------------------------------
def test_wedge_basic():
    """Test Wedge creation and property access."""
    w = Wedge((0.5, 0.5), 1.0, 30, 120, facecolor='red')
    assert w.get_center() == (0.5, 0.5)
    assert w.get_r() == 1.0
    assert w.get_theta1() == 30
    assert w.get_theta2() == 120


# ---------------------------------------------------------------------------
# test_polygon_set_xy (upstream-inspired)
# ---------------------------------------------------------------------------
def test_polygon_set_xy():
    """Test Polygon vertex manipulation."""
    from matplotlib.patches import Polygon
    verts = [(0, 0), (1, 0), (1, 1)]
    p = Polygon(verts)
    assert len(p.get_xy()) == 3
    new_verts = [(0, 0), (2, 0), (2, 2), (0, 2)]
    p.set_xy(new_verts)
    assert len(p.get_xy()) == 4


# ---------------------------------------------------------------------------
# test_circle_properties (upstream-inspired extension)
# ---------------------------------------------------------------------------
def test_circle_set_center():
    """Test Circle set_center/get_center."""
    from matplotlib.patches import Circle
    c = Circle((1, 2), 3)
    assert c.get_center() == (1, 2)
    c.set_center((5, 6))
    assert c.get_center() == (5, 6)
    c.set_radius(10)
    assert c.get_radius() == 10


# ===========================================================================
# Third batch of ported upstream tests (2026-03-19)
# New features: Patch linestyle/antialiased, Polygon closed, Wedge setters
# ===========================================================================


# ---------------------------------------------------------------------------
# test_patch_linestyle (upstream test_default_linestyle + test_patch_custom_linestyle)
# ---------------------------------------------------------------------------
def test_patch_linestyle():
    """Upstream: Patch linestyle get/set."""
    p = Patch()
    assert p.get_linestyle() == 'solid'  # default
    p.set_linestyle('dashed')
    assert p.get_linestyle() == 'dashed'


# ---------------------------------------------------------------------------
# test_patch_antialiased (upstream test_default_antialiased)
# ---------------------------------------------------------------------------
def test_patch_antialiased():
    """Upstream: Patch antialiased get/set."""
    p = Patch()
    assert p.get_antialiased() is True  # default
    p.set_antialiased(False)
    assert p.get_antialiased() is False


# ---------------------------------------------------------------------------
# test_polygon_closed (upstream test_Polygon_close)
# ---------------------------------------------------------------------------
def test_polygon_closed():
    """Upstream: test_Polygon_close — closed property round-trip."""
    from matplotlib.patches import Polygon
    verts = [(0, 0), (1, 0), (1, 1)]
    p = Polygon(verts, closed=True)
    assert p.get_closed() is True
    p.set_closed(False)
    assert p.get_closed() is False
    p.set_closed(True)
    assert p.get_closed() is True


# ---------------------------------------------------------------------------
# test_wedge_movement (upstream test_wedge_movement)
# ---------------------------------------------------------------------------
def test_wedge_movement():
    """Upstream: test_wedge_movement — Wedge setters work."""
    w = Wedge((0, 0), 1.0, 0, 90)

    w.set_center((1, 1))
    assert w.get_center() == (1, 1)

    w.set_radius(2.0)
    assert w.get_r() == 2.0

    w.set_theta1(45)
    assert w.get_theta1() == 45

    w.set_theta2(135)
    assert w.get_theta2() == 135


# ---------------------------------------------------------------------------
# test_wedge_width (upstream-inspired)
# ---------------------------------------------------------------------------
def test_wedge_width():
    """Wedge set_width/get_width angular width."""
    w = Wedge((0, 0), 1.0, 30, 120)
    assert w.get_width() == 90
    w.set_width(45)
    assert w.get_theta2() == 75  # 30 + 45


# ---------------------------------------------------------------------------
# test_add_artist_patch (upstream-inspired)
# ---------------------------------------------------------------------------
def test_add_artist_patch():
    """Adding a Patch via add_artist puts it in ax.patches."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    w = Wedge((0.5, 0.5), 0.3, 0, 180, facecolor='blue')
    ax.add_artist(w)
    assert w in ax.patches
    assert w.axes is ax


# ===========================================================================
# Newly ported upstream tests (2026-03-19, batch 2)
# ===========================================================================


def test_rectangle_xy_roundtrip():
    """Rectangle get_xy / set_xy round-trip."""
    r = Rectangle((1, 2), 3, 4)
    assert r.get_xy() == (1, 2)
    r.set_xy((5, 6))
    assert r.get_xy() == (5, 6)


def test_rectangle_width_height_roundtrip():
    """Rectangle get_width/height / set_width/height."""
    r = Rectangle((0, 0), 10, 20)
    assert r.get_width() == 10
    assert r.get_height() == 20
    r.set_width(30)
    r.set_height(40)
    assert r.get_width() == 30
    assert r.get_height() == 40


def test_patch_facecolor_roundtrip():
    """Patch set_facecolor / get_facecolor round-trip."""
    p = Patch(facecolor='red')
    assert p.get_facecolor()[:3] == (1, 0, 0)
    p.set_facecolor('blue')
    assert p.get_facecolor()[:3] == (0, 0, 1)


def test_patch_edgecolor_roundtrip():
    """Patch set_edgecolor / get_edgecolor round-trip."""
    p = Patch(edgecolor='green')
    fc = p.get_edgecolor()[:3]
    # green ~ (0, 0.5, 0) approximately
    assert fc[1] > 0


def test_patch_alpha_roundtrip():
    """Patch set_alpha / get_alpha."""
    p = Patch()
    p.set_alpha(0.5)
    assert p.get_alpha() == 0.5


def test_patch_visibility():
    """Patch visibility can be toggled."""
    p = Patch()
    assert p.get_visible() is True
    p.set_visible(False)
    assert p.get_visible() is False


def test_patch_zorder():
    """Patch default zorder is 1."""
    p = Patch()
    assert p.get_zorder() == 1


def test_circle_radius_roundtrip():
    """Circle set_radius / get_radius."""
    c = Circle((0, 0), 5)
    assert c.get_radius() == 5
    c.set_radius(10)
    assert c.get_radius() == 10


def test_polygon_xy_roundtrip():
    """Polygon get_xy / set_xy."""
    from matplotlib.patches import Polygon
    verts = [(0, 0), (1, 0), (1, 1)]
    p = Polygon(verts)
    xy = p.get_xy()
    assert len(xy) >= 3


def test_wedge_angles():
    """Wedge theta1/theta2 accessors."""
    w = Wedge((0, 0), 1.0, 45, 135)
    assert w.get_theta1() == 45
    assert w.get_theta2() == 135


def test_wedge_center_radius():
    """Wedge center and r accessors."""
    w = Wedge((1, 2), 3.0, 0, 90)
    assert w.get_center() == (1, 2)
    assert w.get_r() == 3.0


def test_rectangle_from_bar():
    """Rectangles created by bar() have correct geometry."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    bc = ax.bar([1, 2], [3, 4], width=0.5)
    r0 = bc[0]
    # bar at x=1, width=0.5 => xy = (0.75, 0)
    assert abs(r0.get_xy()[0] - 0.75) < 1e-10
    assert r0.get_xy()[1] == 0
    assert abs(r0.get_width() - 0.5) < 1e-10
    assert r0.get_height() == 3


def test_patch_label():
    """Patch label can be set/get."""
    p = Patch()
    p.set_label('my patch')
    assert p.get_label() == 'my patch'


def test_rectangle_corners_standard():
    """Standard positive-dimension rectangle corners."""
    r = Rectangle((0, 0), 5, 3)
    corners = r.get_corners()
    assert corners == [(0, 0), (5, 0), (5, 3), (0, 3)]
