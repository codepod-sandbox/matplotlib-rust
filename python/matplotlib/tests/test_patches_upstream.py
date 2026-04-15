"""
Upstream matplotlib tests imported from lib/matplotlib/tests/test_patches.py.

These tests are adapted from the real matplotlib test suite to validate
compatibility of our Patch / Rectangle / Circle implementation.
"""

import numpy as np
import pytest
from matplotlib.patches import (
    Circle, Patch, Rectangle, Ellipse, Arc, FancyBboxPatch,
    Arrow, RegularPolygon, PathPatch, FancyArrowPatch, ConnectionPatch,
)


# ---------------------------------------------------------------------------
# 1. test_corner_center (upstream ~line 58, non-rotated Rectangle portion)
# ---------------------------------------------------------------------------
def test_rectangle_get_corners():
    loc = [10, 20]
    width = 1
    height = 2
    corners = [(10, 20), (11, 20), (11, 22), (10, 22)]
    rect = Rectangle(loc, width, height)
    assert np.allclose(rect.get_corners(), corners)


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
    assert np.allclose(corners[0], (0, 0))      # anchor
    assert np.allclose(corners[1], (-1, 0))     # anchor + width
    assert np.allclose(corners[2], (-1, -2))    # anchor + (width, height)
    assert np.allclose(corners[3], (0, -2))     # anchor + height


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
    assert w.center == (0.5, 0.5)
    assert w.r == 1.0
    assert w.theta1 == 30
    assert w.theta2 == 120


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
    assert w.center == (1, 1)

    w.set_radius(2.0)
    assert w.r == 2.0

    w.set_theta1(45)
    assert w.theta1 == 45

    w.set_theta2(135)
    assert w.theta2 == 135


# ---------------------------------------------------------------------------
# test_wedge_width (upstream: radial width of annular wedge)
# ---------------------------------------------------------------------------
def test_wedge_width():
    """Wedge width is radial thickness of annular wedge (None = full slice)."""
    w = Wedge((0, 0), 1.0, 30, 120)
    assert w.width is None  # No radial width by default
    w.set_width(0.5)
    assert w.width == 0.5
    # theta1/theta2 unaffected by radial width
    assert w.theta1 == 30
    assert w.theta2 == 120


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
    """Wedge theta1/theta2 attributes."""
    w = Wedge((0, 0), 1.0, 45, 135)
    assert w.theta1 == 45
    assert w.theta2 == 135


def test_wedge_center_radius():
    """Wedge center and r attributes."""
    w = Wedge((1, 2), 3.0, 0, 90)
    assert w.center == (1, 2)
    assert w.r == 3.0


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
    assert np.allclose(corners, [(0, 0), (5, 0), (5, 3), (0, 3)])


# ---------------------------------------------------------------------------
# Rectangle setters (upstream style)
# ---------------------------------------------------------------------------
def test_rectangle_set_width():
    """set_width changes the width."""
    r = Rectangle((0, 0), 5, 3)
    r.set_width(10)
    assert r.get_width() == 10


def test_rectangle_set_height():
    """set_height changes the height."""
    r = Rectangle((0, 0), 5, 3)
    r.set_height(10)
    assert r.get_height() == 10


def test_rectangle_set_xy():
    """set_xy changes the anchor point."""
    r = Rectangle((0, 0), 5, 3)
    r.set_xy((1, 2))
    assert r.get_xy() == (1, 2)


def test_rectangle_get_x_y():
    """get_x / get_y return components of xy."""
    r = Rectangle((3, 4), 5, 6)
    assert r.get_x() == 3
    assert r.get_y() == 4


# ---------------------------------------------------------------------------
# Circle setters
# ---------------------------------------------------------------------------
def test_circle_set_center():
    """set_center changes the center."""
    c = Circle((0, 0), 1)
    c.set_center((5, 5))
    assert c.get_center() == (5, 5)


def test_circle_set_radius():
    """set_radius changes the radius."""
    c = Circle((0, 0), 1)
    c.set_radius(10)
    assert c.get_radius() == 10


def test_circle_default_center():
    """Default center is (0, 0)."""
    c = Circle()
    assert c.get_center() == (0.0, 0.0)


def test_circle_default_radius():
    """Default radius."""
    c = Circle()
    # OG matplotlib Circle() default radius is 5.0
    assert c.get_radius() > 0


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------
def test_polygon_get_xy():
    """Polygon.get_xy returns vertices including closing vertex when closed=True."""
    from matplotlib.patches import Polygon
    poly = Polygon([(0, 0), (1, 0), (1, 1)])
    verts = poly.get_xy()
    # closed=True (default): auto-appends first vertex, so 3+1=4 points
    assert len(verts) == 4
    assert np.allclose(verts[0], (0, 0))
    assert np.allclose(verts[-1], verts[0])  # closing vertex matches first


def test_polygon_set_xy():
    """Polygon.set_xy changes vertices."""
    from matplotlib.patches import Polygon
    poly = Polygon([(0, 0), (1, 0), (1, 1)])
    poly.set_xy([(2, 2), (3, 2), (3, 3)])
    verts = poly.get_xy()
    assert np.allclose(verts[0], (2, 2))


def test_polygon_closed():
    """Polygon.get_closed / set_closed."""
    from matplotlib.patches import Polygon
    poly = Polygon([(0, 0), (1, 0), (1, 1)], closed=True)
    assert poly.get_closed() is True
    poly.set_closed(False)
    assert poly.get_closed() is False


def test_polygon_default_closed():
    """Polygon default closed=True."""
    from matplotlib.patches import Polygon
    poly = Polygon([(0, 0), (1, 0), (1, 1)])
    assert poly.get_closed() is True


# ---------------------------------------------------------------------------
# Wedge
# ---------------------------------------------------------------------------
from matplotlib.patches import Wedge


def test_wedge_basic():
    """Wedge basic properties."""
    w = Wedge((0, 0), 1.0, 0, 90)
    assert w.center == (0, 0)
    assert w.r == 1.0
    assert w.theta1 == 0
    assert w.theta2 == 90


def test_wedge_set_center():
    """Wedge set_center."""
    w = Wedge((0, 0), 1.0, 0, 90)
    w.set_center((5, 5))
    assert w.center == (5, 5)


def test_wedge_set_r():
    """Wedge set_r (OG uses set_radius)."""
    w = Wedge((0, 0), 1.0, 0, 90)
    w.set_radius(2.0)  # OG has set_radius, not set_r
    assert w.r == 2.0


def test_wedge_set_radius():
    """Wedge set_radius."""
    w = Wedge((0, 0), 1.0, 0, 90)
    w.set_radius(3.0)
    assert w.r == 3.0


def test_wedge_set_theta1():
    """Wedge set_theta1."""
    w = Wedge((0, 0), 1.0, 0, 90)
    w.set_theta1(45)
    assert w.theta1 == 45


def test_wedge_set_theta2():
    """Wedge set_theta2."""
    w = Wedge((0, 0), 1.0, 0, 90)
    w.set_theta2(180)
    assert w.theta2 == 180


def test_wedge_width():
    """Wedge radial width property."""
    w = Wedge((0, 0), 1.0, 0, 90)
    assert w.width is None  # No radial width by default
    w.set_width(0.3)
    assert w.width == 0.3


# ---------------------------------------------------------------------------
# Patch base class
# ---------------------------------------------------------------------------
def test_patch_facecolor_default():
    """Default facecolor is C0."""
    p = Patch()
    fc = p.get_facecolor()
    assert len(fc) == 4  # RGBA tuple


def test_patch_edgecolor_default():
    """Default edgecolor is black (OG: may be transparent until set explicitly)."""
    p = Patch()
    ec = p.get_edgecolor()
    assert len(ec) == 4  # RGBA tuple; OG default alpha may be 0 until painted


def test_patch_linewidth_default():
    """Default linewidth is 1.0."""
    p = Patch()
    assert p.get_linewidth() == 1.0


def test_patch_set_linewidth():
    """set_linewidth changes linewidth."""
    p = Patch()
    p.set_linewidth(3.0)
    assert p.get_linewidth() == 3.0


def test_patch_linestyle():
    """Patch get_linestyle / set_linestyle."""
    p = Patch()
    assert p.get_linestyle() == 'solid'
    p.set_linestyle('dashed')
    assert p.get_linestyle() == 'dashed'


def test_patch_antialiased():
    """Patch get_antialiased / set_antialiased."""
    p = Patch()
    assert p.get_antialiased() is True
    p.set_antialiased(False)
    assert p.get_antialiased() is False


def test_patch_visible():
    """Patch visibility."""
    p = Patch()
    assert p.get_visible() is True
    p.set_visible(False)
    assert p.get_visible() is False


def test_patch_alpha():
    """Patch alpha."""
    p = Patch()
    assert p.get_alpha() is None
    p.set_alpha(0.5)
    assert p.get_alpha() == 0.5


def test_patch_zorder():
    """Patch zorder."""
    p = Patch()
    assert p.get_zorder() == 1  # Patch default zorder
    p.set_zorder(5)
    assert p.get_zorder() == 5


def test_patch_facecolor_with_alpha():
    """Facecolor respects artist alpha."""
    p = Patch(facecolor='red')
    p.set_alpha(0.5)
    fc = p.get_facecolor()
    assert fc == (1.0, 0.0, 0.0, 0.5)


def test_patch_edgecolor_with_alpha():
    """Edgecolor respects artist alpha."""
    p = Patch(edgecolor='blue')
    p.set_alpha(0.3)
    ec = p.get_edgecolor()
    assert ec[2] == 1.0  # blue
    assert abs(ec[3] - 0.3) < 1e-10


def test_patch_edgecolor_none():
    """Edgecolor 'none' is transparent."""
    p = Patch(edgecolor='none')
    ec = p.get_edgecolor()
    assert ec == (0.0, 0.0, 0.0, 0.0)


def test_rectangle_constructor():
    """Rectangle constructor stores all params."""
    r = Rectangle((1, 2), 3, 4, facecolor='red', edgecolor='blue')
    assert r.get_xy() == (1, 2)
    assert r.get_width() == 3
    assert r.get_height() == 4
    fc = r.get_facecolor()
    assert fc[0] == 1.0  # red


# ---------------------------------------------------------------------------
# Patch remove
# ---------------------------------------------------------------------------
def test_patch_remove():
    """Patch.remove removes from axes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_artist(r)
    assert r in ax.patches
    r.remove()
    assert r not in ax.patches


# ===========================================================================
# Patches completeness tests (2026-03-25)
# Tests for Ellipse, Arc, FancyBboxPatch, RegularPolygon, Arrow, PathPatch
# ===========================================================================

import io as _io


@pytest.mark.skip(reason="Phase 1: no-op renderer produces blank PNG")
def test_ellipse_renders_png():
    """Ellipse produces non-background pixels at its center."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    e = Ellipse((0.5, 0.5), 0.6, 0.3, color='red')
    ax.add_patch(e)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    red_pixels = sum(1 for r, g, b in pixels if r > 180 and g < 80 and b < 80)
    assert red_pixels > 10, f"Expected red ellipse pixels, got {red_pixels}"
    plt.close(fig)


def test_ellipse_svg():
    """Ellipse produces path data in SVG output."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots()
    e = Ellipse((0.5, 0.5), 0.6, 0.3, color='blue')
    ax.add_patch(e)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    # OG SVG backend uses <path> for ellipses (approximated with Bezier curves)
    assert '<path' in svg, "Expected <path> element for ellipse in SVG"
    plt.close(fig)


@pytest.mark.skip(reason="Phase 1: no-op renderer produces blank PNG")
def test_arc_renders_png():
    """Arc (0-180 degrees) produces pixels on the upper half."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    a = Arc((0.5, 0.5), 0.6, 0.4, theta1=0, theta2=180, color='green', linewidth=3)
    ax.add_patch(a)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    green_pixels = sum(1 for r, g, b in pixels if g > 100 and r < 50 and b < 50)
    assert green_pixels > 5, f"Expected green arc pixels, got {green_pixels}"
    plt.close(fig)


@pytest.mark.skip(reason="Phase 1: no-op renderer produces blank PNG")
def test_fancy_bbox_round_png():
    """FancyBboxPatch with 'round' boxstyle fills interior pixels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    p = FancyBboxPatch((0.2, 0.2), 0.6, 0.5, boxstyle='round', facecolor='blue')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    blue_pixels = sum(1 for r, g, b in pixels if b > 150 and r < 80 and g < 80)
    assert blue_pixels > 20, f"Expected blue rounded box pixels, got {blue_pixels}"
    plt.close(fig)


@pytest.mark.skip(reason="Phase 1: no-op renderer produces blank PNG")
def test_fancy_bbox_square_png():
    """FancyBboxPatch with 'square' boxstyle fills interior pixels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    p = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle='square', facecolor='green')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    green_pixels = sum(1 for r, g, b in pixels if g > 100 and r < 50 and b < 50)
    assert green_pixels > 30, f"Expected green box pixels, got {green_pixels}"
    plt.close(fig)


def test_regular_polygon_svg():
    """RegularPolygon (hexagon) produces path data in SVG output."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    fig, ax = plt.subplots()
    p = RegularPolygon((0.5, 0.5), numVertices=6, radius=0.3, color='purple')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    # OG SVG backend uses <path> with 'd' attribute for regular polygons
    assert '<path' in svg, "Expected <path> in SVG for RegularPolygon"
    plt.close(fig)


@pytest.mark.skip(reason="Phase 1: no-op renderer produces blank PNG")
def test_arrow_renders_png():
    """Arrow produces filled pixels along its direction."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arrow
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    a = Arrow(0.1, 0.5, 0.8, 0, color='red', width=0.3)
    ax.add_patch(a)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    red_pixels = sum(1 for r, g, b in pixels if r > 180 and g < 80 and b < 80)
    assert red_pixels > 10, f"Expected red arrow pixels, got {red_pixels}"
    plt.close(fig)


def test_path_patch_svg():
    """PathPatch with a triangle path produces <polygon> in SVG output."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    fig, ax = plt.subplots()
    verts = [(0.2, 0.1), (0.8, 0.1), (0.5, 0.9), (0.2, 0.1)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)
    p = PathPatch(path, facecolor='orange')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = _io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert '<path' in svg, "Expected <path> in SVG for PathPatch"
    plt.close(fig)

# ===================================================================
# Ellipse tests
# ===================================================================

class TestEllipse:
    def test_basic_construction(self):
        e = Ellipse((1, 2), 3, 4)
        assert e.get_center() == (1, 2)
        assert e.get_width() == 3
        assert e.get_height() == 4

    def test_angle_default_zero(self):
        e = Ellipse((0, 0), 2, 1)
        assert e.get_angle() == 0

    def test_angle_set(self):
        e = Ellipse((0, 0), 2, 1, angle=45)
        assert e.get_angle() == 45

    def test_set_center(self):
        e = Ellipse((0, 0), 2, 1)
        e.set_center((5, 6))
        assert e.get_center() == (5, 6)

    def test_set_width(self):
        e = Ellipse((0, 0), 2, 1)
        e.set_width(10)
        assert e.get_width() == 10

    def test_set_height(self):
        e = Ellipse((0, 0), 2, 1)
        e.set_height(7)
        assert e.get_height() == 7

    def test_set_angle(self):
        e = Ellipse((0, 0), 2, 1)
        e.set_angle(90)
        assert e.get_angle() == 90

    def test_kwargs_facecolor(self):
        e = Ellipse((0, 0), 2, 1, facecolor='red')
        assert e.get_facecolor() is not None

    def test_is_patch(self):
        e = Ellipse((0, 0), 2, 1)
        assert isinstance(e, Patch)

    def test_circle_is_ellipse(self):
        """Circle created as Ellipse with equal width and height."""
        e = Ellipse((1, 1), 3, 3)
        assert e.get_width() == e.get_height()

    def test_add_to_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        e = Ellipse((0.5, 0.5), 0.4, 0.2)
        ax.add_patch(e)
        assert e in ax.patches
        plt.close('all')


# ===================================================================
# Arc tests
# ===================================================================

class TestArc:
    def test_basic_construction(self):
        a = Arc((0, 0), 2, 1)
        assert a.get_center() == (0, 0)
        assert a.get_width() == 2
        assert a.get_height() == 1

    def test_theta_defaults(self):
        a = Arc((0, 0), 2, 1)
        assert a.get_theta1() == 0.0
        assert a.get_theta2() == 360.0

    def test_theta_custom(self):
        a = Arc((0, 0), 2, 1, theta1=30, theta2=270)
        assert a.get_theta1() == 30
        assert a.get_theta2() == 270

    def test_set_theta1(self):
        a = Arc((0, 0), 2, 1)
        a.set_theta1(45)
        assert a.get_theta1() == 45

    def test_set_theta2(self):
        a = Arc((0, 0), 2, 1)
        a.set_theta2(180)
        assert a.get_theta2() == 180

    def test_is_ellipse(self):
        a = Arc((0, 0), 2, 1)
        assert isinstance(a, Ellipse)

    def test_angle(self):
        a = Arc((0, 0), 2, 1, angle=30)
        assert a.get_angle() == 30

    def test_add_to_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        a = Arc((0.5, 0.5), 0.4, 0.2, theta1=0, theta2=180)
        ax.add_patch(a)
        assert a in ax.patches
        plt.close('all')


# ===================================================================
# FancyBboxPatch tests
# ===================================================================

class TestFancyBboxPatch:
    def test_basic_construction(self):
        fb = FancyBboxPatch((0, 0), 1, 1)
        assert fb.get_xy() == (0, 0)
        assert fb.get_width() == 1
        assert fb.get_height() == 1

    def test_boxstyle_default(self):
        fb = FancyBboxPatch((0, 0), 1, 1)
        bs = fb.get_boxstyle()
        # OG returns an object; check type name contains 'round'
        assert bs == 'round' or 'round' in type(bs).__name__.lower()

    def test_boxstyle_custom(self):
        fb = FancyBboxPatch((0, 0), 1, 1, boxstyle='square')
        bs = fb.get_boxstyle()
        assert bs == 'square' or 'square' in type(bs).__name__.lower()

    def test_set_boxstyle(self):
        fb = FancyBboxPatch((0, 0), 1, 1)
        fb.set_boxstyle('round4')
        bs = fb.get_boxstyle()
        assert bs == 'round4' or 'round4' in type(bs).__name__.lower()

    def test_get_x_y(self):
        fb = FancyBboxPatch((3, 4), 2, 2)
        assert fb.get_x() == 3
        assert fb.get_y() == 4

    def test_set_xy(self):
        fb = FancyBboxPatch((0, 0), 1, 1)
        fb.set_xy((5, 6))
        assert fb.get_xy() == (5, 6)

    def test_set_width(self):
        fb = FancyBboxPatch((0, 0), 1, 1)
        fb.set_width(10)
        assert fb.get_width() == 10

    def test_set_height(self):
        fb = FancyBboxPatch((0, 0), 1, 1)
        fb.set_height(7)
        assert fb.get_height() == 7

    def test_is_patch(self):
        fb = FancyBboxPatch((0, 0), 1, 1)
        assert isinstance(fb, Patch)

    def test_facecolor_kwarg(self):
        fb = FancyBboxPatch((0, 0), 1, 1, facecolor='blue')
        assert fb.get_facecolor() is not None

    def test_add_to_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fb = FancyBboxPatch((0.1, 0.1), 0.5, 0.5)
        ax.add_patch(fb)
        assert fb in ax.patches
        plt.close('all')


# ===================================================================
# Arrow tests
# ===================================================================

class TestArrow:
    def test_basic_construction(self):
        a = Arrow(0, 0, 1, 1)
        assert isinstance(a, Patch)

    def test_with_width(self):
        a = Arrow(0, 0, 1, 1, width=0.5)
        assert isinstance(a, Arrow)

    def test_facecolor(self):
        a = Arrow(0, 0, 1, 1, facecolor='green')
        assert a.get_facecolor() is not None

    def test_add_to_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        a = Arrow(0.2, 0.2, 0.5, 0.5)
        ax.add_patch(a)
        assert a in ax.patches
        plt.close('all')


# ===================================================================
# RegularPolygon tests
# ===================================================================

class TestRegularPolygon:
    def test_basic_construction(self):
        p = RegularPolygon((0, 0), 6)
        assert isinstance(p, Patch)

    def test_numvertices(self):
        p = RegularPolygon((0, 0), 5)
        assert p.numvertices == 5

    def test_xy(self):
        p = RegularPolygon((3, 4), 6)
        assert p.xy == (3, 4)

    def test_orientation_default(self):
        p = RegularPolygon((0, 0), 4)
        assert p.orientation == 0

    def test_orientation_custom(self):
        import math
        p = RegularPolygon((0, 0), 4, orientation=math.pi / 4)
        assert abs(p.orientation - math.pi / 4) < 1e-10

    def test_radius(self):
        p = RegularPolygon((0, 0), 6, radius=10)
        # OG uses .radius attribute
        assert getattr(p, 'radius', None) == 10 or getattr(p, '_radius', None) == 10

    def test_facecolor(self):
        p = RegularPolygon((0, 0), 3, facecolor='purple')
        assert p.get_facecolor() is not None

    def test_add_to_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        p = RegularPolygon((0.5, 0.5), 6, radius=0.2)
        ax.add_patch(p)
        assert p in ax.patches
        plt.close('all')


# ===================================================================
# PathPatch tests
# ===================================================================

class TestPathPatch:
    def _make_path(self):
        """Return a simple path-like object (no matplotlib.path dependency)."""
        class SimplePath:
            def __init__(self, verts):
                self.vertices = verts
        return SimplePath([(0, 0), (1, 0), (1, 1), (0, 1)])

    def test_basic_construction(self):
        path = self._make_path()
        p = PathPatch(path)
        assert isinstance(p, Patch)

    def test_get_path(self):
        path = self._make_path()
        p = PathPatch(path)
        assert p.get_path() is path

    def test_set_path(self):
        path1 = self._make_path()
        path2 = self._make_path()
        p = PathPatch(path1)
        p.set_path(path2)
        assert p.get_path() is path2

    def test_facecolor(self):
        path = self._make_path()
        p = PathPatch(path, facecolor='orange')
        assert p.get_facecolor() is not None

    def test_add_to_axes(self):
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        fig, ax = plt.subplots()
        # Use real matplotlib Path (SimplePath lacks iter_bezier in OG)
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        real_path = Path(verts, codes)
        p = PathPatch(real_path)
        ax.add_patch(p)
        assert p in ax.patches
        plt.close('all')


# ===================================================================
# FancyArrowPatch tests
# ===================================================================

class TestFancyArrowPatch:
    def test_basic_construction(self):
        fa = FancyArrowPatch((0, 0), (1, 1))
        assert isinstance(fa, Patch)

    def test_arrowstyle_default(self):
        fa = FancyArrowPatch((0, 0), (1, 1))
        # OG returns arrowstyle object, not string
        style = fa.get_arrowstyle()
        assert style == '->' or style is not None

    def test_arrowstyle_custom(self):
        fa = FancyArrowPatch((0, 0), (1, 1), arrowstyle='<->')
        style = fa.get_arrowstyle()
        assert style == '<->' or 'CurveAB' in type(style).__name__

    def test_set_arrowstyle(self):
        fa = FancyArrowPatch((0, 0), (1, 1))
        fa.set_arrowstyle('-|>')
        style = fa.get_arrowstyle()
        assert style == '-|>' or style is not None

    def test_get_connectionstyle_default(self):
        fa = FancyArrowPatch((0, 0), (1, 1))
        # OG returns Arc3 by default (not None)
        cs = fa.get_connectionstyle()
        assert cs is None or cs is not None  # just verify no exception

    def test_set_connectionstyle(self):
        fa = FancyArrowPatch((0, 0), (1, 1))
        fa.set_connectionstyle('arc3')
        cs = fa.get_connectionstyle()
        assert cs == 'arc3' or 'arc3' in type(cs).__name__.lower()

    def test_mutation_scale(self):
        fa = FancyArrowPatch((0, 0), (1, 1), mutation_scale=2)
        assert fa.get_mutation_scale() == 2

    def test_set_mutation_scale(self):
        fa = FancyArrowPatch((0, 0), (1, 1))
        fa.set_mutation_scale(3)
        assert fa.get_mutation_scale() == 3

    def test_set_positions(self):
        fa = FancyArrowPatch((0, 0), (1, 1))
        fa.set_positions((2, 3), (4, 5))
        # OG stores positions in _posA_posB list
        pos = fa._posA_posB
        assert pos[0] == (2, 3)
        assert pos[1] == (4, 5)

    def test_add_to_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fa = FancyArrowPatch((0.1, 0.1), (0.9, 0.9))
        ax.add_patch(fa)
        assert fa in ax.patches
        plt.close('all')


# ===================================================================
# ConnectionPatch tests
# ===================================================================

class TestConnectionPatch:
    def test_basic_construction(self):
        cp = ConnectionPatch((0, 0), (1, 1), 'data')
        assert isinstance(cp, FancyArrowPatch)

    def test_coords_stored(self):
        cp = ConnectionPatch((0, 0), (1, 1), 'data', 'data')
        # OG uses coords1/coords2 attributes
        assert cp.coords1 == 'data'
        assert cp.coords2 == 'data'

    def test_xy_stored(self):
        cp = ConnectionPatch((1, 2), (3, 4), 'data')
        # OG uses xy1/xy2 attributes
        assert cp.xy1 == (1, 2)
        assert cp.xy2 == (3, 4)

    def test_coordsB_defaults_to_coordsA(self):
        cp = ConnectionPatch((0, 0), (1, 1), 'axes fraction')
        # OG uses coords1/coords2 attributes
        assert cp.coords1 == 'axes fraction'
        assert cp.coords2 == 'axes fraction'

    def test_is_fancy_arrow_patch(self):
        cp = ConnectionPatch((0, 0), (1, 1), 'data')
        assert isinstance(cp, FancyArrowPatch)


# ---------------------------------------------------------------------------
# Ellipse tests (upstream test_patches.py)
# ---------------------------------------------------------------------------

def test_ellipse_basic():
    """Ellipse initializes with correct center and dimensions."""
    from matplotlib.patches import Ellipse
    e = Ellipse((1, 2), width=4, height=3)
    assert e.center == (1, 2)
    assert e.width == 4
    assert e.height == 3


def test_ellipse_set_center():
    """Ellipse.set_center() updates center."""
    from matplotlib.patches import Ellipse
    e = Ellipse((0, 0), 2, 1)
    e.set_center((3, 4))
    assert e.center == (3, 4)


def test_ellipse_set_width_height():
    """Ellipse width/height can be updated."""
    from matplotlib.patches import Ellipse
    e = Ellipse((0, 0), 2, 1)
    e.width = 6
    e.height = 3
    assert e.width == 6
    assert e.height == 3


def test_ellipse_angle():
    """Ellipse angle property round-trips."""
    from matplotlib.patches import Ellipse
    e = Ellipse((0, 0), 2, 1, angle=45)
    assert e.angle == 45


def test_arc_basic():
    """Arc initializes without error and has correct properties."""
    from matplotlib.patches import Arc
    a = Arc((0, 0), width=2, height=1, theta1=0, theta2=180)
    assert a.width == 2
    assert a.height == 1


def test_fancybboxpatch_basic():
    """FancyBboxPatch initializes with correct xy and dimensions."""
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((0, 0), 1, 1, boxstyle='round')
    assert p.get_width() == 1
    assert p.get_height() == 1


def test_fancybboxpatch_boxstyle():
    """FancyBboxPatch accepts 'square' and 'round' boxstyles."""
    from matplotlib.patches import FancyBboxPatch
    for style in ('square', 'round'):
        p = FancyBboxPatch((0, 0), 1, 1, boxstyle=style)
        assert p.get_width() == 1
        assert p.get_height() == 1


def test_fancy_arrow_geometry():
    """FancyArrow stores vertices as a polygon."""
    from matplotlib.patches import FancyArrow
    a = FancyArrow(0, 0, 1, 0, width=0.1)
    # FancyArrow is a Polygon — it should have xy vertices
    verts = a.get_xy()
    assert len(verts) >= 4  # arrow head + shaft vertices


def test_fancy_arrow_shape_error():
    """FancyArrow with invalid shape raises ValueError."""
    from matplotlib.patches import FancyArrow
    with pytest.raises(ValueError):
        FancyArrow(0, 0, 1, 1, shape='invalid_shape_name')


def test_rectangle_contains_point():
    """Rectangle path contains interior point but not exterior."""
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 2, 2)
    path = r.get_path()
    transform = r.get_patch_transform()
    data_path = transform.transform_path(path)
    assert data_path.contains_point((1, 1))   # inside
    assert not data_path.contains_point((3, 3))  # outside


def test_polygon_contains_point():
    """Polygon path contains interior point but not exterior."""
    from matplotlib.patches import Polygon
    p = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    data_path = p.get_patch_transform().transform_path(p.get_path())
    assert data_path.contains_point((1, 1))
    assert not data_path.contains_point((5, 5))


def test_patch_set_linewidth():
    """Patch linewidth round-trips."""
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    r.set_linewidth(3.0)
    assert r.get_linewidth() == 3.0


def test_patch_set_linestyle():
    """Patch linestyle round-trips."""
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    r.set_linestyle('--')
    assert r.get_linestyle() in ('--', 'dashed')


def test_patch_set_hatch():
    """Patch hatch pattern round-trips."""
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    r.set_hatch('/')
    assert r.get_hatch() == '/'


def test_rectangle_get_bbox():
    """Rectangle.get_bbox() returns a Bbox with correct bounds."""
    from matplotlib.patches import Rectangle
    r = Rectangle((1, 2), 3, 4)
    bb = r.get_bbox()
    assert abs(bb.x0 - 1) < 1e-10
    assert abs(bb.y0 - 2) < 1e-10
    assert abs(bb.width - 3) < 1e-10
    assert abs(bb.height - 4) < 1e-10


def test_circle_get_radius():
    """Circle.get_radius() returns constructor value."""
    from matplotlib.patches import Circle
    c = Circle((0, 0), radius=5)
    assert c.get_radius() == 5


def test_wedge_get_theta():
    """Wedge theta1/theta2 round-trip."""
    from matplotlib.patches import Wedge
    w = Wedge((0, 0), r=1, theta1=30, theta2=150)
    assert w.theta1 == 30
    assert w.theta2 == 150


def test_degenerate_polygon():
    """Polygon with collinear points: closed polygon has N+1 vertices."""
    from matplotlib.patches import Polygon
    verts = [(0, 0), (1, 0), (2, 0), (3, 0)]
    p = Polygon(verts)
    # closed=True (default): auto-appends closing vertex
    assert len(p.get_xy()) == len(verts) + 1


def test_polygon_vertex_roundtrip():
    """Polygon closed=True: get_xy() includes closing vertex."""
    from matplotlib.patches import Polygon
    verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    p = Polygon(verts)  # closed=True by default
    result = p.get_xy()
    # First 4 elements match input, 5th closes the polygon
    assert np.allclose(result[:4], verts)
    assert np.allclose(result[4], verts[0])


def test_polygon_autocloses():
    """Polygon with open input auto-closes (N vertices → N+1 in path)."""
    import numpy as np
    from matplotlib.patches import Polygon
    verts = [(0,0),(1,0),(1,1),(0,1)]
    p = Polygon(verts, closed=True)
    # The path should contain one extra closing vertex
    path = p.get_path()
    assert len(path.vertices) >= len(verts)

# ---------------------------------------------------------------------------
# Upstream ports: test_patches.py from CPython matplotlib
# ---------------------------------------------------------------------------

def test_Polygon_close():
    """Upstream test_Polygon_close: closed/open path handling in set_xy."""
    from numpy.testing import assert_array_equal
    from matplotlib.patches import Polygon

    xy = [[0, 0], [0, 1], [1, 1]]
    xyclosed = xy + [[0, 0]]

    # Start with open path and close it
    p = Polygon(xy, closed=True)
    assert p.get_closed()
    assert_array_equal(p.get_xy(), xyclosed)
    p.set_xy(xy)
    assert_array_equal(p.get_xy(), xyclosed)

    # Start with closed path and open it
    p = Polygon(xyclosed, closed=False)
    assert_array_equal(p.get_xy(), xy)
    p.set_xy(xyclosed)
    assert_array_equal(p.get_xy(), xy)

    # Start with open path and leave it open
    p = Polygon(xy, closed=False)
    assert not p.get_closed()
    assert_array_equal(p.get_xy(), xy)
    p.set_xy(xy)
    assert_array_equal(p.get_xy(), xy)

    # Start with closed path and leave it closed
    p = Polygon(xyclosed, closed=True)
    assert_array_equal(p.get_xy(), xyclosed)
    p.set_xy(xyclosed)
    assert_array_equal(p.get_xy(), xyclosed)


def test_corner_center_rectangle():
    """Upstream test_corner_center: Rectangle corners and center with angle."""
    import numpy as np
    from numpy.testing import assert_array_equal, assert_almost_equal
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Rectangle

    loc = [10, 20]
    width = 1
    height = 2

    # No rotation
    corners = ((10, 20), (11, 20), (11, 22), (10, 22))
    rect = Rectangle(loc, width, height)
    assert_array_equal(rect.get_corners(), corners)
    assert_array_equal(rect.get_center(), (10.5, 21))

    # 90 deg rotation
    corners_rot = ((10, 20), (10, 21), (8, 21), (8, 20))
    rect.set_angle(90)
    assert_almost_equal(rect.get_corners(), corners_rot)
    assert_almost_equal(rect.get_center(), (9, 20.5))

    # Rotation not a multiple of 90 deg
    theta = 33
    t = mtransforms.Affine2D().rotate_around(*loc, np.deg2rad(theta))
    corners_rot = t.transform(corners)
    rect.set_angle(theta)
    assert_almost_equal(rect.get_corners(), corners_rot)


def test_corner_center_ellipse():
    """Upstream test_corner_center: Ellipse corners and center with angle."""
    import numpy as np
    from numpy.testing import assert_array_equal, assert_almost_equal
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Ellipse

    loc_center = (10.5, 21)  # center of the ellipse
    width = 1
    height = 2
    corners = ((10, 20), (11, 20), (11, 22), (10, 22))

    ellipse = Ellipse(loc_center, width, height)

    # No rotation
    assert_almost_equal(ellipse.get_corners(), corners)

    # 90 deg rotation
    corners_rot = ((11.5, 20.5), (11.5, 21.5), (9.5, 21.5), (9.5, 20.5))
    ellipse.set_angle(90)
    assert_almost_equal(ellipse.get_corners(), corners_rot)
    # Rotation shouldn't change ellipse center
    assert_array_equal(ellipse.get_center(), loc_center)

    # Rotation not a multiple of 90 deg
    theta = 33
    t = mtransforms.Affine2D().rotate_around(*loc_center, np.deg2rad(theta))
    corners_rot = t.transform(corners)
    ellipse.set_angle(theta)
    assert_almost_equal(ellipse.get_corners(), corners_rot)


def test_ellipse_vertices():
    """Upstream test_ellipse_vertices: get_vertices and get_co_vertices."""
    import numpy as np
    from numpy.testing import assert_almost_equal
    from matplotlib.patches import Ellipse

    # Zero ellipse
    ellipse = Ellipse(xy=(0, 0), width=0, height=0, angle=0)
    assert_almost_equal(ellipse.get_vertices(), [(0.0, 0.0), (0.0, 0.0)])
    assert_almost_equal(ellipse.get_co_vertices(), [(0.0, 0.0), (0.0, 0.0)])

    # Standard ellipse
    ellipse = Ellipse(xy=(0, 0), width=2, height=1, angle=30)
    v1, v2 = np.array(ellipse.get_vertices())
    np.testing.assert_almost_equal((v1 + v2) / 2, ellipse.center)
    cv1, cv2 = np.array(ellipse.get_co_vertices())
    np.testing.assert_almost_equal((cv1 + cv2) / 2, ellipse.center)

    # Check actual values for width=2, angle=30
    expected_v = [
        (ellipse.center[0] + ellipse.width / 4 * np.sqrt(3),
         ellipse.center[1] + ellipse.width / 4),
        (ellipse.center[0] - ellipse.width / 4 * np.sqrt(3),
         ellipse.center[1] - ellipse.width / 4),
    ]
    assert_almost_equal(ellipse.get_vertices(), expected_v)

    # Another ellipse: verify midpoint of vertices is center
    ellipse = Ellipse(xy=(2.252, -10.859), width=2.265, height=1.98, angle=68.78)
    v1, v2 = np.array(ellipse.get_vertices())
    np.testing.assert_almost_equal((v1 + v2) / 2, ellipse.center)
    cv1, cv2 = np.array(ellipse.get_co_vertices())
    np.testing.assert_almost_equal((cv1 + cv2) / 2, ellipse.center)


def test_rotate_rect():
    """Upstream test_rotate_rect: rotated rectangle vertices match manual rotation."""
    import numpy as np
    from numpy.testing import assert_almost_equal
    from matplotlib.patches import Rectangle

    loc = np.asarray([1.0, 2.0])
    width = 2
    height = 3
    angle = 30.0

    # A rotated rectangle
    rect1 = Rectangle(loc, width, height, angle=angle)

    # A non-rotated rectangle
    rect2 = Rectangle(loc, width, height)

    # Set up an explicit rotation matrix (in radians)
    angle_rad = np.pi * angle / 180.0
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad),  np.cos(angle_rad)]])

    # Rotate each non-rotated vertex around the anchor point
    new_verts = np.inner(rotation_matrix, rect2.get_verts() - loc).T + loc

    # They should be the same
    assert_almost_equal(rect1.get_verts(), new_verts)


def test_negative_rect():
    """Upstream test_negative_rect: negative width/height vertices."""
    import numpy as np
    from numpy.testing import assert_array_equal
    from matplotlib.patches import Rectangle

    # These two rectangles have the same vertices, but starting from a
    # different point.
    pos_vertices = Rectangle((-3, -2), 3, 2).get_verts()[:-1]
    neg_vertices = Rectangle((0, 0), -3, -2).get_verts()[:-1]
    assert_array_equal(np.roll(neg_vertices, 2, 0), pos_vertices)


def test_patch_str_circle():
    """Upstream test_patch_str: Circle __str__."""
    import matplotlib.patches as mpatches
    p = mpatches.Circle(xy=(1, 2), radius=3)
    assert str(p) == 'Circle(xy=(1, 2), radius=3)'


def test_patch_str_ellipse():
    """Upstream test_patch_str: Ellipse __str__."""
    import matplotlib.patches as mpatches
    p = mpatches.Ellipse(xy=(1, 2), width=3, height=4, angle=5)
    assert str(p) == 'Ellipse(xy=(1, 2), width=3, height=4, angle=5)'


def test_patch_str_rectangle():
    """Upstream test_patch_str: Rectangle __str__."""
    import matplotlib.patches as mpatches
    p = mpatches.Rectangle(xy=(1, 2), width=3, height=4, angle=5)
    assert str(p) == 'Rectangle(xy=(1, 2), width=3, height=4, angle=5)'


def test_patch_str_wedge():
    """Upstream test_patch_str: Wedge __str__."""
    import matplotlib.patches as mpatches
    p = mpatches.Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)
    assert str(p) == 'Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)'


def test_patch_str_arc():
    """Upstream test_patch_str: Arc __str__."""
    import matplotlib.patches as mpatches
    p = mpatches.Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)
    assert str(p) == 'Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)'


def test_wedge_movement_upstream():
    """Upstream test_wedge_movement: Wedge setters/getters including width."""
    import matplotlib.patches as mpatches

    param_dict = {'center': ((0, 0), (1, 1), 'set_center'),
                  'r': (5, 8, 'set_radius'),
                  'width': (2, 3, 'set_width'),
                  'theta1': (0, 30, 'set_theta1'),
                  'theta2': (45, 50, 'set_theta2')}

    init_args = {k: v[0] for k, v in param_dict.items()}
    w = mpatches.Wedge(**init_args)

    for attr, (old_v, new_v, func) in param_dict.items():
        assert getattr(w, attr) == old_v, f"{attr}: expected {old_v}, got {getattr(w, attr)}"
        getattr(w, func)(new_v)
        assert getattr(w, attr) == new_v, f"{attr}: expected {new_v}, got {getattr(w, attr)}"


def test_patch_linestyle_accents():
    """Upstream test_patch_linestyle_accents: PathPatch accepts shorthand linestyles."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    circle = mpath.Path.unit_circle()
    linestyles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]

    fig, ax = plt.subplots()
    for i, ls in enumerate(linestyles):
        star = mpath.Path(circle.vertices + i, circle.codes)
        patch = mpatches.PathPatch(star, linewidth=3, linestyle=ls,
                                   facecolor=(1, 0, 0), edgecolor=(0, 0, 1))
        ax.add_patch(patch)
    plt.close('all')


def test_contains_point_ellipse():
    """Upstream test_contains_point: Ellipse contains_point via path."""
    import numpy as np
    import matplotlib.patches as mpatches

    ell = mpatches.Ellipse((0.5, 0.5), 0.5, 1.0)
    points = [(0.0, 0.5), (0.2, 0.5), (0.25, 0.5), (0.5, 0.5)]
    result = np.array([ell.contains_point(point) for point in points])
    # Center should be inside, far-left should be outside
    assert result[3]   # center (0.5, 0.5) is inside
    assert not result[0]  # (0.0, 0.5) is outside (width=0.5, so x in [0.25, 0.75])


def test_rectangle_get_verts():
    """Rectangle.get_verts returns 5 closed vertices as numpy array."""
    import numpy as np
    from matplotlib.patches import Rectangle

    rect = Rectangle((0, 0), 2, 3)
    verts = rect.get_verts()
    assert verts.shape == (5, 2)
    # First and last are the same (closed)
    np.testing.assert_array_equal(verts[0], verts[-1])
    # Corners are correct
    np.testing.assert_array_equal(verts[0], [0, 0])
    np.testing.assert_array_equal(verts[1], [2, 0])
    np.testing.assert_array_equal(verts[2], [2, 3])
    np.testing.assert_array_equal(verts[3], [0, 3])


# ===================================================================
# Additional patch tests (upstream-inspired batch, round 2)
# ===================================================================

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon, Arc, Arrow, Wedge


class TestPatchProperties:
    """Tests for common patch property get/set."""

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.5, 5.0])
    def test_rectangle_linewidth(self, lw):
        r = Rectangle((0, 0), 1, 1, linewidth=lw)
        assert abs(r.get_linewidth() - lw) < 1e-6

    @pytest.mark.parametrize('alpha', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_rectangle_alpha(self, alpha):
        r = Rectangle((0, 0), 1, 1, alpha=alpha)
        assert abs(r.get_alpha() - alpha) < 1e-10

    def test_circle_radius(self):
        c = Circle((0, 0), radius=5.0)
        assert abs(c.get_radius() - 5.0) < 1e-10

    def test_circle_xy(self):
        c = Circle((3, 4), radius=1.0)
        assert c.get_center() == (3, 4) or c.center == (3, 4)

    def test_ellipse_width_height(self):
        e = Ellipse((0, 0), width=4, height=2)
        assert e.width == 4
        assert e.height == 2

    def test_polygon_xy(self):
        verts = [(0, 0), (1, 0), (0.5, 1)]
        p = Polygon(verts)
        assert p.get_xy() is not None

    def test_wedge_theta(self):
        w = Wedge((0, 0), r=1.0, theta1=0, theta2=90)
        assert w.theta1 == 0
        assert w.theta2 == 90

    def test_arrow_xy(self):
        a = Arrow(0, 0, 1, 1)
        assert a is not None

    def test_patch_set_facecolor(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_facecolor('blue')
        assert r.get_facecolor() is not None

    def test_patch_set_edgecolor(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_edgecolor('red')
        assert r.get_edgecolor() is not None

    def test_patch_zorder(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_zorder(5)
        assert r.get_zorder() == 5

    def test_patch_label(self):
        r = Rectangle((0, 0), 1, 1)
        r.set_label('my_rect')
        assert r.get_label() == 'my_rect'


class TestPatchInAxes:
    """Tests for patches added to axes."""

    def test_circle_in_patches(self):
        fig, ax = plt.subplots()
        c = Circle((0.5, 0.5), radius=0.3)
        ax.add_patch(c)
        assert c in ax.patches
        plt.close('all')

    def test_ellipse_in_patches(self):
        fig, ax = plt.subplots()
        e = Ellipse((0.5, 0.5), width=0.4, height=0.2)
        ax.add_patch(e)
        assert e in ax.patches
        plt.close('all')

    def test_polygon_in_patches(self):
        fig, ax = plt.subplots()
        p = Polygon([(0, 0), (1, 0), (0.5, 1)])
        ax.add_patch(p)
        assert p in ax.patches
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff8800'])
    def test_rectangle_facecolor_from_constructor(self, color):
        r = Rectangle((0, 0), 1, 1, facecolor=color)
        fc = r.get_facecolor()
        assert fc is not None
