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
