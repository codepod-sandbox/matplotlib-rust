"""Tests for matplotlib.patches module.

Covers Patch defaults, facecolor/edgecolor RGBA conversion, 'none' color
handling, alpha application, Rectangle construction and geometry,
Circle defaults and accessors, and Artist property inheritance.
"""

import pytest

from matplotlib.patches import Patch, Rectangle, Circle
from matplotlib.artist import Artist
from matplotlib.colors import to_rgba


# ===================================================================
# Patch defaults (4 tests)
# ===================================================================

class TestPatchDefaults:
    def test_default_facecolor(self):
        """Default facecolor is 'C0' resolved to its RGBA tuple."""
        p = Patch()
        assert p.get_facecolor() == to_rgba('C0')

    def test_default_edgecolor(self):
        """Default edgecolor is 'black' -> (0, 0, 0, 1)."""
        p = Patch()
        assert p.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)

    def test_default_linewidth(self):
        """Default linewidth is 1.0."""
        p = Patch()
        assert p.get_linewidth() == 1.0

    def test_default_zorder(self):
        """Patch class zorder is 1, overriding Artist's 0."""
        p = Patch()
        assert p.get_zorder() == 1
        assert Patch.zorder == 1


# ===================================================================
# Patch facecolor / edgecolor as RGBA tuples (6 tests)
# ===================================================================

class TestPatchColors:
    def test_facecolor_named(self):
        """Named facecolor resolves to correct RGBA."""
        p = Patch(facecolor='red')
        assert p.get_facecolor() == (1.0, 0.0, 0.0, 1.0)

    def test_edgecolor_named(self):
        """Named edgecolor resolves to correct RGBA."""
        p = Patch(edgecolor='blue')
        assert p.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)

    def test_facecolor_rgb_tuple(self):
        """Setting facecolor as an RGB tuple returns RGBA with alpha=1."""
        p = Patch()
        p.set_facecolor((0.5, 0.25, 0.75))
        assert p.get_facecolor() == (0.5, 0.25, 0.75, 1.0)

    def test_facecolor_rgba_tuple(self):
        """Setting facecolor as an RGBA tuple preserves the alpha."""
        p = Patch()
        p.set_facecolor((0.1, 0.2, 0.3, 0.4))
        assert p.get_facecolor() == (0.1, 0.2, 0.3, 0.4)

    def test_edgecolor_hex(self):
        """Hex edgecolor resolves correctly."""
        p = Patch(edgecolor='#ff0000')
        assert p.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)

    def test_set_facecolor_updates(self):
        """set_facecolor replaces the previous facecolor."""
        p = Patch(facecolor='red')
        p.set_facecolor('green')
        expected = to_rgba('green')
        assert p.get_facecolor() == expected

    def test_set_edgecolor_updates(self):
        """set_edgecolor replaces the previous edgecolor."""
        p = Patch(edgecolor='red')
        p.set_edgecolor('blue')
        assert p.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)


# ===================================================================
# Patch 'none' facecolor / edgecolor (3 tests)
# ===================================================================

class TestPatchNoneColor:
    def test_facecolor_none_string(self):
        """Facecolor 'none' returns fully transparent (0,0,0,0)."""
        p = Patch(facecolor='none')
        assert p.get_facecolor() == (0.0, 0.0, 0.0, 0.0)

    def test_facecolor_none_case_insensitive(self):
        """Facecolor 'None' (mixed case) is also transparent."""
        p = Patch(facecolor='None')
        assert p.get_facecolor() == (0.0, 0.0, 0.0, 0.0)

    def test_edgecolor_none_string(self):
        """Edgecolor 'none' returns fully transparent (0,0,0,0)."""
        p = Patch(edgecolor='none')
        assert p.get_edgecolor() == (0.0, 0.0, 0.0, 0.0)

    def test_none_facecolor_ignores_alpha(self):
        """When facecolor is 'none', artist alpha does not apply."""
        p = Patch(facecolor='none')
        p.set_alpha(0.5)
        assert p.get_facecolor() == (0.0, 0.0, 0.0, 0.0)

    def test_none_edgecolor_ignores_alpha(self):
        """When edgecolor is 'none', artist alpha does not apply."""
        p = Patch(edgecolor='none')
        p.set_alpha(0.5)
        assert p.get_edgecolor() == (0.0, 0.0, 0.0, 0.0)


# ===================================================================
# Patch alpha applied to colors (5 tests)
# ===================================================================

class TestPatchAlpha:
    def test_alpha_applied_to_facecolor(self):
        """Artist alpha overrides the facecolor alpha channel."""
        p = Patch(facecolor='red')
        p.set_alpha(0.5)
        fc = p.get_facecolor()
        assert fc == (1.0, 0.0, 0.0, 0.5)

    def test_alpha_applied_to_edgecolor(self):
        """Artist alpha overrides the edgecolor alpha channel."""
        p = Patch(edgecolor='blue')
        p.set_alpha(0.3)
        ec = p.get_edgecolor()
        assert ec == (0.0, 0.0, 1.0, 0.3)

    def test_alpha_applied_to_both(self):
        """Alpha is applied to both face and edge colors simultaneously."""
        p = Patch(facecolor='red', edgecolor='green')
        p.set_alpha(0.7)
        fc = p.get_facecolor()
        ec = p.get_edgecolor()
        assert fc[3] == 0.7
        assert ec[3] == 0.7

    def test_no_alpha_preserves_original(self):
        """Without artist alpha, the original color alpha is preserved."""
        p = Patch(facecolor=(0.5, 0.5, 0.5, 0.8))
        assert p.get_alpha() is None
        fc = p.get_facecolor()
        assert fc == (0.5, 0.5, 0.5, 0.8)

    def test_alpha_overrides_rgba_tuple_alpha(self):
        """Artist alpha overrides even an explicit RGBA tuple alpha."""
        p = Patch(facecolor=(1.0, 0.0, 0.0, 0.9))
        p.set_alpha(0.2)
        fc = p.get_facecolor()
        assert fc == (1.0, 0.0, 0.0, 0.2)


# ===================================================================
# Patch linewidth (2 tests)
# ===================================================================

class TestPatchLinewidth:
    def test_set_linewidth(self):
        """set_linewidth changes the linewidth."""
        p = Patch()
        p.set_linewidth(3.5)
        assert p.get_linewidth() == 3.5

    def test_linewidth_constructor(self):
        """linewidth can be set via constructor."""
        p = Patch(linewidth=2.0)
        assert p.get_linewidth() == 2.0


# ===================================================================
# Rectangle construction and accessors (7 tests)
# ===================================================================

class TestRectangleConstruction:
    def test_basic_construction(self):
        """Rectangle stores xy, width, and height."""
        r = Rectangle((1, 2), 3, 4)
        assert r.get_x() == 1
        assert r.get_y() == 2
        assert r.get_width() == 3
        assert r.get_height() == 4

    def test_get_xy(self):
        """get_xy returns the anchor point as a tuple."""
        r = Rectangle((5, 10), 20, 30)
        assert r.get_xy() == (5, 10)

    def test_xy_is_tuple(self):
        """xy is stored as a tuple even if passed as a list."""
        r = Rectangle([3, 4], 5, 6)
        xy = r.get_xy()
        assert isinstance(xy, tuple)
        assert xy == (3, 4)

    def test_zero_dimensions(self):
        """Rectangle with zero width and height is valid."""
        r = Rectangle((0, 0), 0, 0)
        assert r.get_width() == 0
        assert r.get_height() == 0

    def test_negative_dimensions(self):
        """Rectangle with negative width/height is valid."""
        r = Rectangle((0, 0), -5, -10)
        assert r.get_width() == -5
        assert r.get_height() == -10

    def test_float_coordinates(self):
        """Rectangle accepts float coordinates."""
        r = Rectangle((1.5, 2.5), 3.5, 4.5)
        assert r.get_x() == 1.5
        assert r.get_y() == 2.5
        assert r.get_width() == 3.5
        assert r.get_height() == 4.5

    def test_inherits_patch_defaults(self):
        """Rectangle inherits Patch default facecolor, edgecolor, linewidth."""
        r = Rectangle((0, 0), 1, 1)
        assert r.get_facecolor() == to_rgba('C0')
        assert r.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
        assert r.get_linewidth() == 1.0


# ===================================================================
# Rectangle setters (3 tests)
# ===================================================================

class TestRectangleSetters:
    def test_set_xy(self):
        """set_xy updates the anchor point."""
        r = Rectangle((0, 0), 1, 1)
        r.set_xy((10, 20))
        assert r.get_xy() == (10, 20)
        assert r.get_x() == 10
        assert r.get_y() == 20

    def test_set_xy_from_list(self):
        """set_xy converts a list to a tuple."""
        r = Rectangle((0, 0), 1, 1)
        r.set_xy([7, 8])
        assert isinstance(r.get_xy(), tuple)
        assert r.get_xy() == (7, 8)

    def test_set_width(self):
        """set_width updates the width."""
        r = Rectangle((0, 0), 1, 1)
        r.set_width(99)
        assert r.get_width() == 99

    def test_set_height(self):
        """set_height updates the height."""
        r = Rectangle((0, 0), 1, 1)
        r.set_height(42)
        assert r.get_height() == 42


# ===================================================================
# Rectangle get_corners (4 tests)
# ===================================================================

class TestRectangleCorners:
    def test_corners_unit_square(self):
        """Unit square at origin has expected corners."""
        r = Rectangle((0, 0), 1, 1)
        corners = r.get_corners()
        assert corners == [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_corners_offset(self):
        """Corners are correctly offset from anchor."""
        r = Rectangle((2, 3), 4, 5)
        corners = r.get_corners()
        assert corners == [(2, 3), (6, 3), (6, 8), (2, 8)]

    def test_corners_order(self):
        """Corners are returned in BL, BR, TR, TL order."""
        r = Rectangle((1, 2), 10, 20)
        bl, br, tr, tl = r.get_corners()
        # bottom-left
        assert bl == (1, 2)
        # bottom-right
        assert br == (11, 2)
        # top-right
        assert tr == (11, 22)
        # top-left
        assert tl == (1, 22)

    def test_corners_after_mutation(self):
        """Corners reflect updated xy, width, and height."""
        r = Rectangle((0, 0), 1, 1)
        r.set_xy((5, 5))
        r.set_width(10)
        r.set_height(20)
        corners = r.get_corners()
        assert corners == [(5, 5), (15, 5), (15, 25), (5, 25)]

    def test_corners_returns_four_tuples(self):
        """get_corners always returns a list of exactly 4 tuples."""
        r = Rectangle((0, 0), 3, 4)
        corners = r.get_corners()
        assert len(corners) == 4
        for c in corners:
            assert isinstance(c, tuple)
            assert len(c) == 2


# ===================================================================
# Rectangle kwargs forwarding (2 tests)
# ===================================================================

class TestRectangleKwargs:
    def test_patch_kwargs_forwarded(self):
        """Rectangle forwards facecolor/edgecolor to Patch."""
        r = Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='blue')
        assert r.get_facecolor() == (1.0, 0.0, 0.0, 1.0)
        assert r.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)

    def test_linewidth_kwarg(self):
        """Rectangle forwards linewidth to Patch."""
        r = Rectangle((0, 0), 1, 1, linewidth=5.0)
        assert r.get_linewidth() == 5.0


# ===================================================================
# Circle defaults (3 tests)
# ===================================================================

class TestCircleDefaults:
    def test_default_center(self):
        """Default center is (0.0, 0.0)."""
        c = Circle()
        assert c.get_center() == (0.0, 0.0)

    def test_default_radius(self):
        """Default radius is 0.5."""
        c = Circle()
        assert c.get_radius() == 0.5

    def test_inherits_patch_defaults(self):
        """Circle inherits Patch defaults for facecolor, edgecolor, linewidth."""
        c = Circle()
        assert c.get_facecolor() == to_rgba('C0')
        assert c.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
        assert c.get_linewidth() == 1.0


# ===================================================================
# Circle get/set center and radius (5 tests)
# ===================================================================

class TestCircleAccessors:
    def test_construction_with_values(self):
        """Circle stores center and radius from constructor."""
        c = Circle(center=(3, 4), radius=2.0)
        assert c.get_center() == (3, 4)
        assert c.get_radius() == 2.0

    def test_set_center(self):
        """set_center updates the center."""
        c = Circle()
        c.set_center((10, 20))
        assert c.get_center() == (10, 20)

    def test_set_center_from_list(self):
        """set_center converts a list to a tuple."""
        c = Circle()
        c.set_center([7, 8])
        assert isinstance(c.get_center(), tuple)
        assert c.get_center() == (7, 8)

    def test_set_radius(self):
        """set_radius updates the radius."""
        c = Circle()
        c.set_radius(10.0)
        assert c.get_radius() == 10.0

    def test_zero_radius(self):
        """Circle with zero radius is valid."""
        c = Circle(radius=0)
        assert c.get_radius() == 0


# ===================================================================
# Circle kwargs forwarding (2 tests)
# ===================================================================

class TestCircleKwargs:
    def test_patch_kwargs_forwarded(self):
        """Circle forwards facecolor/edgecolor to Patch."""
        c = Circle(facecolor='red', edgecolor='blue')
        assert c.get_facecolor() == (1.0, 0.0, 0.0, 1.0)
        assert c.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)

    def test_linewidth_kwarg(self):
        """Circle forwards linewidth to Patch."""
        c = Circle(linewidth=3.0)
        assert c.get_linewidth() == 3.0


# ===================================================================
# Artist property inheritance (8 tests)
# ===================================================================

class TestArtistInheritance:
    def test_patch_is_artist(self):
        """Patch inherits from Artist."""
        p = Patch()
        assert isinstance(p, Artist)

    def test_rectangle_is_artist(self):
        """Rectangle inherits from Artist (via Patch)."""
        r = Rectangle((0, 0), 1, 1)
        assert isinstance(r, Artist)

    def test_circle_is_artist(self):
        """Circle inherits from Artist (via Patch)."""
        c = Circle()
        assert isinstance(c, Artist)

    def test_visible_default_and_setter(self):
        """Patches default to visible=True and support set_visible."""
        p = Patch()
        assert p.get_visible() is True
        p.set_visible(False)
        assert p.get_visible() is False

    def test_alpha_default_and_setter(self):
        """Patches default to alpha=None and support set_alpha."""
        p = Patch()
        assert p.get_alpha() is None
        p.set_alpha(0.5)
        assert p.get_alpha() == 0.5

    def test_label_default_and_setter(self):
        """Patches default to label='' and support set_label."""
        p = Patch()
        assert p.get_label() == ''
        p.set_label('my patch')
        assert p.get_label() == 'my patch'

    def test_zorder_inheritance(self):
        """Patch zorder=1 overrides Artist zorder=0."""
        assert Artist.zorder == 0
        assert Patch.zorder == 1
        p = Patch()
        assert p.get_zorder() == 1

    def test_set_zorder(self):
        """set_zorder changes the zorder for an instance."""
        p = Patch()
        p.set_zorder(10)
        assert p.get_zorder() == 10

    def test_rectangle_zorder(self):
        """Rectangle inherits Patch zorder of 1."""
        r = Rectangle((0, 0), 1, 1)
        assert r.get_zorder() == 1

    def test_circle_zorder(self):
        """Circle inherits Patch zorder of 1."""
        c = Circle()
        assert c.get_zorder() == 1


# ===================================================================
# Batch setter via kwargs (3 tests)
# ===================================================================

class TestBatchSetter:
    def test_patch_set_via_kwargs(self):
        """Patch constructor forwards unknown kwargs through Artist.set()."""
        p = Patch(visible=False, label='test')
        assert p.get_visible() is False
        assert p.get_label() == 'test'

    def test_rectangle_set_via_kwargs(self):
        """Rectangle forwards extra kwargs through to Artist.set()."""
        r = Rectangle((0, 0), 1, 1, visible=False, label='rect')
        assert r.get_visible() is False
        assert r.get_label() == 'rect'

    def test_circle_set_via_kwargs(self):
        """Circle forwards extra kwargs through to Artist.set()."""
        c = Circle(visible=False, label='circ')
        assert c.get_visible() is False
        assert c.get_label() == 'circ'
