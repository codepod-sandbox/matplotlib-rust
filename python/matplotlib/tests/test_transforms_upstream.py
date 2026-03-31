"""Upstream-ported tests for matplotlib.transforms."""

import math
import pytest
from matplotlib.tests._approx import approx

from matplotlib.transforms import (
    Bbox, BboxBase, Affine2D, IdentityTransform, BboxTransform,
    BboxTransformTo, BboxTransformFrom, ScaledTranslation,
    CompositeGenericTransform, BlendedGenericTransform,
    blended_transform_factory, TransformNode, Transform,
    nonsingular,
)


# ===================================================================
# Bbox creation
# ===================================================================

class TestBboxCreation:
    def test_from_bounds(self):
        bb = Bbox.from_bounds(1, 2, 3, 4)
        assert bb.x0 == 1
        assert bb.y0 == 2
        assert bb.width == 3
        assert bb.height == 4
        assert bb.x1 == 4
        assert bb.y1 == 6

    def test_from_extents(self):
        bb = Bbox.from_extents(1, 2, 4, 6)
        assert bb.x0 == 1
        assert bb.y0 == 2
        assert bb.x1 == 4
        assert bb.y1 == 6

    def test_unit(self):
        bb = Bbox.unit()
        assert bb.x0 == 0
        assert bb.y0 == 0
        assert bb.x1 == 1
        assert bb.y1 == 1

    def test_null(self):
        bb = Bbox.null()
        assert bb.x0 == float('inf')
        assert bb.y0 == float('inf')
        assert bb.x1 == float('-inf')
        assert bb.y1 == float('-inf')

    def test_default_constructor(self):
        bb = Bbox()
        assert bb.x0 == 0.0
        assert bb.y0 == 0.0
        assert bb.x1 == 1.0
        assert bb.y1 == 1.0

    def test_points_constructor(self):
        bb = Bbox([[1, 2], [3, 4]])
        assert bb.x0 == 1.0
        assert bb.y0 == 2.0
        assert bb.x1 == 3.0
        assert bb.y1 == 4.0

    def test_points_as_float(self):
        bb = Bbox([[1, 2], [3, 4]])
        assert isinstance(bb.x0, float)
        assert isinstance(bb.y0, float)


# ===================================================================
# Bbox properties
# ===================================================================

class TestBboxProperties:
    def test_width_height(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.width == 4
        assert bb.height == 6

    def test_bounds(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.bounds == (1, 2, 4, 6)

    def test_extents(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.extents == (1, 2, 5, 8)

    def test_min_max(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.min == (1, 2)
        assert bb.max == (5, 8)

    def test_intervalx(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.intervalx == (1, 5)

    def test_intervaly(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.intervaly == (2, 8)

    def test_p0_p1(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.p0 == (1, 2)
        assert bb.p1 == (5, 8)

    def test_size(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        assert bb.size == (4, 6)

    def test_is_unit(self):
        assert Bbox.unit().is_unit()
        assert not Bbox.from_extents(0, 0, 2, 2).is_unit()

    def test_get_points(self):
        bb = Bbox.from_extents(1, 2, 3, 4)
        pts = bb.get_points()
        assert pts == [[1, 2], [3, 4]]

    def test_set_points(self):
        bb = Bbox()
        bb.set_points([[5, 6], [7, 8]])
        assert bb.x0 == 5
        assert bb.y0 == 6
        assert bb.x1 == 7
        assert bb.y1 == 8


# ===================================================================
# Bbox containment
# ===================================================================

class TestBboxContainment:
    def test_contains_inside(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert bb.contains(5, 5)

    def test_contains_on_edge(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert bb.contains(0, 0)
        assert bb.contains(10, 10)
        assert bb.contains(0, 10)

    def test_contains_outside(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert not bb.contains(-1, 5)
        assert not bb.contains(5, 11)
        assert not bb.contains(11, 5)

    def test_containsx(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert bb.containsx(5)
        assert bb.containsx(0)
        assert not bb.containsx(-1)

    def test_containsy(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert bb.containsy(5)
        assert bb.containsy(0)
        assert not bb.containsy(11)

    def test_fully_contains(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert bb.fully_contains(5, 5)
        # Edge is not fully contained
        assert not bb.fully_contains(0, 5)
        assert not bb.fully_contains(5, 0)

    def test_fully_containsx(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert bb.fully_containsx(5)
        assert not bb.fully_containsx(0)
        assert not bb.fully_containsx(10)

    def test_fully_containsy(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        assert bb.fully_containsy(5)
        assert not bb.fully_containsy(0)
        assert not bb.fully_containsy(10)

    def test_count_contains(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        vertices = [(5, 5), (-1, -1), (3, 7), (11, 11)]
        assert bb.count_contains(vertices) == 2

    def test_count_contains_all(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        vertices = [(1, 1), (5, 5), (9, 9)]
        assert bb.count_contains(vertices) == 3

    def test_count_contains_none(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        vertices = [(-1, -1), (11, 11)]
        assert bb.count_contains(vertices) == 0


# ===================================================================
# Bbox overlaps
# ===================================================================

class TestBboxOverlaps:
    def test_overlaps_true(self):
        bb1 = Bbox.from_extents(0, 0, 10, 10)
        bb2 = Bbox.from_extents(5, 5, 15, 15)
        assert bb1.overlaps(bb2)
        assert bb2.overlaps(bb1)

    def test_overlaps_false(self):
        bb1 = Bbox.from_extents(0, 0, 5, 5)
        bb2 = Bbox.from_extents(10, 10, 15, 15)
        assert not bb1.overlaps(bb2)

    def test_overlaps_adjacent(self):
        bb1 = Bbox.from_extents(0, 0, 5, 5)
        bb2 = Bbox.from_extents(5, 0, 10, 5)
        # Adjacent boxes don't overlap (strict inequality)
        assert not bb1.overlaps(bb2)

    def test_count_overlaps(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        others = [
            Bbox.from_extents(5, 5, 15, 15),
            Bbox.from_extents(20, 20, 30, 30),
            Bbox.from_extents(8, 8, 12, 12),
        ]
        assert bb.count_overlaps(others) == 2


# ===================================================================
# Bbox operations
# ===================================================================

class TestBboxOperations:
    def test_expanded(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        bb2 = bb.expanded(2, 2)
        assert bb2.x0 == approx(-5.0)
        assert bb2.y0 == approx(-5.0)
        assert bb2.x1 == approx(15.0)
        assert bb2.y1 == approx(15.0)

    def test_expanded_identity(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        bb2 = bb.expanded(1, 1)
        assert bb2.x0 == approx(0.0)
        assert bb2.y0 == approx(0.0)
        assert bb2.x1 == approx(10.0)
        assert bb2.y1 == approx(10.0)

    def test_translated(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        bb2 = bb.translated(5, -3)
        assert bb2.x0 == 5
        assert bb2.y0 == -3
        assert bb2.x1 == 15
        assert bb2.y1 == 7

    def test_padded(self):
        bb = Bbox.from_extents(2, 3, 8, 7)
        bb2 = bb.padded(1)
        assert bb2.x0 == 1
        assert bb2.y0 == 2
        assert bb2.x1 == 9
        assert bb2.y1 == 8

    def test_shrunk(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        bb2 = bb.shrunk(0.5, 0.5)
        assert bb2.width == approx(5.0)
        assert bb2.height == approx(5.0)
        assert bb2.x0 == 0
        assert bb2.y0 == 0

    def test_shrunk_to_aspect(self):
        bb = Bbox.from_extents(0, 0, 10, 10)
        # aspect = 2: height should be 2*width
        bb2 = bb.shrunk_to_aspect(2.0)
        assert bb2.height == approx(bb2.width * 2, abs=0.1)

    def test_frozen(self):
        bb = Bbox.from_extents(1, 2, 3, 4)
        frozen = bb.frozen()
        bb.x0 = 10
        assert frozen.x0 == 1  # frozen is independent

    def test_rotated(self):
        bb = Bbox.from_extents(0, 0, 2, 2)
        rotated = bb.rotated(math.pi / 4)
        # Rotated bbox should be larger
        assert rotated.width >= 2.0
        assert rotated.height >= 2.0

    def test_rotated_zero(self):
        bb = Bbox.from_extents(0, 0, 2, 2)
        rotated = bb.rotated(0)
        assert rotated.x0 == approx(0.0)
        assert rotated.y0 == approx(0.0)
        assert rotated.x1 == approx(2.0)
        assert rotated.y1 == approx(2.0)


# ===================================================================
# Bbox anchored
# ===================================================================

class TestBboxAnchored:
    def test_anchored_C(self):
        bb = Bbox.from_bounds(0, 0, 4, 4)
        container = Bbox.from_bounds(0, 0, 10, 10)
        result = bb.anchored('C', container)
        assert result.x0 == approx(3.0)
        assert result.y0 == approx(3.0)

    def test_anchored_SW(self):
        bb = Bbox.from_bounds(0, 0, 4, 4)
        container = Bbox.from_bounds(0, 0, 10, 10)
        result = bb.anchored('SW', container)
        assert result.x0 == approx(0.0)
        assert result.y0 == approx(0.0)

    def test_anchored_NE(self):
        bb = Bbox.from_bounds(0, 0, 4, 4)
        container = Bbox.from_bounds(0, 0, 10, 10)
        result = bb.anchored('NE', container)
        assert result.x0 == approx(6.0)
        assert result.y0 == approx(6.0)

    def test_anchored_tuple(self):
        bb = Bbox.from_bounds(0, 0, 4, 4)
        container = Bbox.from_bounds(0, 0, 10, 10)
        result = bb.anchored((0.5, 0.5), container)
        assert result.x0 == approx(3.0)
        assert result.y0 == approx(3.0)

    def test_anchored_N(self):
        bb = Bbox.from_bounds(0, 0, 4, 4)
        container = Bbox.from_bounds(0, 0, 10, 10)
        result = bb.anchored('N', container)
        assert result.x0 == approx(3.0)
        assert result.y0 == approx(6.0)

    def test_anchored_S(self):
        bb = Bbox.from_bounds(0, 0, 4, 4)
        container = Bbox.from_bounds(0, 0, 10, 10)
        result = bb.anchored('S', container)
        assert result.x0 == approx(3.0)
        assert result.y0 == approx(0.0)


# ===================================================================
# Bbox union and intersection
# ===================================================================

class TestBboxUnionIntersection:
    def test_union_two(self):
        bb1 = Bbox.from_extents(0, 0, 5, 5)
        bb2 = Bbox.from_extents(3, 3, 10, 10)
        result = Bbox.union([bb1, bb2])
        assert result.x0 == 0
        assert result.y0 == 0
        assert result.x1 == 10
        assert result.y1 == 10

    def test_union_single(self):
        bb = Bbox.from_extents(1, 2, 3, 4)
        result = Bbox.union([bb])
        assert result.x0 == 1
        assert result.y0 == 2

    def test_union_empty(self):
        result = Bbox.union([])
        assert result.is_unit()

    def test_union_many(self):
        bboxes = [
            Bbox.from_extents(i, i, i + 2, i + 2) for i in range(5)
        ]
        result = Bbox.union(bboxes)
        assert result.x0 == 0
        assert result.y0 == 0
        assert result.x1 == 6
        assert result.y1 == 6

    def test_intersection_overlap(self):
        bb1 = Bbox.from_extents(0, 0, 10, 10)
        bb2 = Bbox.from_extents(5, 5, 15, 15)
        result = Bbox.intersection(bb1, bb2)
        assert result is not None
        assert result.x0 == 5
        assert result.y0 == 5
        assert result.x1 == 10
        assert result.y1 == 10

    def test_intersection_no_overlap(self):
        bb1 = Bbox.from_extents(0, 0, 5, 5)
        bb2 = Bbox.from_extents(10, 10, 15, 15)
        result = Bbox.intersection(bb1, bb2)
        assert result is None

    def test_intersection_contained(self):
        bb1 = Bbox.from_extents(0, 0, 10, 10)
        bb2 = Bbox.from_extents(2, 2, 8, 8)
        result = Bbox.intersection(bb1, bb2)
        assert result is not None
        assert result.x0 == 2
        assert result.y0 == 2
        assert result.x1 == 8
        assert result.y1 == 8


# ===================================================================
# Bbox repr / eq
# ===================================================================

class TestBboxReprEq:
    def test_repr(self):
        bb = Bbox.from_extents(1.0, 2.0, 3.0, 4.0)
        r = repr(bb)
        assert 'Bbox' in r
        assert '1.0' in r
        assert '4.0' in r

    def test_eq(self):
        bb1 = Bbox.from_extents(1, 2, 3, 4)
        bb2 = Bbox.from_extents(1, 2, 3, 4)
        assert bb1 == bb2

    def test_neq(self):
        bb1 = Bbox.from_extents(1, 2, 3, 4)
        bb2 = Bbox.from_extents(1, 2, 3, 5)
        assert bb1 != bb2

    def test_eq_not_bbox(self):
        bb = Bbox.from_extents(1, 2, 3, 4)
        assert bb != "not a bbox"

    def test_iter(self):
        bb = Bbox.from_extents(1, 2, 5, 8)
        x0, y0, w, h = bb
        assert x0 == 1
        assert y0 == 2
        assert w == 4
        assert h == 6


# ===================================================================
# Bbox update
# ===================================================================

class TestBboxUpdate:
    def test_update_from_data_xy_ignore(self):
        bb = Bbox.from_extents(0, 0, 1, 1)
        bb.update_from_data_xy([(5, 5), (10, 10)], ignore=True)
        assert bb.x0 == 5
        assert bb.y0 == 5
        assert bb.x1 == 10
        assert bb.y1 == 10

    def test_update_from_data_xy_extend(self):
        bb = Bbox.from_extents(0, 0, 5, 5)
        bb.update_from_data_xy([(3, 3), (10, 10)], ignore=False)
        assert bb.x0 == 0
        assert bb.y0 == 0
        assert bb.x1 == 10
        assert bb.y1 == 10

    def test_update_from_data_xy_empty(self):
        bb = Bbox.from_extents(0, 0, 1, 1)
        bb.update_from_data_xy([], ignore=True)
        # No change
        assert bb.x0 == 0


# ===================================================================
# Affine2D basic
# ===================================================================

class TestAffine2DBasic:
    def test_identity(self):
        t = Affine2D()
        assert t.is_identity()

    def test_identity_factory(self):
        t = Affine2D.identity()
        assert t.is_identity()

    def test_not_identity_after_translate(self):
        t = Affine2D().translate(1, 0)
        assert not t.is_identity()

    def test_get_matrix(self):
        t = Affine2D()
        m = t.get_matrix()
        assert m[0][0] == 1
        assert m[1][1] == 1
        assert m[2][2] == 1
        assert m[0][1] == 0

    def test_clear(self):
        t = Affine2D().translate(1, 2)
        assert not t.is_identity()
        t.clear()
        assert t.is_identity()

    def test_frozen(self):
        t = Affine2D().translate(5, 10)
        frozen = t.frozen()
        t.clear()
        assert t.is_identity()
        assert not frozen.is_identity()


# ===================================================================
# Affine2D transforms
# ===================================================================

class TestAffine2DTransforms:
    def test_translate(self):
        t = Affine2D().translate(3, 4)
        result = t.transform_point((0, 0))
        assert result[0] == approx(3.0)
        assert result[1] == approx(4.0)

    def test_scale_uniform(self):
        t = Affine2D().scale(2)
        result = t.transform_point((3, 4))
        assert result[0] == approx(6.0)
        assert result[1] == approx(8.0)

    def test_scale_nonuniform(self):
        t = Affine2D().scale(2, 3)
        result = t.transform_point((1, 1))
        assert result[0] == approx(2.0)
        assert result[1] == approx(3.0)

    def test_rotate_90(self):
        t = Affine2D().rotate(math.pi / 2)
        result = t.transform_point((1, 0))
        assert result[0] == approx(0.0)
        assert result[1] == approx(1.0)

    def test_rotate_180(self):
        t = Affine2D().rotate(math.pi)
        result = t.transform_point((1, 0))
        assert result[0] == approx(-1.0)
        assert result[1] == approx(0.0)

    def test_rotate_deg(self):
        t = Affine2D().rotate_deg(90)
        result = t.transform_point((1, 0))
        assert result[0] == approx(0.0)
        assert result[1] == approx(1.0)

    def test_rotate_around(self):
        t = Affine2D().rotate_around(1, 1, math.pi / 2)
        result = t.transform_point((2, 1))
        assert result[0] == approx(1.0)
        assert result[1] == approx(2.0)

    def test_rotate_deg_around(self):
        t = Affine2D().rotate_deg_around(1, 1, 90)
        result = t.transform_point((2, 1))
        assert result[0] == approx(1.0)
        assert result[1] == approx(2.0)

    def test_skew(self):
        t = Affine2D().skew(math.pi / 4, 0)
        result = t.transform_point((0, 1))
        assert result[0] == approx(1.0)
        assert result[1] == approx(1.0)

    def test_skew_deg(self):
        t = Affine2D().skew_deg(45, 0)
        result = t.transform_point((0, 1))
        assert result[0] == approx(1.0)

    def test_transform_multiple_points(self):
        t = Affine2D().translate(1, 2)
        result = t.transform([(0, 0), (1, 1)])
        assert result[0][0] == approx(1.0)
        assert result[0][1] == approx(2.0)
        assert result[1][0] == approx(2.0)
        assert result[1][1] == approx(3.0)

    def test_transform_empty(self):
        t = Affine2D().translate(1, 2)
        result = t.transform([])
        assert result == []

    def test_chained_transforms(self):
        t = Affine2D().scale(2).translate(1, 0)
        result = t.transform_point((1, 0))
        assert result[0] == approx(3.0)
        assert result[1] == approx(0.0)

    def test_scale_then_translate(self):
        t = Affine2D().scale(3, 3).translate(10, 10)
        result = t.transform_point((1, 1))
        assert result[0] == approx(13.0)
        assert result[1] == approx(13.0)


# ===================================================================
# Affine2D inverse
# ===================================================================

class TestAffine2DInverse:
    def test_inverse_translation(self):
        t = Affine2D().translate(5, 10)
        inv = t.inverted()
        result = inv.transform_point((5, 10))
        assert result[0] == approx(0.0)
        assert result[1] == approx(0.0)

    def test_inverse_scale(self):
        t = Affine2D().scale(4, 2)
        inv = t.inverted()
        result = inv.transform_point((8, 6))
        assert result[0] == approx(2.0)
        assert result[1] == approx(3.0)

    def test_inverse_roundtrip(self):
        t = Affine2D().rotate(0.5).translate(3, 4).scale(2)
        inv = t.inverted()
        pt = (7, 11)
        result = inv.transform_point(t.transform_point(pt))
        assert result[0] == approx(pt[0])
        assert result[1] == approx(pt[1])

    def test_inverse_singular(self):
        t = Affine2D().scale(0, 0)
        with pytest.raises(ValueError):
            t.inverted()

    def test_inverse_identity(self):
        t = Affine2D()
        inv = t.inverted()
        assert inv.is_identity()


# ===================================================================
# Affine2D repr / eq
# ===================================================================

class TestAffine2DReprEq:
    def test_repr(self):
        t = Affine2D()
        r = repr(t)
        assert 'Affine2D' in r

    def test_eq(self):
        t1 = Affine2D().translate(1, 2)
        t2 = Affine2D().translate(1, 2)
        assert t1 == t2

    def test_neq(self):
        t1 = Affine2D().translate(1, 2)
        t2 = Affine2D().translate(1, 3)
        assert t1 != t2

    def test_eq_not_affine(self):
        t = Affine2D()
        assert t != "not affine"


# ===================================================================
# Affine2D set
# ===================================================================

class TestAffine2DSet:
    def test_set_copies(self):
        t1 = Affine2D().translate(1, 2)
        t2 = Affine2D()
        t2.set(t1)
        assert t1 == t2
        # Modifying t1 shouldn't affect t2
        t1.translate(10, 10)
        assert t1 != t2


# ===================================================================
# IdentityTransform
# ===================================================================

class TestIdentityTransform:
    def test_transform(self):
        t = IdentityTransform()
        vals = [(1, 2), (3, 4)]
        assert t.transform(vals) == vals

    def test_inverted(self):
        t = IdentityTransform()
        inv = t.inverted()
        assert isinstance(inv, IdentityTransform)

    def test_repr(self):
        t = IdentityTransform()
        assert repr(t) == "IdentityTransform()"

    def test_is_affine(self):
        t = IdentityTransform()
        assert t.is_affine

    def test_has_inverse(self):
        t = IdentityTransform()
        assert t.has_inverse


# ===================================================================
# BboxTransform
# ===================================================================

class TestBboxTransform:
    def test_basic(self):
        boxin = Bbox.from_extents(0, 0, 1, 1)
        boxout = Bbox.from_extents(0, 0, 100, 200)
        t = BboxTransform(boxin, boxout)
        m = t.get_matrix()
        assert m[0][0] == approx(100.0)
        assert m[1][1] == approx(200.0)

    def test_is_affine(self):
        t = BboxTransform(Bbox.unit(), Bbox.unit())
        assert t.is_affine

    def test_unit_to_unit(self):
        t = BboxTransform(Bbox.unit(), Bbox.unit())
        m = t.get_matrix()
        assert m[0][0] == approx(1.0)
        assert m[1][1] == approx(1.0)
        assert m[0][2] == approx(0.0)
        assert m[1][2] == approx(0.0)


# ===================================================================
# BboxTransformTo / BboxTransformFrom
# ===================================================================

class TestBboxTransformTo:
    def test_basic(self):
        box = Bbox.from_extents(10, 20, 110, 220)
        t = BboxTransformTo(box)
        m = t.get_matrix()
        assert m[0][0] == approx(100.0)
        assert m[1][1] == approx(200.0)
        assert m[0][2] == approx(10.0)
        assert m[1][2] == approx(20.0)


class TestBboxTransformFrom:
    def test_basic(self):
        box = Bbox.from_extents(0, 0, 100, 200)
        t = BboxTransformFrom(box)
        m = t.get_matrix()
        assert m[0][0] == approx(0.01)
        assert m[1][1] == approx(0.005)


# ===================================================================
# ScaledTranslation
# ===================================================================

class TestScaledTranslation:
    def test_basic(self):
        t = ScaledTranslation(5, 10)
        m = t.get_matrix()
        assert m[0][2] == 5
        assert m[1][2] == 10
        assert m[0][0] == 1
        assert m[1][1] == 1


# ===================================================================
# CompositeGenericTransform
# ===================================================================

class TestCompositeTransform:
    def test_composition(self):
        t1 = Affine2D().translate(1, 0)
        t2 = Affine2D().translate(0, 1)
        comp = CompositeGenericTransform(t1, t2)
        result = comp.transform_point((0, 0))
        assert result[0] == approx(1.0)
        assert result[1] == approx(1.0)

    def test_add_operator(self):
        t1 = Affine2D().translate(1, 0)
        t2 = Affine2D().translate(0, 1)
        comp = t1 + t2
        result = comp.transform_point((0, 0))
        assert result[0] == approx(1.0)
        assert result[1] == approx(1.0)

    def test_inverse(self):
        t1 = Affine2D().translate(1, 0)
        t2 = Affine2D().scale(2)
        comp = CompositeGenericTransform(t1, t2)
        inv = comp.inverted()
        pt = (5, 3)
        result = inv.transform_point(comp.transform_point(pt))
        assert result[0] == approx(pt[0])
        assert result[1] == approx(pt[1])


# ===================================================================
# BlendedGenericTransform
# ===================================================================

class TestBlendedTransform:
    def test_blended_factory(self):
        tx = IdentityTransform()
        ty = IdentityTransform()
        result = blended_transform_factory(tx, ty)
        assert isinstance(result, BlendedGenericTransform)


# ===================================================================
# TransformNode
# ===================================================================

class TestTransformNode:
    def test_constants(self):
        assert TransformNode.INVALID_NON_AFFINE == 1
        assert TransformNode.INVALID_AFFINE == 2
        assert TransformNode.INVALID == 3

    def test_is_affine(self):
        n = TransformNode()
        assert not n.is_affine

    def test_is_bbox(self):
        n = TransformNode()
        assert not n.is_bbox

    def test_frozen(self):
        n = TransformNode()
        assert n.frozen() is n


# ===================================================================
# Transform base
# ===================================================================

class TestTransform:
    def test_input_output_dims(self):
        t = Transform()
        assert t.input_dims == 2
        assert t.output_dims == 2

    def test_is_separable(self):
        t = Transform()
        assert not t.is_separable

    def test_has_inverse(self):
        t = Transform()
        assert not t.has_inverse


# ===================================================================
# nonsingular
# ===================================================================

class TestNonsingular:
    def test_equal_zero(self):
        vmin, vmax = nonsingular(0, 0)
        assert vmin < 0
        assert vmax > 0

    def test_equal_nonzero(self):
        vmin, vmax = nonsingular(5, 5)
        assert vmin < 5
        assert vmax > 5

    def test_different(self):
        vmin, vmax = nonsingular(1, 10)
        assert vmin == 1
        assert vmax == 10

    def test_reversed(self):
        vmin, vmax = nonsingular(10, 1, increasing=True)
        assert vmin == 1
        assert vmax == 10

    def test_not_increasing(self):
        vmin, vmax = nonsingular(10, 1, increasing=False)
        assert vmin == 10
        assert vmax == 1


# ===================================================================
# Upstream TestAffine2D — from test_transforms.py
# ===================================================================

class TestAffine2DUpstream:
    """Port of upstream TestAffine2D class from test_transforms.py."""

    single_point = [1.0, 1.0]
    multiple_points = [[0.0, 2.0], [3.0, 3.0], [4.0, 0.0]]
    pivot = single_point

    def test_init(self):
        import numpy as np
        Affine2D([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Affine2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], int))
        Affine2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float))

    def test_values(self):
        import numpy as np
        np.random.seed(19680801)
        values = np.random.random(6)
        np.testing.assert_array_equal(
            Affine2D.from_values(*values).to_values(), values)

    def test_modify_inplace(self):
        import numpy as np
        trans = Affine2D()
        mtx = trans.get_matrix()
        mtx[0, 0] = 42
        np.testing.assert_array_equal(
            trans.get_matrix(), [[42, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_clear(self):
        import numpy as np
        a = Affine2D(np.random.rand(3, 3) + 5)
        a.clear()
        np.testing.assert_array_equal(
            a.get_matrix(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_rotate(self):
        import numpy as np
        r_pi_2 = Affine2D().rotate(np.pi / 2)
        r90 = Affine2D().rotate_deg(90)
        np.testing.assert_array_equal(r_pi_2.get_matrix(), r90.get_matrix())
        np.testing.assert_array_almost_equal(r90.transform(self.single_point), [-1, 1])
        np.testing.assert_array_almost_equal(
            r90.transform(self.multiple_points), [[-2, 0], [-3, 3], [0, 4]])

        r_pi = Affine2D().rotate(np.pi)
        r180 = Affine2D().rotate_deg(180)
        np.testing.assert_array_equal(r_pi.get_matrix(), r180.get_matrix())
        np.testing.assert_array_almost_equal(r180.transform(self.single_point), [-1, -1])
        np.testing.assert_array_almost_equal(
            r180.transform(self.multiple_points), [[0, -2], [-3, -3], [-4, 0]])

        r_pi_3_2 = Affine2D().rotate(3 * np.pi / 2)
        r270 = Affine2D().rotate_deg(270)
        np.testing.assert_array_equal(r_pi_3_2.get_matrix(), r270.get_matrix())
        np.testing.assert_array_almost_equal(r270.transform(self.single_point), [1, -1])
        np.testing.assert_array_almost_equal(
            r270.transform(self.multiple_points), [[2, 0], [3, -3], [0, -4]])

        np.testing.assert_array_equal(
            (r90 + r90).get_matrix(), r180.get_matrix())
        np.testing.assert_array_equal(
            (r90 + r180).get_matrix(), r270.get_matrix())

    def test_rotate_around(self):
        import numpy as np
        r_pi_2 = Affine2D().rotate_around(*self.pivot, np.pi / 2)
        r90 = Affine2D().rotate_deg_around(*self.pivot, 90)
        np.testing.assert_array_equal(r_pi_2.get_matrix(), r90.get_matrix())
        np.testing.assert_array_almost_equal(r90.transform(self.single_point), [1, 1])
        np.testing.assert_array_almost_equal(
            r90.transform(self.multiple_points), [[0, 0], [-1, 3], [2, 4]])

        r_pi = Affine2D().rotate_around(*self.pivot, np.pi)
        r180 = Affine2D().rotate_deg_around(*self.pivot, 180)
        np.testing.assert_array_equal(r_pi.get_matrix(), r180.get_matrix())
        np.testing.assert_array_almost_equal(r180.transform(self.single_point), [1, 1])
        np.testing.assert_array_almost_equal(
            r180.transform(self.multiple_points), [[2, 0], [-1, -1], [-2, 2]])

    def test_scale(self):
        import numpy as np
        sx = Affine2D().scale(3, 1)
        sy = Affine2D().scale(1, -2)
        trans = Affine2D().scale(3, -2)
        np.testing.assert_array_equal((sx + sy).get_matrix(), trans.get_matrix())
        np.testing.assert_array_equal(trans.transform(self.single_point), [3, -2])
        np.testing.assert_array_equal(
            trans.transform(self.multiple_points), [[0, -4], [9, -6], [12, 0]])

    def test_skew(self):
        import numpy as np
        trans_rad = Affine2D().skew(np.pi / 8, np.pi / 12)
        trans_deg = Affine2D().skew_deg(22.5, 15)
        np.testing.assert_array_equal(trans_rad.get_matrix(), trans_deg.get_matrix())
        trans = Affine2D().skew_deg(26.5650512, 14.0362435)
        np.testing.assert_array_almost_equal(trans.transform(self.single_point), [1.5, 1.25])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points), [[1, 2], [4.5, 3.75], [4, 1]])

    def test_translate(self):
        import numpy as np
        tx = Affine2D().translate(23, 0)
        ty = Affine2D().translate(0, 42)
        trans = Affine2D().translate(23, 42)
        np.testing.assert_array_equal((tx + ty).get_matrix(), trans.get_matrix())
        np.testing.assert_array_equal(trans.transform(self.single_point), [24, 43])
        np.testing.assert_array_equal(
            trans.transform(self.multiple_points), [[23, 44], [26, 45], [27, 42]])

    def test_rotate_plus_other(self):
        import numpy as np
        trans = Affine2D().rotate_deg(90).rotate_deg_around(*self.pivot, 180)
        trans_added = (Affine2D().rotate_deg(90) +
                       Affine2D().rotate_deg_around(*self.pivot, 180))
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_almost_equal(trans.transform(self.single_point), [3, 1])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points), [[4, 2], [5, -1], [2, -2]])

        trans = Affine2D().rotate_deg(90).scale(3, -2)
        trans_added = Affine2D().rotate_deg(90) + Affine2D().scale(3, -2)
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_almost_equal(
            trans.transform(self.single_point), [-3, -2])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points), [[-6, 0], [-9, -6], [0, -8]])

        trans = Affine2D().rotate_deg(90).translate(23, 42)
        trans_added = Affine2D().rotate_deg(90) + Affine2D().translate(23, 42)
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_almost_equal(
            trans.transform(self.single_point), [22, 43])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points), [[21, 42], [20, 45], [23, 46]])

    def test_scale_plus_other(self):
        import numpy as np
        trans = Affine2D().scale(3, -2).rotate_deg(90)
        trans_added = Affine2D().scale(3, -2) + Affine2D().rotate_deg(90)
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_equal(trans.transform(self.single_point), [2, 3])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points), [[4, 0], [6, 9], [0, 12]])

        trans = Affine2D().scale(3, -2).translate(23, 42)
        trans_added = Affine2D().scale(3, -2) + Affine2D().translate(23, 42)
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_equal(trans.transform(self.single_point), [26, 40])
        np.testing.assert_array_equal(
            trans.transform(self.multiple_points), [[23, 38], [32, 36], [35, 42]])

    def test_skew_plus_other(self):
        import numpy as np
        trans = Affine2D().skew_deg(26.5650512, 14.0362435).translate(23, 42)
        trans_added = (Affine2D().skew_deg(26.5650512, 14.0362435) +
                       Affine2D().translate(23, 42))
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_almost_equal(
            trans.transform(self.single_point), [24.5, 43.25])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points),
            [[24, 44], [27.5, 45.75], [27, 43]])

    def test_translate_plus_other(self):
        import numpy as np
        trans = Affine2D().translate(23, 42).rotate_deg(90)
        trans_added = Affine2D().translate(23, 42) + Affine2D().rotate_deg(90)
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_almost_equal(
            trans.transform(self.single_point), [-43, 24])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points),
            [[-44, 23], [-45, 26], [-42, 27]])

        trans = Affine2D().translate(23, 42).scale(3, -2)
        trans_added = Affine2D().translate(23, 42) + Affine2D().scale(3, -2)
        np.testing.assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        np.testing.assert_array_almost_equal(
            trans.transform(self.single_point), [72, -86])
        np.testing.assert_array_almost_equal(
            trans.transform(self.multiple_points), [[69, -88], [78, -90], [81, -84]])


# ===================================================================
# Upstream standalone transform tests
# ===================================================================

def test_Affine2D_from_values():
    """Upstream test_Affine2D_from_values: from_values creates correct transform."""
    import numpy as np
    from numpy.testing import assert_almost_equal
    from matplotlib.transforms import Affine2D

    points = np.array([[0, 0], [10, 20], [-1, 0]])

    t = Affine2D.from_values(1, 0, 0, 0, 0, 0)
    assert_almost_equal(t.transform(points), [[0, 0], [10, 0], [-1, 0]])

    t = Affine2D.from_values(0, 2, 0, 0, 0, 0)
    assert_almost_equal(t.transform(points), [[0, 0], [0, 20], [0, -2]])

    t = Affine2D.from_values(0, 0, 3, 0, 0, 0)
    assert_almost_equal(t.transform(points), [[0, 0], [60, 0], [0, 0]])

    t = Affine2D.from_values(0, 0, 0, 4, 0, 0)
    assert_almost_equal(t.transform(points), [[0, 0], [0, 80], [0, 0]])

    t = Affine2D.from_values(0, 0, 0, 0, 5, 0)
    assert_almost_equal(t.transform(points), [[5, 0], [5, 0], [5, 0]])

    t = Affine2D.from_values(0, 0, 0, 0, 0, 6)
    assert_almost_equal(t.transform(points), [[0, 6], [0, 6], [0, 6]])


def test_affine_inverted_invalidated():
    """Upstream test_affine_inverted_invalidated: inverse stays valid after translate."""
    from numpy.testing import assert_almost_equal
    from matplotlib.transforms import Affine2D

    point = [1.0, 1.0]
    t = Affine2D()

    result = t.transform(t.inverted().transform(point))
    assert_almost_equal(point, result)

    t.translate(1.0, 1.0).get_matrix()
    result = t.transform(t.inverted().transform(point))
    assert_almost_equal(point, result)


# ===================================================================
# Additional transform tests (upstream-inspired batch, round 2)
# ===================================================================

import pytest
import numpy as np
from matplotlib.transforms import Affine2D, BboxTransform, Bbox


class TestAffine2DOperations:
    """Tests for Affine2D transform composition."""

    def test_identity_transform(self):
        t = Affine2D()
        pt = [3.0, 4.0]
        result = t.transform(pt)
        np.testing.assert_allclose(result, pt, atol=1e-10)

    def test_translate(self):
        t = Affine2D().translate(5, 3)
        result = t.transform([1.0, 2.0])
        np.testing.assert_allclose(result, [6.0, 5.0], atol=1e-10)

    def test_scale(self):
        t = Affine2D().scale(2.0, 3.0)
        result = t.transform([1.0, 1.0])
        np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-10)

    def test_rotate_90(self):
        t = Affine2D().rotate_deg(90)
        result = t.transform([1.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-8)

    def test_rotate_180(self):
        t = Affine2D().rotate_deg(180)
        result = t.transform([1.0, 0.0])
        np.testing.assert_allclose(result, [-1.0, 0.0], atol=1e-8)

    def test_translate_then_scale(self):
        t = Affine2D().translate(1.0, 0.0).scale(2.0, 2.0)
        result = t.transform([0.0, 0.0])
        np.testing.assert_allclose(result, [2.0, 0.0], atol=1e-10)

    def test_inverse_translate(self):
        t = Affine2D().translate(5, 3)
        inv = t.inverted()
        pt = [6.0, 5.0]
        result = inv.transform(pt)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_inverse_scale(self):
        t = Affine2D().scale(2.0, 4.0)
        inv = t.inverted()
        result = inv.transform([4.0, 8.0])
        np.testing.assert_allclose(result, [2.0, 2.0], atol=1e-10)

    @pytest.mark.parametrize('angle', [0, 45, 90, 135, 180, 270])
    def test_rotate_round_trip(self, angle):
        t = Affine2D().rotate_deg(angle)
        pt = [2.0, 3.0]
        result = t.inverted().transform(t.transform(pt))
        np.testing.assert_allclose(result, pt, atol=1e-8)

    def test_array_transform(self):
        t = Affine2D().translate(1, 2)
        pts = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        result = t.transform(pts)
        expected = np.array([[1, 2], [2, 3], [3, 4]], dtype=float)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestBboxTransformOps:
    """Tests for BboxTransform."""

    def test_bbox_transform_basic(self):
        src = Bbox([[0, 0], [1, 1]])
        dst = Bbox([[0, 0], [100, 100]])
        t = BboxTransform(src, dst)
        result = t.transform([0.5, 0.5])
        np.testing.assert_allclose(result, [50.0, 50.0], atol=1e-8)

    def test_bbox_transform_origin(self):
        src = Bbox([[0, 0], [1, 1]])
        dst = Bbox([[10, 20], [110, 120]])
        t = BboxTransform(src, dst)
        result = t.transform([0.0, 0.0])
        np.testing.assert_allclose(result, [10.0, 20.0], atol=1e-8)

    def test_bbox_transform_far_corner(self):
        src = Bbox([[0, 0], [1, 1]])
        dst = Bbox([[0, 0], [200, 100]])
        t = BboxTransform(src, dst)
        result = t.transform([1.0, 1.0])
        np.testing.assert_allclose(result, [200.0, 100.0], atol=1e-8)
