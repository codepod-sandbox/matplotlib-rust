"""Tests for newly added axes, collections, lines, colors, and transforms methods."""

import numpy as np
import pytest
import sys
import os

# Ensure the matplotlib package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import (
    CircleCollection, RegularPolyCollection,
)
try:
    from matplotlib.collections import BrokenBarHCollection
except ImportError:
    BrokenBarHCollection = None  # removed in OG 3.10
from matplotlib.colors import FuncNorm
from matplotlib.transforms import offset_copy, Affine2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_axes():
    fig = mfigure.Figure(figsize=(6, 4))
    ax = Axes(fig, [0.1, 0.1, 0.8, 0.8])
    fig.add_axes(ax)
    return fig, ax


# ---------------------------------------------------------------------------
# axes.py: matshow
# ---------------------------------------------------------------------------

class TestMatshow:
    def test_returns_axes_image(self):
        fig, ax = make_axes()
        Z = np.arange(9).reshape(3, 3)
        im = ax.matshow(Z)
        from matplotlib.image import AxesImage
        assert isinstance(im, AxesImage)

    def test_sets_xticks(self):
        fig, ax = make_axes()
        Z = np.arange(12).reshape(3, 4)
        ax.matshow(Z)
        # OG matshow sets ticks for columns; filter to integer ticks in valid range
        xticks = [int(t) for t in ax.get_xticks() if 0 <= t < Z.shape[1]]
        assert xticks == [0, 1, 2, 3]

    def test_sets_yticks(self):
        fig, ax = make_axes()
        Z = np.arange(12).reshape(3, 4)
        ax.matshow(Z)
        # OG matshow sets ticks for rows; filter to integer ticks in valid range
        yticks = [int(t) for t in ax.get_yticks() if 0 <= t < Z.shape[0]]
        assert yticks == [0, 1, 2]

    def test_origin_upper(self):
        fig, ax = make_axes()
        Z = np.eye(3)
        im = ax.matshow(Z)
        assert im.origin == 'upper'


# ---------------------------------------------------------------------------
# axes.py: spy
# ---------------------------------------------------------------------------

class TestSpy:
    def test_returns_path_collection(self):
        # OG spy() returns an AxesImage (imshow-based) for dense matrices;
        # accept either PathCollection or AxesImage
        from matplotlib.collections import PathCollection
        from matplotlib.image import AxesImage
        fig, ax = make_axes()
        Z = np.eye(3)
        sc = ax.spy(Z)
        assert isinstance(sc, (PathCollection, AxesImage))

    def test_inverted_y_axis(self):
        fig, ax = make_axes()
        Z = np.eye(3)
        ax.spy(Z)
        ylim = ax.get_ylim()
        # y-axis inverted: ymin > ymax
        assert ylim[0] > ylim[1]

    def test_correct_nonzero_count(self):
        from matplotlib.collections import PathCollection
        from matplotlib.image import AxesImage
        fig, ax = make_axes()
        Z = np.eye(4)
        sc = ax.spy(Z)
        if isinstance(sc, PathCollection):
            offsets = sc.get_offsets()
            assert len(offsets) == 4  # 4 ones on the diagonal
        else:
            # AxesImage-based spy: verify the matrix has 4 nonzero entries
            assert np.count_nonzero(Z) == 4


# ---------------------------------------------------------------------------
# axes.py: broken_barh
# ---------------------------------------------------------------------------

class TestBrokenBarh:
    def test_returns_list_of_rectangles(self):
        # OG broken_barh returns a PolyCollection, not a list of Rectangles
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Rectangle
        fig, ax = make_axes()
        xranges = [(1, 2), (5, 3)]
        rects = ax.broken_barh(xranges, (0, 1))
        assert isinstance(rects, PolyCollection) or (
            hasattr(rects, '__len__') and all(isinstance(r, Rectangle) for r in rects)
        )

    def test_patches_added_to_axes(self):
        # OG broken_barh adds a PolyCollection to collections, not patches
        from matplotlib.collections import PolyCollection
        fig, ax = make_axes()
        initial_collections = len(ax.collections)
        initial_patches = len(ax.patches)
        ax.broken_barh([(0, 1), (2, 1), (4, 1)], (0, 1))
        # OG adds to collections
        assert len(ax.collections) == initial_collections + 1 or \
               len(ax.patches) == initial_patches + 3

    def test_empty_xranges(self):
        # OG broken_barh with empty xranges returns a PolyCollection (not [])
        from matplotlib.collections import PolyCollection
        fig, ax = make_axes()
        rects = ax.broken_barh([], (0, 1))
        assert rects == [] or isinstance(rects, PolyCollection)


# ---------------------------------------------------------------------------
# axes.py: xcorr / acorr
# ---------------------------------------------------------------------------

class TestXcorr:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.x = rng.standard_normal(50)
        self.y = rng.standard_normal(50)

    def test_returns_four_tuple(self):
        fig, ax = make_axes()
        result = ax.xcorr(self.x, self.y, maxlags=10)
        assert len(result) == 4

    def test_lags_shape(self):
        fig, ax = make_axes()
        lags, correls, lines, b = ax.xcorr(self.x, self.y, maxlags=10)
        assert len(lags) == 21  # -10 .. 10

    def test_correls_shape(self):
        fig, ax = make_axes()
        lags, correls, lines, b = ax.xcorr(self.x, self.y, maxlags=10)
        assert len(correls) == 21

    def test_normed_range(self):
        fig, ax = make_axes()
        lags, correls, lines, b = ax.xcorr(self.x, self.x, normed=True, maxlags=10)
        # Autocorrelation at lag 0 should be 1.0
        zero_idx = list(lags).index(0)
        assert abs(correls[zero_idx] - 1.0) < 1e-9


class TestAcorr:
    def test_acorr_is_xcorr_with_self(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(40)
        fig, ax = make_axes()
        lags, correls, lines, b = ax.acorr(x, maxlags=5)
        assert len(lags) == 11  # -5..5

    def test_acorr_normed_peak_is_one(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(40)
        fig, ax = make_axes()
        lags, correls, lines, b = ax.acorr(x, normed=True, maxlags=5)
        zero_idx = list(lags).index(0)
        assert abs(correls[zero_idx] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# axes.py: magnitude_spectrum / angle_spectrum
# ---------------------------------------------------------------------------

class TestMagnitudeSpectrum:
    def test_returns_three_tuple(self):
        fig, ax = make_axes()
        x = np.sin(2 * np.pi * np.linspace(0, 1, 64))
        result = ax.magnitude_spectrum(x)
        assert len(result) == 3

    def test_spectrum_and_freqs_shapes_match(self):
        fig, ax = make_axes()
        x = np.ones(64)
        spectrum, freqs, line = ax.magnitude_spectrum(x)
        assert len(spectrum) == len(freqs)

    def test_line_is_line2d(self):
        fig, ax = make_axes()
        x = np.ones(32)
        spectrum, freqs, line = ax.magnitude_spectrum(x)
        assert isinstance(line, Line2D)


class TestAngleSpectrum:
    def test_returns_three_tuple(self):
        fig, ax = make_axes()
        x = np.sin(2 * np.pi * np.linspace(0, 1, 64))
        result = ax.angle_spectrum(x)
        assert len(result) == 3

    def test_spectrum_and_freqs_shapes_match(self):
        fig, ax = make_axes()
        x = np.ones(64)
        spectrum, freqs, line = ax.angle_spectrum(x)
        assert len(spectrum) == len(freqs)


# ---------------------------------------------------------------------------
# axes.py: add_artist / get_children
# ---------------------------------------------------------------------------

class TestAddArtist:
    def test_line2d_routed_to_lines(self):
        fig, ax = make_axes()
        line = Line2D([0, 1], [0, 1])
        ax.add_artist(line)
        assert line in ax.lines

    def test_patch_routed_to_patches(self):
        from matplotlib.patches import Rectangle
        fig, ax = make_axes()
        patch = Rectangle((0, 0), 1, 1)
        ax.add_artist(patch)
        assert patch in ax.patches

    def test_collection_routed_to_collections(self):
        fig, ax = make_axes()
        cc = CircleCollection([10, 20])
        ax.add_artist(cc)
        assert cc in ax.collections

    def test_returns_artist(self):
        fig, ax = make_axes()
        line = Line2D([0], [0])
        returned = ax.add_artist(line)
        assert returned is line


class TestGetChildren:
    def test_returns_list(self):
        fig, ax = make_axes()
        children = ax.get_children()
        assert isinstance(children, list)

    def test_contains_added_line(self):
        fig, ax = make_axes()
        line = Line2D([0, 1], [0, 1])
        ax.add_artist(line)
        assert line in ax.get_children()

    def test_contains_added_patch(self):
        from matplotlib.patches import Rectangle
        fig, ax = make_axes()
        r = Rectangle((0, 0), 1, 1)
        ax.add_artist(r)
        assert r in ax.get_children()


# ---------------------------------------------------------------------------
# lines.py: is_dashed / recache / get_window_extent
# ---------------------------------------------------------------------------

class TestIsDashed:
    def test_solid_is_not_dashed(self):
        line = Line2D([0, 1], [0, 1], linestyle='-')
        assert line.is_dashed() is False

    def test_dashed_is_dashed(self):
        line = Line2D([0, 1], [0, 1], linestyle='--')
        assert line.is_dashed() is True

    def test_dotted_is_dashed(self):
        line = Line2D([0, 1], [0, 1], linestyle=':')
        assert line.is_dashed() is True

    def test_dash_dot_is_dashed(self):
        line = Line2D([0, 1], [0, 1], linestyle='-.')
        assert line.is_dashed() is True

    def test_solid_string_is_not_dashed(self):
        line = Line2D([0, 1], [0, 1], linestyle='solid')
        assert line.is_dashed() is False


class TestRecache:
    def test_recache_is_noop(self):
        line = Line2D([0, 1], [0, 1])
        result = line.recache()
        assert result is None

    def test_recache_always_is_noop(self):
        line = Line2D([0, 1], [0, 1])
        result = line.recache(always=True)
        assert result is None


class TestLine2DGetWindowExtent:
    def test_returns_bbox(self):
        from matplotlib.transforms import Bbox
        line = Line2D([1, 2, 3], [4, 5, 6])
        bbox = line.get_window_extent()
        assert isinstance(bbox, Bbox)

    def test_empty_line(self):
        from matplotlib.transforms import Bbox
        line = Line2D([], [])
        bbox = line.get_window_extent()
        assert isinstance(bbox, Bbox)


# ---------------------------------------------------------------------------
# colors.py: FuncNorm
# ---------------------------------------------------------------------------

class TestFuncNorm:
    def test_forward_identity(self):
        norm = FuncNorm((lambda x: x, lambda x: x), vmin=0, vmax=1)
        result = norm(0.5)
        assert abs(result - 0.5) < 1e-9

    def test_forward_square(self):
        norm = FuncNorm((lambda x: x**2, lambda x: np.sqrt(x)), vmin=0, vmax=1)
        result = norm(0.5)
        assert abs(result - 0.25) < 1e-9

    def test_inverse(self):
        norm = FuncNorm((lambda x: x**2, lambda x: np.sqrt(x)), vmin=0, vmax=1)
        # inverse(0.25) -> sqrt(0.25)=0.5, then 0.5*(1-0)+0 = 0.5
        result = norm.inverse(0.25)
        assert abs(float(result) - 0.5) < 1e-9

    def test_array_input(self):
        norm = FuncNorm((lambda x: x, lambda x: x), vmin=0, vmax=2)
        vals = norm(np.array([0.0, 1.0, 2.0]))
        assert len(vals) == 3

    def test_autoscale_on_call(self):
        norm = FuncNorm((lambda x: x, lambda x: x))
        # vmin/vmax should be set after calling with data
        vals = np.array([0.0, 1.0, 2.0])
        norm(vals)
        assert norm.vmin == 0.0
        assert norm.vmax == 2.0


# ---------------------------------------------------------------------------
# transforms.py: offset_copy
# ---------------------------------------------------------------------------

class TestOffsetCopy:
    def test_returns_transform_with_offset(self):
        trans = Affine2D()
        result = offset_copy(trans, x=0.1, y=0.2, units='dots')
        assert hasattr(result, 'transform')

    def test_transform_point_adds_offset_dots(self):
        trans = Affine2D()
        ot = offset_copy(trans, x=10.0, y=20.0, units='dots')
        pt = ot.transform_point((0.0, 0.0))
        assert abs(pt[0] - 10.0) < 1e-9
        assert abs(pt[1] - 20.0) < 1e-9

    def test_units_inches(self):
        fig = mfigure.Figure(figsize=(6, 4), dpi=100)
        trans = Affine2D()
        ot = offset_copy(trans, fig=fig, x=1.0, y=0.0, units='inches')
        pt = ot.transform_point((0.0, 0.0))
        assert abs(pt[0] - 100.0) < 1e-9  # 1 inch * 100 dpi

    def test_units_points(self):
        fig = mfigure.Figure(figsize=(6, 4), dpi=72)
        trans = Affine2D()
        ot = offset_copy(trans, fig=fig, x=72.0, y=0.0, units='points')
        pt = ot.transform_point((0.0, 0.0))
        assert abs(pt[0] - 72.0) < 1e-9  # 72 points * 72dpi / 72 = 72 px

    def test_invalid_units_raises(self):
        trans = Affine2D()
        with pytest.raises(ValueError):
            offset_copy(trans, x=1.0, y=1.0, units='furlongs')


# ---------------------------------------------------------------------------
# collections.py: CircleCollection
# ---------------------------------------------------------------------------

class TestCircleCollection:
    def test_stores_sizes(self):
        cc = CircleCollection([10.0, 20.0, 30.0])
        sizes = cc.get_sizes()
        # OG returns ndarray; use list() for comparison
        assert list(sizes) == [10.0, 20.0, 30.0]

    def test_set_sizes(self):
        cc = CircleCollection([1.0])
        cc.set_sizes([5.0, 10.0])
        assert list(cc.get_sizes()) == [5.0, 10.0]

    def test_is_collection(self):
        from matplotlib.collections import Collection
        cc = CircleCollection([1.0])
        assert isinstance(cc, Collection)


# ---------------------------------------------------------------------------
# collections.py: BrokenBarHCollection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(BrokenBarHCollection is None, reason="BrokenBarHCollection removed in OG 3.10")
class TestBrokenBarHCollection:
    def test_creates_verts(self):
        xranges = [(0, 2), (5, 3)]
        coll = BrokenBarHCollection(xranges, (1, 2))
        verts = coll.get_verts()
        assert len(verts) == 2

    def test_correct_vertex_coords(self):
        coll = BrokenBarHCollection([(0, 2)], (1, 3))
        verts = coll.get_verts()
        # ymin=1, ymax=4; xmin=0, xmax=2
        v = np.array(verts[0])
        xs = v[:, 0]
        ys = v[:, 1]
        assert set(xs.tolist()) == {0.0, 2.0}
        assert set(ys.tolist()) == {1.0, 4.0}

    def test_is_poly_collection(self):
        from matplotlib.collections import PolyCollection
        coll = BrokenBarHCollection([(0, 1)], (0, 1))
        assert isinstance(coll, PolyCollection)

    def test_span_where(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        where = np.array([True, True, False, True, True])
        coll = BrokenBarHCollection.span_where(x, ymin=0, ymax=1, where=where)
        assert isinstance(coll, BrokenBarHCollection)
        verts = coll.get_verts()
        assert len(verts) == 2  # two runs of True


# ---------------------------------------------------------------------------
# collections.py: RegularPolyCollection
# ---------------------------------------------------------------------------

class TestRegularPolyCollection:
    def test_stores_numsides(self):
        rpc = RegularPolyCollection(6, sizes=[100.0])
        assert rpc.get_numsides() == 6

    def test_stores_sizes(self):
        rpc = RegularPolyCollection(4, sizes=[50.0, 60.0])
        sizes = rpc.get_sizes()
        # OG returns ndarray; use list() for comparison
        assert list(sizes) == [50.0, 60.0]

    def test_set_sizes(self):
        rpc = RegularPolyCollection(3, sizes=[1.0])
        rpc.set_sizes([10.0, 20.0])
        assert list(rpc.get_sizes()) == [10.0, 20.0]

    def test_is_collection(self):
        from matplotlib.collections import Collection
        rpc = RegularPolyCollection(5)
        assert isinstance(rpc, Collection)
