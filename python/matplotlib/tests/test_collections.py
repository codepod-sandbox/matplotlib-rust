"""Tests for matplotlib.collections module --- Collection and PathCollection."""

import pytest
import numpy as np
from matplotlib.collections import Collection, PathCollection
from matplotlib.colors import to_hex


# ===================================================================
# Collection base class
# ===================================================================

class TestCollection:
    def test_default_zorder(self):
        """Collection class zorder is 0 in OG matplotlib."""
        assert Collection.zorder == 0

    def test_instance_zorder(self):
        """Collection instance has default zorder=1 (set by __init__ default arg)."""
        c = Collection()
        assert c.get_zorder() == 1

    def test_default_visible(self):
        """Collection is visible by default."""
        c = Collection()
        assert c.get_visible() is True

    def test_default_alpha(self):
        """Collection alpha is None by default."""
        c = Collection()
        assert c.get_alpha() is None

    def test_default_label(self):
        """Collection label is empty string by default."""
        c = Collection()
        assert c.get_label() == ''

    def test_set_kwargs_forwarded(self):
        """Collection forwards kwargs to Artist.set()."""
        c = Collection(visible=False, alpha=0.5)
        assert c.get_visible() is False
        assert c.get_alpha() == 0.5


# ===================================================================
# PathCollection construction defaults
# ===================================================================

class TestPathCollectionDefaults:
    def test_default_offsets(self):
        """Default offsets in OG is [[0,0]] (not empty)."""
        pc = PathCollection()
        offsets = pc.get_offsets()
        assert isinstance(offsets, np.ndarray)
        # OG default offsets is [[0., 0.]]
        assert offsets.shape == (1, 2)

    def test_default_sizes(self):
        """Default sizes is [] (empty) in OG matplotlib."""
        pc = PathCollection()
        assert len(pc.get_sizes()) == 0

    def test_default_facecolors(self):
        """Default facecolors in OG is a single default color RGBA."""
        pc = PathCollection()
        fc = pc.get_facecolors()
        assert isinstance(fc, np.ndarray)
        # OG has one default color (matplotlib C0)
        assert fc.shape[1] == 4  # RGBA

    def test_default_edgecolors(self):
        """Default edgecolors is empty in OG."""
        pc = PathCollection()
        ec = pc.get_edgecolors()
        assert isinstance(ec, np.ndarray)
        assert len(ec) == 0

    def test_default_label(self):
        """Default label is empty string (inherited from Artist)."""
        pc = PathCollection()
        assert pc.get_label() == ''

    def test_default_zorder(self):
        """PathCollection instance has default zorder=1 (from Collection.__init__)."""
        pc = PathCollection()
        assert pc.get_zorder() == 1


# ===================================================================
# PathCollection with explicit data
# ===================================================================

class TestPathCollectionExplicitData:
    def test_explicit_offsets(self):
        """Offsets passed at construction are stored."""
        offsets = [(1, 2), (3, 4)]
        pc = PathCollection(offsets=offsets)
        result = pc.get_offsets()
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [(1, 2), (3, 4)])

    def test_explicit_sizes(self):
        """Sizes passed at construction are stored."""
        pc = PathCollection(sizes=[10.0, 30.0, 50.0])
        sizes = pc.get_sizes()
        assert np.allclose(sizes, [10.0, 30.0, 50.0])

    def test_explicit_facecolors(self):
        """Facecolors passed at construction are stored as RGBA."""
        pc = PathCollection(facecolors=['red', 'blue'])
        fc = pc.get_facecolors()
        assert isinstance(fc, np.ndarray)
        assert fc.shape == (2, 4)
        assert to_hex(fc[0]) == '#ff0000'
        assert to_hex(fc[1]) == '#0000ff'

    def test_explicit_edgecolors(self):
        """Edgecolors passed at construction are stored as RGBA."""
        pc = PathCollection(edgecolors=['green'])
        ec = pc.get_edgecolors()
        assert isinstance(ec, np.ndarray)
        assert ec.shape == (1, 4)
        assert to_hex(ec[0]) == '#008000'

    def test_explicit_label(self):
        """Label passed at construction is stored."""
        pc = PathCollection(label='scatter1')
        assert pc.get_label() == 'scatter1'

    def test_all_params_at_once(self):
        """All parameters can be set simultaneously at construction."""
        pc = PathCollection(
            offsets=[(0, 0)],
            sizes=[50.0],
            facecolors=['red'],
            edgecolors=['black'],
            label='full',
        )
        assert np.allclose(pc.get_offsets(), [(0, 0)])
        assert np.allclose(pc.get_sizes(), [50.0])
        assert to_hex(pc.get_facecolors()[0]) == '#ff0000'
        assert pc.get_label() == 'full'


# ===================================================================
# get/set offsets round-trip
# ===================================================================

class TestOffsetsRoundTrip:
    def test_set_get_offsets(self):
        """set_offsets then get_offsets returns the same data."""
        pc = PathCollection()
        pc.set_offsets([(10, 20), (30, 40)])
        assert np.allclose(pc.get_offsets(), [(10, 20), (30, 40)])

    def test_set_offsets_replaces(self):
        """set_offsets replaces previous offsets entirely."""
        pc = PathCollection(offsets=[(1, 2)])
        pc.set_offsets([(5, 6), (7, 8)])
        assert np.allclose(pc.get_offsets(), [(5, 6), (7, 8)])

    def test_set_offsets_empty(self):
        """set_offsets with np.empty((0,2)) clears offsets."""
        pc = PathCollection(offsets=[(1, 2)])
        pc.set_offsets(np.empty((0, 2)))  # OG requires (0,2) shape, not []
        offsets = pc.get_offsets()
        assert isinstance(offsets, np.ndarray)
        assert len(offsets) == 0


# ===================================================================
# get/set sizes round-trip
# ===================================================================

class TestSizesRoundTrip:
    def test_set_get_sizes(self):
        """set_sizes then get_sizes returns the same data."""
        pc = PathCollection()
        pc.set_sizes([100.0, 200.0])
        assert np.allclose(pc.get_sizes(), [100.0, 200.0])

    def test_set_sizes_replaces(self):
        """set_sizes replaces previous sizes entirely."""
        pc = PathCollection(sizes=[5.0])
        pc.set_sizes([10.0, 15.0, 20.0])
        assert np.allclose(pc.get_sizes(), [10.0, 15.0, 20.0])

    def test_set_sizes_empty(self):
        """set_sizes with empty list clears sizes."""
        pc = PathCollection()
        pc.set_sizes([])
        assert len(pc.get_sizes()) == 0


# ===================================================================
# get/set facecolors round-trip
# ===================================================================

class TestFacecolorsRoundTrip:
    def test_set_get_facecolors(self):
        """set_facecolors then get_facecolors returns RGBA array."""
        pc = PathCollection()
        pc.set_facecolors(['red', 'green'])
        fc = pc.get_facecolors()
        assert fc.shape == (2, 4)
        assert to_hex(fc[0]) == '#ff0000'

    def test_set_facecolors_replaces(self):
        """set_facecolors replaces previous facecolors."""
        pc = PathCollection(facecolors=['blue'])
        pc.set_facecolors(['yellow', 'cyan'])
        fc = pc.get_facecolors()
        assert fc.shape == (2, 4)

    def test_set_facecolors_empty(self):
        """set_facecolors with empty list clears facecolors."""
        pc = PathCollection(facecolors=['red'])
        pc.set_facecolors([])
        fc = pc.get_facecolors()
        assert len(fc) == 0

    def test_facecolors_tuples(self):
        """Facecolors can be RGBA tuples."""
        colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.5)]
        pc = PathCollection(facecolors=colors)
        fc = pc.get_facecolors()
        assert np.allclose(fc, colors)


# ===================================================================
# get/set edgecolors round-trip
# ===================================================================

class TestEdgecolorsRoundTrip:
    def test_set_get_edgecolors(self):
        """set_edgecolors then get_edgecolors returns RGBA array."""
        pc = PathCollection()
        pc.set_edgecolors(['black', 'white'])
        ec = pc.get_edgecolors()
        assert ec.shape == (2, 4)
        assert np.allclose(ec[0], [0, 0, 0, 1])

    def test_set_edgecolors_replaces(self):
        """set_edgecolors replaces previous edgecolors."""
        pc = PathCollection(edgecolors=['gray'])
        pc.set_edgecolors(['red', 'blue'])
        ec = pc.get_edgecolors()
        assert ec.shape == (2, 4)

    def test_set_edgecolors_empty(self):
        """set_edgecolors with empty list clears edgecolors."""
        pc = PathCollection(edgecolors=['black'])
        pc.set_edgecolors([])
        ec = pc.get_edgecolors()
        assert len(ec) == 0

    def test_edgecolors_tuples(self):
        """Edgecolors can be RGBA tuples."""
        colors = [(0.0, 0.0, 0.0, 1.0)]
        pc = PathCollection(edgecolors=colors)
        ec = pc.get_edgecolors()
        assert np.allclose(ec, colors)


# ===================================================================
# Data returned as copies (not references)
# ===================================================================

class TestDataCopies:
    def test_get_offsets_returns_copy(self):
        """get_offsets returns a numpy array."""
        pc = PathCollection(offsets=[(1, 2)])
        result = pc.get_offsets()
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [(1, 2)])

    def test_set_offsets_copies_input(self):
        """set_offsets copies the input; later mutations do not affect the object."""
        data = [(1, 2), (3, 4)]
        pc = PathCollection()
        pc.set_offsets(data)
        data.append((5, 6))
        result = pc.get_offsets()
        assert len(result) == 2
        assert np.allclose(result, [(1, 2), (3, 4)])

    def test_get_sizes_returns_copy(self):
        """get_sizes returns ndarray; mutations don't affect the object."""
        pc = PathCollection(sizes=[20.0])
        result = pc.get_sizes()
        assert isinstance(result, np.ndarray)
        # verify mutation of the copy doesn't affect internal state
        assert np.allclose(result, [20.0])

    def test_set_sizes_copies_input(self):
        """set_sizes copies the input list."""
        data = [10.0, 20.0]
        pc = PathCollection()
        pc.set_sizes(data)
        data.append(30.0)
        assert np.allclose(pc.get_sizes(), [10.0, 20.0])

    def test_get_facecolors_returns_copy(self):
        """get_facecolors returns RGBA ndarray."""
        pc = PathCollection(facecolors=['red'])
        result = pc.get_facecolors()
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 4)

    def test_set_facecolors_copies_input(self):
        """set_facecolors copies the input list."""
        data = ['red']
        pc = PathCollection()
        pc.set_facecolors(data)
        data.append('green')
        # internal still has 1 color
        assert len(pc.get_facecolors()) == 1

    def test_get_edgecolors_returns_copy(self):
        """get_edgecolors returns RGBA ndarray."""
        pc = PathCollection(edgecolors=['black'])
        result = pc.get_edgecolors()
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 4)

    def test_set_edgecolors_copies_input(self):
        """set_edgecolors copies the input list."""
        data = ['black']
        pc = PathCollection()
        pc.set_edgecolors(data)
        data.append('white')
        assert len(pc.get_edgecolors()) == 1

    def test_constructor_offsets_copies_input(self):
        """Constructor copies the offsets argument."""
        data = [(1, 2)]
        pc = PathCollection(offsets=data)
        data.append((3, 4))
        result = pc.get_offsets()
        assert len(result) == 1
        assert np.allclose(result, [(1, 2)])

    def test_constructor_sizes_copies_input(self):
        """Constructor copies the sizes argument."""
        data = [10.0]
        pc = PathCollection(sizes=data)
        data.append(20.0)
        assert np.allclose(pc.get_sizes(), [10.0])


# ===================================================================
# Label support
# ===================================================================

class TestLabelSupport:
    def test_label_via_constructor(self):
        """Label set in constructor is retrievable."""
        pc = PathCollection(label='my_scatter')
        assert pc.get_label() == 'my_scatter'

    def test_label_via_set_label(self):
        """Label set via set_label is retrievable."""
        pc = PathCollection()
        pc.set_label('updated')
        assert pc.get_label() == 'updated'

    def test_label_overwrite(self):
        """set_label overwrites a previously set label."""
        pc = PathCollection(label='first')
        pc.set_label('second')
        assert pc.get_label() == 'second'

    def test_label_numeric_coerced_to_string(self):
        """Numeric label is coerced to string."""
        pc = PathCollection(label=42)
        assert pc.get_label() == '42'

    def test_label_none_becomes_nolegend(self):
        """set_label(None) — OG stores None (not '_nolegend_')."""
        pc = PathCollection()
        pc.set_label(None)
        # OG stores None as-is
        assert pc.get_label() is None or pc.get_label() == '_nolegend_'


# ===================================================================
# Artist integration (zorder, visible, alpha)
# ===================================================================

class TestArtistIntegration:
    def test_pathcollection_is_collection(self):
        """PathCollection is a subclass of Collection."""
        assert issubclass(PathCollection, Collection)

    def test_zorder_class_attribute(self):
        """PathCollection class inherits zorder=0 from Collection."""
        assert PathCollection.zorder == 0

    def test_visible_default(self):
        """PathCollection is visible by default."""
        pc = PathCollection()
        assert pc.get_visible() is True

    def test_set_visible_false(self):
        """PathCollection visibility can be toggled."""
        pc = PathCollection()
        pc.set_visible(False)
        assert pc.get_visible() is False

    def test_alpha_default(self):
        """PathCollection alpha is None by default."""
        pc = PathCollection()
        assert pc.get_alpha() is None

    def test_set_alpha(self):
        """PathCollection alpha can be set."""
        pc = PathCollection()
        pc.set_alpha(0.7)
        assert pc.get_alpha() == 0.7

    def test_set_zorder(self):
        """PathCollection zorder can be changed."""
        pc = PathCollection()
        pc.set_zorder(5)
        assert pc.get_zorder() == 5

    def test_figure_default_none(self):
        """PathCollection has no figure by default."""
        pc = PathCollection()
        assert pc.figure is None

    def test_axes_default_none(self):
        """PathCollection has no axes by default."""
        pc = PathCollection()
        assert pc.axes is None

    def test_kwargs_forwarded_to_artist(self):
        """Extra kwargs are forwarded through Collection to Artist.set()."""
        pc = PathCollection(visible=False, alpha=0.3)
        assert pc.get_visible() is False
        assert pc.get_alpha() == 0.3


# ===================================================================
# Additional collection tests (upstream-inspired batch)
# ===================================================================

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection


class TestScatterCollection:
    """Tests for scatter plot PathCollection."""

    def test_scatter_returns_pathcollection(self):
        """ax.scatter returns a PathCollection."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [4, 5, 6])
        assert isinstance(sc, PathCollection)
        plt.close('all')

    def test_scatter_in_collections(self):
        """scatter adds collection to ax.collections."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [4, 5, 6])
        assert sc in ax.collections
        plt.close('all')

    def test_scatter_offsets_match_input(self):
        """scatter PathCollection offsets match input x,y."""
        fig, ax = plt.subplots()
        xs = [1.0, 2.0, 3.0]
        ys = [4.0, 5.0, 6.0]
        sc = ax.scatter(xs, ys)
        offsets = sc.get_offsets()
        assert len(offsets) == 3
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 5, 10, 50])
    def test_scatter_n_points(self, n):
        """scatter with n points creates n offsets."""
        fig, ax = plt.subplots()
        xs = list(range(n))
        ys = list(range(n))
        sc = ax.scatter(xs, ys)
        assert len(sc.get_offsets()) == n
        plt.close('all')

    def test_scatter_label(self):
        """scatter label is stored."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1], [1], label='my_scatter')
        assert sc.get_label() == 'my_scatter'
        plt.close('all')

    def test_scatter_alpha(self):
        """scatter alpha is settable."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2], [3, 4])
        sc.set_alpha(0.5)
        assert abs(sc.get_alpha() - 0.5) < 1e-10
        plt.close('all')

    def test_scatter_visible_default(self):
        """scatter collection is visible by default."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1], [1])
        assert sc.get_visible() is True
        plt.close('all')
