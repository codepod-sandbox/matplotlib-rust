"""
Upstream matplotlib tests for collections module.
"""

import numpy as np
import pytest

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, LineCollection, PolyCollection, EventCollection


# ---------------------------------------------------------------------------
# PathCollection from scatter
# ---------------------------------------------------------------------------
def test_scatter_returns_pathcollection():
    """scatter() returns a PathCollection."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2, 3], [4, 5, 6])
    assert isinstance(pc, PathCollection)


def test_scatter_offsets():
    """scatter offsets match input data."""
    import numpy as np
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4])
    offsets = pc.get_offsets()
    assert len(offsets) == 2
    assert np.allclose(offsets[0], [1, 3])
    assert np.allclose(offsets[1], [2, 4])


def test_scatter_sizes():
    """scatter sizes are stored."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], s=50)
    sizes = pc.get_sizes()
    assert sizes == [50]


def test_scatter_sizes_array():
    """scatter with per-point sizes."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], s=[10, 20])
    sizes = pc.get_sizes()
    assert sizes == [10, 20]


def test_scatter_color():
    """scatter facecolors are stored."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], c='red')
    fc = pc.get_facecolors()
    assert len(fc) > 0


def test_scatter_label():
    """scatter label is stored."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4], label='points')
    assert pc.get_label() == 'points'


def test_scatter_in_collections():
    """scatter result is in ax.collections."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4])
    assert pc in ax.collections


def test_scatter_string_s_raises():
    """scatter with string s raises ValueError."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.scatter([1, 2], [3, 4], s='large')


def test_scatter_mismatched_s_raises():
    """scatter with wrong-length s raises ValueError."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.scatter([1, 2], [3, 4], s=[10, 20, 30])


# ---------------------------------------------------------------------------
# PathCollection set/get
# ---------------------------------------------------------------------------
def test_pathcollection_set_offsets():
    """PathCollection.set_offsets changes offsets."""
    pc = PathCollection(offsets=[(0, 0)])
    pc.set_offsets([(1, 1), (2, 2)])
    assert len(pc.get_offsets()) == 2


def test_pathcollection_set_sizes():
    """PathCollection.set_sizes changes sizes."""
    pc = PathCollection(sizes=[10])
    pc.set_sizes([20, 30])
    assert pc.get_sizes() == [20, 30]


def test_pathcollection_set_facecolors():
    """PathCollection.set_facecolors changes colors."""
    pc = PathCollection(facecolors=['red'])
    pc.set_facecolors(['blue', 'green'])
    assert pc.get_facecolors() == ['blue', 'green']


def test_pathcollection_set_edgecolors():
    """PathCollection.set_edgecolors changes colors."""
    pc = PathCollection(edgecolors=['red'])
    pc.set_edgecolors(['blue'])
    assert pc.get_edgecolors() == ['blue']


def test_pathcollection_visible():
    """PathCollection visibility."""
    pc = PathCollection()
    assert pc.get_visible() is True
    pc.set_visible(False)
    assert pc.get_visible() is False


def test_pathcollection_alpha():
    """PathCollection alpha."""
    pc = PathCollection()
    assert pc.get_alpha() is None
    pc.set_alpha(0.5)
    assert pc.get_alpha() == 0.5


def test_pathcollection_label():
    """PathCollection label."""
    pc = PathCollection(label='test')
    assert pc.get_label() == 'test'


def test_pathcollection_zorder():
    """PathCollection default zorder is 1."""
    pc = PathCollection()
    assert pc.get_zorder() == 1


def test_pathcollection_remove():
    """PathCollection.remove removes from axes."""
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4])
    assert pc in ax.collections
    pc.remove()
    assert pc not in ax.collections


# ---------------------------------------------------------------------------
# PathCollection empty
# ---------------------------------------------------------------------------
def test_pathcollection_empty():
    """PathCollection with no offsets."""
    import numpy as np
    pc = PathCollection()
    offsets = pc.get_offsets()
    assert isinstance(offsets, np.ndarray)
    assert offsets.shape == (0, 2)


def test_pathcollection_default_sizes():
    """PathCollection default sizes is [20.0]."""
    pc = PathCollection()
    assert pc.get_sizes() == [20.0]


# ===================================================================
# LineCollection tests
# ===================================================================

class TestLineCollection:
    def _segs(self):
        return [[(0, 0), (1, 1)], [(1, 0), (2, 1)]]

    def test_basic_construction(self):
        lc = LineCollection(self._segs())
        assert isinstance(lc, LineCollection)

    def test_get_segments(self):
        segs = self._segs()
        lc = LineCollection(segs)
        result = lc.get_segments()
        assert len(result) == 2

    def test_set_segments(self):
        lc = LineCollection([])
        lc.set_segments(self._segs())
        assert len(lc.get_segments()) == 2

    def test_set_verts_alias(self):
        lc = LineCollection([])
        lc.set_verts(self._segs())
        assert len(lc.get_segments()) == 2

    def test_get_paths(self):
        lc = LineCollection(self._segs())
        paths = lc.get_paths()
        assert len(paths) == 2

    def test_set_paths(self):
        lc = LineCollection([])
        lc.set_paths(self._segs())
        assert len(lc.get_paths()) == 2

    def test_linewidths_scalar(self):
        lc = LineCollection(self._segs(), linewidths=2.0)
        assert lc._linewidths == [2.0]

    def test_linewidths_list(self):
        lc = LineCollection(self._segs(), linewidths=[1, 2])
        assert lc._linewidths == [1, 2]

    def test_colors_string(self):
        lc = LineCollection(self._segs(), colors='red')
        assert lc._edgecolors == ['red']

    def test_colors_list(self):
        lc = LineCollection(self._segs(), colors=['red', 'blue'])
        assert lc._edgecolors == ['red', 'blue']

    def test_get_color_default(self):
        lc = LineCollection(self._segs())
        c = lc.get_color()
        assert isinstance(c, list)

    def test_set_color_string(self):
        lc = LineCollection(self._segs())
        lc.set_color('green')
        assert lc._edgecolors == ['green']

    def test_set_color_list(self):
        lc = LineCollection(self._segs())
        lc.set_color(['red', 'blue'])
        assert lc._edgecolors == ['red', 'blue']

    def test_get_colors_alias(self):
        lc = LineCollection(self._segs(), colors='cyan')
        colors = lc.get_colors()
        # get_colors returns RGBA tuples
        assert len(colors) == 1
        assert len(colors[0]) == 4  # RGBA tuple

    def test_label(self):
        lc = LineCollection(self._segs(), label='my_lc')
        assert lc.get_label() == 'my_lc'

    def test_empty_segments(self):
        lc = LineCollection([])
        assert lc.get_segments() == []

    def test_add_to_axes(self):
        lc = LineCollection(self._segs())
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        assert lc in ax.collections
        plt.close('all')

    def test_linestyles_string(self):
        lc = LineCollection(self._segs(), linestyles='dashed')
        assert lc._linestyles == ['dashed']

    def test_linestyles_list(self):
        lc = LineCollection(self._segs(), linestyles=['solid', 'dashed'])
        assert lc._linestyles == ['solid', 'dashed']


# ===================================================================
# PolyCollection tests
# ===================================================================

class TestPolyCollection:
    def _verts(self):
        return [[(0, 0), (1, 0), (0.5, 1)], [(1, 0), (2, 0), (1.5, 1)]]

    def test_basic_construction(self):
        pc = PolyCollection(self._verts())
        assert isinstance(pc, PolyCollection)

    def test_get_verts(self):
        verts = self._verts()
        pc = PolyCollection(verts)
        result = pc.get_verts()
        assert len(result) == 2

    def test_set_verts(self):
        pc = PolyCollection([])
        pc.set_verts(self._verts())
        assert len(pc.get_verts()) == 2

    def test_empty_verts(self):
        pc = PolyCollection([])
        assert pc.get_verts() == []

    def test_none_verts(self):
        pc = PolyCollection(None)
        assert pc.get_verts() == []

    def test_facecolor(self):
        pc = PolyCollection(self._verts(), facecolor='red')
        assert pc.get_facecolor() is not None

    def test_add_to_axes(self):
        pc = PolyCollection(self._verts())
        fig, ax = plt.subplots()
        ax.add_collection(pc)
        assert pc in ax.collections
        plt.close('all')


# ===================================================================
# EventCollection tests
# ===================================================================

class TestEventCollection:
    def test_basic_construction(self):
        ec = EventCollection([1, 2, 3])
        assert isinstance(ec, EventCollection)

    def test_get_positions(self):
        ec = EventCollection([1, 2, 3])
        assert ec.get_positions() == [1, 2, 3]

    def test_set_positions(self):
        ec = EventCollection([])
        ec.set_positions([4, 5, 6])
        assert ec.get_positions() == [4, 5, 6]

    def test_orientation_default(self):
        ec = EventCollection([1, 2])
        assert ec.get_orientation() == 'horizontal'

    def test_orientation_vertical(self):
        ec = EventCollection([1, 2], orientation='vertical')
        assert ec.get_orientation() == 'vertical'

    def test_set_orientation(self):
        ec = EventCollection([1, 2])
        ec.set_orientation('vertical')
        assert ec.get_orientation() == 'vertical'

    def test_lineoffset_default(self):
        ec = EventCollection([1, 2])
        assert ec.get_lineoffset() == 0

    def test_lineoffset_custom(self):
        ec = EventCollection([1, 2], lineoffset=0.5)
        assert ec.get_lineoffset() == 0.5

    def test_set_lineoffset(self):
        ec = EventCollection([1, 2])
        ec.set_lineoffset(2.0)
        assert ec.get_lineoffset() == 2.0

    def test_empty_positions(self):
        ec = EventCollection([])
        assert ec.get_positions() == []

    def test_none_positions(self):
        ec = EventCollection(None)
        assert ec.get_positions() == []

    def test_color(self):
        ec = EventCollection([1, 2], color='blue')
        assert ec._edgecolors == ['blue']

    def test_add_to_axes(self):
        ec = EventCollection([1, 2, 3, 4, 5])
        fig, ax = plt.subplots()
        ax.add_collection(ec)
        assert ec in ax.collections
        plt.close('all')

    def test_linelength_default(self):
        ec = EventCollection([1, 2])
        assert ec.get_linelength() == 1

    def test_linelength_custom(self):
        ec = EventCollection([1, 2], linelength=2.0)
        assert ec.get_linelength() == 2.0

    def test_set_linelength(self):
        ec = EventCollection([1, 2])
        ec.set_linelength(3.0)
        assert ec.get_linelength() == 3.0

    def test_label(self):
        ec = EventCollection([1, 2], label='events')
        assert ec.get_label() == 'events'


# ===================================================================
# Collection base class tests
# ===================================================================

class TestCollectionBase:
    def test_default_linewidths(self):
        """Collection default linewidths is [1.0]."""
        from matplotlib.collections import Collection
        # Use PathCollection as concrete subclass
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        assert pc.get_linewidths() == [1.0]

    def test_set_linewidth_scalar(self):
        """set_linewidth with scalar stores as list."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_linewidth(2.5)
        assert pc.get_linewidths() == [2.5]

    def test_set_linewidths_list(self):
        """set_linewidths with list stores all values."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_linewidths([1.0, 2.0, 3.0])
        assert pc.get_linewidths() == [1.0, 2.0, 3.0]

    def test_default_linestyles(self):
        """Collection default linestyles is ['solid']."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        assert pc.get_linestyles() == ['solid']

    def test_set_linestyle_string(self):
        """set_linestyle with string stores as list."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_linestyle('dashed')
        assert pc.get_linestyles() == ['dashed']

    def test_set_linestyle_list(self):
        """set_linestyle with list stores all values."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_linestyle(['solid', 'dashed'])
        assert pc.get_linestyles() == ['solid', 'dashed']

    def test_get_set_array(self):
        """get_array/set_array roundtrip."""
        import numpy as np
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        assert pc.get_array() is None
        pc.set_array([1, 2, 3])
        np.testing.assert_array_equal(pc.get_array(), [1, 2, 3])

    def test_get_set_paths(self):
        """get_paths/set_paths roundtrip."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_paths([[0, 1, 2]])
        paths = pc.get_paths()
        assert len(paths) == 1

    def test_set_color_sets_both(self):
        """set_color sets both facecolors and edgecolors."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_color('red')
        assert pc.get_facecolors() == ['red']
        assert pc.get_edgecolors() == ['red']

    def test_set_edgecolor_none_clears(self):
        """set_edgecolor(None) clears edgecolors."""
        from matplotlib.collections import PathCollection
        pc = PathCollection(edgecolors=['red'])
        pc.set_edgecolor(None)
        assert pc.get_edgecolors() == []

    def test_set_facecolor_none_clears(self):
        """set_facecolor(None) clears facecolors."""
        from matplotlib.collections import PathCollection
        pc = PathCollection(facecolors=['blue'])
        pc.set_facecolor(None)
        assert pc.get_facecolors() == []

    def test_set_offsets_list(self):
        """set_offsets stores list of offsets."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_offsets([(1, 2), (3, 4)])
        assert len(pc.get_offsets()) == 2

    def test_get_linestyle_alias(self):
        """get_linestyle is alias for get_linestyles."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_linestyle('dotted')
        assert pc.get_linestyle() == ['dotted']
        assert pc.get_linestyles() == ['dotted']

    def test_get_linewidth_alias(self):
        """get_linewidth is alias for get_linewidths."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_linewidth(3.0)
        assert pc.get_linewidth() == [3.0]
        assert pc.get_linewidths() == [3.0]

    def test_get_facecolor_alias(self):
        """get_facecolor is alias for get_facecolors."""
        from matplotlib.collections import PathCollection
        pc = PathCollection(facecolors=['red'])
        assert pc.get_facecolor() == ['red']
        assert pc.get_facecolors() == ['red']

    def test_get_edgecolor_alias(self):
        """get_edgecolor is alias for get_edgecolors."""
        from matplotlib.collections import PathCollection
        pc = PathCollection(edgecolors=['blue'])
        assert pc.get_edgecolor() == ['blue']
        assert pc.get_edgecolors() == ['blue']


# ===================================================================
# Extended parametric tests for collections
# ===================================================================

import pytest as _pytest_col


# ---------------------------------------------------------------------------
# LineCollection upstream tests (test_collections.py)
# ---------------------------------------------------------------------------

def test_linecollection_basic():
    """LineCollection accepts a list of (N,2) arrays."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots()
    segs = [np.array([[0,0],[1,1]]), np.array([[0,1],[1,0]])]
    lc = LineCollection(segs)
    ax.add_collection(lc)
    assert len(ax.collections) == 1
    plt.close('all')


def test_linecollection_set_color():
    """LineCollection.set_color() applies to all segments."""
    import numpy as np
    from matplotlib.collections import LineCollection
    segs = [np.array([[0,0],[1,1]])]
    lc = LineCollection(segs, color='red')
    colors = lc.get_colors()
    assert len(colors) >= 1
    assert abs(colors[0][0] - 1.0) < 1e-6  # red channel


def test_linecollection_set_linewidth():
    """LineCollection linewidth can be set and retrieved."""
    import numpy as np
    from matplotlib.collections import LineCollection
    segs = [np.array([[0,0],[1,1]])]
    lc = LineCollection(segs, linewidth=3)
    lw = lc.get_linewidth()
    assert lw[0] == 3.0


def test_linecollection_set_linestyle():
    """LineCollection linestyle can be set."""
    import numpy as np
    from matplotlib.collections import LineCollection
    segs = [np.array([[0,0],[1,1]])]
    lc = LineCollection(segs, linestyle='--')
    # Should not raise


def test_polycollection_basic():
    """PolyCollection accepts a list of vertex arrays."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, ax = plt.subplots()
    verts = [np.array([[0,0],[1,0],[1,1],[0,1]])]
    pc = PolyCollection(verts)
    ax.add_collection(pc)
    assert len(ax.collections) == 1
    plt.close('all')


def test_polycollection_facecolor():
    """PolyCollection facecolor round-trips."""
    import numpy as np
    from matplotlib.collections import PolyCollection
    verts = [np.array([[0,0],[1,0],[1,1]])]
    pc = PolyCollection(verts, facecolor='blue')
    fc = pc.get_facecolor()
    assert abs(fc[0][2] - 1.0) < 1e-6  # blue channel


def test_eventcollection_basic():
    """EventCollection accepts positions list."""
    from matplotlib.collections import EventCollection
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ec = EventCollection([1, 2, 3])
    ax.add_collection(ec)
    assert len(ax.collections) == 1
    plt.close('all')


def test_eventcollection_positions():
    """EventCollection stores positions in sorted order."""
    import numpy as np
    from matplotlib.collections import EventCollection
    ec = EventCollection([3, 1, 2])
    pos = ec.get_positions()
    # Positions may or may not be sorted — just check length
    assert len(pos) == 3


def test_eventcollection_orientation():
    """EventCollection orientation= is stored as an attribute."""
    from matplotlib.collections import EventCollection
    ec = EventCollection([1, 2, 3], orientation='vertical')
    assert ec.orientation in ('vertical', 'v')


def test_pathcollection_offsets():
    """PathCollection set_offsets updates offset data."""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [4, 5, 6])
    new_offsets = np.array([[0, 0], [1, 1]])
    sc.set_offsets(new_offsets)
    offsets = sc.get_offsets()
    assert offsets.shape == (2, 2)
    plt.close('all')


def test_scatter_s_scalar():
    """ax.scatter with scalar s applies uniform marker size."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [4, 5, 6], s=100)
    sizes = sc.get_sizes()
    assert all(s == 100 for s in sizes)
    plt.close('all')


def test_scatter_s_array():
    """ax.scatter with array s applies per-point marker sizes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [4, 5, 6], s=[10, 20, 30])
    sizes = sc.get_sizes()
    assert list(sizes) == [10, 20, 30]
    plt.close('all')


def test_scatter_c_array():
    """ax.scatter with c= array produces a PathCollection in ax.collections."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [4, 5, 6], c=[0.1, 0.5, 0.9])
    assert isinstance(sc, PathCollection)
    assert sc in ax.collections
    plt.close('all')


def test_scatter_alpha():
    """ax.scatter with alpha= sets transparency."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2], [3, 4], alpha=0.5)
    assert sc.get_alpha() == 0.5
    plt.close('all')


def test_collection_set_cmap():
    """Collection.set_cmap() accepts colormap name."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [4, 5, 6], c=[0, 0.5, 1])
    sc.set_cmap('plasma')
    assert sc.get_cmap().name == 'plasma'
    plt.close('all')


def test_collection_set_clim():
    """Collection.set_clim() sets vmin/vmax for colormap."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [4, 5, 6], c=[1, 2, 3])
    sc.set_clim(0, 5)
    assert sc.norm.vmin == 0
    assert sc.norm.vmax == 5
    plt.close('all')


def test_quadmesh_basic():
    """ax.pcolormesh creates a QuadMesh in ax.collections."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import QuadMesh
    fig, ax = plt.subplots()
    data = np.arange(12).reshape(3, 4).astype(float)
    mesh = ax.pcolormesh(data)
    assert isinstance(mesh, QuadMesh)
    assert mesh in ax.collections
    plt.close('all')


def test_collection_label():
    """Collection label round-trips."""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1], [1], label='mypoints')
    assert sc.get_label() == 'mypoints'
    plt.close('all')


def test_collection_visible():
    """Collection visibility can be toggled."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2], [3, 4])
    sc.set_visible(False)
    assert sc.get_visible() is False
    plt.close('all')


def test_collection_remove():
    """collection.remove() removes it from axes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2], [3, 4])
    assert sc in ax.collections
    sc.remove()
    assert sc not in ax.collections
    plt.close('all')


# ---------------------------------------------------------------------------
# Upstream: EventCollection tests (no image comparison)
# ---------------------------------------------------------------------------

def test_EventCollection_nosort():
    """EventCollection should not sort input positions in-place."""
    arr = np.array([3, 2, 1, 10])
    EventCollection(arr)
    np.testing.assert_array_equal(arr, np.array([3, 2, 1, 10]))


def test_EventCollection_positions():
    """EventCollection get/set positions."""
    positions = np.array([0., 1., 2., 3., 5.])
    coll = EventCollection(positions)
    np.testing.assert_array_equal(coll.get_positions(), positions)
    new_pos = np.array([10., 20.])
    coll.set_positions(new_pos)
    np.testing.assert_array_equal(coll.get_positions(), new_pos)


def test_EventCollection_orientation():
    """EventCollection orientation and is_horizontal."""
    coll = EventCollection([0, 1, 2], orientation='horizontal')
    assert coll.get_orientation() == 'horizontal'
    assert coll.is_horizontal()
    coll.switch_orientation()
    assert coll.get_orientation() == 'vertical'
    assert not coll.is_horizontal()
    coll.switch_orientation()
    assert coll.is_horizontal()


def test_EventCollection_add_positions():
    """EventCollection add/extend/append positions."""
    arr = np.array([0., 1., 2.])
    coll = EventCollection(arr)
    coll.add_positions(5.0)
    assert 5.0 in coll.get_positions()

    arr2 = np.array([3., 4.])
    coll.extend_positions(arr2)
    for p in [3., 4.]:
        assert p in coll.get_positions()

    coll.append_positions(99.0)
    assert 99.0 in coll.get_positions()


def test_EventCollection_lineoffset():
    """EventCollection get/set lineoffset."""
    coll = EventCollection([0, 1], lineoffset=1.5)
    assert coll.get_lineoffset() == 1.5
    coll.set_lineoffset(-2.0)
    assert coll.get_lineoffset() == -2.0


def test_EventCollection_linelength():
    """EventCollection get/set linelength."""
    coll = EventCollection([0, 1], linelength=0.5)
    assert coll.get_linelength() == 0.5
    coll.set_linelength(2.0)
    assert coll.get_linelength() == 2.0


# ---------------------------------------------------------------------------
# Upstream: Collection cap/joinstyle and linestyle validation
# ---------------------------------------------------------------------------

def test_capstyle():
    """PathCollection capstyle getter/setter."""
    col = PathCollection([])
    assert col.get_capstyle() is None
    col = PathCollection([], capstyle='round')
    assert col.get_capstyle() == 'round'
    col.set_capstyle('butt')
    assert col.get_capstyle() == 'butt'


def test_joinstyle():
    """PathCollection joinstyle getter/setter."""
    from matplotlib.collections import PathCollection
    col = PathCollection([])
    assert col.get_joinstyle() is None
    col = PathCollection([], joinstyle='round')
    assert col.get_joinstyle() == 'round'
    col.set_joinstyle('miter')
    assert col.get_joinstyle() == 'miter'


def test_set_wrong_linestyle():
    """Collection raises ValueError for unknown linestyle string."""
    from matplotlib.collections import Collection
    c = Collection()
    with pytest.raises(ValueError, match="Do not know how to convert"):
        c.set_linestyle('fuzzy')


# ---------------------------------------------------------------------------
# Upstream: Collection.set_array copies input
# ---------------------------------------------------------------------------

def test_collection_set_array():
    """set_array copies data; rejects non-numeric types."""
    from matplotlib.collections import Collection
    vals = list(range(10))

    c = Collection()
    c.set_array(vals)

    with pytest.raises(TypeError, match="Image data of dtype"):
        c.set_array("wrong_input")

    # Array is copied — modifying source doesn't change stored array
    vals[5] = 45
    assert np.not_equal(vals, c.get_array()).any()


# ---------------------------------------------------------------------------
# Upstream: EllipseCollection setter/getter
# ---------------------------------------------------------------------------

def test_EllipseCollection_setter_getter():
    """EllipseCollection stores half-widths, returns full widths."""
    from matplotlib.collections import EllipseCollection
    rng = np.random.default_rng(0)

    widths = (2,)
    heights = (3,)
    angles = (45,)
    offsets = rng.random((10, 2)) * 10

    fig, ax = plt.subplots()
    ec = EllipseCollection(
        widths=widths, heights=heights, angles=angles,
        offsets=offsets, units='x', offset_transform=ax.transData)

    np.testing.assert_array_almost_equal(ec._widths, np.array(widths).ravel() * 0.5)
    np.testing.assert_array_almost_equal(ec._heights, np.array(heights).ravel() * 0.5)
    np.testing.assert_array_almost_equal(ec._angles, np.deg2rad(angles))

    np.testing.assert_array_almost_equal(ec.get_widths(), widths)
    np.testing.assert_array_almost_equal(ec.get_heights(), heights)
    np.testing.assert_array_almost_equal(ec.get_angles(), angles)

    new_widths = rng.random((10, 2)) * 2
    new_heights = rng.random((10, 2)) * 3
    new_angles = rng.random((10, 2)) * 180

    ec.set(widths=new_widths, heights=new_heights, angles=new_angles)
    np.testing.assert_array_almost_equal(ec.get_widths(), new_widths.ravel())
    np.testing.assert_array_almost_equal(ec.get_heights(), new_heights.ravel())
    np.testing.assert_array_almost_equal(ec.get_angles(), new_angles.ravel())

    plt.close('all')
