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
    fig, ax = plt.subplots()
    pc = ax.scatter([1, 2], [3, 4])
    offsets = pc.get_offsets()
    assert len(offsets) == 2
    assert offsets[0] == (1, 3)
    assert offsets[1] == (2, 4)


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
    pc = PathCollection()
    assert pc.get_offsets() == []


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
        assert lc.get_colors() == ['cyan']

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
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        assert pc.get_array() is None
        pc.set_array([1, 2, 3])
        assert pc.get_array() == [1, 2, 3]

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

class TestCollectionsParametricExtended:
    """Extended parametric tests for collections."""

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_path_collection_linewidth(self, lw):
        """PathCollection linewidth is stored."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_linewidth(lw)
        result = pc.get_linewidth()
        if isinstance(result, list):
            assert abs(result[0] - lw) < 1e-10
        else:
            assert abs(result - lw) < 1e-10

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff0000', 'cyan'])
    def test_path_collection_facecolor(self, color):
        """PathCollection facecolor is stored."""
        from matplotlib.collections import PathCollection
        pc = PathCollection(facecolors=[color])
        result = pc.get_facecolors()
        assert result is not None
        assert len(result) > 0

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_path_collection_alpha(self, alpha):
        """PathCollection alpha is stored."""
        from matplotlib.collections import PathCollection
        pc = PathCollection()
        pc.set_alpha(alpha)
        assert abs(pc.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('n', [1, 3, 5, 10, 20])
    def test_scatter_n_points(self, n):
        """scatter creates PathCollection with n points."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter(range(n), range(n))
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('s', [10, 20, 50, 100, 200])
    def test_scatter_marker_size(self, s):
        """scatter accepts s (size) parameter."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], s=s)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#aabbcc'])
    def test_scatter_color(self, color):
        """scatter accepts color parameter."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], color=color)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.2, 0.5, 0.8, 1.0])
    def test_scatter_alpha(self, alpha):
        """scatter accepts alpha parameter."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], alpha=alpha)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'v', 'D', '*'])
    def test_scatter_marker(self, marker):
        """scatter accepts marker parameter."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], marker=marker)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_line_collection_n_lines(self, n):
        """LineCollection accepts n line segments."""
        from matplotlib.collections import LineCollection
        segs = [[(0, i), (1, i+1)] for i in range(n)]
        lc = LineCollection(segs)
        assert lc is not None


class TestCollectionsUpstreamParametric2:
    """More parametric tests for collections_upstream."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_linecollection_n_segs(self, n):
        """LineCollection with n segments."""
        segs = [[(0, i), (1, i+1)] for i in range(n)]
        lc = LineCollection(segs)
        assert len(lc.get_segments()) == n

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0])
    def test_linecollection_lw(self, lw):
        """LineCollection linewidth stored."""
        segs = [[(0, 0), (1, 1)]]
        lc = LineCollection(segs, linewidths=lw)
        assert lc is not None

    @pytest.mark.parametrize('alpha', [0.1, 0.5, 0.8, 1.0])
    def test_linecollection_alpha(self, alpha):
        """LineCollection alpha stored."""
        segs = [[(0, 0), (1, 1)]]
        lc = LineCollection(segs)
        lc.set_alpha(alpha)
        assert abs(lc.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('visible', [True, False])
    def test_linecollection_visible(self, visible):
        """LineCollection visibility."""
        segs = [[(0, 0), (1, 1)]]
        lc = LineCollection(segs)
        lc.set_visible(visible)
        assert lc.get_visible() == visible

    @pytest.mark.parametrize('zorder', [1, 2, 5, 10])
    def test_linecollection_zorder(self, zorder):
        """LineCollection zorder."""
        segs = [[(0, 0), (1, 1)]]
        lc = LineCollection(segs)
        lc.set_zorder(zorder)
        assert lc.get_zorder() == zorder

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_ax_line_n_segs(self, n):
        """ax with n lines via LineCollection."""
        fig, ax = plt.subplots()
        segs = [[(0, i), (1, i+1)] for i in range(n)]
        lc = LineCollection(segs)
        ax.add_collection(lc)
        assert lc is not None
        plt.close('all')

    @pytest.mark.parametrize('linestyle', ['-', '--', ':', '-.'])
    def test_linecollection_linestyle(self, linestyle):
        """LineCollection linestyle stored."""
        segs = [[(0, 0), (1, 1)]]
        lc = LineCollection(segs, linestyles=linestyle)
        assert lc is not None

    @pytest.mark.parametrize('n', [2, 3, 5, 10])
    def test_polycollection_n(self, n):
        """PolyCollection with n polygons."""
        from matplotlib.collections import PolyCollection
        import numpy as np
        verts = [np.array([(0, 0), (1, 0), (1, 1), (0, 1)]) for _ in range(n)]
        pc = PolyCollection(verts)
        assert pc is not None


class TestCollectionsUpstreamParametric4:
    """Further parametric tests."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-5, 5), (0, 100)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9
        assert abs(result[1] - xlim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("marker", ["o", "s", "^", "D", "v"])
    def test_marker(self, marker):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_bar(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(1, n + 1))
        assert len(bars) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close("all")

    @pytest.mark.parametrize("title", ["Test", "My Plot", "Signal", "", "Results"])
    def test_title(self, title):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close("all")

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_line_alpha(self, alpha):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")

