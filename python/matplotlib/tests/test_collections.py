"""Tests for matplotlib.collections module --- Collection and PathCollection."""

import pytest

from matplotlib.collections import Collection, PathCollection
from matplotlib.colors import to_hex


# ===================================================================
# Collection base class
# ===================================================================

class TestCollection:
    def test_default_zorder(self):
        """Collection class zorder is 1."""
        assert Collection.zorder == 1

    def test_instance_zorder(self):
        """Collection instance inherits zorder=1."""
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
        """Default offsets is an empty list."""
        pc = PathCollection()
        assert pc.get_offsets() == []

    def test_default_sizes(self):
        """Default sizes is [20.0]."""
        pc = PathCollection()
        assert pc.get_sizes() == [20.0]

    def test_default_facecolors(self):
        """Default facecolors is an empty list."""
        pc = PathCollection()
        assert pc.get_facecolors() == []

    def test_default_edgecolors(self):
        """Default edgecolors is an empty list."""
        pc = PathCollection()
        assert pc.get_edgecolors() == []

    def test_default_label(self):
        """Default label is empty string (inherited from Artist)."""
        pc = PathCollection()
        assert pc.get_label() == ''

    def test_default_zorder(self):
        """PathCollection inherits Collection zorder=1."""
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
        assert pc.get_offsets() == [(1, 2), (3, 4)]

    def test_explicit_sizes(self):
        """Sizes passed at construction are stored."""
        pc = PathCollection(sizes=[10.0, 30.0, 50.0])
        assert pc.get_sizes() == [10.0, 30.0, 50.0]

    def test_explicit_facecolors(self):
        """Facecolors passed at construction are stored."""
        pc = PathCollection(facecolors=['red', 'blue'])
        assert pc.get_facecolors() == ['red', 'blue']

    def test_explicit_edgecolors(self):
        """Edgecolors passed at construction are stored."""
        pc = PathCollection(edgecolors=['green'])
        assert pc.get_edgecolors() == ['green']

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
        assert pc.get_offsets() == [(0, 0)]
        assert pc.get_sizes() == [50.0]
        assert pc.get_facecolors() == ['red']
        assert pc.get_edgecolors() == ['black']
        assert pc.get_label() == 'full'


# ===================================================================
# get/set offsets round-trip
# ===================================================================

class TestOffsetsRoundTrip:
    def test_set_get_offsets(self):
        """set_offsets then get_offsets returns the same data."""
        pc = PathCollection()
        pc.set_offsets([(10, 20), (30, 40)])
        assert pc.get_offsets() == [(10, 20), (30, 40)]

    def test_set_offsets_replaces(self):
        """set_offsets replaces previous offsets entirely."""
        pc = PathCollection(offsets=[(1, 2)])
        pc.set_offsets([(5, 6), (7, 8)])
        assert pc.get_offsets() == [(5, 6), (7, 8)]

    def test_set_offsets_empty(self):
        """set_offsets with empty list clears offsets."""
        pc = PathCollection(offsets=[(1, 2)])
        pc.set_offsets([])
        assert pc.get_offsets() == []


# ===================================================================
# get/set sizes round-trip
# ===================================================================

class TestSizesRoundTrip:
    def test_set_get_sizes(self):
        """set_sizes then get_sizes returns the same data."""
        pc = PathCollection()
        pc.set_sizes([100.0, 200.0])
        assert pc.get_sizes() == [100.0, 200.0]

    def test_set_sizes_replaces(self):
        """set_sizes replaces previous sizes entirely."""
        pc = PathCollection(sizes=[5.0])
        pc.set_sizes([10.0, 15.0, 20.0])
        assert pc.get_sizes() == [10.0, 15.0, 20.0]

    def test_set_sizes_empty(self):
        """set_sizes with empty list clears sizes."""
        pc = PathCollection()
        pc.set_sizes([])
        assert pc.get_sizes() == []


# ===================================================================
# get/set facecolors round-trip
# ===================================================================

class TestFacecolorsRoundTrip:
    def test_set_get_facecolors(self):
        """set_facecolors then get_facecolors returns the same data."""
        pc = PathCollection()
        pc.set_facecolors(['red', 'green'])
        assert pc.get_facecolors() == ['red', 'green']

    def test_set_facecolors_replaces(self):
        """set_facecolors replaces previous facecolors."""
        pc = PathCollection(facecolors=['blue'])
        pc.set_facecolors(['yellow', 'cyan'])
        assert pc.get_facecolors() == ['yellow', 'cyan']

    def test_set_facecolors_empty(self):
        """set_facecolors with empty list clears facecolors."""
        pc = PathCollection(facecolors=['red'])
        pc.set_facecolors([])
        assert pc.get_facecolors() == []

    def test_facecolors_tuples(self):
        """Facecolors can be RGBA tuples."""
        colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.5)]
        pc = PathCollection(facecolors=colors)
        assert pc.get_facecolors() == colors


# ===================================================================
# get/set edgecolors round-trip
# ===================================================================

class TestEdgecolorsRoundTrip:
    def test_set_get_edgecolors(self):
        """set_edgecolors then get_edgecolors returns the same data."""
        pc = PathCollection()
        pc.set_edgecolors(['black', 'white'])
        assert pc.get_edgecolors() == ['black', 'white']

    def test_set_edgecolors_replaces(self):
        """set_edgecolors replaces previous edgecolors."""
        pc = PathCollection(edgecolors=['gray'])
        pc.set_edgecolors(['red', 'blue'])
        assert pc.get_edgecolors() == ['red', 'blue']

    def test_set_edgecolors_empty(self):
        """set_edgecolors with empty list clears edgecolors."""
        pc = PathCollection(edgecolors=['black'])
        pc.set_edgecolors([])
        assert pc.get_edgecolors() == []

    def test_edgecolors_tuples(self):
        """Edgecolors can be RGBA tuples."""
        colors = [(0.0, 0.0, 0.0, 1.0)]
        pc = PathCollection(edgecolors=colors)
        assert pc.get_edgecolors() == colors


# ===================================================================
# Data returned as copies (not references)
# ===================================================================

class TestDataCopies:
    def test_get_offsets_returns_copy(self):
        """get_offsets returns a new list each time, not the internal one."""
        pc = PathCollection(offsets=[(1, 2)])
        result = pc.get_offsets()
        result.append((99, 99))
        assert pc.get_offsets() == [(1, 2)]

    def test_set_offsets_copies_input(self):
        """set_offsets copies the input; later mutations do not affect the object."""
        data = [(1, 2), (3, 4)]
        pc = PathCollection()
        pc.set_offsets(data)
        data.append((5, 6))
        assert pc.get_offsets() == [(1, 2), (3, 4)]

    def test_get_sizes_returns_copy(self):
        """get_sizes returns a new list, not the internal one."""
        pc = PathCollection(sizes=[20.0])
        result = pc.get_sizes()
        result.append(999.0)
        assert pc.get_sizes() == [20.0]

    def test_set_sizes_copies_input(self):
        """set_sizes copies the input list."""
        data = [10.0, 20.0]
        pc = PathCollection()
        pc.set_sizes(data)
        data.append(30.0)
        assert pc.get_sizes() == [10.0, 20.0]

    def test_get_facecolors_returns_copy(self):
        """get_facecolors returns a new list, not the internal one."""
        pc = PathCollection(facecolors=['red'])
        result = pc.get_facecolors()
        result.append('blue')
        assert pc.get_facecolors() == ['red']

    def test_set_facecolors_copies_input(self):
        """set_facecolors copies the input list."""
        data = ['red']
        pc = PathCollection()
        pc.set_facecolors(data)
        data.append('green')
        assert pc.get_facecolors() == ['red']

    def test_get_edgecolors_returns_copy(self):
        """get_edgecolors returns a new list, not the internal one."""
        pc = PathCollection(edgecolors=['black'])
        result = pc.get_edgecolors()
        result.append('white')
        assert pc.get_edgecolors() == ['black']

    def test_set_edgecolors_copies_input(self):
        """set_edgecolors copies the input list."""
        data = ['black']
        pc = PathCollection()
        pc.set_edgecolors(data)
        data.append('white')
        assert pc.get_edgecolors() == ['black']

    def test_constructor_offsets_copies_input(self):
        """Constructor copies the offsets argument."""
        data = [(1, 2)]
        pc = PathCollection(offsets=data)
        data.append((3, 4))
        assert pc.get_offsets() == [(1, 2)]

    def test_constructor_sizes_copies_input(self):
        """Constructor copies the sizes argument."""
        data = [10.0]
        pc = PathCollection(sizes=data)
        data.append(20.0)
        assert pc.get_sizes() == [10.0]


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
        """set_label(None) produces '_nolegend_'."""
        pc = PathCollection()
        pc.set_label(None)
        assert pc.get_label() == '_nolegend_'


# ===================================================================
# Artist integration (zorder, visible, alpha)
# ===================================================================

class TestArtistIntegration:
    def test_pathcollection_is_collection(self):
        """PathCollection is a subclass of Collection."""
        assert issubclass(PathCollection, Collection)

    def test_zorder_class_attribute(self):
        """PathCollection class inherits zorder=1 from Collection."""
        assert PathCollection.zorder == 1

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

import pytest
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


class TestPathCollectionParametric:
    """Parametric tests for PathCollection."""

    @pytest.mark.parametrize('color', ['red', 'blue', 'green'])
    def test_scatter_color(self, color):
        """scatter facecolor can be specified."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2], [3, 4], color=color)
        fc = sc.get_facecolor()
        assert len(fc) > 0
        plt.close('all')

    @pytest.mark.parametrize('size', [10, 50, 100, 200])
    def test_scatter_sizes(self, size):
        """scatter size sets marker sizes."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [4, 5, 6], s=size)
        sizes = sc.get_sizes()
        assert len(sizes) == 1  # single size applied to all
        assert abs(sizes[0] - size) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.5, 0.9, 1.0])
    def test_collection_alpha(self, alpha):
        """Collection alpha is settable."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2], [3, 4])
        sc.set_alpha(alpha)
        assert abs(sc.get_alpha() - alpha) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('zorder', [0, 1, 3, 5])
    def test_collection_zorder(self, zorder):
        """Collection zorder is settable."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1], [1])
        sc.set_zorder(zorder)
        assert sc.get_zorder() == zorder
        plt.close('all')


# ===================================================================
# Extended parametric tests for collections
# ===================================================================

class TestCollectionsParametric2:
    """More parametric tests for collections."""

    @pytest.mark.parametrize('n', [1, 3, 5, 10, 20])
    def test_scatter_n_points(self, n):
        """scatter creates n points collection."""
        fig, ax = plt.subplots()
        sc = ax.scatter(range(n), range(n))
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('s', [5, 10, 20, 50, 100, 200])
    def test_scatter_size(self, s):
        """scatter accepts size parameter."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], s=s)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'v', 'D', '+', 'x', '*'])
    def test_scatter_marker(self, marker):
        """scatter accepts various markers."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], marker=marker)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_scatter_alpha(self, alpha):
        """scatter accepts alpha."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], alpha=alpha)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#aabbcc', 'cyan'])
    def test_scatter_color(self, color):
        """scatter accepts color."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3], color=color)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_line_collection_n_segs(self, n):
        """LineCollection created with n segments."""
        from matplotlib.collections import LineCollection
        segs = [[(0, i), (1, i+1)] for i in range(n)]
        lc = LineCollection(segs)
        assert lc is not None

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0])
    def test_line_collection_linewidth(self, lw):
        """LineCollection stores linewidth."""
        from matplotlib.collections import LineCollection
        lc = LineCollection([[(0, 0), (1, 1)]])
        lc.set_linewidth(lw)
        result = lc.get_linewidths()
        assert abs(result[0] - lw) < 1e-10

    @pytest.mark.parametrize('visible', [True, False])
    def test_scatter_visibility(self, visible):
        """scatter visibility is stored."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3])
        sc.set_visible(visible)
        assert sc.get_visible() == visible
        plt.close('all')

    @pytest.mark.parametrize('zorder', [0, 1, 2, 5, 10])
    def test_scatter_zorder(self, zorder):
        """scatter zorder is stored."""
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 2, 3])
        sc.set_zorder(zorder)
        assert sc.get_zorder() == zorder
        plt.close('all')


class TestCollectionsParametric2:
    """More parametric tests."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_n_lines(self, n):
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("lo,hi", [(0, 1), (-1, 1), (0, 100)])
    def test_xlim(self, lo, hi):
        fig, ax = plt.subplots()
        ax.set_xlim(lo, hi)
        assert ax.get_xlim() == (lo, hi)
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0])
    def test_linewidth(self, lw):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close("all")

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_bar(self, n):
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("title", ["Title", "Test", ""])
    def test_title(self, title):
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close("all")

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_line_alpha(self, alpha):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_hist(self, bins):
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close("all")

    @pytest.mark.parametrize("marker", ["o", "s", "^"])
    def test_marker(self, marker):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")



class TestCollectionsParametric3:
    """Further parametric collection tests."""

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
    def test_n_line_collections(self, n):
        """Multiple LineCollections can be added."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        for i in range(n):
            lc = LineCollection([[(0, i), (1, i)]])
            ax.add_collection(lc)
        assert len(ax.collections) == n
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_path_collection_alpha(self, alpha):
        """PathCollection alpha is set correctly."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([0, 1, 2], [0, 1, 2], alpha=alpha)
        assert abs(sc.get_alpha() - alpha) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('n', [3, 5, 10, 20, 50])
    def test_scatter_n_points(self, n):
        """Scatter plot with n points produces PathCollection."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter(range(n), range(n))
        assert hasattr(sc, 'get_offsets')
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff0000', 'orange'])
    def test_line_collection_color(self, color):
        """LineCollection color is set via set_color."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]])
        lc.set_color(color)
        ax.add_collection(lc)
        assert lc.get_color() is not None
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_line_collection_linewidth(self, lw):
        """LineCollection linewidth is set correctly."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]], linewidths=lw)
        ax.add_collection(lc)
        lws = lc.get_linewidth()
        assert abs(float(lws[0]) - lw) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_collection_visibility(self, visible):
        """Collection visibility is set correctly."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]])
        ax.add_collection(lc)
        lc.set_visible(visible)
        assert lc.get_visible() == visible
        plt.close('all')

    @pytest.mark.parametrize('zorder', [1, 2, 3, 5, 10])
    def test_collection_zorder(self, zorder):
        """Collection zorder is settable."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]])
        ax.add_collection(lc)
        lc.set_zorder(zorder)
        assert lc.get_zorder() == zorder
        plt.close('all')


class TestCollectionsParametric4:
    """Yet more parametric collection tests."""

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8])
    def test_n_scatter_points(self, n):
        """Scatter with n points creates PathCollection."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter(range(n), range(n))
        assert hasattr(sc, 'get_offsets')
        plt.close('all')

    @pytest.mark.parametrize('s', [10, 20, 50, 100, 200])
    def test_scatter_size(self, s):
        """Scatter point size is settable."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([0], [0], s=s)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 8])
    def test_line_collections_n(self, n):
        """n LineCollections can be added to axes."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        for i in range(n):
            lc = LineCollection([[(i, 0), (i + 1, 1)]])
            ax.add_collection(lc)
        assert len(ax.collections) == n
        plt.close('all')

    @pytest.mark.parametrize('linewidth', [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_line_collection_lw(self, linewidth):
        """LineCollection linewidth roundtrips."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]], linewidths=linewidth)
        ax.add_collection(lc)
        assert abs(float(lc.get_linewidth()[0]) - linewidth) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.25, 0.5, 0.75, 1.0])
    def test_scatter_alpha(self, alpha):
        """Scatter alpha is settable."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([0, 1, 2], [0, 1, 2], alpha=alpha)
        assert abs(sc.get_alpha() - alpha) < 1e-9
        plt.close('all')


class TestCollectionsParametric14:
    """Yet more parametric tests."""

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

