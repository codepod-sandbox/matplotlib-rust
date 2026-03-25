"""
Upstream tests — batch 19.
Focus: Artist, Container, Cycler, Colormap (cm), and axes advanced operations.
Adapted from matplotlib upstream tests (no canvas rendering, no image comparison).
Note: Avoiding pytest.approx and np.arange(N) single-arg form.
"""
import math
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.container import (
    Container, BarContainer, ErrorbarContainer, StemContainer,
)
from matplotlib.cycler import Cycler, cycler
import matplotlib.cm as cm
from matplotlib.cm import get_cmap, ListedColormap, LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def close(a, b, tol=1e-10):
    """Check approximate equality."""
    return abs(a - b) < tol


# ------------------------------------------------------------------
# Artist tests
# ------------------------------------------------------------------

class TestArtist:
    def test_visible_default(self):
        a = Artist()
        assert a.get_visible()

    def test_visible_set(self):
        a = Artist()
        a.set_visible(False)
        assert not a.get_visible()
        a.set_visible(True)
        assert a.get_visible()

    def test_alpha_default(self):
        a = Artist()
        assert a.get_alpha() is None

    def test_alpha_set(self):
        a = Artist()
        a.set_alpha(0.5)
        assert close(a.get_alpha(), 0.5)

    def test_label_default(self):
        a = Artist()
        assert a.get_label() == ''

    def test_label_set(self):
        a = Artist()
        a.set_label('my label')
        assert a.get_label() == 'my label'

    def test_label_none(self):
        a = Artist()
        a.set_label(None)
        assert a.get_label() == '_nolegend_'

    def test_zorder_default(self):
        a = Artist()
        z = a.get_zorder()
        assert z is not None

    def test_zorder_set(self):
        a = Artist()
        a.set_zorder(5)
        assert a.get_zorder() == 5

    def test_clip_on(self):
        a = Artist()
        assert a.get_clip_on()
        a.set_clip_on(False)
        assert not a.get_clip_on()

    def test_set_method_batch(self):
        a = Artist()
        a.set(visible=False, alpha=0.3, zorder=2)
        assert not a.get_visible()
        assert close(a.get_alpha(), 0.3)
        assert a.get_zorder() == 2

    def test_update_from_dict(self):
        a = Artist()
        a.update({'visible': False, 'alpha': 0.7})
        assert not a.get_visible()
        assert close(a.get_alpha(), 0.7)

    def test_properties_returns_dict(self):
        a = Artist()
        props = a.properties()
        assert isinstance(props, dict)
        assert 'visible' in props
        assert 'alpha' in props

    def test_stale_default(self):
        a = Artist()
        assert a._stale

    def test_figure_none_default(self):
        a = Artist()
        assert a.figure is None

    def test_axes_none_default(self):
        a = Artist()
        assert a.axes is None


# ------------------------------------------------------------------
# Container tests
# ------------------------------------------------------------------

class TestContainer:
    def test_bar_container_length(self):
        patches = [Rectangle((i, 0), 0.8, i+1) for i in range(3)]
        bc = BarContainer(patches)
        assert len(bc) == 3

    def test_bar_container_iter(self):
        patches = [Rectangle((i, 0), 0.8, 1) for i in range(3)]
        bc = BarContainer(patches)
        for p in bc:
            assert isinstance(p, Rectangle)

    def test_bar_container_getitem(self):
        patches = [Rectangle((0, 0), 0.8, 1), Rectangle((1, 0), 0.8, 2)]
        bc = BarContainer(patches)
        assert bc[0] is patches[0]
        assert bc[1] is patches[1]

    def test_bar_container_patches(self):
        patches = [Rectangle((0, 0), 0.8, 1)]
        bc = BarContainer(patches)
        assert len(bc.patches) == 1

    def test_bar_container_label(self):
        patches = [Rectangle((0, 0), 0.8, 1)]
        bc = BarContainer(patches, label='mybar')
        assert bc.get_label() == 'mybar'

    def test_bar_container_errorbar(self):
        patches = [Rectangle((0, 0), 0.8, 1)]
        bc = BarContainer(patches, errorbar=None)
        assert bc.errorbar is None

    def test_errorbar_container(self):
        lines = [Line2D([0], [0]), []]
        ec = ErrorbarContainer(lines, has_xerr=False, has_yerr=True)
        assert ec is not None
        assert ec.has_yerr

    def test_errorbar_container_has_err(self):
        lines = [Line2D([0], [0]), [], []]
        ec = ErrorbarContainer(lines, has_xerr=True, has_yerr=True)
        assert ec.has_xerr
        assert ec.has_yerr

    def test_stem_container(self):
        markerline = Line2D([0, 1], [0, 1])
        stemlines = [Line2D([0, 0], [0, 1])]
        baseline = Line2D([0, 1], [0, 0])
        sc = StemContainer((markerline, stemlines, baseline))
        assert sc is not None

    def test_container_label(self):
        c = Container([])
        c.set_label('test')
        assert c.get_label() == 'test'


# ------------------------------------------------------------------
# Cycler tests
# ------------------------------------------------------------------

class TestCycler:
    def test_basic_creation(self):
        c = Cycler('color', ['r', 'g', 'b'])
        assert len(c) == 3

    def test_iteration(self):
        c = Cycler('color', ['r', 'g', 'b'])
        items = list(c)
        assert items[0] == {'color': 'r'}
        assert items[1] == {'color': 'g'}
        assert items[2] == {'color': 'b'}

    def test_by_key(self):
        c = Cycler('color', ['r', 'g', 'b'])
        bk = c.by_key()
        assert 'color' in bk
        assert bk['color'] == ['r', 'g', 'b']

    def test_keys(self):
        c = Cycler('color', ['r', 'g', 'b'])
        assert 'color' in c.keys

    def test_getitem(self):
        c = Cycler('linewidth', [1, 2, 3])
        assert c[0] == {'linewidth': 1}
        assert c[2] == {'linewidth': 3}

    def test_add_cyclers(self):
        c1 = Cycler('color', ['r', 'g'])
        c2 = Cycler('color', ['b', 'k'])
        c3 = c1 + c2
        assert len(c3) == 4

    def test_mul_int(self):
        c = Cycler('color', ['r', 'g'])
        c3 = c * 3
        assert len(c3) == 6

    def test_cycler_function_basic(self):
        c = cycler('color', ['r', 'g', 'b'])
        assert len(c) == 3

    def test_cycler_function_kwargs(self):
        c = cycler(color=['r', 'g', 'b'])
        assert len(c) == 3

    def test_repr(self):
        c = Cycler('color', ['r', 'g'])
        r = repr(c)
        assert 'cycler' in r

    def test_axes_prop_cycle(self):
        fig, ax = plt.subplots()
        c = cycler('color', ['red', 'green', 'blue'])
        ax.set_prop_cycle(c)
        # Plot to use colors from cycler
        lines = ax.plot([0, 1], [0, 1])
        assert len(lines) == 1


# ------------------------------------------------------------------
# Colormap tests
# ------------------------------------------------------------------

class TestColormaps:
    def test_get_cmap_viridis(self):
        cmap = get_cmap('viridis')
        assert cmap is not None
        assert cmap.name == 'viridis'

    def test_get_cmap_gray(self):
        cmap = get_cmap('gray')
        assert cmap is not None

    def test_get_cmap_jet(self):
        cmap = get_cmap('jet')
        assert cmap is not None

    def test_cmap_call_scalar(self):
        cmap = get_cmap('viridis')
        result = cmap(0.5)
        assert len(result) == 4
        for c in result:
            assert 0.0 <= c <= 1.0

    def test_cmap_call_zero(self):
        cmap = get_cmap('viridis')
        result = cmap(0.0)
        assert len(result) == 4

    def test_cmap_call_one(self):
        cmap = get_cmap('viridis')
        result = cmap(1.0)
        assert len(result) == 4

    def test_cmap_call_list(self):
        cmap = get_cmap('viridis')
        result = cmap([0.0, 0.5, 1.0])
        assert len(result) == 3

    def test_cmap_alpha(self):
        cmap = get_cmap('viridis')
        result = cmap(0.5, alpha=0.3)
        assert close(result[3], 0.3)

    def test_cmap_bytes(self):
        cmap = get_cmap('viridis')
        result = cmap(0.5, bytes=True)
        assert all(isinstance(c, int) for c in result)
        assert all(0 <= c <= 255 for c in result)

    def test_listed_colormap(self):
        colors = ['red', 'green', 'blue']
        cmap = ListedColormap(colors)
        assert cmap is not None
        result = cmap(0.0)
        assert len(result) == 4

    def test_listed_colormap_N(self):
        colors = ['red', 'green', 'blue', 'yellow']
        cmap = ListedColormap(colors, N=4)
        assert cmap.N == 4

    def test_linear_segmented_colormap_from_list(self):
        colors = ['red', 'blue']
        cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
        assert cmap is not None
        result = cmap(0.5)
        assert len(result) == 4

    def test_colormap_set_bad(self):
        cmap = get_cmap('viridis')
        cmap.set_bad('gray')
        bad = cmap.get_bad()
        assert len(bad) == 4

    def test_colormap_set_under(self):
        cmap = get_cmap('viridis')
        cmap.set_under('black')
        under = cmap.get_under()
        assert len(under) == 4

    def test_colormap_set_over(self):
        cmap = get_cmap('viridis')
        cmap.set_over('white')
        over = cmap.get_over()
        assert len(over) == 4

    def test_colormap_name(self):
        cmap = get_cmap('viridis')
        assert cmap.name == 'viridis'

    def test_colormap_repr(self):
        cmap = get_cmap('viridis')
        r = repr(cmap)
        assert 'viridis' in r

    def test_colormap_eq(self):
        cmap1 = get_cmap('viridis')
        cmap2 = get_cmap('viridis')
        assert cmap1 == cmap2

    def test_colormap_copy(self):
        cmap = get_cmap('viridis')
        cmap2 = cmap.copy()
        # The copy may be same or different object depending on implementation
        assert cmap2 == cmap


# ------------------------------------------------------------------
# ScalarMappable tests
# ------------------------------------------------------------------

class TestScalarMappable:
    def test_scatter_with_cmap(self):
        fig, ax = plt.subplots()
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 2, 5, 3]
        c = [0.1, 0.3, 0.5, 0.7, 0.9]
        sc = ax.scatter(x, y, c=c, cmap='viridis')
        assert sc is not None

    def test_scatter_with_norm(self):
        fig, ax = plt.subplots()
        x = [1, 2, 3]
        y = [1, 2, 3]
        norm = Normalize(vmin=0, vmax=1)
        sc = ax.scatter(x, y, c=[0.2, 0.5, 0.8], norm=norm)
        assert sc is not None


# ------------------------------------------------------------------
# Advanced axes operations
# ------------------------------------------------------------------

class TestAdvancedAxes:
    def test_twinx(self):
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        assert ax2 is not None

    def test_twiny(self):
        fig, ax = plt.subplots()
        ax2 = ax.twiny()
        assert ax2 is not None

    def test_add_patch(self):
        fig, ax = plt.subplots()
        r = Rectangle((0, 0), 1, 1)
        ax.add_patch(r)
        assert len(ax.patches) >= 1

    def test_add_line(self):
        fig, ax = plt.subplots()
        line = Line2D([0, 1], [0, 1])
        ax.add_line(line)
        assert len(ax.lines) >= 1

    def test_get_children(self):
        fig, ax = plt.subplots()
        children = ax.get_children()
        assert isinstance(children, list)

    def test_findobj(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        objs = ax.findobj()
        assert isinstance(objs, list)

    def test_relim(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.relim()  # Should not error

    def test_autoscale(self):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.autoscale()  # Should not error

    def test_minorticks_on(self):
        fig, ax = plt.subplots()
        ax.minorticks_on()  # Should not error

    def test_minorticks_off(self):
        fig, ax = plt.subplots()
        ax.minorticks_off()  # Should not error

    def test_set_xmargin(self):
        fig, ax = plt.subplots()
        ax.margins(x=0.1)

    def test_set_ymargin(self):
        fig, ax = plt.subplots()
        ax.margins(y=0.1)

    def test_margins(self):
        fig, ax = plt.subplots()
        ax.margins(0.05)

    def test_format_coord(self):
        fig, ax = plt.subplots()
        result = ax.format_coord(0.5, 0.5)
        assert isinstance(result, str)

    def test_label_outer_topleft(self):
        fig, axes = plt.subplots(2, 2)
        axes[0][0].label_outer()

    def test_remove_artist(self):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        n_before = len(ax.lines)
        ax.lines.remove(line)
        assert len(ax.lines) == n_before - 1

    def test_get_lines(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.plot([0, 1], [1, 0])
        assert len(ax.lines) == 2

    def test_set_prop_cycle_color_list(self):
        fig, ax = plt.subplots()
        ax.set_prop_cycle('color', ['red', 'blue', 'green'])
        line1 = ax.plot([0, 1], [0, 1])[0]
        line2 = ax.plot([0, 1], [1, 0])[0]
        # Colors should come from the cycle

    def test_axis_off(self):
        fig, ax = plt.subplots()
        ax.axis('off')

    def test_axis_on(self):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.axis('on')

    def test_invert_xaxis_xaxis_method(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.xaxis.set_inverted(True)
        assert ax.xaxis.get_inverted()

    def test_tick_top(self):
        fig, ax = plt.subplots()
        ax.xaxis.tick_top()

    def test_tick_bottom(self):
        fig, ax = plt.subplots()
        ax.xaxis.tick_bottom()

    def test_tick_left(self):
        fig, ax = plt.subplots()
        ax.yaxis.tick_left()

    def test_tick_right(self):
        fig, ax = plt.subplots()
        ax.yaxis.tick_right()


# ===================================================================
# Extended parametric tests for batch19
# ===================================================================

class TestBatch19Parametric:
    """Parametric tests for artist, container, cycler."""

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_artist_alpha(self, alpha):
        """Artist alpha is stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('zorder', [0, 1, 2, 5, 10])
    def test_artist_zorder(self, zorder):
        """Artist zorder is stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_zorder(zorder)
        assert line.get_zorder() == zorder
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_artist_visibility(self, visible):
        """Artist visibility is stored."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_visible(visible)
        assert line.get_visible() == visible
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_bar_container_n(self, n):
        """BarContainer has n patches."""
        fig, ax = plt.subplots()
        container = ax.bar(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('colors', [
        ['r', 'g', 'b'],
        ['red', 'blue', 'green', 'orange'],
        ['cyan', 'magenta'],
    ])
    def test_cycler_colors_len(self, colors):
        """Cycler length matches colors."""
        c = cycler(color=colors)
        assert len(c) == len(colors)

    @pytest.mark.parametrize('key', ['color', 'linewidth', 'marker', 'linestyle'])
    def test_cycler_key_stored(self, key):
        """Cycler key is in .keys."""
        c = cycler(key, [1, 2, 3])
        assert key in c.keys

    @pytest.mark.parametrize('n', [2, 3, 4, 5])
    def test_cycler_iteration(self, n):
        """Cycler iterates n times."""
        c = cycler(color=range(n))
        assert len(list(c)) == n

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100)])
    def test_xlim(self, xmin, xmax):
        """set_xlim/get_xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_hist_bins(self, bins):
        """hist bins count is correct."""
        fig, ax = plt.subplots()
        n_counts, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n_counts) == bins
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale(self, scale):
        """set_xscale/get_xscale roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')


class TestBatch19Parametric2:
    """More parametric tests."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_n_lines(self, n):
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('lo,hi', [(0, 1), (-1, 1), (0, 100)])
    def test_xlim(self, lo, hi):
        fig, ax = plt.subplots()
        ax.set_xlim(lo, hi)
        assert ax.get_xlim() == (lo, hi)
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale(self, scale):
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('title', ['Title', 'Test', ''])
    def test_title(self, title):
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0])
    def test_linewidth(self, lw):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'D'])
    def test_marker(self, marker):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close('all')

    @pytest.mark.parametrize('n', [2, 3, 5])
    def test_bar_patches(self, n):
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_hist(self, bins):
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close('all')

    @pytest.mark.parametrize('aspect', ['equal', 'auto'])
    def test_aspect(self, aspect):
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_visible(self, visible):
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close('all')


class TestBatch19Parametric9:
    """Further parametric tests for batch 19."""

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



class TestBatch19Parametric13:
    """Yet more parametric tests for batch 19."""

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

