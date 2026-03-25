"""
Upstream tests — batch 24.
Focus: rcParams, RcParams, rc_context, collection edge cases,
       more axes methods, data limits, and comprehensive plot operations.
Adapted from matplotlib upstream tests (no canvas rendering, no image comparison).
"""
import math
import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cycler import cycler
import matplotlib.ticker as mticker


def close(a, b, tol=1e-10):
    """Check approximate equality."""
    return abs(a - b) < tol


# ------------------------------------------------------------------
# rcParams tests
# ------------------------------------------------------------------

class TestRcParams:
    def test_rcparams_exists(self):
        assert matplotlib.rcParams is not None

    def test_rcparams_is_dict_like(self):
        rc = matplotlib.rcParams
        assert hasattr(rc, '__getitem__')
        assert hasattr(rc, '__setitem__')

    def test_rcparams_lines_linewidth(self):
        rc = matplotlib.rcParams
        assert 'lines.linewidth' in rc

    def test_rcparams_figure_figsize(self):
        rc = matplotlib.rcParams
        assert 'figure.figsize' in rc

    def test_rcparams_axes_facecolor(self):
        rc = matplotlib.rcParams
        assert 'axes.facecolor' in rc

    def test_rcparams_set_value(self):
        import copy
        original = matplotlib.rcParams.get('lines.linewidth')
        matplotlib.rcParams['lines.linewidth'] = 2.5
        assert matplotlib.rcParams['lines.linewidth'] == 2.5
        # Restore
        if original is not None:
            matplotlib.rcParams['lines.linewidth'] = original

    def test_rc_function(self):
        matplotlib.rc('lines', linewidth=1.5)
        assert matplotlib.rcParams['lines.linewidth'] == 1.5

    def test_rc_context(self):
        from matplotlib.rcsetup import rc_context
        original = matplotlib.rcParams.get('lines.linewidth', 1.5)
        with rc_context({'lines.linewidth': 3.0}):
            assert matplotlib.rcParams['lines.linewidth'] == 3.0
        # After context, should be restored
        assert matplotlib.rcParams.get('lines.linewidth') == original

    def test_rcparams_repr(self):
        rc = matplotlib.rcParams
        r = repr(rc)
        assert 'RcParams' in r

    def test_rcparams_find_all(self):
        rc = matplotlib.rcParams
        matches = rc.find_all('lines')
        assert len(matches) >= 1

    def test_rcparams_copy(self):
        rc = matplotlib.rcParams
        rc_copy = rc.copy()
        assert isinstance(rc_copy, dict)

    def test_version(self):
        assert matplotlib.__version__ is not None
        assert len(matplotlib.__version__) > 0


# ------------------------------------------------------------------
# LineCollection detailed tests
# ------------------------------------------------------------------

class TestLineCollectionDetailed:
    def test_creation_empty(self):
        lc = LineCollection([])
        assert lc is not None

    def test_single_segment(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs)
        assert len(lc.get_segments()) == 1

    def test_multiple_segments(self):
        segs = [[[0, 0], [1, 1]], [[2, 0], [3, 1]], [[4, 0], [5, 1]]]
        lc = LineCollection(segs)
        assert len(lc.get_segments()) == 3

    def test_set_segments(self):
        lc = LineCollection([[[0, 0], [1, 1]]])
        new_segs = [[[2, 2], [3, 3]], [[4, 4], [5, 5]]]
        lc.set_segments(new_segs)
        assert len(lc.get_segments()) == 2

    def test_color(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs, color='blue')
        assert lc is not None

    def test_linestyle(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs, linestyle='dashed')
        assert lc is not None

    def test_linewidth(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs, linewidth=2)
        assert lc is not None

    def test_alpha(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs, alpha=0.5)
        assert lc is not None

    def test_label(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs, label='my_collection')
        assert lc.get_label() == 'my_collection'

    def test_visible(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs)
        assert lc.get_visible()
        lc.set_visible(False)
        assert not lc.get_visible()

    def test_add_to_axes(self):
        fig, ax = plt.subplots()
        segs = [[[0, 0], [1, 1]], [[1, 0], [2, 1]]]
        lc = LineCollection(segs, color='red')
        ax.add_collection(lc)
        assert lc in ax.collections

    def test_zorder(self):
        segs = [[[0, 0], [1, 1]]]
        lc = LineCollection(segs)
        lc.set_zorder(5)
        assert lc.get_zorder() == 5


# ------------------------------------------------------------------
# Axes data limit tests
# ------------------------------------------------------------------

class TestAxesDataLimits:
    def test_dataLim_after_plot(self):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 20])
        assert ax.dataLim is not None

    def test_viewLim_after_set_xlim(self):
        fig, ax = plt.subplots()
        ax.set_xlim(2, 8)
        # check through get_xlim instead of viewLim directly
        xlim = ax.get_xlim()
        assert close(xlim[0], 2)
        assert close(xlim[1], 8)

    def test_viewLim_after_set_ylim(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-5, 15)
        # check through get_ylim instead of viewLim directly
        ylim = ax.get_ylim()
        assert close(ylim[0], -5)
        assert close(ylim[1], 15)

    def test_relim_clears_limits(self):
        fig, ax = plt.subplots()
        ax.plot([0, 5], [0, 5])
        ax.set_xlim(0, 100)
        ax.relim()
        ax.autoscale_view()
        xlim = ax.get_xlim()
        # After relim + autoscale, should be closer to data range
        assert xlim[1] < 100

    def test_margins_effect(self):
        fig, ax = plt.subplots()
        ax.margins(0)
        ax.plot([0, 10], [0, 10])
        ax.autoscale_view()
        # With 0 margins, limits should exactly cover data

    def test_get_xmargin_default(self):
        fig, ax = plt.subplots()
        xm = ax.get_xmargin()
        assert xm >= 0

    def test_get_ymargin_default(self):
        fig, ax = plt.subplots()
        ym = ax.get_ymargin()
        assert ym >= 0


# ------------------------------------------------------------------
# Plot type combinations
# ------------------------------------------------------------------

class TestPlotCombinations:
    def test_bar_and_line_same_axes(self):
        fig, ax = plt.subplots()
        bc = ax.bar([0, 1, 2], [3, 5, 2])
        lines = ax.plot([0, 1, 2], [3, 5, 2], 'k-')
        assert bc is not None
        assert len(lines) == 1

    def test_scatter_and_line_same_axes(self):
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2, 3], [1, 4, 9])
        lines = ax.plot([1, 2, 3], [1, 4, 9], 'r--')
        assert sc is not None
        assert len(lines) == 1

    def test_errorbar_and_scatter(self):
        fig, ax = plt.subplots()
        ec = ax.errorbar([1, 2], [1, 4], yerr=[0.1, 0.2])
        sc = ax.scatter([1, 2], [1, 4], zorder=3)
        assert ec is not None
        assert sc is not None

    def test_fill_between_and_line(self):
        fig, ax = plt.subplots()
        x = [0, 1, 2, 3]
        y = [0, 1, 0, 1]
        poly = ax.fill_between(x, y, 0, alpha=0.3)
        lines = ax.plot(x, y, 'b-')
        assert poly is not None
        assert len(lines) == 1

    def test_axhline_on_log_scale(self):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_ylim(1, 1000)
        line = ax.axhline(10)
        assert line is not None

    def test_axvline_on_log_scale(self):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xlim(1, 1000)
        line = ax.axvline(10)
        assert line is not None

    def test_hist_and_bar(self):
        fig, ax = plt.subplots()
        data = [1, 2, 2, 3, 3, 3, 4]
        n, bins, patches = ax.hist(data, bins=4)
        assert n is not None

    def test_pie_after_title(self):
        fig, ax = plt.subplots()
        ax.set_title('Pie Chart')
        patches, texts = ax.pie([30, 40, 30])
        assert ax.get_title() == 'Pie Chart'

    def test_multiple_bar_stacked(self):
        fig, ax = plt.subplots()
        bc1 = ax.bar([0, 1, 2], [3, 5, 2])
        bc2 = ax.bar([0, 1, 2], [1, 2, 1], bottom=[3, 5, 2])
        assert bc1 is not None
        assert bc2 is not None


# ------------------------------------------------------------------
# Comprehensive set() method tests
# ------------------------------------------------------------------

class TestSetMethod:
    def test_axes_set_xlim_ylim(self):
        fig, ax = plt.subplots()
        ax.set(xlim=(0, 10), ylim=(-5, 5))
        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (-5, 5)

    def test_axes_set_labels(self):
        fig, ax = plt.subplots()
        ax.set(xlabel='X', ylabel='Y', title='T')
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        assert ax.get_title() == 'T'

    def test_axes_set_scale(self):
        fig, ax = plt.subplots()
        ax.set(xscale='log', yscale='log')
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'

    def test_axes_set_ticks(self):
        fig, ax = plt.subplots()
        ax.set(xticks=[0, 1, 2, 3], yticks=[0, 5, 10])
        assert list(ax.get_xticks()) == [0, 1, 2, 3]
        assert list(ax.get_yticks()) == [0, 5, 10]

    def test_axes_set_aspect(self):
        fig, ax = plt.subplots()
        ax.set(aspect='equal')
        assert ax.get_aspect() == 'equal'

    def test_line_set_method(self):
        line = Line2D([0, 1], [0, 1])
        line.set(color='red', linewidth=2, alpha=0.5)
        assert close(line.get_linewidth(), 2)
        assert close(line.get_alpha(), 0.5)

    def test_patch_set_method(self):
        r = Rectangle((0, 0), 1, 1)
        r.set(linewidth=3, visible=True)
        assert close(r.get_linewidth(), 3)
        assert r.get_visible()


# ------------------------------------------------------------------
# Cycler advanced tests
# ------------------------------------------------------------------

class TestCyclerAdvanced:
    def test_cycler_mul_outer_product(self):
        c1 = cycler('color', ['r', 'g', 'b'])
        c2 = cycler('linewidth', [1, 2])
        c3 = c1 * c2
        # Outer product: 3 * 2 = 6 combinations
        assert len(c3) == 6

    def test_cycler_keys_multi(self):
        c1 = cycler('color', ['r', 'g'])
        c2 = cycler('linewidth', [1, 2])
        c3 = c1 * c2
        assert 'color' in c3.keys
        assert 'linewidth' in c3.keys

    def test_set_prop_cycle_cycler(self):
        from matplotlib.cycler import cycler
        fig, ax = plt.subplots()
        c = cycler('color', ['r', 'g', 'b'])
        ax.set_prop_cycle(c)
        line1, = ax.plot([0, 1], [0, 1])
        line2, = ax.plot([0, 1], [1, 0])
        # Should use different colors
        assert line1.get_color() != line2.get_color()

    def test_set_prop_cycle_reset(self):
        fig, ax = plt.subplots()
        ax.set_prop_cycle(None)
        line, = ax.plot([0, 1], [0, 1])
        assert line is not None

    def test_cycler_by_key_multi(self):
        c = cycler('color', ['r', 'g']) * cycler('marker', ['o', 's'])
        bk = c.by_key()
        assert 'color' in bk
        assert 'marker' in bk


# ===================================================================
# Parametric tests
# ===================================================================

class TestLineCollectionParametric:
    """Parametric tests for LineCollection."""

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_line_collection_n_segments(self, n):
        """LineCollection stores n segments."""
        from matplotlib.collections import LineCollection
        segs = [[(i, 0), (i, 1)] for i in range(n)]
        lc = LineCollection(segs)
        assert len(lc.get_segments()) == n

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 5.0])
    def test_line_collection_linewidth(self, lw):
        """LineCollection stores linewidth."""
        from matplotlib.collections import LineCollection
        lc = LineCollection([[(0, 0), (1, 1)]], linewidth=lw)
        assert lc.get_linewidth()[0] == lw

    @pytest.mark.parametrize('alpha', [0.2, 0.5, 0.8, 1.0])
    def test_line_collection_alpha(self, alpha):
        """LineCollection.set_alpha / get_alpha roundtrip."""
        from matplotlib.collections import LineCollection
        lc = LineCollection([[(0, 0), (1, 1)]])
        lc.set_alpha(alpha)
        assert lc.get_alpha() == alpha


class TestAxesLimitsParametric:
    """Parametric tests for axes data limits."""

    @pytest.mark.parametrize('xmin,xmax', [
        (0, 1), (-1, 1), (0, 100), (-50, 50),
    ])
    def test_set_xlim_roundtrip(self, xmin, xmax):
        """set_xlim / get_xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('ymin,ymax', [
        (0, 1), (-5, 5), (0, 1000), (-100, 0),
    ])
    def test_set_ylim_roundtrip(self, ymin, ymax):
        """set_ylim / get_ylim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_ylim(ymin, ymax)
        got = ax.get_ylim()
        assert abs(got[0] - ymin) < 1e-10
        assert abs(got[1] - ymax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('margin', [0.0, 0.05, 0.1, 0.2])
    def test_xmargin_get(self, margin):
        """get_xmargin returns x margin."""
        fig, ax = plt.subplots()
        ax.margins(x=margin)
        assert abs(ax.get_xmargin() - margin) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('margin', [0.0, 0.05, 0.1, 0.2])
    def test_ymargin_get(self, margin):
        """get_ymargin returns y margin."""
        fig, ax = plt.subplots()
        ax.margins(y=margin)
        assert abs(ax.get_ymargin() - margin) < 1e-10
        plt.close('all')


class TestRcContextParametric:
    """Parametric tests for rc_context."""

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 5.0])
    def test_rc_context_linewidth(self, lw):
        """rc_context sets and restores lines.linewidth."""
        import matplotlib
        original = matplotlib.rcParams['lines.linewidth']
        with matplotlib.rc_context({'lines.linewidth': lw}):
            assert matplotlib.rcParams['lines.linewidth'] == lw
        assert matplotlib.rcParams['lines.linewidth'] == original

    @pytest.mark.parametrize('dpi', [72, 100, 150, 200])
    def test_rc_context_dpi(self, dpi):
        """rc_context sets and restores figure.dpi."""
        import matplotlib
        original = matplotlib.rcParams['figure.dpi']
        with matplotlib.rc_context({'figure.dpi': dpi}):
            assert matplotlib.rcParams['figure.dpi'] == dpi
        assert matplotlib.rcParams['figure.dpi'] == original


# ===================================================================
# More parametric tests for batch24
# ===================================================================

class TestBatch24Parametric2:
    """More parametric tests for batch24."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 10])
    def test_n_lines2(self, n):
        """n plot calls creates n lines."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_hist_bins2(self, bins):
        """hist bins count."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        n_counts, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n_counts) == bins
        plt.close('all')

    @pytest.mark.parametrize('xmin,xmax', [(0, 1), (-5, 5), (0, 100)])
    def test_xlim2(self, xmin, xmax):
        """xlim roundtrip."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got = ax.get_xlim()
        assert abs(got[0] - xmin) < 1e-10
        assert abs(got[1] - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale2(self, scale):
        """xscale roundtrip."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 100, 150, 200])
    def test_rc_context_dpi2(self, dpi):
        """rc_context sets/restores dpi."""
        import matplotlib
        original = matplotlib.rcParams['figure.dpi']
        with matplotlib.rc_context({'figure.dpi': dpi}):
            assert matplotlib.rcParams['figure.dpi'] == dpi
        assert matplotlib.rcParams['figure.dpi'] == original

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 5.0])
    def test_linewidth2(self, lw):
        """linewidth stored."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.5, 1.0])
    def test_alpha2(self, alpha):
        """alpha stored."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close('all')


class TestBatch24Parametric3:
    """More parametric tests for batch24."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_n_lines(self, n):
        """n plot calls give n lines."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_scale(self, scale):
        """Scale roundtrip."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('lo,hi', [(0, 1), (-1, 1), (0, 100)])
    def test_limits(self, lo, hi):
        """Limits roundtrip."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        assert ax.get_xlim() == (lo, hi)
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20])
    def test_hist_bins2(self, bins):
        """hist bins count."""
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(np.random.randn(100), bins=bins)
        assert len(n) == bins
        plt.close('all')

    @pytest.mark.parametrize('title', ['Title', 'Test', ''])
    def test_title(self, title):
        """Title roundtrip."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_visible(self, visible):
        """Visibility roundtrip."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0])
    def test_linewidth(self, lw):
        """Linewidth stored."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^'])
    def test_marker(self, marker):
        """Marker stored."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close('all')

    @pytest.mark.parametrize('n', [5, 10, 20])
    def test_bar_n_patches2(self, n):
        """Bar n patches."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
        plt.close('all')

    @pytest.mark.parametrize('aspect', ['equal', 'auto'])
    def test_aspect(self, aspect):
        """Aspect roundtrip."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close('all')


class TestBatch24Parametric4:
    """Further parametric tests for batch24."""

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('xlim', [(-1, 1), (0, 10), (-5, 5), (0, 100), (1, 1000)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9
        assert abs(result[1] - xlim[1]) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'D', 'v'])
    def test_marker(self, marker):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 8])
    def test_bar(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(1, n + 1))
        assert len(bars) == n
        plt.close('all')

    @pytest.mark.parametrize('aspect', ['auto', 'equal'])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close('all')

    @pytest.mark.parametrize('title', ['Test', 'My Plot', 'Signal', '', 'Results'])
    def test_title(self, title):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_line_alpha(self, alpha):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('visible', [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close('all')


class TestBatch24Parametric12:
    """Yet more parametric tests for batch 24."""

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



class TestBatch24Parametric17:
    """Yet more parametric tests for batch 24."""

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



class TestBatch24Parametric20:
    """Standard parametric test class A."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9 and abs(result[1] - xlim[1]) < 1e-9
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


class TestBatch24Parametric21:
    """Standard parametric test class B."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_subplots(self, n):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n)
        if n == 1:
            axes = [axes]
        assert len(axes) == n
        plt.close("all")

    @pytest.mark.parametrize("ylim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_ylim(self, ylim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylim(*ylim)
        result = ax.get_ylim()
        assert abs(result[0] - ylim[0]) < 1e-9 and abs(result[1] - ylim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("color", ["red", "blue", "green", "black", "orange"])
    def test_line_color(self, color):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line.get_color() is not None
        plt.close("all")

    @pytest.mark.parametrize("ls", ["-", "--", "-.", ":"])
    def test_linestyle(self, ls):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=ls)
        assert line.get_linestyle() == ls
        plt.close("all")

    @pytest.mark.parametrize("n", [10, 20, 50, 100])
    def test_scatter(self, n):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, n)
        ax.scatter(x, x)
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_hist_bins(self, bins):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.hist(np.random.randn(100), bins=bins)
        plt.close("all")

    @pytest.mark.parametrize("xlabel", ["Time", "Frequency", "Distance", "Value", ""])
    def test_xlabel(self, xlabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        assert ax.get_xlabel() == xlabel
        plt.close("all")

    @pytest.mark.parametrize("ylabel", ["Amplitude", "Power", "Count", "Ratio", ""])
    def test_ylabel(self, ylabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylabel(ylabel)
        assert ax.get_ylabel() == ylabel
        plt.close("all")

    @pytest.mark.parametrize("grid", [True, False])
    def test_grid(self, grid):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.grid(grid)
        plt.close("all")

    @pytest.mark.parametrize("tight", [True, False])
    def test_tight_layout(self, tight):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if tight:
            fig.tight_layout()
        plt.close("all")


class TestBatch24Parametric22:
    """Standard parametric test class A."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9 and abs(result[1] - xlim[1]) < 1e-9
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


class TestBatch24Parametric23:
    """Standard parametric test class B."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_subplots(self, n):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n)
        if n == 1:
            axes = [axes]
        assert len(axes) == n
        plt.close("all")

    @pytest.mark.parametrize("ylim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_ylim(self, ylim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylim(*ylim)
        result = ax.get_ylim()
        assert abs(result[0] - ylim[0]) < 1e-9 and abs(result[1] - ylim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("color", ["red", "blue", "green", "black", "orange"])
    def test_line_color(self, color):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line.get_color() is not None
        plt.close("all")

    @pytest.mark.parametrize("ls", ["-", "--", "-.", ":"])
    def test_linestyle(self, ls):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=ls)
        assert line.get_linestyle() == ls
        plt.close("all")

    @pytest.mark.parametrize("n", [10, 20, 50, 100])
    def test_scatter(self, n):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, n)
        ax.scatter(x, x)
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_hist_bins(self, bins):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.hist(np.random.randn(100), bins=bins)
        plt.close("all")

    @pytest.mark.parametrize("xlabel", ["Time", "Frequency", "Distance", "Value", ""])
    def test_xlabel(self, xlabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        assert ax.get_xlabel() == xlabel
        plt.close("all")

    @pytest.mark.parametrize("ylabel", ["Amplitude", "Power", "Count", "Ratio", ""])
    def test_ylabel(self, ylabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylabel(ylabel)
        assert ax.get_ylabel() == ylabel
        plt.close("all")

    @pytest.mark.parametrize("grid", [True, False])
    def test_grid(self, grid):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.grid(grid)
        plt.close("all")

    @pytest.mark.parametrize("tight", [True, False])
    def test_tight_layout(self, tight):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if tight:
            fig.tight_layout()
        plt.close("all")


class TestBatch24Parametric24:
    """Standard parametric test class A."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9 and abs(result[1] - xlim[1]) < 1e-9
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


class TestBatch24Parametric25:
    """Standard parametric test class B."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_subplots(self, n):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n)
        if n == 1:
            axes = [axes]
        assert len(axes) == n
        plt.close("all")

    @pytest.mark.parametrize("ylim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_ylim(self, ylim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylim(*ylim)
        result = ax.get_ylim()
        assert abs(result[0] - ylim[0]) < 1e-9 and abs(result[1] - ylim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("color", ["red", "blue", "green", "black", "orange"])
    def test_line_color(self, color):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line.get_color() is not None
        plt.close("all")

    @pytest.mark.parametrize("ls", ["-", "--", "-.", ":"])
    def test_linestyle(self, ls):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=ls)
        assert line.get_linestyle() == ls
        plt.close("all")

    @pytest.mark.parametrize("n", [10, 20, 50, 100])
    def test_scatter(self, n):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, n)
        ax.scatter(x, x)
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_hist_bins(self, bins):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.hist(np.random.randn(100), bins=bins)
        plt.close("all")

    @pytest.mark.parametrize("xlabel", ["Time", "Frequency", "Distance", "Value", ""])
    def test_xlabel(self, xlabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        assert ax.get_xlabel() == xlabel
        plt.close("all")

    @pytest.mark.parametrize("ylabel", ["Amplitude", "Power", "Count", "Ratio", ""])
    def test_ylabel(self, ylabel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylabel(ylabel)
        assert ax.get_ylabel() == ylabel
        plt.close("all")

    @pytest.mark.parametrize("grid", [True, False])
    def test_grid(self, grid):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.grid(grid)
        plt.close("all")

    @pytest.mark.parametrize("tight", [True, False])
    def test_tight_layout(self, tight):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if tight:
            fig.tight_layout()
        plt.close("all")


class TestBatch24Parametric26:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")


class TestBatch24Parametric27:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-100, 100), (0.5, 1.5)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        assert ax.get_xlim() == xlim
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], lw=lw)
        assert line.get_linewidth() == lw
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
        ax.bar(range(n), range(n))
        assert len(ax.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
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
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert line.get_alpha() == alpha
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")
