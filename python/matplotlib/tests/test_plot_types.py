"""Tests for new plot types."""

import math


class TestDrawWedge:
    def test_renderer_base_raises(self):
        from matplotlib.backend_bases import RendererBase
        r = RendererBase(100, 100, 72)
        try:
            r.draw_wedge(50, 50, 40, 0, 90, '#ff0000')
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass

    def test_svg_draw_wedge_quarter(self):
        from matplotlib._svg_backend import RendererSVG
        r = RendererSVG(200, 200, 72)
        r.draw_wedge(100, 100, 50, 0, 90, '#ff0000')
        svg = r.get_result()
        assert '<path' in svg
        assert '#ff0000' in svg

    def test_svg_draw_wedge_full_circle(self):
        from matplotlib._svg_backend import RendererSVG
        r = RendererSVG(200, 200, 72)
        r.draw_wedge(100, 100, 50, 0, 360, '#00ff00')
        svg = r.get_result()
        assert '#00ff00' in svg
        assert '<circle' in svg

    def test_pil_draw_wedge(self):
        from matplotlib._pil_backend import RendererPIL
        r = RendererPIL(200, 200, 72)
        r.draw_wedge(100, 100, 50, 0, 90, '#ff0000')
        result = r.get_result()
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestWedgePatch:
    def test_wedge_creation(self):
        from matplotlib.patches import Wedge
        w = Wedge((0, 0), 1.0, 0, 90)
        assert w._center == (0, 0)
        assert w._r == 1.0
        assert w._theta1 == 0
        assert w._theta2 == 90

    def test_wedge_color(self):
        from matplotlib.patches import Wedge
        w = Wedge((0, 0), 1.0, 0, 90, facecolor='red')
        fc = w.get_facecolor()
        assert fc[0] == 1.0  # red channel

    def test_wedge_draw(self):
        from matplotlib.patches import Wedge
        from matplotlib._svg_backend import RendererSVG
        from matplotlib.backend_bases import AxesLayout
        w = Wedge((5, 5), 3.0, 0, 180, facecolor='blue')
        renderer = RendererSVG(200, 200, 72)
        layout = AxesLayout(10, 10, 180, 180, 0, 10, 0, 10)
        w.draw(renderer, layout)
        svg = renderer.get_result()
        assert '<path' in svg or '<circle' in svg


class TestStemContainer:
    def test_stem_container_creation(self):
        from matplotlib.container import StemContainer
        sc = StemContainer(('marker', ['s1', 's2'], 'base'), label='test')
        assert sc.markerline == 'marker'
        assert sc.stemlines == ['s1', 's2']
        assert sc.baseline == 'base'
        assert sc.get_label() == 'test'


class TestStep:
    def test_step_pre(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = ax.step([1, 2, 3], [1, 4, 2], where='pre')
        assert len(lines) == 1
        line = lines[0]
        xd = line.get_xdata()
        yd = line.get_ydata()
        assert len(xd) == 5

    def test_step_post(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = ax.step([1, 2, 3], [1, 4, 2], where='post')
        line = lines[0]
        xd = line.get_xdata()
        assert len(xd) == 5

    def test_step_mid(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = ax.step([1, 2, 3], [1, 4, 2], where='mid')
        line = lines[0]
        xd = line.get_xdata()
        assert len(xd) == 7

    def test_step_returns_line_list(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.step([0, 1], [0, 1])
        assert isinstance(result, list)

    def test_step_invalid_where(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            ax.step([1, 2], [1, 2], where='invalid')
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestStairs:
    def test_stairs_basic(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line = ax.stairs([3, 2, 5, 1])
        xd = line.get_xdata()
        yd = line.get_ydata()
        assert len(xd) == 2 * 4

    def test_stairs_with_edges(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line = ax.stairs([3, 2, 5], [10, 20, 30, 40])
        xd = line.get_xdata()
        assert xd[0] == 10
        assert xd[-1] == 40

    def test_stairs_is_line(self):
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stairs([1, 2, 3])
        assert isinstance(result, Line2D)


class TestStackplot:
    def test_stackplot_basic(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stackplot([1, 2, 3], [1, 2, 3], [2, 1, 2])
        assert len(result) == 2

    def test_stackplot_returns_polygons(self):
        from matplotlib.patches import Polygon
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stackplot([1, 2, 3], [1, 2, 3])
        assert len(result) == 1
        assert isinstance(result[0], Polygon)

    def test_stackplot_labels(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stackplot([1, 2, 3], [1, 2, 3], [2, 1, 2],
                              labels=['A', 'B'])
        assert result[0].get_label() == 'A'
        assert result[1].get_label() == 'B'

    def test_stackplot_cumulative(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        polys = ax.stackplot([1, 2], [10, 20], [5, 10])
        assert len(ax.patches) >= 2


class TestStem:
    def test_stem_basic(self):
        from matplotlib.container import StemContainer
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stem([1, 2, 3], [4, 5, 6])
        assert isinstance(result, StemContainer)

    def test_stem_has_markerline(self):
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2, 3], [4, 5, 6])
        assert isinstance(sc.markerline, Line2D)

    def test_stem_has_baseline(self):
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2, 3], [4, 5, 6])
        assert isinstance(sc.baseline, Line2D)

    def test_stem_stemlines_count(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2, 3], [4, 5, 6])
        assert len(sc.stemlines) == 3

    def test_stem_y_only(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.stem([4, 5, 6])
        assert sc.markerline.get_xdata() == [0, 1, 2]

    def test_stem_custom_bottom(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2], [3, 4], bottom=1)
        assert sc.baseline.get_ydata() == [1, 1]


class TestPie:
    def test_pie_basic(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.pie([1, 2, 3])
        wedges, texts = result[0], result[1]
        assert len(wedges) == 3

    def test_pie_wedge_angles_sum_to_360(self):
        from matplotlib.patches import Wedge
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        wedges, texts = ax.pie([1, 1, 1, 1])
        assert len(wedges) == 4
        for w in wedges:
            assert isinstance(w, Wedge)
            span = w._theta2 - w._theta1
            assert abs(span - 90.0) < 0.01

    def test_pie_labels(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        wedges, texts = ax.pie([1, 2], labels=['A', 'B'])
        assert len(texts) == 2
        assert texts[0].get_text() == 'A'
        assert texts[1].get_text() == 'B'

    def test_pie_colors(self):
        from matplotlib.colors import to_hex
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        wedges, texts = ax.pie([1, 1], colors=['red', 'blue'])
        assert to_hex(wedges[0]._facecolor) == to_hex('red')
        assert to_hex(wedges[1]._facecolor) == to_hex('blue')

    def test_pie_startangle(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        wedges, texts = ax.pie([1], startangle=90)
        assert wedges[0]._theta1 == 90

    def test_pie_autopct(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie([1, 3], autopct='%1.0f%%')
        assert len(autotexts) == 2
        assert autotexts[0].get_text() == '25%'
        assert autotexts[1].get_text() == '75%'

    def test_pie_sets_equal_aspect(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3])
        assert ax.get_aspect() == 'equal'


class TestBoxplot:
    def test_boxplot_single(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.boxplot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert 'boxes' in result
        assert 'medians' in result
        assert 'whiskers' in result
        assert len(result['boxes']) == 1
        assert len(result['medians']) == 1

    def test_boxplot_multiple(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.boxplot([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
        assert len(result['boxes']) == 2
        assert len(result['medians']) == 2

    def test_boxplot_median_value(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.boxplot([1, 2, 3, 4, 5])
        med_line = result['medians'][0]
        assert med_line.get_ydata()[0] == 3

    def test_boxplot_fliers(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        result = ax.boxplot(data)
        assert 'fliers' in result
        assert len(result['fliers']) == 1

    def test_boxplot_no_fliers(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.boxplot([1, 2, 3, 4, 5], showfliers=False)
        assert len(result['fliers']) == 0

    def test_boxplot_vert_false(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.boxplot([1, 2, 3, 4, 5], vert=False)
        assert len(result['boxes']) == 1


class TestViolinplot:
    def test_violinplot_basic(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert 'bodies' in result

    def test_violinplot_returns_bodies(self):
        from matplotlib.patches import Polygon
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5])
        assert len(result['bodies']) == 1
        assert isinstance(result['bodies'][0], Polygon)

    def test_violinplot_multiple(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([[1, 2, 3, 4, 5], [10, 20, 30]])
        assert len(result['bodies']) == 2

    def test_violinplot_showmedians(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5], showmedians=True)
        assert 'cmedians' in result
        assert len(result['cmedians']) == 1

    def test_violinplot_showmeans(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5], showmeans=True)
        assert 'cmeans' in result
        assert len(result['cmeans']) == 1

    def test_violinplot_showextrema(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5], showextrema=True)
        assert 'cmins' in result
        assert 'cmaxes' in result
        assert 'cbars' in result

    def test_violinplot_vert_false(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5], vert=False)
        assert len(result['bodies']) == 1


class TestPyplotWrappers:
    def test_step(self):
        import matplotlib.pyplot as plt
        result = plt.step([1, 2], [3, 4])
        assert result is not None

    def test_stairs(self):
        import matplotlib.pyplot as plt
        result = plt.stairs([1, 2, 3])
        assert result is not None

    def test_stackplot(self):
        import matplotlib.pyplot as plt
        result = plt.stackplot([1, 2], [3, 4])
        assert result is not None

    def test_stem(self):
        import matplotlib.pyplot as plt
        result = plt.stem([1, 2, 3])
        assert result is not None

    def test_pie(self):
        import matplotlib.pyplot as plt
        result = plt.pie([1, 2, 3])
        assert result is not None

    def test_boxplot(self):
        import matplotlib.pyplot as plt
        result = plt.boxplot([1, 2, 3, 4, 5])
        assert result is not None

    def test_violinplot(self):
        import matplotlib.pyplot as plt
        result = plt.violinplot([1, 2, 3, 4, 5])
        assert result is not None


class TestAutoLimitsNewTypes:
    def test_pie_auto_limits(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= -1.0
        assert xlim[1] >= 1.0
        assert ylim[0] <= -1.0
        assert ylim[1] >= 1.0

    def test_boxplot_auto_limits(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.boxplot([1, 2, 3, 4, 5])
        ylim = ax.get_ylim()
        assert ylim[0] <= 1.0
        assert ylim[1] >= 5.0


# ===================================================================
# Additional plot types tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt


class TestBoxplotParametric:
    """Parametric tests for boxplot."""

    @pytest.mark.parametrize('n', [5, 10, 20, 50])
    def test_boxplot_n_data_points(self, n):
        """Boxplot works for various data sizes."""
        import numpy as np
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).normal(0, 1, n).tolist()
        result = ax.boxplot(data)
        assert result is not None
        plt.close('all')

    @pytest.mark.parametrize('whis', [1.5, 2.0, 3.0])
    def test_boxplot_whis(self, whis):
        """Boxplot accepts whis parameter."""
        fig, ax = plt.subplots()
        result = ax.boxplot([1, 2, 3, 4, 5], whis=whis)
        assert result is not None
        plt.close('all')


class TestPieParametric:
    """Parametric tests for pie chart."""

    @pytest.mark.parametrize('n', [2, 3, 4, 5, 8])
    def test_pie_n_slices(self, n):
        """Pie chart creates n wedges."""
        fig, ax = plt.subplots()
        sizes = [1] * n
        patches, _ = ax.pie(sizes)
        assert len(patches) == n
        plt.close('all')

    @pytest.mark.parametrize('startangle', [0, 45, 90, 135, 180])
    def test_pie_startangle(self, startangle):
        """Pie chart accepts startangle parameter."""
        fig, ax = plt.subplots()
        patches, _ = ax.pie([1, 2, 3], startangle=startangle)
        assert len(patches) == 3
        plt.close('all')


class TestStepParametric:
    """Parametric tests for step plot."""

    @pytest.mark.parametrize('where', ['pre', 'post', 'mid'])
    def test_step_where(self, where):
        """Step plot accepts where parameter."""
        fig, ax = plt.subplots()
        lines = ax.step([1, 2, 3], [4, 5, 6], where=where)
        assert len(lines) == 1
        plt.close('all')

    @pytest.mark.parametrize('n', [2, 5, 10, 20])
    def test_step_n_points(self, n):
        """Step plot works for various data sizes."""
        fig, ax = plt.subplots()
        xs = list(range(n))
        ys = list(range(n))
        lines = ax.step(xs, ys)
        assert len(lines) == 1
        plt.close('all')


class TestStackplotParametric:
    """Parametric tests for stackplot."""

    @pytest.mark.parametrize('n_series', [2, 3, 4, 5])
    def test_stackplot_n_series(self, n_series):
        """Stackplot handles n data series."""
        fig, ax = plt.subplots()
        x = [1, 2, 3]
        ys = [[1, 2, 3]] * n_series
        polys = ax.stackplot(x, *ys)
        assert len(polys) == n_series
        plt.close('all')


class TestStemParametric:
    """Parametric tests for stem plot."""

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_stem_n_points(self, n):
        """Stem plot works for various data sizes."""
        fig, ax = plt.subplots()
        ys = list(range(1, n + 1))
        sc = ax.stem(ys)
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('bottom', [0, 1, -1, 0.5])
    def test_stem_bottom(self, bottom):
        """Stem plot accepts bottom parameter."""
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2, 3], bottom=bottom)
        assert sc is not None
        plt.close('all')


# ===================================================================
# Extended parametric tests for plot types
# ===================================================================

class TestPlotTypesParametricExtended:
    """Extended parametric tests for various plot types."""

    @pytest.mark.parametrize('n', [1, 3, 5, 10, 20])
    def test_bar_n(self, n):
        """bar() creates n patches."""
        fig, ax = plt.subplots()
        container = ax.bar(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_barh_n(self, n):
        """barh() creates n patches."""
        fig, ax = plt.subplots()
        container = ax.barh(range(n), range(n))
        assert len(container) == n
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 20, 50])
    def test_hist_bins_count(self, bins):
        """hist returns correct bin count."""
        fig, ax = plt.subplots()
        n_counts, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n_counts) == bins
        plt.close('all')

    @pytest.mark.parametrize('n', [3, 5, 10, 20])
    def test_scatter_n(self, n):
        """scatter with n points."""
        fig, ax = plt.subplots()
        sc = ax.scatter(range(n), range(n))
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5, 10])
    def test_stem_n(self, n):
        """stem with n points."""
        fig, ax = plt.subplots()
        sc = ax.stem(range(1, n+1))
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('linestyle', ['-', '--', ':', '-.'])
    def test_plot_linestyle(self, linestyle):
        """plot accepts linestyle."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=linestyle)
        assert line is not None
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'v', 'D', '+', 'x', '*'])
    def test_plot_marker(self, marker):
        """plot accepts various markers."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1, 2], [0, 1, 0], marker=marker)
        assert line.get_marker() == marker
        plt.close('all')

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', 'black', '#ff0000'])
    def test_plot_color(self, color):
        """plot accepts various colors."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], color=color)
        assert line is not None
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_plot_lw(self, lw):
        """plot accepts linewidth."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_plot_alpha(self, alpha):
        """plot accepts alpha."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], alpha=alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('histtype', ['bar', 'step', 'stepfilled'])
    def test_hist_type(self, histtype):
        """hist accepts histtype."""
        fig, ax = plt.subplots()
        n_counts, _, _ = ax.hist(list(range(50)), bins=10, histtype=histtype)
        assert len(n_counts) == 10
        plt.close('all')


class TestPlotTypesParametric2:
    """Further parametric plot type tests."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 10])
    def test_plot_n_lines(self, n):
        """Axes can hold n line plots."""
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 10])
    def test_bar_n(self, n):
        """bar() creates n patches."""
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(1, n + 1))
        assert len(bars) == n
        plt.close('all')

    @pytest.mark.parametrize('n', [3, 5, 10, 20, 50])
    def test_scatter_n(self, n):
        """scatter() handles n points."""
        fig, ax = plt.subplots()
        sc = ax.scatter(range(n), range(n))
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('bins', [5, 10, 15, 20, 30])
    def test_hist_bins(self, bins):
        """hist() produces correct bin count."""
        fig, ax = plt.subplots()
        n_counts, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n_counts) == bins
        plt.close('all')

    @pytest.mark.parametrize('linestyle', ['-', '--', ':', '-.'])
    def test_linestyle(self, linestyle):
        """Line linestyle roundtrips."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linestyle=linestyle)
        assert line.get_linestyle() == linestyle
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_line_alpha(self, alpha):
        """Line alpha roundtrips."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-9
        plt.close('all')

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        """Line width roundtrips."""
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-9
        plt.close('all')


class TestPlotTypesParametric7:
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



class TestPlotTypesParametric13:
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

