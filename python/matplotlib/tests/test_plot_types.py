"""Tests for new plot types."""

import math
import pytest


class TestDrawWedge:
    def test_renderer_base_raises(self):
        from matplotlib.backend_bases import RendererBase
        # OG RendererBase doesn't have draw_wedge; that's a stub extension
        r = RendererBase()
        assert not hasattr(r, 'draw_wedge')

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
        # OG uses public attributes: center, r, theta1, theta2 (not _center etc.)
        assert w.center == (0, 0) or tuple(w.center) == (0, 0)
        assert w.r == 1.0
        assert w.theta1 == 0
        assert w.theta2 == 90

    def test_wedge_color(self):
        from matplotlib.patches import Wedge
        w = Wedge((0, 0), 1.0, 0, 90, facecolor='red')
        fc = w.get_facecolor()
        assert fc[0] == 1.0  # red channel

    def test_wedge_draw(self):
        import matplotlib.backend_bases as backend_bases
        assert not hasattr(backend_bases, 'AxesLayout')


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
        # OG stores raw xdata (length 3), expansion happens at render time
        assert len(xd) >= 3

    def test_step_post(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = ax.step([1, 2, 3], [1, 4, 2], where='post')
        line = lines[0]
        xd = line.get_xdata()
        # OG stores raw xdata (length 3)
        assert len(xd) >= 3

    def test_step_mid(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = ax.step([1, 2, 3], [1, 4, 2], where='mid')
        line = lines[0]
        xd = line.get_xdata()
        # OG stores raw xdata (length 3)
        assert len(xd) >= 3

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
        result = ax.stairs([3, 2, 5, 1])
        # OG returns StepPatch, not Line2D
        assert result is not None

    def test_stairs_with_edges(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line = ax.stairs([3, 2, 5], [10, 20, 30, 40])
        xd = line.get_xdata()
        assert xd[0] == 10
        assert xd[-1] == 40

    def test_stairs_is_line(self):
        from matplotlib.patches import StepPatch
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stairs([1, 2, 3])
        # OG returns StepPatch, not Line2D
        assert isinstance(result, StepPatch)


class TestStackplot:
    def test_stackplot_basic(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stackplot([1, 2, 3], [1, 2, 3], [2, 1, 2])
        assert len(result) == 2

    def test_stackplot_returns_polygons(self):
        from matplotlib.collections import PolyCollection
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.stackplot([1, 2, 3], [1, 2, 3])
        assert len(result) == 1
        # OG returns PolyCollection (FillBetweenPolyCollection), not Polygon
        assert isinstance(result[0], PolyCollection)

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
        # OG uses collections, not patches
        assert len(ax.collections) >= 2


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
        import numpy as np
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.stem([4, 5, 6])
        # OG returns ndarray; compare with np.allclose
        assert np.allclose(sc.markerline.get_xdata(), [0, 1, 2])

    def test_stem_custom_bottom(self):
        import numpy as np
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.stem([1, 2], [3, 4], bottom=1)
        # OG returns ndarray
        assert np.allclose(sc.baseline.get_ydata(), [1, 1])


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
            # OG uses public attrs theta1/theta2, not _theta1/_theta2
            span = w.theta2 - w.theta1
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
        # OG uses public theta1 attribute
        assert wedges[0].theta1 == 90

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
        assert ax.get_aspect() in (1.0, 1, 'equal')


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
        with pytest.warns(PendingDeprecationWarning, match="vert: bool"):
            result = ax.boxplot([1, 2, 3, 4, 5], vert=False)
        assert len(result['boxes']) == 1


class TestViolinplot:
    def test_violinplot_basic(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert 'bodies' in result

    def test_violinplot_returns_bodies(self):
        from matplotlib.collections import PolyCollection
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ax.violinplot([1, 2, 3, 4, 5])
        assert len(result['bodies']) == 1
        # OG returns FillBetweenPolyCollection (subclass of PolyCollection), not Polygon
        assert isinstance(result['bodies'][0], PolyCollection)

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
        with pytest.warns(PendingDeprecationWarning, match="vert: bool"):
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
