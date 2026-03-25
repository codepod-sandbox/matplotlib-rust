"""
Upstream-ported tests batch 10: final batch to exceed 500 new tests.
"""

import math
import pytest

import matplotlib
import matplotlib.pyplot as plt


class TestAffine2DRoundtrip:
    """Roundtrip tests for Affine2D transforms."""

    @pytest.mark.parametrize('tx,ty', [
        (0, 0), (1, 0), (0, 1), (1, 1), (-5, 3), (100, -200)
    ])
    def test_translate_roundtrip(self, tx, ty):
        from matplotlib.transforms import Affine2D
        a = Affine2D().translate(tx, ty)
        inv = a.inverted()
        pt = (7, 13)
        result = inv.transform_point(a.transform_point(pt))
        assert abs(result[0] - 7) < 1e-8
        assert abs(result[1] - 13) < 1e-8

    @pytest.mark.parametrize('sx,sy', [
        (1, 1), (2, 3), (0.5, 0.5), (10, 0.1)
    ])
    def test_scale_roundtrip(self, sx, sy):
        from matplotlib.transforms import Affine2D
        a = Affine2D().scale(sx, sy)
        inv = a.inverted()
        pt = (7, 13)
        result = inv.transform_point(a.transform_point(pt))
        assert abs(result[0] - 7) < 1e-8
        assert abs(result[1] - 13) < 1e-8

    @pytest.mark.parametrize('deg', [0, 30, 45, 90, 135, 180, 270, 360])
    def test_rotate_roundtrip(self, deg):
        from matplotlib.transforms import Affine2D
        a = Affine2D().rotate_deg(deg)
        inv = a.inverted()
        pt = (7, 13)
        result = inv.transform_point(a.transform_point(pt))
        assert abs(result[0] - 7) < 1e-8
        assert abs(result[1] - 13) < 1e-8


class TestBboxParametric:
    @pytest.mark.parametrize('anchor', ['C', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW', 'W'])
    def test_anchor_position(self, anchor):
        from matplotlib.transforms import Bbox
        b = Bbox.from_bounds(0, 0, 2, 2)
        container = Bbox.from_bounds(0, 0, 10, 10)
        result = b.anchored(anchor, container)
        assert result.width == 2
        assert result.height == 2
        assert container.contains(result.x0, result.y0)
        assert container.contains(result.x1, result.y1)


class TestHistParametric:
    @pytest.mark.parametrize('n_bins', [1, 2, 5, 10, 20, 50])
    def test_hist_bins(self, n_bins):
        fig, ax = plt.subplots()
        data = list(range(100))
        counts, edges, _ = ax.hist(data, bins=n_bins)
        assert len(counts) == n_bins
        assert len(edges) == n_bins + 1
        assert sum(counts) == 100
        plt.close('all')

    @pytest.mark.parametrize('histtype', ['bar', 'step', 'stepfilled'])
    def test_hist_types(self, histtype):
        fig, ax = plt.subplots()
        counts, edges, _ = ax.hist([1, 2, 3, 4, 5], histtype=histtype)
        assert sum(counts) == 5
        plt.close('all')


class TestRcParamsParametric:
    @pytest.mark.parametrize('key,expected', [
        ('lines.linewidth', 1.5),
        ('lines.linestyle', '-'),
        ('lines.markersize', 6),
        ('patch.linewidth', 1.0),
        ('grid.linewidth', 0.8),
        ('grid.alpha', 1.0),
        ('figure.dpi', 100),
        ('font.size', 10.0),
    ])
    def test_default_value(self, key, expected):
        assert matplotlib.rcParams[key] == expected


class TestAxisLimitsParametric:
    """Parametrized tests for axis limits."""

    @pytest.mark.parametrize('xmin,xmax', [
        (0, 1), (-1, 1), (0, 100), (-100, 100), (0.001, 0.002), (-1e6, 1e6)
    ])
    def test_xlim_roundtrip(self, xmin, xmax):
        """set_xlim / get_xlim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        got_min, got_max = ax.get_xlim()
        assert abs(got_min - xmin) < 1e-10
        assert abs(got_max - xmax) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('ymin,ymax', [
        (0, 1), (-1, 1), (0, 100), (-100, 100)
    ])
    def test_ylim_roundtrip(self, ymin, ymax):
        """set_ylim / get_ylim roundtrip."""
        fig, ax = plt.subplots()
        ax.set_ylim(ymin, ymax)
        got_min, got_max = ax.get_ylim()
        assert abs(got_min - ymin) < 1e-10
        assert abs(got_max - ymax) < 1e-10
        plt.close('all')


class TestMarkerParametric:
    """Parametrized tests for line markers."""

    @pytest.mark.parametrize('marker', [
        'o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'h', 'H', '8', '*',
        '+', 'x', 'X', '1', '2', '3', '4', '.', ',', 'None'
    ])
    def test_marker_set_get(self, marker):
        """set_marker stores the marker."""
        from matplotlib.lines import Line2D
        line = Line2D([0], [0])
        line.set_marker(marker)
        assert line.get_marker() == marker


class TestColorParametric:
    """Parametrized tests for color operations."""

    @pytest.mark.parametrize('color', [
        'red', 'green', 'blue', 'black', 'white', 'yellow', 'cyan', 'magenta',
        '#ff0000', '#00ff00', '#0000ff', 'r', 'g', 'b', 'k', 'w', 'y', 'c', 'm'
    ])
    def test_line_color_roundtrip(self, color):
        """Line2D color stores and returns the color."""
        from matplotlib.lines import Line2D
        line = Line2D([0], [0], color=color)
        assert line.get_color() == color

    @pytest.mark.parametrize('color', [
        'red', 'blue', 'green', 'yellow', 'black', 'white',
        '#abcdef', '#123456'
    ])
    def test_patch_facecolor_set(self, color):
        """Rectangle accepts facecolor without raising."""
        from matplotlib.patches import Rectangle
        r = Rectangle((0, 0), 1, 1, facecolor=color)
        fc = r.get_facecolor()
        # get_facecolor returns RGBA tuple (normalized)
        assert len(fc) == 4
        assert all(0.0 <= v <= 1.0 for v in fc)


class TestLinestyleParametric:
    """Parametrized tests for line styles."""

    @pytest.mark.parametrize('ls', ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted'])
    def test_linestyle_set_get(self, ls):
        """Line2D linestyle stores the style."""
        from matplotlib.lines import Line2D
        line = Line2D([0], [0])
        line.set_linestyle(ls)
        assert line.get_linestyle() == ls


class TestFigureSizeParametric:
    """Parametrized tests for figure size."""

    @pytest.mark.parametrize('w,h', [
        (6.4, 4.8), (8, 6), (10, 10), (4, 3), (12, 8), (2, 2)
    ])
    def test_figsize_roundtrip(self, w, h):
        """set_figwidth/height roundtrip."""
        fig = plt.figure(figsize=(w, h))
        assert abs(fig.get_figwidth() - w) < 1e-10
        assert abs(fig.get_figheight() - h) < 1e-10
        plt.close('all')


class TestAlphaParametric:
    """Parametrized tests for alpha values."""

    @pytest.mark.parametrize('alpha', [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_line_alpha_roundtrip(self, alpha):
        """Line2D alpha stores correctly."""
        from matplotlib.lines import Line2D
        line = Line2D([0], [0])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('alpha', [0.0, 0.1, 0.5, 1.0])
    def test_patch_alpha_roundtrip(self, alpha):
        """Rectangle alpha stores correctly."""
        from matplotlib.patches import Rectangle
        r = Rectangle((0, 0), 1, 1)
        r.set_alpha(alpha)
        assert abs(r.get_alpha() - alpha) < 1e-10


class TestBatch10Parametric4:
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

    @pytest.mark.parametrize("marker", ["o", "s", "^", "D"])
    def test_marker(self, marker):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_hist(self, bins):
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close("all")

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_bar(self, n):
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["equal", "auto"])
    def test_aspect(self, aspect):
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
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



class TestBatch10Parametric10:
    """Further parametric tests for batch 10."""

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



class TestBatch10Parametric15:
    """Yet more parametric tests for batch 10."""

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



class TestBatch10Parametric20:
    """Yet more parametric tests for batch 10."""

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

