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
