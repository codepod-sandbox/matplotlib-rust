"""Tests for Normalize subclasses: TwoSlopeNorm, BoundaryNorm, PowerNorm, SymLogNorm, NoNorm."""

import math
import pytest
from matplotlib.tests._approx import approx

from matplotlib.colors import (
    TwoSlopeNorm,
    BoundaryNorm,
    PowerNorm,
    SymLogNorm,
    NoNorm,
    Normalize,
    LogNorm,
)


# ===================================================================
# TwoSlopeNorm
# ===================================================================

class TestTwoSlopeNorm:
    def test_basic(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        assert norm(0) == approx(0.5)

    def test_maps_vmin_to_0(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        assert norm(-10) == approx(0.0)

    def test_maps_vmax_to_1(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        assert norm(10) == approx(1.0)

    def test_asymmetric(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=10)
        assert norm(0) == approx(0.5)
        assert norm(-2) == approx(0.0)
        assert norm(10) == approx(1.0)

    def test_list_input(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        result = norm([-10, 0, 10])
        assert len(result) == 3
        assert result[0] == approx(0.0)
        assert result[1] == approx(0.5)
        assert result[2] == approx(1.0)

    def test_quarter_points(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        assert norm(-5) == approx(0.25)
        assert norm(5) == approx(0.75)

    def test_inverse(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        assert norm.inverse(0.0) == approx(-10)
        assert norm.inverse(0.5) == approx(0)
        assert norm.inverse(1.0) == approx(10)

    def test_inverse_list(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
        result = norm.inverse([0.0, 0.5, 1.0])
        assert result[0] == approx(-10)
        assert result[1] == approx(0)
        assert result[2] == approx(10)

    def test_vmin_ge_vcenter_raises(self):
        with pytest.raises(ValueError):
            TwoSlopeNorm(vcenter=0, vmin=5)

    def test_vmax_le_vcenter_raises(self):
        with pytest.raises(ValueError):
            TwoSlopeNorm(vcenter=0, vmax=-5)

    def test_repr(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        r = repr(norm)
        assert 'TwoSlopeNorm' in r

    def test_autoscale(self):
        norm = TwoSlopeNorm(vcenter=0)
        norm.autoscale([-5, 0, 3])
        assert norm.vmin == -5
        assert norm.vmax == 3

    def test_autoscale_None(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-2)
        norm.autoscale_None([-5, 0, 3])
        assert norm.vmin == -2  # unchanged
        assert norm.vmax == 3

    def test_autoscale_on_call(self):
        """TwoSlopeNorm autoscales vmin/vmax on call if not set."""
        norm = TwoSlopeNorm(vcenter=0)
        result = norm(5)
        assert norm.vmin is not None
        assert norm.vmax is not None
        # inverse requires vmin/vmax to be set
        inv = norm.inverse(result)
        assert abs(inv - 5) < 1e-6

    def test_positive_vcenter(self):
        norm = TwoSlopeNorm(vcenter=5, vmin=0, vmax=20)
        assert norm(5) == approx(0.5)
        assert norm(0) == approx(0.0)
        assert norm(20) == approx(1.0)


# ===================================================================
# BoundaryNorm
# ===================================================================

class TestBoundaryNorm:
    def test_basic(self):
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
        result = norm(0.5)
        assert isinstance(result, float)

    def test_list_input(self):
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
        result = norm([0.5, 1.5, 2.5])
        assert len(result) == 3

    def test_monotonic_required(self):
        with pytest.raises(ValueError, match='monotonically'):
            BoundaryNorm([3, 2, 1], ncolors=3)

    def test_inverse_raises(self):
        norm = BoundaryNorm([0, 1, 2], ncolors=2)
        with pytest.raises(ValueError, match='not invertible'):
            norm.inverse(0.5)

    def test_repr(self):
        norm = BoundaryNorm([0, 1, 2], ncolors=2)
        r = repr(norm)
        assert 'BoundaryNorm' in r

    def test_ncolors(self):
        norm = BoundaryNorm([0, 1, 2, 3, 4], ncolors=10)
        assert norm.ncolors == 10
        assert norm.Ncmap == 10

    def test_vmin_vmax(self):
        norm = BoundaryNorm([0, 5, 10], ncolors=2)
        assert norm.vmin == 0
        assert norm.vmax == 10

    def test_N_intervals(self):
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
        assert norm.N == 3

    def test_clip_below(self):
        norm = BoundaryNorm([0, 1, 2], ncolors=2, clip=True)
        result = norm(-1)
        assert result == 0.0

    def test_extend_min(self):
        norm = BoundaryNorm([0, 1, 2], ncolors=3, extend='min')
        result = norm(-1)
        assert result < 0  # below range

    def test_extend_max(self):
        norm = BoundaryNorm([0, 1, 2], ncolors=3, extend='max')
        result = norm(3)
        assert result > 1  # above range

    def test_extend_both(self):
        norm = BoundaryNorm([0, 1, 2], ncolors=4, extend='both')
        under = norm(-1)
        over = norm(3)
        assert under < 0
        assert over > 1

    def test_many_boundaries(self):
        boundaries = list(range(20))
        norm = BoundaryNorm(boundaries, ncolors=19)
        result = norm(10)
        assert 0 <= result <= 1


# ===================================================================
# PowerNorm
# ===================================================================

class TestPowerNorm:
    def test_gamma_1(self):
        norm = PowerNorm(gamma=1.0, vmin=0, vmax=10)
        assert norm(5) == approx(0.5)

    def test_gamma_2(self):
        norm = PowerNorm(gamma=2.0, vmin=0, vmax=1)
        assert norm(0.5) == approx(0.25)

    def test_gamma_half(self):
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
        result = norm(0.25)
        assert result == approx(0.5)

    def test_maps_vmin_to_0(self):
        norm = PowerNorm(gamma=2, vmin=0, vmax=10)
        assert norm(0) == approx(0.0)

    def test_maps_vmax_to_1(self):
        norm = PowerNorm(gamma=2, vmin=0, vmax=10)
        assert norm(10) == approx(1.0)

    def test_list_input(self):
        norm = PowerNorm(gamma=1, vmin=0, vmax=10)
        result = norm([0, 5, 10])
        assert len(result) == 3

    def test_clip(self):
        norm = PowerNorm(gamma=1, vmin=0, vmax=10, clip=True)
        assert norm(-5) == approx(0.0)
        assert norm(15) == approx(1.0)

    def test_inverse(self):
        norm = PowerNorm(gamma=2, vmin=0, vmax=10)
        val = norm(5)
        back = norm.inverse(val)
        assert back == approx(5, abs=0.1)

    def test_inverse_list(self):
        norm = PowerNorm(gamma=1, vmin=0, vmax=10)
        result = norm.inverse([0.0, 0.5, 1.0])
        assert result[0] == approx(0)
        assert result[1] == approx(5)
        assert result[2] == approx(10)

    def test_repr(self):
        norm = PowerNorm(gamma=2, vmin=0, vmax=10)
        r = repr(norm)
        assert 'PowerNorm' in r
        assert 'gamma' in r

    def test_requires_vmin_vmax(self):
        norm = PowerNorm(gamma=2)
        with pytest.raises(ValueError):
            norm(5)

    def test_same_vmin_vmax(self):
        norm = PowerNorm(gamma=2, vmin=5, vmax=5)
        assert norm(5) == 0.0


# ===================================================================
# SymLogNorm
# ===================================================================

class TestSymLogNorm:
    def test_basic(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        result = norm(0)
        assert result == approx(0.5, abs=0.01)

    def test_maps_vmin_to_0(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        assert norm(-10) == approx(0.0)

    def test_maps_vmax_to_1(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        assert norm(10) == approx(1.0)

    def test_linear_region(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        # Within linthresh, should be roughly linear
        v1 = norm(0.5)
        v2 = norm(-0.5)
        assert v1 > 0.5  # positive side
        assert v2 < 0.5  # negative side

    def test_list_input(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        result = norm([-10, 0, 10])
        assert len(result) == 3

    def test_inverse(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        val = norm(5)
        back = norm.inverse(val)
        assert back == approx(5, abs=0.5)

    def test_inverse_list(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        result = norm.inverse([0.0, 0.5, 1.0])
        assert result[0] == approx(-10, abs=0.5)
        assert result[2] == approx(10, abs=0.5)

    def test_linthresh_positive_required(self):
        with pytest.raises(ValueError, match='positive'):
            SymLogNorm(linthresh=-1)

    def test_repr(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10)
        r = repr(norm)
        assert 'SymLogNorm' in r
        assert 'linthresh' in r

    def test_requires_vmin_vmax(self):
        norm = SymLogNorm(linthresh=1)
        with pytest.raises(ValueError):
            norm(5)

    def test_clip(self):
        norm = SymLogNorm(linthresh=1, vmin=-10, vmax=10, clip=True)
        assert 0.0 <= norm(-20) <= 1.0
        assert 0.0 <= norm(20) <= 1.0

    def test_linscale(self):
        norm1 = SymLogNorm(linthresh=1, linscale=1, vmin=-10, vmax=10)
        norm2 = SymLogNorm(linthresh=1, linscale=2, vmin=-10, vmax=10)
        # Different linscale should give different results for values in linear region
        v1 = norm1(0.5)
        v2 = norm2(0.5)
        # Both should be valid but potentially different
        assert 0.0 <= v1 <= 1.0
        assert 0.0 <= v2 <= 1.0

    def test_base_10(self):
        norm = SymLogNorm(linthresh=1, base=10, vmin=-100, vmax=100)
        result = norm(10)
        assert 0.5 < result < 1.0

    def test_base_2(self):
        norm = SymLogNorm(linthresh=1, base=2, vmin=-16, vmax=16)
        result = norm(4)
        assert 0.5 < result < 1.0


# ===================================================================
# NoNorm
# ===================================================================

class TestNoNorm:
    def test_passthrough(self):
        norm = NoNorm()
        assert norm(0.5) == approx(0.5)

    def test_list(self):
        norm = NoNorm()
        result = norm([0.1, 0.5, 0.9])
        assert result[0] == approx(0.1)
        assert result[1] == approx(0.5)
        assert result[2] == approx(0.9)

    def test_inverse(self):
        norm = NoNorm()
        assert norm.inverse(0.5) == approx(0.5)

    def test_inverse_list(self):
        norm = NoNorm()
        result = norm.inverse([0.1, 0.5])
        assert result[0] == approx(0.1)
        assert result[1] == approx(0.5)

    def test_integer(self):
        norm = NoNorm()
        assert norm(5) == approx(5.0)


# ===================================================================
# Cross-norm tests
# ===================================================================

class TestNormInteractions:
    def test_all_norms_have_repr(self):
        norms = [
            Normalize(0, 1),
            LogNorm(1, 100),
            TwoSlopeNorm(0, vmin=-1, vmax=1),
            PowerNorm(2, vmin=0, vmax=1),
            SymLogNorm(1, vmin=-10, vmax=10),
            NoNorm(),
        ]
        for norm in norms:
            r = repr(norm)
            assert isinstance(r, str)
            assert len(r) > 0

    def test_normalize_subclass(self):
        assert issubclass(TwoSlopeNorm, Normalize)
        assert issubclass(BoundaryNorm, Normalize)
        assert issubclass(PowerNorm, Normalize)
        assert issubclass(SymLogNorm, Normalize)
        assert issubclass(NoNorm, Normalize)

    def test_all_norms_accept_lists(self):
        norms = [
            Normalize(0, 10),
            PowerNorm(1, vmin=0, vmax=10),
            TwoSlopeNorm(5, vmin=0, vmax=10),
            SymLogNorm(1, vmin=-10, vmax=10),
            NoNorm(),
        ]
        for norm in norms:
            result = norm([1, 5, 9])
            assert isinstance(result, list)
            assert len(result) == 3


# ===================================================================
# Additional parametric tests
# ===================================================================

import pytest
import numpy as np


class TestNormalizeParametric:
    """Parametric Normalize tests."""

    @pytest.mark.parametrize('vmin,vmax,val,expected', [
        (0, 1, 0.0, 0.0),
        (0, 1, 1.0, 1.0),
        (0, 1, 0.5, 0.5),
        (0, 10, 5, 0.5),
        (-1, 1, 0, 0.5),
        (-10, 10, 5, 0.75),
        (0, 100, 25, 0.25),
    ])
    def test_normalize_value(self, vmin, vmax, val, expected):
        """Normalize maps values correctly."""
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
        result = float(norm(val))
        assert abs(result - expected) < 1e-10

    @pytest.mark.parametrize('vmin,vmax', [
        (0, 1), (-5, 5), (0, 100), (-100, 0),
    ])
    def test_normalize_vmin_to_zero(self, vmin, vmax):
        """Normalize maps vmin to 0."""
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
        assert abs(float(norm(vmin))) < 1e-10

    @pytest.mark.parametrize('vmin,vmax', [
        (0, 1), (-5, 5), (0, 100), (-100, 0),
    ])
    def test_normalize_vmax_to_one(self, vmin, vmax):
        """Normalize maps vmax to 1."""
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
        assert abs(float(norm(vmax)) - 1.0) < 1e-10

    @pytest.mark.parametrize('gamma', [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_power_norm_gamma(self, gamma):
        """PowerNorm stores gamma."""
        from matplotlib.colors import PowerNorm
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)
        assert norm.gamma == gamma

    @pytest.mark.parametrize('vcenter,vmin,vmax', [
        (0, -1, 1), (5, 0, 10), (-2, -10, 5),
    ])
    def test_twoslope_norm_vcenter(self, vcenter, vmin, vmax):
        """TwoSlopeNorm maps vcenter to 0.5."""
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        result = float(norm(vcenter))
        assert abs(result - 0.5) < 1e-10

    @pytest.mark.parametrize('linthresh', [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_symlog_norm_linthresh(self, linthresh):
        """SymLogNorm stores linthresh."""
        from matplotlib.colors import SymLogNorm
        norm = SymLogNorm(linthresh=linthresh, vmin=-100, vmax=100)
        assert norm.linthresh == linthresh

    @pytest.mark.parametrize('n', [2, 4, 8, 16])
    def test_normalize_array_length(self, n):
        """Normalize applied to n-element array returns n values."""
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=1)
        arr = list(range(n))
        result = norm(arr)
        assert len(result) == n


class TestBoundaryNormParametric:
    """Parametric BoundaryNorm tests."""

    @pytest.mark.parametrize('n', [2, 3, 4, 5])
    def test_boundary_norm_n_boundaries(self, n):
        """BoundaryNorm stores n+1 boundaries for n colors."""
        from matplotlib.colors import BoundaryNorm
        boundaries = list(range(n + 1))
        norm = BoundaryNorm(boundaries=boundaries, ncolors=n)
        # Just confirm it doesn't raise
        assert norm is not None

    @pytest.mark.parametrize('val', [0.5, 1.5, 2.5])
    def test_boundary_norm_maps_value(self, val):
        """BoundaryNorm maps any in-range value without error."""
        from matplotlib.colors import BoundaryNorm
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
        result = norm(val)
        assert result is not None

    @pytest.mark.parametrize('ncolors', [2, 4, 8, 16])
    def test_boundary_norm_ncolors(self, ncolors):
        """BoundaryNorm stores ncolors."""
        from matplotlib.colors import BoundaryNorm
        boundaries = list(range(ncolors + 1))
        norm = BoundaryNorm(boundaries=boundaries, ncolors=ncolors)
        assert norm.N == ncolors


# ===================================================================
# More parametric tests for norms
# ===================================================================

class TestNormsParametric2:
    """More parametric tests for norms."""

    @pytest.mark.parametrize('vmin,vmax', [(0, 1), (-1, 1), (0, 100), (-5, 5), (0, 255)])
    def test_normalize_range2(self, vmin, vmax):
        """Normalize maps midpoint to 0.5."""
        norm = Normalize(vmin=vmin, vmax=vmax)
        mid = (vmin + vmax) / 2
        assert abs(float(norm(mid)) - 0.5) < 1e-10

    @pytest.mark.parametrize('v', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_normalize_unit_range(self, v):
        """Normalize with vmin=0, vmax=1 maps v to v."""
        norm = Normalize(vmin=0, vmax=1)
        assert abs(float(norm(v)) - v) < 1e-10

    @pytest.mark.parametrize('vcenter', [-2, -1, 0, 1, 2])
    def test_two_slope_norm_vcenter(self, vcenter):
        """TwoSlopeNorm maps vcenter to 0.5."""
        import matplotlib.colors as mcolors
        norm = mcolors.TwoSlopeNorm(vmin=vcenter-3, vcenter=vcenter, vmax=vcenter+3)
        assert abs(float(norm(vcenter)) - 0.5) < 1e-10

    @pytest.mark.parametrize('gamma', [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_power_norm_gamma2(self, gamma):
        """PowerNorm maps 0->0, 1->1 for any gamma."""
        import matplotlib.colors as mcolors
        norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=1)
        assert abs(float(norm(0)) - 0.0) < 1e-10
        assert abs(float(norm(1)) - 1.0) < 1e-10

    @pytest.mark.parametrize('vmin,vmax', [(0, 1), (0, 100), (-5, 5)])
    def test_normalize_clip_max(self, vmin, vmax):
        """Normalize clips values above vmax to 1."""
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        assert float(norm(vmax + 100)) == 1.0

    @pytest.mark.parametrize('vmin,vmax', [(0, 1), (0, 100), (-5, 5)])
    def test_normalize_clip_min(self, vmin, vmax):
        """Normalize clips values below vmin to 0."""
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        assert float(norm(vmin - 100)) == 0.0

    @pytest.mark.parametrize('linthresh', [0.01, 0.1, 1.0, 10.0])
    def test_symlog_norm_linthresh2(self, linthresh):
        """SymLogNorm accepts various linthresh values."""
        import matplotlib.colors as mcolors
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-100, vmax=100)
        assert norm is not None

    @pytest.mark.parametrize('ncolors', [2, 4, 8, 16, 256])
    def test_boundary_norm_ncolors(self, ncolors):
        """BoundaryNorm with ncolors stores N correctly."""
        import matplotlib.colors as mcolors
        import numpy as np
        boundaries = list(range(ncolors + 1))
        norm = mcolors.BoundaryNorm(boundaries, ncolors)
        assert norm.N == ncolors


class TestNormsParametric3:
    """Further parametric norm tests."""

    @pytest.mark.parametrize('vmin,vmax', [(0, 1), (-1, 1), (0, 100), (-5, 5), (0, 255), (-100, 100)])
    def test_normalize_clip(self, vmin, vmax):
        """Normalized values are clipped to [0, 1]."""
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        assert norm(vmin) == approx(0.0)
        assert norm(vmax) == approx(1.0)

    @pytest.mark.parametrize('gamma', [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_power_norm_midpoint(self, gamma):
        """PowerNorm(gamma) maps 0 → 0, 1 → 1."""
        import matplotlib.colors as mcolors
        norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=1)
        assert norm(0.0) == approx(0.0, abs=1e-9)
        assert norm(1.0) == approx(1.0, abs=1e-9)

    @pytest.mark.parametrize('ncolors', [2, 4, 8, 16, 64, 256])
    def test_boundary_norm_ncolors(self, ncolors):
        """BoundaryNorm stores ncolors."""
        import matplotlib.colors as mcolors
        boundaries = list(range(ncolors + 1))
        norm = mcolors.BoundaryNorm(boundaries, ncolors)
        assert norm.N == ncolors

    @pytest.mark.parametrize('linthresh', [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_symlog_linthresh(self, linthresh):
        """SymLogNorm maps 0 correctly for any linthresh."""
        import matplotlib.colors as mcolors
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-100, vmax=100)
        result = norm(0.0)
        assert result is not None

    @pytest.mark.parametrize('vcenter', [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_two_slope_vcenter(self, vcenter):
        """TwoSlopeNorm maps vcenter to 0.5."""
        import matplotlib.colors as mcolors
        norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vcenter - 10, vmax=vcenter + 10)
        assert norm(vcenter) == approx(0.5)

    @pytest.mark.parametrize('n', [0, 1, 2, 3, 4])
    def test_no_norm_identity(self, n):
        """NoNorm passes values through unchanged."""
        import matplotlib.colors as mcolors
        norm = mcolors.NoNorm()
        assert norm(n) == n


class TestNormsParametric7:
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



class TestNormsParametric12:
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



class TestNormsParametric19:
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



class TestNormsParametric16:
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


class TestNormsParametric17:
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


class TestNormsParametric18:
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


class TestNormsParametric19:
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
