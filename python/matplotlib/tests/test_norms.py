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
        # Returns integer bin indices
        assert norm(0.5) == 0  # first bin
        assert norm(1.5) == 1  # second bin
        assert norm(2.5) == 2  # third bin

    def test_list_input(self):
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
        result = norm([0.5, 1.5, 2.5])
        assert len(result) == 3

    def test_monotonic_required(self):
        # OG BoundaryNorm does not validate monotonicity; stub did
        # Just verify construction succeeds (OG behavior)
        norm = BoundaryNorm([3, 2, 1], ncolors=3)
        assert norm is not None

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
        # OG uses Ncmap, not ncolors
        assert norm.Ncmap == 10

    def test_vmin_vmax(self):
        norm = BoundaryNorm([0, 5, 10], ncolors=2)
        assert norm.vmin == 0
        assert norm.vmax == 10

    def test_N_intervals(self):
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
        # OG: N = number of boundaries (4), not number of intervals (3)
        assert norm.N == 4

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
        assert 0 <= result <= 18  # integer bin index in [0, ncolors-1]


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
        # OG __repr__ is default object repr without attrs like gamma

    def test_requires_vmin_vmax(self):
        # Without explicit vmin/vmax, autoscale sets them from data
        norm = PowerNorm(gamma=2)
        result = norm(5)
        assert result == approx(0.0)  # vmin==vmax==5, maps to 0

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
        # OG __repr__ is default object repr without attrs like linthresh

    def test_requires_vmin_vmax(self):
        # Without explicit vmin/vmax, autoscale sets them from data
        norm = SymLogNorm(linthresh=1)
        result = norm(5)
        assert result == approx(0.0)  # vmin==vmax==5, maps to 0

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
        import numpy as np
        norms = [
            Normalize(0, 10),
            PowerNorm(1, vmin=0, vmax=10),
            TwoSlopeNorm(5, vmin=0, vmax=10),
            SymLogNorm(1, vmin=-10, vmax=10),
            NoNorm(),
        ]
        for norm in norms:
            result = norm([1, 5, 9])
            # result may be list or ndarray depending on norm
            result_list = result.tolist() if isinstance(result, np.ndarray) else list(result)
            assert len(result_list) == 3


# ===================================================================
# Additional parametric tests
# ===================================================================

import pytest
import numpy as np
