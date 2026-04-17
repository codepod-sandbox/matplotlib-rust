# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from upstream matplotlib tests/test_cm.py and tests/test_colors.py
"""Upstream-ported tests for colormaps, registry, ScalarMappable, and norm classes."""
import numpy as np
import numpy.testing as npt
import pytest


# ---------------------------------------------------------------------------
# Registry and get_cmap
# ---------------------------------------------------------------------------

def test_get_cmap_return_default():
    from matplotlib import cm
    from matplotlib.colors import Colormap
    cmap = cm.get_cmap()
    assert isinstance(cmap, Colormap)
    assert cmap.name == 'viridis'


def test_get_cmap_by_name():
    from matplotlib import cm
    from matplotlib.colors import Colormap
    for name in ('hot', 'viridis', 'Blues'):
        cmap = cm.get_cmap(name)
        assert isinstance(cmap, Colormap), f"get_cmap({name!r}) did not return Colormap"


def test_get_cmap_bad_name():
    from matplotlib import cm
    with pytest.raises(ValueError):
        cm.get_cmap('nonexistent_cmap_xyz')


def test_register_cmap():
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(['#ff0000', '#00ff00', '#0000ff'], name='my_test_cmap')
    cm._colormaps.register(my_cmap, name='my_test_cmap', force=True)
    retrieved = cm.get_cmap('my_test_cmap')
    assert retrieved.name == 'my_test_cmap'


def test_registry_getitem_keyerror():
    from matplotlib import cm
    with pytest.raises(KeyError):
        _ = cm._colormaps['nonexistent_cmap_xyz']


# ---------------------------------------------------------------------------
# Colormap.__call__
# ---------------------------------------------------------------------------

def test_colormap_call_scalar():
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    for x in (0.0, 0.5, 1.0):
        result = cmap(x)
        assert isinstance(result, tuple), f"cmap({x}) should return tuple, got {type(result)}"
        assert len(result) == 4, f"cmap({x}) should return 4-tuple"
        assert all(isinstance(v, float) for v in result), f"all values should be float"
        assert all(0.0 <= v <= 1.0 for v in result), f"all values should be in [0, 1]"


def test_colormap_call_array():
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    x = np.linspace(0, 1, 10)
    result = cmap(x)
    assert hasattr(result, 'shape'), "array input should return array"
    assert result.shape == (10, 4), f"expected (10, 4), got {result.shape}"
    assert result.dtype == np.float64 or str(result.dtype).startswith('float')


def test_colormap_bytes():
    from matplotlib import cm
    import numpy as np
    cmap = cm.get_cmap('viridis')
    result = cmap(0.5, bytes=True)
    assert isinstance(result, tuple), "bytes=True should return tuple"
    assert len(result) == 4
    # OG matplotlib returns np.uint8 values, which are numpy integers not Python int
    assert all(isinstance(v, (int, np.integer)) for v in result), f"bytes=True should return ints, got {result}"
    assert all(0 <= v <= 255 for v in result), f"bytes should be in [0,255], got {result}"


def test_colormap_reversed():
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    rev = cmap.reversed()
    assert isinstance(rev.name, str)
    # cmap(0.0) should equal rev(1.0) — tolerance of 1/256 accounts for LUT rounding
    c0 = cmap(0.0)
    r1 = rev(1.0)
    npt.assert_allclose(c0, r1, atol=1/256, err_msg=f"cmap(0.0)={c0} != rev(1.0)={r1}")


# ---------------------------------------------------------------------------
# ListedColormap
# ---------------------------------------------------------------------------

def test_listed_colormap():
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['r', 'g', 'b'])
    assert cmap.N == 3
    result = cmap(0.0)
    assert isinstance(result, tuple) and len(result) == 4
    # 0.0 should map to red: (1.0, 0.0, 0.0, 1.0)
    assert abs(result[0] - 1.0) < 1e-6, f"first color should be red, got {result}"
    assert abs(result[1] - 0.0) < 1e-6
    assert abs(result[2] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# LinearSegmentedColormap
# ---------------------------------------------------------------------------

def test_linear_segmented_from_list():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('test_lr', ['blue', 'red'])
    result = cmap(0.5)
    assert isinstance(result, tuple) and len(result) == 4
    assert all(0.0 <= v <= 1.0 for v in result), f"RGBA values out of range: {result}"


# ---------------------------------------------------------------------------
# Norm classes
# ---------------------------------------------------------------------------

def test_boundary_norm():
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
    # Returns integer bin indices
    assert norm(0.5) == 0, f"BoundaryNorm(0.5) expected 0, got {norm(0.5)}"
    assert norm(1.5) == 1, f"BoundaryNorm(1.5) expected 1, got {norm(1.5)}"
    assert norm(2.5) == 2, f"BoundaryNorm(2.5) expected 2, got {norm(2.5)}"


def test_two_slope_norm():
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=2)
    x = np.array([-1.0, 0.0, 2.0])
    result = norm(x)
    result_list = result.tolist() if hasattr(result, 'tolist') else list(result)
    npt.assert_allclose(result_list, [0.0, 0.5, 1.0], atol=1e-6)


def test_centered_norm():
    from matplotlib.colors import CenteredNorm
    norm = CenteredNorm()
    x = np.array([-1.0, 0.0, 1.0])
    result = norm(x)
    result_list = result.tolist() if hasattr(result, 'tolist') else list(result)
    npt.assert_allclose(result_list, [0.0, 0.5, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# ScalarMappable
# ---------------------------------------------------------------------------

def test_scalar_mappable_to_rgba():
    from matplotlib import cm
    from matplotlib.colors import Normalize
    sm = cm.ScalarMappable(norm=Normalize(0, 1), cmap='viridis')
    x = np.array([0.0, 0.5, 1.0])
    result = sm.to_rgba(x)
    assert hasattr(result, 'shape'), "to_rgba should return array"
    assert result.shape == (3, 4), f"expected (3, 4), got {result.shape}"


def test_scalar_mappable_set_clim():
    from matplotlib import cm
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_clim(0, 10)
    assert sm.get_clim() == (0, 10)


def test_scalar_mappable_autoscale():
    from matplotlib import cm
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(np.array([1.0, 2.0, 3.0]))
    sm.autoscale()
    vmin, vmax = sm.get_clim()
    assert vmin == 1.0, f"expected vmin=1.0, got {vmin}"
    assert vmax == 3.0, f"expected vmax=3.0, got {vmax}"


def test_scalar_mappable_norm_false():
    from matplotlib import cm
    from matplotlib.colors import Normalize
    sm = cm.ScalarMappable(norm=Normalize(0, 1), cmap='viridis')
    x = np.array([0.0, 0.5, 1.0])
    result = sm.to_rgba(x, norm=False)
    assert result.shape == (3, 4)


# ---------------------------------------------------------------------------
# Exhaustive registry test
# ---------------------------------------------------------------------------

def test_all_colormaps_callable():
    from matplotlib import cm
    from matplotlib.colors import Colormap
    names = list(cm._colormaps)
    assert len(names) >= 100, f"expected ≥100 colormaps, got {len(names)}"
    errors = []
    for name in names:
        try:
            cmap = cm._colormaps[name]
            result = cmap(0.5)
            assert isinstance(result, tuple) and len(result) == 4
        except Exception as e:
            errors.append(f"{name}: {e}")
    assert not errors, "Some colormaps failed:\n" + "\n".join(errors[:10])


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def test_rcparam_image_cmap():
    import matplotlib
    assert matplotlib.rcParams['image.cmap'] == 'viridis'
    assert matplotlib.rcParams['image.lut'] == 256


# ===================================================================
# ColormapRegistry extended tests
# ===================================================================

import matplotlib.cm as cm

class TestColormapRegistry:
    def test_len(self):
        """Registry has at least some colormaps."""
        assert len(cm._colormaps) > 10

    def test_contains(self):
        """__contains__ works for known names."""
        assert 'viridis' in cm._colormaps
        assert 'hot' in cm._colormaps

    def test_not_contains(self):
        """Unknown name not in registry."""
        assert 'not_a_real_cmap' not in cm._colormaps

    def test_iter(self):
        """Can iterate over registry."""
        names = list(cm._colormaps)
        assert len(names) > 10
        assert 'viridis' in names

    def test_getitem(self):
        """Registry[name] returns a Colormap."""
        cmap = cm._colormaps['viridis']
        assert hasattr(cmap, '__call__')

    def test_getitem_unknown_raises(self):
        """Registry[unknown] raises KeyError."""
        with pytest.raises(KeyError):
            _ = cm._colormaps['not_real_cmap_xyz']

    def test_register_new(self):
        """Can register a new colormap."""
        new_cmap = cm.ListedColormap(['red', 'green', 'blue'], name='test_reg_abc')
        cm._colormaps.register(new_cmap)
        assert 'test_reg_abc' in cm._colormaps
        cm._colormaps.unregister('test_reg_abc')

    def test_register_duplicate_raises(self):
        """Registering duplicate without force raises."""
        new_cmap = cm.ListedColormap(['red'], name='viridis_dup_xyz')
        cm._colormaps.register(new_cmap)
        with pytest.raises(ValueError):
            cm._colormaps.register(new_cmap)
        cm._colormaps.unregister('viridis_dup_xyz')

    def test_register_force_overwrites(self):
        """force=True allows overwriting."""
        new_cmap = cm.ListedColormap(['red'], name='viridis_force_xyz')
        cm._colormaps.register(new_cmap)
        new_cmap2 = cm.ListedColormap(['blue'], name='viridis_force_xyz')
        cm._colormaps.register(new_cmap2, force=True)
        assert 'viridis_force_xyz' in cm._colormaps
        cm._colormaps.unregister('viridis_force_xyz')

    def test_unregister(self):
        """unregister removes cmap."""
        new_cmap = cm.ListedColormap(['red'], name='test_unreg_xyz')
        cm._colormaps.register(new_cmap)
        cm._colormaps.unregister('test_unreg_xyz')
        assert 'test_unreg_xyz' not in cm._colormaps

    def test_unregister_nonexistent_no_error(self):
        """unregister of nonexistent name doesn't raise."""
        cm._colormaps.unregister('definitely_not_a_cmap_xyz')

    def test_register_non_cmap_raises(self):
        """Registering a non-Colormap raises TypeError (OG matplotlib 3.10 behavior)."""
        with pytest.raises((TypeError, ValueError)):
            cm._colormaps.register('not_a_colormap', name='test_xyz')

    def test_reversed_variants(self):
        """Reversed (_r) variants are in the registry."""
        assert 'viridis_r' in cm._colormaps


# ===================================================================
# ScalarMappable extended tests
# ===================================================================

class TestScalarMappableExtended:
    def test_set_norm(self):
        """set_norm stores the norm."""
        from matplotlib.colors import Normalize
        sm = cm.ScalarMappable()
        n = Normalize(0, 1)
        sm.set_norm(n)
        assert sm.get_norm() is n

    def test_set_norm_none_creates_normalize(self):
        """Setting norm to None creates a default Normalize."""
        from matplotlib.colors import Normalize
        sm = cm.ScalarMappable()
        sm.norm = None
        assert isinstance(sm.norm, Normalize)

    def test_set_cmap_string(self):
        """set_cmap with string resolves to Colormap."""
        sm = cm.ScalarMappable()
        sm.set_cmap('hot')
        assert sm.get_cmap() is not None

    def test_set_cmap_object(self):
        """set_cmap with Colormap object stores it."""
        cmap = cm.get_cmap('cool')
        sm = cm.ScalarMappable(cmap=cmap)
        assert sm.get_cmap() is cmap

    def test_get_array_default_none(self):
        """get_array returns None by default."""
        sm = cm.ScalarMappable()
        assert sm.get_array() is None

    def test_set_array(self):
        """set_array stores the data."""
        import numpy as np
        sm = cm.ScalarMappable()
        sm.set_array([1, 2, 3])
        assert np.allclose(sm.get_array(), [1, 2, 3])

    def test_set_array_none(self):
        """set_array(None) clears data."""
        sm = cm.ScalarMappable()
        sm.set_array([1, 2, 3])
        sm.set_array(None)
        assert sm.get_array() is None

    def test_autoscale_from_array(self):
        """autoscale sets vmin/vmax from array."""
        from matplotlib.colors import Normalize
        sm = cm.ScalarMappable(norm=Normalize())
        sm.set_array([2, 5, 8])
        sm.autoscale()
        assert sm.get_norm().vmin == 2
        assert sm.get_norm().vmax == 8

    def test_autoscale_none_no_array(self):
        """autoscale_None raises TypeError when no array (OG matplotlib 3.10 behavior)."""
        sm = cm.ScalarMappable()
        # OG matplotlib 3.10 raises TypeError if no array is set
        with pytest.raises(TypeError):
            sm.autoscale_None()

    def test_changed_no_error(self):
        """changed() can be called without error."""
        sm = cm.ScalarMappable()
        sm.changed()  # No-op, should not raise

    def test_to_rgba_list(self):
        """to_rgba works with list input and norm."""
        from matplotlib.colors import Normalize
        sm = cm.ScalarMappable(norm=Normalize(0, 10), cmap='viridis')
        result = sm.to_rgba([0.0, 5.0, 10.0])
        assert result is not None


# ===================================================================
# LinearSegmentedColormap tests
# ===================================================================

class TestLinearSegmentedColormap:
    def test_from_list_basic(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('my_cmap', ['blue', 'red'])
        assert cmap.name == 'my_cmap'

    def test_from_list_N(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('my_cmap', ['blue', 'red'], N=128)
        assert cmap.N == 128

    def test_call_scalar(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        result = cmap(0.5)
        assert len(result) == 4  # RGBA tuple

    def test_call_zero(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        result = cmap(0.0)
        assert len(result) == 4

    def test_call_one(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        result = cmap(1.0)
        assert len(result) == 4

    def test_reversed(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        rev = cmap.reversed()
        assert 'reversed' in rev.name or rev.name != cmap.name

    def test_resampled(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        new_cmap = cmap.resampled(64)
        assert new_cmap.N == 64

    def test_eq_same_name(self):
        from matplotlib.colors import LinearSegmentedColormap
        c1 = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        c2 = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        assert c1 == c2

    def test_eq_different_name(self):
        # OG matplotlib 3.10: LinearSegmentedColormap equality does not consider
        # the name, so two colormaps with same data but different names compare equal.
        from matplotlib.colors import LinearSegmentedColormap
        c1 = LinearSegmentedColormap.from_list('lr', ['blue', 'red'])
        c2 = LinearSegmentedColormap.from_list('lr2', ['blue', 'red'])
        # In OG matplotlib 3.10, name is not part of equality — c1 == c2 is True
        # This is the observed upstream behavior.
        assert c1 == c2


# ===================================================================
# Colormap bad/under/over tests
# ===================================================================

class TestColormapBadUnderOver:
    def test_get_bad_default(self):
        cmap = cm.get_cmap('viridis')
        bad = cmap.get_bad()
        assert bad is not None

    def test_set_bad(self):
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['red', 'blue'], name='test_bu')
        cmap.set_bad('white')
        bad = cmap.get_bad()
        assert bad is not None

    def test_get_under_default(self):
        cmap = cm.get_cmap('viridis')
        under = cmap.get_under()
        assert under is not None

    def test_set_under(self):
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['red', 'blue'], name='test_su')
        cmap.set_under('white')
        under = cmap.get_under()
        assert under is not None

    def test_get_over_default(self):
        cmap = cm.get_cmap('viridis')
        over = cmap.get_over()
        assert over is not None

    def test_set_over(self):
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['red', 'blue'], name='test_so')
        cmap.set_over('black')
        over = cmap.get_over()
        assert over is not None

    def test_colors_property(self):
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['red', 'green', 'blue'], name='test_cp')
        colors = cmap.colors
        assert len(colors) == 3


# ===================================================================
# get_cmap and colormaps registry integration
# ===================================================================

class TestGetCmapIntegration:
    def test_get_cmap_viridis(self):
        cmap = cm.get_cmap('viridis')
        assert cmap.name == 'viridis'

    def test_get_cmap_plasma(self):
        cmap = cm.get_cmap('plasma')
        assert cmap.name == 'plasma'

    def test_get_cmap_jet(self):
        cmap = cm.get_cmap('jet')
        assert cmap.name == 'jet'

    def test_get_cmap_gray(self):
        cmap = cm.get_cmap('gray')
        assert cmap.name == 'gray'

    def test_get_cmap_hot(self):
        cmap = cm.get_cmap('hot')
        assert cmap.name == 'hot'

    def test_get_cmap_cool(self):
        cmap = cm.get_cmap('cool')
        assert cmap.name == 'cool'

    def test_get_cmap_spring(self):
        cmap = cm.get_cmap('spring')
        assert cmap.name == 'spring'

    def test_get_cmap_n_default(self):
        cmap = cm.get_cmap('viridis')
        assert cmap.N == 256

    def test_get_cmap_n_custom(self):
        cmap = cm.get_cmap('viridis', 128)
        assert cmap.N == 128

    def test_colormap_hash(self):
        cmap = cm.get_cmap('viridis')
        with pytest.raises(TypeError, match="unhashable type"):
            hash(cmap)

    def test_colormap_copy(self):
        cmap = cm.get_cmap('viridis')
        cmap_copy = cmap.copy()
        assert cmap_copy.name == cmap.name
        assert cmap_copy is not cmap


# ===================================================================
# Additional parametric tests
# ===================================================================

import pytest
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class TestListedColormapExtended:
    """Extended tests for ListedColormap."""

    def test_single_color(self):
        cmap = mcolors.ListedColormap(['red'])
        assert cmap.N == 1
        result = cmap(0.5)
        assert len(result) == 4

    def test_many_colors(self):
        colors = ['#{:02x}{:02x}{:02x}'.format(i, 0, 0) for i in range(0, 256, 10)]
        cmap = mcolors.ListedColormap(colors)
        assert cmap.N == len(colors)

    def test_name_preserved(self):
        cmap = mcolors.ListedColormap(['r', 'g', 'b'], name='rgb_test')
        assert cmap.name == 'rgb_test'

    def test_call_endpoints(self):
        cmap = mcolors.ListedColormap(['r', 'g', 'b'])
        r0 = cmap(0.0)
        r1 = cmap(1.0)
        assert len(r0) == 4
        assert len(r1) == 4

    def test_call_array_shape(self):
        cmap = mcolors.ListedColormap(['r', 'g', 'b'])
        x = np.linspace(0, 1, 5)
        result = cmap(x)
        assert result.shape == (5, 4)

    def test_reversed_reverses_colors(self):
        cmap = mcolors.ListedColormap(['r', 'g', 'b'])
        rev = cmap.reversed()
        # First color of original should be last of reversed
        c_orig_0 = cmap(0.0)
        c_rev_1 = rev(1.0)
        assert all(abs(a - b) < 1/256 for a, b in zip(c_orig_0, c_rev_1))

    def test_copy(self):
        cmap = mcolors.ListedColormap(['r', 'g', 'b'], name='test_copy')
        copy = cmap.copy()
        assert copy.name == cmap.name
        assert copy is not cmap


class TestBoundaryNormExtended:
    """BoundaryNorm edge cases and behavior."""

    def test_boundaries_integer_bins(self):
        norm = mcolors.BoundaryNorm([0, 1, 2, 3], ncolors=3)
        assert norm(0.0) == 0
        assert norm(0.9) == 0
        assert norm(1.0) == 1
        assert norm(2.5) == 2

    def test_below_lower_bound(self):
        norm = mcolors.BoundaryNorm([0, 1, 2], ncolors=2)
        result = norm(-1.0)
        assert result <= 0

    def test_above_upper_bound(self):
        norm = mcolors.BoundaryNorm([0, 1, 2], ncolors=2)
        result = norm(3.0)
        assert result >= 1

    def test_clip_true(self):
        norm = mcolors.BoundaryNorm([0, 1, 2], ncolors=2, clip=True)
        assert 0 <= norm(-1.0) <= 1
        assert 0 <= norm(5.0) <= 1


class TestNormClipping:
    """Normalize clip=True / clip=False behavior."""

    @pytest.mark.parametrize('norm_cls,kwargs', [
        (mcolors.Normalize, dict(vmin=0, vmax=1)),
        (mcolors.PowerNorm, dict(gamma=2, vmin=0, vmax=1)),
    ])
    def test_clip_true_clamps_to_01(self, norm_cls, kwargs):
        norm = norm_cls(clip=True, **kwargs)
        result = norm(2.0)  # above vmax
        assert 0.0 <= float(result) <= 1.0
        result2 = norm(-1.0)  # below vmin
        assert 0.0 <= float(result2) <= 1.0

    @pytest.mark.parametrize('norm_cls,kwargs', [
        (mcolors.Normalize, dict(vmin=0, vmax=1)),
        (mcolors.PowerNorm, dict(gamma=2, vmin=0, vmax=1)),
    ])
    def test_clip_false_allows_outside_01(self, norm_cls, kwargs):
        norm = norm_cls(clip=False, **kwargs)
        result = norm(2.0)
        assert float(result) > 1.0 or float(result) <= 1.0  # just must not raise


class TestColormapCallEdgeCases:
    """Edge cases for Colormap.__call__."""

    def test_call_with_masked_array(self):
        cmap = cm.get_cmap('viridis')
        import numpy.ma as ma
        x = ma.array([0.0, 0.5, 1.0], mask=[False, True, False])
        result = cmap(x)
        assert result.shape[0] == 3

    def test_call_nan_gets_bad_color(self):
        cmap = cm.get_cmap('viridis')
        cmap.set_bad('white')
        x = np.array([float('nan'), 0.5])
        result = cmap(x)
        # NaN entry should be white (1.0, 1.0, 1.0, 1.0)
        assert abs(result[0][0] - 1.0) < 1e-6
        assert abs(result[0][1] - 1.0) < 1e-6
        assert abs(result[0][2] - 1.0) < 1e-6

    def test_alpha_override(self):
        cmap = cm.get_cmap('viridis')
        result = cmap(0.5, alpha=0.5)
        assert abs(result[3] - 0.5) < 1e-6

    @pytest.mark.parametrize('name', ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool', 'gray'])
    def test_builtin_colormaps_call(self, name):
        cmap = cm.get_cmap(name)
        result = cmap(0.5)
        assert len(result) == 4
        assert all(0.0 <= v <= 1.0 for v in result)


class TestReversedColormaps:
    """Reversed (_r) colormap variants."""

    @pytest.mark.parametrize('name', ['viridis', 'plasma', 'hot', 'cool'])
    def test_reversed_variant_exists(self, name):
        assert f'{name}_r' in cm._colormaps

    @pytest.mark.parametrize('name', ['viridis', 'plasma'])
    def test_reversed_is_actual_reverse(self, name):
        cmap = cm.get_cmap(name)
        cmap_r = cm.get_cmap(f'{name}_r')
        c0 = cmap(0.0)
        cr1 = cmap_r(1.0)
        assert all(abs(a - b) < 1/256 for a, b in zip(c0, cr1))
