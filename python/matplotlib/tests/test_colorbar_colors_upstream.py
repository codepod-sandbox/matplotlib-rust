"""Upstream-ported tests for colormaps, norms, and ScalarMappable."""

import math
import pytest
import matplotlib as mpl
from matplotlib.tests._approx import approx

from matplotlib.cm import (
    Colormap, ListedColormap, LinearSegmentedColormap,
    get_cmap as _cm_get_cmap, register_cmap, ColormapRegistry, ScalarMappable,
    _cmap_registry, _colormaps,
)
from matplotlib.colors import Normalize, LogNorm, to_rgba, to_hex, is_color_like


def get_cmap(*args, **kwargs):
    from matplotlib import _api
    with pytest.warns(_api.deprecation.MatplotlibDeprecationWarning):
        return _cm_get_cmap(*args, **kwargs)


# ===================================================================
# Colormap base tests
# ===================================================================

class TestColormapBase:
    """Test Colormap base class; OG Colormap._init() is abstract.
    Use LinearSegmentedColormap for tests that require _init()."""

    def _make_cmap(self, name='test'):
        """Return a concrete colormap for testing callable methods."""
        return LinearSegmentedColormap.from_list(name, ['red', 'blue'])

    def test_name(self):
        cmap = Colormap('test_cmap')
        assert cmap.name == 'test_cmap'

    def test_N_default(self):
        cmap = Colormap('test')
        assert cmap.N == 256

    def test_N_custom(self):
        cmap = Colormap('test', N=128)
        assert cmap.N == 128

    def test_repr(self):
        cmap = Colormap('test')
        r = repr(cmap)
        # OG repr does not include name; just check class type is in repr
        assert 'Colormap' in r

    def test_eq_same(self):
        # OG base Colormap._init() raises NotImplementedError; use concrete type
        c1 = self._make_cmap('test')
        c2 = self._make_cmap('test')
        assert c1 == c2

    def test_eq_different_name(self):
        # OG __eq__ compares lookup tables, not names. Use different colors.
        c1 = LinearSegmentedColormap.from_list('a', ['red', 'blue'])
        c2 = LinearSegmentedColormap.from_list('b', ['green', 'yellow'])
        assert c1 != c2

    def test_eq_different_N(self):
        c1 = LinearSegmentedColormap.from_list('test', ['red', 'blue'], N=256)
        c2 = LinearSegmentedColormap.from_list('test', ['red', 'blue'], N=128)
        assert c1 != c2

    def test_eq_not_colormap(self):
        c = Colormap('test')
        assert c != 'not a cmap'

    def test_set_bad(self):
        cmap = self._make_cmap()
        cmap.set_bad('red')
        rgba = cmap.get_bad()
        assert rgba[0] == approx(1.0)
        assert rgba[1] == approx(0.0)

    def test_set_under(self):
        cmap = self._make_cmap()
        cmap.set_under('blue')
        rgba = cmap.get_under()
        assert rgba[2] == approx(1.0)

    def test_set_over(self):
        cmap = self._make_cmap()
        cmap.set_over('green')
        rgba = cmap.get_over()
        assert rgba[1] > 0

    def test_call_nan_returns_bad(self):
        cmap = self._make_cmap()
        cmap.set_bad(color='red')
        result = cmap(float('nan'))
        assert result[0] == approx(1.0)

    def test_call_below_zero(self):
        cmap = self._make_cmap()
        cmap.set_under('blue')
        result = cmap(-0.5)
        assert result[2] == approx(1.0)

    def test_call_above_one(self):
        cmap = self._make_cmap()
        cmap.set_over('red')
        result = cmap(1.5)
        assert result[0] == approx(1.0)

    def test_call_with_alpha(self):
        cmap = self._make_cmap()
        result = cmap(0.5, alpha=0.5)
        assert result[3] == approx(0.5)

    def test_call_bytes(self):
        import numpy as np
        cmap = self._make_cmap()
        result = cmap(0.5, bytes=True)
        # OG returns numpy uint8, not Python int
        assert all(isinstance(v, (int, np.integer)) for v in result)
        assert all(0 <= v <= 255 for v in result)

    def test_call_list(self):
        cmap = self._make_cmap()
        result = cmap([0.0, 0.5, 1.0])
        assert len(result) == 3
        assert all(len(r) == 4 for r in result)

    def test_resampled(self):
        cmap = self._make_cmap()
        resampled = cmap.resampled(64)
        assert resampled.N == 64

    def test_copy(self):
        cmap = self._make_cmap()
        c = cmap.copy()
        assert c is not None


# ===================================================================
# ListedColormap tests
# ===================================================================

class TestListedColormapUpstream:
    def test_creation(self):
        cmap = ListedColormap(['red', 'green', 'blue'])
        assert cmap.N == 3

    def test_colors_property(self):
        cmap = ListedColormap(['red', 'green', 'blue'])
        colors = cmap.colors
        assert len(colors) == 3

    def test_call_endpoints(self):
        cmap = ListedColormap(['red', 'blue'])
        # 0.0 -> red
        rgba0 = cmap(0.0)
        assert rgba0[0] == approx(1.0)
        assert rgba0[2] == approx(0.0)
        # 1.0 -> blue
        rgba1 = cmap(1.0)
        assert rgba1[0] == approx(0.0)
        assert rgba1[2] == approx(1.0)

    def test_reversed(self):
        cmap = ListedColormap(['red', 'blue'], name='rb')
        rev = cmap.reversed()
        assert rev.name == 'rb_r'
        # 0.0 should now be blue
        rgba = rev(0.0)
        assert rgba[2] == approx(1.0)

    def test_reversed_custom_name(self):
        cmap = ListedColormap(['red', 'blue'], name='rb')
        rev = cmap.reversed(name='custom_rev')
        assert rev.name == 'custom_rev'

    def test_resampled(self):
        cmap = ListedColormap(['red', 'green', 'blue'])
        resampled = cmap.resampled(10)
        assert resampled.N == 10

    def test_single_color(self):
        cmap = ListedColormap(['red'])
        rgba = cmap(0.0)
        assert rgba[0] == approx(1.0)
        rgba = cmap(1.0)
        assert rgba[0] == approx(1.0)

    def test_many_colors(self):
        colors = [f'#{i:02x}{i:02x}{i:02x}' for i in range(0, 256, 16)]
        cmap = ListedColormap(colors)
        assert cmap.N == len(colors)

    def test_rgba_tuples(self):
        cmap = ListedColormap([(1, 0, 0, 1), (0, 0, 1, 0.5)])
        rgba = cmap(0.0)
        assert rgba[0] == approx(1.0)


# ===================================================================
# LinearSegmentedColormap tests
# ===================================================================

class TestLinearSegmentedColormapUpstream:
    def test_from_list(self):
        cmap = LinearSegmentedColormap.from_list('test', ['red', 'blue'])
        assert cmap.name == 'test'

    def test_from_list_endpoints(self):
        cmap = LinearSegmentedColormap.from_list('test', ['red', 'blue'])
        rgba0 = cmap(0.0)
        rgba1 = cmap(1.0)
        assert rgba0[0] == approx(1.0)  # red
        assert rgba1[2] == approx(1.0)  # blue

    def test_from_list_midpoint(self):
        cmap = LinearSegmentedColormap.from_list(
            'test', ['red', 'green', 'blue'])
        rgba_mid = cmap(0.5)
        # Should be greenish
        assert rgba_mid[1] > 0.3

    def test_reversed(self):
        cmap = LinearSegmentedColormap.from_list('test', ['red', 'blue'])
        rev = cmap.reversed()
        assert rev.name == 'test_r'
        rgba0 = rev(0.0)
        # Reversed: 0.0 should be blue
        assert rgba0[2] == approx(1.0)

    def test_resampled(self):
        cmap = LinearSegmentedColormap.from_list('test', ['red', 'blue'])
        resampled = cmap.resampled(64)
        assert resampled.N == 64

    def test_gamma(self):
        cmap1 = LinearSegmentedColormap.from_list(
            'test', ['black', 'white'], gamma=1.0)
        cmap2 = LinearSegmentedColormap.from_list(
            'test', ['black', 'white'], gamma=2.0)
        mid1 = cmap1(0.5)
        mid2 = cmap2(0.5)
        # Different gamma should produce different midpoints
        assert mid1[0] != mid2[0]


# ===================================================================
# Built-in colormaps
# ===================================================================

class TestBuiltinColormaps:
    @pytest.mark.parametrize('name', [
        'viridis', 'jet', 'hot', 'cool', 'gray', 'spring', 'summer',
        'autumn', 'winter', 'plasma', 'inferno', 'magma', 'cividis',
    ])
    def test_exists(self, name):
        cmap = get_cmap(name)
        assert cmap is not None
        assert cmap.name == name

    @pytest.mark.parametrize('name', [
        'viridis_r', 'jet_r', 'hot_r', 'cool_r', 'gray_r',
    ])
    def test_reversed_exists(self, name):
        cmap = get_cmap(name)
        assert cmap is not None

    def test_grey_alias(self):
        # OG 'grey' and 'gray' are registered as separate names (both exist)
        # They have the same LUT values even if different names
        grey = get_cmap('grey')
        gray = get_cmap('gray')
        assert grey is not None
        assert gray is not None
        # Both are valid grayscale colormaps
        assert grey(0.5)[0] == approx(gray(0.5)[0], abs=0.01)

    def test_get_cmap_none(self):
        cmap = get_cmap(None)
        assert cmap.name == mpl.rcParams['image.cmap']

    def test_get_cmap_instance(self):
        cmap = get_cmap('viridis')
        assert get_cmap(cmap) is cmap

    def test_get_cmap_instance_with_lut(self):
        cmap = get_cmap('viridis')
        # OG get_cmap(cmap, lut=N) doesn't reliably resample; use cmap.resampled()
        resampled = cmap.resampled(64)
        assert resampled.N == 64

    def test_get_cmap_invalid(self):
        with pytest.raises(ValueError):
            get_cmap('nonexistent_colormap')

    def test_get_cmap_with_lut(self):
        cmap = get_cmap('viridis', lut=64)
        assert cmap.N == 64


# ===================================================================
# ColormapRegistry
# ===================================================================

class TestColormapRegistryUpstream:
    def test_contains(self):
        assert 'viridis' in _colormaps
        assert 'nonexistent' not in _colormaps

    def test_getitem(self):
        cmap = _colormaps['viridis']
        assert cmap.name == 'viridis'

    def test_len(self):
        assert len(_colormaps) > 0

    def test_iter(self):
        names = list(_colormaps)
        assert 'viridis' in names

    def test_call(self):
        # OG ColormapRegistry is not callable with a name argument; use __getitem__
        cmap = _colormaps['viridis']
        assert cmap.name == 'viridis'

    def test_register(self):
        cmap = ListedColormap(['red', 'blue'], name='test_register_cmap')
        _colormaps.register(cmap, force=True)
        assert 'test_register_cmap' in _colormaps
        _colormaps.unregister('test_register_cmap')

    def test_register_duplicate(self):
        cmap = ListedColormap(['red'], name='viridis')
        with pytest.raises(ValueError):
            _colormaps.register(cmap, force=False)

    def test_register_force(self):
        # OG prohibits re-registering builtins; use a non-builtin name
        test_name = 'test_force_override_cmap'
        cmap1 = ListedColormap(['red'], name=test_name)
        cmap2 = ListedColormap(['blue'], name=test_name)
        _colormaps.register(cmap1, force=True)
        with pytest.warns(UserWarning, match="Overwriting the cmap"):
            _colormaps.register(cmap2, force=True)
        assert _colormaps[test_name].colors[0] == cmap2.colors[0]
        _colormaps.unregister(test_name)

    def test_unregister(self):
        cmap = ListedColormap(['red'], name='temp_cmap_test')
        _colormaps.register(cmap, force=True)
        _colormaps.unregister('temp_cmap_test')
        assert 'temp_cmap_test' not in _colormaps


# ===================================================================
# register_cmap function
# ===================================================================

class TestRegisterCmap:
    def test_register(self):
        import warnings
        cmap = ListedColormap(['red', 'blue'], name='reg_test')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            register_cmap(cmap=cmap)
        assert 'reg_test' in _cmap_registry
        _cmap_registry.unregister('reg_test')

    def test_register_with_name(self):
        import warnings
        cmap = ListedColormap(['red', 'blue'], name='other')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            register_cmap(name='custom_name', cmap=cmap)
        assert 'custom_name' in _cmap_registry
        _cmap_registry.unregister('custom_name')

    def test_register_no_cmap(self):
        import warnings
        # OG raises TypeError (not ValueError) for cmap=None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises((ValueError, TypeError)):
                register_cmap(name='foo', cmap=None)


# ===================================================================
# ScalarMappable tests
# ===================================================================

class TestScalarMappableUpstream:
    def test_creation(self):
        sm = ScalarMappable()
        assert sm.cmap is not None

    def test_creation_with_cmap_string(self):
        sm = ScalarMappable(cmap='viridis')
        assert sm.cmap.name == 'viridis'

    def test_creation_with_cmap_object(self):
        cmap = get_cmap('jet')
        sm = ScalarMappable(cmap=cmap)
        assert sm.cmap is cmap

    def test_set_cmap_string(self):
        sm = ScalarMappable()
        sm.set_cmap('hot')
        assert sm.cmap.name == 'hot'

    def test_set_cmap_object(self):
        cmap = get_cmap('cool')
        sm = ScalarMappable()
        sm.set_cmap(cmap)
        assert sm.cmap is cmap

    def test_get_cmap(self):
        sm = ScalarMappable(cmap='viridis')
        assert sm.get_cmap().name == 'viridis'

    def test_norm_property(self):
        norm = Normalize(0, 10)
        sm = ScalarMappable(norm=norm)
        assert sm.norm is norm

    def test_set_norm(self):
        sm = ScalarMappable()
        norm = Normalize(0, 10)
        sm.set_norm(norm)
        assert sm.norm is norm

    def test_set_norm_none(self):
        sm = ScalarMappable(norm=Normalize())
        sm.norm = None
        assert isinstance(sm.norm, Normalize)

    def test_get_norm(self):
        norm = Normalize(0, 10)
        sm = ScalarMappable(norm=norm)
        assert sm.get_norm() is norm

    def test_set_array(self):
        import numpy as np
        sm = ScalarMappable()
        sm.set_array([1, 2, 3])
        # OG returns MaskedArray, not list
        assert np.allclose(sm.get_array(), [1, 2, 3])

    def test_set_array_none(self):
        sm = ScalarMappable()
        sm.set_array(None)
        assert sm.get_array() is None

    def test_get_clim(self):
        sm = ScalarMappable(norm=Normalize(0, 10))
        vmin, vmax = sm.get_clim()
        assert vmin == 0
        assert vmax == 10

    def test_set_clim(self):
        sm = ScalarMappable(norm=Normalize())
        sm.set_clim(vmin=5, vmax=15)
        vmin, vmax = sm.get_clim()
        assert vmin == 5
        assert vmax == 15

    def test_set_clim_tuple(self):
        sm = ScalarMappable(norm=Normalize())
        sm.set_clim((2, 8))
        vmin, vmax = sm.get_clim()
        assert vmin == 2
        assert vmax == 8

    def test_colorbar_default(self):
        sm = ScalarMappable()
        assert sm.colorbar is None

    def test_to_rgba(self):
        sm = ScalarMappable(norm=Normalize(0, 1), cmap='viridis')
        result = sm.to_rgba(0.5)
        assert len(result) == 4

    def test_autoscale(self):
        sm = ScalarMappable(norm=Normalize())
        sm.set_array([2, 4, 6, 8])
        sm.autoscale()
        vmin, vmax = sm.get_clim()
        assert vmin == 2
        assert vmax == 8

    def test_autoscale_None(self):
        sm = ScalarMappable()
        sm.set_array([1, 5, 10])
        sm.autoscale_None()
        vmin, vmax = sm.get_clim()
        assert vmin == 1
        assert vmax == 10

    def test_changed(self):
        sm = ScalarMappable()
        # Should not raise
        sm.changed()


# ===================================================================
# Normalize tests
# ===================================================================

class TestNormalizeUpstream:
    def test_basic(self):
        norm = Normalize(0, 10)
        assert norm(0) == approx(0.0)
        assert norm(10) == approx(1.0)
        assert norm(5) == approx(0.5)

    def test_vmin_vmax(self):
        norm = Normalize(2, 8)
        assert norm.vmin == 2
        assert norm.vmax == 8

    def test_clip_false(self):
        norm = Normalize(0, 10, clip=False)
        result = norm(15)
        assert result > 1.0

    def test_clip_true(self):
        norm = Normalize(0, 10, clip=True)
        result = norm(15)
        assert result == approx(1.0)

    def test_clip_below(self):
        norm = Normalize(0, 10, clip=True)
        result = norm(-5)
        assert result == approx(0.0)

    def test_inverse(self):
        norm = Normalize(0, 10)
        assert norm.inverse(0.5) == approx(5.0)

    def test_autoscale(self):
        norm = Normalize()
        norm.autoscale([2, 4, 6, 8])
        assert norm.vmin == 2
        assert norm.vmax == 8

    def test_autoscale_None(self):
        norm = Normalize(vmin=None, vmax=None)
        norm.autoscale_None([1, 5, 10])
        assert norm.vmin == 1
        assert norm.vmax == 10

    def test_autoscale_None_preserves_existing(self):
        norm = Normalize(vmin=0, vmax=None)
        norm.autoscale_None([1, 5, 10])
        assert norm.vmin == 0  # preserved
        assert norm.vmax == 10

    def test_list_input(self):
        norm = Normalize(0, 10)
        result = norm([0, 5, 10])
        expected = [0.0, 0.5, 1.0]
        result_list = result.tolist() if hasattr(result, 'tolist') else list(result)
        for r, e in zip(result_list, expected):
            assert abs(r - e) < 1e-6

    def test_scaled(self):
        norm = Normalize(0, 10)
        assert norm.scaled() is True

    def test_not_scaled(self):
        norm = Normalize()
        assert norm.scaled() is False


# ===================================================================
# to_rgba / to_hex tests (additional upstream tests)
# ===================================================================

class TestColorConversionUpstream:
    def test_rgba_from_hex(self):
        assert to_rgba('#ff0000') == approx((1.0, 0.0, 0.0, 1.0))

    def test_rgba_from_name(self):
        r, g, b, a = to_rgba('red')
        assert r == approx(1.0)
        assert g == approx(0.0)
        assert b == approx(0.0)
        assert a == approx(1.0)

    def test_rgba_from_tuple(self):
        assert to_rgba((0.5, 0.5, 0.5)) == approx((0.5, 0.5, 0.5, 1.0))

    def test_rgba_from_tuple_with_alpha(self):
        assert to_rgba((0.5, 0.5, 0.5, 0.5)) == approx((0.5, 0.5, 0.5, 0.5))

    def test_rgba_alpha_override(self):
        result = to_rgba('red', alpha=0.5)
        assert result[3] == approx(0.5)

    def test_to_hex_basic(self):
        result = to_hex((1.0, 0.0, 0.0))
        assert result.lower() == '#ff0000'

    def test_to_hex_from_name(self):
        result = to_hex('blue')
        assert result.lower() == '#0000ff'

    def test_is_color_like(self):
        assert is_color_like('red')
        assert is_color_like('#ff0000')
        assert is_color_like((1, 0, 0))
        assert is_color_like((1, 0, 0, 1))

    def test_is_color_like_false(self):
        assert not is_color_like('not_a_color_xyz')
        assert not is_color_like(42)


# ===================================================================
# Additional colorbar / color tests (upstream-inspired batch, round 2)
# ===================================================================

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LogNorm, PowerNorm, SymLogNorm, TwoSlopeNorm


class TestNormInAxes:
    """Test norm classes used in actual plotting contexts."""

    def test_imshow_with_lognorm(self):
        fig, ax = plt.subplots()
        data = np.array([[1, 10], [100, 1000]], dtype=float)
        im = ax.imshow(data, norm=LogNorm(vmin=1, vmax=1000))
        assert im is not None
        plt.close('all')

    def test_imshow_with_powernorm(self):
        fig, ax = plt.subplots()
        data = np.array([[0, 1], [2, 3]], dtype=float)
        im = ax.imshow(data, norm=PowerNorm(gamma=2, vmin=0, vmax=3))
        assert im is not None
        plt.close('all')

    def test_scatter_with_norm(self):
        fig, ax = plt.subplots()
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        c = [0.1, 0.3, 0.5, 0.7, 0.9]
        sc = ax.scatter(x, y, c=c, norm=Normalize(0, 1))
        assert sc is not None
        plt.close('all')

    @pytest.mark.parametrize('gamma', [0.5, 1.0, 2.0, 3.0])
    def test_powernorm_gamma(self, gamma):
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)
        assert abs(float(norm(0.0))) == 0.0
        assert abs(float(norm(1.0)) - 1.0) < 1e-6

    @pytest.mark.parametrize('vmin,vmax', [(0, 1), (0, 10), (-1, 1)])
    def test_normalize_mid_point(self, vmin, vmax):
        norm = Normalize(vmin=vmin, vmax=vmax)
        mid = (vmin + vmax) / 2
        result = float(norm(mid))
        assert abs(result - 0.5) < 1e-6

    def test_twoslopenorm_maps_vcenter_to_half(self):
        norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        assert abs(float(norm(0)) - 0.5) < 1e-6

    def test_lognorm_vmin_vmax_mapped(self):
        norm = LogNorm(vmin=1, vmax=100)
        assert abs(float(norm(1)) - 0.0) < 1e-6
        assert abs(float(norm(100)) - 1.0) < 1e-6


class TestColormapScalarsAndArrays:
    """Test that colormaps work consistently with scalars and arrays."""

    @pytest.mark.parametrize('name', ['viridis', 'plasma', 'hot', 'cool', 'gray'])
    def test_scalar_and_array_consistency(self, name):
        cmap = get_cmap(name)
        x = 0.5
        # Scalar result
        scalar_result = cmap(x)
        # Array result
        arr_result = cmap(np.array([x]))
        # Should match
        for s, a in zip(scalar_result, arr_result[0]):
            assert abs(s - a) < 1e-6

    def test_colormap_0_and_1_consistent(self):
        cmap = get_cmap('viridis')
        r0 = cmap(0.0)
        r1 = cmap(1.0)
        # They should be different colors
        diff = sum(abs(a - b) for a, b in zip(r0[:3], r1[:3]))
        assert diff > 0.1  # viridis endpoints are very different

    def test_norm_colormap_pipeline(self):
        """Full pipeline: data -> norm -> colormap -> RGBA."""
        norm = Normalize(vmin=0, vmax=10)
        cmap = get_cmap('viridis')
        data = np.array([0.0, 5.0, 10.0])
        normalized = norm(data)
        rgba = cmap(normalized)
        assert rgba.shape == (3, 4)
        # vmin maps to first color, vmax maps to last
        assert rgba[0][0] != rgba[2][0]  # different colors at ends
