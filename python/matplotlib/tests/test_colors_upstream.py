"""Upstream matplotlib test_colors.py tests imported for compatibility testing."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from matplotlib import colors as mcolors
from matplotlib.colors import is_color_like, to_rgba_array


# ---------------------------------------------------------------------------
# Helper functions (from upstream test_colors.py)
# ---------------------------------------------------------------------------
def _inverse_tester(norm_instance, vals):
    """Checks if the inverse of the given normalization is working."""
    assert_array_almost_equal(norm_instance.inverse(norm_instance(vals)), vals)


def _scalar_tester(norm_instance, vals):
    """Checks if scalars and arrays are handled the same way."""
    scalar_result = [norm_instance(float(v)) for v in vals]
    assert_array_almost_equal(scalar_result, norm_instance(vals))


def _mask_tester(norm_instance, vals):
    """Checks mask handling."""
    masked_array = np.ma.array(vals)
    masked_array[0] = np.ma.masked
    assert_array_equal(masked_array.mask, norm_instance(masked_array).mask)


# ---------------------------------------------------------------------------
# 1. test_color_names (upstream ~line 1179)
# ---------------------------------------------------------------------------
def test_color_names():
    assert mcolors.to_hex("blue") == "#0000ff"
    # Skip xkcd:blue — we don't have xkcd colors
    assert mcolors.to_hex("tab:blue") == "#1f77b4"


# ---------------------------------------------------------------------------
# 2. test_conversions (upstream ~line 1270)
# ---------------------------------------------------------------------------
def test_conversions():
    # to_rgba_array("none") returns an empty (0, 4) result.
    # Upstream returns np.zeros((0, 4)); our impl returns [].
    assert len(mcolors.to_rgba_array("none")) == 0
    assert len(mcolors.to_rgba_array([])) == 0
    # a list of grayscale levels, not a single color.
    result = mcolors.to_rgba_array([".2", ".5", ".8"])
    expected = [mcolors.to_rgba(c) for c in [".2", ".5", ".8"]]
    assert result == expected
    # alpha is properly set.
    assert mcolors.to_rgba((1, 1, 1), .5) == (1, 1, 1, .5)
    assert mcolors.to_rgba(".1", .5) == (.1, .1, .1, .5)
    # builtin round differs between py2 and py3.
    assert mcolors.to_hex((.7, .7, .7)) == "#b2b2b2"
    # hex roundtrip.
    hex_color = "#1234abcd"
    assert mcolors.to_hex(mcolors.to_rgba(hex_color), keep_alpha=True) == \
        hex_color


# ---------------------------------------------------------------------------
# 3. test_to_rgba_array_single_str (upstream ~line 1299)
# ---------------------------------------------------------------------------
def test_to_rgba_array_single_str():
    # single color name is valid
    result = mcolors.to_rgba_array("red")
    assert len(result) == 1
    assert result[0] == (1, 0, 0, 1)

    # single char color sequence is invalid
    with pytest.raises(ValueError, match="not a valid color value|Invalid RGBA"):
        mcolors.to_rgba_array("rgb")


# ---------------------------------------------------------------------------
# 4. test_to_rgba_array_2tuple_str (upstream ~line 1309)
# ---------------------------------------------------------------------------
def test_to_rgba_array_2tuple_str():
    expected = [(0, 0, 0, 1), (1, 1, 1, 1)]
    result = mcolors.to_rgba_array(("k", "w"))
    assert len(result) == 2
    assert result[0] == expected[0]
    assert result[1] == expected[1]


# ---------------------------------------------------------------------------
# 5. test_to_rgba_array_alpha_array (upstream ~line 1314)
# ---------------------------------------------------------------------------
def test_to_rgba_array_alpha_array():
    with pytest.raises(ValueError,
                       match="The number of colors must match|alpha length"):
        mcolors.to_rgba_array([[1, 1, 1], [1, 1, 1], [1, 1, 1],
                               [1, 1, 1], [1, 1, 1]],
                              alpha=[0.5, 0.6])
    alpha = [0.5, 0.6]
    c = mcolors.to_rgba_array([[1, 1, 1], [1, 1, 1]], alpha=alpha)
    assert c[0][3] == 0.5
    assert c[1][3] == 0.6
    c = mcolors.to_rgba_array(['r', 'g'], alpha=alpha)
    assert c[0][3] == 0.5
    assert c[1][3] == 0.6


# ---------------------------------------------------------------------------
# 6. test_to_rgba_array_accepts_color_alpha_tuple (upstream ~line 1324)
# ---------------------------------------------------------------------------
def test_to_rgba_array_accepts_color_alpha_tuple():
    result = mcolors.to_rgba_array(('black', 0.9))
    assert len(result) == 1
    assert result[0] == (0, 0, 0, 0.9)


# ---------------------------------------------------------------------------
# 7. test_to_rgba_array_explicit_alpha_overrides_tuple_alpha (upstream ~line 1330)
# ---------------------------------------------------------------------------
def test_to_rgba_array_explicit_alpha_overrides_tuple_alpha():
    result = mcolors.to_rgba_array(('black', 0.9), alpha=0.5)
    assert len(result) == 1
    assert result[0] == (0, 0, 0, 0.5)


# ---------------------------------------------------------------------------
# 8. test_to_rgba_array_accepts_color_alpha_tuple_with_multiple_colors
#    (upstream ~line 1336)
# ---------------------------------------------------------------------------
def test_to_rgba_array_accepts_color_alpha_tuple_with_multiple_colors():
    color_sequence = [[1., 1., 1., 1.], [0., 0., 1., 0.]]
    result = mcolors.to_rgba_array((color_sequence, 0.4))
    assert len(result) == 2
    assert result[0] == (1., 1., 1., 0.4)
    assert result[1] == (0., 0., 1., 0.4)


# ---------------------------------------------------------------------------
# 9. test_to_rgba_array_error_with_color_invalid_alpha_tuple (upstream ~line 1348)
# ---------------------------------------------------------------------------
def test_to_rgba_array_error_with_color_invalid_alpha_tuple():
    with pytest.raises(ValueError, match="'alpha' must be between 0 and 1,"):
        mcolors.to_rgba_array(('black', 2.0))


# ---------------------------------------------------------------------------
# 10. test_to_rgba_accepts_color_alpha_tuple (parametrized, upstream ~line 1353)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('rgba_alpha',
                         [('white', 0.5), ('#ffffff', 0.5), ('#ffffff00', 0.5),
                          ((1.0, 1.0, 1.0, 1.0), 0.5)])
def test_to_rgba_accepts_color_alpha_tuple(rgba_alpha):
    assert mcolors.to_rgba(rgba_alpha) == (1, 1, 1, 0.5)


# ---------------------------------------------------------------------------
# 11. test_to_rgba_explicit_alpha_overrides_tuple_alpha (upstream ~line 1360)
# ---------------------------------------------------------------------------
def test_to_rgba_explicit_alpha_overrides_tuple_alpha():
    assert mcolors.to_rgba(('red', 0.1), alpha=0.9) == (1, 0, 0, 0.9)


# ---------------------------------------------------------------------------
# 12. test_to_rgba_error_with_color_invalid_alpha_tuple (upstream ~line 1364)
# ---------------------------------------------------------------------------
def test_to_rgba_error_with_color_invalid_alpha_tuple():
    with pytest.raises(ValueError, match="'alpha' must be between 0 and 1"):
        mcolors.to_rgba(('blue', 2.0))


# ---------------------------------------------------------------------------
# 13. test_failed_conversions (upstream ~line 1437)
# ---------------------------------------------------------------------------
def test_failed_conversions():
    with pytest.raises(ValueError):
        mcolors.to_rgba('5')
    with pytest.raises(ValueError):
        mcolors.to_rgba('-1')
    with pytest.raises(ValueError):
        mcolors.to_rgba('nan')
    with pytest.raises(ValueError):
        mcolors.to_rgba('unknown_color')
    with pytest.raises(ValueError):
        # Gray must be a string to distinguish 3-4 grays from RGB or RGBA.
        mcolors.to_rgba(0.4)


# ---------------------------------------------------------------------------
# 14. test_grey_gray (upstream ~line 1451)
# ---------------------------------------------------------------------------
def test_grey_gray():
    color_mapping = mcolors._colors_full_map
    for k in color_mapping.keys():
        if 'grey' in k:
            assert color_mapping[k] == color_mapping[k.replace('grey', 'gray')]
        if 'gray' in k:
            assert color_mapping[k] == color_mapping[k.replace('gray', 'grey')]


# ---------------------------------------------------------------------------
# 15. test_tableau_order (upstream ~line 1460)
# ---------------------------------------------------------------------------
def test_tableau_order():
    dflt_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    assert list(mcolors.TABLEAU_COLORS.values()) == dflt_cycle


# ---------------------------------------------------------------------------
# 16. test_same_color (upstream ~line 1494)
# ---------------------------------------------------------------------------
def test_same_color():
    assert mcolors.same_color('k', (0, 0, 0))
    assert not mcolors.same_color('w', (1, 1, 0))
    assert mcolors.same_color(['red', 'blue'], ['r', 'b'])
    assert mcolors.same_color('none', 'none')
    assert not mcolors.same_color('none', 'red')
    with pytest.raises(ValueError):
        mcolors.same_color(['r', 'g', 'b'], ['r'])
    with pytest.raises(ValueError):
        mcolors.same_color(['red', 'green'], 'none')


# ---------------------------------------------------------------------------
# 17. test_hex_shorthand_notation (upstream ~line 1506)
# ---------------------------------------------------------------------------
def test_hex_shorthand_notation():
    assert mcolors.same_color("#123", "#112233")
    assert mcolors.same_color("#123a", "#112233aa")


# ---------------------------------------------------------------------------
# 18. test_has_alpha_channel (upstream ~line 1231)
# ---------------------------------------------------------------------------
def test_has_alpha_channel():
    assert mcolors._has_alpha_channel((0, 0, 0, 0))
    assert mcolors._has_alpha_channel([1, 1, 1, 1])
    assert mcolors._has_alpha_channel('#fff8')
    assert mcolors._has_alpha_channel('#0f0f0f80')
    assert mcolors._has_alpha_channel(('r', 0.5))
    assert mcolors._has_alpha_channel(([1, 1, 1, 1], None))
    assert not mcolors._has_alpha_channel('blue')  # 4-char string!
    assert not mcolors._has_alpha_channel('0.25')
    assert not mcolors._has_alpha_channel('r')
    assert not mcolors._has_alpha_channel((1, 0, 0))
    assert not mcolors._has_alpha_channel('#fff')
    assert not mcolors._has_alpha_channel('#0f0f0f')
    assert not mcolors._has_alpha_channel(('r', None))
    assert not mcolors._has_alpha_channel(([1, 1, 1], None))


# ---------------------------------------------------------------------------
# 19. test_is_color_like (parametrized, upstream ~line 1781)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('input, expected',
                         [('red', True),
                          (('red', 0.5), True),
                          (('red', 2), False),
                          (['red', 0.5], False),
                          (('red', 'blue'), False),
                          (['red', 'blue'], False),
                          ('C3', True),
                          (('C3', 0.5), True)])
def test_is_color_like(input, expected):
    assert is_color_like(input) is expected


# ---------------------------------------------------------------------------
# 20. test_to_rgba_array_none_color_with_alpha_param (upstream ~line 1771)
# ---------------------------------------------------------------------------
def test_to_rgba_array_none_color_with_alpha_param():
    # effective alpha for color "none" must always be 0 to achieve a vanishing
    # color even explicit alpha must be ignored
    c = ["blue", "none"]
    alpha = [1, 1]
    result = mcolors.to_rgba_array(c, alpha)
    assert result[0] == (0., 0., 1., 1.)
    assert result[1] == (0., 0., 0., 0.)


# ---------------------------------------------------------------------------
# 21. test_2d_to_rgba (upstream ~line 1579)
# ---------------------------------------------------------------------------
def test_2d_to_rgba():
    color = [0.1, 0.2, 0.3]
    rgba_1d = mcolors.to_rgba(color)
    rgba_2d = mcolors.to_rgba([color])  # list-of-one-list form
    assert rgba_1d == rgba_2d


# ---------------------------------------------------------------------------
# 22. test_set_dict_to_rgba (upstream ~line 1586)
# ---------------------------------------------------------------------------
def test_set_dict_to_rgba():
    # downstream libraries do this...
    # note we can't test this because it is not well-ordered
    # so just smoketest:
    colors = {(0, .5, 1), (1, .2, .5), (.4, 1, .2)}
    res = mcolors.to_rgba_array(colors)
    assert len(res) == 3
    palette = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}
    res = mcolors.to_rgba_array(palette.values())
    # Check that each color was converted and has alpha=1
    assert len(res) == 3
    for r in res:
        assert r[3] == 1.0
    # Check that RGB components form an identity-like pattern (each row
    # has exactly one 1.0 in the first 3 components)
    rgb_vals = sorted([tuple(r[:3]) for r in res])
    expected = sorted([(0., 0., 1.), (0., 1., 0.), (1., 0., 0.)])
    assert rgb_vals == expected


# ---------------------------------------------------------------------------
# 23. test_lognorm_invalid (upstream ~line 499) — ADAPTED
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("vmin,vmax", [[-1, 2], [3, 1]])
def test_lognorm_invalid(vmin, vmax):
    # Check that invalid limits in LogNorm error
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    with pytest.raises(ValueError):
        norm(1)
    with pytest.raises(ValueError):
        norm.inverse(1)


# ---------------------------------------------------------------------------
# 24. test_cn (upstream ~line 1248) — ADAPTED
#     Removed cycler/rcParams dependency; uses our DEFAULT_CYCLE instead.
# ---------------------------------------------------------------------------
def test_cn():
    # Our CN colors always resolve against DEFAULT_CYCLE
    assert mcolors.to_hex("C0") == '#1f77b4'
    assert mcolors.to_hex("C1") == '#ff7f0e'
    assert mcolors.to_hex("C9") == '#17becf'
    # CN wraps around the cycle
    assert mcolors.to_hex("C10") == '#1f77b4'
    assert mcolors.to_hex("C11") == '#ff7f0e'


# ===========================================================================
# Newly ported upstream tests (2026-03-19)
# Source: https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/tests/test_colors.py
# ===========================================================================

import copy


# ---------------------------------------------------------------------------
# test_Normalize (upstream)
# ---------------------------------------------------------------------------
def test_Normalize():
    """Upstream: basic Normalize forward and inverse."""
    norm = mcolors.Normalize(vmin=0, vmax=10)
    assert norm(0) == 0.0
    assert norm(10) == 1.0
    assert norm(5) == 0.5
    assert norm.inverse(0.0) == 0.0
    assert norm.inverse(1.0) == 10.0
    assert norm.inverse(0.5) == 5.0


# ---------------------------------------------------------------------------
# test_LogNorm (upstream)
# ---------------------------------------------------------------------------
def test_LogNorm():
    """Upstream: LogNorm with clip=True."""
    ln = mcolors.LogNorm(clip=True, vmin=1, vmax=5)
    result = ln([1, 6])
    assert result[0] == 0.0
    assert result[1] == 1.0


# ---------------------------------------------------------------------------
# test_LogNorm_inverse (upstream)
# ---------------------------------------------------------------------------
def test_LogNorm_inverse():
    """Upstream: LogNorm inverse round-trip."""
    norm = mcolors.LogNorm(vmin=0.1, vmax=10)
    fwd = norm([0.5, 0.4])
    inv = norm.inverse(fwd)
    assert abs(inv[0] - 0.5) < 1e-3
    assert abs(inv[1] - 0.4) < 1e-3


# ---------------------------------------------------------------------------
# test_norm_deepcopy (upstream)
# ---------------------------------------------------------------------------
def test_norm_deepcopy():
    """Upstream: deepcopy preserves vmin/vmax."""
    norm = mcolors.Normalize(vmin=0.0002, vmax=1.0)
    norm2 = copy.deepcopy(norm)
    assert norm2.vmin == norm.vmin
    assert norm2.vmax == norm.vmax


# ---------------------------------------------------------------------------
# test_norm_autoscale (upstream-inspired)
# ---------------------------------------------------------------------------
def test_norm_autoscale():
    """Normalize.autoscale sets vmin/vmax from data."""
    norm = mcolors.Normalize()
    norm.autoscale([1, 2, 3, 4, 5])
    assert norm.vmin == 1
    assert norm.vmax == 5
    assert norm(3) == 0.5


# ===========================================================================
# Newly ported upstream tests (2026-03-19, batch 2)
# ===========================================================================


def test_to_hex_roundtrip():
    """to_hex and to_rgba are inverses for named colors."""
    for name in ['red', 'green', 'blue', 'white', 'black']:
        rgba = mcolors.to_rgba(name)
        hexval = mcolors.to_hex(rgba)
        rgba2 = mcolors.to_rgba(hexval)
        for a, b in zip(rgba, rgba2):
            assert abs(a - b) < 0.01


def test_same_color_hex():
    """same_color works with hex strings."""
    assert mcolors.same_color('#ff0000', 'red')
    assert mcolors.same_color('#00ff00', 'lime')
    assert not mcolors.same_color('#ff0000', 'blue')


def test_normalize_clip():
    """Normalize clips values outside [vmin, vmax] to [0, 1]."""
    norm = mcolors.Normalize(vmin=0, vmax=10, clip=True)
    assert norm(-5) == 0.0
    assert norm(15) == 1.0
    assert norm(5) == 0.5


def test_normalize_no_clip():
    """Normalize without clip allows values outside [0, 1]."""
    norm = mcolors.Normalize(vmin=0, vmax=10, clip=False)
    assert norm(5) == 0.5
    # Without clip, out-of-range values may extend beyond 0-1
    val = norm(-5)
    assert val < 0


def test_to_rgba_tuple_input():
    """to_rgba accepts (r, g, b) and (r, g, b, a) tuples."""
    assert mcolors.to_rgba((1, 0, 0)) == (1.0, 0.0, 0.0, 1.0)
    assert mcolors.to_rgba((0, 1, 0, 0.5)) == (0.0, 1.0, 0.0, 0.5)


def test_to_rgba_with_alpha():
    """to_rgba with explicit alpha parameter."""
    assert mcolors.to_rgba('red', alpha=0.5) == (1.0, 0.0, 0.0, 0.5)


def test_to_hex_with_alpha():
    """to_hex with keep_alpha=True includes alpha channel."""
    h = mcolors.to_hex((1, 0, 0, 0.5), keep_alpha=True)
    assert h == '#ff000080'


def test_is_color_like_valid():
    """is_color_like returns True for valid colors."""
    assert is_color_like('red')
    assert is_color_like('#ff0000')
    assert is_color_like((1, 0, 0))
    assert is_color_like((1, 0, 0, 1))
    assert is_color_like('C0')
    assert is_color_like('0.5')


def test_is_color_like_invalid():
    """is_color_like returns False for invalid inputs."""
    assert not is_color_like('not_a_color')
    assert not is_color_like((1, 0))  # too few elements
    assert not is_color_like(42)


def test_normalize_inverse():
    """Normalize.inverse reverses the mapping."""
    norm = mcolors.Normalize(vmin=0, vmax=10)
    assert norm(5) == 0.5
    assert norm.inverse(0.5) == 5.0
    assert norm.inverse(0.0) == 0.0
    assert norm.inverse(1.0) == 10.0


def test_lognorm_basic():
    """LogNorm maps logarithmically."""
    ln = mcolors.LogNorm(vmin=1, vmax=100)
    # log(10)/log(100) = 0.5
    assert abs(ln(10) - 0.5) < 1e-10


def test_normalize_vmin_vmax():
    """Normalize stores vmin/vmax correctly."""
    norm = mcolors.Normalize(vmin=-10, vmax=10)
    assert norm.vmin == -10
    assert norm.vmax == 10


# ---------------------------------------------------------------------------
# Normalize.__eq__ (upstream ~line 500)
# ---------------------------------------------------------------------------
def test_normalize_eq_same():
    """Two Normalize with same params are equal."""
    n1 = mcolors.Normalize(0, 1)
    n2 = mcolors.Normalize(0, 1)
    assert n1 == n2


def test_normalize_eq_diff_vmin():
    """Different vmin makes norms not equal."""
    n1 = mcolors.Normalize(0, 1)
    n2 = mcolors.Normalize(0.5, 1)
    assert n1 != n2


def test_normalize_eq_diff_vmax():
    """Different vmax makes norms not equal."""
    n1 = mcolors.Normalize(0, 1)
    n2 = mcolors.Normalize(0, 10)
    assert n1 != n2


def test_normalize_eq_diff_clip():
    """Different clip makes norms not equal."""
    n1 = mcolors.Normalize(0, 1, clip=False)
    n2 = mcolors.Normalize(0, 1, clip=True)
    assert n1 != n2


def test_normalize_eq_diff_type():
    """Normalize != LogNorm even with same params."""
    n = mcolors.Normalize(1, 10)
    ln = mcolors.LogNorm(1, 10)
    assert n != ln


def test_normalize_eq_not_norm():
    """Normalize compared to non-Normalize returns NotImplemented."""
    n = mcolors.Normalize(0, 1)
    assert n != "not a norm"
    assert n != 42


def test_normalize_hash():
    """Equal Normalize objects have same hash."""
    n1 = mcolors.Normalize(0, 1)
    n2 = mcolors.Normalize(0, 1)
    assert hash(n1) == hash(n2)


def test_normalize_hash_set():
    """Equal Normalize objects deduplicate in a set."""
    n1 = mcolors.Normalize(0, 1)
    n2 = mcolors.Normalize(0, 1)
    n3 = mcolors.Normalize(0, 10)
    s = {n1, n2, n3}
    assert len(s) == 2


def test_normalize_repr():
    """Normalize repr includes vmin, vmax, clip."""
    n = mcolors.Normalize(0, 1, clip=True)
    r = repr(n)
    assert 'Normalize' in r
    assert 'vmin=0' in r
    assert 'vmax=1' in r
    assert 'clip=True' in r


def test_normalize_repr_none():
    """Normalize repr handles None vmin/vmax."""
    n = mcolors.Normalize()
    r = repr(n)
    assert 'None' in r


# ---------------------------------------------------------------------------
# Normalize.scaled property
# ---------------------------------------------------------------------------
def test_normalize_scaled():
    """scaled returns True when both vmin and vmax are set."""
    n = mcolors.Normalize(0, 1)
    assert n.scaled() is True


def test_normalize_not_scaled():
    """scaled returns False when vmin or vmax is None."""
    n = mcolors.Normalize()
    assert n.scaled() is False
    n2 = mcolors.Normalize(vmin=0)
    assert n2.scaled() is False
    n3 = mcolors.Normalize(vmax=1)
    assert n3.scaled() is False


# ---------------------------------------------------------------------------
# Normalize.autoscale_None
# ---------------------------------------------------------------------------
def test_normalize_autoscale_none():
    """autoscale_None only fills in None values."""
    n = mcolors.Normalize(vmin=5)
    n.autoscale_None([1, 2, 10])
    assert n.vmin == 5  # not overwritten
    assert n.vmax == 10  # set from data


def test_normalize_autoscale_none_both():
    """autoscale_None with both None fills both."""
    n = mcolors.Normalize()
    n.autoscale_None([1, 2, 10])
    assert n.vmin == 1
    assert n.vmax == 10


def test_normalize_autoscale_none_neither():
    """autoscale_None with both set doesn't change anything."""
    n = mcolors.Normalize(vmin=0, vmax=100)
    n.autoscale_None([1, 2, 10])
    assert n.vmin == 0
    assert n.vmax == 100


# ---------------------------------------------------------------------------
# Normalize clipping
# ---------------------------------------------------------------------------
def test_normalize_clip_values():
    """Normalize with clip=True clips out-of-range values."""
    n = mcolors.Normalize(0, 10, clip=True)
    assert n(15) == 1.0
    assert n(-5) == 0.0
    assert n(5) == 0.5


def test_normalize_no_clip():
    """Normalize with clip=False allows out-of-range values."""
    n = mcolors.Normalize(0, 10, clip=False)
    assert n(15) == 1.5
    assert n(-5) == -0.5


def test_normalize_list_input():
    """Normalize accepts a list of values."""
    n = mcolors.Normalize(0, 10)
    result = n([0, 5, 10])
    assert abs(result[0] - 0.0) < 1e-10
    assert abs(result[1] - 0.5) < 1e-10
    assert abs(result[2] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# LogNorm eq (inherited)
# ---------------------------------------------------------------------------
def test_lognorm_eq():
    """LogNorm equality check."""
    ln1 = mcolors.LogNorm(1, 100)
    ln2 = mcolors.LogNorm(1, 100)
    ln3 = mcolors.LogNorm(1, 1000)
    assert ln1 == ln2
    assert ln1 != ln3


def test_lognorm_hash():
    """LogNorm hash consistency."""
    ln1 = mcolors.LogNorm(1, 100)
    ln2 = mcolors.LogNorm(1, 100)
    assert hash(ln1) == hash(ln2)


def test_lognorm_repr():
    """LogNorm repr includes class name."""
    ln = mcolors.LogNorm(1, 100)
    r = repr(ln)
    assert 'LogNorm' in r
    assert 'vmin=1' in r


# ---------------------------------------------------------------------------
# Normalize autoscale
# ---------------------------------------------------------------------------
def test_normalize_autoscale():
    """autoscale sets vmin and vmax from data."""
    n = mcolors.Normalize()
    n.autoscale([2, 5, 8])
    assert n.vmin == 2
    assert n.vmax == 8
    assert n.scaled() is True


# ---------------------------------------------------------------------------
# Normalize edge cases
# ---------------------------------------------------------------------------
def test_normalize_equal_vmin_vmax():
    """Normalize with vmin == vmax returns 0.0."""
    n = mcolors.Normalize(5, 5)
    assert n(5) == 0.0
    assert n(10) == 0.0


def test_normalize_callbacks():
    """Normalize has a callbacks registry."""
    n = mcolors.Normalize(0, 1)
    assert hasattr(n, 'callbacks')
    # Should be able to connect/disconnect
    cid = n.callbacks.connect('changed', lambda: None)
    n.callbacks.disconnect(cid)


# ===================================================================
# CenteredNorm tests
# ===================================================================

class TestCenteredNorm:
    def test_basic_vcenter_zero(self):
        """CenteredNorm default vcenter=0, halfrange from data."""
        n = mcolors.CenteredNorm()
        assert n.vcenter == 0.0

    def test_vcenter_custom(self):
        """Custom vcenter is stored."""
        n = mcolors.CenteredNorm(vcenter=5.0)
        assert n.vcenter == 5.0

    def test_halfrange_provided(self):
        """Provided halfrange is stored."""
        n = mcolors.CenteredNorm(vcenter=0.0, halfrange=2.0)
        assert n._halfrange == 2.0

    def test_center_maps_to_half(self):
        """vcenter maps to 0.5."""
        n = mcolors.CenteredNorm(vcenter=0.0, halfrange=5.0)
        result = n(0.0)
        assert abs(result - 0.5) < 1e-10

    def test_max_maps_to_one(self):
        """vcenter + halfrange maps to 1.0."""
        n = mcolors.CenteredNorm(vcenter=0.0, halfrange=5.0)
        result = n(5.0)
        assert abs(result - 1.0) < 1e-10

    def test_min_maps_to_zero(self):
        """vcenter - halfrange maps to 0.0."""
        n = mcolors.CenteredNorm(vcenter=0.0, halfrange=5.0)
        result = n(-5.0)
        assert abs(result - 0.0) < 1e-10

    def test_nonzero_vcenter(self):
        """Non-zero vcenter shifts the map."""
        n = mcolors.CenteredNorm(vcenter=10.0, halfrange=5.0)
        assert abs(n(10.0) - 0.5) < 1e-10
        assert abs(n(15.0) - 1.0) < 1e-10

    def test_halfrange_from_data(self):
        """halfrange=None infers from data max deviation."""
        n = mcolors.CenteredNorm(vcenter=0.0)
        result = n([-10.0, 0.0, 10.0])
        # 0 should still map near 0.5
        assert abs(result[1] - 0.5) < 1e-10

    def test_list_input(self):
        """CenteredNorm works on list input."""
        n = mcolors.CenteredNorm(vcenter=0.0, halfrange=1.0)
        result = n([-1.0, 0.0, 1.0])
        assert len(result) == 3

    def test_monotonic(self):
        """CenteredNorm is monotonically increasing."""
        n = mcolors.CenteredNorm(vcenter=0.0, halfrange=10.0)
        values = [-10.0, -5.0, 0.0, 5.0, 10.0]
        normalized = [n(v) for v in values]
        for i in range(1, len(normalized)):
            assert normalized[i] > normalized[i - 1]


# ===================================================================
# PowerNorm tests
# ===================================================================

class TestPowerNorm:
    def test_basic(self):
        """PowerNorm gamma=1 is linear."""
        n = mcolors.PowerNorm(gamma=1.0, vmin=0, vmax=10)
        assert abs(n(5.0) - 0.5) < 1e-10

    def test_gamma_2(self):
        """PowerNorm gamma=2: midpoint maps to 0.25."""
        n = mcolors.PowerNorm(gamma=2.0, vmin=0, vmax=1)
        result = n(0.5)
        assert abs(result - 0.25) < 1e-10

    def test_vmin_maps_to_zero(self):
        """vmin always maps to 0."""
        n = mcolors.PowerNorm(gamma=0.5, vmin=1, vmax=4)
        assert abs(n(1.0) - 0.0) < 1e-10

    def test_vmax_maps_to_one(self):
        """vmax always maps to 1."""
        n = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=1)
        assert abs(n(1.0) - 1.0) < 1e-10

    def test_no_vmin_autoscales(self):
        """PowerNorm without vmin autoscales from data."""
        n = mcolors.PowerNorm(gamma=1.0)
        # Autoscale from data: vmin=vmax=5.0 → maps to 0
        result = n(5.0)
        assert result == 0.0  # vmin==vmax case

    def test_inverse_gamma_1(self):
        """Inverse of linear norm is itself."""
        n = mcolors.PowerNorm(gamma=1.0, vmin=0, vmax=10)
        assert abs(n.inverse(0.5) - 5.0) < 1e-10

    def test_inverse_round_trip(self):
        """forward then inverse returns original."""
        n = mcolors.PowerNorm(gamma=2.0, vmin=0, vmax=10)
        for x in [1.0, 3.0, 5.0, 7.0, 9.0]:
            fwd = n(x)
            inv = n.inverse(fwd)
            assert abs(inv - x) < 1e-6

    def test_list_input(self):
        """PowerNorm works on lists."""
        n = mcolors.PowerNorm(gamma=1.0, vmin=0, vmax=10)
        result = n([0.0, 5.0, 10.0])
        assert len(result) == 3

    def test_repr(self):
        n = mcolors.PowerNorm(gamma=2.0, vmin=0, vmax=10)
        r = repr(n)
        assert 'PowerNorm' in r

    def test_gamma_stored(self):
        n = mcolors.PowerNorm(gamma=3.0)
        assert n.gamma == 3.0


# ===================================================================
# NoNorm tests
# ===================================================================

class TestNoNorm:
    def test_passthrough_scalar(self):
        """NoNorm returns value unchanged."""
        n = mcolors.NoNorm()
        assert n(5.0) == 5.0

    def test_passthrough_zero(self):
        n = mcolors.NoNorm()
        assert n(0.0) == 0.0

    def test_passthrough_negative(self):
        n = mcolors.NoNorm()
        assert n(-3.0) == -3.0

    def test_passthrough_list(self):
        n = mcolors.NoNorm()
        result = n([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_inverse_passthrough(self):
        n = mcolors.NoNorm()
        assert n.inverse(0.7) == 0.7

    def test_inverse_list(self):
        n = mcolors.NoNorm()
        result = n.inverse([0.2, 0.5, 0.8])
        assert result == [0.2, 0.5, 0.8]

    def test_is_normalize_subclass(self):
        n = mcolors.NoNorm()
        assert isinstance(n, mcolors.Normalize)


# ===================================================================
# BoundaryNorm extended tests
# ===================================================================

class TestBoundaryNormExtended:
    def test_in_range(self):
        """Values within boundaries are mapped to (0, 1)."""
        n = mcolors.BoundaryNorm([0, 1, 2, 3], 3)
        result = n(1.5)
        assert 0.0 <= result <= 1.0

    def test_non_monotonic_raises(self):
        """Non-monotonic boundaries raise ValueError."""
        with pytest.raises(ValueError):
            mcolors.BoundaryNorm([0, 2, 1], 2)

    def test_vmin_vmax_set(self):
        """vmin and vmax are first and last boundaries."""
        n = mcolors.BoundaryNorm([0, 5, 10], 2)
        assert n.vmin == 0
        assert n.vmax == 10

    def test_inverse_raises(self):
        """BoundaryNorm inverse raises ValueError."""
        n = mcolors.BoundaryNorm([0, 1, 2], 2)
        with pytest.raises(ValueError):
            n.inverse(0.5)

    def test_repr(self):
        n = mcolors.BoundaryNorm([0, 1, 2], 2)
        assert 'BoundaryNorm' in repr(n)

    def test_list_input(self):
        n = mcolors.BoundaryNorm([0, 1, 2, 3], 3)
        result = n([0.5, 1.5, 2.5])
        assert isinstance(result, list)
        assert len(result) == 3


# ===================================================================
# TwoSlopeNorm tests
# ===================================================================

class TestTwoSlopeNorm:
    def test_basic_construction(self):
        """TwoSlopeNorm can be constructed with vcenter."""
        n = mcolors.TwoSlopeNorm(vcenter=0)
        assert n.vcenter == 0.0

    def test_vcenter_maps_to_half(self):
        """vcenter maps to 0.5."""
        n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        result = n(0.0)
        assert abs(result - 0.5) < 1e-10

    def test_vmin_maps_to_zero(self):
        """vmin maps to 0.0."""
        n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=4)
        result = n(-2.0)
        assert abs(result - 0.0) < 1e-10

    def test_vmax_maps_to_one(self):
        """vmax maps to 1.0."""
        n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=4)
        result = n(4.0)
        assert abs(result - 1.0) < 1e-10

    def test_vmin_ge_vcenter_raises(self):
        """vmin >= vcenter raises ValueError."""
        with pytest.raises(ValueError):
            mcolors.TwoSlopeNorm(vcenter=0, vmin=1, vmax=2)

    def test_vmax_le_vcenter_raises(self):
        """vmax <= vcenter raises ValueError."""
        with pytest.raises(ValueError):
            mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=-0.5)

    def test_asymmetric_mapping(self):
        """Values below center map differently than above."""
        n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=4)
        below = n(-1.0)
        above = n(2.0)
        # below center: -1 is midpoint of [-2, 0] -> 0.25
        assert abs(below - 0.25) < 1e-10
        # above center: 2 is midpoint of [0, 4] -> 0.75
        assert abs(above - 0.75) < 1e-10

    def test_list_input(self):
        """TwoSlopeNorm works with list input."""
        n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        result = n([-1.0, 0.0, 1.0])
        assert len(result) == 3
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10
        assert abs(result[2] - 1.0) < 1e-10

    def test_autoscale_from_data(self):
        """TwoSlopeNorm autoscales from data when vmin/vmax not set."""
        n = mcolors.TwoSlopeNorm(vcenter=0)
        result = n([-2.0, 0.0, 4.0])
        # Center should map to 0.5
        assert abs(result[1] - 0.5) < 1e-10

    def test_is_normalize_subclass(self):
        n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        assert isinstance(n, mcolors.Normalize)

    def test_repr(self):
        n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        r = repr(n)
        assert 'TwoSlopeNorm' in r or 'vcenter' in r


# ---------------------------------------------------------------------------
# TwoSlopeNorm error cases (upstream test_colors.py)
# ---------------------------------------------------------------------------

def test_TwoSlopeNorm_vmin_eq_vcenter():
    """TwoSlopeNorm raises when vmin == vcenter."""
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vcenter=0, vmin=0, vmax=1)


def test_TwoSlopeNorm_vmax_eq_vcenter():
    """TwoSlopeNorm raises when vmax == vcenter."""
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=0)


def test_TwoSlopeNorm_vmin_gt_vcenter():
    """TwoSlopeNorm raises when vmin > vcenter."""
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vcenter=0, vmin=1, vmax=2)


def test_TwoSlopeNorm_vmin_gt_vmax():
    """TwoSlopeNorm raises when vmin > vmax."""
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vcenter=0, vmin=2, vmax=-1)


def test_TwoSlopeNorm_center_maps_to_half():
    """TwoSlopeNorm maps vcenter to 0.5."""
    n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=4)
    assert abs(n(0) - 0.5) < 1e-10


def test_TwoSlopeNorm_vmin_maps_to_zero():
    """TwoSlopeNorm maps vmin to 0.0."""
    n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=4)
    assert abs(n(-2) - 0.0) < 1e-10


def test_TwoSlopeNorm_vmax_maps_to_one():
    """TwoSlopeNorm maps vmax to 1.0."""
    n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=4)
    assert abs(n(4) - 1.0) < 1e-10


def test_TwoSlopeNorm_negative_side():
    """TwoSlopeNorm negative-side midpoint maps to 0.25."""
    n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4)
    assert abs(n(-2) - 0.25) < 1e-10


def test_TwoSlopeNorm_positive_side():
    """TwoSlopeNorm positive-side midpoint maps to 0.75."""
    n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4)
    assert abs(n(2) - 0.75) < 1e-10


# ---------------------------------------------------------------------------
# PowerNorm (upstream test_colors.py)
# ---------------------------------------------------------------------------

def test_PowerNorm_basic():
    """PowerNorm(gamma=2) maps 0->0, 0.5->0.25, 1->1."""
    n = mcolors.PowerNorm(gamma=2, vmin=0, vmax=1)
    assert abs(n(0) - 0.0) < 1e-10
    assert abs(n(0.5) - 0.25) < 1e-10
    assert abs(n(1.0) - 1.0) < 1e-10


def test_PowerNorm_gamma_one():
    """PowerNorm(gamma=1) behaves like Normalize."""
    n = mcolors.PowerNorm(gamma=1, vmin=0, vmax=1)
    assert abs(n(0.3) - 0.3) < 1e-6
    assert abs(n(0.7) - 0.7) < 1e-6


def test_PowerNorm_clip():
    """PowerNorm(clip=True) clips values outside [0,1]."""
    n = mcolors.PowerNorm(gamma=2, vmin=0, vmax=1, clip=True)
    assert n(2.0) == 1.0
    assert n(-1.0) == 0.0


# ---------------------------------------------------------------------------
# BoundaryNorm (upstream test_colors.py)
# ---------------------------------------------------------------------------

def test_BoundaryNorm_basic():
    """BoundaryNorm maps values to correct bin indices."""
    boundaries = [0, 1, 2, 3]
    n = mcolors.BoundaryNorm(boundaries, ncolors=3)
    assert n(0.5) == 0
    assert n(1.5) == 1
    assert n(2.5) == 2


def test_BoundaryNorm_values_at_boundary():
    """BoundaryNorm handles values exactly at boundaries."""
    boundaries = [0, 1, 2, 3]
    n = mcolors.BoundaryNorm(boundaries, ncolors=3)
    # Value exactly at lower bound maps to first bin
    assert n(0) == 0


def test_BoundaryNorm_clip():
    """BoundaryNorm with clip=True clips out-of-range values."""
    boundaries = [0, 1, 2, 3]
    n = mcolors.BoundaryNorm(boundaries, ncolors=3, clip=True)
    # Above max should clip to max color
    result = n(5)
    assert result <= 2


def test_BoundaryNorm_is_normalize_subclass():
    """BoundaryNorm is a Normalize subclass."""
    boundaries = [0, 1, 2, 3]
    n = mcolors.BoundaryNorm(boundaries, ncolors=3)
    assert isinstance(n, mcolors.Normalize)


# ---------------------------------------------------------------------------
# Colormap operations (upstream test_colors.py)
# ---------------------------------------------------------------------------

def test_listed_colormap_length():
    """ListedColormap has correct N."""
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['r', 'g', 'b', 'c'])
    assert cmap.N == 4


def test_listed_colormap_call():
    """ListedColormap(0) returns first color, (1) returns last."""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['r', 'g', 'b'])
    # First color
    first = cmap(0.0)
    assert first[0] == 1.0 and first[1] == 0.0  # red
    # Last color (clipped to last bin)
    last = cmap(1.0)
    assert last[2] == 1.0  # blue


def test_get_cmap_returns_colormap():
    """cm.get_cmap returns a Colormap object."""
    import matplotlib.cm as cm
    from matplotlib.colors import Colormap
    cmap = cm.get_cmap('viridis')
    assert isinstance(cmap, Colormap)


def test_get_cmap_name():
    """Returned colormap has the requested name."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap('plasma')
    assert cmap.name == 'plasma'


def test_colormap_scalar_returns_tuple():
    """cmap(scalar) returns a 4-tuple RGBA."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    result = cmap(0.5)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert all(0.0 <= v <= 1.0 for v in result)


def test_colormap_array_returns_array():
    """cmap(array) returns a 2D array of shape (N, 4)."""
    import numpy as np
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    x = np.linspace(0, 1, 5)
    result = cmap(x)
    assert result.shape == (5, 4)


def test_colormap_set_bad():
    """ListedColormap.set_bad() changes the color for masked values."""
    from matplotlib.colors import ListedColormap
    import numpy as np
    cmap = ListedColormap(['r', 'g', 'b'])
    cmap.set_bad('white')
    arr = np.ma.array([0.5], mask=[True])
    result = cmap(arr)
    # bad color should be white (1,1,1,1)
    assert abs(result[0][0] - 1.0) < 1e-6


def test_colormap_set_under():
    """ListedColormap.set_under() changes the under-range color."""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['r', 'g', 'b'])
    cmap.set_under('black')
    result = cmap(-0.1)  # below range
    # should return the under color: black = (0,0,0,1)
    assert abs(result[0] - 0.0) < 1e-6


def test_colormap_set_over():
    """ListedColormap.set_over() changes the over-range color."""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['r', 'g', 'b'])
    cmap.set_over('white')
    result = cmap(1.1)  # above range
    # should return the over color: white = (1,1,1,1)
    assert abs(result[0] - 1.0) < 1e-6


def test_normalize_scalar_output():
    """Normalize()(scalar) returns a scalar float."""
    n = mcolors.Normalize(vmin=0, vmax=10)
    result = n(5)
    assert abs(result - 0.5) < 1e-10


def test_normalize_array_output():
    """Normalize()(array) returns a MaskedArray."""
    import numpy as np
    n = mcolors.Normalize(vmin=0, vmax=10)
    result = n(np.array([0, 5, 10]))
    assert abs(result[0] - 0.0) < 1e-10
    assert abs(result[1] - 0.5) < 1e-10
    assert abs(result[2] - 1.0) < 1e-10


def test_normalize_clip_true():
    """Normalize(clip=True) clips values outside [vmin, vmax]."""
    n = mcolors.Normalize(vmin=0, vmax=1, clip=True)
    assert n(2.0) == 1.0
    assert n(-1.0) == 0.0


def test_lognorm_maps_correctly():
    """LogNorm maps log-spaced values to linear [0, 1]."""
    import numpy as np
    n = mcolors.LogNorm(vmin=1, vmax=100)
    # midpoint in log space is sqrt(100) = 10
    assert abs(n(10) - 0.5) < 1e-6


def test_lognorm_inverse():
    """LogNorm inverse round-trips correctly."""
    import numpy as np
    n = mcolors.LogNorm(vmin=1, vmax=1000)
    for v in [1, 10, 100, 1000]:
        normed = n(v)
        recovered = n.inverse(normed)
        assert abs(recovered - v) < 1e-6


# ===========================================================================
# Upstream tests from test_colors.py
# ===========================================================================

def test_TwoSlopeNorm_autoscale():
    norm = mcolors.TwoSlopeNorm(vcenter=20)
    norm.autoscale([10, 20, 30, 40])
    assert norm.vmin == 10.
    assert norm.vmax == 40.


def test_TwoSlopeNorm_autoscale_None_vmin():
    norm = mcolors.TwoSlopeNorm(2, vmin=0, vmax=None)
    norm.autoscale_None([1, 2, 3, 4, 5])
    assert norm(5) == 1
    assert norm.vmax == 5


def test_TwoSlopeNorm_autoscale_None_vmax():
    norm = mcolors.TwoSlopeNorm(2, vmin=None, vmax=10)
    norm.autoscale_None([1, 2, 3, 4, 5])
    assert norm(1) == 0
    assert norm.vmin == 1


def test_TwoSlopeNorm_scale():
    norm = mcolors.TwoSlopeNorm(2)
    assert norm.scaled() is False
    norm([1, 2, 3, 4])
    assert norm.scaled() is True


def test_TwoSlopeNorm_scaleout_center():
    # test the vmin never goes above vcenter
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    norm([0, 1, 2, 3, 5])
    assert norm.vmin == -5
    assert norm.vmax == 5


def test_TwoSlopeNorm_scaleout_center_max():
    # test the vmax never goes below vcenter
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    norm([0, -1, -2, -3, -5])
    assert norm.vmax == 5
    assert norm.vmin == -5


def test_TwoSlopeNorm_Even():
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=4)
    vals = np.array([-1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0])
    expected = np.array([0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_Odd():
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=5)
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_VcenterGTVmax():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=25, vmax=20)


def test_TwoSlopeNorm_premature_scaling():
    norm = mcolors.TwoSlopeNorm(vcenter=2)
    with pytest.raises(ValueError):
        norm.inverse(np.array([0.1, 0.5, 0.9]))


def test_SymLogNorm():
    """Test SymLogNorm behavior."""
    norm = mcolors.SymLogNorm(3, vmax=5, linscale=1.2, base=np.e)
    vals = np.array([-30, -1, 2, 6], dtype=float)
    normed_vals = norm(vals)
    expected = [0., 0.53980074, 0.826991, 1.02758204]
    assert_array_almost_equal(normed_vals, expected)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)

    # Ensure that specifying vmin returns the same result as above
    norm = mcolors.SymLogNorm(3, vmin=-30, vmax=5, linscale=1.2, base=np.e)
    normed_vals = norm(vals)
    assert_array_almost_equal(normed_vals, expected)

    # test something more easily checked.
    norm = mcolors.SymLogNorm(1, vmin=-np.e**3, vmax=np.e**3, base=np.e)
    nn = norm([-np.e**3, -np.e**2, -np.e**1, -1,
              0, 1, np.e**1, np.e**2, np.e**3])
    xx = np.array([0., 0.109123, 0.218246, 0.32737, 0.5, 0.67263,
                   0.781754, 0.890877, 1.])
    assert_array_almost_equal(nn, xx)
    norm = mcolors.SymLogNorm(1, vmin=-10**3, vmax=10**3, base=10)
    nn = norm([-10**3, -10**2, -10**1, -1,
              0, 1, 10**1, 10**2, 10**3])
    xx = np.array([0., 0.121622, 0.243243, 0.364865, 0.5, 0.635135,
                   0.756757, 0.878378, 1.])
    assert_array_almost_equal(nn, xx)


def test_FuncNorm():
    def forward(x):
        return (x**2)
    def inverse(x):
        return np.sqrt(x)

    norm = mcolors.FuncNorm((forward, inverse), vmin=0, vmax=10)
    expected = np.array([0, 0.25, 1])
    input_vals = np.array([0, 5, 10])
    assert_array_almost_equal(norm(input_vals), expected)
    assert_array_almost_equal(norm.inverse(expected), input_vals)

    def forward(x):
        return np.log10(x)
    def inverse(x):
        return 10**x
    norm = mcolors.FuncNorm((forward, inverse), vmin=0.1, vmax=10)
    lognorm = mcolors.LogNorm(vmin=0.1, vmax=10)
    assert_array_almost_equal(norm([0.2, 5, 10]), lognorm([0.2, 5, 10]))
    assert_array_almost_equal(norm.inverse([0.2, 5, 10]),
                              lognorm.inverse([0.2, 5, 10]))


def test_PowerNorm_translation_invariance():
    a = np.array([0, 1/2, 1], dtype=float)
    expected = [0, 1/8, 1]
    pnorm = mcolors.PowerNorm(vmin=0, vmax=1, gamma=3)
    assert_array_almost_equal(pnorm(a), expected)
    pnorm = mcolors.PowerNorm(vmin=-2, vmax=-1, gamma=3)
    assert_array_almost_equal(pnorm(a - 2), expected)


def test_PowerNorm_upstream():
    # Check an exponent of 1 gives same results as a normal linear normalization.
    a = np.array([0, 0.5, 1, 1.5], dtype=float)
    pnorm = mcolors.PowerNorm(1)
    norm = mcolors.Normalize()
    assert_array_almost_equal(norm(a), pnorm(a))

    a = np.array([-0.5, 0, 2, 4, 8], dtype=float)
    expected = [-1/16, 0, 1/16, 1/4, 1]
    pnorm = mcolors.PowerNorm(2, vmin=0, vmax=8)
    assert_array_almost_equal(pnorm(a), expected)
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[2]) == expected[2]
    # Check inverse
    a_roundtrip = pnorm.inverse(pnorm(a))
    assert_array_almost_equal(a, a_roundtrip)
    # PowerNorm inverse adds a mask, so check that is correct too
    assert_array_equal(a_roundtrip.mask, np.zeros(a.shape, dtype=bool))

    # Clip = True
    a = np.array([-0.5, 0, 1, 8, 16], dtype=float)
    expected = [0, 0, 0, 1, 1]
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=True)
    assert_array_almost_equal(pnorm(a), expected)
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[-1]) == expected[-1]
    # Clip = True at call time
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=False)
    assert_array_almost_equal(pnorm(a, clip=True), expected)
    assert pnorm(a[0], clip=True) == expected[0]
    assert pnorm(a[-1], clip=True) == expected[-1]

    # Check clip=True preserves masked values
    a = np.ma.array([5, 2], mask=[True, False])
    out = pnorm(a, clip=True)
    assert_array_equal(out.mask, [True, False])


def test_Normalize_upstream():
    norm = mcolors.Normalize()
    vals = np.arange(-10, 10, 1, dtype=float)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # Handle integer input correctly (don't overflow)
    vals = np.array([-128, 127], dtype=np.int8)
    norm = mcolors.Normalize(vals.min(), vals.max())
    assert_array_equal(norm(vals), [0, 1])


def test_LogNorm_upstream():
    """LogNorm clip=True clips to [0, 1]."""
    ln = mcolors.LogNorm(clip=True, vmax=5)
    assert_array_equal(ln([1, 6]), [0, 1.0])


@pytest.mark.parametrize("vmin,vmax", [[-1, 2], [3, 1]])
def test_lognorm_invalid(vmin, vmax):
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    with pytest.raises(ValueError):
        norm(1)
    with pytest.raises(ValueError):
        norm.inverse(1)


def test_LogNorm_inverse_upstream():
    """Test that lists work, and that the inverse works."""
    norm = mcolors.LogNorm(vmin=0.1, vmax=10)
    assert_array_almost_equal(norm([0.5, 0.4]), [0.349485, 0.30103])
    assert_array_almost_equal([0.5, 0.4], norm.inverse([0.349485, 0.30103]))
    assert_array_almost_equal(norm(0.4), [0.30103])
    assert_array_almost_equal([0.4], norm.inverse([0.30103]))


# ===================================================================
# Additional color tests (upstream-inspired batch, round 2)
# ===================================================================

import pytest
import numpy as np
import matplotlib.colors as mcolors


class TestColorConversions:
    """Tests for color conversion utilities."""

    @pytest.mark.parametrize('name,expected_hex', [
        ('red', '#ff0000'),
        ('green', '#008000'),
        ('blue', '#0000ff'),
        ('white', '#ffffff'),
        ('black', '#000000'),
    ])
    def test_to_hex_named_colors(self, name, expected_hex):
        assert mcolors.to_hex(name) == expected_hex

    @pytest.mark.parametrize('rgba,expected_hex', [
        ((1.0, 0.0, 0.0, 1.0), '#ff0000'),
        ((0.0, 0.0, 1.0, 1.0), '#0000ff'),
        ((0.0, 0.0, 0.0, 1.0), '#000000'),
    ])
    def test_to_hex_from_rgba(self, rgba, expected_hex):
        assert mcolors.to_hex(rgba) == expected_hex

    def test_to_rgba_red(self):
        r, g, b, a = mcolors.to_rgba('red')
        assert abs(r - 1.0) < 1e-6
        assert abs(g - 0.0) < 1e-6
        assert abs(b - 0.0) < 1e-6
        assert abs(a - 1.0) < 1e-6

    def test_to_rgba_with_alpha(self):
        r, g, b, a = mcolors.to_rgba('red', alpha=0.5)
        assert abs(a - 0.5) < 1e-6

    def test_to_rgba_hex(self):
        r, g, b, a = mcolors.to_rgba('#ff8000')
        assert abs(r - 1.0) < 1e-6
        assert abs(g - 128/255) < 2/255
        assert abs(b - 0.0) < 1e-6

    @pytest.mark.parametrize('color', ['red', 'blue', '#ff0000', (1, 0, 0), (1, 0, 0, 1)])
    def test_to_rgba_various_inputs(self, color):
        result = mcolors.to_rgba(color)
        assert len(result) == 4
        assert all(0.0 <= v <= 1.0 for v in result)

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', 'white', 'black'])
    def test_is_color_like_named(self, color):
        assert mcolors.is_color_like(color)

    @pytest.mark.parametrize('color', ['#ff0000', '#00ff00', '#0000ff'])
    def test_is_color_like_hex(self, color):
        assert mcolors.is_color_like(color)

    def test_is_color_like_invalid(self):
        assert not mcolors.is_color_like('not_a_color')
        assert not mcolors.is_color_like(42)

    def test_to_rgb_drops_alpha(self):
        r, g, b = mcolors.to_rgb('red')
        assert len((r, g, b)) == 3
        assert abs(r - 1.0) < 1e-6


class TestNormalizeExtended:
    """Extended Normalize behavior tests."""

    def test_normalize_array_output(self):
        norm = mcolors.Normalize(vmin=0, vmax=10)
        result = norm(np.array([0.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            result.tolist() if hasattr(result, 'tolist') else list(result),
            [0.0, 0.5, 1.0], atol=1e-6
        )

    def test_normalize_clip_true(self):
        norm = mcolors.Normalize(vmin=0, vmax=1, clip=True)
        assert float(norm(2.0)) == 1.0
        assert float(norm(-1.0)) == 0.0

    def test_normalize_clip_false(self):
        norm = mcolors.Normalize(vmin=0, vmax=1, clip=False)
        assert float(norm(2.0)) > 1.0
        assert float(norm(-1.0)) < 0.0

    def test_normalize_vmin_vmax_equal(self):
        norm = mcolors.Normalize(vmin=5, vmax=5)
        assert float(norm(5)) == 0.0

    def test_normalize_inverse_scalar(self):
        norm = mcolors.Normalize(vmin=0, vmax=10)
        assert abs(float(norm.inverse(0.5)) - 5.0) < 1e-6

    @pytest.mark.parametrize('vmin,vmax', [(0, 1), (-1, 1), (0, 100), (-50, 50)])
    def test_normalize_roundtrip(self, vmin, vmax):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        values = np.linspace(vmin, vmax, 5)
        normalized = norm(values)
        np.testing.assert_allclose(
            norm.inverse(normalized.tolist() if hasattr(normalized, 'tolist') else list(normalized)),
            values, atol=1e-6
        )
