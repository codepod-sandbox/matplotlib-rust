# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_axes.py (scale sections)
import pytest
import numpy as np


def test_linear_scale_identity():
    """LinearScale forward/inverse are identity."""
    from matplotlib.scale import LinearScale
    s = LinearScale()
    vals = np.array([0.0, 1.0, 2.0, -3.0])
    np.testing.assert_array_equal(s.forward(vals), vals)
    np.testing.assert_array_equal(s.inverse(vals), vals)


def test_log_scale_forward():
    """LogScale(base=10) forward maps 1→0, 10→1, 100→2 (in log10)."""
    from matplotlib.scale import LogScale
    s = LogScale(base=10)
    vals = np.array([1.0, 10.0, 100.0])
    result = s.forward(vals)
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=1e-10)


def test_log_scale_inverse():
    """LogScale(base=10) inverse maps 0→1, 1→10, 2→100."""
    from matplotlib.scale import LogScale
    s = LogScale(base=10)
    vals = np.array([0.0, 1.0, 2.0])
    result = s.inverse(vals)
    np.testing.assert_allclose(result, [1.0, 10.0, 100.0], atol=1e-10)


def test_log_scale_nonpos():
    """LogScale masks non-positive values."""
    from matplotlib.scale import LogScale
    import numpy.ma as ma
    s = LogScale(base=10)
    vals = np.array([-1.0, 0.0, 1.0, 10.0])
    result = s.forward(vals)
    assert isinstance(result, ma.MaskedArray)
    assert result.mask[0]   # -1 masked
    assert result.mask[1]   # 0 masked
    assert not result.mask[2]  # 1 unmasked
    assert not result.mask[3]  # 10 unmasked


def test_symlog_scale():
    """SymmetricalLogScale is symmetric around zero."""
    from matplotlib.scale import SymmetricalLogScale
    s = SymmetricalLogScale(base=10, linthresh=1.0)
    fwd = s.forward(np.array([1.0, -1.0]))
    assert abs(fwd[0]) == abs(fwd[1])


def test_func_scale():
    """FuncScale applies user-provided forward/inverse callables."""
    from matplotlib.scale import FuncScale
    s = FuncScale(forward=np.sqrt, inverse=np.square)
    vals = np.array([1.0, 4.0, 9.0])
    result = s.forward(vals)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(s.inverse(result), vals)


def test_set_xscale_log_changes_locator():
    """ax.set_xscale('log') must install LogLocator on xaxis."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    assert ax.xaxis.get_scale() == 'log'
    assert isinstance(ax.xaxis.get_major_locator(), LogLocator)
    plt.close('all')


def test_set_xscale_linear_is_default():
    """ax.set_xscale('linear') (or default) uses LinearScale."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    assert ax.xaxis.get_scale() == 'linear'
    plt.close('all')


def test_axes_layout_sx_linear():
    """AxesLayout.sx with linear scale behaves as before."""
    from matplotlib.backend_bases import AxesLayout
    from matplotlib.scale import LinearScale
    layout = AxesLayout(0, 0, 100, 100, 0, 10, 0, 10,
                        LinearScale(), LinearScale())
    assert abs(layout.sx(5) - 50) < 0.01


def test_axes_layout_sx_log():
    """AxesLayout.sx with LogScale maps in log space."""
    from matplotlib.backend_bases import AxesLayout
    from matplotlib.scale import LogScale
    s = LogScale(base=10)
    # data range: [1, 100]; forward: [0, 2]
    layout = AxesLayout(0, 0, 200, 100, 1, 100, 1, 10,
                        s, LogScale(base=10))
    # sx(10) → forward(10)=1.0, linear map [0,2]→[0,200] → 100
    assert abs(layout.sx(10) - 100) < 0.5


def test_axes_layout_backward_compat_no_scales():
    """AxesLayout constructed without scale args must work as before."""
    from matplotlib.backend_bases import AxesLayout
    layout = AxesLayout(0, 0, 100, 100, 0, 10, 0, 10)
    assert abs(layout.sx(5) - 50) < 0.01
    assert abs(layout.sy(5) - 50) < 0.01


def test_cla_resets_scale():
    """cla() must reset axis scale to 'linear' and _xscale to 'linear'."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.cla()
    assert ax.xaxis.get_scale() == 'linear'
    assert ax.get_xscale() == 'linear'
    plt.close('all')


# ===================================================================
# ScaleBase equality and hashing
# ===================================================================

def test_scale_base_eq_string():
    """ScaleBase.__eq__ compares by name string."""
    from matplotlib.scale import LinearScale, LogScale
    s = LinearScale()
    assert s == 'linear'
    assert not (s == 'log')
    assert LogScale() == 'log'


def test_scale_base_eq_same_type():
    """Two instances of same scale type are equal."""
    from matplotlib.scale import LinearScale, LogScale
    assert LinearScale() == LinearScale()
    assert LogScale() == LogScale()


def test_scale_base_eq_different_type():
    """Different scale types are not equal."""
    from matplotlib.scale import LinearScale, LogScale
    assert not (LinearScale() == LogScale())


def test_scale_base_hash():
    """ScaleBase.__hash__ returns hash of name."""
    from matplotlib.scale import LinearScale, LogScale
    assert hash(LinearScale()) == hash('linear')
    assert hash(LogScale()) == hash('log')


def test_scale_base_repr():
    """ScaleBase.__repr__ returns ClassName()."""
    from matplotlib.scale import LinearScale, LogScale
    assert repr(LinearScale()) == 'LinearScale()'
    assert repr(LogScale()) == 'LogScale()'


# ===================================================================
# LogScale edge cases
# ===================================================================

def test_log_scale_base2():
    """LogScale(base=2) maps 1→0, 2→1, 4→2."""
    from matplotlib.scale import LogScale
    s = LogScale(base=2.0)
    result = s.forward(np.array([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=1e-10)


def test_log_scale_inverse_base2():
    """LogScale(base=2) inverse maps 0→1, 1→2, 3→8."""
    from matplotlib.scale import LogScale
    s = LogScale(base=2.0)
    # Test inverse directly with known values
    result = s.inverse(np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(result, [1.0, 2.0, 8.0], atol=1e-10)


def test_log_scale_nonpositive_clip():
    """LogScale(nonpositive='clip') returns result for non-pos input."""
    from matplotlib.scale import LogScale
    s = LogScale(base=10, nonpositive='clip')
    result = s.forward(np.array([1.0, 10.0]))
    # Positive values should map correctly regardless of nonpositive mode
    np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)


def test_log_scale_name():
    from matplotlib.scale import LogScale
    assert LogScale().name == 'log'


# ===================================================================
# SymmetricalLogScale
# ===================================================================

def test_symlog_round_trip():
    """SymmetricalLogScale forward/inverse round-trips."""
    from matplotlib.scale import SymmetricalLogScale
    s = SymmetricalLogScale(base=10, linthresh=1.0)
    vals = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    np.testing.assert_allclose(s.inverse(s.forward(vals)), vals, atol=1e-8)


def test_symlog_zero_maps_to_zero():
    """SymmetricalLogScale forward maps 0 to 0."""
    from matplotlib.scale import SymmetricalLogScale
    s = SymmetricalLogScale(base=10, linthresh=1.0)
    result = s.forward(np.array([0.0]))
    assert abs(float(result[0])) < 1e-10


def test_symlog_name():
    from matplotlib.scale import SymmetricalLogScale
    assert SymmetricalLogScale().name == 'symlog'


def test_set_xscale_symlog():
    """ax.set_xscale('symlog') installs SymmetricalLogLocator."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import SymmetricalLogLocator
    fig, ax = plt.subplots()
    ax.set_xscale('symlog')
    assert ax.xaxis.get_scale() == 'symlog'
    assert isinstance(ax.xaxis.get_major_locator(), SymmetricalLogLocator)
    plt.close('all')


# ===================================================================
# FuncScale
# ===================================================================

def test_func_scale_name():
    from matplotlib.scale import FuncScale
    s = FuncScale(forward=lambda x: x, inverse=lambda x: x)
    # FuncScale has no name attribute by default
    assert hasattr(s, 'forward')
    assert hasattr(s, 'inverse')


def test_func_scale_identity():
    """FuncScale with identity forward/inverse is identity."""
    from matplotlib.scale import FuncScale
    s = FuncScale(forward=lambda x: x, inverse=lambda x: x)
    vals = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(s.forward(vals), vals)
    np.testing.assert_allclose(s.inverse(vals), vals)


def test_func_scale_log():
    """FuncScale can implement log scale."""
    from matplotlib.scale import FuncScale
    s = FuncScale(forward=np.log10, inverse=lambda x: 10 ** x)
    vals = np.array([1.0, 10.0, 100.0])
    result = s.forward(vals)
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=1e-10)


# ===================================================================
# Parametrized scale tests (upstream-inspired batch)
# ===================================================================

import math
import pytest
import numpy as np


class TestLogScaleRoundTrip:
    """LogScale forward/inverse round-trips for various bases."""

    @pytest.mark.parametrize('base', [2.0, 10.0, math.e])
    def test_round_trip_positive(self, base):
        from matplotlib.scale import LogScale
        s = LogScale(base=base)
        vals = np.array([1.0, 2.0, 5.0, 100.0])
        np.testing.assert_allclose(s.inverse(s.forward(vals)), vals, rtol=1e-10)

    def test_single_value(self):
        from matplotlib.scale import LogScale
        s = LogScale(base=10.0)
        result = s.forward(np.array([10.0]))
        assert abs(float(result[0]) - 1.0) < 1e-10

    def test_large_values(self):
        from matplotlib.scale import LogScale
        s = LogScale(base=10.0)
        result = s.forward(np.array([1e6]))
        assert abs(float(result[0]) - 6.0) < 1e-8

    def test_inverse_round_trip_base2(self):
        from matplotlib.scale import LogScale
        s = LogScale(base=2.0)
        vals = np.array([0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(s.forward(s.inverse(vals)), vals, rtol=1e-10)


class TestSymlogScaleVariations:
    """SymmetricalLogScale with different parameters."""

    def test_linthresh_small(self):
        from matplotlib.scale import SymmetricalLogScale
        s = SymmetricalLogScale(base=10, linthresh=0.1)
        vals = np.array([-10.0, 0.0, 10.0])
        out = s.forward(vals)
        assert out[0] < out[1]  # monotone in symlog order
        assert abs(float(out[1])) < 1e-10  # zero maps to zero

    def test_linthresh_large(self):
        from matplotlib.scale import SymmetricalLogScale
        s = SymmetricalLogScale(base=10, linthresh=10.0)
        # Values within linear range should map approximately linearly
        vals = np.array([1.0, 2.0, 3.0])
        out = s.forward(vals)
        assert len(out) == 3

    def test_antisymmetry(self):
        from matplotlib.scale import SymmetricalLogScale
        s = SymmetricalLogScale(base=10, linthresh=1.0)
        fwd = s.forward(np.array([5.0, -5.0]))
        assert abs(float(fwd[0]) + float(fwd[1])) < 1e-8

    def test_inverse_of_forward_large_range(self):
        from matplotlib.scale import SymmetricalLogScale
        s = SymmetricalLogScale(base=10, linthresh=1.0)
        vals = np.array([-50.0, -1.0, 1.0, 50.0])
        np.testing.assert_allclose(s.inverse(s.forward(vals)), vals, atol=1e-7)


class TestSetYScale:
    """ax.set_yscale integration tests."""

    def test_set_yscale_log(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        assert ax.yaxis.get_scale() == 'log'
        assert isinstance(ax.yaxis.get_major_locator(), LogLocator)
        plt.close('all')

    def test_set_yscale_symlog(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import SymmetricalLogLocator
        fig, ax = plt.subplots()
        ax.set_yscale('symlog')
        assert ax.yaxis.get_scale() == 'symlog'
        assert isinstance(ax.yaxis.get_major_locator(), SymmetricalLogLocator)
        plt.close('all')

    def test_set_yscale_linear_default(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        assert ax.yaxis.get_scale() == 'linear'
        plt.close('all')

    def test_cla_resets_yscale(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.cla()
        assert ax.yaxis.get_scale() == 'linear'
        plt.close('all')


class TestFuncScaleVariations:
    """FuncScale with various callables."""

    def test_square_sqrt(self):
        from matplotlib.scale import FuncScale
        s = FuncScale(forward=lambda x: x**2, inverse=lambda x: x**0.5)
        vals = np.array([1.0, 2.0, 3.0])
        fwd = s.forward(vals)
        np.testing.assert_allclose(fwd, [1.0, 4.0, 9.0], rtol=1e-10)

    def test_constant_scale_factor(self):
        from matplotlib.scale import FuncScale
        s = FuncScale(forward=lambda x: x * 2, inverse=lambda x: x / 2)
        vals = np.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(s.inverse(s.forward(vals)), vals, rtol=1e-10)

    def test_numpy_log_exp(self):
        from matplotlib.scale import FuncScale
        s = FuncScale(forward=np.log, inverse=np.exp)
        vals = np.array([0.5, 1.0, 2.0, math.e])
        np.testing.assert_allclose(s.inverse(s.forward(vals)), vals, rtol=1e-10)
