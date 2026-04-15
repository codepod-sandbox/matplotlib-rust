"""Smoke tests for the Rust matplotlib._image extension (Phase 3A).

These tests pin the contract that `matplotlib.image` and
`matplotlib.colors` rely on: the filter constants exist, resample()
runs without raising for the NEAREST and BILINEAR paths, identity
transforms return the input unchanged, and RGBA alpha scaling hits
only the alpha channel.
"""

import numpy as np

from matplotlib import _image
from matplotlib.transforms import Affine2D


def test_interpolation_constants_present():
    # All 17 constants used by matplotlib.image._interpd_ must exist.
    for name in (
        "NEAREST", "BILINEAR", "BICUBIC", "SPLINE16", "SPLINE36",
        "HANNING", "HAMMING", "HERMITE", "KAISER", "QUADRIC", "CATROM",
        "GAUSSIAN", "BESSEL", "MITCHELL", "SINC", "LANCZOS", "BLACKMAN",
    ):
        assert hasattr(_image, name), f"_image.{name} missing"
    assert _image.NEAREST == 0
    assert _image.BILINEAR == 1
    assert _image.LANCZOS == 15


def test_resample_identity_nearest_f64():
    inp = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = np.zeros((4, 4), dtype=np.float64)
    _image.resample(inp, out, Affine2D(), _image.NEAREST)
    np.testing.assert_array_equal(out, inp)


def test_resample_identity_bilinear_f64():
    inp = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = np.zeros((4, 4), dtype=np.float64)
    _image.resample(inp, out, Affine2D(), _image.BILINEAR)
    np.testing.assert_allclose(out, inp)


def test_resample_identity_u8():
    inp = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = np.zeros((4, 4), dtype=np.uint8)
    _image.resample(inp, out, Affine2D(), _image.NEAREST)
    np.testing.assert_array_equal(out, inp)


def test_resample_rgba_alpha_scales_only_alpha():
    # alpha=0.5 must halve the A channel and leave RGB alone.
    rgba = np.full((2, 2, 4), 200, dtype=np.uint8)
    out = np.zeros_like(rgba)
    _image.resample(rgba, out, Affine2D(), _image.NEAREST, alpha=0.5)
    assert out[0, 0, 0] == 200  # R
    assert out[0, 0, 1] == 200  # G
    assert out[0, 0, 2] == 200  # B
    assert out[0, 0, 3] == 100  # A halved


def test_resample_out_of_bounds_yields_zero():
    # Translate the whole input off-screen — every output pixel should
    # fall outside the source and read 0.
    inp = np.full((4, 4), 42.0)
    out = np.zeros((4, 4))
    trans = Affine2D().translate(100, 100)
    _image.resample(inp, out, trans, _image.NEAREST)
    assert np.all(out == 0)


def test_resample_accepts_all_filter_constants():
    # Every filter constant should at least run without raising on an
    # identity transform — even if the windowed filters (3A.3) haven't
    # landed yet, dispatch must not throw.
    inp = np.ones((4, 4), dtype=np.float64)
    out = np.zeros((4, 4), dtype=np.float64)
    for name in (
        "NEAREST", "BILINEAR", "BICUBIC", "SPLINE16", "SPLINE36",
        "HANNING", "HAMMING", "HERMITE", "KAISER", "QUADRIC", "CATROM",
        "GAUSSIAN", "BESSEL", "MITCHELL", "SINC", "LANCZOS", "BLACKMAN",
    ):
        interp = getattr(_image, name)
        out[:] = 0
        _image.resample(inp, out, Affine2D(), interp)
