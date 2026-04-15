"""Smoke tests for the Rust matplotlib._image extension (Phase 3A).

These tests pin the contract that `matplotlib.image` and
`matplotlib.colors` rely on: the filter constants exist, resample()
runs without raising for all filter paths, identity transforms return
the input unchanged, RGBA alpha scaling hits only the alpha channel,
and 2D scalar inputs are not incorrectly scaled by the alpha argument.
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
    # Every filter constant must run without raising on an identity
    # transform and return something non-trivially non-zero for a uniform
    # input array (guards against silent fall-through to NEAREST on all
    # filters, which would still "not raise" but give wrong results for
    # the antialiased path that actively chooses HANNING).
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
        # A uniform-1 input through any filter must produce non-zero output
        # on interior pixels — catches the case where a filter silently
        # returns all zeros (e.g. unimplemented kernel).
        assert out[1, 1] > 0, f"{name} filter returned 0 for interior pixel"


def test_resample_2d_alpha_not_applied():
    # The alpha kwarg must NOT scale 2D scalar output. Alpha composition
    # for scalar/mask planes is done by the Python image pipeline after
    # this call (image.py:480-525). Guard against the regression where
    # the Rust 2D path multiplied the sample by alpha directly.
    inp = np.full((4, 4), 100.0)
    out = np.zeros((4, 4))
    _image.resample(inp, out, Affine2D(), _image.NEAREST, alpha=0.5)
    np.testing.assert_array_equal(
        out, inp,
        err_msg="2D resample must not apply alpha to scalar output",
    )


def test_resample_hanning_not_nearest():
    # HANNING and NEAREST are identical on an identity transform because
    # HANNING is zero at all non-zero integer offsets (same taps contribute).
    # Use 2x upsampling (4x4 → 8x8) so output pixels land at half-integer
    # source coords — that is where the kernels diverge.
    inp = np.zeros((4, 4), dtype=np.float64)
    inp[::2, ::2] = 1.0
    inp[1::2, 1::2] = 1.0
    out_near = np.zeros((8, 8), dtype=np.float64)
    out_hann = np.zeros((8, 8), dtype=np.float64)
    # scale(2): forward transform maps input → output at 2x.
    # The inverse (output→input) scales by 0.5, placing output pixel
    # centres at fractional input coords where the kernels differ.
    trans = Affine2D().scale(2.0)
    _image.resample(inp, out_near, trans, _image.NEAREST)
    _image.resample(inp, out_hann, trans, _image.HANNING)
    # HANNING smooths at sub-pixel boundaries → lower variance than the
    # hard block edges NEAREST produces on the upsampled checkerboard.
    assert out_hann.var() < out_near.var(), (
        "HANNING output has same variance as NEAREST on 2x upsampled "
        "checkerboard — likely falling back to nearest-neighbor instead "
        "of applying the Hann window kernel"
    )


def test_resample_lanczos_identity():
    # LANCZOS on an identity transform must reproduce the input exactly
    # (within f64 precision). Lanczos is an interpolating kernel:
    # sinc(n) = 0 for nonzero integers, so only the tap at distance 0
    # contributes and normalization gives the input pixel unchanged.
    inp = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = np.zeros_like(inp)
    _image.resample(inp, out, Affine2D(), _image.LANCZOS, radius=3.0)
    np.testing.assert_allclose(out, inp, atol=1e-10)
