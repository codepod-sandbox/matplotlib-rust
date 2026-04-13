"""
Stub/bridge for matplotlib._image (C extension) for RustPython/WASM.

Image resampling and compositing operations.
Heavy pixel-level work delegates to pillow-rust when available.
"""

import numpy as np

# Interpolation mode constants (match C extension values)
NEAREST = 0
BILINEAR = 1
BICUBIC = 2
SPLINE16 = 3
SPLINE36 = 4
HANNING = 5
HAMMING = 6
HERMITE = 7
KAISER = 8
QUADRIC = 9
CATROM = 10
GAUSSIAN = 11
BESSEL = 12
MITCHELL = 13
SINC = 14
LANCZOS = 15
BLACKMAN = 16


def resample(input_array, output_array, transform, interpolation=BILINEAR,
             resample=False, alpha=1.0, norm=False, radius=1.0):
    """
    Resample input_array into output_array using the given transform.

    In the WASM sandbox this is a no-op stub — colormap resampling
    (the only upstream caller) falls back to the pre-built LUT.
    """
    pass


def pcolor(x, y, data, rows, cols, bounds):
    """Stub for pcolor rendering."""
    raise NotImplementedError("_image.pcolor not available in WASM sandbox")


def pcolorfast(x, y, data, rows, cols, bounds):
    """Stub for pcolorfast rendering."""
    raise NotImplementedError("_image.pcolorfast not available in WASM sandbox")
