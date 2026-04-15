"""Smoke tests for the Rust-backed matplotlib.backends._backend_agg.

These verify the low-level RendererAgg extension directly, independent
of the Python wrapper in backends/backend_agg.py. They assert the C
extension has been loaded (not the fallback .py stub), that every
method the wrapper touches is callable, and that the buffer protocol
returns an (H, W, 4) uint8 ndarray after drawing.

Phase 1A pass criterion: all tests in this file pass.
"""

import numpy as np
import pytest

from matplotlib.backends import _backend_agg


def test_extension_loaded():
    """The Rust .so should win over the Python stub fallback."""
    # If the stub .py is loaded, __file__ ends with .py. If the Rust
    # extension is loaded, it ends with .so (Unix) or .pyd (Windows).
    path = _backend_agg.__file__
    assert path.endswith((".so", ".pyd")), (
        f"Expected Rust extension, got {path}. Run `make build-ext` first."
    )


def test_renderer_construction():
    r = _backend_agg.RendererAgg(100, 80, 72.0)
    assert r.width == 100
    assert r.height == 80
    assert r.dpi == 72.0


def test_get_hinting_flag():
    assert _backend_agg.get_hinting_flag() == 0


def test_buffer_rgba_shape_and_dtype():
    r = _backend_agg.RendererAgg(50, 40, 72.0)
    buf = r.buffer_rgba()
    assert buf.shape == (40, 50, 4)
    assert buf.dtype == np.uint8


def test_np_asarray_returns_same_shape():
    r = _backend_agg.RendererAgg(32, 24, 72.0)
    arr = np.asarray(r)
    assert arr.shape == (24, 32, 4)
    assert arr.dtype == np.uint8


def test_clear_zeros_buffer():
    r = _backend_agg.RendererAgg(10, 10, 72.0)
    r.clear()
    arr = np.asarray(r)
    assert (arr == 0).all()


def test_tostring_rgb_size():
    r = _backend_agg.RendererAgg(10, 8, 72.0)
    data = r.tostring_rgb()
    assert isinstance(data, bytes)
    assert len(data) == 10 * 8 * 3


def test_tostring_argb_size():
    r = _backend_agg.RendererAgg(10, 8, 72.0)
    data = r.tostring_argb()
    assert isinstance(data, bytes)
    assert len(data) == 10 * 8 * 4


def test_draw_text_image_does_not_raise_on_ndarray():
    """With a 2D uint8 bitmap (TeX path), draw_text_image must not crash."""
    r = _backend_agg.RendererAgg(50, 50, 72.0)
    bitmap = np.zeros((5, 5), dtype=np.uint8)

    class MinimalGC:
        def get_rgb(self):
            return (0.0, 0.0, 0.0, 1.0)

        def get_alpha(self):
            return 1.0

        def get_linewidth(self):
            return 1.0

        def get_clip_rectangle(self):
            return None

    r.draw_text_image(bitmap, 10, 10, 0.0, MinimalGC())


def test_region_lifecycle_noop():
    """copy_from_bbox / restore_region should be callable without error."""
    r = _backend_agg.RendererAgg(20, 20, 72.0)
    region = r.copy_from_bbox(None)
    # Single-arg shape
    r.restore_region(region)
    # 7-arg shape
    r.restore_region(region, 0, 0, 10, 10, 0, 0)


def test_filter_stack_noop():
    r = _backend_agg.RendererAgg(20, 20, 72.0)
    r.start_filter()
    r.stop_filter(None)
    r.start_rasterizing()
    r.stop_rasterizing()


def test_draw_path_produces_pixels():
    """A filled red rectangle should put red pixels in the buffer."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import io

    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, color='red'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    arr = np.asarray(img)
    red = ((arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)).sum()
    assert red > 100, f"Expected red rectangle pixels, got {red}"
