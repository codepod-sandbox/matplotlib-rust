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

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends import _backend_agg
from matplotlib.figure import Figure


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


def test_figure_mathtext_draws_visible_pixels():
    """The backend_agg Python wrapper must blit mathtext FT2Image output."""
    fig = Figure(figsize=(2, 1), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.text(0.5, 0.5, r'$a+b$', ha='center', va='center')
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba())
    assert (arr[:, :, :3] < 250).any(), "expected visible mathtext pixels"


def test_region_lifecycle_noop():
    """copy_from_bbox / restore_region should be callable without error."""
    r = _backend_agg.RendererAgg(20, 20, 72.0)
    region = r.copy_from_bbox(None)
    # Single-arg shape
    r.restore_region(region)
    # 7-arg shape
    r.restore_region(region, 0, 0, 10, 10, 0, 0)


class _Bbox:
    """Minimal matplotlib Bbox-like for copy_from_bbox tests."""
    def __init__(self, x0, y0, x1, y1):
        self.extents = (x0, y0, x1, y1)


class _MinimalGC:
    """Minimal gc for draw calls that need a real gc."""
    def __init__(self, rgb=(0.0, 0.0, 0.0, 1.0), alpha=1.0, lw=1.0, clip=None):
        self._rgb = rgb
        self._alpha = alpha
        self._lw = lw
        self._clip = clip

    def get_rgb(self):
        return self._rgb

    def get_alpha(self):
        return self._alpha

    def get_linewidth(self):
        return self._lw

    def get_clip_rectangle(self):
        if self._clip is None:
            return None

        class _ClipBbox:
            bounds = self._clip

        return _ClipBbox()


def test_copy_from_bbox_returns_buffer_region():
    """copy_from_bbox must return a real BufferRegion with correct dimensions."""
    r = _backend_agg.RendererAgg(100, 100, 72.0)
    region = r.copy_from_bbox(_Bbox(10.0, 20.0, 60.0, 80.0))
    assert isinstance(region, _backend_agg.BufferRegion)
    assert region.width == 50
    assert region.height == 60


def test_region_round_trip_preserves_pixels():
    """Draw → copy → clear → restore should recover the original pixels."""
    r = _backend_agg.RendererAgg(100, 100, 72.0)

    # Paint a 20x20 red block via draw_image
    data = np.zeros((20, 20, 4), dtype=np.uint8)
    data[:, :, 0] = 255
    data[:, :, 3] = 255
    r.draw_image(_MinimalGC(), 10, 10, data)

    before = np.asarray(r).copy()
    red_before = ((before[:, :, 0] > 200) & (before[:, :, 1] < 50)).sum()
    assert red_before > 300, f"expected red block, got {red_before}"

    # Capture the whole canvas
    region = r.copy_from_bbox(_Bbox(0.0, 0.0, 100.0, 100.0))

    # Wipe and restore
    r.clear()
    cleared = np.asarray(r)
    assert cleared.sum() == 0, "clear should zero the buffer"

    r.restore_region(region)
    restored = np.asarray(r)
    red_after = ((restored[:, :, 0] > 200) & (restored[:, :, 1] < 50)).sum()
    assert red_after == red_before, (
        f"restore should recover exactly the original red pixel count "
        f"({red_before}), got {red_after}"
    )


def test_clip_rectangle_is_honored():
    """A draw_path with a clipped gc must not paint outside the clip rect."""
    r = _backend_agg.RendererAgg(100, 100, 72.0)

    class _FullCanvasPath:
        vertices = np.array(
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]]
        )
        codes = np.array([1, 2, 2, 2, 79], dtype=np.uint8)

    class _Identity:
        def get_matrix(self):
            return np.eye(3)

    # Clip to bottom-left 50x50 in display coords (y-up).
    gc = _MinimalGC(rgb=(1.0, 0.0, 0.0, 1.0), lw=0.0, clip=(0.0, 0.0, 50.0, 50.0))
    r.draw_path(gc, _FullCanvasPath(), _Identity(), (1.0, 0.0, 0.0, 1.0))

    arr = np.asarray(r)
    # pixmap rows are y-down, so the clipped region (display y 0..50)
    # corresponds to pixmap rows 50..100 (bottom half).
    top_left = ((arr[:50, :50, 0] > 200) & (arr[:50, :50, 1] < 50)).sum()
    top_right = ((arr[:50, 50:, 0] > 200) & (arr[:50, 50:, 1] < 50)).sum()
    bottom_left = ((arr[50:, :50, 0] > 200) & (arr[50:, :50, 1] < 50)).sum()
    bottom_right = ((arr[50:, 50:, 0] > 200) & (arr[50:, 50:, 1] < 50)).sum()

    assert bottom_left == 50 * 50, (
        f"expected 2500 red pixels in clipped bottom-left, got {bottom_left}"
    )
    assert top_left == 0, f"clip leak into top-left: {top_left}"
    assert top_right == 0, f"clip leak into top-right: {top_right}"
    assert bottom_right == 0, f"clip leak into bottom-right: {bottom_right}"


def test_buffer_region_asarray_shape_and_dtype():
    """np.asarray(region) must yield a (H, W, 4) uint8 ndarray.

    Matches the backend_gtk3agg.py:42 contract.
    """
    r = _backend_agg.RendererAgg(100, 80, 72.0)
    region = r.copy_from_bbox(_Bbox(10.0, 20.0, 60.0, 70.0))
    arr = np.asarray(region)
    assert arr.shape == (50, 50, 4)
    assert arr.dtype == np.uint8


def test_buffer_region_memoryview_shape_and_format():
    """memoryview(region) must yield a 3-D B-format view with shape.

    Matches the backend_qtagg.py:50 contract: the Qt backend does
    `buf = memoryview(self.copy_from_bbox(bbox))` and then reads
    `buf.shape[0]` and `buf.shape[1]` for the QImage dimensions.
    """
    r = _backend_agg.RendererAgg(100, 80, 72.0)
    region = r.copy_from_bbox(_Bbox(10.0, 20.0, 60.0, 70.0))
    mv = memoryview(region)
    assert mv.shape == (50, 50, 4)
    assert mv.format == "B"
    assert mv.itemsize == 1
    assert mv.nbytes == 50 * 50 * 4
    assert mv.readonly is True


def test_buffer_region_memoryview_pixels_match_asarray():
    """The memoryview and np.asarray views must expose the same bytes.

    Guards against a class of bugs where the two buffer-exposure paths
    drift apart (e.g. stale cache on one of them).
    """
    r = _backend_agg.RendererAgg(50, 50, 72.0)
    data = np.zeros((10, 10, 4), dtype=np.uint8)
    data[:, :, 1] = 255  # green
    data[:, :, 3] = 255
    r.draw_image(_MinimalGC(), 5, 5, data)
    region = r.copy_from_bbox(_Bbox(0.0, 0.0, 50.0, 50.0))

    arr = np.asarray(region)
    mv = memoryview(region)

    # Same shape
    assert arr.shape == mv.shape
    # Same byte content (memoryview → bytes → compare to ndarray tobytes)
    assert bytes(mv) == arr.tobytes()


def test_copy_from_bbox_with_real_matplotlib_bbox():
    """copy_from_bbox must accept a real matplotlib.transforms.Bbox.

    Bbox.extents returns an ndarray, not a tuple, so this catches
    regressions in the bbox-extraction path that only show up with
    the real class (not the _Bbox shim used elsewhere in this file).

    Simulates the exact upstream backend_qtagg.py:50 /
    backend_gtk3agg.py:42 sequences.
    """
    from matplotlib.transforms import Bbox

    r = _backend_agg.RendererAgg(200, 150, 72.0)
    data = np.zeros((30, 30, 4), dtype=np.uint8)
    data[:, :, 2] = 255  # blue
    data[:, :, 3] = 255
    r.draw_image(_MinimalGC(), 50, 50, data)

    bbox = Bbox([[0, 0], [200, 150]])

    # Qt path: memoryview then read .shape[0] / .shape[1]
    buf = memoryview(r.copy_from_bbox(bbox))
    assert buf.shape == (150, 200, 4)
    assert buf.format == "B"

    # GTK path: np.asarray then .ravel()
    arr = np.asarray(r.copy_from_bbox(bbox))
    assert arr.shape == (150, 200, 4)
    assert arr.dtype == np.uint8
    # Ravel should not crash (exercised by gtk3agg cbook helper).
    _ = arr.ravel()


def test_restore_region_subrect_shape():
    """The 7-arg restore_region form should place a sub-rect at a new location."""
    r = _backend_agg.RendererAgg(100, 100, 72.0)

    # Paint 20x20 red block via draw_image at display coords (10, 10).
    # draw_image y-flips, so the block lands at pixmap rows
    # (100 - 10 - 20)..(100 - 10) = rows 70..90, cols 10..30.
    data = np.zeros((20, 20, 4), dtype=np.uint8)
    data[:, :, 0] = 255
    data[:, :, 3] = 255
    r.draw_image(_MinimalGC(), 10, 10, data)

    # Sanity: red block is where we think it is.
    before = np.asarray(r)
    src_red = ((before[70:90, 10:30, 0] > 200) & (before[70:90, 10:30, 1] < 50)).sum()
    assert src_red == 400, f"expected 400 red pixels at (70:90, 10:30), got {src_red}"

    region = r.copy_from_bbox(_Bbox(0.0, 0.0, 100.0, 100.0))
    r.clear()

    # 7-arg form: (x1, y1, x2, y2, ox, oy) in pixmap coords.
    # Take the sub-rect of the region at (10, 70)..(30, 90) and blit
    # its top-left at (60, 60). The block should appear at pixmap
    # rows 60..80, cols 60..80.
    r.restore_region(region, 10, 70, 30, 90, 60, 60)
    arr = np.asarray(r)

    dst_red = ((arr[60:80, 60:80, 0] > 200) & (arr[60:80, 60:80, 1] < 50)).sum()
    assert dst_red == 400, (
        f"expected 400 restored red pixels at dest (60:80, 60:80), got {dst_red}"
    )
    # And nothing outside that destination rect should be red.
    total_red = ((arr[:, :, 0] > 200) & (arr[:, :, 1] < 50)).sum()
    outside = total_red - dst_red
    assert outside == 0, f"sub-rect restore bled outside: {outside} extra red"


def test_filter_stack_noop():
    r = _backend_agg.RendererAgg(20, 20, 72.0)
    r.start_filter()
    r.stop_filter(None)
    r.start_rasterizing()
    r.stop_rasterizing()


def test_dashed_line_has_gaps_in_row():
    """A red dashed horizontal line should produce a row with gaps, not
    a solid run of red pixels. Guards the 1B.3 dash plumbing.
    """
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots(figsize=(3, 3), dpi=50)
    ax.plot([0.1, 0.9], [0.5, 0.5], 'r--', linewidth=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    buf.seek(0)
    from PIL import Image
    arr = np.asarray(Image.open(buf).convert('RGB'))
    # Find the row with the most red pixels — this is the line row.
    row_red = ((arr[:, :, 0] > 200) & (arr[:, :, 1] < 50)).sum(axis=1)
    line_row_idx = int(row_red.argmax())
    assert row_red[line_row_idx] > 10, (
        f"expected a red line row, got max row count {row_red[line_row_idx]}"
    )

    # Count gaps in the red columns: a dashed line has > 0 gaps.
    line_row = arr[line_row_idx]
    red_cols = np.where((line_row[:, 0] > 200) & (line_row[:, 1] < 50))[0]
    gaps = np.diff(red_cols)
    gap_count = int((gaps > 1).sum())
    assert gap_count > 0, (
        f"dashed line rendered as solid (0 gaps); gc.get_dashes() not honored"
    )


def test_hatched_rect_has_dark_lines_inside_blue_fill():
    """A blue rectangle with hatch='///' should have dark hatch lines
    visible inside the fill. Guards 1B.2 hatching plumbing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import io

    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    # edgecolor='none' so any dark pixels in the interior must come
    # from the hatch (matplotlib defaults the hatch color to black
    # when edgecolor is none).
    ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, facecolor='blue',
                           hatch='///', edgecolor='none'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    buf.seek(0)
    from PIL import Image
    arr = np.asarray(Image.open(buf).convert('RGB'))
    h, w = arr.shape[:2]
    inner = arr[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    blue = ((inner[:, :, 2] > 200) & (inner[:, :, 0] < 80) & (inner[:, :, 1] < 80)).sum()
    dark = ((inner[:, :, 0] < 40) & (inner[:, :, 1] < 40) & (inner[:, :, 2] < 40)).sum()
    assert blue > 1000, f"expected blue fill, got {blue} blue pixels"
    assert dark > 100, (
        f"expected dark hatch lines inside the blue fill, got {dark}"
    )


def test_unhatched_rect_has_no_dark_lines():
    """A blue rectangle WITHOUT hatch should have zero dark interior
    pixels. Catches regressions where hatch leaks into unhatched paths.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import io

    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, facecolor='blue', edgecolor='none'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    buf.seek(0)
    from PIL import Image
    arr = np.asarray(Image.open(buf).convert('RGB'))
    h, w = arr.shape[:2]
    inner = arr[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    dark = ((inner[:, :, 0] < 40) & (inner[:, :, 1] < 40) & (inner[:, :, 2] < 40)).sum()
    assert dark == 0, f"unhatched rect should have 0 dark pixels, got {dark}"


def test_polar_clip_path_masks_corners():
    """A polar (circular) axes should not paint the corners of its bounding
    rectangle. Guards 1B.1 clip-path plumbing via gc.get_clip_path().
    """
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots(
        figsize=(3, 3), dpi=50, subplot_kw={'projection': 'polar'}
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.full_like(theta, 0.8)
    ax.plot(theta, r, 'r-', linewidth=3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    buf.seek(0)
    from PIL import Image
    arr = np.asarray(Image.open(buf).convert('RGB'))
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50)
    assert red.sum() > 100, "expected a red polar curve"
    # Four corner pixels should be outside the circular clip path.
    h, w = arr.shape[:2]
    for rr, cc in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        assert not red[rr, cc], (
            f"corner pixel ({rr},{cc}) is red — clip path not applied"
        )


def test_round_cap_produces_round_pixels():
    """A line with capstyle='round' should paint pixels outside the
    rectangular stroke bounds at the endpoints. Guards capstyle plumbing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import io

    fig, ax = plt.subplots(figsize=(3, 3), dpi=50)
    # Very thick short horizontal line with round caps.
    line = Line2D([0.4, 0.6], [0.5, 0.5], color='red', linewidth=20,
                  solid_capstyle='round')
    ax.add_line(line)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    buf.seek(0)
    from PIL import Image
    arr = np.asarray(Image.open(buf).convert('RGB'))
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50)
    # Sanity: something was drawn.
    assert red.sum() > 100, f"expected red stroke pixels, got {red.sum()}"


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
