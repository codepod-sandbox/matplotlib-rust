"""Smoke tests for the Rust-backed matplotlib.ft2font.

Phase 2A pass criterion: rasterizing a simple string produces real
glyph pixels, metrics are in 26.6 subpixel units, and the extension
replaces the old Python stub.
"""

import os
import numpy as np
import pytest

from matplotlib import ft2font


def _find_dejavu_sans():
    """Find the bundled DejaVuSans.ttf under mpl-data/fonts/ttf/."""
    import matplotlib
    base = os.path.dirname(os.path.abspath(matplotlib.__file__))
    p = os.path.join(base, 'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf')
    if not os.path.exists(p):
        pytest.skip(f"DejaVuSans.ttf not found at {p}")
    return p


def test_extension_loaded():
    """The Rust .so should replace the old Python stub entirely."""
    path = ft2font.__file__
    assert path.endswith((".so", ".pyd")), (
        f"Expected Rust extension, got {path}. Run `make build-ext` first."
    )


def test_freetype_version_constants():
    assert hasattr(ft2font, '__freetype_version__')
    assert hasattr(ft2font, '__freetype_build_type__')
    # The Rust extension reports "fontdue" as the build type.
    assert ft2font.__freetype_build_type__ == 'fontdue'


def test_load_flags_exposed():
    """LoadFlags enum-like class must be importable — backend_agg.py:34
    does `from matplotlib.ft2font import LoadFlags`.
    """
    assert hasattr(ft2font, 'LoadFlags')
    assert ft2font.LoadFlags.DEFAULT == 0
    assert ft2font.LoadFlags.FORCE_AUTOHINT == 32


def test_ft2image_constructible():
    """_mathtext.py imports FT2Image and does draw_rect_filled."""
    img = ft2font.FT2Image(10, 8)
    assert img.width == 10
    assert img.height == 8
    img.draw_rect_filled(1, 1, 8, 6)
    arr = np.asarray(img)
    assert arr.shape == (8, 10)
    assert arr.dtype == np.uint8
    # Filled region should have 255 values.
    assert (arr[1:7, 1:9] == 255).all()
    # Outside should be 0.
    assert (arr[0, :] == 0).all()


def test_construct_from_dejavu_sans():
    font = ft2font.FT2Font(_find_dejavu_sans())
    assert font.family_name == 'DejaVu Sans'
    assert font.style_name == 'Book'
    assert font.units_per_EM == 2048
    assert font.num_glyphs > 100  # DejaVu has thousands


def test_render_hello_returns_pixels():
    """The 2A pass criterion: set_text + draw_glyphs_to_bitmap + get_image
    produces a non-zero grayscale bitmap for a real string.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    font.set_text("Hello", 0.0, 0)

    w_26_6, h_26_6 = font.get_width_height()
    # At 12 pt / 72 dpi, "Hello" should be roughly 30 px wide.
    # In 26.6 subpixels: 30 * 64 = 1920 ± slack.
    assert 500 < w_26_6 < 5000, f"unexpected width: {w_26_6}"
    assert 200 < h_26_6 < 3000, f"unexpected height: {h_26_6}"

    font.draw_glyphs_to_bitmap(antialiased=True)
    bitmap = font.get_image()
    assert bitmap.dtype == np.uint8
    assert bitmap.ndim == 2
    assert bitmap.shape[0] > 0 and bitmap.shape[1] > 0
    assert (bitmap > 0).sum() > 30, (
        f"expected rasterized glyph pixels, got {(bitmap > 0).sum()} non-zero "
        f"in bitmap of shape {bitmap.shape}"
    )


def test_render_empty_string_does_not_crash():
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    font.set_text("", 0.0, 0)
    font.draw_glyphs_to_bitmap(antialiased=True)
    bitmap = font.get_image()
    assert bitmap.dtype == np.uint8


def test_metrics_scale_with_size():
    """Doubling the pt size should roughly double the bitmap width."""
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    font.set_text("M", 0.0, 0)
    w_small, _ = font.get_width_height()

    font.set_size(24.0, 72.0)
    font.set_text("M", 0.0, 0)
    w_big, _ = font.get_width_height()

    # Allow 10% tolerance on the ratio.
    ratio = w_big / w_small
    assert 1.7 < ratio < 2.3, f"expected ~2x scaling, got {ratio:.2f}"


def test_clear_resets_state():
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    font.set_text("Hello", 0.0, 0)
    font.draw_glyphs_to_bitmap(antialiased=True)
    assert font.get_image().size > 1

    font.clear()
    w, h = font.get_width_height()
    assert w == 0.0 and h == 0.0


def test_get_char_index_known_ascii():
    """get_char_index should return a non-zero glyph ID for 'A'."""
    font = ft2font.FT2Font(_find_dejavu_sans())
    idx = font.get_char_index(ord('A'))
    assert idx > 0, f"expected non-zero glyph index for 'A', got {idx}"


# ===================================================================
# Phase 2B — get_path via ttf-parser OutlineBuilder
# ===================================================================


def test_load_char_then_get_path_returns_outline():
    """The text2path pipeline calls load_char() then get_path() per
    character. The returned outline should have valid matplotlib codes.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    font.load_char(ord('A'), flags=0)

    verts, codes = font.get_path()
    assert verts.ndim == 2
    assert verts.shape[1] == 2
    assert verts.dtype == np.float64
    assert codes.ndim == 1
    assert codes.dtype == np.uint8
    assert verts.shape[0] == codes.shape[0]
    assert verts.shape[0] > 0, "expected non-empty outline for 'A'"

    # 'A' must contain at least one MOVETO and one LINETO.
    code_set = set(codes.tolist())
    assert 1 in code_set, f"missing MOVETO; codes seen: {code_set}"
    assert 2 in code_set, f"missing LINETO; codes seen: {code_set}"


def test_get_path_changes_per_character():
    """load_char(A) and load_char(B) should yield different outlines."""
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)

    font.load_char(ord('A'), flags=0)
    a_verts, a_codes = font.get_path()

    font.load_char(ord('B'), flags=0)
    b_verts, b_codes = font.get_path()

    assert a_verts.shape != b_verts.shape or not np.array_equal(a_verts, b_verts), (
        "A and B should produce different outlines"
    )


def test_get_path_for_space_is_empty():
    """Whitespace glyphs have no outline."""
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    font.load_char(ord(' '), flags=0)
    verts, codes = font.get_path()
    assert verts.shape[0] == 0
    assert codes.shape[0] == 0


def test_fontmap_returns_self_per_char():
    """_get_fontmap('Hi') should return {'H': self, 'i': self}."""
    font = ft2font.FT2Font(_find_dejavu_sans())
    fm = font._get_fontmap('Hi!')
    assert set(fm.keys()) == {'H', 'i', '!'}
    # All values should be the same FT2Font instance (single-font fallback).
    values = list(fm.values())
    assert all(v is values[0] for v in values)
    assert isinstance(values[0], ft2font.FT2Font)


def test_savefig_svg_contains_path_elements():
    """End-to-end: rendering text via savefig(svg) should produce <path>
    elements from the OG backend_svg text2path pipeline.
    """
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots()
    ax.set_title('SVGTITLE')
    ax.plot([1, 2, 3], [4, 5, 6])
    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue().decode('utf-8')
    assert '<svg' in svg
    # text2path emits each glyph as a <path d="..."> element. With a
    # title, axis labels, and tick labels, expect many paths.
    path_count = svg.count('<path')
    assert path_count > 5, f"expected many <path> elements, got {path_count}"
