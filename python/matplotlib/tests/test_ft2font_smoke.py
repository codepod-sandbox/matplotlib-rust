"""Smoke tests for the Rust-backed matplotlib.ft2font.

Phase 2A pass criterion: rasterizing a simple string produces real
glyph pixels, metrics are in 26.6 subpixel units, and the extension
replaces the old Python stub.
"""

import os
import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import ft2font
from matplotlib.mathtext import MathTextParser


def _find_dejavu_sans():
    """Find the bundled DejaVuSans.ttf under mpl-data/fonts/ttf/."""
    import matplotlib
    base = os.path.dirname(os.path.abspath(matplotlib.__file__))
    p = os.path.join(base, 'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf')
    assert os.path.exists(p), f"DejaVuSans.ttf not found at {p}"
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


def test_mathtext_raster_parse_returns_nonempty_image():
    """MathTextParser('agg') should draw glyph pixels into FT2Image."""
    mpl.rcParams['mathtext.fontset'] = 'cm'
    out = MathTextParser('agg').parse(r'$a+b+\dot s$', dpi=100)
    bitmap = np.asarray(out.image)
    assert bitmap.dtype == np.uint8
    assert bitmap.ndim == 2
    assert bitmap.shape[0] > 0 and bitmap.shape[1] > 0
    assert (bitmap > 0).any(), "mathtext raster parse returned a blank image"


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


def test_linear_hori_advance_is_16_16_fixed_point():
    """glyph.linearHoriAdvance is 16.16 fixed-point (divide by 65536
    for pixels), while horiAdvance is 26.6 (divide by 64).
    matplotlib._text_helpers.layout does
        x += glyph.linearHoriAdvance / 65536
    so linearHoriAdvance MUST be in 16.16 units or the pen
    under-advances by 1024× and multi-glyph text layout collapses.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    g = font.load_char(ord('A'), flags=0)
    # Both should agree on the glyph's advance in pixels.
    advance_pxel_from_26_6 = g.horiAdvance / 64
    advance_pxel_from_16_16 = g.linearHoriAdvance / 65536
    assert advance_pxel_from_26_6 > 0
    assert advance_pxel_from_16_16 > 0
    # Within 0.5 px (the 26.6 truncation + 16.16 representation error)
    assert abs(advance_pxel_from_26_6 - advance_pxel_from_16_16) < 0.5, (
        f"advance mismatch: 26.6={advance_pxel_from_26_6:.3f}, "
        f"16.16={advance_pxel_from_16_16:.3f}"
    )


def test_text_helpers_layout_advances_monotonically():
    """End-to-end verification of linearHoriAdvance via the real
    matplotlib._text_helpers.layout() routine, which is what
    backend_svg.py's text2path pipeline uses to position glyphs.
    """
    from matplotlib import _text_helpers
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(100.0, 100.0)
    items = list(_text_helpers.layout('ABC', font))
    assert len(items) == 3
    assert items[0].x == 0.0
    assert items[1].x > items[0].x
    assert items[2].x > items[1].x
    # Advances should be ~roughly 1 em wide at 100pt/100dpi. Much less
    # than that would indicate the 1024× under-advance bug.
    assert items[1].x > 30.0, (
        f"layout advance too small, got {items[1].x:.2f}; "
        f"indicates linearHoriAdvance is in wrong fixed-point units"
    )


def test_get_ps_font_info_is_dict_with_string_keys():
    """get_ps_font_info() must return a dict indexable by string key.
    font_manager.py:418 does `font.get_ps_font_info()["weight"]`, so
    the stubbed tuple form would raise TypeError.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    info = font.get_ps_font_info()
    assert isinstance(info, dict)
    # The keys that font_manager.ttfFontProperty and other callers touch:
    for key in ('weight', 'family_name', 'italic_angle', 'is_fixed_pitch',
                'underline_position', 'underline_thickness'):
        assert key in info, f"missing key {key!r}"
    # weight is a string; .replace(' ', '') is called on it.
    assert isinstance(info['weight'], str)
    info['weight'].replace(' ', '')  # must not raise


# ===================================================================
# Phase 2C — SFNT metadata tables + kerning + charmap
# ===================================================================


def test_os2_table_matches_font_manager_contract():
    """font_manager.ttfFontProperty reads os2['version'] and
    os2['usWeightClass'] to pick the font weight. Both must be present
    and usWeightClass must be in the OS/2-defined range (1..1000).
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    os2 = font.get_sfnt_table("OS/2")
    assert isinstance(os2, dict)
    assert os2["version"] != 0xffff, "0xffff means missing OS/2 table"
    assert 1 <= os2["usWeightClass"] <= 1000
    # DejaVu Sans is Regular (weight ~400).
    assert 300 <= os2["usWeightClass"] <= 500
    # fsSelection must be an int so bitwise ops work.
    assert isinstance(os2["fsSelection"], int)


def test_head_table_has_units_per_em():
    font = ft2font.FT2Font(_find_dejavu_sans())
    head = font.get_sfnt_table("head")
    assert isinstance(head, dict)
    assert head["unitsPerEm"] == font.units_per_EM
    assert head["unitsPerEm"] == 2048  # DejaVu Sans


def test_hhea_table_has_vertical_metrics():
    font = ft2font.FT2Font(_find_dejavu_sans())
    hhea = font.get_sfnt_table("hhea")
    assert isinstance(hhea, dict)
    assert hhea["ascent"] > 0
    assert hhea["descent"] < 0


def test_post_table_has_italic_angle():
    font = ft2font.FT2Font(_find_dejavu_sans())
    post = font.get_sfnt_table("post")
    assert isinstance(post, dict)
    # italicAngle and format are both 16.16 fixed-point (major, minor)
    # tuples per _SfntPostDict. backend_pdf.py:1446 does
    #     post['italicAngle'][1]
    # which would TypeError if this were a float.
    ia = post["italicAngle"]
    assert isinstance(ia, tuple), f"expected tuple, got {type(ia).__name__}"
    assert len(ia) == 2
    assert all(isinstance(x, int) for x in ia)
    # DejaVu Sans is upright → (0, 0)
    assert ia == (0, 0)
    # backend_pdf's [1] index must not raise
    _ = post["italicAngle"][1]

    fmt = post["format"]
    assert isinstance(fmt, tuple) and len(fmt) == 2
    assert all(isinstance(x, int) for x in fmt)

    # post['underlinePosition'] used by font metadata introspection.
    assert post["underlinePosition"] < 0


def test_head_version_and_revision_are_fixed_tuples():
    """_SfntHeadDict specifies version and fontRevision as 16.16
    fixed-point (major, minor) tuples.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    head = font.get_sfnt_table("head")
    for key in ("version", "fontRevision", "created", "modified"):
        val = head[key]
        assert isinstance(val, tuple), f"{key}: expected tuple, got {type(val).__name__}"
        assert len(val) == 2


def test_hhea_maxp_versions_are_fixed_tuples():
    font = ft2font.FT2Font(_find_dejavu_sans())
    hhea = font.get_sfnt_table("hhea")
    assert isinstance(hhea["version"], tuple) and len(hhea["version"]) == 2
    maxp = font.get_sfnt_table("maxp")
    assert isinstance(maxp["version"], tuple) and len(maxp["version"]) == 2
    assert maxp["numGlyphs"] > 100


def test_os2_ulcharrange_is_4_tuple_not_bytes():
    """_SfntOs2Dict specifies ulCharRange as tuple[int, int, int, int]
    (the four 32-bit Unicode-range bitmasks), NOT bytes.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    os2 = font.get_sfnt_table("OS/2")
    ucr = os2["ulCharRange"]
    assert isinstance(ucr, tuple)
    assert len(ucr) == 4
    assert all(isinstance(x, int) for x in ucr)


def test_backend_pdf_italic_angle_index_does_not_raise():
    """Guards the exact line backend_pdf.py:1446 uses:
        post['italicAngle'][1]
    against a future regression that would return a float or
    something else non-subscriptable.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    post = font.get_sfnt_table("post")
    # The exact operation from backend_pdf.py:
    italic_minor = post['italicAngle'][1]
    assert isinstance(italic_minor, int)


def test_unknown_sfnt_table_returns_none():
    font = ft2font.FT2Font(_find_dejavu_sans())
    assert font.get_sfnt_table("not-a-real-table") is None


def test_get_sfnt_name_table_populated():
    """get_sfnt() should return a dict keyed by 4-tuples of the name
    table header fields. DejaVu Sans has a real name table with the
    family name and style at well-known name_ids.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    names = font.get_sfnt()
    assert isinstance(names, dict)
    assert len(names) > 0
    # Every key is a (platform_id, encoding_id, language_id, name_id)
    # 4-tuple, every value is bytes.
    k = next(iter(names.keys()))
    assert isinstance(k, tuple) and len(k) == 4
    assert all(isinstance(x, int) for x in k)
    assert isinstance(names[k], bytes)


def test_charmap_maps_ascii_letters():
    """get_charmap() must cover at least ASCII letters for DejaVu Sans."""
    font = ft2font.FT2Font(_find_dejavu_sans())
    cm = font.get_charmap()
    assert isinstance(cm, dict)
    assert len(cm) > 100
    for ch in "ABCabc0123":
        assert ord(ch) in cm, f"charmap missing {ch!r}"
        assert cm[ord(ch)] > 0


def test_kerning_pair_av_is_negative():
    """A-V is a classic negative kerning pair — the pair pulls V
    leftward so its slant tucks under A's right edge. Real Latin
    fonts include this pair in their kern table.
    """
    font = ft2font.FT2Font(_find_dejavu_sans())
    font.set_size(12.0, 72.0)
    cm = font.get_charmap()
    a_gid = cm[ord('A')]
    v_gid = cm[ord('V')]
    kerning = font.get_kerning(a_gid, v_gid, 0)
    # Negative means the V glyph is pulled leftward.
    assert kerning < 0, f"expected negative A-V kerning, got {kerning}"


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
