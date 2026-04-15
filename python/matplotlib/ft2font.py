"""Stub for matplotlib.ft2font.

Replaced by crates/matplotlib-ft2font (fontdue) in Phase 2.
"""

__freetype_version__ = "2.6.1"
__freetype_build_type__ = "local"

LOAD_DEFAULT = 0
LOAD_NO_SCALE = 1

# Enum-like classes expected by OG _text_helpers.py
class LoadFlags:
    """Stub for ft2font.LoadFlags."""
    DEFAULT = 0
    NO_SCALE = 1
    NO_HINTING = 2
    RENDER = 4
    NO_BITMAP = 8
    VERTICAL_LAYOUT = 16
    FORCE_AUTOHINT = 32
    CROP_BITMAP = 64
    PEDANTIC = 128
    IGNORE_GLOBAL_ADVANCE_WIDTH = 512
    NO_RECURSE = 1024
    IGNORE_TRANSFORM = 2048
    MONOCHROME = 4096
    LINEAR_DESIGN = 8192
    NO_AUTOHINT = 32768


class Kerning:
    """Stub for ft2font.Kerning."""
    DEFAULT = 0
    UNFITTED = 1
    UNSCALED = 2


class FaceFlags:
    """Stub for ft2font.FaceFlags."""
    SCALABLE = 1
    FIXED_SIZES = 2
    FIXED_WIDTH = 4
    SFNT = 8
    HORIZONTAL = 16
    VERTICAL = 32
    KERNING = 64
    FAST_GLYPHS = 128
    MULTIPLE_MASTERS = 256
    GLYPH_NAMES = 512
    EXTERNAL_STREAM = 1024
    HINTER = 2048
    CID_KEYED = 4096
    TRICKY = 8192
    COLOR = 16384


class StyleFlags:
    """Stub for ft2font.StyleFlags."""
    NORMAL = 0
    ITALIC = 1
    BOLD = 2



LOAD_NO_HINTING = 2
LOAD_RENDER = 4
LOAD_NO_BITMAP = 8
LOAD_VERTICAL_LAYOUT = 16
LOAD_FORCE_AUTOHINT = 32
LOAD_CROP_BITMAP = 64
LOAD_PEDANTIC = 128
LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH = 512
LOAD_NO_RECURSE = 1024
LOAD_IGNORE_TRANSFORM = 2048
LOAD_MONOCHROME = 4096
LOAD_LINEAR_DESIGN = 8192
LOAD_NO_AUTOHINT = 32768

KERNING_DEFAULT = 0
KERNING_UNFITTED = 1
KERNING_UNSCALED = 2

FACE_FLAG_SCALABLE = 1
FACE_FLAG_FIXED_SIZES = 2
FACE_FLAG_FIXED_WIDTH = 4
FACE_FLAG_SFNT = 8
FACE_FLAG_HORIZONTAL = 16
FACE_FLAG_VERTICAL = 32
FACE_FLAG_KERNING = 64
FACE_FLAG_FAST_GLYPHS = 128
FACE_FLAG_MULTIPLE_MASTERS = 256
FACE_FLAG_GLYPH_NAMES = 512
FACE_FLAG_BOLD = 1
FACE_FLAG_ITALIC = 2
STYLE_FLAG_ITALIC = 1
STYLE_FLAG_BOLD = 2


class FT2Font:
    """Stub FT2Font.

    Attributes are populated so font_manager can scan fonts without crashing.
    Rendering calls raise NotImplementedError.
    """
    def __init__(self, filename, hinting_factor=8, *, _fallback_list=None,
                 _kerning=False, _kerning_factor=None):
        self.fname = filename
        self.family_name = ""
        self.style_name = ""
        self.face_flags = FACE_FLAG_SCALABLE | FACE_FLAG_HORIZONTAL
        self.style_flags = 0
        self.num_fixed_sizes = 0
        self.num_charmaps = 0
        self.scalable = True
        self.units_per_EM = 2048
        self.underline_position = -100
        self.underline_thickness = 50
        self.bbox = (0, 0, 2048, 2048)
        self.ascender = 800
        self.descender = -200
        self.height = 1200
        self.max_advance_width = 1200
        self.max_advance_height = 1200
        self.num_glyphs = 0
        self.postscript_name = ""

    def set_size(self, ptsize, dpi):
        # Store for later metric estimation
        self._ptsize = float(ptsize)
        self._dpi = float(dpi)

    def set_text(self, s, angle=0.0, flags=LOAD_FORCE_AUTOHINT):
        # Store for later metric estimation
        self._current_text = str(s) if s is not None else ""

    def get_width_height(self):
        # Returns values in FT 26.6 fixed-point (1 unit = 1/64 pixel)
        # Rough estimate: char width 0.6 * ptsize * dpi / 72, line height 1.2 *
        size = getattr(self, '_ptsize', 10.0) * getattr(self, '_dpi', 72.0) / 72.0
        n = len(getattr(self, '_current_text', ''))
        w = n * size * 0.6 * 64.0
        h = size * 1.2 * 64.0
        return w, h

    def get_descent(self):
        size = getattr(self, '_ptsize', 10.0) * getattr(self, '_dpi', 72.0) / 72.0
        return size * 0.2 * 64.0

    def draw_glyphs_to_bitmap(self, antialiased=True):
        # No-op: we don't rasterize glyphs; return a fake empty bitmap
        import numpy as np
        self._bitmap = np.zeros((1, 1), dtype=np.uint8)

    def draw_glyph_to_bitmap(self, image, x, y, glyph, antialiased=True):
        # No-op: individual glyph rendering for mathtext
        pass

    def get_bitmap_offset(self):
        return 0, 0

    def get_image(self):
        import numpy as np
        return np.zeros((1, 1), dtype=np.uint8)

    def load_char(self, charcode, flags=LOAD_FORCE_AUTOHINT):
        # Return a fake Glyph-like object
        class _FakeGlyph:
            width = 0
            height = 0
            horiBearingX = 0
            horiBearingY = 0
            horiAdvance = 0
            linearHoriAdvance = 0
            vertBearingX = 0
            vertBearingY = 0
            vertAdvance = 0
            bbox = (0, 0, 0, 0)
        return _FakeGlyph()

    def load_glyph(self, glyphindex, flags=LOAD_FORCE_AUTOHINT):
        return self.load_char(glyphindex, flags)

    def get_kerning(self, left, right, mode):
        return 0

    def get_charmap(self):
        return {}

    def get_name_index(self, name):
        return 0

    def get_char_index(self, codepoint):
        return 0

    def get_sfnt(self):
        return {}

    def get_sfnt_table(self, name):
        return None

    def get_xys(self, antialiased=True):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_glyph_name(self, index):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_num_glyphs(self):
        return 0

    def get_path(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_ps_font_info(self):
        return None

    def select_charmap(self, i):
        pass

    def clear(self):
        """Clear the current glyph set (stub)."""
        pass


class FT2Image:
    """Stub FT2Image."""
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)

    def draw_rect(self, x0, y0, x1, y1):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def draw_rect_filled(self, x0, y0, x1, y1):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")
