"""Stub for matplotlib.ft2font.

Replaced by crates/matplotlib-ft2font (fontdue) in Phase 2.
"""

LOAD_DEFAULT = 0
LOAD_NO_SCALE = 1
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
STYLE_FLAG_ITALIC = 1
STYLE_FLAG_BOLD = 2


class FT2Font:
    def __init__(self, filename, hinting_factor=8, *, _fallback_list=None,
                 _kerning=False):
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
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def set_text(self, s, angle=0.0, flags=LOAD_FORCE_AUTOHINT):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_width_height(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_descent(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def draw_glyphs_to_bitmap(self, antialiased=True):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_image(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def load_char(self, charcode, flags=LOAD_FORCE_AUTOHINT):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def load_glyph(self, glyphindex, flags=LOAD_FORCE_AUTOHINT):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_kerning(self, left, right, mode):
        return 0

    def get_charmap(self):
        return {}

    def get_name_index(self, name):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

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


class FT2Image:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)

    def draw_rect(self, x0, y0, x1, y1):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def draw_rect_filled(self, x0, y0, x1, y1):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")
