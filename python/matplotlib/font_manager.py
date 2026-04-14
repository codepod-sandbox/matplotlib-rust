"""
Stub for matplotlib.font_manager for RustPython/WASM sandbox.

Font file resolution is not meaningful in WASM; all font lookups
return a dummy sentinel so callers can detect the no-font case.
"""


class FontProperties:
    """Minimal FontProperties stub."""

    def __init__(self, family=None, style=None, variant=None, weight=None,
                 stretch=None, size=None, fname=None, math_fontfamily=None):
        self._family = family
        self._style = style
        self._variant = variant
        self._weight = weight
        self._stretch = stretch
        self._size = size
        self._fname = fname
        self._math_fontfamily = math_fontfamily

    def get_family(self):
        return [self._family] if self._family else ['sans-serif']

    def get_size(self):
        return self._size if self._size is not None else 10.0

    def get_size_in_points(self):
        return self.get_size()

    def get_weight(self):
        return self._weight or 'normal'

    def get_style(self):
        return self._style or 'normal'

    def get_file(self):
        return self._fname

    def copy(self):
        """Return a copy of this FontProperties."""
        return FontProperties(
            family=self._family,
            style=self._style,
            variant=self._variant,
            weight=self._weight,
            stretch=self._stretch,
            size=self._size,
            fname=self._fname,
            math_fontfamily=self._math_fontfamily,
        )

    def set_size(self, size):
        self._size = size

    def set_weight(self, weight):
        self._weight = weight

    def set_family(self, family):
        self._family = family

    def set_style(self, style):
        self._style = style

    def set_name(self, name):
        self._family = name


def findfont(prop, fontext='ttf', directory=None, fallback_to_default=True,
             rebuild_if_missing=True):
    """Return a dummy font path — no real font lookup in WASM sandbox."""
    return ''


def get_font_names():
    """Return empty list — no fonts available in WASM sandbox."""
    return []
