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
        self._fname = fname

    def get_family(self):
        return [self._family] if self._family else ['sans-serif']

    def get_size(self):
        return 10.0

    def get_file(self):
        return self._fname


def findfont(prop, fontext='ttf', directory=None, fallback_to_default=True,
             rebuild_if_missing=True):
    """Return a dummy font path — no real font lookup in WASM sandbox."""
    return ''


def get_font_names():
    """Return empty list — no fonts available in WASM sandbox."""
    return []
