"""Stub for matplotlib.colorizer for RustPython/WASM sandbox."""

from matplotlib.colors import Normalize


class Colorizer:
    """Minimal Colorizer stub."""

    def __init__(self, cmap=None, norm=None):
        self.cmap = cmap
        self.norm = norm if norm is not None else Normalize()

    @property
    def vmin(self):
        return self.norm.vmin

    @vmin.setter
    def vmin(self, value):
        self.norm.vmin = float(value) if value is not None else None

    @property
    def vmax(self):
        return self.norm.vmax

    @vmax.setter
    def vmax(self, value):
        self.norm.vmax = float(value) if value is not None else None


class ColorizerMixin:
    """Mixin stub providing colorizer-related methods."""
    pass
