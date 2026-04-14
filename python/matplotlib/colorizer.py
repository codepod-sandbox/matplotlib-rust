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

    @staticmethod
    def _check_exclusionary_keywords(colorizer, **kwargs):
        """Raise ValueError if colorizer is given alongside exclusive kwargs."""
        if colorizer is not None:
            if any(val is not None for val in kwargs.values()):
                raise ValueError(
                    "The `colorizer` keyword cannot be used simultaneously with: "
                    + ", ".join(f'`{k}`' for k in kwargs)
                )

    def _scale_norm(self, norm, vmin, vmax):
        """Helper for initial scaling."""
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)

    def set_clim(self, vmin=None, vmax=None):
        """Set color limits."""
        self._clim = (vmin, vmax)

    def get_clim(self):
        """Get color limits."""
        return getattr(self, '_clim', (None, None))

    def set_norm(self, norm):
        """Set normalization."""
        self._norm = norm

    def get_norm(self):
        """Get normalization."""
        return getattr(self, '_norm', None)

    def set_cmap(self, cmap):
        """Set colormap."""
        self._cmap = cmap

    def get_cmap(self):
        """Get colormap."""
        return getattr(self, '_cmap', None)


class _ScalarMappable(ColorizerMixin):
    """Stub _ScalarMappable base class."""

    @staticmethod
    def _check_exclusionary_keywords(colorizer, **kwargs):
        """Raise ValueError if colorizer is given alongside exclusive kwargs."""
        if colorizer is not None:
            if any(val is not None for val in kwargs.values()):
                raise ValueError(
                    "The `colorizer` keyword cannot be used simultaneously with: "
                    + ", ".join(f'`{k}`' for k in kwargs)
                )

    @staticmethod
    def _get_colorizer(cmap, norm, colorizer):
        if isinstance(colorizer, Colorizer):
            return colorizer
        return Colorizer(cmap, norm)


class ColorizingArtist(_ScalarMappable):
    """Stub ColorizingArtist."""
    pass


# Register docstring interpolation keys used by axes/_axes.py
import matplotlib as _mpl
_mpl._docstring.interpd.register(
    colorizer_doc="colorizer : Colorizer or None, default: None",
    cmap_doc="cmap : str or Colormap, default: :rc:`image.cmap`\n    The colormap.",
    norm_doc="norm : str or Normalize, optional\n    The normalization.",
    vmin_vmax_doc="vmin, vmax : float, optional\n    Colormap range.",
)
