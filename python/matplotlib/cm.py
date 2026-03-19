"""
matplotlib.cm --- Colormap registry and ScalarMappable base class.

Provides colormap objects (ListedColormap, LinearSegmentedColormap),
a global registry, get_cmap(), and the ScalarMappable mixin.
"""

from matplotlib.colors import Normalize, to_rgba


# ===================================================================
# Colormap base class
# ===================================================================

class Colormap:
    """Base class for all colormaps.

    A colormap maps scalar values in [0, 1] to RGBA tuples.
    """

    def __init__(self, name, N=256):
        self.name = name
        self.N = N
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)   # for masked/NaN
        self._rgba_under = None
        self._rgba_over = None

    def __call__(self, X, alpha=None, bytes=False):
        """Map scalar value(s) *X* in [0, 1] to RGBA.

        Parameters
        ----------
        X : float or list of float
            Values in [0, 1].
        alpha : float, optional
            Override alpha.
        bytes : bool
            If True return 0-255 integers instead of 0-1 floats.

        Returns
        -------
        tuple or list of tuple
        """
        scalar = not hasattr(X, '__iter__')
        if scalar:
            X = [X]

        result = []
        for x in X:
            if x != x:  # NaN check
                rgba = self._rgba_bad
            elif x < 0.0:
                rgba = self._rgba_under if self._rgba_under is not None else self._map_scalar(0.0)
            elif x > 1.0:
                rgba = self._rgba_over if self._rgba_over is not None else self._map_scalar(1.0)
            else:
                rgba = self._map_scalar(x)
            if alpha is not None:
                rgba = (rgba[0], rgba[1], rgba[2], float(alpha))
            if bytes:
                rgba = (int(round(rgba[0] * 255)),
                        int(round(rgba[1] * 255)),
                        int(round(rgba[2] * 255)),
                        int(round(rgba[3] * 255)))
            result.append(rgba)

        if scalar:
            return result[0]
        return result

    def _map_scalar(self, x):
        """Map a single float in [0, 1] to RGBA. Subclasses override."""
        return (x, x, x, 1.0)

    def set_bad(self, color='k', alpha=None):
        """Set the color for masked/NaN values."""
        self._rgba_bad = to_rgba(color, alpha=alpha)

    def set_under(self, color='k', alpha=None):
        """Set the color for values below vmin."""
        self._rgba_under = to_rgba(color, alpha=alpha)

    def set_over(self, color='k', alpha=None):
        """Set the color for values above vmax."""
        self._rgba_over = to_rgba(color, alpha=alpha)

    def get_bad(self):
        """Return RGBA for masked/bad values."""
        return self._rgba_bad

    def get_under(self):
        """Return RGBA for under values."""
        return self._rgba_under if self._rgba_under is not None else self._map_scalar(0.0)

    def get_over(self):
        """Return RGBA for over values."""
        return self._rgba_over if self._rgba_over is not None else self._map_scalar(1.0)

    @property
    def colors(self):
        """Return the list of colors (only meaningful for ListedColormap)."""
        raise AttributeError(f"{type(self).__name__} has no 'colors' attribute")

    def reversed(self, name=None):
        """Return a reversed copy of this colormap."""
        if name is None:
            name = self.name + '_r'
        # Default implementation: subclasses should override
        return self  # pragma: no cover

    def resampled(self, lutsize):
        """Return a resampled copy with *lutsize* entries."""
        colors = [self._map_scalar(i / max(1, lutsize - 1))
                  for i in range(lutsize)]
        return ListedColormap(colors, name=self.name, N=lutsize)

    def __eq__(self, other):
        if not isinstance(other, Colormap):
            return NotImplemented
        return (type(self) is type(other)
                and self.name == other.name
                and self.N == other.N)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, N={self.N})"

    def __copy__(self):
        return self

    def copy(self):
        """Return a copy of the colormap."""
        import copy
        return copy.copy(self)


# ===================================================================
# ListedColormap
# ===================================================================

class ListedColormap(Colormap):
    """A colormap defined by a list of colors.

    Parameters
    ----------
    colors : list of color
        Colour specifications.
    name : str
        Name of the colormap.
    N : int, optional
        Number of entries. If None, len(colors) is used.
    """

    def __init__(self, colors, name='from_list', N=None):
        if N is None:
            N = len(colors)
        super().__init__(name, N)
        # Store as RGBA tuples
        self._colors_list = [to_rgba(c) for c in colors]

    @property
    def colors(self):
        return list(self._colors_list)

    def _map_scalar(self, x):
        if not self._colors_list:
            return (0.0, 0.0, 0.0, 1.0)
        n = len(self._colors_list)
        if n == 1:
            return self._colors_list[0]
        idx = int(x * (n - 1) + 0.5)
        idx = max(0, min(n - 1, idx))
        return self._colors_list[idx]

    def reversed(self, name=None):
        if name is None:
            name = self.name + '_r'
        return ListedColormap(list(reversed(self._colors_list)),
                              name=name, N=self.N)

    def resampled(self, lutsize):
        """Return a resampled ListedColormap."""
        colors = [self._map_scalar(i / max(1, lutsize - 1))
                  for i in range(lutsize)]
        return ListedColormap(colors, name=self.name, N=lutsize)


# ===================================================================
# LinearSegmentedColormap
# ===================================================================

class LinearSegmentedColormap(Colormap):
    """A colormap defined by linear interpolation segments.

    Parameters
    ----------
    name : str
        Name of the colormap.
    segmentdata : dict
        Dictionary with keys 'red', 'green', 'blue' (and optionally 'alpha').
        Each maps to a list of (x, y0, y1) tuples.
    N : int
        Number of RGB quantization levels.
    gamma : float
        Gamma correction factor.
    """

    def __init__(self, name, segmentdata, N=256, gamma=1.0):
        super().__init__(name, N)
        self._segmentdata = segmentdata
        self._gamma = gamma

    @staticmethod
    def from_list(name, colors, N=256, gamma=1.0):
        """Create a LinearSegmentedColormap from a list of colors.

        Parameters
        ----------
        name : str
        colors : list of color
            Evenly spaced or (value, color) pairs.
        N : int
        gamma : float
        """
        # Handle (value, color) pairs
        if colors and isinstance(colors[0], (tuple, list)) and len(colors[0]) == 2:
            # Check if it's (value, color) or just an RGBA tuple
            first = colors[0]
            if isinstance(first[0], (int, float)) and isinstance(first[1], (str, tuple, list)):
                vals = [c[0] for c in colors]
                rgbas = [to_rgba(c[1]) for c in colors]
            else:
                # It's a list of color tuples
                n = len(colors)
                vals = [i / (n - 1) for i in range(n)]
                rgbas = [to_rgba(c) for c in colors]
        else:
            n = len(colors)
            vals = [i / max(1, n - 1) for i in range(n)]
            rgbas = [to_rgba(c) for c in colors]

        segmentdata = {}
        for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
            data = []
            for i, (v, rgba) in enumerate(zip(vals, rgbas)):
                data.append((v, rgba[channel_idx], rgba[channel_idx]))
            segmentdata[channel_name] = data

        # Alpha channel
        alpha_data = []
        for v, rgba in zip(vals, rgbas):
            alpha_data.append((v, rgba[3], rgba[3]))
        segmentdata['alpha'] = alpha_data

        return LinearSegmentedColormap(name, segmentdata, N=N, gamma=gamma)

    def _map_scalar(self, x):
        r = self._interpolate_channel(x, self._segmentdata.get('red', [(0, 0, 0), (1, 1, 1)]))
        g = self._interpolate_channel(x, self._segmentdata.get('green', [(0, 0, 0), (1, 1, 1)]))
        b = self._interpolate_channel(x, self._segmentdata.get('blue', [(0, 0, 0), (1, 1, 1)]))
        a = self._interpolate_channel(x, self._segmentdata.get('alpha', [(0, 1, 1), (1, 1, 1)]))
        return (max(0.0, min(1.0, r)),
                max(0.0, min(1.0, g)),
                max(0.0, min(1.0, b)),
                max(0.0, min(1.0, a)))

    def _interpolate_channel(self, x, data):
        """Linearly interpolate through segment data."""
        if not data:
            return 0.0
        # Apply gamma
        x = x ** self._gamma if self._gamma != 1.0 else x

        # data is list of (x_i, y0_i, y1_i)
        if x <= data[0][0]:
            return data[0][1]
        if x >= data[-1][0]:
            return data[-1][2]

        for i in range(len(data) - 1):
            x0, _, y1_left = data[i]
            x1, y0_right, _ = data[i + 1]
            if x0 <= x <= x1:
                if x1 == x0:
                    return y1_left
                t = (x - x0) / (x1 - x0)
                return y1_left + t * (y0_right - y1_left)

        return data[-1][2]

    def reversed(self, name=None):
        if name is None:
            name = self.name + '_r'
        new_segdata = {}
        for key, data in self._segmentdata.items():
            new_segdata[key] = [(1.0 - x, y1, y0) for x, y0, y1 in reversed(data)]
        return LinearSegmentedColormap(name, new_segdata, self.N, self._gamma)

    def resampled(self, lutsize):
        """Return a resampled copy."""
        return LinearSegmentedColormap(
            self.name, self._segmentdata, N=lutsize, gamma=self._gamma)


# ===================================================================
# Built-in colormaps
# ===================================================================

def _make_viridis():
    """Create the 'viridis' colormap (simplified 8-stop version)."""
    colors = [
        (0.267004, 0.004874, 0.329415),
        (0.282327, 0.140926, 0.457517),
        (0.253935, 0.265254, 0.529983),
        (0.206756, 0.371758, 0.553117),
        (0.163625, 0.471133, 0.558148),
        (0.127568, 0.566949, 0.550556),
        (0.134692, 0.658636, 0.517649),
        (0.477504, 0.821444, 0.318195),
        (0.741388, 0.873449, 0.149561),
        (0.993248, 0.906157, 0.143936),
    ]
    return LinearSegmentedColormap.from_list('viridis', colors, N=256)


def _make_jet():
    """Create the 'jet' colormap."""
    segmentdata = {
        'red':   [(0.0, 0.0, 0.0), (0.35, 0.0, 0.0), (0.66, 1.0, 1.0), (0.89, 1.0, 1.0), (1.0, 0.5, 0.5)],
        'green': [(0.0, 0.0, 0.0), (0.125, 0.0, 0.0), (0.375, 1.0, 1.0), (0.64, 1.0, 1.0), (0.91, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 0.5, 0.5), (0.11, 1.0, 1.0), (0.34, 1.0, 1.0), (0.65, 0.0, 0.0), (1.0, 0.0, 0.0)],
    }
    return LinearSegmentedColormap('jet', segmentdata, N=256)


def _make_hot():
    """Create the 'hot' colormap."""
    segmentdata = {
        'red':   [(0.0, 0.0416, 0.0416), (0.365079, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'green': [(0.0, 0.0, 0.0), (0.365079, 0.0, 0.0), (0.746032, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'blue':  [(0.0, 0.0, 0.0), (0.746032, 0.0, 0.0), (1.0, 1.0, 1.0)],
    }
    return LinearSegmentedColormap('hot', segmentdata, N=256)


def _make_cool():
    """Create the 'cool' colormap."""
    return LinearSegmentedColormap.from_list(
        'cool', [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0)], N=256)


def _make_gray():
    """Create the 'gray' colormap."""
    return LinearSegmentedColormap.from_list(
        'gray', [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], N=256)


def _make_spring():
    return LinearSegmentedColormap.from_list(
        'spring', [(1.0, 0.0, 1.0), (1.0, 1.0, 0.0)], N=256)


def _make_summer():
    return LinearSegmentedColormap.from_list(
        'summer', [(0.0, 0.5, 0.4), (1.0, 1.0, 0.4)], N=256)


def _make_autumn():
    return LinearSegmentedColormap.from_list(
        'autumn', [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], N=256)


def _make_winter():
    return LinearSegmentedColormap.from_list(
        'winter', [(0.0, 0.0, 1.0), (0.0, 1.0, 0.5)], N=256)


def _make_plasma():
    colors = [
        (0.050383, 0.029803, 0.527975),
        (0.274191, 0.014444, 0.635363),
        (0.452780, 0.002472, 0.638153),
        (0.607840, 0.063010, 0.542920),
        (0.735683, 0.175158, 0.402049),
        (0.835270, 0.313349, 0.261530),
        (0.908627, 0.472975, 0.135920),
        (0.949217, 0.643310, 0.028508),
        (0.940015, 0.817580, 0.095953),
    ]
    return LinearSegmentedColormap.from_list('plasma', colors, N=256)


def _make_inferno():
    colors = [
        (0.001462, 0.000466, 0.013866),
        (0.120638, 0.047674, 0.283397),
        (0.316822, 0.071690, 0.428685),
        (0.512831, 0.116472, 0.422491),
        (0.698325, 0.174138, 0.325932),
        (0.854063, 0.272394, 0.161962),
        (0.953099, 0.440640, 0.013552),
        (0.976580, 0.645640, 0.068880),
        (0.929644, 0.862325, 0.297150),
    ]
    return LinearSegmentedColormap.from_list('inferno', colors, N=256)


def _make_magma():
    colors = [
        (0.001462, 0.000466, 0.013866),
        (0.109384, 0.054213, 0.306905),
        (0.305819, 0.082080, 0.484932),
        (0.497960, 0.125675, 0.505780),
        (0.679408, 0.190631, 0.441945),
        (0.845561, 0.296706, 0.328218),
        (0.953099, 0.459929, 0.230600),
        (0.985891, 0.649882, 0.208770),
        (0.987387, 0.862325, 0.395490),
    ]
    return LinearSegmentedColormap.from_list('magma', colors, N=256)


def _make_cividis():
    colors = [
        (0.0, 0.135112, 0.304751),
        (0.127568, 0.258073, 0.425230),
        (0.289555, 0.371758, 0.487059),
        (0.430983, 0.488889, 0.489307),
        (0.583761, 0.597073, 0.440933),
        (0.759356, 0.699644, 0.350556),
        (0.946260, 0.813913, 0.226135),
    ]
    return LinearSegmentedColormap.from_list('cividis', colors, N=256)


# ===================================================================
# Colormap registry
# ===================================================================

_cmap_registry = {}


def _register_builtin_cmaps():
    """Register all built-in colormaps."""
    builders = {
        'viridis': _make_viridis,
        'jet': _make_jet,
        'hot': _make_hot,
        'cool': _make_cool,
        'gray': _make_gray,
        'spring': _make_spring,
        'summer': _make_summer,
        'autumn': _make_autumn,
        'winter': _make_winter,
        'plasma': _make_plasma,
        'inferno': _make_inferno,
        'magma': _make_magma,
        'cividis': _make_cividis,
    }
    for name, builder in builders.items():
        cmap = builder()
        _cmap_registry[name] = cmap
        # Register reversed version
        _cmap_registry[name + '_r'] = cmap.reversed()

    # grey alias
    _cmap_registry['grey'] = _cmap_registry['gray']
    _cmap_registry['grey_r'] = _cmap_registry['gray_r']


_register_builtin_cmaps()


def get_cmap(name=None, lut=None):
    """Return a colormap by name.

    Parameters
    ----------
    name : str or Colormap or None
        If None, return the default ('viridis').
        If a Colormap instance, return it directly.
    lut : int, optional
        Number of colors in the lookup table.

    Returns
    -------
    Colormap
    """
    if name is None:
        name = 'viridis'
    if isinstance(name, Colormap):
        if lut is not None:
            return name.resampled(lut)
        return name
    if name not in _cmap_registry:
        raise ValueError(f"'{name}' is not a known colormap name")
    cmap = _cmap_registry[name]
    if lut is not None:
        cmap = cmap.resampled(lut)
    return cmap


def register_cmap(name=None, cmap=None):
    """Register a colormap.

    Parameters
    ----------
    name : str, optional
        Name for the colormap. If None, uses cmap.name.
    cmap : Colormap
        The colormap to register.
    """
    if cmap is None:
        raise ValueError("A Colormap must be provided")
    if name is None:
        name = cmap.name
    _cmap_registry[name] = cmap


class ColormapRegistry:
    """A dict-like registry of colormaps."""

    def __getitem__(self, name):
        return get_cmap(name)

    def __contains__(self, name):
        return name in _cmap_registry

    def __iter__(self):
        return iter(sorted(_cmap_registry))

    def __len__(self):
        return len(_cmap_registry)

    def __call__(self, name=None, lut=None):
        """Get a colormap by name (like get_cmap)."""
        return get_cmap(name, lut)

    def register(self, cmap, name=None, force=False):
        """Register a colormap."""
        if name is None:
            name = cmap.name
        if name in _cmap_registry and not force:
            raise ValueError(
                f"'{name}' is already registered. Use force=True to override.")
        _cmap_registry[name] = cmap

    def unregister(self, name):
        """Unregister a colormap."""
        _cmap_registry.pop(name, None)


# Module-level registry instance
_colormaps = ColormapRegistry()


# ===================================================================
# ScalarMappable
# ===================================================================

class ScalarMappable:
    """Mixin for objects that map scalar data to RGBA colors.

    Uses a Normalize + Colormap to map data values to colors.
    """

    def __init__(self, norm=None, cmap=None):
        self._norm = norm
        self._cmap = get_cmap(cmap) if isinstance(cmap, str) or cmap is None else cmap
        self._A = None  # data array
        self.colorbar = None  # set by colorbar
        self._update_dict = {'array': False}

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm):
        if norm is None:
            norm = Normalize()
        self._norm = norm

    def set_norm(self, norm):
        self.norm = norm

    def get_norm(self):
        return self._norm

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        if isinstance(cmap, str):
            cmap = get_cmap(cmap)
        self._cmap = cmap

    def set_cmap(self, cmap):
        self.cmap = cmap

    def get_cmap(self):
        return self._cmap

    def get_array(self):
        return self._A

    def set_array(self, A):
        if A is not None and hasattr(A, '__iter__'):
            self._A = list(A)
        else:
            self._A = A

    def get_clim(self):
        if self._norm is not None:
            return (self._norm.vmin, self._norm.vmax)
        return (None, None)

    def set_clim(self, vmin=None, vmax=None):
        if vmin is not None and hasattr(vmin, '__iter__'):
            vmin, vmax = vmin
        if self._norm is None:
            self._norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            if vmin is not None:
                self._norm.vmin = vmin
            if vmax is not None:
                self._norm.vmax = vmax

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """Map data values to RGBA.

        Parameters
        ----------
        x : float or list
            Data values.
        alpha : float, optional
        bytes : bool
        norm : bool
            If True, normalize x first.
        """
        if norm and self._norm is not None:
            x_normed = self._norm(x)
        else:
            x_normed = x
        return self._cmap(x_normed, alpha=alpha, bytes=bytes)

    def changed(self):
        """Call to notify that the mappable has changed."""
        pass

    def autoscale(self):
        """Autoscale the norm from the current array."""
        if self._A is not None and self._norm is not None:
            self._norm.autoscale(self._A)

    def autoscale_None(self):
        """Autoscale only None vmin/vmax."""
        if self._A is not None:
            if self._norm is None:
                self._norm = Normalize()
            self._norm.autoscale_None(self._A)
