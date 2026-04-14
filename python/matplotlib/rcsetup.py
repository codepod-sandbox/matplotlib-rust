"""
matplotlib.rcsetup — default parameters and RcParams dict subclass.
"""

from contextlib import contextmanager
try:
    from cycler import cycler  # re-export for matplotlib.rcsetup.cycler compatibility
    _prop_cycle_default = cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ])
except ImportError:
    cycler = None  # type: ignore[assignment]
    _prop_cycle_default = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

# Default parameter values mirroring real matplotlib defaults.
_default_params = {
    # Axes
    'axes.prop_cycle': _prop_cycle_default,
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.axisbelow': 'line',
    'axes.formatter.limits': (-5, 6),
    'axes.formatter.use_locale': False,
    'axes.formatter.use_mathtext': False,
    'axes.formatter.min_exponent': 0,
    'axes.formatter.useoffset': True,
    'axes.formatter.offset_threshold': 4,
    'axes.unicode_minus': True,
    'axes.autolimit_mode': 'data',
    'axes.xmargin': 0.05,
    'axes.ymargin': 0.05,
    'axes.zmargin': 0.05,
    'axes.grid.which': 'major',
    'axes.grid.axis': 'both',
    'axes.titlesize': 'large',
    'axes.titlepad': 6.0,
    'axes.titlelocation': 'center',
    'axes.labelsize': 'medium',
    'axes.labelpad': 4.0,

    # Formatter behavior
    'axes.formatter.limits': [-5, 6],
    'axes.formatter.use_locale': False,
    'axes.formatter.use_mathtext': False,
    'axes.formatter.min_exponent': 0,
    'axes.formatter.useoffset': True,
    'axes.formatter.offset_threshold': 2,
    'axes.unicode_minus': True,
    'axes.autolimit_mode': 'data',

    # Figure
    'figure.figsize': [6.4, 4.8],
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'figure.max_open_warning': 20,

    # Lines
    'lines.linewidth': 1.5,
    'lines.linestyle': '-',
    'lines.color': 'C0',
    'lines.marker': 'None',
    'lines.markersize': 6,
    'lines.markerfacecolor': 'auto',
    'lines.markeredgecolor': 'auto',
    'lines.markeredgewidth': 1.0,
    'lines.antialiased': True,
    'lines.dash_joinstyle': 'round',
    'lines.dash_capstyle': 'butt',
    'lines.solid_joinstyle': 'round',
    'lines.solid_capstyle': 'projecting',
    'lines.scale_dashes': True,

    # Markers
    'markers.fillstyle': 'full',

    # Patch
    'patch.linewidth': 1.0,
    'patch.facecolor': 'C0',
    'patch.edgecolor': 'black',
    'patch.antialiased': True,
    'patch.force_edgecolor': False,

    # Hatch
    'hatch.color': 'black',
    'hatch.linewidth': 1.0,

    # Histogram
    'hist.bins': 10,

    # Boxplot
    'boxplot.notch': False,
    'boxplot.vertical': True,
    'boxplot.whiskers': 1.5,
    'boxplot.bootstrap': None,
    'boxplot.patchartist': False,
    'boxplot.showmeans': False,
    'boxplot.showcaps': True,
    'boxplot.showbox': True,
    'boxplot.showfliers': True,
    'boxplot.meanline': False,

    # Scatter
    'scatter.marker': 'o',
    'scatter.edgecolors': 'face',

    # Legend
    'legend.loc': 'best',
    'legend.frameon': True,
    'legend.fontsize': 'medium',
    'legend.framealpha': 0.8,
    'legend.facecolor': 'inherit',
    'legend.edgecolor': '0.8',
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.markerscale': 1.0,
    'legend.title_fontsize': None,
    'legend.columnspacing': 2.0,
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.5,
    'legend.handlelength': 2.0,
    'legend.handleheight': 0.7,
    'legend.handletextpad': 0.8,
    'legend.borderaxespad': 0.5,

    # Grid
    'grid.color': '#b0b0b0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.alpha': 1.0,

    # Ticks
    'xtick.top': False,
    'xtick.bottom': True,
    'xtick.labeltop': False,
    'xtick.labelbottom': True,
    'xtick.major.size': 3.5,
    'xtick.minor.size': 2.0,
    'xtick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'xtick.major.pad': 3.5,
    'xtick.minor.pad': 3.4,
    'xtick.color': 'black',
    'xtick.labelcolor': 'inherit',
    'xtick.labelsize': 'medium',
    'xtick.direction': 'out',
    'xtick.minor.visible': False,
    'xtick.major.top': True,
    'xtick.major.bottom': True,
    'xtick.minor.top': True,
    'xtick.minor.bottom': True,
    'xtick.alignment': 'center',
    'xtick.minor.ndivs': 'auto',
    'ytick.left': True,
    'ytick.right': False,
    'ytick.labelleft': True,
    'ytick.labelright': False,
    'ytick.major.size': 3.5,
    'ytick.minor.size': 2.0,
    'ytick.major.width': 0.8,
    'ytick.minor.width': 0.6,
    'ytick.major.pad': 3.5,
    'ytick.minor.pad': 3.4,
    'ytick.color': 'black',
    'ytick.labelcolor': 'inherit',
    'ytick.labelsize': 'medium',
    'ytick.direction': 'out',
    'ytick.minor.visible': False,
    'ytick.major.left': True,
    'ytick.major.right': True,
    'ytick.minor.left': True,
    'ytick.minor.right': True,
    'ytick.alignment': 'center_baseline',
    'ytick.minor.ndivs': 'auto',

    # Text
    'text.antialiased': True,
    'text.color': 'black',
    'text.usetex': False,

    # Font
    'font.family': ['sans-serif'],
    'font.style': 'normal',
    'font.variant': 'normal',
    'font.weight': 'normal',
    'font.stretch': 'normal',
    'font.size': 10.0,
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.monospace': ['DejaVu Sans Mono', 'Courier New'],

    # Image
    'image.cmap': 'viridis',
    'image.interpolation': 'antialiased',
    'image.aspect': 'equal',
    'image.origin': 'upper',
    'image.lut': 256,

    # Errorbar
    'errorbar.capsize': 0,

    # Saving
    'savefig.dpi': 'figure',
    'savefig.format': 'png',
    'savefig.bbox': None,
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'auto',
    'savefig.edgecolor': 'auto',
    'savefig.transparent': False,

    # Date
    'date.autoformatter.year': '%Y',
    'date.autoformatter.month': '%Y-%m',
    'date.autoformatter.day': '%Y-%m-%d',

    # Axes3d
    'axes3d.grid': True,

    # Animation
    'animation.html': 'none',

    # Path
    'path.simplify': True,
    'path.simplify_threshold': 0.111111111111,
    'path.snap': True,
    'path.sketch': None,
    'path.effects': [],
    'agg.path.chunksize': 0,

    # Internal flags used by real matplotlib ticker/scale modules
    '_internal.classic_mode': False,

    # Axes title/label extras
    'axes.titleweight': 'normal',
    'axes.titley': None,
    'axes.titlecolor': 'auto',
    'axes.labelcolor': 'black',
    'axes.labelweight': 'normal',

    # Axis label location
    'xaxis.labellocation': 'center',
    'yaxis.labellocation': 'center',

    # Boxplot extras
    'boxplot.flierprops.color': 'black',
    'boxplot.flierprops.linestyle': 'none',
    'boxplot.flierprops.linewidth': 1.0,
    'boxplot.flierprops.marker': 'o',
    'boxplot.flierprops.markeredgecolor': 'black',
    'boxplot.flierprops.markeredgewidth': 1.0,
    'boxplot.flierprops.markerfacecolor': 'auto',
    'boxplot.flierprops.markersize': 6.0,
    'boxplot.boxprops.color': 'black',
    'boxplot.boxprops.linestyle': '-',
    'boxplot.boxprops.linewidth': 1.0,
    'boxplot.whiskerprops.color': 'black',
    'boxplot.whiskerprops.linestyle': '-',
    'boxplot.whiskerprops.linewidth': 1.0,
    'boxplot.capprops.color': 'black',
    'boxplot.capprops.linestyle': '-',
    'boxplot.capprops.linewidth': 1.0,
    'boxplot.medianprops.color': 'C1',
    'boxplot.medianprops.linestyle': '-',
    'boxplot.medianprops.linewidth': 1.0,
    'boxplot.meanprops.color': 'C2',
    'boxplot.meanprops.linestyle': '--',
    'boxplot.meanprops.linewidth': 1.0,
    'boxplot.meanprops.marker': '^',
    'boxplot.meanprops.markerfacecolor': 'C2',
    'boxplot.meanprops.markeredgecolor': 'C2',
    'boxplot.meanprops.markersize': 6.0,

    # pcolor/pcolormesh
    'pcolor.shading': 'auto',
    'pcolormesh.snap': True,

    # image extras
    'image.resample': True,
    'image.composite_image': True,

    # Scatter extras
    'scatter.edgecolors': 'face',

    # Contour
    'contour.negative_linestyle': 'dashed',
    'contour.corner_mask': True,
    'contour.linewidth': None,
    'contour.algorithm': 'mpl2014',

    # Hatching
    'hatch.color': 'black',
    'hatch.linewidth': 1.0,

    # Polaraxes
    'polaraxes.grid': True,

    # Misc axes
    'axes3d.grid': True,
    'axes.labelsize': 'medium',
}


def validate_bool(b):
    """Convert b to ``bool`` or raise."""
    if isinstance(b, str):
        b = b.lower()
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError(f'Cannot convert {b!r} to bool')


def validate_axisbelow(s):
    try:
        return validate_bool(s)
    except ValueError:
        if isinstance(s, str):
            if s == 'line':
                return 'line'
    raise ValueError(f'{s!r} cannot be interpreted as True, False, or "line"')


class RcParams(dict):
    """A dictionary subclass for storing matplotlib configuration parameters.

    Behaves like a regular dict but can be extended with validation
    in the future.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        lines = [f"RcParams({{"]
        for k in sorted(self):
            lines.append(f"    {k!r}: {self[k]!r},")
        lines.append("})")
        return '\n'.join(lines)

    def __str__(self):
        return self.__repr__()

    def find_all(self, pattern):
        """Return a dict of entries whose keys contain *pattern*."""
        import re
        regex = re.compile(pattern)
        return {k: v for k, v in self.items() if regex.search(k)}

    def copy(self):
        """Return a copy of this RcParams."""
        return RcParams(super().copy())


class _RcContext:
    """Context manager that temporarily overrides rcParams."""

    def __init__(self, rc_params_obj, overrides=None):
        self._rc = rc_params_obj
        self._overrides = overrides or {}
        self._saved = {}

    def __enter__(self):
        # Save current values for keys we will override
        for k in self._overrides:
            if k in self._rc:
                self._saved[k] = self._rc[k]
            else:
                self._saved[k] = _sentinel
        # Apply overrides
        self._rc.update(self._overrides)
        return self._rc

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for k, v in self._saved.items():
            if v is _sentinel:
                self._rc.pop(k, None)
            else:
                self._rc[k] = v
        return False


# Sentinel for keys that did not exist before context entry
_sentinel = object()


@contextmanager
def rc_context(rc=None):
    """Context manager for temporarily changing rcParams.

    Parameters
    ----------
    rc : dict, optional
        Mapping of rcParams keys to values to set within the context.

    Example
    -------
    >>> import matplotlib
    >>> with matplotlib.rc_context({'lines.linewidth': 3.0}):
    ...     print(matplotlib.rcParams['lines.linewidth'])
    3.0
    """
    # Import here to avoid circular imports at module level
    import matplotlib
    ctx = _RcContext(matplotlib.rcParams, rc)
    ctx.__enter__()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)
