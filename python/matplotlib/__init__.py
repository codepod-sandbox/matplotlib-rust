"""
matplotlib — plotting library for codepod.

SVG primary output with optional PIL/PNG backend.
"""

__version__ = "3.8.0"

import os as _os
import pathlib as _pathlib

# Data path stub — no bundled data in WASM sandbox
_DATA_PATH = _pathlib.Path(_os.path.dirname(_os.path.abspath(__file__)), "_data")


def get_data_path():
    """Return path to matplotlib data directory (stub for WASM sandbox)."""
    return str(_DATA_PATH)


from matplotlib.rcsetup import RcParams, _default_params, rc_context
from matplotlib import style
from matplotlib.cm import _colormaps as colormaps
from matplotlib.colors import _color_sequences as color_sequences

# Global configuration parameters
rcParams = RcParams(_default_params)

# Interactive mode flag
_interactive = False


def is_interactive():
    """Return whether interactive mode is enabled."""
    return _interactive


def _val_or_rc(val, rc_name):
    """If *val* is None, return ``rcParams[rc_name]``, otherwise return val."""
    return val if val is not None else rcParams[rc_name]


def rc(group, **kwargs):
    """Set rcParams for a *group* of parameters.

    Parameters
    ----------
    group : str
        The group name, e.g. ``'lines'``, ``'axes'``, ``'figure'``.
    **kwargs
        Key/value pairs to set.  Each key is prefixed with
        ``group + '.'`` to form the full rcParams key.

    Example
    -------
    >>> import matplotlib
    >>> matplotlib.rc('lines', linewidth=2.0, linestyle='-')
    """
    for k, v in kwargs.items():
        key = f'{group}.{k}'
        rcParams[key] = v
