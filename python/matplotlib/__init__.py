"""
matplotlib — plotting library for codepod.

SVG primary output with optional PIL/PNG backend.
"""

__version__ = "3.8.0"

from matplotlib.rcsetup import RcParams, _default_params, rc_context

# Global configuration parameters
rcParams = RcParams(_default_params)

# Interactive mode flag
_interactive = False


def is_interactive():
    """Return whether interactive mode is enabled."""
    return _interactive


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
