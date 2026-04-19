"""
Core functions and attributes for the matplotlib style library.
"""

from __future__ import annotations

import contextlib
import importlib.resources
import logging
import os
from pathlib import Path
import warnings

import matplotlib as mpl
from matplotlib import _api, _docstring, _rc_params_in_file, rcParamsDefault


_log = logging.getLogger(__name__)

__all__ = ["use", "context", "available", "library", "reload_library"]


BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), "stylelib")
USER_LIBRARY_PATHS = [os.path.join(mpl.get_configdir(), "stylelib")]
STYLE_EXTENSION = "mplstyle"
STYLE_BLACKLIST = {
    "interactive",
    "backend",
    "webagg.port",
    "webagg.address",
    "webagg.port_retries",
    "webagg.open_in_browser",
    "backend_fallback",
    "toolbar",
    "timezone",
    "figure.max_open_warning",
    "figure.raise_window",
    "savefig.directory",
    "tk.window_focus",
    "docstring.hardcopy",
    "date.epoch",
}


def use(style):
    """
    Use Matplotlib style settings from a style specification.
    """
    if isinstance(style, (str, Path)) or hasattr(style, "keys"):
        styles = [style]
    else:
        styles = style

    style_alias = {
        "mpl20": "default",
        "mpl15": "classic",
        "_classic_test": ["classic", "_classic_test_patch"],
    }

    for style in styles:
        if isinstance(style, str):
            style = style_alias.get(style, style)
            if isinstance(style, list):
                use(style)
                continue
            if style == "default":
                with _api.suppress_matplotlib_deprecation_warning():
                    style = {k: rcParamsDefault[k] for k in rcParamsDefault if k not in STYLE_BLACKLIST}
            elif style in library:
                style = library[style]
            elif "." in style:
                pkg, _, name = style.rpartition(".")
                try:
                    path = importlib.resources.files(pkg) / f"{name}.{STYLE_EXTENSION}"
                    style = _rc_params_in_file(path)
                except (ModuleNotFoundError, OSError, TypeError):
                    pass
        if isinstance(style, (str, Path)):
            try:
                style = _rc_params_in_file(style)
            except OSError as err:
                raise OSError(
                    f"{style!r} is not a valid package style, path of style file, "
                    f"URL of style file, or library style name (library styles are "
                    f"listed in `style.available`)"
                ) from err

        filtered = {}
        for k in style:
            if k in STYLE_BLACKLIST:
                _api.warn_external(
                    f"Style includes a parameter, {k!r}, that is not related to style.  "
                    f"Ignoring this parameter."
                )
            else:
                filtered[k] = style[k]
        for key, value in filtered.items():
            mpl.rcParams[key] = value


@contextlib.contextmanager
def context(style, after_reset=False):
    """
    Context manager for using style settings temporarily.
    """
    with mpl.rc_context():
        if after_reset:
            mpl.rcdefaults()
        use(style)
        yield


def update_user_library(library):
    """Update style library with user-defined rc files."""
    for stylelib_path in map(os.path.expanduser, USER_LIBRARY_PATHS):
        styles = read_style_directory(stylelib_path)
        update_nested_dict(library, styles)
    return library


def read_style_directory(style_dir):
    """Return dictionary of styles defined in *style_dir*."""
    styles = {}
    for path in Path(style_dir).glob(f"*.{STYLE_EXTENSION}"):
        with warnings.catch_warnings(record=True) as warns:
            styles[path.stem] = _rc_params_in_file(path)
        for w in warns:
            _log.warning("In %s: %s", path, w.message)
    return styles


def update_nested_dict(main_dict, new_dict):
    """
    Update nested dict (one level of nesting) with new values.
    """
    for name, rc_dict in new_dict.items():
        main_dict.setdefault(name, {}).update(rc_dict)
    return main_dict


_base_library = read_style_directory(BASE_LIBRARY_PATH)
library = {}
available = []


def reload_library():
    """Reload the style library."""
    library.clear()
    library.update(update_user_library(_base_library))
    available[:] = sorted(library.keys())


reload_library()
