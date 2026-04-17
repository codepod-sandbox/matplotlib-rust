"""Minimal matplotlib.style.core compatibility layer."""

from __future__ import annotations

from contextlib import contextmanager


# Matplotlib excludes a small set of runtime-managed rcParams from style resets.
# This minimal list is enough to support rcdefaults()/rc_file_defaults() in this
# repo without pulling in the full upstream style subsystem.
STYLE_BLACKLIST = {
    "interactive",
    "backend",
    "backend_fallback",
    "webagg.address",
    "webagg.open_in_browser",
    "webagg.port",
    "webagg.port_retries",
    "toolbar",
    "timezone",
    "figure.max_open_warning",
    "savefig.directory",
    "tk.window_focus",
    "docstring.hardcopy",
    "date.epoch",
}


@contextmanager
def context(style, after_reset=False):
    yield


def use(style):
    return None


available = []
