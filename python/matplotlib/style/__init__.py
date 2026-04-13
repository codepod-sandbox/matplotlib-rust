"""Stub for matplotlib.style for RustPython/WASM sandbox."""

from contextlib import contextmanager


@contextmanager
def context(style, after_reset=False):
    """No-op style context — styles not supported in WASM sandbox."""
    yield


def use(style):
    """No-op — styles not supported in WASM sandbox."""
    pass


def available():
    return []
