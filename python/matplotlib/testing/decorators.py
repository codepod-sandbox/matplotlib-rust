"""
Stub for matplotlib.testing.decorators for RustPython/WASM sandbox.

Image-comparison tests are skipped — no reference images in sandbox.
"""

import pytest
import functools


def image_comparison(baseline_images, extensions=None, tol=0, **kwargs):
    """Skip image comparison tests in WASM sandbox."""
    def decorator(func):
        @pytest.mark.skip(reason="image_comparison not available in WASM sandbox")
        @functools.wraps(func)
        def wrapper(*args, **kw):
            return func(*args, **kw)
        return wrapper
    return decorator


def check_figures_equal(*, extensions=('png',), tol=0, **kwargs):
    """Skip figure-comparison tests in WASM sandbox."""
    def decorator(func):
        @pytest.mark.skip(reason="check_figures_equal not available in WASM sandbox")
        @functools.wraps(func)
        def wrapper(*args, **kw):
            return func(*args, **kw)
        return wrapper
    return decorator
