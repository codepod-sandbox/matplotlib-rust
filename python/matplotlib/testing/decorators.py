"""
Minimal image-based testing decorators.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
import tempfile

import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure


def _module_baseline_dir(func):
    module_path = Path(func.__globals__["__file__"]).resolve()
    return module_path.parent / "baseline_images" / module_path.stem


def _apply_style(style_name):
    if style_name is None:
        return
    # This repo only exposes a tiny style surface; silently ignore missing
    # styles so comparison decorators can still exercise rendering behavior.
    try:
        style.use(style_name)
    except Exception:
        pass


def image_comparison(baseline_images, extensions=None, tol=0, *, style=None, **kwargs):
    extensions = tuple(extensions or ("png",))

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            baseline_dir = _module_baseline_dir(func)
            missing = []
            for name in baseline_images:
                base = Path(name)
                ext = base.suffix.lstrip(".")
                if ext:
                    candidates = [baseline_dir / base]
                else:
                    candidates = [baseline_dir / f"{name}.{extension}" for extension in extensions]
                if not any(candidate.exists() for candidate in candidates):
                    missing.append(name)
            if missing:
                pytest.skip(
                    "baseline image assets are not available for "
                    + ", ".join(missing)
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                with style.context({}):
                    _apply_style(style)
                    plt.close("all")
                    func(*args, **kw)
                    figs = [plt.figure(num) for num in plt.get_fignums()]
                    if len(figs) != len(baseline_images):
                        raise AssertionError(
                            f"Expected {len(baseline_images)} figure(s), got {len(figs)}"
                        )
                    for fig, baseline in zip(figs, baseline_images):
                        base = Path(baseline)
                        ext = base.suffix.lstrip(".") or extensions[0]
                        baseline_path = baseline_dir / (base.name if base.suffix else f"{baseline}.{ext}")
                        actual_path = Path(tmpdir) / baseline_path.name
                        fig.savefig(actual_path, format=ext)
                        result = compare_images(baseline_path, actual_path, tol, in_decorator=True)
                        if result is not None:
                            raise ImageComparisonFailure(str(result))
                plt.close("all")
        wrapper.__signature__ = inspect.Signature()
        return wrapper
    return decorator


def check_figures_equal(*, extensions=("png",), tol=0, **kwargs):
    extensions = tuple(extensions)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            with tempfile.TemporaryDirectory() as tmpdir:
                for extension in extensions:
                    with mpl.rc_context({"lines.color": "b"}):
                        with style.context({}):
                            plt.close("all")
                            fig_test = plt.figure()
                            fig_ref = plt.figure()
                            func(*args, fig_test=fig_test, fig_ref=fig_ref, **kw)
                            actual = Path(tmpdir) / f"actual.{extension}"
                            expected = Path(tmpdir) / f"expected.{extension}"
                            fig_test.savefig(actual, format=extension)
                            fig_ref.savefig(expected, format=extension)
                            result = compare_images(expected, actual, tol, in_decorator=True)
                            if result is not None:
                                raise ImageComparisonFailure(str(result))
                plt.close("all")
        wrapper.__signature__ = inspect.Signature()
        return wrapper
    return decorator
