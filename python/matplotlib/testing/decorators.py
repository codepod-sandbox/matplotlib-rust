"""Minimal but upstream-compatible image-based testing decorators."""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
import string
import tempfile
import warnings
import contextlib

import pytest
from cycler import cycler

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.units as munits
from matplotlib import _pylab_helpers
from matplotlib import style as mpl_style
from matplotlib.text import Text
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure


def _module_baseline_dir(func):
    module_path = Path(func.__globals__["__file__"]).resolve()
    return module_path.parent / "baseline_images" / module_path.stem


def _module_result_dir(func):
    module_path = Path(inspect.getfile(func)).resolve()
    result_dir = Path.cwd() / "result_images" / module_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _image_directories(func):
    """
    Compute the baseline and result image directories for *func*.

    This matches upstream's testing contract so imported tests can use the
    helper directly instead of rebuilding local path logic.
    """
    return _module_baseline_dir(func), _module_result_dir(func)


def _raise_on_image_difference(expected, actual, tol):
    result = compare_images(expected, actual, tol, in_decorator=True)
    if result is not None:
        raise ImageComparisonFailure(str(result))


@contextlib.contextmanager
def _cleanup_cm():
    orig_units_registry = munits.registry.copy()
    try:
        with warnings.catch_warnings(), mpl.rc_context():
            yield
    finally:
        munits.registry.clear()
        munits.registry.update(orig_units_registry)
        plt.close("all")


class _ExtRequest:
    def __init__(self, ext):
        self._ext = ext

    def getfixturevalue(self, name):
        if name == "ext":
            return self._ext
        raise LookupError(f"Unsupported synthetic fixture lookup: {name}")


def _collect_new_figures():
    managers = _pylab_helpers.Gcf.figs
    preexisting = [manager for manager in managers.values()]
    new_figs = []

    class _Collector:
        def __enter__(self):
            return new_figs

        def __exit__(self, exc_type, exc, tb):
            new_managers = sorted(
                [manager for manager in managers.values() if manager not in preexisting],
                key=lambda manager: manager.num,
            )
            new_figs[:] = [manager.canvas.figure for manager in new_managers]
            return False

    return _Collector()


def _apply_style(style_name):
    if style_name is None:
        return
    # This repo only exposes a tiny style surface; silently ignore missing
    # styles so comparison decorators can still exercise rendering behavior.
    try:
        mpl_style.use(style_name)
    except Exception:
        pass


def _style_context(style):
    base_style = ["classic", "_classic_test_patch"]
    if style is None:
        return mpl_style.context(base_style)
    if isinstance(style, (list, tuple)):
        return mpl_style.context([*base_style, *style])
    return mpl_style.context([*base_style, style])


def _optional_style_context(style):
    return _style_context(style)


def _remove_figure_text(fig):
    for artist in fig.findobj(match=Text):
        artist.set_text("")
        artist.set_visible(False)
    for ax in fig.get_axes():
        ax.title.set_visible(False)
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )


def remove_ticks_and_titles(figure):
    for ax in figure.get_axes():
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])


def image_comparison(baseline_images, extensions=None, tol=0, *, style=None, **kwargs):
    extensions = tuple(extensions or ("png",))
    savefig_kwarg = dict(kwargs.get("savefig_kwarg", {}))
    remove_text = kwargs.get("remove_text", False)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            resolved_baselines = baseline_images
            if resolved_baselines is None:
                resolved_baselines = kw.get("baseline_images")
            if resolved_baselines is None:
                raise TypeError(
                    "image_comparison requires baseline_images either in the "
                    "decorator or as a test argument")
            if isinstance(resolved_baselines, (str, Path)):
                resolved_baselines = [resolved_baselines]

            baseline_dir = _module_baseline_dir(func)
            missing = []
            for name in resolved_baselines:
                base = Path(name)
                candidate_exts = (base.suffix.lstrip("."),) if base.suffix else extensions
                candidates = [
                    baseline_dir / (base.name if base.suffix else f"{name}.{extension}")
                    for extension in candidate_exts
                ]
                if not any(candidate.exists() for candidate in candidates):
                    missing.append(name)
            if missing:
                pytest.skip(
                    "baseline image assets are not available for "
                    + ", ".join(missing)
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                with _optional_style_context(style):
                    plt.close("all")
                    func(*args, **kw)
                    figs = [plt.figure(num) for num in plt.get_fignums()]
                    if len(figs) != len(resolved_baselines):
                        raise AssertionError(
                            f"Expected {len(resolved_baselines)} figure(s), got {len(figs)}"
                        )
                    for fig, baseline in zip(figs, resolved_baselines):
                        if remove_text:
                            _remove_figure_text(fig)
                        base = Path(baseline)
                        candidate_exts = (base.suffix.lstrip("."),) if base.suffix else extensions
                        for ext in candidate_exts:
                            baseline_path = baseline_dir / (
                                base.name if base.suffix else f"{baseline}.{ext}"
                            )
                            actual_path = Path(tmpdir) / baseline_path.name
                            figure_savefig_kwarg = dict(savefig_kwarg)
                            figure_savefig_kwarg.setdefault("transparent", False)
                            fig.savefig(actual_path, format=ext, **figure_savefig_kwarg)
                            result = compare_images(
                                baseline_path, actual_path, tol, in_decorator=True
                            )
                            if result is not None:
                                raise ImageComparisonFailure(str(result))
                    plt.close("all")
        return wrapper
    return decorator


def check_figures_equal(*, extensions=("png",), tol=0, **kwargs):
    extensions = tuple(extensions)
    allowed_chars = set(string.digits + string.ascii_letters + "_-[]()")
    keyword_only = inspect.Parameter.KEYWORD_ONLY

    def decorator(func):
        old_sig = inspect.signature(func)
        if not {"fig_test", "fig_ref"}.issubset(old_sig.parameters):
            raise ValueError(
                "The decorated function must have at least the "
                "parameters 'fig_test' and 'fig_ref', but your "
                f"function has the signature {old_sig}"
            )

        @pytest.mark.parametrize("ext", extensions)
        def wrapper(*args, ext, request, **kw):
            result_dir = _module_result_dir(func)
            with tempfile.TemporaryDirectory():
                with mpl.rc_context({"axes.prop_cycle": cycler(color=["b"])}):
                    with mpl_style.context({}):
                        plt.close("all")
                        fig_test = plt.figure("test")
                        fig_ref = plt.figure("reference")
                        call_kw = dict(kw)
                        if "ext" in old_sig.parameters:
                            call_kw["ext"] = ext
                        if "request" in old_sig.parameters:
                            call_kw["request"] = request
                        try:
                            with _collect_new_figures() as figs:
                                func(*args, fig_test=fig_test, fig_ref=fig_ref, **call_kw)
                            if figs:
                                raise RuntimeError(
                                    "Number of open figures changed during test. "
                                    "Plot to fig_test/fig_ref or close new figures explicitly."
                                )
                            file_name = "".join(
                                c for c in request.node.name if c in allowed_chars
                            )
                            actual = result_dir / f"{file_name}.{ext}"
                            expected = result_dir / f"{file_name}-expected.{ext}"
                            fig_test.savefig(actual, format=ext)
                            fig_ref.savefig(expected, format=ext)
                            _raise_on_image_difference(expected, actual, tol)
                        finally:
                            plt.close(fig_test)
                            plt.close(fig_ref)
        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__module__ = func.__module__
        wrapper.__doc__ = func.__doc__
        parameters = [
            param
            for param in old_sig.parameters.values()
            if param.name not in {"fig_test", "fig_ref"}
        ]
        if "ext" not in old_sig.parameters:
            parameters.append(inspect.Parameter("ext", keyword_only))
        if "request" not in old_sig.parameters:
            parameters.append(inspect.Parameter("request", keyword_only))
        wrapper.__signature__ = old_sig.replace(parameters=parameters)
        wrapper.pytestmark = getattr(func, "pytestmark", []) + getattr(wrapper, "pytestmark", [])
        return wrapper
    return decorator
