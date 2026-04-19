"""matplotlib.testing — test infrastructure."""

from __future__ import annotations

import importlib.util
import inspect
import locale
import logging
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
from matplotlib import _api

_log = logging.getLogger(__name__)


def set_font_settings_for_testing():
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["text.hinting"] = "none"
    mpl.rcParams["text.hinting_factor"] = 8


def set_reproducibility_for_testing():
    mpl.rcParams["svg.hashsalt"] = "matplotlib"


def setup():
    """Initialize matplotlib for testing."""
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, "English_United States.1252")
        except locale.Error:
            _log.warning(
                "Could not set locale to English/United States. "
                "Some date-related tests may fail."
            )

    mpl.use("Agg")
    with _api.suppress_matplotlib_deprecation_warning():
        mpl.rcdefaults()
    set_font_settings_for_testing()
    set_reproducibility_for_testing()


def subprocess_run_for_testing(
    command,
    env=None,
    timeout=180,
    stdout=None,
    stderr=None,
    check=False,
    text=True,
    capture_output=False,
):
    """
    Run a subprocess with the compatibility contract used by upstream tests.
    """
    if capture_output:
        stdout = stderr = subprocess.PIPE
    try:
        return subprocess.run(
            command,
            env=env,
            timeout=timeout,
            check=check,
            stdout=stdout,
            stderr=stderr,
            text=text,
        )
    except BlockingIOError:
        if sys.platform == "cygwin":
            import pytest

            pytest.xfail("Fork failure")
        raise


def subprocess_run_helper(func, *args, timeout=180, extra_env=None):
    """
    Run a source-backed helper function in a fresh Python subprocess.
    """
    source = inspect.getsourcefile(func)
    if source is None:
        raise ValueError("subprocess_run_helper requires a source-backed function")

    spec = importlib.util.spec_from_file_location(func.__module__, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not build import spec for {source!r}")

    env = {**os.environ, "SOURCE_DATE_EPOCH": "0", **(extra_env or {})}
    command = [
        sys.executable,
        "-c",
        (
            f"import importlib.util;"
            f"_spec=importlib.util.spec_from_file_location({func.__module__!r}, {source!r});"
            f"_module=importlib.util.module_from_spec(_spec);"
            f"_spec.loader.exec_module(_module);"
            f"_module.{func.__name__}()"
        ),
        *args,
    ]
    return subprocess_run_for_testing(
        command,
        env=env,
        timeout=timeout,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _check_for_pgf(texsystem):
    """
    Check whether a TeX system with pgf support is available.
    """
    with TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir, "test.tex")
        tex_path.write_text(
            r"""
            \documentclass{article}
            \usepackage{pgf}
            \begin{document}
            \typeout{pgfversion=\pgfversion}
            \makeatletter
            \@@end
        """,
            encoding="utf-8",
        )
        try:
            subprocess.check_call(
                [texsystem, "-halt-on-error", str(tex_path)],
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return False
        return True


def _has_tex_package(package):
    try:
        mpl.dviread.find_tex_file(f"{package}.sty")
        return True
    except FileNotFoundError:
        return False


def is_ci_environment():
    ci_environment_variables = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS",
        "GITLAB_CI",
        "GITHUB_ACTIONS",
        "TEAMCITY_VERSION",
    ]
    return any(os.getenv(env_var) for env_var in ci_environment_variables)


def _gen_multi_font_text():
    """
    Return a deterministic multi-font sample for fallback tests.

    ASCII glyphs should come from Computer Modern, while the non-ASCII glyphs
    are expected to fall back to DejaVu Sans.
    """
    fonts = ["cmr10", "DejaVu Sans"]
    test_str = "Hello World\nПривет мир"
    return fonts, test_str
