"""matplotlib.testing — test infrastructure."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def setup():
    """Initialize matplotlib for testing."""
    pass


def subprocess_run_helper(func, *, timeout=60, extra_env=None):
    """
    Run a top-level helper function in a fresh Python subprocess.

    The helper is loaded from its defining source file via ``runpy.run_path``
    so test-local imports and globals are initialized exactly as in-process.
    """
    source = inspect.getsourcefile(func)
    if source is None:
        raise ValueError("subprocess_run_helper requires a source-backed function")

    source_path = Path(source).resolve()
    qualname = func.__qualname__
    script = textwrap.dedent(
        f"""
        import runpy
        namespace = runpy.run_path({str(source_path)!r}, run_name="__main__")
        obj = namespace[{qualname.split(".")[0]!r}]
        for part in {qualname.split(".")[1:]!r}:
            obj = getattr(obj, part)
        obj()
        """
    )

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(source_path.parent.parent.parent),
    )
