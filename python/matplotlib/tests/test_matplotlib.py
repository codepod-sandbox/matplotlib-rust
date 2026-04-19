import os
import subprocess
import sys
from unittest.mock import patch

import pytest

import matplotlib
from matplotlib.testing import subprocess_run_for_testing


@pytest.mark.parametrize(
    "version_str, version_tuple",
    [
        ("3.5.0", (3, 5, 0, "final", 0)),
        ("3.5.0rc2", (3, 5, 0, "candidate", 2)),
        ("3.5.0.dev820+g6768ef8c4c", (3, 5, 0, "alpha", 820)),
        ("3.5.0.post820+g6768ef8c4c", (3, 5, 1, "alpha", 820)),
    ],
)
def test_parse_to_version_info(version_str, version_tuple):
    assert matplotlib._parse_to_version_info(version_str) == version_tuple


@pytest.mark.skipif(sys.platform == "win32", reason="chmod() doesn't work as is on Windows")
@pytest.mark.skipif(
    sys.platform != "win32" and os.geteuid() == 0,
    reason="chmod() doesn't work as root",
)
def test_tmpconfigdir_warning(tmp_path):
    mode = os.stat(tmp_path).st_mode
    cache_home = tmp_path.parent / "cache-home"
    cache_home.mkdir(exist_ok=True)
    try:
        os.chmod(tmp_path, 0)
        try:
            proc = subprocess_run_for_testing(
                [sys.executable, "-c", "import matplotlib"],
                env={
                    **os.environ,
                    "MPLCONFIGDIR": str(tmp_path),
                    "XDG_CACHE_HOME": str(cache_home),
                },
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
        assert "set the MPLCONFIGDIR" in stderr
    finally:
        os.chmod(tmp_path, mode)


def test_importable_with_no_home():
    subprocess_run_for_testing(
        [
            sys.executable,
            "-c",
            "import pathlib; pathlib.Path.home = lambda *args: 1/0; "
            "import matplotlib.pyplot",
        ],
        env={**os.environ, "MPLCONFIGDIR": matplotlib.get_configdir()},
        check=True,
    )


def test_use_doc_standard_backends():
    def parse(key):
        backends = []
        for line in matplotlib.use.__doc__.split(key)[1].split("\n"):
            if not line.strip():
                break
            backends += [entry.strip().lower() for entry in line.split(",") if entry]
        return backends

    from matplotlib.backends import BackendFilter, backend_registry

    assert set(parse("- interactive backends:\n")) == set(
        backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    )
    assert set(parse("- non-interactive backends:\n")) == set(
        backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
    )


def test_importable_with__OO():
    program = (
        "import matplotlib as mpl; "
        "import matplotlib.pyplot as plt; "
        "import matplotlib.cbook as cbook; "
        "import matplotlib.patches as mpatches"
    )
    subprocess_run_for_testing(
        [sys.executable, "-OO", "-c", program],
        env={**os.environ, "MPLBACKEND": ""},
        check=True,
    )


@patch("matplotlib.subprocess.check_output")
def test_get_executable_info_timeout(mock_check_output):
    mock_check_output.side_effect = subprocess.TimeoutExpired(cmd=["mock"], timeout=30)
    with pytest.raises(matplotlib.ExecutableNotFoundError, match="Timed out"):
        matplotlib._get_executable_info.__wrapped__("inkscape")
