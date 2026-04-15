.PHONY: setup build build-ext build-path test test-cpython test-rustpython clean

# Platform detection: pick the right cargo cdylib suffix for the host
# and the right Python extension suffix for the destination. Keeps the
# repo build-clean on Linux/macOS/Windows.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  DYLIB_EXT := dylib
  DYLIB_PREFIX := lib
  PY_EXT := so
else ifeq ($(UNAME_S),Linux)
  DYLIB_EXT := so
  DYLIB_PREFIX := lib
  PY_EXT := so
else
  # MSYS / MinGW / Git Bash on Windows → uname reports MINGW64_NT-* etc.
  DYLIB_EXT := dll
  DYLIB_PREFIX :=
  PY_EXT := pyd
endif

PATH_OUT := python/matplotlib/_path.$(PY_EXT)
AGG_OUT  := python/matplotlib/backends/_backend_agg.$(PY_EXT)

# Pin PyO3 to the venv Python so the extension's ABI matches what
# pytest will import. Without this, cargo picks whatever `python3` is
# on PATH, which may be a different minor version than the venv and
# will fail at import with missing Py_* symbols.
#
# Note: in a git worktree, `--show-toplevel` gives the worktree root,
# not the main repo root. The venv lives in the main repo
# (`dirname .git-common-dir`). `make test` also reuses this path.
MAIN_REPO := $(shell dirname $$(git rev-parse --git-common-dir))
VENV_PYTHON := $(MAIN_REPO)/.venv/bin/python
export PYO3_PYTHON := $(VENV_PYTHON)

setup:
	git config core.hooksPath .githooks

# Build the RustPython interpreter binary
build:
	cargo build -p matplotlib-python

# Build CPython extensions. Currently:
#   - matplotlib-path: the committed _path.so is kept as-is. Rebuilding
#     it against the current numpy (2.4.3) + pyo3-numpy 0.25 introduces
#     a wave of "ndarray cannot be converted to PyArray<T, D>" failures
#     because the source in lib.rs hasn't been ported to the stricter
#     extraction rules. Tracked as a follow-up; see `build-path`.
#   - matplotlib-agg: fresh build on every `make test`.
build-ext:
	cargo build -p matplotlib-agg
	cp target/debug/$(DYLIB_PREFIX)matplotlib_agg.$(DYLIB_EXT) $(AGG_OUT)

# Explicit rebuild for _path.so (committed binary works, don't use
# this unless you're tackling the numpy 2.x ABI cleanup).
build-path:
	cargo build -p matplotlib-path
	cp target/debug/$(DYLIB_PREFIX)_path.$(DYLIB_EXT) $(PATH_OUT)

# CPython test run (primary dev loop)
test: build-ext
	$(VENV_PYTHON) -m pytest python/matplotlib/tests/

# RustPython test run (final compatibility check)
test-rustpython: build
	target/debug/matplotlib-python -m pytest python/matplotlib/tests/

clean:
	cargo clean
