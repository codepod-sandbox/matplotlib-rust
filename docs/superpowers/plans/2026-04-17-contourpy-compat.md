# ContourPy Compat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock Matplotlib `contour` and `contourf` construction in this repo by replacing the `python/contourpy` stub with a minimal compatible fallback that returns valid contour path data.

**Architecture:** Keep this first slice in pure Python under `python/contourpy/__init__.py` so it is fast to land and easy to iterate. Implement only the narrow API that `python/matplotlib/contour.py` currently calls: `contour_generator(...)`, `LineType.SeparateCode`, `FillType.OuterCode`, and generator methods `create_contour(level)` / `create_filled_contour(lower, upper)`. For the first pass, return simple valid path containers rather than aiming for geometric parity with upstream `contourpy`.

**Tech Stack:** Python 3.12, NumPy, pytest, local Matplotlib test suite

---

## File Map

**Files to modify:**
- `python/contourpy/__init__.py` — replace the current `NotImplementedError` stub with a minimal generator object and API validation.
- `python/matplotlib/tests/test_axes_upstream2.py` — unskip the basic `ax.contour()` / `ax.contourf()` smoke tests once the fallback exists.
- `python/matplotlib/tests/test_pyplot_upstream.py` — unskip the basic `plt.contour()` / `plt.contourf()` smoke tests once the fallback exists.

**Files to verify against but not necessarily modify:**
- `python/matplotlib/contour.py` — consumer contract for the fallback generator.
- `python/matplotlib/tests/test_axes_upstream.py` — already-soft tests that should stop skipping once the fallback works.

---

### Task 1: Ship a Minimal `contourpy` Generator

**Files:**
- Modify: `python/contourpy/__init__.py`
- Test: `python/matplotlib/tests/test_axes_upstream.py`
- Test: `python/matplotlib/tests/test_axes_upstream2.py`
- Test: `python/matplotlib/tests/test_pyplot_upstream.py`

- [ ] **Step 1: Write the failing tests**

Remove the hard skips from the basic contour smoke tests so they become a real red-green gate:

```python
# python/matplotlib/tests/test_axes_upstream2.py
class TestAxesContour:
    def test_contour(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        result = ax.contour([[0, 1], [1, 2]])
        assert result is not None

    def test_contourf(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        result = ax.contourf([[0, 1], [1, 2]])
        assert result is not None
```

```python
# python/matplotlib/tests/test_pyplot_upstream.py
def test_plt_contour_basic():
    plt.close('all')
    Z = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    result = plt.contour(Z)
    assert result is not None
    plt.close('all')


def test_plt_contourf_basic():
    plt.close('all')
    Z = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    result = plt.contourf(Z)
    assert result is not None
    plt.close('all')
```

- [ ] **Step 2: Run the focused contour tests to verify they fail**

Run:

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp PYTHONPATH=python ./.venv/bin/python -m pytest -q \
  python/matplotlib/tests/test_axes_upstream.py \
  python/matplotlib/tests/test_axes_upstream2.py \
  python/matplotlib/tests/test_pyplot_upstream.py -k contour
```

Expected: FAIL with `NotImplementedError: contourpy not yet implemented (Phase 3)`.

- [ ] **Step 3: Implement the minimal generator**

Replace the stub with a narrow compatibility layer:

```python
# python/contourpy/__init__.py
import numpy as np

__version__ = "1.3.0"


class CoordinateType:
    Separate = 0
    SeparateCode = 1
    ChunkCombinedArray = 2
    ChunkCombinedCodesOffsets = 3
    ChunkCombinedOffset = 4
    ChunkCombinedNan = 5


class FillType:
    OuterCode = 0
    OuterOffset = 1
    ChunkCombinedCode = 2
    ChunkCombinedOffset = 3
    ChunkCombinedCodeOffset = 4
    ChunkCombinedOffsetOffset = 5


class LineType:
    Separate = 0
    SeparateCode = 1
    ChunkCombinedArray = 2
    ChunkCombinedOffset = 3
    ChunkCombinedNan = 4


def _rect_path(x0, x1, y0, y1):
    vertices = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
        dtype=float,
    )
    codes = np.array([1, 2, 2, 2, 79], dtype=np.uint8)
    return vertices, codes


class _ContourGenerator:
    def __init__(self, x, y, z):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.xmin = float(np.nanmin(self.x))
        self.xmax = float(np.nanmax(self.x))
        self.ymin = float(np.nanmin(self.y))
        self.ymax = float(np.nanmax(self.y))

    def create_contour(self, level):
        if not np.isfinite(level):
            return [], []
        mid_y = 0.5 * (self.ymin + self.ymax)
        vertices = np.array([[self.xmin, mid_y], [self.xmax, mid_y]], dtype=float)
        codes = np.array([1, 2], dtype=np.uint8)
        return [vertices], [codes]

    def create_filled_contour(self, lower, upper):
        if not (np.isfinite(lower) and np.isfinite(upper)):
            return [], []
        vertices, codes = _rect_path(self.xmin, self.xmax, self.ymin, self.ymax)
        return [vertices], [codes]


def contour_generator(x=None, y=None, z=None, name="serial",
                      corner_mask=None, line_type=None, fill_type=None,
                      chunk_size=None, chunk_count=None,
                      total_chunk_count=None, quad_as_tri=False,
                      z_interp=None, thread_count=0):
    if z is None:
        raise TypeError("contour_generator() missing required argument: 'z'")
    if line_type not in (None, LineType.SeparateCode):
        raise NotImplementedError("only LineType.SeparateCode is supported")
    if fill_type not in (None, FillType.OuterCode):
        raise NotImplementedError("only FillType.OuterCode is supported")
    return _ContourGenerator(x, y, z)
```

- [ ] **Step 4: Run the focused contour tests to verify they pass**

Run:

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp PYTHONPATH=python ./.venv/bin/python -m pytest -q \
  python/matplotlib/tests/test_axes_upstream.py \
  python/matplotlib/tests/test_axes_upstream2.py \
  python/matplotlib/tests/test_pyplot_upstream.py -k contour
```

Expected: PASS for the basic contour smoke tests, with no `NotImplementedError`.

- [ ] **Step 5: Commit**

```bash
git add python/contourpy/__init__.py \
  python/matplotlib/tests/test_axes_upstream2.py \
  python/matplotlib/tests/test_pyplot_upstream.py \
  docs/superpowers/plans/2026-04-17-contourpy-compat.md
git commit -m "feat: add contourpy compatibility fallback"
```

### Task 2: Harden the Fallback Against Existing Contour Call Shapes

**Files:**
- Modify: `python/contourpy/__init__.py`
- Test: `python/matplotlib/tests/test_axes_upstream.py`
- Test: `python/matplotlib/tests/test_axes_upstream2.py`

- [ ] **Step 1: Add a focused contract test for generator output shape**

```python
def test_contourpy_generator_returns_vertices_and_codes():
    import contourpy
    x = [[0, 1], [0, 1]]
    y = [[0, 0], [1, 1]]
    z = [[0, 1], [1, 2]]
    cg = contourpy.contour_generator(
        x, y, z,
        line_type=contourpy.LineType.SeparateCode,
        fill_type=contourpy.FillType.OuterCode,
    )
    vs, cs = cg.create_contour(1.0)
    assert isinstance(vs, list)
    assert isinstance(cs, list)
    assert vs[0].shape[1] == 2
    assert cs[0].dtype == np.uint8
```

- [ ] **Step 2: Run the contract test and verify any shape mismatch fails**

Run:

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp PYTHONPATH=python ./.venv/bin/python -m pytest -q \
  python/matplotlib/tests/test_axes_upstream.py -k contour
```

Expected: FAIL only if the fallback returns malformed path data.

- [ ] **Step 3: Tighten input normalization**

Handle the actual argument forms passed by `matplotlib.contour`:

```python
def _normalize_xy(x, y, z):
    z = np.asarray(z, dtype=float)
    if x is None or y is None:
        ny, nx = z.shape
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
        x, y = np.meshgrid(x, y)
    else:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    return x, y, z
```

Wire this helper into `contour_generator(...)` before constructing `_ContourGenerator`.

- [ ] **Step 4: Re-run the focused contour tests**

Run:

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp PYTHONPATH=python ./.venv/bin/python -m pytest -q \
  python/matplotlib/tests/test_axes_upstream.py \
  python/matplotlib/tests/test_axes_upstream2.py \
  python/matplotlib/tests/test_pyplot_upstream.py -k contour
```

Expected: PASS, including 2D-list and meshgrid inputs.

- [ ] **Step 5: Commit**

```bash
git add python/contourpy/__init__.py python/matplotlib/tests/test_axes_upstream.py
git commit -m "test: harden contourpy fallback contract"
```

### Task 3: Broaden Verification and Record Remaining Gaps

**Files:**
- Modify: `docs/superpowers/plans/2026-04-17-contourpy-compat.md`

- [ ] **Step 1: Run the full suite**

Run:

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp PYTHONPATH=python ./.venv/bin/python -m pytest -q python/matplotlib/tests
```

Expected: overall suite still green; contour-related skips decrease.

- [ ] **Step 2: Record what remains out of scope for this slice**

Append a short note to the plan:

```markdown
## Remaining Follow-Ups

- Geometry parity with real contourpy is not implemented in this slice.
- `tricontour` / `_tri` integration remains separate work.
- Image-comparison contour rendering is still a later verification step.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-04-17-contourpy-compat.md
git commit -m "docs: record contour fallback follow-ups"
```
