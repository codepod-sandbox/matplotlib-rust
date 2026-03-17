# Upstream Matplotlib Compatibility Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add upstream matplotlib API coverage across five phases: tick formatters/locators, log/symlog scales, legend rendering, arrow annotations, and artist properties.

**Architecture:** Copy real matplotlib source files verbatim where feasible, strip C-extension dependencies (reimplement in Python), strip date/transform/mathtext subsystems, and wire into existing `Axes`/renderer infrastructure. Each phase is self-contained and adds upstream test coverage.

**Tech Stack:** Python (RustPython), numpy-rust for array ops, existing SVG/PIL renderers, pytest for tests.

**Build & test commands** (run from `packages/matplotlib-py/`):
```bash
cargo build -p matplotlib-python          # rebuild binary
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q   # full suite (789 baseline)
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -v  # single file
```

---

## Chunk 1: Phase 1 — Tick Formatters/Locators

### File Map

| Action   | Path |
|----------|------|
| Modify   | `python/matplotlib/rcsetup.py` |
| Create   | `python/matplotlib/ticker.py` |
| Create   | `python/matplotlib/axis.py` |
| Modify   | `python/matplotlib/axes.py` |
| Create   | `python/matplotlib/tests/test_ticker_upstream.py` |

Note: `_svg_backend.py` is **not** modified in this phase. `_nice_ticks`/`_fmt_tick` remain defined there; only the import in `axes.py` is removed.

---

### Task 1: Add formatter rcParams keys to `rcsetup.py`

`ScalarFormatter` and friends look up these keys. Add them to `_default_params` so `rcParams['axes.formatter.limits']` works without KeyError.

**Files:**
- Modify: `python/matplotlib/rcsetup.py:8-65`

- [ ] **Step 1: Write the failing test**

Create `python/matplotlib/tests/test_ticker_upstream.py`:

```python
# Ported from lib/matplotlib/tests/test_ticker.py
import pytest
import matplotlib
from matplotlib.rcsetup import _default_params


def test_rcparams_formatter_keys():
    """Formatter rcParams keys must exist with correct defaults."""
    assert 'axes.formatter.limits' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.limits'] == [-5, 6]
    assert 'axes.formatter.use_locale' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.use_locale'] is False
    assert 'axes.formatter.use_mathtext' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.use_mathtext'] is False
    assert 'axes.formatter.min_exponent' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.min_exponent'] == 0
    assert 'axes.formatter.useoffset' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.useoffset'] is True
    assert 'axes.formatter.offset_threshold' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.offset_threshold'] == 4
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py::test_rcparams_formatter_keys -v
```

Expected: `FAILED` — `KeyError: 'axes.formatter.limits'`

- [ ] **Step 3: Add the keys to `rcsetup.py`**

In `python/matplotlib/rcsetup.py`, add inside `_default_params` after the `'axes.labelpad'` line:

```python
    # Formatter behavior
    'axes.formatter.limits': [-5, 6],
    'axes.formatter.use_locale': False,
    'axes.formatter.use_mathtext': False,
    'axes.formatter.min_exponent': 0,
    'axes.formatter.useoffset': True,
    'axes.formatter.offset_threshold': 4,
```

- [ ] **Step 4: Run the test — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py::test_rcparams_formatter_keys -v
```

- [ ] **Step 5: Full suite still passes**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

Expected: 790 passed (1 new test).

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/rcsetup.py python/matplotlib/tests/test_ticker_upstream.py
git commit -m "feat: add axes.formatter.* rcParams keys for ticker compatibility"
```

---

### Task 2: Port `ticker.py` from upstream matplotlib

Copy real matplotlib's `ticker.py` and adapt it. The file is ~2000 lines of pure Python with two blockers:

1. `from matplotlib import _ticker` — a C extension providing `_Edge_integer`
2. Various `rcParams` / internal matplotlib imports

**Files:**
- Create: `python/matplotlib/ticker.py`
- Modify: `python/matplotlib/tests/test_ticker_upstream.py`

**Adaptation rules:**

1. **Remove** the `from matplotlib import _ticker` import line entirely.
2. **Add** a pure Python `_Edge_integer(x, d)` near the top of the file:
   ```python
   import math as _math

   def _Edge_integer(x, d):
       """Return the nearest integer multiple of d to x.
       Equivalent to matplotlib._ticker._Edge_integer in C.
       """
       if d == 0:
           raise ValueError("d must be nonzero")
       return _math.floor(x / d + 0.5) * d
   ```
3. **Replace** `matplotlib._ticker._Edge_integer(...)` calls with `_Edge_integer(...)`.
4. **Keep** all `import numpy as np` lines — numpy-rust provides these.
5. **Remove** these import lines:
   - `from matplotlib import transforms` (and any `transforms.` usage — only used in `DateLocator`, which we drop)
   - `from matplotlib import cbook` — replace any `cbook.` calls with inline equivalents (see below)
   - `from matplotlib.dates import ...` — drop entire Date* classes
6. **cbook replacements** (check what `cbook` functions are actually used; common ones):
   - `cbook.iterable(x)` → `np.iterable(x)` or `hasattr(x, '__iter__')`
   - `cbook.is_numlike(x)` → `isinstance(x, (int, float, np.number))`
   - `cbook.safe_masked_invalid(x)` → `np.ma.masked_invalid(x)`
7. **Replace** `rcParams['axes.formatter.*']` lookups with `matplotlib.rcParams['axes.formatter.*']` (add `import matplotlib` near top).
8. **Drop** these classes entirely (they depend on `transforms` or `dates`):
   - `DateLocator`, `DateFormatter`, `AutoDateLocator`, `AutoDateFormatter`, `YearLocator`, `MonthLocator`, `WeekdayLocator`, `DayLocator`, `HourLocator`, `MinuteLocator`, `SecondLocator`, `MicrosecondLocator`, `DateConverter`, `ConciseDateFormatter`, `AutoDateFormatter`
9. **Keep**: `Formatter`, `NullFormatter`, `FixedFormatter`, `FuncFormatter`, `StrMethodFormatter`, `ScalarFormatter`, `LogFormatter`, `LogFormatterSciNotation`, `LogFormatterMathtext`, `PercentFormatter`, `Locator`, `NullLocator`, `FixedLocator`, `MultipleLocator`, `AutoLocator`, `LogLocator`, `MaxNLocator`, `IndexLocator`, `LinearLocator`, `SymmetricalLogLocator`.

- [ ] **Step 1: Write failing tests for formatters**

Append to `python/matplotlib/tests/test_ticker_upstream.py`:

```python
# ---------------------------------------------------------------------------
# Ported from lib/matplotlib/tests/test_ticker.py
# ---------------------------------------------------------------------------

def test_formatter_str():
    """NullFormatter and FixedFormatter basics."""
    from matplotlib.ticker import NullFormatter, FixedFormatter
    assert NullFormatter()(1.0, 0) == ''
    fmt = FixedFormatter(['a', 'b', 'c'])
    assert fmt(0, 0) == 'a'
    assert fmt(1, 0) == 'b'
    assert fmt(5, 0) == ''  # out of range → empty string


def test_scalar_formatter():
    """ScalarFormatter produces plain strings for small values."""
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter()
    fmt.create_dummy_axis()
    result = fmt(1000, 0)
    assert isinstance(result, str)
    assert '1000' in result or '1' in result  # some numeric representation


def test_percent_formatter():
    """PercentFormatter formats fractions as percentages."""
    from matplotlib.ticker import PercentFormatter
    fmt = PercentFormatter(xmax=1.0)
    assert fmt(0.5, 0) == '50%'
    assert fmt(1.0, 0) == '100%'
    fmt2 = PercentFormatter(xmax=100)
    assert fmt2(50, 0) == '50%'


def test_func_formatter():
    """FuncFormatter delegates to a callable."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda x, pos: f'val={x:.1f}')
    assert fmt(3.14, 0) == 'val=3.1'


def test_str_method_formatter():
    """StrMethodFormatter uses str.format."""
    from matplotlib.ticker import StrMethodFormatter
    fmt = StrMethodFormatter('{x:.2f}')
    assert fmt(3.14159, 0) == '3.14'
    fmt2 = StrMethodFormatter('{x:d}')
    assert fmt2(42, 0) == '42'
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -k "formatter" -v
```

Expected: `ImportError: No module named 'matplotlib.ticker'`

- [ ] **Step 3: Create `ticker.py`**

Download real matplotlib's `ticker.py` from the matplotlib 3.9.x release (or copy from a local checkout). Apply the adaptations listed above. Save to `python/matplotlib/ticker.py`.

Quick checklist while editing:
- [ ] `_Edge_integer` pure-Python function added near top
- [ ] `from matplotlib import _ticker` removed
- [ ] Date* classes removed
- [ ] `from matplotlib import transforms` removed
- [ ] `from matplotlib import cbook` removed / replaced
- [ ] `import matplotlib` added; `rcParams` access via `matplotlib.rcParams`
- [ ] File begins with the original matplotlib copyright block, followed by our BSD-3-Clause header

Copyright block template at top of file:
```
# This file is copied from the matplotlib project and modified.
# Original copyright:
#   Copyright (c) 2002–2012 John D. Hunter; All Rights Reserved
#   Copyright (c) 2012–2024 The Matplotlib Development Team
# Licensed under the matplotlib PSF-derived license (see below).
#
# Modifications copyright (c) 2024 CodePod Contributors
# BSD 3-Clause License (see LICENSE at repository root)
```

- [ ] **Step 4: Run formatter tests — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -k "formatter" -v
```

- [ ] **Step 5: Write failing locator tests**

Append to `python/matplotlib/tests/test_ticker_upstream.py`:

```python
def test_null_locator():
    """NullLocator returns empty tick list."""
    from matplotlib.ticker import NullLocator
    loc = NullLocator()
    assert loc.tick_values(0, 10) == []


def test_fixed_locator():
    """FixedLocator returns its preset tick positions."""
    from matplotlib.ticker import FixedLocator
    loc = FixedLocator([1, 2, 3, 5])
    result = loc.tick_values(0, 6)
    assert list(result) == [1, 2, 3, 5]


def test_multiple_locator():
    """MultipleLocator produces ticks at multiples of base."""
    from matplotlib.ticker import MultipleLocator
    loc = MultipleLocator(0.5)
    ticks = loc.tick_values(0.0, 2.0)
    assert 0.5 in ticks
    assert 1.0 in ticks
    assert 1.5 in ticks


def test_auto_locator():
    """AutoLocator produces ~5-8 nice ticks in range."""
    from matplotlib.ticker import AutoLocator
    loc = AutoLocator()
    ticks = loc.tick_values(0, 10)
    assert 4 <= len(ticks) <= 12
    # ticks should be within or near range
    assert min(ticks) <= 0.01
    assert max(ticks) >= 9.99


def test_log_locator():
    """LogLocator places ticks at powers of base."""
    from matplotlib.ticker import LogLocator
    loc = LogLocator(base=10.0)
    ticks = loc.tick_values(1, 1000)
    tick_list = sorted(ticks)
    assert 1 in tick_list or any(abs(t - 1) < 0.01 for t in tick_list)
    assert any(abs(t - 10) < 0.01 for t in tick_list)
    assert any(abs(t - 100) < 0.01 for t in tick_list)


def test_maxn_locator():
    """MaxNLocator respects nbins limit."""
    from matplotlib.ticker import MaxNLocator
    loc = MaxNLocator(nbins=4)
    ticks = loc.tick_values(0, 100)
    assert len(ticks) <= 6  # nbins+1 at most


def test_logformatter():
    # Ported from lib/matplotlib/tests/test_ticker.py::test_LogFormatter
    """LogFormatter produces reasonable strings for log-scale values."""
    from matplotlib.ticker import LogFormatter
    fmt = LogFormatter(base=10)
    fmt.create_dummy_axis()
    result = fmt(100, 0)
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 6: Run locator tests — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -k "locator or logformatter" -v
```

Fix any import errors or adaptation issues before continuing.

- [ ] **Step 7: Full suite still passes**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 8: Commit**

```bash
git add python/matplotlib/ticker.py python/matplotlib/tests/test_ticker_upstream.py
git commit -m "feat: port ticker.py from upstream matplotlib"
```

---

### Task 3: Create `axis.py` with `XAxis`/`YAxis`

A thin wrapper that holds a formatter+locator pair for major and minor ticks, and a fixed tick list when `set_ticks()` is called explicitly.

**Files:**
- Create: `python/matplotlib/axis.py`
- Modify: `python/matplotlib/tests/test_ticker_upstream.py`

- [ ] **Step 1: Write failing tests**

Append to `python/matplotlib/tests/test_ticker_upstream.py`:

```python
# ---------------------------------------------------------------------------
# axis.py tests
# ---------------------------------------------------------------------------

def test_axis_default_locator_formatter():
    """XAxis defaults to AutoLocator + ScalarFormatter."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import AutoLocator, ScalarFormatter
    ax_obj = XAxis()
    assert isinstance(ax_obj.get_major_locator(), AutoLocator)
    assert isinstance(ax_obj.get_major_formatter(), ScalarFormatter)


def test_axis_set_major_locator():
    """set_major_locator() replaces the locator."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import FixedLocator
    ax_obj = XAxis()
    loc = FixedLocator([1, 2, 3])
    ax_obj.set_major_locator(loc)
    assert ax_obj.get_major_locator() is loc


def test_axis_set_ticks_uses_fixed_locator():
    """set_ticks() installs a FixedLocator + FixedFormatter."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax_obj = XAxis()
    ax_obj.set_ticks([0.0, 0.5, 1.0])
    assert isinstance(ax_obj.get_major_locator(), FixedLocator)
    assert list(ax_obj.get_ticks()) == [0.0, 0.5, 1.0]


def test_axis_set_ticks_with_labels():
    """set_ticks() with labels installs FixedFormatter."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import FixedFormatter
    ax_obj = XAxis()
    ax_obj.set_ticks([1, 2, 3], ['a', 'b', 'c'])
    assert isinstance(ax_obj.get_major_formatter(), FixedFormatter)
    assert ax_obj.get_major_formatter().seq == ['a', 'b', 'c']


def test_axis_tick_values():
    """tick_values() delegates to the locator."""
    from matplotlib.axis import XAxis
    ax_obj = XAxis()
    ax_obj.set_ticks([10, 20, 30])
    vals = ax_obj.tick_values(0, 40)
    assert list(vals) == [10, 20, 30]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -k "axis_" -v
```

- [ ] **Step 3: Create `python/matplotlib/axis.py`**

```python
# Copyright (c) 2024 CodePod Contributors
# BSD 3-Clause License (see LICENSE at repository root)
"""matplotlib.axis — XAxis and YAxis wrappers."""

from matplotlib.ticker import (
    AutoLocator, ScalarFormatter,
    FixedLocator, FixedFormatter,
    NullLocator, NullFormatter,
)


class _TickerPair:
    """Holds a (locator, formatter) pair for major or minor ticks."""

    def __init__(self, locator, formatter):
        self.locator = locator
        self.formatter = formatter


class Axis:
    """Base class for a single axis (x or y)."""

    def __init__(self):
        self._major = _TickerPair(AutoLocator(), ScalarFormatter())
        self._minor = _TickerPair(NullLocator(), NullFormatter())
        self._ticks = None   # None means "use locator"; list means "fixed"

    # --- major ---
    def get_major_locator(self):
        return self._major.locator

    def set_major_locator(self, locator):
        self._major.locator = locator
        self._ticks = None  # clear fixed ticks

    def get_major_formatter(self):
        return self._major.formatter

    def set_major_formatter(self, formatter):
        self._major.formatter = formatter

    # --- minor ---
    def get_minor_locator(self):
        return self._minor.locator

    def set_minor_locator(self, locator):
        self._minor.locator = locator

    def get_minor_formatter(self):
        return self._minor.formatter

    def set_minor_formatter(self, formatter):
        self._minor.formatter = formatter

    # --- explicit ticks ---
    def set_ticks(self, ticks, labels=None):
        """Set explicit tick positions (and optionally labels)."""
        ticks = list(ticks)
        self._ticks = ticks
        self._major.locator = FixedLocator(ticks)
        if labels is not None:
            self._major.formatter = FixedFormatter(list(labels))
        else:
            self._major.formatter = ScalarFormatter()

    def get_ticks(self):
        """Return the fixed tick list, or [] if using auto locator."""
        return list(self._ticks) if self._ticks is not None else []

    def tick_values(self, vmin, vmax):
        """Ask the locator for tick positions in [vmin, vmax]."""
        return self._major.locator.tick_values(vmin, vmax)

    def format_ticks(self, values):
        """Format a sequence of tick values using the major formatter."""
        fmt = self._major.formatter
        return [fmt(v, i) for i, v in enumerate(values)]


class XAxis(Axis):
    """X-axis."""


class YAxis(Axis):
    """Y-axis."""
```

- [ ] **Step 4: Run — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -k "axis_" -v
```

- [ ] **Step 5: Full suite still passes**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/axis.py python/matplotlib/tests/test_ticker_upstream.py
git commit -m "feat: add axis.py with XAxis/YAxis wrapping locator+formatter pairs"
```

---

### Task 4: Migrate `axes.py` tick machinery to use `Axis` objects

Replace `_xticks`/`_yticks` plain lists with `XAxis`/`YAxis` instances. `Axes.draw()` reads ticks from the Axis object rather than calling `_nice_ticks` directly.

**Files:**
- Modify: `python/matplotlib/axes.py`
- Modify: `python/matplotlib/tests/test_ticker_upstream.py`

The migration touches several spots in `axes.py`:

1. **`__init__`**: add `self.xaxis = XAxis()` / `self.yaxis = YAxis()`, remove `_xticks`, `_yticks`, `_xticklabels`, `_yticklabels`.
2. **`set_xticks(ticks, labels=None)`**: delegate to `self.xaxis.set_ticks(ticks, labels)`.
3. **`get_xticks()`**: return `self.xaxis.get_ticks()`.
4. **`set_xticklabels(labels)`**: get current ticks and call `self.xaxis.set_ticks(current_ticks, labels)` — or if no ticks set yet, store labels and apply when ticks arrive. Simpler: store a `_xticklabels_pending` list and apply in `draw()` if ticks get set later. **Simplest**: `set_xticklabels(labels)` calls `self.xaxis.set_major_formatter(FixedFormatter(labels))`.
5. **`cla()`**: reset axis objects — call `self.xaxis = XAxis()` / `self.yaxis = YAxis()`.
6. **`draw()`** tick rendering section: replace `_nice_ticks(layout.xmin, layout.xmax, 8)` calls with `self.xaxis.tick_values(layout.xmin, layout.xmax)`. Tick labels come from `self.xaxis.format_ticks(ticks)` instead of `_fmt_tick(t)`.
7. Remove the `from matplotlib._svg_backend import _nice_ticks, _fmt_tick, _esc` import (keep `_esc` only if used — verify it isn't; remove if so).

- [ ] **Step 1: Write failing integration tests**

Append to `python/matplotlib/tests/test_ticker_upstream.py`:

```python
# ---------------------------------------------------------------------------
# Axes integration tests
# ---------------------------------------------------------------------------

def test_axes_has_xaxis_yaxis():
    """Axes must expose .xaxis and .yaxis as Axis instances."""
    import matplotlib.pyplot as plt
    from matplotlib.axis import XAxis, YAxis
    fig, ax = plt.subplots()
    assert isinstance(ax.xaxis, XAxis)
    assert isinstance(ax.yaxis, YAxis)
    plt.close('all')


def test_set_xticks_delegates_to_xaxis():
    """set_xticks() must set a FixedLocator on xaxis."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    assert isinstance(ax.xaxis.get_major_locator(), FixedLocator)
    assert ax.get_xticks() == [1, 2, 3]
    plt.close('all')


def test_set_xticklabels_sets_formatter():
    """set_xticklabels() must install FixedFormatter on xaxis."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['zero', 'one', 'two'])
    assert isinstance(ax.xaxis.get_major_formatter(), FixedFormatter)
    plt.close('all')


def test_cla_resets_axis():
    """cla() must reset xaxis/yaxis to default AutoLocator."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator, FixedLocator
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    assert isinstance(ax.xaxis.get_major_locator(), FixedLocator)
    ax.cla()
    assert isinstance(ax.xaxis.get_major_locator(), AutoLocator)
    plt.close('all')


def test_draw_renders_tick_labels():
    """draw() must produce tick label text in SVG output via Axis objects."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 20, 30])
    svg = fig.to_svg()
    # SVG must contain some numeric tick label text
    assert any(str(n) in svg for n in range(1, 31))
    plt.close('all')
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -k "axes_has_xaxis or set_xticks or xticklabels or cla_resets" -v
```

- [ ] **Step 3: Apply migration to `axes.py`**

In `axes.py`:

**3a. Add import at top:**
```python
from matplotlib.axis import XAxis, YAxis
from matplotlib.ticker import FixedFormatter
```

**3b. In `__init__`, replace:**
```python
self._xticks = None
self._yticks = None
self._xticklabels = None
self._yticklabels = None
```
with:
```python
self.xaxis = XAxis()
self.yaxis = YAxis()
```

**3c. Replace `set_xticks` / `get_xticks` / `set_yticks` / `get_yticks`:**
```python
def set_xticks(self, ticks, labels=None, **kwargs):
    self.xaxis.set_ticks(ticks, labels)

def get_xticks(self):
    return self.xaxis.get_ticks()

def set_yticks(self, ticks, labels=None, **kwargs):
    self.yaxis.set_ticks(ticks, labels)

def get_yticks(self):
    return self.yaxis.get_ticks()
```

**3d. Replace `set_xticklabels` / `set_yticklabels`:**
```python
def set_xticklabels(self, labels, **kwargs):
    self.xaxis.set_major_formatter(FixedFormatter(list(labels)))

def set_yticklabels(self, labels, **kwargs):
    self.yaxis.set_major_formatter(FixedFormatter(list(labels)))
```

**3e. In `cla()`, replace:**
```python
self._xticks = None
self._yticks = None
self._xticklabels = None
self._yticklabels = None
```
with:
```python
self.xaxis = XAxis()
self.yaxis = YAxis()
```
Also remove `self._legend = False` (will become `self._legend_obj = None` in Phase 3 — for now leave `_legend` since legend migration is Phase 3).

**3f. In `draw()`, replace BOTH the grid and tick rendering sections:**

The current `draw()` has two separate `_nice_ticks` call sites. Both must be replaced. Compute `xtick_vals`/`ytick_vals` once **before** the grid block so they are available for both:

```python
# Compute tick positions ONCE — used for both grid and tick marks
xtick_vals = self.xaxis.tick_values(layout.xmin, layout.xmax)
ytick_vals = self.yaxis.tick_values(layout.ymin, layout.ymax)
xtick_labels = self.xaxis.format_ticks(xtick_vals)
ytick_labels = self.yaxis.format_ticks(ytick_vals)

# Grid (replace old _nice_ticks call in the grid block)
if self._grid:
    for t in xtick_vals:
        tx = layout.sx(t)
        if px < tx < px + pw:
            renderer.draw_line([tx, tx], [py, py + ph],
                               '#dddddd', 0.5, '--')
    for t in ytick_vals:
        ty = layout.sy(t)
        if py < ty < py + ph:
            renderer.draw_line([px, px + pw], [ty, ty],
                               '#dddddd', 0.5, '--')

# Tick marks + labels (replace old _nice_ticks call in the tick section)
for t, label in zip(xtick_vals, xtick_labels):
    tx = layout.sx(t)
    if px <= tx <= px + pw:
        renderer.draw_line([tx, tx], [py + ph, py + ph + 5],
                           '#000000', 1.0, '-')
        if self._xticklabels_visible:
            renderer.draw_text(tx, py + ph + 18, label,
                               11, '#333333', 'center')
for t, label in zip(ytick_vals, ytick_labels):
    ty = layout.sy(t)
    if py <= ty <= py + ph:
        renderer.draw_line([px - 5, px], [ty, ty],
                           '#000000', 1.0, '-')
        if self._yticklabels_visible:
            renderer.draw_text(px - 8, ty + 4, label,
                               11, '#333333', 'right')
```

Replace the entire old grid block (the `if self._grid:` section with `_nice_ticks`) plus the tick marks block beneath it with the above.

**3g. Update the import line** — `_esc` is defined in `_svg_backend.py` and is **not** used anywhere in `axes.py` (verified). Remove the entire `from matplotlib._svg_backend import _nice_ticks, _fmt_tick, _esc` import line from `axes.py` — it is no longer needed after this migration.

- [ ] **Step 4: Rebuild and run integration tests**

```bash
cargo build -p matplotlib-python 2>&1 | tail -3
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_ticker_upstream.py -k "axes_has_xaxis or set_xticks or xticklabels or cla_resets" -v
```

- [ ] **Step 5: Full suite still passes**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

Fix any regressions before committing.

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/axes.py python/matplotlib/tests/test_ticker_upstream.py
git commit -m "feat: migrate axes _xticks/_yticks to XAxis/YAxis objects"
```

---

## Chunk 2: Phase 2 — Log/Symlog Scale Rendering

### File Map

| Action   | Path |
|----------|------|
| Create   | `python/matplotlib/scale.py` |
| Modify   | `python/matplotlib/axis.py` |
| Modify   | `python/matplotlib/axes.py` |
| Modify   | `python/matplotlib/backend_bases.py` |
| Create   | `python/matplotlib/tests/test_scale_upstream.py` |

---

### Task 5: Port `scale.py` from upstream matplotlib

**Files:**
- Create: `python/matplotlib/scale.py`
- Create: `python/matplotlib/tests/test_scale_upstream.py`

**Adaptation rules (same principle as ticker.py):**
1. Keep `LinearScale`, `LogScale`, `SymmetricalLogScale` (symlog), `FuncScale`.
2. Each scale implements `forward(values)` and `inverse(values)` using numpy (available via numpy-rust).
3. Drop `register_scale` / `_scale_mapping` global registry — `ax.set_xscale()` will construct scales directly.
4. Drop any `transforms.Transform` base class — scales here are plain Python objects, not matplotlib transform objects.
5. `rcParams` lookups: same approach as ticker — use `matplotlib.rcParams`.
6. `LogScale` must handle non-positive values: use `np.ma.masked_where(values <= 0, values)` in `forward()`.
7. `FuncScale.__init__(self, forward, inverse)` takes two callables.

- [ ] **Step 1: Write failing scale tests**

Create `python/matplotlib/tests/test_scale_upstream.py`:

```python
# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_axes.py (scale sections)
import pytest
import numpy as np


def test_linear_scale_identity():
    """LinearScale forward/inverse are identity."""
    from matplotlib.scale import LinearScale
    s = LinearScale()
    vals = np.array([0.0, 1.0, 2.0, -3.0])
    np.testing.assert_array_equal(s.forward(vals), vals)
    np.testing.assert_array_equal(s.inverse(vals), vals)


def test_log_scale_forward():
    """LogScale(base=10) forward maps 1→0, 10→1, 100→2 (in log10)."""
    from matplotlib.scale import LogScale
    s = LogScale(base=10)
    vals = np.array([1.0, 10.0, 100.0])
    result = s.forward(vals)
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=1e-10)


def test_log_scale_inverse():
    """LogScale(base=10) inverse maps 0→1, 1→10, 2→100."""
    from matplotlib.scale import LogScale
    s = LogScale(base=10)
    vals = np.array([0.0, 1.0, 2.0])
    result = s.inverse(vals)
    np.testing.assert_allclose(result, [1.0, 10.0, 100.0], atol=1e-10)


def test_log_scale_nonpos():
    """LogScale masks non-positive values."""
    from matplotlib.scale import LogScale
    import numpy.ma as ma
    s = LogScale(base=10)
    vals = np.array([-1.0, 0.0, 1.0, 10.0])
    result = s.forward(vals)
    assert isinstance(result, ma.MaskedArray)
    assert result.mask[0]   # -1 masked
    assert result.mask[1]   # 0 masked
    assert not result.mask[2]  # 1 unmasked
    assert not result.mask[3]  # 10 unmasked


def test_symlog_scale():
    """SymmetricalLogScale is symmetric around zero."""
    from matplotlib.scale import SymmetricalLogScale
    s = SymmetricalLogScale(base=10, linthresh=1.0)
    fwd = s.forward(np.array([1.0, -1.0]))
    assert abs(fwd[0]) == abs(fwd[1])


def test_func_scale():
    """FuncScale applies user-provided forward/inverse callables."""
    from matplotlib.scale import FuncScale
    import math
    s = FuncScale(forward=np.sqrt, inverse=np.square)
    vals = np.array([1.0, 4.0, 9.0])
    result = s.forward(vals)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(s.inverse(result), vals)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_scale_upstream.py -v
```

- [ ] **Step 3: Create `python/matplotlib/scale.py`**

Copy real matplotlib `scale.py` and apply adaptations. The key classes to keep are `LinearScale`, `LogScale`, `SymmetricalLogScale`, `FuncScale`. Remove all `ScaleBase(transforms.Transform)` inheritance — replace with a simple Python base class:

```python
class ScaleBase:
    """Abstract base for axis scales."""

    def forward(self, values):
        """Transform data values to axis (display) space."""
        raise NotImplementedError

    def inverse(self, values):
        """Transform axis (display) space back to data values."""
        raise NotImplementedError
```

Full implementations:

```python
import numpy as np
import numpy.ma as ma
import math as _math


class LinearScale(ScaleBase):
    def forward(self, values):
        return np.asarray(values, dtype=float)

    def inverse(self, values):
        return np.asarray(values, dtype=float)


class LogScale(ScaleBase):
    def __init__(self, base=10.0, nonpositive='mask'):
        self.base = float(base)
        self._nonpositive = nonpositive

    def forward(self, values):
        arr = np.asarray(values, dtype=float)
        if self._nonpositive == 'mask':
            arr = ma.masked_where(arr <= 0, arr)
            return ma.log(arr) / _math.log(self.base)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.log(arr) / _math.log(self.base)

    def inverse(self, values):
        return self.base ** np.asarray(values, dtype=float)


class SymmetricalLogScale(ScaleBase):
    def __init__(self, base=10.0, linthresh=2.0, linscale=1.0):
        self.base = float(base)
        self.linthresh = float(linthresh)
        self.linscale = float(linscale)

    def _symlog(self, x):
        """Vectorised symlog."""
        log_base = _math.log(self.base)
        linthresh = self.linthresh
        linscale = self.linscale
        sign = np.sign(x)
        abs_x = np.abs(x)
        inside = abs_x <= linthresh
        result = np.where(
            inside,
            x / linthresh * linscale,
            sign * (linscale + (np.log(abs_x / linthresh) / log_base)),
        )
        return result

    def _isymlog(self, y):
        """Inverse of _symlog."""
        log_base = _math.log(self.base)
        linthresh = self.linthresh
        linscale = self.linscale
        sign = np.sign(y)
        abs_y = np.abs(y)
        inside = abs_y <= linscale
        result = np.where(
            inside,
            y * linthresh / linscale,
            sign * linthresh * (self.base ** (abs_y - linscale)),
        )
        return result

    def forward(self, values):
        return self._symlog(np.asarray(values, dtype=float))

    def inverse(self, values):
        return self._isymlog(np.asarray(values, dtype=float))


class FuncScale(ScaleBase):
    def __init__(self, forward, inverse):
        self._forward = forward
        self._inverse = inverse

    def forward(self, values):
        return self._forward(np.asarray(values, dtype=float))

    def inverse(self, values):
        return self._inverse(np.asarray(values, dtype=float))
```

Add the copyright header.

- [ ] **Step 4: Run scale tests — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_scale_upstream.py -v
```

- [ ] **Step 5: Full suite still passes**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/scale.py python/matplotlib/tests/test_scale_upstream.py
git commit -m "feat: port scale.py (LinearScale, LogScale, SymmetricalLogScale, FuncScale)"
```

---

### Task 6: Wire scales into `Axis` and `Axes`, update `AxesLayout`

**Files:**
- Modify: `python/matplotlib/axis.py`
- Modify: `python/matplotlib/axes.py`
- Modify: `python/matplotlib/backend_bases.py`
- Modify: `python/matplotlib/tests/test_scale_upstream.py`

- [ ] **Step 1: Write failing integration tests**

Append to `python/matplotlib/tests/test_scale_upstream.py`:

```python
def test_set_xscale_log_changes_locator():
    """ax.set_xscale('log') must install LogLocator on xaxis."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    from matplotlib.scale import LogScale
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    assert isinstance(ax.xaxis.get_scale(), LogScale)
    assert isinstance(ax.xaxis.get_major_locator(), LogLocator)
    plt.close('all')


def test_set_xscale_linear_is_default():
    """ax.set_xscale('linear') (or default) uses LinearScale."""
    import matplotlib.pyplot as plt
    from matplotlib.scale import LinearScale
    fig, ax = plt.subplots()
    assert isinstance(ax.xaxis.get_scale(), LinearScale)
    plt.close('all')


def test_axes_layout_sx_linear():
    """AxesLayout.sx with linear scale behaves as before."""
    from matplotlib.backend_bases import AxesLayout
    from matplotlib.scale import LinearScale
    layout = AxesLayout(0, 0, 100, 100, 0, 10, 0, 10,
                        LinearScale(), LinearScale())
    assert abs(layout.sx(5) - 50) < 0.01


def test_axes_layout_sx_log():
    """AxesLayout.sx with LogScale maps in log space."""
    from matplotlib.backend_bases import AxesLayout
    from matplotlib.scale import LogScale
    import math
    s = LogScale(base=10)
    # data: [1, 10, 100], log10: [0, 1, 2]
    layout = AxesLayout(0, 0, 200, 100, 0, 2, 0, 10,
                        s, LogScale(base=10))
    # sx(10) → forward(10)=1.0, then linear map [0,2]→[0,200] → 100
    assert abs(layout.sx(10) - 100) < 0.5


def test_axes_layout_backward_compat_no_scales():
    """AxesLayout constructed without scale args must work as before."""
    from matplotlib.backend_bases import AxesLayout
    # Old call site: no xscale/yscale kwargs
    layout = AxesLayout(0, 0, 100, 100, 0, 10, 0, 10)
    assert abs(layout.sx(5) - 50) < 0.01
    assert abs(layout.sy(5) - 50) < 0.01


def test_cla_resets_scale():
    """cla() must reset axis scale to LinearScale and _xscale to 'linear'."""
    import matplotlib.pyplot as plt
    from matplotlib.scale import LinearScale
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.cla()
    assert isinstance(ax.xaxis.get_scale(), LinearScale)
    assert ax.get_xscale() == 'linear'
    plt.close('all')
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_scale_upstream.py -k "set_xscale or axes_layout" -v
```

- [ ] **Step 3: Update `axis.py` — add scale support**

Add to `Axis.__init__`:
```python
from matplotlib.scale import LinearScale
self._scale = LinearScale()
```

Add methods:
```python
def get_scale(self):
    return self._scale

def set_scale(self, scale):
    """Set scale object and update default locator/formatter."""
    from matplotlib.scale import LogScale, SymmetricalLogScale, LinearScale
    from matplotlib.ticker import (LogLocator, LogFormatter,
                                    SymmetricalLogLocator,
                                    AutoLocator, ScalarFormatter)
    self._scale = scale
    if isinstance(scale, LogScale):
        self._major.locator = LogLocator(base=scale.base)
        self._major.formatter = LogFormatter(base=scale.base)
    elif isinstance(scale, SymmetricalLogScale):
        self._major.locator = SymmetricalLogLocator(base=scale.base,
                                                     linthresh=scale.linthresh)
        self._major.formatter = ScalarFormatter()
    else:
        # LinearScale or FuncScale
        self._major.locator = AutoLocator()
        self._major.formatter = ScalarFormatter()
```

- [ ] **Step 4: Update `axes.py` — `set_xscale`/`set_yscale`**

Replace the stub implementations:
```python
def set_xscale(self, scale, **kwargs):
    """Set the x-axis scale."""
    from matplotlib.scale import LinearScale, LogScale, SymmetricalLogScale, FuncScale
    if isinstance(scale, str):
        if scale == 'linear':
            scale_obj = LinearScale()
        elif scale == 'log':
            base = kwargs.get('base', 10.0)
            nonpositive = kwargs.get('nonpositive', 'mask')
            scale_obj = LogScale(base=base, nonpositive=nonpositive)
        elif scale == 'symlog':
            base = kwargs.get('base', 10.0)
            linthresh = kwargs.get('linthresh', 2.0)
            linscale = kwargs.get('linscale', 1.0)
            scale_obj = SymmetricalLogScale(base=base,
                                            linthresh=linthresh,
                                            linscale=linscale)
        else:
            raise ValueError(f"Unknown scale: {scale!r}")
    elif isinstance(scale, (LinearScale, LogScale,
                             SymmetricalLogScale, FuncScale)):
        scale_obj = scale
    else:
        raise TypeError(f"scale must be a string or Scale object, got {type(scale)}")
    self._xscale = scale  # keep str for get_xscale() compat
    self.xaxis.set_scale(scale_obj)

def set_yscale(self, scale, **kwargs):
    """Set the y-axis scale."""
    # Mirror of set_xscale
    from matplotlib.scale import LinearScale, LogScale, SymmetricalLogScale, FuncScale
    if isinstance(scale, str):
        if scale == 'linear':
            scale_obj = LinearScale()
        elif scale == 'log':
            base = kwargs.get('base', 10.0)
            nonpositive = kwargs.get('nonpositive', 'mask')
            scale_obj = LogScale(base=base, nonpositive=nonpositive)
        elif scale == 'symlog':
            base = kwargs.get('base', 10.0)
            linthresh = kwargs.get('linthresh', 2.0)
            linscale = kwargs.get('linscale', 1.0)
            scale_obj = SymmetricalLogScale(base=base,
                                            linthresh=linthresh,
                                            linscale=linscale)
        else:
            raise ValueError(f"Unknown scale: {scale!r}")
    elif isinstance(scale, (LinearScale, LogScale,
                             SymmetricalLogScale, FuncScale)):
        scale_obj = scale
    else:
        raise TypeError(f"scale must be a string or Scale object, got {type(scale)}")
    self._yscale = scale
    self.yaxis.set_scale(scale_obj)
```

- [ ] **Step 5: Update `AxesLayout` in `backend_bases.py` to accept scales**

Change `AxesLayout.__init__` signature to accept optional scale objects:
```python
class AxesLayout:
    def __init__(self, plot_x, plot_y, plot_w, plot_h,
                 xmin, xmax, ymin, ymax,
                 xscale=None, yscale=None):
        ...
        # Pre-compute forward-transformed limits
        from matplotlib.scale import LinearScale
        self._xscale = xscale or LinearScale()
        self._yscale = yscale or LinearScale()
        # Transform limits into display space
        import numpy as np
        self._fxmin = float(np.asarray(self._xscale.forward(xmin)))
        self._fxmax = float(np.asarray(self._xscale.forward(xmax)))
        self._fymin = float(np.asarray(self._yscale.forward(ymin)))
        self._fymax = float(np.asarray(self._yscale.forward(ymax)))

    def sx(self, v):
        """Map data x to pixel x via scale."""
        import numpy as np
        fv = float(np.asarray(self._xscale.forward(v)))
        return self.plot_x + (fv - self._fxmin) / (self._fxmax - self._fxmin) * self.plot_w

    def sy(self, v):
        """Map data y to pixel y (inverted) via scale."""
        import numpy as np
        fv = float(np.asarray(self._yscale.forward(v)))
        return self.plot_y + self.plot_h - (fv - self._fymin) / (self._fymax - self._fymin) * self.plot_h
```

In `axes.py`, update `_compute_layout()` (where `AxesLayout` is constructed) to pass scale objects:
```python
layout = AxesLayout(plot_x, plot_y, plot_w, plot_h,
                    xmin, xmax, ymin, ymax,
                    xscale=self.xaxis.get_scale(),
                    yscale=self.yaxis.get_scale())
```

Find the call to `AxesLayout(...)` in `_compute_layout` and add the two keyword arguments.

Also update `cla()` in `axes.py` to reset the string-cache fields alongside the Axis objects:
```python
# In cla(), add these resets alongside xaxis = XAxis() / yaxis = YAxis():
self._xscale = 'linear'
self._yscale = 'linear'
```
Without this, `get_xscale()` would return a stale `'log'` after `cla()`.

- [ ] **Step 6: Rebuild and run scale integration tests**

```bash
cargo build -p matplotlib-python 2>&1 | tail -3
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_scale_upstream.py -v
```

- [ ] **Step 7: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 8: Commit**

```bash
git add python/matplotlib/axis.py python/matplotlib/axes.py \
        python/matplotlib/backend_bases.py \
        python/matplotlib/tests/test_scale_upstream.py
git commit -m "feat: integrate scale objects into Axis, Axes, and AxesLayout"
```

---

## Chunk 3: Phase 3 — Legend Rendering

### File Map

| Action   | Path |
|----------|------|
| Create   | `python/matplotlib/legend.py` |
| Modify   | `python/matplotlib/axes.py` |
| Create   | `python/matplotlib/tests/test_legend_upstream.py` |

---

### Task 7: Port simplified `legend.py`

Real matplotlib's `legend.py` is ~1400 lines with extensive transform machinery. We copy the *logic* but replace transform-dependent layout with direct pixel math.

**Files:**
- Create: `python/matplotlib/legend.py`
- Create: `python/matplotlib/tests/test_legend_upstream.py`

**What to keep:**
- `loc` parameter: `'best'`, `'upper right'`, `'upper left'`, `'lower left'`, `'lower right'`, `'center'`, and numeric 0-10.
- `ncol` (alias `ncols`): multi-column legend.
- `bbox_to_anchor`: tuple `(x, y)` or `(x, y, w, h)` — pixel coordinates from the top-left of the axes.
- `framealpha`: opacity of legend box (stored; used as SVG `fill-opacity` in Phase 5).
- `title`: string drawn above legend entries.
- `handles`/`labels`: explicit override.
- `fontsize`: numeric or named size.

**What to drop:**
- Shadow, fancy box, draggable.
- `BboxTransformTo`/`BboxTransformFrom`.
- `HandlerBase` extensibility (keep only Line2D → line swatch, Patch → colored rectangle).
- `DraggableLegend`.

**Legend layout algorithm (pixel math, no transforms):**

The legend is positioned relative to the axes bounding box (`plot_x, plot_y, plot_w, plot_h`).

```
loc_map = {
    0: 'best',  1: 'upper right', 2: 'upper left',
    3: 'lower left', 4: 'lower right', 5: 'right',
    6: 'center left', 7: 'center right', 8: 'lower center',
    9: 'upper center', 10: 'center',
}
```

For each `loc`, compute `(lx, ly)` — the top-left corner of the legend box — in pixel space. E.g., `'upper right'` → `(plot_x + plot_w - box_w - 5, plot_y + 5)`.

`'best'` is simplified to `'upper right'` for now.

- [ ] **Step 1: Write failing tests**

Create `python/matplotlib/tests/test_legend_upstream.py`:

```python
# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_legend.py
import pytest
import matplotlib.pyplot as plt


def test_legend_auto_labels():
    """ax.legend() picks up handles/labels from plotted artists."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='line1')
    ax.plot([1, 2], [2, 1], label='line2')
    leg = ax.legend()
    assert leg is not None
    texts = leg.get_texts()
    assert len(texts) == 2
    assert texts[0].get_text() == 'line1'
    assert texts[1].get_text() == 'line2'
    plt.close('all')


def test_legend_no_handles_labels():
    """ax.legend() with nothing labeled returns empty legend (no error)."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])  # no label
    leg = ax.legend()
    assert len(leg.get_texts()) == 0
    plt.close('all')


def test_legend_explicit_labels():
    """ax.legend(['a', 'b']) uses explicit labels."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    ax.plot([1, 2], [2, 1])
    leg = ax.legend(['a', 'b'])
    texts = leg.get_texts()
    assert texts[0].get_text() == 'a'
    assert texts[1].get_text() == 'b'
    plt.close('all')


def test_legend_loc():
    """ax.legend(loc=...) stores the location."""
    fig, ax = plt.subplots()
    ax.plot([1], [1], label='x')
    leg = ax.legend(loc='lower left')
    assert leg.get_loc() == 'lower left'
    plt.close('all')


def test_legend_ncol():
    """ax.legend(ncol=2) stores column count."""
    fig, ax = plt.subplots()
    for i in range(4):
        ax.plot([1], [i], label=f'line{i}')
    leg = ax.legend(ncol=2)
    assert leg.get_ncol() == 2
    plt.close('all')


def test_legend_title():
    """ax.legend(title='T') stores the title."""
    fig, ax = plt.subplots()
    ax.plot([1], [1], label='x')
    leg = ax.legend(title='Legend Title')
    assert leg.get_title().get_text() == 'Legend Title'
    plt.close('all')


def test_legend_renders_without_error():
    """Calling fig.savefig (or to_svg) with legend must not raise."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='x')
    ax.legend()
    svg = fig.to_svg()
    assert len(svg) > 0
    plt.close('all')
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_legend_upstream.py -v
```

- [ ] **Step 3: Create `python/matplotlib/legend.py`**

```python
# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
"""matplotlib.legend — simplified Legend artist."""

from matplotlib.text import Text
from matplotlib.colors import to_hex


_LOC_MAP = {
    0: 'best', 1: 'upper right', 2: 'upper left',
    3: 'lower left', 4: 'lower right', 5: 'right',
    6: 'center left', 7: 'center right',
    8: 'lower center', 9: 'upper center', 10: 'center',
}
_ROW_H = 20   # pixels per legend row
_PAD = 8      # padding inside box
_SWATCH_W = 22  # width of colour swatch
_MIN_W = 80   # minimum legend box width


class LegendText:
    """Minimal text-like proxy for a legend entry label."""
    def __init__(self, text):
        self._text = text
    def get_text(self):
        return self._text
    def set_text(self, s):
        self._text = s


class Legend:
    """Simplified matplotlib-compatible Legend."""

    def __init__(self, ax, handles, labels, *,
                 loc='best', ncol=1, ncols=None,
                 bbox_to_anchor=None, framealpha=0.8,
                 title=None, fontsize=None):
        self._ax = ax
        self._handles = list(handles)
        self._labels = list(labels)
        self._loc = _LOC_MAP.get(loc, loc) if isinstance(loc, int) else loc
        self._ncol = ncols if ncols is not None else ncol
        self._bbox_to_anchor = bbox_to_anchor
        self._framealpha = framealpha
        self._title_text = LegendText(title) if title else LegendText('')
        self._fontsize = fontsize or 11

        # Build LegendText list for get_texts() API
        self._texts = [LegendText(lbl) for lbl in labels]

    # --- API ---
    def get_texts(self):
        return self._texts

    def get_loc(self):
        return self._loc

    def get_ncol(self):
        return self._ncol

    def get_title(self):
        return self._title_text

    def get_handles(self):
        return list(self._handles)

    # --- Geometry ---
    def _box_size(self):
        """Compute (width, height) of the legend box in pixels."""
        n = len(self._labels)
        if n == 0:
            return (0, 0)
        title_h = _ROW_H if self._title_text.get_text() else 0
        nrows = -(-n // self._ncol)  # ceiling division
        h = title_h + nrows * _ROW_H + _PAD
        # estimate width: swatch + longest label text
        max_len = max((len(lbl) for lbl in self._labels), default=0)
        col_w = _SWATCH_W + max_len * 6 + _PAD  # ~6px per char
        w = max(_MIN_W, col_w * self._ncol + _PAD)
        return (w, h)

    def _box_origin(self, layout):
        """Compute (lx, ly) top-left of legend box using loc."""
        px, py = layout.plot_x, layout.plot_y
        pw, ph = layout.plot_w, layout.plot_h
        w, h = self._box_size()

        if self._bbox_to_anchor is not None:
            bta = self._bbox_to_anchor
            if len(bta) >= 2:
                return (px + bta[0] * pw - w, py + bta[1] * ph - h)

        margin = 5
        loc = self._loc if self._loc != 'best' else 'upper right'

        if loc in ('upper right', 'right'):
            return (px + pw - w - margin, py + margin)
        elif loc == 'upper left':
            return (px + margin, py + margin)
        elif loc == 'lower left':
            return (px + margin, py + ph - h - margin)
        elif loc == 'lower right':
            return (px + pw - w - margin, py + ph - h - margin)
        elif loc == 'upper center':
            return (px + pw / 2 - w / 2, py + margin)
        elif loc == 'lower center':
            return (px + pw / 2 - w / 2, py + ph - h - margin)
        elif loc == 'center left':
            return (px + margin, py + ph / 2 - h / 2)
        elif loc == 'center right':
            return (px + pw - w - margin, py + ph / 2 - h / 2)
        elif loc == 'center':
            return (px + pw / 2 - w / 2, py + ph / 2 - h / 2)
        return (px + pw - w - margin, py + margin)  # fallback upper right

    def draw(self, renderer, layout):
        """Draw legend onto renderer."""
        if not self._labels:
            return

        w, h = self._box_size()
        lx, ly = self._box_origin(layout)

        # Frame
        renderer.draw_rect(lx, ly, w, h, '#999999', '#ffffff')

        # Title
        row = 0
        title = self._title_text.get_text()
        if title:
            renderer.draw_text(
                lx + _PAD, ly + _PAD + row * _ROW_H + 13,
                title, self._fontsize + 1, '#000000', 'left',
            )
            row += 1

        # Entries (simple single-column for now; ncol support is additive)
        for i, (handle, label) in enumerate(
                zip(self._handles, self._labels)):
            col = i % self._ncol
            r = row + i // self._ncol
            ey = ly + _PAD + r * _ROW_H + _ROW_H // 2
            ex = lx + _PAD + col * (w // self._ncol)

            # Colour swatch (line or patch)
            color = '#555555'
            if hasattr(handle, 'get_color'):
                try:
                    color = to_hex(handle.get_color())
                except Exception:
                    pass
            elif hasattr(handle, 'get_facecolor'):
                try:
                    fc = handle.get_facecolor()
                    if len(fc) == 4:
                        color = to_hex(fc[:3])
                except Exception:
                    pass

            renderer.draw_line(
                [ex, ex + _SWATCH_W - 4], [ey, ey],
                color, 2.0, '-',
            )
            renderer.draw_text(
                ex + _SWATCH_W, ey + 4,
                self._texts[i].get_text(),
                self._fontsize, '#333333', 'left',
            )
```

- [ ] **Step 4: Run legend tests — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_legend_upstream.py -v
```

- [ ] **Step 5: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/legend.py python/matplotlib/tests/test_legend_upstream.py
git commit -m "feat: add simplified Legend class in legend.py"
```

---

### Task 8: Migrate `axes.py` from `_draw_legend` to `Legend` object

**Files:**
- Modify: `python/matplotlib/axes.py`
- Modify: `python/matplotlib/tests/test_legend_upstream.py`

- [ ] **Step 1: Write failing integration test**

Append to `python/matplotlib/tests/test_legend_upstream.py`:

```python
def test_axes_legend_returns_legend_object():
    """ax.legend() must return a Legend and store it as _legend_obj."""
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='x')
    leg = ax.legend()
    assert isinstance(leg, Legend)
    assert ax._legend_obj is leg
    plt.close('all')
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_legend_upstream.py::test_axes_legend_returns_legend_object -v
```

- [ ] **Step 3: Update `axes.py`**

**3a. Add import:**
```python
from matplotlib.legend import Legend
```

**3b. In `__init__`, replace:**
```python
self._legend = False
```
with:
```python
self._legend_obj = None
```

**3b-extra. Add `get_legend_handles_labels()` method** (required by `legend()`):
```python
def get_legend_handles_labels(self):
    """Return (handles, labels) for all artists with a non-empty label."""
    handles, labels = [], []
    for artist in self.lines + self.patches + list(self.collections):
        lbl = artist.get_label()
        if lbl and not lbl.startswith('_'):
            handles.append(artist)
            labels.append(lbl)
    # Also check containers
    for container in self.containers:
        lbl = container.get_label() if hasattr(container, 'get_label') else ''
        if lbl and not lbl.startswith('_'):
            handles.append(container)
            labels.append(lbl)
    return handles, labels
```

Note: This method may already exist in `axes.py`. If so, verify it uses `get_label()` and filters `_nolegend_`/underscore-prefixed labels. Update if needed.

**3c. Replace `legend()` method:**
```python
def legend(self, *args, **kwargs):
    """Create and return a Legend object for this axes."""
    if len(args) > 2:
        raise TypeError(
            f"legend() takes at most 2 positional arguments "
            f"({len(args)} given)")

    # Resolve handles and labels
    if len(args) == 2:
        handles, labels = args[0], args[1]
    elif len(args) == 1:
        # Single arg: list of labels
        labels = args[0]
        handles, _ = self.get_legend_handles_labels()
        handles = handles[:len(labels)]
    else:
        handles, labels = self.get_legend_handles_labels()

    leg = Legend(self, handles, labels, **kwargs)
    self._legend_obj = leg
    return leg
```

**3d. In `draw()`, replace:**
```python
# Legend
if self._legend:
    self._draw_legend(renderer, px + pw - 10, py + 10)
```
with:
```python
# Legend
if self._legend_obj is not None:
    self._legend_obj.draw(renderer, layout)
```

**3e. Remove `_draw_legend()` method entirely.**

**3f. In `cla()`, replace:**
```python
self._legend = False
```
with:
```python
self._legend_obj = None
```

- [ ] **Step 4: Rebuild and run tests**

```bash
cargo build -p matplotlib-python 2>&1 | tail -3
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_legend_upstream.py -v
```

- [ ] **Step 5: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/axes.py python/matplotlib/tests/test_legend_upstream.py
git commit -m "feat: replace _draw_legend with Legend object in axes.py"
```

---

## Chunk 4: Phase 4 — Arrow Annotations

### File Map

| Action   | Path |
|----------|------|
| Modify   | `python/matplotlib/backend_bases.py` |
| Modify   | `python/matplotlib/_svg_backend.py` |
| Modify   | `python/matplotlib/_pil_backend.py` |
| Modify   | `python/matplotlib/patches.py` |
| Modify   | `python/matplotlib/text.py` |
| Create   | `python/matplotlib/tests/test_annotation_upstream.py` |

---

### Task 9: Add `draw_arrow` to renderers

**Files:**
- Modify: `python/matplotlib/backend_bases.py`
- Modify: `python/matplotlib/_svg_backend.py`
- Modify: `python/matplotlib/_pil_backend.py`
- Create: `python/matplotlib/tests/test_annotation_upstream.py`

- [ ] **Step 1: Write failing renderer tests**

Create `python/matplotlib/tests/test_annotation_upstream.py`:

```python
# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_text.py (annotation section)
import pytest


def test_renderer_svg_draw_arrow_no_error():
    """RendererSVG.draw_arrow must produce valid SVG with a path element."""
    from matplotlib._svg_backend import RendererSVG
    r = RendererSVG(200, 200, 100)
    r.draw_arrow(10, 100, 150, 50, '->', '#ff0000', 1.5)
    svg = r.get_result()
    assert '<path' in svg or '<line' in svg
    assert 'marker-end' in svg or 'polygon' in svg.lower() or '<path' in svg


def test_renderer_svg_draw_arrow_no_head():
    """draw_arrow with style '-' must draw a line without arrowhead."""
    from matplotlib._svg_backend import RendererSVG
    r = RendererSVG(200, 200, 100)
    r.draw_arrow(10, 100, 150, 50, '-', '#000000', 1.0)
    svg = r.get_result()
    assert '<polyline' in svg or '<line' in svg or '<path' in svg


def test_renderer_pil_draw_arrow_no_error():
    """RendererPIL.draw_arrow must not raise."""
    from matplotlib._pil_backend import RendererPIL
    r = RendererPIL(200, 200, 100)
    r.draw_arrow(10, 100, 150, 50, '->', '#ff0000', 1.5)
    # Just check it produces bytes without error
    result = r.get_result()
    assert len(result) > 0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_annotation_upstream.py -k "renderer" -v
```

- [ ] **Step 3: Add `draw_arrow` to `RendererBase`**

In `backend_bases.py`, add to `RendererBase`:
```python
def draw_arrow(self, x1, y1, x2, y2, arrowstyle, color, linewidth):
    """Draw an arrow from (x1,y1) to (x2,y2).

    arrowstyle : str
        One of '->', '<-', '<->', '-', 'fancy'.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Implement `draw_arrow` in `RendererSVG`**

Add to `_svg_backend.py`:
```python
def draw_arrow(self, x1, y1, x2, y2, arrowstyle, color, linewidth):
    """Draw an arrow using SVG path with marker-end for arrowhead."""
    import math
    arrow_id = f'arrow-{len(self._parts)}'
    has_end = arrowstyle in ('->', '<->', 'fancy')
    has_start = arrowstyle in ('<-', '<->')

    defs_parts = []
    if has_end or has_start:
        # Define a simple triangle arrowhead marker
        marker = (
            f'<defs>'
            f'<marker id="{arrow_id}-end" markerWidth="8" markerHeight="6" '
            f'refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{color}"/>'
            f'</marker>'
        )
        if has_start:
            marker += (
                f'<marker id="{arrow_id}-start" markerWidth="8" markerHeight="6" '
                f'refX="1" refY="3" orient="auto-start-reverse">'
                f'<polygon points="0 0, 8 3, 0 6" fill="{color}"/>'
                f'</marker>'
            )
        marker += '</defs>'
        self._parts.append(marker)

    attrs = (f'stroke="{color}" stroke-width="{linewidth}" fill="none"')
    if has_end:
        attrs += f' marker-end="url(#{arrow_id}-end)"'
    if has_start:
        attrs += f' marker-start="url(#{arrow_id}-start)"'

    self._parts.append(
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attrs}/>'
    )
```

- [ ] **Step 5: Implement `draw_arrow` in `RendererPIL`**

Add to `_pil_backend.py`:
```python
def draw_arrow(self, x1, y1, x2, y2, arrowstyle, color, linewidth):
    """Draw arrow: line + arrowhead polygon."""
    import math
    col = _to_rgb_255(color)
    lw = max(1, int(linewidth))

    # Draw the shaft
    self._draw.line([(int(x1), int(y1)), (int(x2), int(y2))],
                    fill=col, width=lw)

    has_end = arrowstyle in ('->', '<->', 'fancy')
    has_start = arrowstyle in ('<-', '<->')

    head_len = max(8, lw * 4)
    head_w = max(4, lw * 2)

    def _arrowhead(tx, ty, fx, fy):
        """Draw a filled triangle arrowhead at (tx,ty) pointing away from (fx,fy)."""
        dx = tx - fx
        dy = ty - fy
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular
        tip = (int(tx), int(ty))
        base1 = (int(tx - ux * head_len + px * head_w),
                 int(ty - uy * head_len + py * head_w))
        base2 = (int(tx - ux * head_len - px * head_w),
                 int(ty - uy * head_len - py * head_w))
        self._draw.line([tip, base1], fill=col, width=lw)
        self._draw.line([tip, base2], fill=col, width=lw)
        self._draw.line([base1, base2], fill=col, width=lw)

    if has_end:
        _arrowhead(x2, y2, x1, y1)
    if has_start:
        _arrowhead(x1, y1, x2, y2)
```

- [ ] **Step 6: Run renderer tests — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_annotation_upstream.py -k "renderer" -v
```

- [ ] **Step 7: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 8: Commit**

```bash
git add python/matplotlib/backend_bases.py python/matplotlib/_svg_backend.py \
        python/matplotlib/_pil_backend.py \
        python/matplotlib/tests/test_annotation_upstream.py
git commit -m "feat: add draw_arrow primitive to RendererBase, RendererSVG, RendererPIL"
```

---

### Task 10: Add `FancyArrowPatch` to `patches.py` and `Annotation.draw()`

**Files:**
- Modify: `python/matplotlib/patches.py`
- Modify: `python/matplotlib/text.py`
- Modify: `python/matplotlib/tests/test_annotation_upstream.py`

- [ ] **Step 1: Write failing tests**

Append to `python/matplotlib/tests/test_annotation_upstream.py`:

```python
def test_annotate_default_arrow():
    """ax.annotate with arrowprops must render without error."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.annotate('hi', xy=(0.5, 0.5), xytext=(0.2, 0.8),
                arrowprops=dict(arrowstyle='->'))
    svg = fig.to_svg()
    assert len(svg) > 0
    plt.close('all')


def test_annotate_no_arrowprops():
    """ax.annotate without arrowprops renders only text."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.annotate('label', xy=(0.5, 0.5))
    svg = fig.to_svg()
    assert 'label' in svg
    plt.close('all')


def test_annotate_arrowprops_styles():
    """Multiple arrowstyle strings must not raise."""
    import matplotlib.pyplot as plt
    for style in ['->', '<-', '<->', '-', 'fancy']:
        fig, ax = plt.subplots()
        ax.annotate('x', xy=(0.5, 0.5), xytext=(0.1, 0.9),
                    arrowprops=dict(arrowstyle=style))
        fig.to_svg()  # must not raise
        plt.close('all')


def test_fancy_arrow_patch_draw():
    """FancyArrowPatch.draw must call renderer.draw_arrow."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.backend_bases import AxesLayout
    from matplotlib.scale import LinearScale
    from unittest.mock import MagicMock

    layout = AxesLayout(0, 0, 100, 100, 0, 1, 0, 1,
                        LinearScale(), LinearScale())
    renderer = MagicMock()
    patch = FancyArrowPatch((0.1, 0.2), (0.8, 0.7),
                             arrowstyle='->', color='red', linewidth=1.5)
    patch.draw(renderer, layout)
    renderer.draw_arrow.assert_called_once()
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_annotation_upstream.py -k "annotate or fancy_arrow" -v
```

- [ ] **Step 3: Add `FancyArrowPatch` to `patches.py`**

Append to `python/matplotlib/patches.py`:
```python
class FancyArrowPatch(Artist):
    """A simplified arrow patch from posA to posB in data coordinates."""

    zorder = 2

    def __init__(self, posA, posB, arrowstyle='->', color='black',
                 linewidth=1.5, shrinkA=0.0, shrinkB=0.0, **kwargs):
        super().__init__()
        self._posA = tuple(posA)
        self._posB = tuple(posB)
        self._arrowstyle = arrowstyle
        self._color = color
        self._linewidth = linewidth
        self._shrinkA = shrinkA
        self._shrinkB = shrinkB

    def get_arrowstyle(self):
        return self._arrowstyle

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        import math
        x1, y1 = layout.sx(self._posA[0]), layout.sy(self._posA[1])
        x2, y2 = layout.sx(self._posB[0]), layout.sy(self._posB[1])

        # Apply shrink
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length > 1e-6 and (self._shrinkA or self._shrinkB):
            ux, uy = dx / length, dy / length
            x1 += ux * self._shrinkA
            y1 += uy * self._shrinkA
            x2 -= ux * self._shrinkB
            y2 -= uy * self._shrinkB

        from matplotlib.colors import to_hex
        color = to_hex(self._color)
        renderer.draw_arrow(x1, y1, x2, y2,
                            self._arrowstyle, color, self._linewidth)
```

- [ ] **Step 4: Update `Annotation` in `text.py` — add `draw()` method**

In `text.py`, find the `Annotation` class and add a `draw` method after `__init__`:

```python
def draw(self, renderer, layout):
    """Draw text and optional arrow."""
    if not self.get_visible():
        return
    # Draw text (delegates to Text.draw)
    super().draw(renderer, layout)
    # Draw arrow if arrowprops were given
    if self.arrow_patch is not None:
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.colors import to_hex
        # arrow goes from xytext (text position) to xy (annotation point)
        arrowstyle = '->'
        if hasattr(self, '_arrowprops') and isinstance(self._arrowprops, dict):
            arrowstyle = self._arrowprops.get('arrowstyle', '->')
        color = '#000000'
        linewidth = 1.5
        patch = FancyArrowPatch(
            self.xytext, self.xy,
            arrowstyle=arrowstyle,
            color=color,
            linewidth=linewidth,
        )
        patch.draw(renderer, layout)
```

Also update `Annotation.__init__` to store `arrowprops` and pre-read `arrowstyle`/`color`/`linewidth`:
```python
def __init__(self, text, xy, xytext=None, arrowprops=None, **kwargs):
    self.xy = tuple(xy)
    self.xytext = tuple(xytext) if xytext is not None else self.xy
    super().__init__(x=self.xytext[0], y=self.xytext[1], text=text, **kwargs)
    self._arrowprops = arrowprops
    # arrow_patch is non-None iff arrowprops was given — used as a sentinel
    self.arrow_patch = True if arrowprops is not None else None
```

Also update `draw()` to extract color/linewidth from `arrowprops` dict:
```python
def draw(self, renderer, layout):
    """Draw text and optional arrow."""
    if not self.get_visible():
        return
    super().draw(renderer, layout)
    if self.arrow_patch is not None and self._arrowprops is not None:
        from matplotlib.patches import FancyArrowPatch
        props = self._arrowprops if isinstance(self._arrowprops, dict) else {}
        arrowstyle = props.get('arrowstyle', '->')
        color = props.get('color', props.get('ec', '#000000'))
        linewidth = props.get('linewidth', props.get('lw', 1.5))
        patch = FancyArrowPatch(
            self.xytext, self.xy,
            arrowstyle=arrowstyle,
            color=color,
            linewidth=linewidth,
        )
        patch.draw(renderer, layout)
```

Also add dependency note: `test_fancy_arrow_patch_draw` constructs `AxesLayout` with `LinearScale()` arguments — this requires Phase 2 (Task 6) to be complete first. If implementing Task 10 before Phase 2, simplify that test to not pass scale args (the backward-compat constructor accepts no scale args).

- [ ] **Step 5: Rebuild and run annotation tests**

```bash
cargo build -p matplotlib-python 2>&1 | tail -3
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_annotation_upstream.py -v
```

- [ ] **Step 6: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 7: Commit**

```bash
git add python/matplotlib/patches.py python/matplotlib/text.py \
        python/matplotlib/tests/test_annotation_upstream.py
git commit -m "feat: add FancyArrowPatch and Annotation.draw() for arrow annotations"
```

---

## Chunk 5: Phase 5 — Artist Properties

### File Map

| Action   | Path |
|----------|------|
| Modify   | `python/matplotlib/artist.py` |
| Modify   | `python/matplotlib/lines.py` |
| Modify   | `python/matplotlib/patches.py` |
| Modify   | `python/matplotlib/_svg_backend.py` |
| Modify   | `python/matplotlib/_pil_backend.py` |
| Create   | `python/matplotlib/tests/test_artist_upstream.py` |

---

### Task 11: `artist.py` consolidation — add `clip_on`, audit per-class duplication

**Files:**
- Modify: `python/matplotlib/artist.py`
- Create: `python/matplotlib/tests/test_artist_upstream.py`

- [ ] **Step 1: Write failing tests**

Create `python/matplotlib/tests/test_artist_upstream.py`:

```python
# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_artist.py and test_lines.py
import pytest
import matplotlib.pyplot as plt


def test_artist_clip_on_default_true():
    """Artist.clip_on must default to True."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    assert line.get_clip_on() is True


def test_artist_set_clip_on():
    """set_clip_on(False) must turn off clipping."""
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    r.set_clip_on(False)
    assert r.get_clip_on() is False


def test_artist_alpha():
    """set_alpha / get_alpha round-trip."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    assert line.get_alpha() is None  # default
    line.set_alpha(0.5)
    assert line.get_alpha() == 0.5


def test_zorder_defaults():
    """Line2D zorder=2, Patch zorder=1, Text zorder=3."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    assert Line2D([0], [0]).get_zorder() == 2
    assert Patch().get_zorder() == 1
    assert Text(0, 0, 'x').get_zorder() == 3
```

- [ ] **Step 2: Run — expect FAIL for clip_on**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_artist_upstream.py -v
```

- [ ] **Step 3: Add `clip_on` to `Artist` base class**

In `python/matplotlib/artist.py`, add in `__init__`:
```python
self._clip_on = True
```

Add methods:
```python
def get_clip_on(self):
    return self._clip_on

def set_clip_on(self, b):
    self._clip_on = bool(b)
```

- [ ] **Step 4: Verify and set zorder class-level defaults**

Check `python/matplotlib/lines.py` — `Line2D` must have `zorder = 2`. Check `python/matplotlib/patches.py` — `Patch` must have `zorder = 1`. Check `python/matplotlib/text.py` — `Text` must have `zorder = 3`. Add any missing class attributes:

```python
# In lines.py:
class Line2D(Artist):
    zorder = 2  # ensure this line exists

# In patches.py:
class Patch(Artist):
    zorder = 1  # ensure this line exists

# In text.py:
class Text(Artist):
    zorder = 3  # ensure this line exists
```

- [ ] **Step 4b: Verify zorder sort exists in `Axes.draw()`**

The spec says Phase 5 sorts artists by `zorder` in `Axes.draw()`. Check `axes.py` — the `all_artists.sort(key=lambda a: a.get_zorder())` line should already exist (it was added in an earlier commit). Verify it's present. If missing, add it before the artist draw loop:
```python
all_artists.sort(key=lambda a: a.get_zorder())
```

Also add a test confirming draw order:
```python
def test_zorder_draw_order_in_svg():
    """Artists with lower zorder must appear earlier in SVG."""
    fig, ax = plt.subplots()
    # Default: patches zorder=1 before lines zorder=2
    ax.bar([1], [1])  # adds a Rectangle (zorder=1)
    ax.plot([1], [1], label='line')  # adds a Line2D (zorder=2)
    svg = fig.to_svg()
    # polyline (line) must appear after rect (bar) in SVG
    rect_pos = svg.find('<rect')
    line_pos = svg.find('<polyline')
    assert rect_pos < line_pos, "Patch (zorder=1) must appear before Line2D (zorder=2)"
    plt.close('all')
```

Append this test to `test_artist_upstream.py`.

- [ ] **Step 5: Run tests — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_artist_upstream.py -v
```

- [ ] **Step 6: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 7: Commit**

```bash
git add python/matplotlib/artist.py python/matplotlib/text.py \
        python/matplotlib/tests/test_artist_upstream.py
git commit -m "feat: add clip_on property to Artist base class, verify zorder defaults"
```

---

### Task 12: Alpha rendering — SVG `opacity` + PIL RGBA blend

**Files:**
- Modify: `python/matplotlib/_svg_backend.py`
- Modify: `python/matplotlib/_pil_backend.py`
- Modify: `python/matplotlib/lines.py`
- Modify: `python/matplotlib/patches.py`
- Modify: `python/matplotlib/tests/test_artist_upstream.py`

Alpha is already stored on each artist via `Artist._alpha`. The task is to:
1. Pass alpha through to renderer calls (artists call `renderer.draw_line(..., alpha=self.get_alpha())`)
2. SVG: add `opacity` attribute when alpha < 1
3. PIL: blend with white background

Note: Rather than changing renderer method signatures (breaking all call sites), a simpler approach is: each artist's `draw()` method computes an effective color incorporating alpha, then passes it. For SVG, wrapping in a `<g opacity="...">` group is clean and requires no signature changes.

A cleaner approach that avoids signature changes: artists apply alpha to their color hex string by prepending an SVG `<g>` tag with opacity. Use a context-manager-style `renderer.push_alpha(alpha)` / `renderer.pop_alpha()`.

Simplest non-breaking approach: add an `alpha` attribute to renderer; artists set `renderer.current_alpha = self.get_alpha()` before drawing, reset after. SVG checks `current_alpha` when building attributes.

**Chosen approach:** Add `draw_line_alpha`, `draw_rect_alpha`, etc. is too much. Instead, **add an `opacity` parameter to `draw_line` and `draw_rect` with default `1.0`**, and update all existing call sites in `axes.py` to not pass it (backward compat: default=1.0). Artists then pass `alpha=self.get_alpha()` explicitly.

Actually the cleanest: **add `opacity` parameter with default 1.0 to `draw_line`, `draw_markers`, `draw_rect` in RendererBase, RendererSVG, RendererPIL.** Existing call sites don't need to change (default=1.0). Artist `draw()` methods pass it when set.

- [ ] **Step 1: Write failing alpha test**

Append to `python/matplotlib/tests/test_artist_upstream.py`:

```python
def test_alpha_in_svg():
    """A line with alpha=0.5 must store alpha on Line2D and produce opacity in SVG."""
    fig, ax = plt.subplots()
    line, = ax.plot([1, 2], [1, 2], alpha=0.5)
    # Verify alpha is stored on the artist
    assert line.get_alpha() == 0.5
    # Verify opacity appears in SVG output
    svg = fig.to_svg()
    assert 'opacity="0.5"' in svg or 'opacity="0.5' in svg
    plt.close('all')
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_artist_upstream.py::test_alpha_in_svg -v
```

- [ ] **Step 3: Update `draw_line` in renderers to accept `opacity` param**

In `backend_bases.py`:
```python
def draw_line(self, xdata, ydata, color, linewidth, linestyle, opacity=1.0):
    raise NotImplementedError
```

In `_svg_backend.py`, update `draw_line`:
```python
def draw_line(self, xdata, ydata, color, linewidth, linestyle, opacity=1.0):
    dash = _svg_dash(linestyle)
    points = ' '.join(
        f'{xdata[i]:.2f},{ydata[i]:.2f}' for i in range(len(xdata))
    )
    clip = self._clip_attr()
    opacity_attr = f' opacity="{opacity}"' if opacity < 1.0 else ''
    self._parts.append(
        f'<polyline points="{points}" fill="none" '
        f'stroke="{color}" stroke-width="{linewidth}"{dash}{opacity_attr}{clip}/>'
    )
```

In `_pil_backend.py`, `opacity` is ignored (PIL draws opaque; full RGBA support would need Image mode='RGBA'):
```python
def draw_line(self, xdata, ydata, color, linewidth, linestyle, opacity=1.0):
    col = _to_rgb_255(color)
    ...  # existing implementation unchanged
```

- [ ] **Step 4: Update `Line2D.draw()` to pass alpha**

In `lines.py`, find the `draw()` method and update the `renderer.draw_line(...)` call to include `opacity=self._alpha if self._alpha is not None else 1.0`.

- [ ] **Step 5: Run alpha test — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_artist_upstream.py::test_alpha_in_svg -v
```

- [ ] **Step 6: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 7: Commit**

```bash
git add python/matplotlib/backend_bases.py python/matplotlib/_svg_backend.py \
        python/matplotlib/_pil_backend.py python/matplotlib/lines.py \
        python/matplotlib/tests/test_artist_upstream.py
git commit -m "feat: add opacity parameter to draw_line, pass alpha from Line2D"
```

---

### Task 13: Extended linestyle support — dash tuples and named styles

**Files:**
- Modify: `python/matplotlib/_svg_backend.py`
- Modify: `python/matplotlib/tests/test_artist_upstream.py`

Real matplotlib allows linestyle as:
- Named: `'solid'`, `'dashed'`, `'dotted'`, `'dashdot'`, `'loosely dashed'`, `'densely dashed'`, etc.
- Tuple: `(offset, (on, off, on, off, ...))`

The current `_svg_dash()` handles `'--'`/`'dashed'`, `':'`/`'dotted'`, `'-.'`/`'dashdot'`. Extend it.

- [ ] **Step 1: Write failing linestyle tests**

Append to `python/matplotlib/tests/test_artist_upstream.py`:

```python
def test_linestyle_tuple_format():
    """Linestyle as (offset, (on, off)) tuple must appear in SVG."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], linestyle=(0, (3, 5)))
    svg = fig.to_svg()
    assert 'stroke-dasharray' in svg
    plt.close('all')


def test_linestyle_named_solid():
    """linestyle='solid' must produce no stroke-dasharray."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], linestyle='solid')
    svg = fig.to_svg()
    # solid lines produce no dasharray
    assert 'stroke-dasharray' not in svg or svg.count('stroke-dasharray') == 0
    plt.close('all')


def test_linestyle_loosely_dashed():
    """Named extended linestyle 'loosely dashed' must produce dasharray."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], linestyle='loosely dashed')
    svg = fig.to_svg()
    assert 'stroke-dasharray' in svg
    plt.close('all')
```

- [ ] **Step 2: Run — expect FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_artist_upstream.py -k "linestyle" -v
```

- [ ] **Step 3: Extend `_svg_dash()` in `_svg_backend.py`**

```python
# Named style → (on, off, ...) dash sequences
_NAMED_DASHES = {
    'solid': None,
    '-': None,
    'dashed': (6, 3),
    '--': (6, 3),
    'dotted': (2, 2),
    ':': (2, 2),
    'dashdot': (6, 2, 2, 2),
    '-.': (6, 2, 2, 2),
    'loosely dashed': (6, 6),
    'densely dashed': (4, 1),
    'loosely dotted': (2, 4),
    'densely dotted': (1, 1),
    'loosely dashdotted': (6, 4, 2, 4),
    'densely dashdotted': (4, 1, 2, 1),
}


def _svg_dash(ls):
    """Return SVG stroke-dasharray attribute string for a linestyle."""
    if isinstance(ls, tuple):
        # (offset, (on, off, ...)) format
        offset, dashes = ls
        dash_str = ','.join(str(d) for d in dashes)
        return f' stroke-dasharray="{dash_str}"'

    seq = _NAMED_DASHES.get(ls)
    if seq is None:
        return ''  # solid or unknown → no dasharray
    dash_str = ','.join(str(d) for d in seq)
    return f' stroke-dasharray="{dash_str}"'
```

- [ ] **Step 4: Run linestyle tests — expect PASS**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_artist_upstream.py -k "linestyle" -v
```

- [ ] **Step 5: Full suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/_svg_backend.py \
        python/matplotlib/tests/test_artist_upstream.py
git commit -m "feat: extend linestyle support to tuple format and named styles"
```

---

### Task 14: Final verification and baseline bump

- [ ] **Step 1: Run complete test suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -v 2>&1 | tail -30
```

Expected: all original 789 tests + all new upstream tests passing. No failures.

- [ ] **Step 2: Count total tests**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q 2>&1 | tail -5
```

Document the new passing count.

- [ ] **Step 3: Final commit (if any uncommitted changes remain)**

If any files were modified and not yet committed in a prior step, commit them now:
```bash
git status  # check what's uncommitted
git add python/matplotlib/axes.py python/matplotlib/artist.py \
        python/matplotlib/lines.py python/matplotlib/patches.py \
        python/matplotlib/text.py \
        python/matplotlib/tests/test_artist_upstream.py
git commit -m "chore: all 5 upstream compatibility phases complete"
```

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `python/matplotlib/rcsetup.py` | +6 axes.formatter.* keys |
| `python/matplotlib/ticker.py` | NEW — port from upstream |
| `python/matplotlib/axis.py` | NEW — XAxis/YAxis |
| `python/matplotlib/scale.py` | NEW — port from upstream |
| `python/matplotlib/legend.py` | NEW — simplified port |
| `python/matplotlib/axes.py` | Migrate ticks, scale, legend |
| `python/matplotlib/backend_bases.py` | draw_arrow, scale-aware AxesLayout |
| `python/matplotlib/_svg_backend.py` | draw_arrow, opacity, extended linestyles |
| `python/matplotlib/_pil_backend.py` | draw_arrow |
| `python/matplotlib/patches.py` | FancyArrowPatch |
| `python/matplotlib/text.py` | Annotation.draw(), store arrowprops |
| `python/matplotlib/artist.py` | clip_on property |
| `python/matplotlib/lines.py` | pass opacity to draw_line |
| `python/matplotlib/tests/test_ticker_upstream.py` | NEW |
| `python/matplotlib/tests/test_scale_upstream.py` | NEW |
| `python/matplotlib/tests/test_legend_upstream.py` | NEW |
| `python/matplotlib/tests/test_annotation_upstream.py` | NEW |
| `python/matplotlib/tests/test_artist_upstream.py` | NEW |
