# Patches Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `draw()` methods to 6 stub patch classes (Ellipse, Arc, FancyBboxPatch, RegularPolygon, Arrow, PathPatch) plus one new renderer primitive (`draw_ellipse`) and a minimal `Path` class.

**Architecture:** Each patch's `draw()` converts data-space coordinates to display-space via `layout.sx/sy`, then calls renderer primitives. A new `draw_ellipse` primitive handles ellipse rendering in both PIL and SVG backends. A minimal `matplotlib.path.Path` class is created to support `PathPatch` and the test.

**Tech Stack:** Python (RustPython/WASM), PIL (via pillow-rust), SVG string generation, existing renderer backends.

**Worktree:** `packages/matplotlib-py/.worktrees/patches/`
**Branch:** `feat/patches`
**Binary:** `target/debug/matplotlib-python`
**Test command:** `target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q`
**Baseline:** 875 passed, 0 failed.

---

## Critical Patterns — Read Before Coding

### Layout coordinate conversion
```python
# From Circle.draw() — the correct pattern:
cx_px = layout.sx(self._center[0])       # data x → pixel x
cy_px = layout.sy(self._center[1])       # data y → pixel y (inverted)
r_px = abs(layout.sx(self._center[0] + self._radius) - cx_px)  # radius in pixels

# For y-axis extent (e.g. height/2), use the same trick:
ry = abs(layout.sy(self._center[1] + self._height / 2) - cy_px)
# abs() handles the y-inversion automatically
```

**There is NO `layout.sy_raw` method.** Use `abs(layout.sy(center_y + half_height) - cy_px)` for vertical extents.

### Color helpers
```python
# Use these Patch methods — don't call to_hex() directly:
fc = self._resolved_facecolor_hex()   # returns 'none' or hex string
ec = self._resolved_edgecolor_hex()   # returns 'none' or hex string
alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
```

### Screen y-axis
Screen y increases downward; data y increases upward. For trigonometric shapes:
```python
# Correct:
py = cy_px - r_px * math.sin(angle)   # negate sin for screen coords
# Wrong:
py = cy_px + r_px * math.sin(angle)   # this goes in the wrong direction
```

### SVG parts list
In `_svg_backend.py`, append to `self._parts` (not `self._elements`).

### PIL draw object
In `_pil_backend.py`, use `self._draw` (the ImageDraw object).

---

## File Map

| File | Action |
|------|--------|
| `python/matplotlib/path.py` | **Create** — minimal `Path` class with code constants |
| `python/matplotlib/patches.py` | **Modify** — add `draw()` to 6 classes + `_rounded_rect_points` helper |
| `python/matplotlib/backend_bases.py` | **Modify** — add `draw_ellipse()` stub |
| `python/matplotlib/_pil_backend.py` | **Modify** — implement `draw_ellipse()` |
| `python/matplotlib/_svg_backend.py` | **Modify** — implement `draw_ellipse()` |
| `python/matplotlib/tests/test_patches_upstream.py` | **Create** — 8 tests |

---

## Task 0: Set Up Worktree and Baseline

**Files:** worktree setup only

- [ ] **Step 1: Create worktree**

```bash
cd /Users/sunny/work/codepod/codepod/packages/matplotlib-py
git worktree add .worktrees/patches -b feat/patches
```

- [ ] **Step 2: Copy binary into worktree**

```bash
cp target/debug/matplotlib-python .worktrees/patches/target/debug/matplotlib-python
```

Wait — the worktree shares the same `target/` via symlink or the binary is at the repo root. Check:
```bash
ls .worktrees/patches/target/debug/matplotlib-python 2>/dev/null || echo "need to copy"
```

If missing:
```bash
mkdir -p .worktrees/patches/target/debug
cp target/debug/matplotlib-python .worktrees/patches/target/debug/matplotlib-python
```

- [ ] **Step 3: Verify baseline from worktree**

```bash
cd /Users/sunny/work/codepod/codepod/packages/matplotlib-py/.worktrees/patches
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q 2>&1 | tail -3
```

Expected: `875 passed` (or current baseline). Note the exact number.

---

## Task 1: Create Test File (All 8 Tests)

**Files:**
- Create: `python/matplotlib/tests/test_patches_upstream.py`

- [ ] **Step 1: Create the test file**

```python
"""Tests for patch rendering completeness (sub-project C).

Baseline: 875 passed, 0 failed. Any regression is a bug.
"""
import io
import pytest


def test_ellipse_renders_png():
    """Ellipse produces non-background pixels at its center."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    e = Ellipse((0.5, 0.5), 0.6, 0.3, color='red')
    ax.add_patch(e)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    red_pixels = sum(1 for r, g, b in pixels if r > 180 and g < 80 and b < 80)
    assert red_pixels > 10, f"Expected red ellipse pixels, got {red_pixels}"
    plt.close(fig)


def test_ellipse_svg():
    """Ellipse produces <ellipse> element in SVG output."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots()
    e = Ellipse((0.5, 0.5), 0.6, 0.3, color='blue')
    ax.add_patch(e)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert '<ellipse' in svg, "Expected <ellipse> element in SVG"
    plt.close(fig)


def test_arc_renders_png():
    """Arc (0-180 degrees) produces pixels on the upper half."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    a = Arc((0.5, 0.5), 0.6, 0.4, theta1=0, theta2=180, color='green', linewidth=3)
    ax.add_patch(a)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    green_pixels = sum(1 for r, g, b in pixels if g > 150 and r < 80 and b < 80)
    assert green_pixels > 5, f"Expected green arc pixels, got {green_pixels}"
    plt.close(fig)


def test_fancy_bbox_round_png():
    """FancyBboxPatch with 'round' boxstyle fills interior pixels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    p = FancyBboxPatch((0.2, 0.2), 0.6, 0.5, boxstyle='round', facecolor='blue')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    blue_pixels = sum(1 for r, g, b in pixels if b > 150 and r < 80 and g < 80)
    assert blue_pixels > 20, f"Expected blue rounded box pixels, got {blue_pixels}"
    plt.close(fig)


def test_fancy_bbox_square_png():
    """FancyBboxPatch with 'square' boxstyle fills interior pixels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    p = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle='square', facecolor='green')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    green_pixels = sum(1 for r, g, b in pixels if g > 150 and r < 80 and b < 80)
    assert green_pixels > 30, f"Expected green box pixels, got {green_pixels}"
    plt.close(fig)


def test_regular_polygon_svg():
    """RegularPolygon (hexagon) produces <polygon> in SVG output."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    fig, ax = plt.subplots()
    p = RegularPolygon((0.5, 0.5), numVertices=6, radius=0.3, color='purple')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert '<polygon' in svg, "Expected <polygon> in SVG for RegularPolygon"
    plt.close(fig)


def test_arrow_renders_png():
    """Arrow produces filled pixels along its direction."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arrow
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    a = Arrow(0.1, 0.5, 0.8, 0, color='red', width=0.3)
    ax.add_patch(a)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = list(img.getdata())
    red_pixels = sum(1 for r, g, b in pixels if r > 180 and g < 80 and b < 80)
    assert red_pixels > 10, f"Expected red arrow pixels, got {red_pixels}"
    plt.close(fig)


def test_path_patch_svg():
    """PathPatch with a triangle path produces <polygon> in SVG output."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    fig, ax = plt.subplots()
    verts = [(0.2, 0.1), (0.8, 0.1), (0.5, 0.9), (0.2, 0.1)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)
    p = PathPatch(path, facecolor='orange')
    ax.add_patch(p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert '<polygon' in svg, "Expected <polygon> in SVG for PathPatch triangle"
    plt.close(fig)
```

- [ ] **Step 2: Run all 8 tests — all must FAIL**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py -q
```

Expected: 8 failed (stubs don't render; `matplotlib.path` missing for last test).

---

## Task 2: Create `matplotlib.path` Module

**Files:**
- Create: `python/matplotlib/path.py`

The test imports `from matplotlib.path import Path`. This module doesn't exist yet.

- [ ] **Step 1: Create `python/matplotlib/path.py`**

```python
"""matplotlib.path — Path class for defining arbitrary paths."""


class Path:
    """A series of possibly disconnected, possibly closed, line and curve segments.

    Parameters
    ----------
    vertices : list of (x, y) tuples
        The vertices of the path.
    codes : list of int, optional
        Path codes (MOVETO, LINETO, etc.). If None, all are LINETO
        except the first which is MOVETO.
    """

    # Path code constants (match upstream matplotlib values)
    STOP = 0
    MOVETO = 1
    LINETO = 2
    CURVE3 = 3
    CURVE4 = 4
    CLOSEPOLY = 79

    def __init__(self, vertices, codes=None):
        self.vertices = list(vertices)
        if codes is None:
            codes = [self.MOVETO] + [self.LINETO] * (len(vertices) - 1)
        self.codes = list(codes)

    def __len__(self):
        return len(self.vertices)

    def __repr__(self):
        return f"Path({self.vertices!r}, {self.codes!r})"
```

- [ ] **Step 2: Run last test to see updated failure**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_path_patch_svg -q
```

Expected: FAIL — but now with a rendering error (not `ModuleNotFoundError`). The `Path` import works.

- [ ] **Step 3: Commit `path.py`**

```bash
git add python/matplotlib/path.py python/matplotlib/tests/test_patches_upstream.py
git commit -m "feat: add matplotlib.path.Path class and patch test stubs"
```

---

## Task 3: Add `draw_ellipse` to All Three Renderer Files

**Files:**
- Modify: `python/matplotlib/backend_bases.py`
- Modify: `python/matplotlib/_pil_backend.py`
- Modify: `python/matplotlib/_svg_backend.py`

### 3a. Add stub to `backend_bases.py`

Read `backend_bases.py` first. Find where `draw_circle` is defined. Add `draw_ellipse` right after it:

```python
def draw_ellipse(self, cx, cy, rx, ry, angle, facecolor, edgecolor, alpha):
    """Draw an ellipse.

    Parameters
    ----------
    cx, cy : float  Center in display coordinates.
    rx, ry : float  Semi-axis lengths in display pixels.
    angle : float   Rotation in degrees (counter-clockwise).
    facecolor : str or None  Fill color hex string, or None.
    edgecolor : str or None  Stroke color hex string, or None.
    alpha : float   Opacity 0.0–1.0.
    """
    pass
```

- [ ] **Step 1: Add stub to `backend_bases.py`** (after `draw_circle`)

### 3b. Implement in `_pil_backend.py`

Read `_pil_backend.py`. Find where `draw_circle` is. Add `draw_ellipse` right after it:

```python
def draw_ellipse(self, cx, cy, rx, ry, angle, facecolor, edgecolor, alpha):
    import math
    fc = self._parse_color(facecolor) if facecolor and facecolor != 'none' else None
    ec = self._parse_color(edgecolor) if edgecolor and edgecolor != 'none' else None

    if angle == 0:
        # Use PIL's native ellipse for non-rotated ellipses
        bbox = [(int(cx - rx), int(cy - ry)), (int(cx + rx), int(cy + ry))]
        if fc is not None:
            self._draw.ellipse(bbox, fill=fc)
        if ec is not None:
            self._draw.ellipse(bbox, outline=ec)
    else:
        # Approximate rotated ellipse with 36-point polygon
        pts = []
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        for i in range(36):
            t = 2 * math.pi * i / 36
            x = rx * math.cos(t)
            y = ry * math.sin(t)
            px = cx + x * cos_a - y * sin_a
            py = cy - (x * sin_a + y * cos_a)  # negate: screen y-down
            pts.append((int(px), int(py)))
        if fc is not None:
            self._draw.polygon(pts, fill=fc)
        if ec is not None:
            for i in range(len(pts)):
                self._draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=ec, width=1)
```

**IMPORTANT:** Check the actual color-parsing helper name in `_pil_backend.py`. It might be `_to_rgb_255()`, `_parse_color()`, or inline. Read the file and adapt accordingly. Do NOT guess.

- [ ] **Step 2: Add `draw_ellipse` to `_pil_backend.py`**

### 3c. Implement in `_svg_backend.py`

Read `_svg_backend.py`. Find where `draw_circle` is. Add `draw_ellipse` right after it:

```python
def draw_ellipse(self, cx, cy, rx, ry, angle, facecolor, edgecolor, alpha):
    fill = facecolor if facecolor and facecolor != 'none' else 'none'
    stroke = edgecolor if edgecolor and edgecolor != 'none' else 'none'
    transform = ''
    if angle != 0:
        transform = f' transform="rotate({-angle:.2f} {cx:.2f} {cy:.2f})"'
    self._parts.append(
        f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
        f'fill="{fill}" stroke="{stroke}"{transform} />'
    )
```

Note: SVG angles are clockwise; matplotlib uses counter-clockwise. Negate `angle` in the SVG `rotate()`.

**IMPORTANT:** Check the actual list name in `_svg_backend.py` — it's `self._parts`, but verify by reading the file first.

- [ ] **Step 3: Add `draw_ellipse` to `_svg_backend.py`**

- [ ] **Step 4: Run ellipse tests to check draw_ellipse works**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_ellipse_renders_png python/matplotlib/tests/test_patches_upstream.py::test_ellipse_svg -q
```

Expected: both still FAIL (Ellipse has no `draw()` yet). But there should be no AttributeError on the renderer — the error should be in patches.py.

---

## Task 4: Implement `Ellipse.draw()` and `Arc.draw()`

**Files:**
- Modify: `python/matplotlib/patches.py`

Read `patches.py`. Find the `Ellipse` class (around line 215). Add `draw()` right after `set_angle()`.

### 4a. `Ellipse.draw()`

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    cx = layout.sx(self._center[0])
    cy = layout.sy(self._center[1])
    rx = abs(layout.sx(self._center[0] + self._width / 2) - cx)
    ry = abs(layout.sy(self._center[1] + self._height / 2) - cy)
    if rx <= 0 or ry <= 0:
        return
    fc = self._resolved_facecolor_hex()
    ec = self._resolved_edgecolor_hex()
    alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
    renderer.draw_ellipse(cx, cy, rx, ry, self._angle,
                          fc if fc != 'none' else None,
                          ec if ec != 'none' else None,
                          alpha)
```

- [ ] **Step 1: Add `draw()` to `Ellipse`**

- [ ] **Step 2: Run ellipse tests**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_ellipse_renders_png python/matplotlib/tests/test_patches_upstream.py::test_ellipse_svg -q
```

Expected: both PASS.

### 4b. `Arc.draw()`

Find the `Arc` class (around line 250). Add `draw()` after `set_theta2()`.

`Arc` is a **stroked arc** (no fill), drawn as connected line segments:

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    import math
    cx = layout.sx(self._center[0])
    cy = layout.sy(self._center[1])
    rx = abs(layout.sx(self._center[0] + self._width / 2) - cx)
    ry = abs(layout.sy(self._center[1] + self._height / 2) - cy)
    if rx <= 0 or ry <= 0:
        return
    t1 = math.radians(self._theta1)
    t2 = math.radians(self._theta2)
    sweep = abs(t2 - t1)
    n = max(32, int(sweep / (2 * math.pi) * 64))
    xdata = []
    ydata = []
    for i in range(n + 1):
        t = t1 + (t2 - t1) * i / n
        xdata.append(cx + rx * math.cos(t))
        ydata.append(cy - ry * math.sin(t))  # negate: screen y-down
    ec = self._resolved_edgecolor_hex()
    lw = self._linewidth if self._linewidth is not None else 1.0
    renderer.draw_line(xdata, ydata, ec if ec != 'none' else 'black', lw, '-', 1.0)
```

- [ ] **Step 3: Add `draw()` to `Arc`**

- [ ] **Step 4: Run arc test**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_arc_renders_png -q
```

Expected: PASS.

- [ ] **Step 5: Run full suite — no regressions**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q 2>&1 | tail -3
```

Expected: baseline + 3 new = 878 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add python/matplotlib/patches.py \
        python/matplotlib/backend_bases.py python/matplotlib/_pil_backend.py \
        python/matplotlib/_svg_backend.py
git commit -m "feat: Ellipse and Arc draw() + draw_ellipse renderer primitive"
```

---

## Task 5: Implement `FancyBboxPatch.draw()`

**Files:**
- Modify: `python/matplotlib/patches.py`

Find `FancyBboxPatch` (around line 272). Add `_rounded_rect_points` at module level (before class definitions, or as a module-level function near the bottom), then add `draw()` to the class.

### 5a. Module-level helper

Add this function to `patches.py` at module level (e.g., after all imports, before `class Patch`):

```python
def _rounded_rect_points(x0, y0, x1, y1, radius):
    """Generate polygon points approximating a rounded rectangle.

    Parameters
    ----------
    x0, y0 : float  Top-left corner (display coords, y0 < y1).
    x1, y1 : float  Bottom-right corner.
    radius : float  Corner radius in pixels.

    Returns
    -------
    list of (x, y) tuples
    """
    import math
    w = x1 - x0
    h = y1 - y0
    r = min(radius, w / 2, h / 2)
    pts = []
    # 4 corners, 8 arc points each (quarter circle)
    corners = [
        (x0 + r, y0 + r, math.pi,     1.5 * math.pi),  # top-left
        (x1 - r, y0 + r, 1.5 * math.pi, 2 * math.pi),  # top-right
        (x1 - r, y1 - r, 0,           0.5 * math.pi),  # bottom-right
        (x0 + r, y1 - r, 0.5 * math.pi, math.pi),      # bottom-left
    ]
    n_arc = 8
    for cx, cy, t_start, t_end in corners:
        for i in range(n_arc + 1):
            t = t_start + (t_end - t_start) * i / n_arc
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts
```

### 5b. `FancyBboxPatch.draw()`

Add `draw()` to `FancyBboxPatch` after `set_boxstyle()`:

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    # Map data coords to display coords
    x0 = layout.sx(self._xy[0])
    y0 = layout.sy(self._xy[1])
    x1 = layout.sx(self._xy[0] + self._width)
    y1 = layout.sy(self._xy[1] + self._height)
    # Normalise: ensure top < bottom, left < right in screen coords
    x_left, x_right = min(x0, x1), max(x0, x1)
    y_top, y_bot = min(y0, y1), max(y0, y1)
    fc = self._resolved_facecolor_hex()
    alpha = self.get_alpha() if self.get_alpha() is not None else 1.0

    if 'round' in str(self._boxstyle).lower():
        pts = _rounded_rect_points(x_left, y_top, x_right, y_bot, radius=8)
        renderer.draw_polygon(pts, fc if fc != 'none' else '#ffffff', alpha)
    else:
        w = x_right - x_left
        h = y_bot - y_top
        renderer.draw_rect(x_left, y_top, w, h,
                           stroke=None, fill=fc if fc != 'none' else None)
```

- [ ] **Step 1: Add `_rounded_rect_points` to module level in `patches.py`**
- [ ] **Step 2: Add `draw()` to `FancyBboxPatch`**

- [ ] **Step 3: Run FancyBboxPatch tests**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_fancy_bbox_round_png python/matplotlib/tests/test_patches_upstream.py::test_fancy_bbox_square_png -q
```

Expected: both PASS.

- [ ] **Step 4: Run full suite — no regressions**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q 2>&1 | tail -3
```

Expected: 880 passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add python/matplotlib/patches.py
git commit -m "feat: FancyBboxPatch draw() with round and square boxstyles"
```

---

## Task 6: Implement `RegularPolygon.draw()`

**Files:**
- Modify: `python/matplotlib/patches.py`

Find `RegularPolygon` (around line 399). Add `draw()` after `orientation` property:

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    import math
    cx = layout.sx(self._xy_center[0])
    cy = layout.sy(self._xy_center[1])
    # Map radius to display pixels (use x-axis scale)
    r_px = abs(layout.sx(self._xy_center[0] + self._radius) - cx)
    if r_px <= 0:
        return
    n = self._numVertices
    pts = []
    for i in range(n):
        angle = self._orientation + 2 * math.pi * i / n
        px = cx + r_px * math.cos(angle)
        py = cy - r_px * math.sin(angle)  # negate: screen y-down
        pts.append((px, py))
    fc = self._resolved_facecolor_hex()
    alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
    renderer.draw_polygon(pts, fc if fc != 'none' else '#ffffff', alpha)
```

- [ ] **Step 1: Add `draw()` to `RegularPolygon`**

- [ ] **Step 2: Run test**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_regular_polygon_svg -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add python/matplotlib/patches.py
git commit -m "feat: RegularPolygon draw()"
```

---

## Task 7: Implement `Arrow.draw()`

**Files:**
- Modify: `python/matplotlib/patches.py`

Find `Arrow` (around line 387). Add `draw()` after `__init__`:

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    import math
    x0 = layout.sx(self._x)
    y0 = layout.sy(self._y)
    x1 = layout.sx(self._x + self._dx)
    y1 = layout.sy(self._y + self._dy)
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    # Unit vector along arrow direction
    ux = (x1 - x0) / length
    uy = (y1 - y0) / length
    # Perpendicular unit vector
    px = -uy
    py = ux
    # Arrow dimensions in display pixels
    # _arrow_width is a fraction of length; /2 gives half-width for polygon symmetry
    shaft_half_w = self._arrow_width / 2 * length * 0.15
    head_half_w = shaft_half_w * 3.0
    head_len = min(length * 0.35, head_half_w * 2.5)
    shaft_end = length - head_len
    # 7-point polygon: shaft rectangle + arrowhead triangle
    pts = [
        (x0 + px * shaft_half_w,              y0 + py * shaft_half_w),
        (x0 + ux * shaft_end + px * shaft_half_w, y0 + uy * shaft_end + py * shaft_half_w),
        (x0 + ux * shaft_end + px * head_half_w,  y0 + uy * shaft_end + py * head_half_w),
        (x1,                                   y1),
        (x0 + ux * shaft_end - px * head_half_w,  y0 + uy * shaft_end - py * head_half_w),
        (x0 + ux * shaft_end - px * shaft_half_w, y0 + uy * shaft_end - py * shaft_half_w),
        (x0 - px * shaft_half_w,              y0 - py * shaft_half_w),
    ]
    fc = self._resolved_facecolor_hex()
    alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
    renderer.draw_polygon(pts, fc if fc != 'none' else '#000000', alpha)
```

- [ ] **Step 1: Add `draw()` to `Arrow`**

- [ ] **Step 2: Run test**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_arrow_renders_png -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add python/matplotlib/patches.py
git commit -m "feat: Arrow draw() with 7-point polygon"
```

---

## Task 8: Implement `PathPatch.draw()`

**Files:**
- Modify: `python/matplotlib/patches.py`

Find `PathPatch` (around line 422). Add `draw()` after `set_path()`:

```python
def draw(self, renderer, layout):
    if not self.get_visible() or self._path is None:
        return
    # Path code constants (match matplotlib.path.Path)
    MOVETO = 1
    LINETO = 2
    CURVE3 = 3
    CURVE4 = 4
    CLOSEPOLY = 79

    codes = self._path.codes
    verts = self._path.vertices
    fc = self._resolved_facecolor_hex()
    alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
    fc_draw = fc if fc != 'none' else '#ffffff'

    current = []
    for code, vertex in zip(codes, verts):
        x, y = vertex
        sx = layout.sx(x)
        sy = layout.sy(y)
        if code == MOVETO:
            if len(current) >= 3:
                renderer.draw_polygon(current, fc_draw, alpha)
            current = [(sx, sy)]
        elif code == LINETO:
            current.append((sx, sy))
        elif code == CLOSEPOLY:
            if len(current) >= 3:
                renderer.draw_polygon(current, fc_draw, alpha)
            current = []
        elif code in (CURVE3, CURVE4):
            # Simple linearisation: treat control points as vertices
            current.append((sx, sy))
    if len(current) >= 3:
        renderer.draw_polygon(current, fc_draw, alpha)
```

- [ ] **Step 1: Add `draw()` to `PathPatch`**

- [ ] **Step 2: Run test**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py::test_path_patch_svg -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add python/matplotlib/patches.py
git commit -m "feat: PathPatch draw() with path code traversal"
```

---

## Task 9: Final Verification

- [ ] **Step 1: Run all 8 new tests**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_patches_upstream.py -v
```

Expected: 8 passed, 0 failed.

- [ ] **Step 2: Run full test suite**

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

Expected: **883 passed, 0 failed** (875 baseline + 8 new).

If any pre-existing test fails, fix it before proceeding. Do not skip or mark as xfail.

---

## Gotchas Reference

| Issue | Fix |
|-------|-----|
| `sy_raw` doesn't exist on `AxesLayout` | Use `abs(layout.sy(center_y + half_height) - cy_px)` |
| `to_hex` on 'none' string | `_resolved_facecolor_hex()` returns `'none'` — check before passing to renderer |
| PIL `ellipse()` ignores rotation | Use 36-pt polygon approximation for `angle != 0` |
| SVG rotate direction | Negate `angle` in `rotate()` transform (SVG is CW, mpl is CCW) |
| `matplotlib.path` missing | Created in Task 2; constants also defined locally in `PathPatch.draw()` |
| Alpha is `None` by default | Always guard with `if self.get_alpha() is not None else 1.0` |
| Screen y is inverted | Negate `sin()` for y coordinates in all trig calculations |
| `_pil_backend.py` color helper | Check actual name (`_to_rgb_255`, `_parse_color`, or inline) before using |
| `_svg_backend.py` parts list | Verify `self._parts` by reading the file — not `self._elements` |
