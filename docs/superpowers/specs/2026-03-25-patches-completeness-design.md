# Spec: Sub-project C — Patches Completeness

**Date:** 2026-03-25
**Branch:** `feat/patches`
**Worktree:** `packages/matplotlib-py/.worktrees/patches/`

## Goal

Add `draw()` methods to 6 stub patch classes (`Ellipse`, `Arc`, `FancyBboxPatch`, `RegularPolygon`, `Arrow`, `PathPatch`) and one new renderer primitive (`draw_ellipse`). No Rust changes required.

## Baseline

875 passed, 0 failed. No regressions allowed.

## Architecture

The renderer already exposes `draw_polygon`, `draw_circle`, `draw_line`, `draw_rect`, `draw_arrow`. This sub-project adds one new primitive (`draw_ellipse`) and uses existing primitives for all 6 patches.

```
patches.py (Ellipse/Arc/FancyBboxPatch/RegularPolygon/Arrow/PathPatch)
    ↓ layout.sx/sy (data→display coords)
    ↓
backend_bases.RendererBase.draw_ellipse / draw_polygon / draw_line / draw_rect
    ↓
RendererPIL: PIL ImageDraw methods
RendererSVG: SVG element strings → self._parts
```

## File Map

| File | Action |
|------|--------|
| `python/matplotlib/patches.py` | Add `draw()` to 6 classes |
| `python/matplotlib/backend_bases.py` | Add `draw_ellipse()` signature |
| `python/matplotlib/_pil_backend.py` | Implement `draw_ellipse()` |
| `python/matplotlib/_svg_backend.py` | Implement `draw_ellipse()` |
| `python/matplotlib/tests/test_patches_upstream.py` | 8 new tests |

## Renderer Primitive: `draw_ellipse`

Signature:

```python
def draw_ellipse(self, cx, cy, rx, ry, angle, facecolor, edgecolor, alpha):
    """Draw a filled (or outlined) ellipse.

    Parameters
    ----------
    cx, cy : float
        Center in display coordinates.
    rx, ry : float
        Semi-axis lengths in display coordinates.
    angle : float
        Rotation angle in degrees (counter-clockwise).
    facecolor : str or None
        Fill color (hex string or None for no fill).
    edgecolor : str or None
        Stroke color (hex string or None for no stroke).
    alpha : float
        Opacity, 0.0–1.0.
    """
    pass
```

**PIL implementation:** Compute the bounding box `[(cx-rx, cy-ry), (cx+rx, cy+ry)]` and call `self._draw.ellipse(bbox, fill=fc)`. For `angle != 0`, approximate the rotated ellipse with a 36-point polygon via trigonometry (PIL's native ellipse doesn't support rotation).

**SVG implementation:**

```xml
<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"
         fill="{facecolor}" stroke="{edgecolor}"
         transform="rotate({angle} {cx} {cy})" />
```

If `facecolor` is None, use `fill="none"`. If `edgecolor` is None, use `stroke="none"`.

## Patch Implementations

### Ellipse

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    cx, cy = layout.sx(self._center[0]), layout.sy(self._center[1])
    # Scale half-widths to display coords
    rx = abs(layout.sx(self._center[0] + self._width / 2) - cx)
    ry = abs(layout.sy_raw(self._height / 2))  # height in display units
    fc = to_hex(self._facecolor) if self._facecolor not in (None, 'none') else None
    ec = to_hex(self._edgecolor) if self._edgecolor not in (None, 'none') else None
    renderer.draw_ellipse(cx, cy, rx, ry, self._angle, fc, ec, self._alpha)
```

Note: `layout.sy` maps data y to screen y. For extents, use the layout scale factor directly (data→display conversion without the offset flip). Check existing patterns in `Circle.draw()` and `Wedge.draw()` for how size scaling is done.

### Arc

An Arc is an open elliptical arc (no fill, just the stroke from `theta1` to `theta2`). Implement by generating points along the ellipse perimeter between the two angles and drawing line segments:

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    import math
    cx, cy = layout.sx(self._center[0]), layout.sy(self._center[1])
    rx = abs(layout.sx(self._center[0] + self._width / 2) - cx)
    ry = abs(layout.sy_raw(self._height / 2))
    # Generate arc points
    t1, t2 = math.radians(self._theta1), math.radians(self._theta2)
    n = max(32, int(abs(t2 - t1) / (2 * math.pi) * 64))
    angles = [t1 + (t2 - t1) * i / n for i in range(n + 1)]
    xdata = [cx + rx * math.cos(a) for a in angles]
    ydata = [cy - ry * math.sin(a) for a in angles]  # negate: screen y-down
    ec = to_hex(self._edgecolor) if self._edgecolor else 'black'
    lw = self._linewidth if hasattr(self, '_linewidth') else 1.0
    renderer.draw_line(xdata, ydata, ec, lw, '-', 1.0)
```

### FancyBboxPatch

Two boxstyle modes:

- **`'square'` (default or any non-round style):** Delegate to `draw_rect`.
- **`'round'`:** Generate a rounded-rectangle polygon (4 straight edges + 4 quarter-circle arcs at corners, 8 points per arc). Pad is extracted from boxstyle string if present (e.g., `'round,pad=0.1'`).

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    x0, y0 = layout.sx(self._xy[0]), layout.sy(self._xy[1])
    x1 = layout.sx(self._xy[0] + self._width)
    y1 = layout.sy(self._xy[1] + self._height)  # y1 < y0 in screen coords
    # Normalise so y_top < y_bottom in display coords
    y_top, y_bot = min(y0, y1), max(y0, y1)
    x_left, x_right = min(x0, x1), max(x0, x1)
    fc = to_hex(self._facecolor) if self._facecolor not in (None, 'none') else None
    if 'round' in str(self._boxstyle):
        pts = _rounded_rect_points(x_left, y_top, x_right, y_bot, radius=8)
        renderer.draw_polygon(pts, fc or '#ffffff', self._alpha)
    else:
        w, h = x_right - x_left, y_bot - y_top
        renderer.draw_rect(x_left, y_top, w, h, stroke=None, fill=fc)
```

Module-level helper `_rounded_rect_points(x0, y0, x1, y1, radius)` generates a polygon approximating a rounded rectangle (8 quarter-circle points per corner, clamped so radius ≤ min(w,h)/2).

### RegularPolygon

Compute `numVertices` evenly-spaced vertices around `(cx, cy)` with `radius` in data units, rotated by `orientation` radians:

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    import math
    cx, cy = layout.sx(self._xy_center[0]), layout.sy(self._xy_center[1])
    # Map radius to display coords (use x-axis scale)
    r_display = abs(layout.sx(self._xy_center[0] + self._radius) - cx)
    n = self._numVertices
    pts = []
    for i in range(n):
        angle = self._orientation + 2 * math.pi * i / n
        px = cx + r_display * math.cos(angle)
        py = cy - r_display * math.sin(angle)  # negate: screen y-down
        pts.append((px, py))
    fc = to_hex(self._facecolor) if self._facecolor not in (None, 'none') else '#ffffff'
    renderer.draw_polygon(pts, fc, self._alpha)
```

### Arrow

Compute a 7-point arrowhead polygon from `(x, y)` in direction `(dx, dy)`:

```python
def draw(self, renderer, layout):
    if not self.get_visible():
        return
    import math
    x0, y0 = layout.sx(self._x), layout.sy(self._y)
    x1 = layout.sx(self._x + self._dx)
    y1 = layout.sy(self._y + self._dy)
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    # Unit vectors
    ux = (x1 - x0) / length
    uy = (y1 - y0) / length
    px, py = -uy, ux  # perpendicular
    hw = self._arrow_width / 2 * length * 0.15  # half-shaft width
    hw_head = hw * 3                              # half-head width
    head_len = min(length * 0.3, hw_head * 2)     # arrowhead length
    shaft_end = length - head_len
    # 7-point polygon (shaft + triangle)
    pts = [
        (x0 + px * hw,       y0 + py * hw),
        (x0 + ux * shaft_end + px * hw, y0 + uy * shaft_end + py * hw),
        (x0 + ux * shaft_end + px * hw_head, y0 + uy * shaft_end + py * hw_head),
        (x1, y1),
        (x0 + ux * shaft_end - px * hw_head, y0 + uy * shaft_end - py * hw_head),
        (x0 + ux * shaft_end - px * hw, y0 + uy * shaft_end - py * hw),
        (x0 - px * hw,       y0 - py * hw),
    ]
    fc = to_hex(self._facecolor) if self._facecolor not in (None, 'none') else '#000000'
    renderer.draw_polygon(pts, fc, self._alpha)
```

### PathPatch

Walk `self._path.codes` and `self._path.vertices`. Group MOVETO+LINETO sequences into closed sub-polygons. CURVE3/CURVE4 control points are linearised with 8 intermediate segments using de Casteljau's algorithm.

```python
def draw(self, renderer, layout):
    if not self.get_visible() or self._path is None:
        return
    codes = self._path.codes   # list of Path.code ints
    verts = self._path.vertices  # list of (x, y)
    fc = to_hex(self._facecolor) if self._facecolor not in (None, 'none') else None
    ec = to_hex(self._edgecolor) if self._edgecolor not in (None, 'none') else 'black'
    current = []
    for code, (x, y) in zip(codes, verts):
        sx, sy = layout.sx(x), layout.sy(y)
        if code == Path.MOVETO:
            if current:
                renderer.draw_polygon(current, fc or '#ffffff', self._alpha)
            current = [(sx, sy)]
        elif code == Path.LINETO:
            current.append((sx, sy))
        elif code == Path.CLOSEPOLY:
            if current:
                renderer.draw_polygon(current, fc or '#ffffff', self._alpha)
            current = []
        elif code in (Path.CURVE3, Path.CURVE4):
            # Linearise: append endpoint only (simple approximation)
            current.append((sx, sy))
    if current:
        renderer.draw_polygon(current, fc or '#ffffff', self._alpha)
```

Note: `Path` must be imported from `matplotlib.path` or defined locally. Check if `matplotlib.path.Path` exists; if not, use integer constants directly (MOVETO=1, LINETO=2, CURVE3=3, CURVE4=4, CLOSEPOLY=79).

## Test File

`python/matplotlib/tests/test_patches_upstream.py` — 8 tests:

```python
"""Tests for patch rendering completeness (sub-project C).

Baseline: 875 passed, 0 failed.
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
    """Arc (0°–180°) produces pixels on the upper half."""
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
    """FancyBboxPatch with 'square' boxstyle is equivalent to a filled rectangle."""
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

## Execution Order

1. Create test file with all 8 tests. Verify all 8 FAIL (stub patches don't render).
2. Add `draw_ellipse()` to `backend_bases.py`, `_pil_backend.py`, `_svg_backend.py`.
3. Add `draw()` to `Ellipse`. Run tests 1–2 → pass. Commit.
4. Add `draw()` to `Arc`. Run test 3 → pass. Commit.
5. Add `draw()` to `FancyBboxPatch` + `_rounded_rect_points` helper. Run tests 4–5 → pass. Commit.
6. Add `draw()` to `RegularPolygon`. Run test 6 → pass. Commit.
7. Add `draw()` to `Arrow`. Run test 7 → pass. Commit.
8. Add `draw()` to `PathPatch`. Run test 8 → pass. Commit.
9. Full suite: `target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q` → 883 passed, 0 failed.

## Gotchas

- **`layout.sy` flip:** Screen y increases downward; data y increases upward. For extent/radius mapping, only use the scale factor, not the offset. Check how `Circle.draw()` and `Wedge.draw()` handle this — use the same pattern.
- **`to_hex` helper:** Already exists in `patches.py` (imported from `colors.py`). Use it for facecolor/edgecolor conversion.
- **`Path` constants:** If `matplotlib.path` module exists, import from it. Otherwise define constants locally: `MOVETO=1, LINETO=2, CURVE3=3, CURVE4=4, CLOSEPOLY=79`.
- **Ellipse rotation in PIL:** PIL's native `ellipse()` doesn't support rotation. For `angle != 0`, use the 36-point polygon approximation. For `angle == 0`, use native `ellipse()`.
- **SVG `transform`:** The `rotate(angle cx cy)` form rotates around `(cx, cy)`. Angle is in degrees.
- **Baseline invariant:** 875 passed must not regress. Run full suite after each commit.
