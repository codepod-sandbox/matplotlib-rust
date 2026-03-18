# Colormaps Sub-project Design Spec

**Date:** 2026-03-18
**Status:** Approved

## Goal

Add full upstream matplotlib colormap support to matplotlib-rust: all ~150 named colormaps, the `Colormap`/`LinearSegmentedColormap`/`ListedColormap` class hierarchy, `get_cmap`, `ScalarMappable`, and associated norm classes. This is the foundational sub-project that unblocks `imshow`, `colorbar`, `contourf`, and scatter per-point coloring.

## Architecture

### Files

| File | Action | Description |
|------|--------|-------------|
| `python/matplotlib/_cm.py` | Copy verbatim | Segment data dicts for all classic colormaps (jet, hot, cool, bone, copper, etc.) |
| `python/matplotlib/_cm_listed.py` | Copy verbatim | Float RGBA arrays for perceptual maps (viridis, plasma, inferno, magma, cividis, turbo, tab10, tab20, Set1, Set2, Paired, etc.) |
| `python/matplotlib/colors.py` | Extend | Add `Colormap`, `LinearSegmentedColormap`, `ListedColormap`, `BoundaryNorm`, `TwoSlopeNorm`, `CenteredNorm` alongside existing `Normalize`, `LogNorm` |
| `python/matplotlib/cm.py` | New | `ColormapRegistry`, `_colormaps` global registry, `get_cmap`, `ScalarMappable` |
| `python/matplotlib/tests/test_cm_upstream.py` | New | Upstream-ported tests |

### Dependency graph

```
_cm.py          (pure data, no imports)
_cm_listed.py   (pure data, no imports)
    ↓
colors.py       (Colormap classes use numpy)
    ↓
cm.py           (registry + ScalarMappable, imports colors + _cm + _cm_listed)
    ↓
axes.py / figure.py  (consumers — unchanged in this sub-project)
```

## Component Design

### `_cm.py` and `_cm_listed.py`

Copy verbatim from upstream matplotlib 3.9.x. No adaptation needed — these are pure Python data files with no imports beyond `numpy` (which numpy-rust provides).

### `colors.py` extensions

Copy the following classes from upstream `matplotlib/colors.py` and add to the existing file:

**`Colormap`** — abstract base:
- `__init__(name, N=256)` — stores name and number of LUT entries
- `__call__(X, alpha=None, bytes=False)` — maps scalar/array in [0,1] to RGBA; must handle scalars, 1-D arrays, masked arrays
- `reversed(name=None)` — returns a reversed copy
- `__copy__` / `__eq__`
- Drop: `_init` abstract method (inline into subclasses); keep `is_gray()`

**`LinearSegmentedColormap(Colormap)`** — piecewise-linear LUT:
- `__init__(name, segmentdata, N=256, gamma=1.0)`
- `_init()` — builds `_lut` array from segmentdata using numpy interpolation
- `from_list(name, colors, N=256)` — class method
- `set_gamma(gamma)`

**`ListedColormap(Colormap)`** — fixed list of colors:
- `__init__(colors, name='from_list', N=None)`
- `_init()` — builds `_lut` from colors list

**Adaptation rules for colormap classes:**
- Remove `from matplotlib import _api` — replace `_api.check_isinstance` with `isinstance` checks
- Remove `from matplotlib import cbook` — inline any `cbook` usage
- Keep `import numpy as np` — numpy-rust provides this
- Remove `matplotlib._cm_multivar` references (not needed)
- `set_bad`, `set_under`, `set_over` — keep (used by LogNorm masked values)

**New norm classes to add:**
- `BoundaryNorm(Normalize)` — maps values to integer bins defined by boundaries array
- `TwoSlopeNorm(Normalize)` — separate linear scaling below/above a center value
- `CenteredNorm(Normalize)` — centers norm at zero with symmetric limits

**Adaptation rules for norm classes:** same as colormap classes. Remove `_api`, `cbook`. Keep numpy.

### `cm.py`

New file. Copy and adapt from upstream `matplotlib/cm.py`:

**`ColormapRegistry`** — simplified dict-like registry:
- `__init__(cmaps)` — populates from `_cm.py` + `_cm_listed.py` data
- `__getitem__(item)` — returns colormap by name, raises `KeyError` for unknown
- `__call__(name=None, lut=None)` — alias for `get_cmap`
- `register(cmap, name=None, force=False)` — user registration
- `get_cmap(name=None, lut=None)` — returns colormap; `None` → default (`rcParams['image.cmap']`)
- Drop: GUI cache invalidation callbacks, `_stale_caches`, `_builtin_cmaps` locking

**`_colormaps`** — module-level `ColormapRegistry` instance pre-populated with all upstream maps. Constructed at module import time from `_cm.py` + `_cm_listed.py`.

**`get_cmap(name=None, lut=None)`** — module-level convenience function delegating to `_colormaps`.

**`ScalarMappable`** — mixin for artists that map scalar data to colors:
- `__init__(norm=None, cmap=None)`
- `set_array(A)` — stores data array
- `get_array()` — returns stored array
- `set_cmap(cmap)` — accepts name string or `Colormap` instance
- `set_norm(norm)` — accepts `Normalize` instance or `None`
- `get_clim()` → `(vmin, vmax)`
- `set_clim(vmin=None, vmax=None)`
- `autoscale()` — sets vmin/vmax from data array
- `autoscale_None()` — only sets limits not already set
- `to_rgba(x, alpha=None, bytes=False)` — applies norm then cmap to array `x`
- Drop: `callbacksSM`, `stale_callback`, `_A` event infrastructure

**`rcParams` default to add in `rcsetup.py`:**
```python
'image.cmap': 'viridis',
'image.lut': 256,
```

## Testing

**File:** `python/matplotlib/tests/test_cm_upstream.py`

Ported from upstream `lib/matplotlib/tests/test_cm.py` and colormap sections of `lib/matplotlib/tests/test_colors.py`. Required tests:

| Test | What it covers |
|------|----------------|
| `test_get_cmap_return_default` | `get_cmap()` returns viridis (default) |
| `test_get_cmap_by_name` | `get_cmap('hot')`, `get_cmap('viridis')`, `get_cmap('Blues')` all return Colormap |
| `test_get_cmap_bad_name` | `get_cmap('nonexistent')` raises `ValueError` |
| `test_register_cmap` | User-defined `ListedColormap` can be registered and retrieved |
| `test_colormap_call_scalar` | `cmap(0.0)`, `cmap(0.5)`, `cmap(1.0)` return RGBA 4-tuples |
| `test_colormap_call_array` | `cmap(np.linspace(0, 1, 10))` returns shape `(10, 4)` array |
| `test_colormap_bytes` | `cmap(0.5, bytes=True)` returns uint8 RGBA tuple |
| `test_listed_colormap` | `ListedColormap(['r', 'g', 'b'])(0.0)` returns red RGBA |
| `test_linear_segmented_from_list` | `LinearSegmentedColormap.from_list('x', ['blue','red'])` works |
| `test_colormap_reversed` | `cmap.reversed()(0.0) == cmap(1.0)` |
| `test_scalar_mappable_to_rgba` | `ScalarMappable(norm=Normalize(0,1), cmap='viridis').to_rgba(array)` returns correct shape |
| `test_scalar_mappable_set_clim` | `set_clim(0, 10)` → `get_clim() == (0, 10)` |
| `test_scalar_mappable_autoscale` | `autoscale()` sets vmin/vmax from array |
| `test_boundary_norm` | `BoundaryNorm([0,1,2,3], ncolors=3)(1.5)` returns correct bin |
| `test_two_slope_norm` | `TwoSlopeNorm(vcenter=0)(array)` maps negative/positive separately |
| `test_centered_norm` | `CenteredNorm()([-1, 0, 1])` → `[0.0, 0.5, 1.0]` |
| `test_all_colormaps_callable` | Every name in `cm._colormaps` can be called with `cmap(0.5)` |
| `test_rcparam_image_cmap` | `rcParams['image.cmap']` exists and defaults to `'viridis'` |

## Build & Test Commands

```bash
# From packages/matplotlib-py/
cargo build -p matplotlib-python
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_cm_upstream.py -v
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q  # full suite
```

## What This Does NOT Include

- `imshow` / `AxesImage` — next sub-project (D)
- `fig.colorbar()` — next sub-project (D)
- `contour` / `contourf` — sub-project F
- Per-point scatter colors — sub-project F
- GUI colormap picker / interactive registration

## Copyright

`_cm.py` and `_cm_listed.py` copied verbatim from matplotlib — retain original copyright. `cm.py` and `colors.py` additions: copied and modified — add CodePod BSD-3 header alongside original matplotlib copyright.
