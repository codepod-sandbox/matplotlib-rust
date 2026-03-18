# Colormaps Sub-project Design Spec

**Date:** 2026-03-18
**Status:** Approved

## Goal

Add full upstream matplotlib colormap support to matplotlib-rust: all ~150 named colormaps, the `Colormap`/`LinearSegmentedColormap`/`ListedColormap` class hierarchy, `get_cmap`, `ScalarMappable`, and associated norm classes. This is the foundational sub-project that unblocks `imshow`, `colorbar`, `contourf`, and scatter per-point coloring.

## Architecture

### Files

| File | Action | Description |
|------|--------|-------------|
| `python/matplotlib/_cm.py` | Copy + adapt | Segment data dicts for all classic colormaps (jet, hot, cool, bone, copper, etc.) — imports `functools.partial` and `numpy` |
| `python/matplotlib/_cm_listed.py` | Copy + adapt | Raw float RGBA list data for perceptual maps (viridis, plasma, inferno, magma, cividis, turbo, tab10, tab20, Set1, Set2, Paired, etc.) — strip `ListedColormap` import, keep raw data only |
| `python/matplotlib/colors.py` | Extend | Add `Colormap`, `LinearSegmentedColormap`, `ListedColormap`, `BoundaryNorm`, `TwoSlopeNorm`, `CenteredNorm` alongside existing `Normalize`, `LogNorm` |
| `python/matplotlib/cm.py` | New | `ColormapRegistry`, `_colormaps` global registry, `get_cmap`, `ScalarMappable` |
| `python/matplotlib/rcsetup.py` | Modify | Add `'image.cmap': 'viridis'` and `'image.lut': 256` to `_default_params` |
| `python/matplotlib/tests/test_cm_upstream.py` | New | Upstream-ported tests |

### Dependency graph

```
_cm.py          (imports functools.partial + numpy; no matplotlib imports)
_cm_listed.py   (raw list data only; no imports)
    ↓
colors.py       (Colormap classes; imports numpy)
    ↓
cm.py           (registry + ScalarMappable; imports colors + _cm + _cm_listed)
    ↓
axes.py / figure.py  (consumers — unchanged in this sub-project)
```

## Component Design

### `_cm.py`

Copy from upstream matplotlib 3.9.x. `functools.partial` is used and should be kept as-is (RustPython supports `functools`). Remove only `from matplotlib import ...` imports. Keep `import numpy as np` and `import functools`.

### `_cm_listed.py`

Copy from upstream, but **strip the `from .colors import ListedColormap` import and all `ListedColormap(...)` constructor calls**. Store only the raw Python lists of RGBA floats (e.g., `_viridis_data = [[0.267, 0.004, 0.329, 1.0], ...]`). `cm.py` will wrap these in `ListedColormap` at registry construction time, avoiding the circular import (`_cm_listed` → `colors` → `cm` → `_cm_listed`).

### `colors.py` extensions

Copy the following classes from upstream `matplotlib/colors.py` and append to the existing file.

**RustPython adaptation rules (apply to all classes):**
- Remove `from matplotlib import _api` — replace `_api.check_isinstance` / `_api.check_in_list` with plain `isinstance` / `if x not in [...]` checks
- Remove `from matplotlib import cbook` — inline or drop any `cbook` usage
- Keep `import numpy as np` — numpy-rust provides this
- Remove `matplotlib._cm_multivar` references (not needed)
- **Masked array adaptation** — `numpy.ma` is a stub in numpy-rust. Replace all `np.ma.*` usage in `Colormap.__call__` with NaN-based equivalents:
  - `np.ma.is_masked(X)` → `np.any(np.isnan(X))`
  - `np.ma.masked_array(...)` → use `np.where(np.isnan(X), np.nan, X)`
  - `np.ma.filled(X, fill_value)` → `np.where(np.isnan(X), fill_value, X)`
  - Out-of-range (under/over) handling: use `np.clip` + explicit NaN checks
- **`np.interp` may not be in numpy-rust** — check at implementation time. If missing, add a pure-Python fallback:
  ```python
  def _interp(x, xp, fp):
      """Pure-Python piecewise linear interpolation (fallback for np.interp).
      Assumes xp is strictly monotone increasing (segment data always is)."""
      result = []
      for xi in x:
          if xi <= xp[0]:
              result.append(fp[0])
          elif xi >= xp[-1]:
              result.append(fp[-1])
          else:
              for i in range(len(xp) - 1):
                  if xp[i] <= xi <= xp[i+1]:
                      dx = xp[i+1] - xp[i]
                      if dx == 0:
                          result.append(fp[i])
                      else:
                          t = (xi - xp[i]) / dx
                          result.append(fp[i] + t * (fp[i+1] - fp[i]))
                      break
      return np.array(result)
  ```
  Then replace `np.interp(...)` calls in `LinearSegmentedColormap._init()` with `_interp(...)`.

**`_lut` initialization — lazy strategy:**
Both `LinearSegmentedColormap` and `ListedColormap` build a `_lut` array in their `_init()` method. Call `_init()` lazily on first `__call__` (check `if not hasattr(self, '_lut'): self._init()`). With ~150 colormaps registered at module import time, eager initialization would be very slow in WASM.

---

**`Colormap`** — abstract base:
- `__init__(name, N=256)` — stores name and number of LUT entries
- `__call__(X, alpha=None, bytes=False)` — maps scalar/array in [0,1] to RGBA 4-tuples; adapted to use NaN instead of masked arrays (see above)
- `reversed(name=None)` — returns a reversed copy
- `__copy__` / `__eq__`
- `is_gray()` — keep
- `set_bad(color, alpha=None)` / `set_under(color, alpha=None)` / `set_over(color, alpha=None)` — keep

**`LinearSegmentedColormap(Colormap)`** — piecewise-linear LUT:
- `__init__(name, segmentdata, N=256, gamma=1.0)`
- `_init()` — builds `_lut` array from segmentdata using `np.interp` (or `_interp` fallback); called lazily
- `from_list(name, colors, N=256)` — classmethod
- `set_gamma(gamma)` — resets `_lut` cache

**`ListedColormap(Colormap)`** — fixed list of colors:
- `__init__(colors, name='from_list', N=None)` — `N=None` defaults to `len(colors)`
- `_init()` — builds `_lut` from colors list via `to_rgba_array`; called lazily

**New norm classes:**
- `BoundaryNorm(Normalize)` — `__init__(boundaries, ncolors, clip=False)`; maps values to integer bins; `boundaries` must be monotonically increasing
- `TwoSlopeNorm(Normalize)` — `__init__(vcenter, vmin=None, vmax=None)`; separate linear scaling below/above center
- `CenteredNorm(Normalize)` — `__init__(vcenter=0, halfrange=None)`; symmetric around center

### `cm.py`

New file. Copy and adapt from upstream `matplotlib/cm.py`.

**`ColormapRegistry`** — simplified dict-like registry:
- `__init__(cmaps)` — populates internal `_cmaps` dict from provided mapping
- `__getitem__(item)` — returns colormap by name; raises `KeyError` for unknown names
- `__iter__` / `__len__` / `__contains__`
- `register(cmap, *, name=None, force=False)` — user registration; raises `ValueError` if name exists and `force=False`
- `get_cmap(name=None, lut=None)` — main lookup: `None` → `rcParams['image.cmap']`; string → registry lookup (raises `ValueError` for unknown, not `KeyError`); `Colormap` instance → return as-is; `lut` resamples if provided
- Drop: GUI cache invalidation callbacks (`_stale_caches`), `_builtin_cmaps` locking, `__call__` alias

**`_colormaps`** — module-level `ColormapRegistry` pre-populated at import time. Construction:
1. From `_cm.py`: iterate `datad` dict, wrap each in `LinearSegmentedColormap(name, data, N=256)`
2. From `_cm_listed.py`: wrap each raw list in `ListedColormap(data, name=name)`
3. Add `_r` (reversed) variants for each

**`get_cmap(name=None, lut=None)`** — module-level function, delegates to `_colormaps.get_cmap(name, lut)`.

**`ScalarMappable`** — mixin for artists mapping scalar data to colors:
- `__init__(norm=None, cmap=None)` — `cmap` accepts name string or `Colormap`; defaults to `rcParams['image.cmap']`
- `set_array(A)` — stores data array as `self._A`
- `get_array()` → `self._A`
- `set_cmap(cmap)` — accepts name string or `Colormap` instance; resolves via `get_cmap`
- `get_cmap()` → stored `Colormap`
- `set_norm(norm)` — accepts `Normalize` instance or `None`
- `get_norm()` → stored `Normalize`
- `get_clim()` → `(self.norm.vmin, self.norm.vmax)`
- `set_clim(vmin=None, vmax=None)` — updates `norm.vmin` / `norm.vmax`
- `autoscale()` — sets `vmin`/`vmax` from `self._A` (requires array set first)
- `autoscale_None()` — only sets unset limits
- `to_rgba(x, alpha=None, bytes=False, norm=True)` — applies norm (if `norm=True`) then cmap to array `x`; returns RGBA array of shape `(*x.shape, 4)` or `(*x.shape, 4)` uint8 if `bytes=True`
- Drop: `callbacksSM`, `stale_callback`, event infrastructure

**`rcsetup.py` changes** — add to `_default_params` dict:
```python
'image.cmap': 'viridis',
'image.lut': 256,
```

## Testing

**File:** `python/matplotlib/tests/test_cm_upstream.py`

Ported from upstream `lib/matplotlib/tests/test_cm.py` and colormap sections of `lib/matplotlib/tests/test_colors.py`.

| Test | What it covers |
|------|----------------|
| `test_get_cmap_return_default` | `get_cmap()` returns viridis (default from rcParams) |
| `test_get_cmap_by_name` | `get_cmap('hot')`, `get_cmap('viridis')`, `get_cmap('Blues')` all return `Colormap` |
| `test_get_cmap_bad_name` | `get_cmap('nonexistent')` raises `ValueError` (not `KeyError`) |
| `test_register_cmap` | User-defined `ListedColormap` can be registered and retrieved |
| `test_colormap_call_scalar` | `cmap(0.0)`, `cmap(0.5)`, `cmap(1.0)` each return a 4-tuple of floats |
| `test_colormap_call_array` | `cmap(np.linspace(0, 1, 10))` returns shape `(10, 4)` float array |
| `test_colormap_bytes` | `cmap(0.5, bytes=True)` returns 4-tuple of ints in [0, 255] |
| `test_listed_colormap` | `ListedColormap(['r', 'g', 'b'])(0.0)` returns red RGBA; `N=None` defaults to 3 |
| `test_linear_segmented_from_list` | `LinearSegmentedColormap.from_list('x', ['blue', 'red'])(0.5)` returns valid RGBA |
| `test_colormap_reversed` | `cmap.reversed()(0.0)` equals `cmap(1.0)` within tolerance |
| `test_scalar_mappable_to_rgba` | `ScalarMappable(norm=Normalize(0,1), cmap='viridis').to_rgba(np.array([0., 0.5, 1.]))` has shape `(3, 4)` |
| `test_scalar_mappable_set_clim` | `sm.set_clim(0, 10)` → `sm.get_clim() == (0, 10)` |
| `test_scalar_mappable_autoscale` | After `set_array([1,2,3])`, `autoscale()` sets vmin=1, vmax=3 |
| `test_boundary_norm` | `BoundaryNorm([0,1,2,3], ncolors=3)(1.5)` returns value in middle bin |
| `test_two_slope_norm` | `TwoSlopeNorm(vcenter=0, vmin=-1, vmax=2)(np.array([-1,0,2]))` → `[0, 0.5, 1]` |
| `test_centered_norm` | `CenteredNorm()(np.array([-1., 0., 1.]))` → `[0.0, 0.5, 1.0]` |
| `test_all_colormaps_callable` | Every name in `cm._colormaps` can be called with `cmap(0.5)` without error |
| `test_rcparam_image_cmap` | `rcParams['image.cmap']` exists and defaults to `'viridis'` |
| `test_registry_getitem_keyerror` | `cm._colormaps['bad']` raises `KeyError` |
| `test_scalar_mappable_norm_false` | `sm.to_rgba(x, norm=False)` skips normalization step |

## Build & Test Commands

```bash
# From packages/matplotlib-py/
cargo build -p matplotlib-python
target/debug/matplotlib-python -m pytest python/matplotlib/tests/test_cm_upstream.py -v
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q  # full suite (848 baseline)
```

## What This Does NOT Include

- `imshow` / `AxesImage` — sub-project D
- `fig.colorbar()` — sub-project D
- `contour` / `contourf` — sub-project F
- Per-point scatter colors — sub-project F
- GUI colormap picker / interactive registration

## Copyright

`_cm.py` and `_cm_listed.py` copied and adapted from matplotlib — retain original copyright block. `cm.py` and `colors.py` additions: copied and modified — add CodePod BSD-3 header alongside original matplotlib copyright.
