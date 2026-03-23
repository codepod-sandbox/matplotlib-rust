# Upstream Matplotlib Compatibility — Design

**Date:** 2026-03-17
**Status:** Approved

## Goal

Expand matplotlib-rust's upstream API coverage across five areas: tick formatters/locators, log/symlog scales, legend rendering, arrow annotations, and artist properties (alpha, zorder, linestyle). Each phase adds working implementation plus upstream tests ported from the real matplotlib test suite.

## Repo Rename

`matplotlib-py` → `matplotlib-rust`. Rename the GitHub repository accordingly. Follows the `numpy-rust` / `pillow-rust` naming convention and correctly reflects that the project includes a Rust backend.

## Licensing

- **Repo root:** Add a `LICENSE` file (BSD-3-Clause) and set the license field on the GitHub repository.
- **Copied files** (from matplotlib): retain the original matplotlib copyright block verbatim at the top, followed by our BSD-3-Clause header.
- **New files:** BSD-3-Clause header only.
- matplotlib uses a PSF-derived BSD-compatible license; BSD-3 is compatible.

## Architecture

```
matplotlib-rust/
  python/matplotlib/
    ticker.py          ← copied from matplotlib, adapted
    scale.py           ← copied from matplotlib, adapted
    axis.py            ← new: XAxis/YAxis wrappers (Phase 1), Scale integration (Phase 2)
    axes.py            ← modified: integrate scale/axis objects, migrate _xticks/_yticks
    legend.py          ← copied from matplotlib, adapted
    patches.py         ← modified: add FancyArrowPatch
    backend_bases.py   ← modified: draw_arrow primitive
    _svg_backend.py    ← modified: draw_arrow (SVG)
    _pil_backend.py    ← modified: draw_arrow (PIL)
    artist.py          ← exists: consolidate per-class property duplication
  crates/
    matplotlib-python/ ← existing RustPython binary
    matplotlib-ticker/ ← new (if _ticker C extension needed)
```

**Dependency boundary:** `numpy-rust` is used as-is for array operations. Any functionality that real matplotlib implements via its own C extensions (e.g., `matplotlib._ticker`) is owned here — reimplemented in Python first, with a Rust crate fallback if performance or missing runtime APIs require it.

## Phase 1 — `matplotlib.ticker`

**Files:** `python/matplotlib/ticker.py`, `python/matplotlib/axis.py` (new), `python/matplotlib/axes.py`, optionally `crates/matplotlib-ticker/`

Copy real matplotlib's `ticker.py` verbatim. Required adaptations:
- Replace `matplotlib._ticker` (C extension) imports: reimplement `_Edge_integer` and related in Python; promote to Rust crate only if needed
- Replace `rcParams` lookups with hardcoded defaults for the following keys: `axes.formatter.limits`, `axes.formatter.use_locale`, `axes.formatter.use_mathtext`, `axes.formatter.min_exponent`, `axes.formatter.useoffset`, `axes.formatter.offset_threshold` — or add these keys to `rcsetup.py`
- Drop `DateLocator`/`DateFormatter` (out of scope; date transforms not implemented)
- All numpy usage copies as-is (numpy-rust provides `np.log`, `np.power`, etc.)

**Formatters included:** `Formatter`, `NullFormatter`, `FixedFormatter`, `FuncFormatter`, `ScalarFormatter`, `LogFormatter`, `LogFormatterSciNotation`, `PercentFormatter`, `StrMethodFormatter`

**Locators included:** `Locator`, `NullLocator`, `FixedLocator`, `MultipleLocator`, `AutoLocator`, `LogLocator`, `MaxNLocator`

**New `axis.py`:** Thin `Axis` class holding a `major` and `minor` ticker (formatter + locator pair). Exposes:
- `set_major_formatter(fmt)` / `get_major_formatter()`
- `set_major_locator(loc)` / `get_major_locator()`
- `set_minor_formatter(fmt)` / `get_minor_formatter()`
- `set_minor_locator(loc)` / `get_minor_locator()`
- `set_ticks(ticks, labels=None)` / `get_ticks()`

`Axes.xaxis` and `Axes.yaxis` are `XAxis`/`YAxis` instances.

**Migration of `_xticks`/`_yticks`:** `Axes` currently stores tick positions and labels as `self._xticks` / `self._yticks` plain lists and renders them directly in the SVG/PIL draw routines. After Phase 1:
- `set_xticks(ticks)` / `set_xticklabels(labels)` delegate to `ax.xaxis.set_ticks()` — `_xticks`/`_yticks` are removed
- `get_xticks()` / `get_yticks()` delegate to `ax.xaxis.get_ticks()`
- Draw routines read tick positions and labels from the `Axis` object (via `AutoLocator` if no explicit ticks set, or from the fixed list if `set_xticks` was called)
- The `FixedLocator` + `FixedFormatter` pair is used when the user calls `set_xticks` explicitly, preserving existing behavior

**Upstream tests:** Port `test_formatter_str`, `test_scalar_formatter`, `test_logformatter`, `test_auto_locator`, `test_multiple_locator`, `test_maxn_locator` from `lib/matplotlib/tests/test_ticker.py`. New file: `test_ticker_upstream.py`.

## Phase 2 — Log/Symlog Scale Rendering

**Files:** `python/matplotlib/scale.py`, `python/matplotlib/axis.py` (extended), `python/matplotlib/axes.py`, `python/matplotlib/backend_bases.py`

Copy real matplotlib's `scale.py` verbatim. Adaptations: same rcParams and C-extension rules as Phase 1.

**Scale objects:** `LinearScale`, `LogScale`, `SymmetricalLogScale` (symlog), `FuncScale`. Each exposes:
- `forward(values)` — data → transformed coordinate
- `inverse(values)` — transformed → data

Named scales are constructed by `ax.set_xscale(name, **kwargs)`: `'linear'` → `LinearScale()`, `'log'` → `LogScale(base=10)`, `'symlog'` → `SymmetricalLogScale(linthresh=2)`. `FuncScale` is constructed directly: `ax.set_xscale(FuncScale(forward_fn, inverse_fn))`.

`axis.py` Phase 2 additions: each `Axis` gains a `set_scale(scale_obj)` method and holds the current `Scale` instance (default `LinearScale`). `ax.set_xscale('log')` constructs a `LogScale` and calls `ax.xaxis.set_scale()`, then updates the default locator/formatter to `LogLocator`/`LogFormatter`.

**`AxesLayout` integration:** `AxesLayout` in `backend_bases.py` has `sx(x)` / `sy(y)` methods that perform direct linear data→pixel mapping. Phase 2 changes these to: `sx(x) = linear_map(xaxis.scale.forward(x))` and `sy(y) = linear_map(yaxis.scale.forward(y))`. For `LinearScale`, `forward(x) == x`, so existing behavior is unchanged.

**Upstream tests:** Port `test_logscale_nonpos`, `test_logscale_mask`, `test_symlog`, `test_symlog2` from `lib/matplotlib/tests/test_axes.py`. New file: `test_scale_upstream.py`.

## Phase 3 — `ax.legend()`

**Files:** `python/matplotlib/legend.py`, `python/matplotlib/axes.py`

Copy real matplotlib's `legend.py`. Adaptations:
- Drop handler map extensibility (keep default handlers: Line2D → line swatch, Patch → colored box)
- Drop shadow, fancy box, draggable
- Drop `BboxTransformTo` / `BboxTransformFrom` (replace with direct pixel math)
- Retain: `loc`, `ncol`, `bbox_to_anchor`, `framealpha`, `title`, `handles`/`labels`, `fontsize`

**Migration of `_draw_legend`:** `Axes` currently has `_draw_legend(renderer, layout)` called directly from `Axes.draw()`, and `ax.legend()` sets `self._legend = True`. After Phase 3:
- `ax.legend(...)` constructs and stores a `Legend` object as `self._legend_obj`
- `Axes.draw()` calls `self._legend_obj.draw(renderer, layout)` instead of `_draw_legend`
- `_draw_legend` and the `_legend` bool are removed

**Rendering:** Legend box is drawn using existing renderer primitives (`draw_rect`, `draw_text`, `draw_line`). No new renderer primitives needed.

**Upstream tests:** Port `test_legend_auto`, `test_legend_loc`, `test_legend_ncol`, `test_no_handles_labels`, `test_legend_title` from `lib/matplotlib/tests/test_legend.py`. New file: `test_legend_upstream.py`.

## Phase 4 — Arrow Annotations

**Files:** `python/matplotlib/patches.py`, `python/matplotlib/text.py`, `python/matplotlib/backend_bases.py`, `_svg_backend.py`, `_pil_backend.py`

**New renderer primitive:** `draw_arrow(x1, y1, x2, y2, arrowstyle, color, linewidth)` added to `RendererBase`, `RendererSVG`, `RendererPIL`.
- SVG: `<path>` with `marker-end` referencing a `<marker>` arrowhead definition
- PIL: draw line + filled polygon for arrowhead

**`FancyArrowPatch` (simplified):** Add to `patches.py`. Supports `arrowstyle` strings: `'->'`, `'<-'`, `'<->'`, `'-'`, `'fancy'`. Handles `shrinkA`/`shrinkB` (shorten arrow at endpoints). Geometry copied from real matplotlib's `patches.py`, stripping transform-dependent code in favor of direct pixel coordinates.

**`Annotation` update:** `Annotation` currently has no `draw()` method (text and arrow are placeholders). Phase 4 adds `Annotation.draw(renderer, layout)` which:
1. Renders the text portion (same as `Text.draw()`)
2. If `arrowprops` is set, constructs a `FancyArrowPatch` from `xytext` → `xy` and calls `draw_arrow`

**Upstream tests:** Port `test_annotate_default_arrow`, `test_annotate_arrowprops` from `lib/matplotlib/tests/test_text.py`. New file: `test_annotation_upstream.py`.

## Phase 5 — Artist Properties

**Files:** `python/matplotlib/artist.py`, `axes.py`, `lines.py`, `patches.py`, `text.py`, renderers

**`artist.py` consolidation:** The existing `Artist` base class already has `alpha`, `zorder`, `visible`, `label`. Phase 5 audits per-class duplications in `Line2D`, `Patch`, `Text` and migrates them to use the base class properties, adding `clip_on`. No new base class is created — this is a cleanup pass.

**Zorder:** `Axes.draw()` currently iterates its artists (Line2D, Patch, Text, etc.) in insertion order. Phase 5 sorts them by `zorder` inside `Axes.draw()` before the per-artist `draw()` loop. Default zorder: lines=2, patches=1, text=3 (matching real matplotlib defaults).

**Alpha:** Passed through to renderer color functions. SVG: `opacity` attribute. PIL: `RGBA` blend.

**Linestyle dashes:** Extend current `'solid'`/`'dashed'`/`'dotted'` to support `(offset, (on, off, ...))` tuple format and named styles `'dashdot'`, `'loosely dashed'`, etc. SVG: `stroke-dasharray`. PIL: manual segment iteration.

**Upstream tests:** Port `test_alpha`, `test_zorder`, `test_linestyle_variants` from `lib/matplotlib/tests/test_artist.py` and `test_lines.py`. New file: `test_artist_upstream.py`.

## Test Strategy

Each phase adds a new `test_<topic>_upstream.py` file. Tests are ported from real matplotlib with the original source file path and function name noted in a comment. All 789 existing tests must continue passing throughout all phases.

## Out of Scope

- `imshow` / colorbar / colormaps
- Date formatters/locators
- LaTeX/mathtext rendering
- Interactive/pick events
- `tight_layout` implementation
- `constrained_layout`
