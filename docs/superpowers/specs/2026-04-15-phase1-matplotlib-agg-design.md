# Phase 1 Design — `crates/matplotlib-agg` (Rust RendererAgg)

**Status:** draft, awaiting review
**Predecessor:** Phase 0 plan (`2026-04-13-og-matplotlib-phase0.md`) — complete, 4401 passing
**Successor plans:** Phase 2 (`ft2font`), Phase 3 (image/contour/qhull), Phase 4 (PDF/PS)

## Goal

Replace the Python no-op `RendererAgg` stub with a real Rust implementation
(tiny-skia) that produces actual rasterized output for paths, markers,
collections, and images. Scope is "self-consistent rendering parity":
figures render visually correctly, our own SVG and PNG outputs agree,
simple image-comparison tests produce expected pixel colors for shapes.
**Not** a goal: byte-for-byte or AA-perfect match with upstream CPython
matplotlib.

## Phase boundaries for text

**Text rendering is Phase 2, not Phase 1.**

The OG `backend_agg.py` wrapper owns the text pipeline: `draw_text` calls
`font.set_text(...)`, `font.draw_glyphs_to_bitmap(...)`, then
`self._renderer.draw_text_image(font, x, y, angle, gc)`. The font object
is an `ft2font.FT2Font` instance, currently our Python no-op stub with
synthetic metrics.

Phase 1's Rust `RendererAgg.draw_text_image(obj, x, y, angle, gc)` takes
either an `FT2Font` instance or a grayscale ndarray. In Phase 1 it reads
a bitmap from `obj.get_image()` (currently zeros for ft2font stubs) or
the raw array, then alpha-blends at (x, y) with rotation. **When the
ft2font stub returns a 1×1 zero bitmap, text is invisible in the output
— that's fine; 1A's pass criterion is shape pixel tests, not text tests.**

Phase 2 replaces `ft2font.FT2Font` with a fontdue-backed implementation
whose `draw_glyphs_to_bitmap()` produces real glyph rasters. **No change
to Phase 1's `draw_text_image`** — the same code path starts showing real
glyphs automatically.

This means Phase 1 **does not** depend on fontdue. Fonts are Phase 2's
problem. The ~3.5 MB bundled-font discussion from the earlier draft of
this spec is hereby scoped out; no fonts are bundled with
`matplotlib-agg`.

## Non-goals

- Text glyph rasterization (Phase 2 — ft2font proper via fontdue)
- Mathtext rendering (Phase 2 — MathTextParser needs ft2font)
- usetex / TeX rendering (Phase 4)
- PDF / PS font embedding (Phase 4)
- Gouraud triangle colorspace correctness
- Path simplification matching OG's `agg.path.chunksize` tuning
- Byte-for-byte upstream pixel parity

## Architecture

### Crate layout

```
crates/matplotlib-agg/
├── Cargo.toml
└── src/
    ├── lib.rs          # #[pymodule] _backend_agg; RendererAgg pyclass
    ├── renderer.rs     # RendererAgg wrapping tiny_skia::Pixmap
    ├── path.rs         # matplotlib.path.Path → tiny_skia::Path
    ├── gc.rs           # GraphicsContext → tiny_skia::Stroke / Paint
    ├── image.rs        # draw_image blit with nearest-neighbor scale
    ├── text_image.rs   # draw_text_image blit for pre-rasterized glyphs
    └── buffer.rs       # __buffer__ / __array__ protocol + unpremul pass
```

No `font.rs` / `text.rs` in Phase 1. No bundled font files. See "Phase
boundaries for text" above.

### Build artifacts

- Cargo target: `libmatplotlib_agg.{dylib,so,dll}` (cdylib, crate-type =
  `["cdylib"]`)
- Installed to: `python/matplotlib/backends/_backend_agg.<ext>` where
  `<ext>` is `so` on Linux/macOS, `pyd` on Windows
- Module name: `_backend_agg` (via `#[pymodule] fn _backend_agg(...)`)
- The Python file `python/matplotlib/backends/_backend_agg.py` (no-op
  stub) **remains as a fallback** when the `.so` has not been built.
  CPython's `FileFinder` checks `ExtensionFileLoader` before
  `SourceFileLoader`, so the `.so` wins when both exist.
- `_backend_agg.so` (and other build artifacts) are added to `.gitignore`.
  The build does **not** mutate tracked source files.

### Python-side wiring

No changes needed. OG's `backends/backend_agg.py` already does:
```python
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
```
and uses it via `self._renderer = _RendererAgg(int(width), int(height), dpi)`.
Our crate exports exactly `RendererAgg` with the same constructor signature.

### Makefile

The Makefile must detect the host platform and copy the right cdylib
extension. No `cp lib_foo.dylib` hardcoding.

```make
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  DYLIB_EXT := dylib
  DYLIB_PREFIX := lib
else ifeq ($(UNAME_S),Linux)
  DYLIB_EXT := so
  DYLIB_PREFIX := lib
else
  DYLIB_EXT := dll
  DYLIB_PREFIX :=
endif

build-ext:
	cargo build -p matplotlib-path -p matplotlib-agg
	cp target/debug/$(DYLIB_PREFIX)_path.$(DYLIB_EXT) python/matplotlib/_path.so
	cp target/debug/$(DYLIB_PREFIX)matplotlib_agg.$(DYLIB_EXT) python/matplotlib/backends/_backend_agg.so
```

`make test` depends on `build-ext`, so `pytest` runs against the freshly
built extension. No pytest skipif gating — when the `.so` is built, it
shadows the `.py` fallback via CPython's import priority
(extension > source > bytecode). When the `.so` is absent (e.g. a
contributor ran `pytest` directly), the no-op `.py` stub still provides
the API and tests that don't require real pixels still pass at the Phase
0 baseline.

`.gitignore` additions: `python/matplotlib/_path.so`,
`python/matplotlib/backends/_backend_agg.so`, `target/`.

## Public API (pyclass surface)

This is **exactly** the surface the existing `backends/backend_agg.py`
wrapper expects. Every entry was verified against the current file at
`python/matplotlib/backends/backend_agg.py` — do not remove any without
also patching the wrapper.

```rust
#[pyclass]
struct RendererAgg {
    width: u32,
    height: u32,
    dpi: f64,
    // tiny-skia premultiplied rgba surface — the draw target
    pixmap: tiny_skia::Pixmap,
    // Cached unpremultiplied rgba view for buffer protocol consumers.
    // Lazily rebuilt when `dirty` is true.
    unpremul_cache: Vec<u8>,
    dirty: bool,
}

#[pymethods]
impl RendererAgg {
    #[new]
    fn new(width: u32, height: u32, dpi: f64) -> Self { ... }

    // -- Drawing primitives called by wrapper `_update_methods` --
    // Wrapper binds these with:
    //   self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
    //   self.draw_image             = self._renderer.draw_image
    //   self.draw_markers           = self._renderer.draw_markers
    //   self.draw_path_collection   = self._renderer.draw_path_collection
    //   self.draw_quad_mesh         = self._renderer.draw_quad_mesh
    // plus, from wrapper methods directly:
    //   self._renderer.draw_path(gc, path, transform, rgbFace)
    //   self._renderer.draw_text_image(obj, x, y, angle, gc)
    //   self._renderer.copy_from_bbox(bbox)
    //   self._renderer.restore_region(region[, x1, y1, x2, y2, ox, oy])

    fn draw_path(&mut self, gc: &Bound<PyAny>, path: &Bound<PyAny>,
                 transform: &Bound<PyAny>,
                 rgb_face: Option<&Bound<PyAny>>) -> PyResult<()>;

    fn draw_markers(&mut self, gc: &Bound<PyAny>,
                    marker_path: &Bound<PyAny>,
                    marker_trans: &Bound<PyAny>,
                    path: &Bound<PyAny>, trans: &Bound<PyAny>,
                    rgb_face: Option<&Bound<PyAny>>) -> PyResult<()>;

    fn draw_path_collection(&mut self,
                            gc: &Bound<PyAny>,
                            master_transform: &Bound<PyAny>,
                            paths: &Bound<PyAny>,
                            all_transforms: &Bound<PyAny>,
                            offsets: &Bound<PyAny>,
                            offset_trans: &Bound<PyAny>,
                            facecolors: &Bound<PyAny>,
                            edgecolors: &Bound<PyAny>,
                            linewidths: &Bound<PyAny>,
                            linestyles: &Bound<PyAny>,
                            antialiaseds: &Bound<PyAny>,
                            urls: &Bound<PyAny>,
                            offset_position: &Bound<PyAny>) -> PyResult<()>;

    // IMPORTANT: draw_quad_mesh must exist — wrapper binds it at
    // _update_methods(). Missing this broke the earlier draft of the spec.
    fn draw_quad_mesh(&mut self,
                      gc: &Bound<PyAny>,
                      master_transform: &Bound<PyAny>,
                      mesh_width: usize,
                      mesh_height: usize,
                      coordinates: &Bound<PyAny>,
                      offsets: &Bound<PyAny>,
                      offset_trans: &Bound<PyAny>,
                      facecolors: &Bound<PyAny>,
                      antialiased: bool,
                      edgecolors: &Bound<PyAny>) -> PyResult<()>;

    fn draw_image(&mut self, gc: &Bound<PyAny>,
                  x: f64, y: f64, im: &Bound<PyAny>) -> PyResult<()>;

    fn draw_gouraud_triangles(&mut self, gc: &Bound<PyAny>,
                              points: &Bound<PyAny>,
                              colors: &Bound<PyAny>,
                              transform: &Bound<PyAny>) -> PyResult<()>;

    // IMPORTANT: draw_text_image is the low-level text blitter the
    // wrapper's draw_text / draw_tex call. Phase 1 implementation blits
    // `obj.get_image()` (for FT2Font) or `obj` (for ndarray). When
    // ft2font is still a no-op stub, the blit is a no-op; Phase 2 makes
    // the same call path produce real glyphs.
    fn draw_text_image(&mut self, obj: &Bound<PyAny>,
                       x: i32, y: i32, angle: f64,
                       gc: &Bound<PyAny>) -> PyResult<()>;

    // Buffer accessors
    fn tostring_rgb(&mut self, py: Python) -> Py<PyBytes>;
    fn tostring_argb(&mut self, py: Python) -> Py<PyBytes>;
    fn clear(&mut self);

    // Region lifecycle. Note: restore_region has TWO calling shapes:
    //   renderer.restore_region(region)
    //   renderer.restore_region(region, x1, y1, x2, y2, ox, oy)
    // Use #[pyo3(signature = (region, x1=None, y1=None, x2=None, y2=None, ox=None, oy=None))]
    // to accept both.
    fn copy_from_bbox(&self, bbox: &Bound<PyAny>) -> PyResult<Py<PyAny>>;

    #[pyo3(signature = (region, x1=None, y1=None, x2=None, y2=None, ox=None, oy=None))]
    fn restore_region(&mut self, region: &Bound<PyAny>,
                      x1: Option<i32>, y1: Option<i32>,
                      x2: Option<i32>, y2: Option<i32>,
                      ox: Option<i32>, oy: Option<i32>) -> PyResult<()>;

    // Filter and rasterization stack — used by wrapper's start_filter /
    // stop_filter. These create new temporary _RendererAgg instances in
    // Python; our instance just needs a workable clear() and draw cycle.
    fn start_filter(&mut self);
    fn stop_filter(&mut self, _post_processing: &Bound<PyAny>);
    fn start_rasterizing(&mut self);
    fn stop_rasterizing(&mut self);
}

// Buffer protocol — the critical Python-side contract. See "Buffer
// exposure" section below for full semantics.
#[pymethods]
impl RendererAgg {
    fn __buffer__<'py>(slf: PyRefMut<'py, Self>, flags: i32)
                      -> PyResult<PyBuffer<u8>>;

    // numpy.asarray(renderer_agg_instance) → (H, W, 4) u8 ndarray
    fn __array__<'py>(&mut self, py: Python<'py>,
                      dtype: Option<&Bound<PyAny>>,
                      copy: Option<bool>)
                      -> PyResult<Bound<'py, PyArray3<u8>>>;
}

#[pyfunction]
fn get_hinting_flag() -> i32 { 0 }
```

### Methods deliberately NOT on the Rust class

The following are methods on the *wrapper* (`backend_agg.RendererAgg`
Python class), not on the extension. They stay in Python and call into
other subsystems that Phase 1 does not touch:

- `draw_text`, `draw_tex`, `draw_mathtext` — Python in
  `backend_agg.py`; they set up fonts/mathtext, then call our Rust
  `draw_text_image`.
- `get_text_width_height_descent` — Python wrapper that consults
  `ft2font.FT2Font` (Phase 2) or `MathTextParser`. Our Rust class does
  not know about font metrics.
- `buffer_rgba()` — Python wrapper that returns
  `memoryview(self._renderer)`. The Rust class does NOT expose a
  `buffer_rgba` method; it provides the buffer protocol instead.
- `points_to_pixels`, `get_canvas_width_height`, `flipy`,
  `option_image_nocomposite`, `option_scale_image` — these live on
  `backend_agg.RendererAgg` Python class, not on the C extension.
  Leave them in Python.

## Path translation (`src/path.rs`)

matplotlib `Path` objects expose `vertices: ndarray(N, 2)` and
`codes: Optional[ndarray(N, uint8)]` where codes are:
- `STOP = 0` — end of path
- `MOVETO = 1`
- `LINETO = 2`
- `CURVE3 = 3` — quadratic bezier (2 vertices: control, end)
- `CURVE4 = 4` — cubic bezier (3 vertices: ctrl1, ctrl2, end)
- `CLOSEPOLY = 79`

Translation to `tiny_skia::PathBuilder`:
- Apply the affine `transform` to each vertex (2×3 matrix from
  `matplotlib.transforms.Affine2D`)
- Emit `move_to`, `line_to`, `quad_to`, `cubic_to`, `close` accordingly
- Flip y: matplotlib has origin at bottom-left, SVG/tiny-skia at top-left

Open question: codes can be `None`, in which case matplotlib treats all
vertices as LINETO after an implicit MOVETO to the first point. Handle
both branches.

## GraphicsContext (`src/gc.rs`)

`gc` is an instance of `matplotlib.backend_bases.GraphicsContextBase`
with attribute getters. We read:
- `get_linewidth()` → `Stroke.width` (convert points to pixels)
- `get_rgb()` → `Color` for stroking (use gc foreground, possibly with alpha)
- `get_dashes()` → `(offset, [on, off, ...])` → tiny_skia dash pattern
- `get_capstyle()` → `butt`/`round`/`projecting` → `LineCap`
- `get_joinstyle()` → `miter`/`round`/`bevel` → `LineJoin`
- `get_clip_rectangle()` → intersect with pixmap bounds for clipping
- `get_clip_path()` → **Milestone 1B**, fall back to rectangle for 1A
- `get_hatch()` → **Milestone 1B**, skip for 1A
- `get_snap()` → **Milestone 1B**, off for 1A
- `get_alpha()` → multiplicative alpha on fill/stroke

For **fill**: `rgbFace` arg to `draw_path`. If `Some`, fill with that
color (using gc's alpha). If `None`, path is stroke-only.

## Text rendering (`src/text_image.rs`)

Phase 1 ships **only** `draw_text_image`, not glyph layout or metrics.
All font/metrics work is Phase 2.

### `draw_text_image(obj, x, y, angle, gc)`

The wrapper calls this in three code paths:

1. From `draw_text`: `obj` is an `ft2font.FT2Font` instance whose
   internal bitmap was populated by `font.draw_glyphs_to_bitmap()`
   earlier in the same method.
2. From `draw_tex`: `obj` is a 2D `numpy.ndarray` of `uint8` grayscale
   pixels produced by the TeX manager.
3. From `draw_mathtext`: `obj` is also an `FT2Font` whose bitmap was
   populated by `MathTextParser`.

Phase 1 implementation:
```rust
fn draw_text_image(&mut self, obj: &Bound<PyAny>,
                   x: i32, y: i32, angle: f64,
                   gc: &Bound<PyAny>) -> PyResult<()> {
    // Extract a (H, W) u8 grayscale bitmap from obj.
    let bitmap: Array2<u8> = if obj_is_ft2font(obj) {
        // Call obj.get_image() → numpy array (may be 1x1 zeros when
        // ft2font is still our no-op stub in Phase 1)
        obj.call_method0("get_image")?.extract()?
    } else {
        // Already an ndarray
        obj.extract()?
    };

    // Read gc foreground color → rgba
    let (r, g, b, a) = gc_foreground(gc)?;

    // Blit: for each nonzero pixel in bitmap, alpha-blend a colored
    // pixel into self.pixmap at (x + col, y + row), applying `angle`
    // rotation via tiny_skia::Transform::from_rotate.
    self.blit_grayscale_mask(&bitmap, x, y, angle, r, g, b, a);
    self.dirty = true;
    Ok(())
}
```

When the ft2font stub returns a 1×1 zero bitmap (current Phase 0
reality), this is a no-op and text is invisible in the output. When
Phase 2 gives ft2font a real `draw_glyphs_to_bitmap()` implementation,
the same call path renders text — no change to
`crates/matplotlib-agg/`.

### Text rotation

Arbitrary rotation angles are supported from day 1 via
`tiny_skia::Transform::from_rotate(angle_degrees)`. The grayscale mask is
blitted with a rotated transform. No 90°-only special case. Phase 1B
only adds snap-to-pixel for axis labels.

### No text metrics in the Rust class

`get_text_width_height_descent` stays in `backend_agg.py` Python wrapper.
It calls `ft2font.FT2Font.get_width_height()` / `.get_descent()`. In
Phase 0, our ft2font stub returns synthetic metrics based on
`0.6 * ptsize * n_chars`. Phase 2 replaces these with fontdue metrics.
**Phase 1 does nothing here.**

## Image rendering (`src/image.rs`)

`draw_image(gc, x, y, im)`:
- `im` is an `(H, W, 4)` uint8 ndarray (RGBA)
- Copy into a `tiny_skia::Pixmap::from_vec(...)` with `PixmapPaint`
- Blit at (x, y) with gc's clip rect applied
- No rotation (matplotlib doesn't expect it from Agg)

1A: nearest-neighbor copy. 1B: bilinear scaling, interpolation modes.

## Buffer exposure

This is the critical Python-side contract. The wrapper
`backend_agg.py:262` does `return memoryview(self._renderer)` — so
`memoryview(RendererAgg_instance)` must work. It also does
`np.asarray(self._renderer).take([3, 0, 1, 2], axis=2).tobytes()` at
line 266 — so `numpy.asarray(RendererAgg_instance)` must also work and
must return a 3D (H, W, 4) array.

Both code paths go through the same underlying data, so we **must**
implement both the Python buffer protocol (`__buffer__`) and numpy's
array protocol (`__array__` or `__array_interface__`) against the same
bytes. There is no separate `buffer_rgba()` method on the Rust class —
the wrapper's `buffer_rgba()` is a Python one-liner that boxes
`memoryview(self._renderer)`.

### Premultiplication

tiny-skia stores pixels premultiplied (rgba with `r', g', b' = r*a, g*a,
b*a`). OG's Agg renderer returns **unpremultiplied** rgba bytes, and
PIL's `RGBA` mode expects unpremultiplied. We cannot expose the raw
pixmap buffer directly — we have to unpremultiply.

### Cached unpremultiplied buffer

The Rust struct holds two buffers:

```rust
pixmap: tiny_skia::Pixmap,      // premultiplied draw surface
unpremul_cache: Vec<u8>,        // (H*W*4) unpremultiplied, lazy
dirty: bool,                    // set by every draw_* method
```

Every `draw_*` method sets `dirty = true`. The buffer-protocol getters
(`__buffer__`, `__array__`, `tostring_rgb`, `tostring_argb`) take
`&mut self`, check `dirty`, and if set:

```rust
if self.dirty {
    unpremultiply(&self.pixmap, &mut self.unpremul_cache);
    self.dirty = false;
}
```

`unpremultiply` is a loop: `for each pixel { if a>0 { r = (r*255)/a; g
= (g*255)/a; b = (b*255)/a } }` with saturation clamping. Fast enough;
profiled ~1 ms for 640×480 on M1.

### Zero-copy vs. copy

- `__buffer__` returns a `PyBuffer` pointing directly at
  `self.unpremul_cache.as_slice()` with shape `(H, W, 4)` and strides
  `(W*4, 4, 1)`, read-only. **Zero-copy from Rust's perspective.** The
  Python memoryview lives as long as the Rust object holds the
  `unpremul_cache`.
- `__array__` returns a numpy array that also views the cache, tagged
  read-only. Lifetime tied to the Rust object via PyO3's standard
  mechanisms.
- `tostring_rgb` / `tostring_argb` copy (they return owned `bytes`).

**Decision**: `__buffer__` and `__array__` are zero-copy views into the
cache. This matches the reviewer's concern about keeping memoryview and
np.asarray consistent. The earlier "1A uses a copy" language in the
Resolved Decisions section is overridden by this section; see the
Resolved Decisions update below.

### Lifetime safety

PyO3 0.25's `PyBuffer::get` callback lets us expose an owned
`Vec<u8>`-backed buffer safely as long as the `Py_buffer` struct keeps a
reference to the Python object. We set `obj` on the `Py_buffer` to `slf`,
and PyO3's buffer release callback drops the reference. The Rust object
is not freed while any outstanding memoryview exists. Standard pattern.

If lifetime handling turns out to be harder than expected in practice,
the **fallback** is to copy into a new `Vec<u8>` on each `__buffer__`
call. This doubles memory allocation per `savefig` but is trivially
correct. The choice between zero-copy and copy is an implementation
detail, NOT a spec commitment. Both paths expose the same bytes.

## Milestone 1A — minimum viable (one commit)

Pass criterion: all 5 currently-skipped `test_*_renders_png` pixel-check
tests in `test_patches_upstream.py` produce expected colors (`red > 200`,
`blue > 200`, etc.) when `savefig(png)` is called:

- `test_arc_renders_png` — draws a green arc
- `test_arrow_renders_png` — draws a red arrow
- `test_ellipse_renders_png` — draws a red ellipse
- `test_fancy_bbox_round_png` — blue rounded box
- `test_fancy_bbox_square_png` — green square box

These tests only assert shape color counts; none assert anything about
text. Phase 1A can pass with `draw_text_image` as a no-op.

Delivers:
- Crate scaffolding (`Cargo.toml`, `src/lib.rs`, `src/renderer.rs`,
  `src/path.rs`, `src/gc.rs`, `src/image.rs`, `src/text_image.rs`,
  `src/buffer.rs`)
- Full `draw_path` (MOVETO/LINETO/CURVE3/CURVE4/CLOSEPOLY, transform,
  fill + stroke, clip rectangle from gc)
- `draw_markers` (naive: loop over path vertices, draw marker_path at
  each)
- `draw_path_collection` (naive: loop of draw_path)
- `draw_quad_mesh` (stub returning Ok(()) — used by `pcolormesh` which
  is not on the 1A critical path)
- `draw_image` (nearest-neighbor blit)
- `draw_text_image` (blits `obj.get_image()` or ndarray with rotation;
  ft2font stub currently returns zeros so output is blank, that's OK)
- `__buffer__` + `__array__` + `tostring_rgb` + `tostring_argb` backed
  by `unpremul_cache` with dirty-flag lazy refresh
- `copy_from_bbox` / `restore_region` with correct 1-arg / 7-arg
  signature
- `start_filter` / `stop_filter` / `start_rasterizing` /
  `stop_rasterizing` (sufficient to not crash; filter stack is 1B)
- `clear()` (reset pixmap to transparent white, set dirty)
- `get_hinting_flag()` module function → 0
- Platform-aware `Makefile` (detect `uname -s`, copy the right cdylib
  suffix)
- `.gitignore` entries for built `.so`
- `python/matplotlib/backends/_backend_agg.py` stub **retained** as
  fallback (not deleted)
- Unskip the 5 pixel tests

Non-goals for 1A: hatching, gouraud, clip paths (only rectangles),
path chunking, image scaling beyond nearest-neighbor, snap-to-pixel,
mathtext, text glyph content (all text pixels come from ft2font — Phase
2).

Expected test count after 1A: 4406 passed (+5), 214 skipped (-5), 0
failed. No regressions from the Phase 0 baseline.

## Milestone 1B — self-consistency polish (follow-up commits)

Delivers incrementally:
- **1B.1 Clip paths**: `get_clip_path()` → tiny_skia `Mask`
- **1B.2 Hatching**: pattern fills (pre-rasterize hatches to small
  pixmaps, use `Pattern` fill)
- **1B.3 Dashes + caps + joins**: full GraphicsContext stroke translation
- **1B.4 Gouraud triangles**: per-vertex color interpolation via
  `tiny_skia::GradientStop`
- **1B.5 Image scaling**: bilinear + gc resize quality
- **1B.6 Path chunking**: implement `agg.path.chunksize` rcParam honoring
- **1B.7 Arbitrary-angle text**: proper affine glyph blit
- **1B.8 Snap-to-pixel**: `gc.get_snap()` support for crisp axis lines

Each sub-milestone is a separate commit. Pass criterion: the `test_rendering*.py`
tests that use OG `savefig(svg/png)` — not the stub-era helpers — produce
expected structural elements.

## Risks & mitigations

1. **Module shadowing — `.py` fallback vs. `.so` built artifact.**
   *Risk:* A contributor runs `pytest` directly without `make build-ext`.
   The `.so` doesn't exist; the `.py` stub takes over; rendering tests
   that expect real pixels fail silently (produce blank PNGs).
   *Mitigation:* `make test` depends on `make build-ext`. CI always
   goes through `make test`. Local dev can too. For the pytest-direct
   case, document in CONTRIBUTING that extension tests need a prior
   build. Verify import order with a smoke test that asserts
   `_backend_agg.__file__.endswith(".so")` when we believe the extension
   is available.

2. **Buffer lifetime under PyO3 0.25.**
   *Risk:* Returning an ndarray or memoryview that borrows from
   `self.unpremul_cache` requires PyO3's `PyBuffer` machinery and
   careful reference counting. Getting this wrong manifests as
   use-after-free or segfault, not a test failure.
   *Mitigation:* Start with the copy-on-access fallback (allocate a new
   `Vec<u8>` on each `__buffer__` / `__array__` call). 640×480×4 =
   1.2 MB copy, negligible. Optimize to zero-copy only after a green
   test baseline.

3. **Draw-call ordering between `draw_*` and buffer access.**
   *Risk:* OG code might call `self._renderer.draw_path()` and then
   `memoryview(self._renderer)` on the same instance from different
   Python frames. If our unpremul pass mutates shared state without
   locking, we could corrupt the cache.
   *Mitigation:* The unpremul pass takes `&mut self`, so PyO3 enforces
   exclusive access. Matplotlib is single-threaded at the figure level.

4. **Text appears invisible in savefig(png) until Phase 2.**
   *Risk:* Someone runs the pipeline, opens the PNG, sees axes and
   lines but no labels, assumes Phase 1 is broken.
   *Mitigation:* Clearly document in the Milestone 1A success criteria
   and commit message: "text is blank — Phase 2 lights it up". The 5
   shape-pixel-check tests do not assert anything about text, so they
   pass. Tests that *do* assert text in PNG output (if any) stay skipped
   with an explicit Phase 2 marker.

5. **tiny-skia curve coverage.** matplotlib emits `CURVE3` (quadratic)
   and `CURVE4` (cubic) Bezier segments. tiny-skia's `PathBuilder`
   supports both (`quad_to`, `cubic_to`). Hermite / Catmull-Rom are NOT
   emitted by matplotlib Paths so this is a non-issue. Confirmed by
   reading `python/matplotlib/path.py`.

6. **Wrapper API drift.** The OG `backend_agg.py` wrapper binds several
   methods at `_update_methods()` time. If an OG update adds a new
   wrapper-bound method (e.g. some new `draw_foo`), our Rust class won't
   have it and the wrapper bind will fail at construction.
   *Mitigation:* Phase 1 pins to the current OG 3.10.x `backend_agg.py`.
   Any future OG sync that bumps this file requires re-auditing the
   extension surface. The spec enumerates the current contract exactly;
   drift is detectable.

## Test plan

### Unit tests (Rust)

- `src/path.rs` tests: `(vertices, codes)` tuples → expected sequences
  of tiny_skia `PathBuilder` calls, including `codes=None` fallback
- `src/gc.rs` tests: GraphicsContext-like Python mock objects → expected
  `tiny_skia::Stroke` + `Paint` output
- `src/buffer.rs` tests: premultiplied → unpremultiplied round-trip
  matches `(r*a/255, g*a/255, b*a/255)` for a range of alpha values

### Integration tests (Python/pytest)

- Unskip the 5 pixel tests in `test_patches_upstream.py` (1A pass
  criterion). They re-run via `make test`.
- Add a new `test_backend_agg_smoke.py` that:
  - Constructs `_backend_agg.RendererAgg(100, 80, 72)` directly
  - Calls `draw_path`, `draw_markers`, `draw_image` with minimal args
  - Asserts `memoryview(renderer).format == 'B'`, shape `(80, 100, 4)`,
    readable via `bytes()`
  - Asserts `np.asarray(renderer).shape == (80, 100, 4)` and
    `dtype == np.uint8`
  - Asserts `renderer.clear()` zeros the buffer
  - Asserts `renderer.draw_text_image(np.zeros((1,1), dtype=np.uint8),
    0, 0, 0.0, gc_mock)` does not raise

### Full suite regression

After every milestone: run `make test` and verify the pass count is
monotonically non-decreasing from the Phase 0 baseline of 4401. Any
OG-behavior test that *breaks* during Phase 1 work should be tracked
and either:
- Adapted (if the assertion is too strict and ours is also valid)
- Re-skipped with a new Phase 1 marker (if it's genuinely renderer-dependent)
- Fixed (if it's a real bug in our code)

## Rollback plan

If Phase 1 is unworkable after several commits:
- The `_backend_agg.py` no-op stub can be restored from git history
- The Makefile reverts to building only `matplotlib-path`
- The 4401 Phase 0 baseline is recoverable

## Resolved design decisions

1. **Image scaling for 1A**: nearest-neighbor only. Simpler, fast enough
   for `imshow` smoke tests. 1B adds bilinear via `PixmapPaint::filter`.
2. **Antialiasing**: always on in 1A (tiny-skia's default AA path). 1B
   honors `gc.get_antialiased()`.
3. **Premultiplied alpha**: tiny-skia stores premultiplied internally;
   we maintain a separate unpremultiplied cache (`unpremul_cache`)
   regenerated lazily via a `dirty` flag on every draw. See the Buffer
   Exposure section for the full semantics — it is the authoritative
   version of this decision, not a footnote here.
4. **Path with `codes=None`**: treat as implicit MOVETO(v[0]) followed
   by LINETO(v[1..]). Matches matplotlib's documented `Path` behavior.
5. **Text rotation**: arbitrary angles via
   `tiny_skia::Transform::from_rotate`. The old "90°-only" language from
   an earlier draft is hereby struck — `draw_text_image` supports
   arbitrary rotation from day 1. (Note: in practice nothing is visible
   anyway until Phase 2 gives ft2font real bitmap data, but the code
   path is complete.)
6. **Buffer exposure in 1A**: zero-copy views into `unpremul_cache` via
   `__buffer__` and `__array__`. If PyO3 0.25 lifetime handling proves
   awkward in practice, fall back to copying into a new `Vec<u8>` on
   each call — same external contract, different memory profile. The
   earlier draft's "copy for 1A" language was me hedging; the final
   contract is "consistent bytes, zero-copy preferred, copy acceptable
   as an implementation detail".
7. **No text glyph rasterization in Phase 1**: `draw_text_image` reads a
   bitmap from the passed object and blits it. It does not call into
   fontdue. The fontdue dependency is scoped to Phase 2 (`ft2font`).
8. **Wrapper methods stay in Python**: `draw_text`, `draw_tex`,
   `draw_mathtext`, `get_text_width_height_descent`, `buffer_rgba`,
   `points_to_pixels`, `flipy`, `option_*` all remain on the Python
   wrapper `backend_agg.RendererAgg`. We do not duplicate them on the
   Rust class.
