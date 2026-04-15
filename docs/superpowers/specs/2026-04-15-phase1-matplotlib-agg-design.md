# Phase 1 Design — `crates/matplotlib-agg` (Rust RendererAgg)

**Status:** draft, awaiting review
**Predecessor:** Phase 0 plan (`2026-04-13-og-matplotlib-phase0.md`) — complete, 4401 passing
**Successor plans:** Phase 2 (`ft2font`), Phase 3 (image/contour/qhull), Phase 4 (PDF/PS)

## Goal

Replace the Python no-op `RendererAgg` stub with a real Rust implementation
(tiny-skia + fontdue) that produces actual rasterized output. Scope is
"self-consistent rendering parity": figures render visually correctly, our
own SVG and PNG outputs agree, simple image-comparison tests produce
expected pixel colors. **Not** a goal: byte-for-byte or AA-perfect match
with upstream CPython matplotlib.

## Non-goals

- Mathtext rendering (Phase 2 — ft2font proper)
- usetex / TeX rendering
- PDF / PS font embedding (Phase 4)
- Gouraud triangle colorspace correctness
- Per-glyph kerning tuned to FreeType output
- Path simplification matching OG's `agg.path.chunksize` tuning

## Architecture

### Crate layout

```
crates/matplotlib-agg/
├── Cargo.toml
├── fonts/
│   ├── DejaVuSans.ttf
│   ├── DejaVuSans-Bold.ttf
│   ├── DejaVuSans-Oblique.ttf
│   ├── DejaVuSansMono.ttf
│   └── DejaVuSerif.ttf
└── src/
    ├── lib.rs          # #[pymodule] _backend_agg; RendererAgg pyclass
    ├── renderer.rs     # RendererAgg wrapping tiny_skia::Pixmap
    ├── path.rs         # matplotlib.path.Path → tiny_skia::Path
    ├── gc.rs           # GraphicsContext → tiny_skia::Stroke / Paint
    ├── text.rs         # fontdue layout + glyph rasterization + blit
    ├── font.rs         # Bundled font registry (include_bytes!)
    └── image.rs        # draw_image blit with nearest-neighbor scale
```

### Build artifacts

- Cargo target: `lib_backend_agg.dylib` (cdylib, crate-type = `["cdylib"]`)
- Installed to: `python/matplotlib/backends/_backend_agg.so`
- Module name: `_backend_agg` (via `#[pymodule] fn _backend_agg(...)`)
- The Python file `python/matplotlib/backends/_backend_agg.py` (no-op
  stub) is **deleted** — the `.so` is the only source for this module

### Python-side wiring

No changes needed. OG's `backends/backend_agg.py` already does:
```python
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
```
and uses it via `self._renderer = _RendererAgg(int(width), int(height), dpi)`.
Our crate exports exactly `RendererAgg` with the same constructor signature.

### Makefile

```make
build-ext:
	cargo build -p matplotlib-path -p matplotlib-agg
	cp target/debug/lib_path.dylib python/matplotlib/_path.so
	cp target/debug/lib_backend_agg.dylib python/matplotlib/backends/_backend_agg.so
```

`make test` depends on `build-ext`, so `pytest` runs against the freshly
built extension. No pytest skipif gating — a missing `.so` is a hard
failure, which is what we want for CI green-ness.

## Public API (pyclass surface)

The class mirrors the Python no-op stub. OG code accesses these methods
directly; any missing method is a regression.

```rust
#[pyclass]
struct RendererAgg {
    width: u32,
    height: u32,
    dpi: f64,
    pixmap: tiny_skia::Pixmap,       // rgba8 premultiplied
    font_registry: FontRegistry,     // shared bundled fonts
}

#[pymethods]
impl RendererAgg {
    #[new]
    fn new(width: u32, height: u32, dpi: f64) -> Self { ... }

    // Drawing
    fn draw_path(&mut self, py: Python, gc: &Bound<PyAny>,
                 path: &Bound<PyAny>, transform: &Bound<PyAny>,
                 rgb_face: Option<&Bound<PyAny>>) -> PyResult<()>;
    fn draw_markers(&mut self, py: Python, gc: &Bound<PyAny>,
                    marker_path: &Bound<PyAny>,
                    marker_trans: &Bound<PyAny>,
                    path: &Bound<PyAny>, trans: &Bound<PyAny>,
                    rgb_face: Option<&Bound<PyAny>>) -> PyResult<()>;
    fn draw_path_collection(&mut self, py: Python,
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
    fn draw_image(&mut self, py: Python, gc: &Bound<PyAny>,
                  x: f64, y: f64, im: &Bound<PyAny>) -> PyResult<()>;
    fn draw_gouraud_triangles(&mut self, py: Python, gc: &Bound<PyAny>,
                              points: &Bound<PyAny>,
                              colors: &Bound<PyAny>,
                              transform: &Bound<PyAny>) -> PyResult<()>;
    fn draw_text(&mut self, py: Python, gc: &Bound<PyAny>,
                 x: f64, y: f64, s: &str,
                 prop: &Bound<PyAny>, angle: f64,
                 ismath: Option<&Bound<PyAny>>,
                 mtext: Option<&Bound<PyAny>>) -> PyResult<()>;

    // Metrics (callable from Python without rasterizing)
    fn get_text_width_height_descent(&self, s: &str,
                                     prop: &Bound<PyAny>,
                                     ismath: &Bound<PyAny>)
                                     -> PyResult<(f64, f64, f64)>;
    fn points_to_pixels(&self, points: f64) -> f64;
    fn get_canvas_width_height(&self) -> (u32, u32);
    fn get_image_magnification(&self) -> f64;

    // Buffer exposure
    fn buffer_rgba<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>>;
    fn tostring_rgb(&self, py: Python) -> Py<PyBytes>;
    fn tostring_argb(&self, py: Python) -> Py<PyBytes>;
    fn clear(&mut self);

    // Lifecycle (no-ops for stateless rasterizer)
    fn copy_from_bbox(&self, _bbox: &Bound<PyAny>) -> Option<()>;
    fn restore_region(&self, _region: &Bound<PyAny>,
                      _bbox: Option<&Bound<PyAny>>,
                      _xy: Option<&Bound<PyAny>>);
    fn start_filter(&mut self);
    fn stop_filter(&mut self, _filter_func: &Bound<PyAny>);
    fn start_rasterizing(&mut self);
    fn stop_rasterizing(&mut self);
    fn flipy(&self) -> bool { true }
    fn option_image_nocomposite(&self) -> bool { false }
    fn option_scale_image(&self) -> bool { true }
}

// Buffer protocol
impl RendererAgg {
    fn __buffer__(&self, flags: i32) -> PyBufferView { ... }
}

#[pyfunction]
fn get_hinting_flag() -> i32 { 0 }
```

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

## Text rendering (`src/text.rs` + `src/font.rs`)

### Font registry

```rust
pub struct FontRegistry {
    sans: fontdue::Font,       // DejaVuSans
    sans_bold: fontdue::Font,  // DejaVuSans-Bold
    sans_italic: fontdue::Font,// DejaVuSans-Oblique
    mono: fontdue::Font,       // DejaVuSansMono
    serif: fontdue::Font,      // DejaVuSerif
}
```

Loaded once via `include_bytes!` at crate init. Phase 1 uses a simple
family-to-font mapping; Phase 2 will integrate with `font_manager` properly.

Mapping rule:
- `font.family` ∈ `["serif"]` → serif
- `font.family` ∈ `["monospace"]` → mono
- `font.weight == "bold"` → sans_bold (or ignored for non-sans)
- `font.style ∈ ("italic", "oblique")` → sans_italic (or ignored for
  non-sans in 1A)
- Everything else → sans

### Metrics

`get_text_width_height_descent(s, prop, ismath)`:
- If `ismath == "TeX"`: return super-class fallback (raises or returns
  zero-size for now; math text is Phase 2)
- Otherwise: pick font from `prop`, layout with fontdue at
  `prop.get_size_in_points() * dpi / 72.0` px, accumulate glyph advances
  for width, use `font.horizontal_line_metrics(px)` for height/descent

These values deliberately diverge from FreeType output by a small amount
— that's expected. tight_layout will be "close enough".

### Rasterization

`draw_text(gc, x, y, s, prop, angle, ismath, mtext)`:
1. Compute glyph positions via fontdue layout
2. For each glyph:
   - Rasterize to a small grayscale bitmap via `font.rasterize()`
   - Rotate by `angle` using bilinear sampling
   - Blend into `self.pixmap` at `(x + offset_x, y + offset_y)` using
     gc's foreground color and alpha
3. Anchor handling: x, y are baseline-origin coords in matplotlib space;
   flip y → pixmap space

Rotation: for 1A, support only multiples of 90° cleanly; arbitrary angles
get a rotated bounding box blit (bilinear). 1B: proper affine transform.

## Image rendering (`src/image.rs`)

`draw_image(gc, x, y, im)`:
- `im` is an `(H, W, 4)` uint8 ndarray (RGBA)
- Copy into a `tiny_skia::Pixmap::from_vec(...)` with `PixmapPaint`
- Blit at (x, y) with gc's clip rect applied
- No rotation (matplotlib doesn't expect it from Agg)

1A: nearest-neighbor copy. 1B: bilinear scaling, interpolation modes.

## Buffer exposure

`buffer_rgba()` returns a zero-copy `numpy.ndarray` view onto
`self.pixmap.data()`. Use `numpy::PyArray3::from_slice_bound` with
`(height, width, 4)` shape. The array is *read-only* from Python's
perspective; PIL can then save it as PNG via `image.imsave`.

`__buffer__` — implement the Python buffer protocol so that
`memoryview(renderer_agg_instance)` works. This is what OG's
`backend_agg.buffer_rgba()` wrapper calls.

## Milestone 1A — minimum viable (one commit)

Pass criterion: all 5 currently-skipped `test_*_renders_png` pixel-check
tests in `test_patches_upstream.py` produce expected colors (`red > 200`,
`blue > 200`, etc.) when `savefig(png)` is called:

- `test_arc_renders_png` — draws a green arc
- `test_arrow_renders_png` — draws a red arrow
- `test_ellipse_renders_png` — draws a red ellipse
- `test_fancy_bbox_round_png` — blue rounded box
- `test_fancy_bbox_square_png` — green square box

Delivers:
- Crate scaffolding (`Cargo.toml`, `src/lib.rs`, `src/renderer.rs`, path/gc/text/font/image modules)
- Bundled DejaVu fonts
- Full `draw_path` (MOVETO/LINETO/CURVE3/CURVE4/CLOSEPOLY, transform, fill + stroke)
- `draw_markers` (naive: loop over path vertices, draw marker_path at each)
- `draw_path_collection` (naive: loop of draw_path)
- `draw_image` (nearest-neighbor blit)
- `draw_text` (axis-aligned + 90° rotations; unaligned rotation allowed but may be rough)
- `get_text_width_height_descent` via fontdue
- `buffer_rgba` + `__buffer__` + `tostring_rgb`
- `Makefile` updated
- `_backend_agg.py` stub deleted
- Unskip the 5 pixel tests

Non-goals for 1A: hatching, gouraud, clip paths, path chunking, image
scaling beyond nearest-neighbor, snap-to-pixel, mathtext, arbitrary-angle
text (best-effort only).

Expected test count after 1A: 4406 passed (+5), 214 skipped (-5).

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

1. **fontdue metrics ≠ FreeType metrics.**
   *Risk:* Layout tests that compare text positions pixel-by-pixel fail.
   *Mitigation:* Document the divergence in the spec; adapt any tests
   that assume FreeType numbers to use tolerances; Phase 2 can tighten.

2. **`get_text_width_height_descent` is called *a lot* (once per tick
   label).**
   *Risk:* Slow tests.
   *Mitigation:* OG caches this via `_get_text_metrics_with_cache_impl`;
   we just need consistent return values. Fontdue is fast enough.

3. **Module shadow loop**: if both `_backend_agg.py` and `_backend_agg.so`
   exist, CPython prefers the `.so` — but this is platform-specific.
   *Mitigation:* Delete the `.py` in the build step and have `.gitignore`
   catch it. Verify import order with a test.

4. **numpy buffer lifetime**: returning an ndarray that borrows from
   `self.pixmap` requires careful lifetime handling in PyO3 0.25.
   *Mitigation:* Return a *copy* for 1A (`to_owned()`), optimize to
   zero-copy in 1B. 640×480×4 = 1.2MB per draw, acceptable.

5. **tiny-skia lacks Hermite / Catmull-Rom curves**: matplotlib emits
   these via `CURVE4` (cubic) only, but we should double-check. fontdue
   ships with its own glyph outline tessellator, so no extra curve support
   needed for text.

## Test plan

### Unit tests (Rust)

- `src/path.rs` tests: code sequences → expected tiny_skia path commands
- `src/gc.rs` tests: GraphicsContext-like Python objects → expected Stroke
- `src/text.rs` tests: fontdue layout gives reasonable width estimates

### Integration tests (Python/pytest)

- Unskip 5 pixel tests in `test_patches_upstream.py` (1A pass criterion)
- Add a new `test_renderer_agg_smoke.py` that:
  - Creates a `RendererAgg` directly
  - Calls each method
  - Verifies `buffer_rgba()` shape and dtype
  - Verifies `get_text_width_height_descent` returns positive floats

### Full suite regression

After every milestone: run `make test` and verify test count is
monotonically non-decreasing. Any OG-behavior test that *breaks* because
our metrics differ from FreeType should be tracked and either:
- Adapted (if the assertion is too strict)
- Re-skipped (if it truly needs FreeType)
- Fixed (if it's a real bug in our code)

## Rollback plan

If Phase 1 is unworkable after several commits:
- The `_backend_agg.py` no-op stub can be restored from git history
- The Makefile reverts to building only `matplotlib-path`
- The 4401 Phase 0 baseline is recoverable

## Resolved design decisions

1. **Image scaling for 1A**: nearest-neighbor only. Simpler, fast enough
   for `imshow` smoke tests. 1B adds bilinear via `PixmapPaint::filter`.
2. **Antialiasing**: always on in 1A (hardcoded `FillRule::EvenOdd` and
   `BlendMode::SourceOver` with AA). 1B honors `gc.get_antialiased()`.
3. **Premultiplied alpha**: tiny-skia stores premultiplied internally.
   `buffer_rgba()` must **unpremultiply** before returning, because OG's
   agg returns unpremultiplied RGBA and PIL expects unpremultiplied for
   `RGBA` mode. Implement as a post-draw pass: `for px in data: if a > 0:
   r /= a; g /= a; b /= a` with saturation clamping.
4. **Path with None codes**: treat as implicit MOVETO(v[0]) followed by
   LINETO(v[1..]). Matches matplotlib's documented behavior.
5. **Text rotation in 1A**: implemented via `tiny_skia::Transform::from_rotate`
   applied to a glyph mask blit. Arbitrary angles work from day 1
   (tiny-skia handles the affine for us); 1B only tightens snap-to-pixel.
6. **Buffer copy in 1A**: return an owned numpy array via
   `PyArray3::from_slice_bound(py, &[h, w, 4], &unpremul_buffer)` to
   avoid lifetime gymnastics. 1B optimizes to zero-copy.
