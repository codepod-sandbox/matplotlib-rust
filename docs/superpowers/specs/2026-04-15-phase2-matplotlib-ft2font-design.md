# Phase 2 Design — `crates/matplotlib-ft2font` (Rust FT2Font)

**Status:** approved, implementation in progress
**Predecessor:** Phase 1 spec (`2026-04-15-phase1-matplotlib-agg-design.md`) — complete, 4431 passing
**Successor plans:** Phase 3 (image/contour/qhull), Phase 4 (PDF/PS backends)

## Goal

Replace the Python `ft2font.py` no-op stub (synthetic metrics, blank
bitmaps, empty paths) with a Rust extension backed by `fontdue`
(rasterization) and `ttf-parser` (outline extraction + metadata).

After Phase 2, text in `savefig(png)`, `savefig(svg)`, and
`canvas.draw()` renders with real glyphs. The ~17 tests currently
skipped behind ft2font / synthetic-metrics issues become runnable.

## Phase boundary note

Phase 1 (`matplotlib-agg`) already handles the rendering side:
`RendererAgg::draw_text_image(obj, x, y, angle, gc)` extracts a
grayscale bitmap from `obj.get_image()` and blits it with rotation and
clip. Phase 2 doesn't touch `matplotlib-agg` — it just replaces the
`FT2Font.get_image()` implementation so the bitmap is real instead of
a 1×1 zero array. Same call path; new pixels.

For SVG, OG's `backends/backend_svg.py:_draw_text_as_path` calls
`text2path.get_glyphs_with_font` → `font._get_fontmap(s)` and
`font.get_path()`. Phase 2 implements both.

## Delete the Python stub

Per the user directive "remove stubs once we have the real thing":

- `python/matplotlib/ft2font.py` is **deleted** once the `.so` ships in
  2A. The extension is the only source.
- Retroactively, `python/matplotlib/backends/_backend_agg.py` is also
  **deleted** since Phase 1 already shipped the real Rust extension.
- `make test` now hard-depends on `make build-ext`. No graceful
  fallback; a missing `.so` is a build error.

This changes the Phase 1 "keep `.py` as fallback" policy. Upside:
there is one authoritative implementation per extension, and the
`.py` shadowing ambiguity goes away. Downside: contributors without
Rust installed can't run tests. Acceptable given the project's Rust
focus.

## Architecture

### Crate layout

```
crates/matplotlib-ft2font/
├── Cargo.toml          # pyo3 0.25, numpy, ndarray, fontdue, ttf-parser
└── src/
    ├── lib.rs          # #[pymodule] ft2font; class + constants
    ├── font.rs         # FT2Font pyclass: loading, metrics, layout
    ├── raster.rs       # draw_glyphs_to_bitmap via fontdue
    └── outline.rs      # get_path via ttf_parser::OutlineBuilder
```

### Why two font libraries

- **`fontdue 0.9`** — pure-Rust TTF/OTF rasterizer. Fast glyph bitmap
  rendering. Used for `draw_glyphs_to_bitmap` / `get_image`.
- **`ttf-parser 0.24`** — pure-Rust TTF parser with `Face::outline_glyph`
  that yields move/line/quad/curve segments via `OutlineBuilder`. Used
  for `get_path`, font metadata (`get_sfnt_table`, `units_per_EM`,
  `ascender`, `descender`), kerning, glyph name lookup.

Both libraries consume the same `Vec<u8>` font bytes read once via
`std::fs::read(filename)`. The FT2Font struct owns the bytes; fontdue
owns its parsed state; ttf-parser's `Face` is borrowed and rebuilt
per call (cheap — it's just index lookups into the owned Vec).

### Build artifacts

- Cargo cdylib → `python/matplotlib/ft2font.{so,pyd}`
- Module name `ft2font` via `#[pymodule] fn ft2font(...)`
- Installed by updated `Makefile:build-ext`

### Makefile changes

```make
FT2FONT_OUT := python/matplotlib/ft2font.$(PY_EXT)

build-ext:
	cargo build -p matplotlib-agg -p matplotlib-ft2font
	cp target/debug/$(DYLIB_PREFIX)matplotlib_agg.$(DYLIB_EXT) $(AGG_OUT)
	cp target/debug/$(DYLIB_PREFIX)matplotlib_ft2font.$(DYLIB_EXT) $(FT2FONT_OUT)
```

`.gitignore`: add `python/matplotlib/ft2font.so` and
`python/matplotlib/ft2font.pyd`.

## Public API (pyclass surface)

Surface verified against the current `python/matplotlib/ft2font.py`
stub and every usage site in the OG Python code:

```rust
#[pyclass(unsendable, module = "matplotlib.ft2font")]
struct FT2Font {
    // Owned font bytes (so ttf-parser's Face can be rebuilt per call)
    font_data: Vec<u8>,
    fontdue: fontdue::Font,
    // Rendering state
    ptsize: f32,
    dpi: f32,
    current_text: String,
    current_angle: f32,
    // Laid-out glyph state from set_text, consumed by
    // draw_glyphs_to_bitmap.
    laid_out: Vec<Positioned>,
    bitmap: ndarray::Array2<u8>,     // (H, W) u8 grayscale, zero on clear
    bitmap_offset: (i32, i32),       // 26.6 subpixels
    width: f32,                      // 26.6 subpixels
    height: f32,                     // 26.6 subpixels
    descent: f32,                    // 26.6 subpixels

    // Public attributes matching OG
    #[pyo3(get)] fname: String,
    #[pyo3(get)] family_name: String,
    #[pyo3(get)] style_name: String,
    #[pyo3(get)] postscript_name: String,
    #[pyo3(get)] num_faces: u32,
    #[pyo3(get)] face_flags: u32,
    #[pyo3(get)] style_flags: u32,
    #[pyo3(get)] num_glyphs: u32,
    #[pyo3(get)] num_fixed_sizes: u32,
    #[pyo3(get)] num_charmaps: u32,
    #[pyo3(get)] scalable: bool,
    #[pyo3(get)] units_per_EM: u16,
    #[pyo3(get)] underline_position: i16,
    #[pyo3(get)] underline_thickness: i16,
    #[pyo3(get)] bbox: (i16, i16, i16, i16),
    #[pyo3(get)] ascender: i16,
    #[pyo3(get)] descender: i16,
    #[pyo3(get)] height: i16,
    #[pyo3(get)] max_advance_width: u16,
    #[pyo3(get)] max_advance_height: i16,
}
```

```rust
#[pymethods]
impl FT2Font {
    #[new]
    #[pyo3(signature = (filename, hinting_factor=8, *, _fallback_list=None, _kerning=false, _kerning_factor=None))]
    fn new(filename: &str, hinting_factor: u32,
           _fallback_list: Option<&Bound<PyAny>>,
           _kerning: bool, _kerning_factor: Option<f32>) -> PyResult<Self>;

    fn set_size(&mut self, ptsize: f32, dpi: f32);
    fn set_text(&mut self, s: &str, angle: f32, flags: i32);
    fn get_width_height(&self) -> (f32, f32);  // 26.6 fixed-point
    fn get_descent(&self) -> f32;              // 26.6 fixed-point

    fn draw_glyphs_to_bitmap(&mut self, antialiased: bool);
    fn draw_glyph_to_bitmap(&mut self, image: &Bound<PyAny>, x: i32, y: i32,
                            glyph: &Bound<PyAny>, antialiased: bool);
    fn get_image<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>>;
    fn get_bitmap_offset(&self) -> (i32, i32);

    fn load_char(&self, codepoint: u32, flags: i32) -> PyResult<Glyph>;
    fn load_glyph(&self, glyph_index: u32, flags: i32) -> PyResult<Glyph>;
    fn get_char_index(&self, codepoint: u32) -> u32;
    fn get_name_index(&self, name: &str) -> u32;
    fn get_glyph_name(&self, index: u32) -> String;
    fn get_num_glyphs(&self) -> u32;
    fn get_charmap(&self) -> HashMap<u32, u32>;
    fn select_charmap(&mut self, i: u32);
    fn set_charmap(&mut self, i: u32);
    fn get_kerning(&self, left: u32, right: u32, mode: i32) -> i32;

    fn get_sfnt(&self) -> HashMap<(u32, u32, u32, u32), Vec<u8>>;
    fn get_sfnt_table(&self, name: &str) -> Option<HashMap<String, PyObject>>;
    fn get_ps_font_info(&self) -> (String, String, String, String, String,
                                    u32, i32, i32, i32, i32, i32, i32, i32, i32);

    fn _get_fontmap(&self, s: &str, py: Python<'_>)
                    -> PyResult<HashMap<String, PyObject>>;
    fn get_path<'py>(&self, py: Python<'py>)
                     -> PyResult<(Bound<'py, PyArray2<f64>>,
                                  Bound<'py, PyArray1<u8>>)>;

    fn clear(&mut self);
}
```

Plus module-level:

```rust
// Glyph wrapper (load_char / load_glyph return this)
#[pyclass] struct Glyph {
    #[pyo3(get)] width: i32,
    #[pyo3(get)] height: i32,
    #[pyo3(get)] horiBearingX: i32,
    #[pyo3(get)] horiBearingY: i32,
    #[pyo3(get)] horiAdvance: i32,
    #[pyo3(get)] linearHoriAdvance: i32,
    #[pyo3(get)] vertBearingX: i32,
    #[pyo3(get)] vertBearingY: i32,
    #[pyo3(get)] vertAdvance: i32,
    #[pyo3(get)] bbox: (i32, i32, i32, i32),
}

// Enum-like classes (already present on the Python stub;
// reproduced for API compatibility)
#[pyclass] enum Kerning { DEFAULT=0, UNFITTED=1, UNSCALED=2 }
#[pyclass] struct LoadFlags { DEFAULT, NO_SCALE, NO_HINTING, ... }
#[pyclass] struct FaceFlags { SCALABLE, FIXED_SIZES, ... }
#[pyclass] struct StyleFlags { NORMAL, ITALIC, BOLD }

// Module constants (some callers check these)
__freetype_version__ = "2.6.1"
__freetype_build_type__ = "fontdue"

LOAD_DEFAULT = 0
LOAD_NO_SCALE = 1
LOAD_NO_HINTING = 2
LOAD_RENDER = 4
// ... and the FACE_FLAG_*, KERNING_*, LOAD_* module-level int consts
// the stub currently exposes.
```

## 26.6 subpixel convention

matplotlib's `backend_agg.get_text_width_height_descent` does `w /= 64.0`
to convert from ft2font output to pixels. Our metric getters (`get_width_height`, `get_descent`, `get_kerning`, glyph `horiAdvance` etc.) must return 26.6 fixed-point (`px * 64.0` as f32 or i32). Single conversion point at each API boundary.

## Milestones

### 2A — bitmap rendering (one commit)

Pass criterion:
1. `test_ft2font_smoke.py::test_render_hello_returns_pixels` — construct `FT2Font(DejaVuSans.ttf)`, `set_size(12, 72)`, `set_text("Hello", 0, LOAD_FORCE_AUTOHINT)`, `draw_glyphs_to_bitmap()`, `get_image()` returns an `(H, W) uint8` array with non-zero pixels.
2. `test_backend_agg_smoke.py::test_title_pixels_in_png` — `ax.set_title("TITLE"); fig.savefig(buf, 'png')` has non-background pixels in the title region (row above the axes).

Scope:
- `FT2Font::new` loads via `std::fs::read` + `fontdue::Font::from_bytes`
- ttf-parser `Face::parse` once to read `family_name`, `units_per_EM`, `ascender`, `descender`, `height`, `bbox`, `num_glyphs`, `postscript_name`, etc.
- `set_size`, `set_text` — layout using fontdue glyph metrics (width, advance) + kerning pairs. Store `(char, x_offset, glyph_metrics)` per glyph.
- `draw_glyphs_to_bitmap` — rasterize each glyph via `fontdue::Font::rasterize`, composite into a shared `ndarray::Array2<u8>` with pen advances. Compute tight bitmap bbox for `bitmap_offset`.
- `get_image` — return the ndarray via `IntoPyArray`.
- `get_width_height`, `get_descent` — 26.6 scaled real metrics.
- Everything else stays as sensible stub (returns valid shape, no crashes) so `_prepare_font` / font_manager probes don't fail.

Stub removal in 2A:
- Delete `python/matplotlib/ft2font.py`.
- Delete `python/matplotlib/backends/_backend_agg.py` (orphan from Phase 1).
- Update `.gitignore` to not reference them.
- Run full suite; expect 4431 → ≥ 4431 passing.

### 2B — path-based text for SVG (follow-up commit)

Pass criterion: `ax.set_title('SVGTITLE'); fig.savefig('out.svg')` output contains the glyph outlines of 'SVGTITLE' as `<path d="...">` elements. Unskip any remaining `test_text_upstream` / `test_annotation_upstream` tests that depend on SVG text.

Scope:
- `_get_fontmap(s)` returns `{char: self}` for every character (single-font fallback; matplotlib handles fallback chains at the font_manager level which is Phase 3-ish).
- `get_path()` walks the currently-loaded string's glyphs, calls `Face::outline_glyph` on each with an `OutlineBuilder` impl that translates:
    - `move_to(x, y)` → vertex + MOVETO
    - `line_to(x, y)` → vertex + LINETO
    - `quad_to(cx, cy, x, y)` → two vertices + CURVE3, CURVE3
    - `curve_to(c1x, c1y, c2x, c2y, x, y)` → three vertices + CURVE4 × 3
    - `close()` → CLOSEPOLY
- Accumulates into `(vertices: Array2<f64>, codes: Array1<u8>)` which `get_path` returns as `(PyArray2, PyArray1)`.
- Coordinates are in font units; matplotlib scales them based on pt size and dpi at a higher layer.

### 2C — metadata tables + glyph lookup (follow-up commit)

Scope:
- `load_char`, `load_glyph` — populate `Glyph` with bearings/advance from fontdue metrics and ttf-parser horizontal metrics.
- `get_sfnt_table(name)` — return a dict for 'name', 'head', 'os/2' etc. via ttf-parser `Face::tables()`.
- `get_kerning` — ttf-parser `Face::glyph_kerning_horizontal` (legacy `kern` table; GPOS is beyond 2C).
- `get_ps_font_info` — from `Face::tables().post` with fallbacks.

Unskip: mathtext tests gated on real font metrics, if any.

## Risks

1. **Font cache collision.** `matplotlib.font_manager.FontManager` caches a JSON list of known fonts keyed by path. Existing stub cache files (`~/.cache/matplotlib/fontlist-*.json`) remain valid since we return the same list of system fonts. Mitigation: none needed unless tests start failing on stale caches.

2. **fontdue metrics differ from FreeType.** Expected. `tight_layout` positions shift slightly. No regression since Phase 1 already uses synthetic metrics; 2A is strictly an improvement.

3. **`get_path` coordinate system.** ttf-parser emits outlines in font units with y-up and origin at the glyph's advance baseline. matplotlib's `TextToPath.get_glyphs_with_font` applies its own transform before rendering, so we emit in font-native coords and let matplotlib scale. Smoke test: render `"A"` to SVG, manually verify the `<path d="...">` has sensible coordinates.

4. **Unsafe buffer protocol not required.** `get_image` returns a freshly-allocated `PyArray2<u8>`, so no `__getbuffer__` ffi is needed (unlike `BufferRegion`).

5. **Module import order under pytest.** `matplotlib.ft2font` is imported early by `font_manager`. If the `.so` has any linker issue (wrong Python ABI), the error surfaces at matplotlib import time rather than during a specific test. `PYO3_PYTHON` pinning in the Makefile already mitigates this.

6. **Deleting `.py` stubs breaks `pytest` without `make build-ext`.** Intentional; documented in the commit message and CONTRIBUTING note.

## Test plan

### New file: `test_ft2font_smoke.py`

```python
def test_extension_loaded():
    assert ft2font.__file__.endswith((".so", ".pyd"))

def test_render_hello_returns_pixels():
    font = ft2font.FT2Font(str(find('DejaVuSans.ttf')))
    font.set_size(12, 72)
    font.set_text("Hello", 0.0, ft2font.LOAD_FORCE_AUTOHINT)
    w, h = font.get_width_height()
    assert w > 0 and h > 0
    font.draw_glyphs_to_bitmap(antialiased=True)
    bitmap = font.get_image()
    assert bitmap.dtype == np.uint8
    assert bitmap.ndim == 2
    assert (bitmap > 0).sum() > 50, "expected rasterized glyph pixels"

def test_metrics_are_subpixel_units():
    font = ft2font.FT2Font(str(find('DejaVuSans.ttf')))
    font.set_size(12, 72)
    font.set_text("M", 0.0, 0)
    w, _ = font.get_width_height()
    # 26.6 fixed: 'M' at 12pt/72dpi ≈ 9 px × 64 = ~576 subpixels
    assert 300 < w < 1500

def test_get_path_returns_outline_for_A():
    # 2B
    font = ft2font.FT2Font(str(find('DejaVuSans.ttf')))
    font.set_size(12, 72)
    font.set_text("A", 0.0, 0)
    verts, codes = font.get_path()
    assert verts.ndim == 2 and verts.shape[1] == 2
    assert codes.ndim == 1 and codes.shape[0] == verts.shape[0]
    assert (codes == 1).sum() >= 1, "expected at least one MOVETO"
```

### Extend `test_backend_agg_smoke.py`

```python
def test_title_pixels_in_savefig_png():
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.set_title("TTT")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    arr = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('L'))
    top_strip = arr[0:int(arr.shape[0] * 0.12), :]
    non_bg = (top_strip < 250).sum()
    assert non_bg > 20, f"expected title pixels in top strip, got {non_bg}"
```

### Full suite expectation

After 2A:
- `ft2font.py` stub deleted → the 17-ish `"Phase 2: needs real cmr10 font"` / `"needs ft2font"` skips can be audited. Some are genuine (mathtext), some unblock trivially.
- Expect ≥ 4431 passing (no regressions) and ideally +5-15 from unskipped tests.

After 2B:
- SVG text tests that currently check `'SVGTITLE' in svg` start passing when targeting the OG backend_svg path.

After 2C:
- Whatever mathtext / font-metadata tests are still skipped start running.

## Rollback plan

Each milestone is a separate commit. If 2A breaks things, revert to the
pre-2A baseline via `git revert`. The stub `ft2font.py` can be restored
from git history. `.gitignore` updates are trivially reversible.

## Open questions (resolved)

1. **2A scope includes `get_path`?** No — 2B owns all outline work. 2A
   is bitmap-only and delivers PNG text.
2. **`ttf-parser` vs `ab_glyph`?** `ttf-parser`. Smaller API surface,
   direct outline builder, explicit metadata tables. `ab_glyph` adds
   shaping indirection we don't need.
3. **Smoke test with 2A?** Yes, committed with 2A.
4. **Also delete `_backend_agg.py`?** Yes, retroactively in 2A.
