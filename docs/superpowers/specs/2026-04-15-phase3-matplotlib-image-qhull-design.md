# Phase 3 — `matplotlib-image` + `matplotlib-qhull`

Date: 2026-04-15
Status: approved

## Goal

Replace the two remaining matplotlib C-extension stubs (`_image` and
`_qhull`) with PyO3 Rust crates, matching the pattern established by
Phase 1 (`matplotlib-agg`) and Phase 2 (`matplotlib-ft2font`). `_contour`
is not a matplotlib C extension — it is the third-party `contourpy`
library — so it is out of scope.

## Non-goals

- No pixel-perfect parity with AGG's C++ resampler. Goal is rendering
  parity: visually indistinguishable output for the common filters,
  matching numeric output for NEAREST + BILINEAR.
- No port of Qhull itself. We use `spade` (pure-Rust Delaunay) whose
  output shape matches scipy/qhull for the operations matplotlib calls.

## Layout

```
crates/
  matplotlib-image/          # new — Phase 3A
    Cargo.toml
    src/lib.rs               # #[pymodule] _image
    src/resample.rs          # sampler + filter kernels
  matplotlib-qhull/          # new — Phase 3B
    Cargo.toml
    src/lib.rs               # #[pymodule] _qhull
    src/delaunay.rs          # spade-backed Delaunay
```

Install targets (per `Makefile` platform detection):
- `python/matplotlib/_image.so` / `.pyd`
- `python/matplotlib/_qhull.so` / `.pyd`

Stubs `python/matplotlib/_image.py` and `python/matplotlib/_qhull.py`
are deleted when their crate ships, same as the Phase 2 policy.

## Milestone 3A — `matplotlib-image`

### API surface

One function + 17 module constants. Surface is validated against every
call site in `python/matplotlib/image.py` and `python/matplotlib/colors.py`.

```python
NEAREST   = 0   BILINEAR = 1   BICUBIC  = 2   SPLINE16 = 3
SPLINE36  = 4   HANNING  = 5   HAMMING  = 6   HERMITE  = 7
KAISER    = 8   QUADRIC  = 9   GAUSSIAN = 11  BESSEL   = 12
MITCHELL  = 13  SINC     = 14  LANCZOS  = 15  BLACKMAN = 16
CATROM    = 10

def resample(input, output, transform,
             interpolation=BILINEAR, resample=False,
             alpha=1.0, norm=False, radius=1.0) -> None
```

`input` and `output` are numpy arrays, either 2D (`float32` / `float64`
/ `uint8`) or 3D with last dim 4 (RGBA). `transform` is a
`matplotlib.transforms.Affine2DBase` — we read `.get_matrix()` to get
the 3×3 forward matrix and invert it for output→input sampling.
`output` is written in place.

### Algorithm

For each output pixel `(ox, oy)`:

1. Compute input coordinate `(ix, iy) = inv_transform · (ox + 0.5, oy + 0.5)`.
2. Sample input at `(ix, iy)` using the selected filter.
3. If `alpha != 1.0`, scale sample; if output is RGBA, premultiply alpha
   into RGB during composite.
4. Store into `output[oy, ox]`.

Filters:
- **NEAREST** — `input[round(iy), round(ix)]`, zero outside bounds.
- **BILINEAR** — 2×2 weighted sample.
- **Windowed kernels** (everything else) — share one path. Each filter
  defines a 1D kernel `k(x)` and a half-width `support`. For each output
  pixel, we sum `input[iy + dy, ix + dx] * k(dx) * k(dy)` over
  `(dx, dy) ∈ [-support, +support]`. `radius` scales the support for
  sinc-family filters. `norm=True` divides the accumulated sample by
  the summed kernel weight.

Kernel table:
| filter    | kernel                                           | support |
|-----------|--------------------------------------------------|---------|
| HANNING   | `0.5 + 0.5·cos(π·x)` for `|x|<1`                 | 1       |
| HAMMING   | `0.54 + 0.46·cos(π·x)` for `|x|<1`               | 1       |
| HERMITE   | `(2|x|³ − 3x² + 1)` for `|x|<1`                  | 1       |
| QUADRIC   | piecewise quadratic                              | 1.5     |
| BICUBIC   | Mitchell–Netravali B=0, C=1 (Catmull–Rom)        | 2       |
| CATROM    | Catmull–Rom                                      | 2       |
| MITCHELL  | Mitchell–Netravali B=1/3, C=1/3                  | 2       |
| GAUSSIAN  | `exp(−2x²)·√(2/π)`                               | 2       |
| SPLINE16  | piecewise cubic                                  | 2       |
| SPLINE36  | piecewise cubic                                  | 3       |
| KAISER    | Kaiser, α=6.33                                   | 1       |
| SINC      | `sinc(x)`                                        | `radius`|
| LANCZOS   | `sinc(x)·sinc(x/radius)`                         | `radius`|
| BESSEL    | Bessel J₁                                        | 3.2383  |
| BLACKMAN  | Blackman window                                  | 1       |

(Kernel formulas are standard; table values cross-checked against AGG's
`agg_image_filters.cpp` and matplotlib's `src/_image_resample.h`.)

### Sequencing

1. **3A.1** — scaffold (Cargo.toml, lib.rs with module constants + empty
   `resample` stub).
2. **3A.2** — NEAREST + BILINEAR. That alone unblocks
   `composite_images()` and default `imshow` with `interpolation='none'`.
3. **3A.3** — the full filter table via the shared windowed-resample
   path. Tests compare against AGG where we have baselines and
   against self-consistency (identity transform, scale-2× round-trip)
   where we don't.
4. **3A.4** — smoke tests, delete stub, Makefile, commit.

## Milestone 3B — `matplotlib-qhull`

### API surface

```python
class Delaunay:
    def __init__(self, points: np.ndarray[float, (N, 2)]): ...
    @property
    def simplices(self) -> np.ndarray[int32, (M, 3)]: ...
    @property
    def neighbors(self) -> np.ndarray[int32, (M, 3)]: ...
    def find_simplex(self, xi: np.ndarray[float, (K, 2)]
                    ) -> np.ndarray[int32, (K,)]: ...
```

`simplices[i]` is three vertex indices (into `points`) for the `i`th
triangle. `neighbors[i, j]` is the index of the triangle opposite
vertex `j`, or `-1` if that edge is on the convex hull. `find_simplex`
returns the containing triangle index for each query point, or `-1`.

### Implementation

`spade::DelaunayTriangulation<Point2<f64>>`:

- Build once in `__init__` by inserting each point. Store in `Py<Self>`
  as `DelaunayTriangulation` + an index map (spade's handles → input
  row).
- `simplices` — walk `triangulation.inner_faces()`, collect three vertex
  indices per face into a cached `(M, 3)` i32 ndarray. Cache because
  matplotlib reads this repeatedly.
- `neighbors` — for each face, for each of its three edges, follow the
  twin half-edge to its face (or -1 if outer). Cache likewise.
- `find_simplex` — for each query point, `triangulation.locate(point)`
  returns a face handle or a position result; translate to triangle
  index or -1.

### Sequencing

1. **3B.1** — scaffold (Cargo.toml with spade dep, lib.rs with stub
   `Delaunay` class).
2. **3B.2** — full implementation: `__init__`, `simplices`, `neighbors`,
   `find_simplex`.
3. **3B.3** — smoke tests, delete stub, Makefile, commit.

## Testing

Smoke tests per crate, analogous to `test_backend_agg_smoke.py` and
`test_ft2font_smoke.py`:

**`test_image_smoke.py`:**
- `test_resample_identity_nearest` — identity transform, NEAREST,
  output == input.
- `test_resample_identity_bilinear` — identity transform, BILINEAR,
  output == input.
- `test_resample_scale_2x_nearest` — 2×2 → 4×4 via scale-2 transform.
- `test_resample_scale_half_bilinear` — 4×4 → 2×2 downsample.
- `test_resample_rgba_alpha_multiply` — alpha=0.5 halves RGB channels.
- `test_resample_lanczos_identity` — identity + LANCZOS returns input.
- `test_interpolation_constants_present` — all 17 constants exposed.

**`test_qhull_smoke.py`:**
- `test_delaunay_square_grid` — 3×3 grid, expect 8 triangles.
- `test_simplices_shape` — returned array is `(M, 3)` int32.
- `test_neighbors_shape_and_hull` — hull edges are -1.
- `test_find_simplex_inside` — point at grid center finds a triangle.
- `test_find_simplex_outside` — point outside returns -1.

Plus the full pytest suite runs after each milestone. Expect
currently-skipped `image` and `tri` tests to flip to passing.

## Build

Extend `Makefile` `build-ext` target:

```make
build-ext: build-agg build-ft2font build-image build-qhull

build-image:
	PYO3_PYTHON=$(VENV_PYTHON) cargo build --release \
	    -p matplotlib-image
	cp target/release/$(DYLIB_PREFIX)matplotlib_image$(DYLIB_EXT) \
	    python/matplotlib/_image.$(PY_EXT)

build-qhull:
	PYO3_PYTHON=$(VENV_PYTHON) cargo build --release \
	    -p matplotlib-qhull
	cp target/release/$(DYLIB_PREFIX)matplotlib_qhull$(DYLIB_EXT) \
	    python/matplotlib/_qhull.$(PY_EXT)
```

`.gitignore` gains `_image.so`/`.pyd` and `_qhull.so`/`.pyd`.

## Risk / known unknowns

- **spade ↔ qhull triangulation shape divergence.** Both produce valid
  Delaunay triangulations of the same point set but may order vertices
  or faces differently. matplotlib doesn't depend on ordering — it uses
  `find_simplex` + neighbor walks — so this should be safe, but the
  `tri` test suite is the ground truth.
- **Filter numerical divergence from AGG.** Our kernels are standard
  textbook formulas; AGG's may have specific coefficient quirks. If
  matplotlib's baseline-image tests diverge, we tighten per-filter.
- **`transform.get_matrix()` coordinate conventions.** matplotlib's
  affine transforms are column vectors with origin at the lower-left
  corner of the image array; `resample` must handle the y-flip that
  composite_images does not pre-apply.
