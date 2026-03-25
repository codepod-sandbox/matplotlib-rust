# matplotlib-rust

A Matplotlib implementation in Python for code running in sandboxed environments (RustPython/WASM).

**35,479 tests passing (`2026-03-25`)**

## How it works

```
Python code (import matplotlib.pyplot as plt)
        │
   matplotlib/ package (pure Python)
        │
   RustPython runtime (WASM sandbox)
        │
   numpy-rust + pillow-rust (Rust native backends)
```

The entire matplotlib API surface is implemented in pure Python. Rendering backends (SVG, PIL) delegate pixel operations to Rust via RustPython bindings.

## Test coverage

| Suite | Result |
|---|---|
| Core axes & plot types | 2,800+ passed |
| Figure, subplots, gridspec | 700+ passed |
| Lines, patches, text, collections | 900+ passed |
| Colors, colormaps, norms | 600+ passed |
| Ticker, transforms, legend, table | 700+ passed |
| Cycler, artist, container, pyplot | 700+ passed |
| Scale, rendering, backend, rcsetup | 600+ passed |

### Supported plot types

`plot`, `scatter`, `bar`, `barh`, `hist`, `pie`, `stem`, `step`, `fill_between`, `stackplot`, `errorbar`, `boxplot`, `violinplot`, `hlines`, `vlines`, `axhline`, `axvline`, `axhspan`, `axvspan`, `imshow`, `pcolormesh`, `contour`, `contourf`, `loglog`, `semilogx`, `semilogy`

### Supported modules

```
matplotlib/
├── pyplot.py          # plt.* wrappers
├── axes.py            # Axes class (~3,200 lines)
├── figure.py          # Figure, subplots
├── lines.py           # Line2D
├── patches.py         # Rectangle, Circle, Polygon, Wedge, ...
├── text.py            # Text, Annotation
├── collections.py     # LineCollection, PolyCollection, EventCollection
├── colors.py          # Normalize, TwoSlopeNorm, BoundaryNorm, ...
├── cm.py              # Colormap, get_cmap, ScalarMappable
├── ticker.py          # Locators, Formatters
├── transforms.py      # Bbox, Affine2D, BboxTransform, ...
├── legend.py          # Legend
├── table.py           # Table, Cell
├── cycler.py          # Cycler
├── artist.py          # Artist base class
├── container.py       # BarContainer, ErrorbarContainer, StemContainer
├── gridspec.py        # GridSpec, GridSpecFromSubplotSpec
└── rcsetup.py         # rcParams (100+ keys)
```

## Running tests

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/ -q
```

## Sister projects

- [numpy-rust](../numpy-rust) — NumPy in Rust (12,133 tests)
- [pillow-rust](../pillow-rust) — Pillow image backend in Rust
