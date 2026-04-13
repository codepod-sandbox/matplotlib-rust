# OG Matplotlib Phase 0 — Python File Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stub `axes.py` and `figure.py` with verbatim OG matplotlib 3.10.x source; create import stubs for all C extensions so non-rendering tests keep passing and the test count does not decrease.

**Architecture:** Clone `matplotlib/matplotlib` at `v3.10.0` as a reference; write C extension stubs first (so OG imports resolve immediately); then copy Python files verbatim preserving the `axes/` package structure and `backends/` directory; keep `rcsetup.py`, `_codepod_compat.py`, `_pil_backend.py`, `_svg_backend.py`, and all custom test files unchanged; run full suite and fix any remaining import gaps.

**Tech Stack:** Python 3, RustPython, PyO3 0.25, pytest, git

---

> **Scope note:** This plan covers **Phase 0 only** (Python file migration + C extension stubs). Phases 1–4 (Rust crates for `_backend_agg`, `ft2font`, `_image`, `_contour`, `_qhull`) require separate plans, to be written after Phase 0 completes and a new test baseline is measured.

---

## File Map

**Files to CREATE (new):**
- `python/matplotlib/_backend_agg.py` — RendererAgg import stub
- `python/matplotlib/ft2font.py` — FT2Font import stub
- `python/matplotlib/_qhull.py` — Delaunay import stub
- `python/contourpy/__init__.py` — contourpy package stub
- `python/matplotlib/axes/__init__.py` — OG package (replaces axes.py)
- `python/matplotlib/axes/_axes.py` — OG Axes class
- `python/matplotlib/axes/_base.py` — OG _AxesBase
- `python/matplotlib/axes/_secondary_axes.py` — OG SecondaryAxis
- `python/matplotlib/axes/_subplots.py` — OG subplot helpers
- `python/matplotlib/backends/__init__.py` — OG backends package
- `python/matplotlib/backends/backend_agg.py` — OG Agg backend
- `python/matplotlib/backends/backend_svg.py` — OG SVG backend
- `python/matplotlib/backends/backend_pdf.py` — OG PDF backend
- `python/matplotlib/backends/backend_ps.py` — OG PS backend
- `python/matplotlib/backends/backend_template.py` — OG template

**Files to DELETE:**
- `python/matplotlib/axes.py` — replaced by axes/ package

**Files to COPY VERBATIM from OG (replace current versions):**
- `python/matplotlib/figure.py`
- `python/matplotlib/backend_bases.py`
- `python/matplotlib/image.py`
- `python/matplotlib/contour.py`
- `python/matplotlib/text.py`
- `python/matplotlib/font_manager.py`
- `python/matplotlib/mathtext.py`
- `python/matplotlib/tri/` (entire package if it exists in 3.10.x)

**Files to KEEP (our versions):**
- `python/matplotlib/rcsetup.py`
- `python/matplotlib/__init__.py` (update `__version__` field only)
- `python/matplotlib/_codepod_compat.py`
- `python/matplotlib/_pil_backend.py`
- `python/matplotlib/_svg_backend.py`
- `python/matplotlib/_image.py`
- `python/matplotlib/testing/conftest.py`
- `python/matplotlib/tests/` (all our test files)

---

### Task 1: Create `_backend_agg` import stub

**Files:**
- Create: `python/matplotlib/_backend_agg.py`

The stub must be importable and expose the same class/function names that `backends/backend_agg.py` imports. No rendering needed — any rendering call raises `NotImplementedError`.

- [ ] **Step 1: Write the stub**

```python
# python/matplotlib/_backend_agg.py
"""Stub for matplotlib._backend_agg.

Replaced by crates/matplotlib-agg (tiny-skia + fontdue) in Phase 1.
"""


def get_hinting_flag():
    return 0


class RendererAgg:
    """Stub RendererAgg — raises NotImplementedError on all drawing calls."""

    def __init__(self, width, height, dpi):
        self.width = int(width)
        self.height = int(height)
        self.dpi = dpi

    def get_image_magnification(self):
        return 1.0

    def clear(self):
        pass

    def draw_path(self, gc, path, transform, rgbFace=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_markers(self, gc, marker_path, marker_trans, path, trans,
                     rgbFace=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_path_collection(self, gc, master_transform, paths,
                             all_transforms, offsets, offset_trans, facecolors,
                             edgecolors, linewidths, linestyles, antialiaseds,
                             urls, offset_position):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_gouraud_triangle(self, gc, points, colors, transform):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_image(self, gc, x, y, im):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def tostring_rgb(self):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def tostring_argb(self):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def buffer_rgba(self):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def copy_from_bbox(self, bbox):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def restore_region(self, region, bbox=None, xy=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")
```

- [ ] **Step 2: Verify the stub is importable**

```bash
target/debug/matplotlib-python -c "from matplotlib._backend_agg import RendererAgg, get_hinting_flag; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit stub**

```bash
git add python/matplotlib/_backend_agg.py
git commit -m "feat: add _backend_agg import stub (Phase 0)"
```

---

### Task 2: Create `ft2font` import stub

**Files:**
- Create: `python/matplotlib/ft2font.py`

`font_manager.py` reads attributes (`family_name`, `style_flags`, etc.) from `FT2Font` instances when scanning fonts during module initialization. The stub must set those attributes in `__init__` without raising — so font scanning completes without crashing. Actual rendering calls raise `NotImplementedError`.

- [ ] **Step 1: Write the stub**

```python
# python/matplotlib/ft2font.py
"""Stub for matplotlib.ft2font.

Replaced by crates/matplotlib-ft2font (fontdue) in Phase 2.
"""

# Load flags (values match OG ft2font)
LOAD_DEFAULT = 0
LOAD_NO_SCALE = 1
LOAD_NO_HINTING = 2
LOAD_RENDER = 4
LOAD_NO_BITMAP = 8
LOAD_VERTICAL_LAYOUT = 16
LOAD_FORCE_AUTOHINT = 32
LOAD_CROP_BITMAP = 64
LOAD_PEDANTIC = 128
LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH = 512
LOAD_NO_RECURSE = 1024
LOAD_IGNORE_TRANSFORM = 2048
LOAD_MONOCHROME = 4096
LOAD_LINEAR_DESIGN = 8192
LOAD_NO_AUTOHINT = 32768

# Kerning modes
KERNING_DEFAULT = 0
KERNING_UNFITTED = 1
KERNING_UNSCALED = 2

# Face flags
FACE_FLAG_SCALABLE = 1
FACE_FLAG_FIXED_SIZES = 2
FACE_FLAG_FIXED_WIDTH = 4
FACE_FLAG_SFNT = 8
FACE_FLAG_HORIZONTAL = 16
FACE_FLAG_VERTICAL = 32
FACE_FLAG_KERNING = 64
FACE_FLAG_FAST_GLYPHS = 128
FACE_FLAG_MULTIPLE_MASTERS = 256
FACE_FLAG_GLYPH_NAMES = 512
FACE_FLAG_BOLD = 1
FACE_FLAG_ITALIC = 2

# Style flags
STYLE_FLAG_ITALIC = 1
STYLE_FLAG_BOLD = 2


class FT2Font:
    """Stub FT2Font.

    Attributes are populated so font_manager can scan fonts without crashing.
    Rendering calls raise NotImplementedError.
    """

    def __init__(self, filename, hinting_factor=8, *, _fallback_list=None,
                 _kerning=False):
        # font_manager reads these attributes after construction
        self.fname = filename
        self.family_name = ""
        self.style_name = ""
        self.face_flags = FACE_FLAG_SCALABLE | FACE_FLAG_HORIZONTAL
        self.style_flags = 0
        self.num_fixed_sizes = 0
        self.num_charmaps = 0
        self.scalable = True
        self.units_per_EM = 2048
        self.underline_position = -100
        self.underline_thickness = 50
        self.bbox = (0, 0, 2048, 2048)
        self.ascender = 800
        self.descender = -200
        self.height = 1200
        self.max_advance_width = 1200
        self.max_advance_height = 1200
        self.num_glyphs = 0
        self.postscript_name = ""

    def set_size(self, ptsize, dpi):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def set_text(self, s, angle=0.0, flags=LOAD_FORCE_AUTOHINT):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_width_height(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_descent(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def draw_glyphs_to_bitmap(self, antialiased=True):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_image(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def load_char(self, charcode, flags=LOAD_FORCE_AUTOHINT):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def load_glyph(self, glyphindex, flags=LOAD_FORCE_AUTOHINT):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_kerning(self, left, right, mode):
        return 0

    def get_charmap(self):
        return {}

    def get_name_index(self, name):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_sfnt(self):
        return {}

    def get_sfnt_table(self, name):
        return None

    def get_xys(self, antialiased=True):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_glyph_name(self, index):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_num_glyphs(self):
        return 0

    def get_path(self):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def get_ps_font_info(self):
        return None

    def select_charmap(self, i):
        pass


class FT2Image:
    """Stub FT2Image."""

    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)

    def draw_rect(self, x0, y0, x1, y1):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")

    def draw_rect_filled(self, x0, y0, x1, y1):
        raise NotImplementedError("ft2font not yet implemented (Phase 2)")
```

- [ ] **Step 2: Verify importable**

```bash
target/debug/matplotlib-python -c "from matplotlib.ft2font import FT2Font, FT2Image, LOAD_FORCE_AUTOHINT; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit stub**

```bash
git add python/matplotlib/ft2font.py
git commit -m "feat: add ft2font import stub (Phase 0)"
```

---

### Task 3: Create `_qhull` import stub

**Files:**
- Create: `python/matplotlib/_qhull.py`

Used by `matplotlib.tri` for Delaunay triangulation. Stub raises `NotImplementedError`.

- [ ] **Step 1: Write the stub**

```python
# python/matplotlib/_qhull.py
"""Stub for matplotlib._qhull.

Replaced by crates/matplotlib-qhull (spade crate) in Phase 3.
"""


class Delaunay:
    """Stub Delaunay triangulation."""

    def __init__(self, points):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")

    @property
    def simplices(self):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")

    @property
    def neighbors(self):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")

    def find_simplex(self, xi):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")
```

- [ ] **Step 2: Verify importable**

```bash
target/debug/matplotlib-python -c "from matplotlib._qhull import Delaunay; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit stub**

```bash
git add python/matplotlib/_qhull.py
git commit -m "feat: add _qhull import stub (Phase 0)"
```

---

### Task 4: Create `contourpy` stub package

**Files:**
- Create: `python/contourpy/__init__.py`

OG matplotlib 3.10.x imports `contour_generator`, `CoordinateType`, `FillType`, `LineType` from `contourpy` at the top of `contour.py`. The stub must make these importable; actual contour generation raises `NotImplementedError`.

- [ ] **Step 1: Create package directory and `__init__.py`**

```bash
mkdir -p python/contourpy
```

```python
# python/contourpy/__init__.py
"""Stub for contourpy — contour line/polygon generation.

Replaced by actual contourpy or crates/matplotlib-contour in Phase 3.
See https://contourpy.readthedocs.io/
"""

__version__ = "1.3.0"


class CoordinateType:
    Separate = 0
    SeparateCode = 1
    ChunkCombinedArray = 2
    ChunkCombinedCodesOffsets = 3
    ChunkCombinedOffset = 4
    ChunkCombinedNan = 5


class FillType:
    OuterCode = 0
    OuterOffset = 1
    ChunkCombinedCode = 2
    ChunkCombinedOffset = 3
    ChunkCombinedCodeOffset = 4
    ChunkCombinedOffsetOffset = 5


class LineType:
    Separate = 0
    SeparateCode = 1
    ChunkCombinedArray = 2
    ChunkCombinedOffset = 3
    ChunkCombinedNan = 4


def contour_generator(x=None, y=None, z=None, name="serial",
                      corner_mask=None, line_type=None, fill_type=None,
                      chunk_size=None, chunk_count=None,
                      total_chunk_count=None, quad_as_tri=False,
                      z_interp=None, thread_count=0):
    """Stub — raises NotImplementedError (Phase 3)."""
    raise NotImplementedError("contourpy not yet implemented (Phase 3)")
```

- [ ] **Step 2: Verify importable**

```bash
target/debug/matplotlib-python -c "from contourpy import contour_generator, CoordinateType, FillType, LineType; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit stub**

```bash
git add python/contourpy/
git commit -m "feat: add contourpy import stub (Phase 0)"
```

---

### Task 5: Clone matplotlib 3.10.x reference source

**Files:** None in-repo (clone to a temp directory outside the repo)

The clone is read-only reference; we copy from it but never commit it.

- [ ] **Step 1: Check for latest 3.10.x tag**

```bash
git ls-remote --tags https://github.com/matplotlib/matplotlib.git | grep 'refs/tags/v3\.10\.' | sort -V | tail -5
```

Note the latest `v3.10.x` tag (e.g. `v3.10.1`). Use it in the next step.

- [ ] **Step 2: Clone (shallow) to `/tmp/matplotlib-og`**

```bash
git clone --depth 1 --branch v3.10.0 https://github.com/matplotlib/matplotlib.git /tmp/matplotlib-og
```

Replace `v3.10.0` with the latest tag found above if different.

- [ ] **Step 3: Verify the clone has the expected structure**

```bash
ls /tmp/matplotlib-og/lib/matplotlib/
ls /tmp/matplotlib-og/lib/matplotlib/axes/
ls /tmp/matplotlib-og/lib/matplotlib/backends/
```

Expected: `axes/` is a package (directory with `_axes.py`, `_base.py`, etc.); `backends/` directory contains `backend_agg.py`, `backend_svg.py`, etc.

---

### Task 6: Create axes/ package (replaces axes.py)

**Files:**
- Create: `python/matplotlib/axes/__init__.py`
- Create: `python/matplotlib/axes/_axes.py`
- Create: `python/matplotlib/axes/_base.py`
- Create: `python/matplotlib/axes/_secondary_axes.py`
- Create: `python/matplotlib/axes/_subplots.py`
- Delete: `python/matplotlib/axes.py`

- [ ] **Step 1: Copy the axes/ package verbatim**

```bash
cp -r /tmp/matplotlib-og/lib/matplotlib/axes python/matplotlib/axes
```

- [ ] **Step 2: Remove the old axes.py stub**

```bash
rm python/matplotlib/axes.py
```

- [ ] **Step 3: Verify axes imports still work**

```bash
target/debug/matplotlib-python -c "from matplotlib.axes import Axes; from matplotlib import axes; print('ok')"
```
Expected: `ok`

If this raises `ImportError` about a missing dependency (e.g. `kiwisolver`, `pyparsing`), note the error and continue — it will be addressed in Task 9.

- [ ] **Step 4: Commit**

```bash
git add python/matplotlib/axes/
git rm python/matplotlib/axes.py
git commit -m "feat: replace axes.py stub with OG matplotlib 3.10.x axes/ package"
```

---

### Task 7: Create backends/ package

**Files:**
- Create: `python/matplotlib/backends/__init__.py`
- Create: `python/matplotlib/backends/backend_agg.py`
- Create: `python/matplotlib/backends/backend_svg.py`
- Create: `python/matplotlib/backends/backend_pdf.py`
- Create: `python/matplotlib/backends/backend_ps.py`
- Create: `python/matplotlib/backends/backend_template.py`

The OG `backends/__init__.py` and the backend files live alongside our existing `_pil_backend.py` and `_svg_backend.py` (which remain untouched).

- [ ] **Step 1: Copy the backends/ package verbatim**

```bash
cp -r /tmp/matplotlib-og/lib/matplotlib/backends python/matplotlib/backends
```

- [ ] **Step 2: Verify backends imports**

```bash
target/debug/matplotlib-python -c "from matplotlib.backends.backend_agg import FigureCanvasAgg; print('ok')"
```

Expected: `ok` (FigureCanvasAgg is defined in the Python file; RendererAgg is imported from our stub).

- [ ] **Step 3: Commit**

```bash
git add python/matplotlib/backends/
git commit -m "feat: add OG matplotlib 3.10.x backends/ package (Phase 0)"
```

---

### Task 8: Copy remaining Python modules from OG source

**Files:** Multiple — see list below

Copy verbatim from `/tmp/matplotlib-og/lib/matplotlib/` into `python/matplotlib/`. **Skip** the files listed in the "keep our versions" section of the file map above.

- [ ] **Step 1: Record the pre-copy test count**

```bash
make test 2>&1 | tail -5
```

Record the number of passing tests as the Phase 0 baseline floor.

- [ ] **Step 2: Copy figure.py**

```bash
cp /tmp/matplotlib-og/lib/matplotlib/figure.py python/matplotlib/figure.py
```

- [ ] **Step 3: Copy backend_bases.py**

```bash
cp /tmp/matplotlib-og/lib/matplotlib/backend_bases.py python/matplotlib/backend_bases.py
```

- [ ] **Step 4: Copy image.py**

```bash
cp /tmp/matplotlib-og/lib/matplotlib/image.py python/matplotlib/image.py
```

- [ ] **Step 5: Copy contour.py**

```bash
cp /tmp/matplotlib-og/lib/matplotlib/contour.py python/matplotlib/contour.py
```

- [ ] **Step 6: Copy text.py**

```bash
cp /tmp/matplotlib-og/lib/matplotlib/text.py python/matplotlib/text.py
```

- [ ] **Step 7: Copy font_manager.py**

```bash
cp /tmp/matplotlib-og/lib/matplotlib/font_manager.py python/matplotlib/font_manager.py
```

- [ ] **Step 8: Copy remaining top-level Python modules**

Copy all `.py` files that exist in OG but that we haven't explicitly handled yet. **Skip** files listed in the "keep our versions" section of the file map.

```bash
# Copy all .py files from OG, skipping our custom files
OG=/tmp/matplotlib-og/lib/matplotlib
DEST=python/matplotlib
SKIP="rcsetup.py __init__.py _codepod_compat.py _pil_backend.py _svg_backend.py _image.py"

for f in $OG/*.py; do
  name=$(basename "$f")
  skip=0
  for s in $SKIP; do [ "$name" = "$s" ] && skip=1; done
  if [ $skip -eq 0 ] && [ -f "$DEST/$name" ]; then
    cp "$f" "$DEST/$name"
    echo "updated: $name"
  elif [ $skip -eq 0 ] && [ ! -f "$DEST/$name" ]; then
    cp "$f" "$DEST/$name"
    echo "added: $name"
  else
    echo "skipped (keep ours): $name"
  fi
done
```

- [ ] **Step 9: Copy any OG subdirectory packages (tri/, style/, _api/) we don't already have**

```bash
# Copy tri/ package if present in OG
[ -d /tmp/matplotlib-og/lib/matplotlib/tri ] && \
  cp -r /tmp/matplotlib-og/lib/matplotlib/tri python/matplotlib/tri

# Copy _api/ if OG has an updated version
cp -r /tmp/matplotlib-og/lib/matplotlib/_api python/matplotlib/_api

# Copy style/ package
cp -r /tmp/matplotlib-og/lib/matplotlib/style python/matplotlib/style
```

- [ ] **Step 10: Run tests to check for import errors**

```bash
make test 2>&1 | grep -E "(ERROR|ImportError|ModuleNotFoundError)" | head -30
```

Record every distinct import error. Proceed to Task 9 to fix them.

- [ ] **Step 11: Commit the batch copy**

```bash
git add python/matplotlib/
git commit -m "feat: copy OG matplotlib 3.10.x Python modules verbatim (Phase 0)"
```

---

### Task 9: Fix import errors from missing external packages

After copying OG files, some imports will fail because OG matplotlib expects third-party packages (pyparsing, packaging, python-dateutil, fonttools) that may not be available in RustPython. Fix each one.

- [ ] **Step 1: Check which packages are missing**

```bash
target/debug/matplotlib-python -c "
import sys
for pkg in ['pyparsing', 'packaging', 'dateutil', 'fonttools', 'cycler', 'PIL']:
    try:
        __import__(pkg)
        print(f'ok: {pkg}')
    except ImportError as e:
        print(f'MISSING: {pkg} -- {e}')
"
```

- [ ] **Step 2: Create a `pyparsing` stub if missing**

`pyparsing` is used for mathtext. If missing, create a minimal stub so the import doesn't crash module load. Full mathtext parsing requires Task 9 step 3+.

If `pyparsing` is missing, create `python/pyparsing/__init__.py`:

```python
# python/pyparsing/__init__.py
"""Minimal pyparsing stub for RustPython.

OG pyparsing is a pure-Python library; install it as a real package if possible.
This stub prevents ImportError during module load but does not support parsing.
"""

__version__ = "3.1.0"


class ParseException(Exception):
    def __init__(self, msg="", loc=0, pstr="", elem=None):
        super().__init__(msg)
        self.loc = loc
        self.pstr = pstr
        self.msg = msg


class ParserElement:
    pass


class Token(ParserElement):
    pass


class Literal(Token):
    def __init__(self, matchString):
        self.matchString = matchString

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("pyparsing.Literal not implemented (Phase 2)")


class Regex(Token):
    def __init__(self, pattern, flags=0):
        import re
        self.pattern = pattern
        self._re = re.compile(pattern, flags)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("pyparsing.Regex not implemented (Phase 2)")


class Word(Token):
    def __init__(self, initChars, bodyChars=None, min=1, max=0,
                 exact=0, asKeyword=False, excludeChars=None):
        self.initChars = initChars

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("pyparsing.Word not implemented (Phase 2)")


class Optional(ParserElement):
    def __init__(self, expr, default=None):
        self.expr = expr
        self.default = default


class ZeroOrMore(ParserElement):
    def __init__(self, expr):
        self.expr = expr


class OneOrMore(ParserElement):
    def __init__(self, expr):
        self.expr = expr


class Group(ParserElement):
    def __init__(self, expr):
        self.expr = expr


class Forward(ParserElement):
    def __lshift__(self, other):
        return self


class Empty(ParserElement):
    pass


class StringEnd(ParserElement):
    pass


class FollowedBy(ParserElement):
    def __init__(self, expr):
        self.expr = expr


class NotAny(ParserElement):
    def __init__(self, expr):
        self.expr = expr


class pyparsing_common:
    signed_integer = Regex(r'[-+]?\d+')
    number = Regex(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?')


class QuotedString(Token):
    def __init__(self, quoteChar, escChar=None, unquoteResults=True,
                 endQuoteChar=None):
        self.quoteChar = quoteChar


def Suppress(expr):
    return expr


def Combine(*args, **kwargs):
    return ParserElement()


def MatchFirst(exprs):
    return exprs[0] if exprs else ParserElement()


def And(exprs):
    return exprs[0] if exprs else ParserElement()


def Or(exprs):
    return exprs[0] if exprs else ParserElement()


infixNotation = None  # used by mathtext; will be stubbed per usage
operatorPrecedence = infixNotation
```

- [ ] **Step 3: Create a `packaging` stub if missing**

`packaging` is used for version comparison in `matplotlib.__init__` and elsewhere.

If missing, create `python/packaging/__init__.py`:

```python
# python/packaging/__init__.py
"""Minimal packaging stub for RustPython."""

__version__ = "24.0"
```

And `python/packaging/version.py`:

```python
# python/packaging/version.py
"""Minimal packaging.version stub."""


class Version:
    def __init__(self, version_string):
        self._s = version_string
        parts = version_string.split(".")
        self._parts = tuple(int(p) for p in parts if p.isdigit())

    def __str__(self):
        return self._s

    def __repr__(self):
        return f"<Version('{self._s}')>"

    def __lt__(self, other):
        return self._parts < other._parts

    def __le__(self, other):
        return self._parts <= other._parts

    def __gt__(self, other):
        return self._parts > other._parts

    def __ge__(self, other):
        return self._parts >= other._parts

    def __eq__(self, other):
        return self._parts == other._parts


def parse(version_string):
    return Version(version_string)
```

- [ ] **Step 4: Create a `dateutil` stub if missing**

`dateutil` is used for date axis formatting. If missing:

```bash
mkdir -p python/dateutil
```

`python/dateutil/__init__.py`:
```python
# python/dateutil/__init__.py
"""Minimal python-dateutil stub for RustPython."""
__version__ = "2.9.0"
```

`python/dateutil/parser.py`:
```python
# python/dateutil/parser.py
"""Stub dateutil.parser."""

def parse(timestr, default=None, ignoretz=False, tzinfos=None, **kwargs):
    raise NotImplementedError("dateutil.parser.parse not implemented")
```

`python/dateutil/tz/__init__.py` and `python/dateutil/relativedelta.py` — create empty files if imported:
```bash
mkdir -p python/dateutil/tz
touch python/dateutil/tz/__init__.py
touch python/dateutil/relativedelta.py
```

- [ ] **Step 5: Handle `fonttools` if missing**

`fonttools` is used in `font_manager.py` for font subsetting/loading. Wrap its import in a try/except by checking if OG `font_manager.py` already handles it:

```bash
grep -n "fonttools\|import ft" python/matplotlib/font_manager.py | head -10
```

If fonttools is imported unconditionally, create a minimal stub `python/fonttools/__init__.py`:

```python
# python/fonttools/__init__.py
"""Minimal fonttools stub for RustPython."""
__version__ = "4.55.0"
```

- [ ] **Step 6: Verify all imports now resolve**

```bash
target/debug/matplotlib-python -c "
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.backends.backend_agg
import matplotlib.contour
import matplotlib.image
print('all imports ok')
"
```

Expected: `all imports ok`

If any import still fails, read the traceback and add whatever minimal stub is needed.

- [ ] **Step 7: Commit all stubs**

```bash
git add python/pyparsing/ python/packaging/ python/dateutil/ python/fonttools/
git commit -m "feat: add minimal external package stubs for RustPython (Phase 0)"
```

---

### Task 10: Update `__init__.py` version and backend defaults

**Files:**
- Modify: `python/matplotlib/__init__.py`

The only required change is bumping `__version__` to `"3.10.0"`. Do not copy the OG `__init__.py` verbatim — keep our custom WASM/RustPython setup.

- [ ] **Step 1: Update version string**

In `python/matplotlib/__init__.py`, change line:
```python
__version__ = "3.8.0"
```
to:
```python
__version__ = "3.10.0"
```

- [ ] **Step 2: Verify version**

```bash
target/debug/matplotlib-python -c "import matplotlib; print(matplotlib.__version__)"
```
Expected: `3.10.0`

- [ ] **Step 3: Commit**

```bash
git add python/matplotlib/__init__.py
git commit -m "chore: bump matplotlib.__version__ to 3.10.0 (Phase 0)"
```

---

### Task 11: Run full test suite and triage failures

**Files:** `python/matplotlib/tests/conftest.py` (update if needed)

- [ ] **Step 1: Run the full test suite**

```bash
make test 2>&1 | tee /tmp/phase0-test-results.txt
```

- [ ] **Step 2: Count passing vs failing**

```bash
grep -E "passed|failed|error" /tmp/phase0-test-results.txt | tail -3
```

Record: **N passed, M failed/error**.

- [ ] **Step 3: Categorize failures**

```bash
# Show all FAILED and ERROR test names
grep -E "^(FAILED|ERROR)" /tmp/phase0-test-results.txt | head -50
```

Expected failure categories:
- Tests that call `RendererAgg` directly → expected NotImplementedError (Phase 1)
- Tests that call `FT2Font.set_size` → expected NotImplementedError (Phase 2)
- Tests that use `contour_generator` → expected NotImplementedError (Phase 3)
- Tests that use `Delaunay` → expected NotImplementedError (Phase 3)

Unexpected failures (must fix before declaring Phase 0 done):
- Any `ImportError` or `AttributeError` at module load time
- Non-rendering logic tests that used to pass (regression)

- [ ] **Step 4: Fix any regressions in non-rendering tests**

For each test that was passing before and now fails without rendering involvement, trace the failure:

```bash
target/debug/matplotlib-python -m pytest python/matplotlib/tests/<test_file.py>::<test_name> -v -s 2>&1 | head -40
```

Common causes:
- A new OG module expects an attribute we don't provide in `rcsetup.py` → add the missing key with a sensible default
- `axes/__init__.py` exports something our tests import directly → verify import paths

- [ ] **Step 5: Mark known rendering failures as xfail in conftest**

Add to `python/matplotlib/testing/conftest.py`:

```python
# Phase 0 xfail markers: rendering tests that need Phase 1-3 Rust crates
_PHASE1_UNIMPLEMENTED = pytest.mark.xfail(
    reason="requires _backend_agg Rust crate (Phase 1)",
    raises=NotImplementedError,
    strict=False,
)
```

Alternatively, rely on the tests raising `NotImplementedError` and being counted as errors (not regressions from the prior baseline). Choose whichever approach keeps the count ≥ baseline.

- [ ] **Step 6: Verify count ≥ pre-Phase-0 baseline**

```bash
make test 2>&1 | tail -3
```

The passing test count must be **≥ the count recorded in Task 8 Step 1**. If it dropped, fix the regression before proceeding.

---

### Task 12: Final Phase 0 commit

- [ ] **Step 1: Check nothing is left unstaged**

```bash
git status
```

All changes should be committed. If any stub or fix is unstaged, add and commit it.

- [ ] **Step 2: Final commit with summary**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: Phase 0 complete — OG matplotlib 3.10.x Python files + import stubs

- Replace axes.py stub with OG axes/ package (axes/_axes.py, _base.py, etc.)
- Replace figure.py stub with OG matplotlib 3.10.x figure.py
- Copy backends/, backend_bases.py, image.py, contour.py, text.py verbatim
- Add _backend_agg.py stub (Phase 1 placeholder)
- Add ft2font.py stub (Phase 2 placeholder)
- Add _qhull.py stub (Phase 3 placeholder)
- Add contourpy/ stub package (Phase 3 placeholder)
- Add minimal external package stubs (pyparsing, packaging, dateutil)
- Bump __version__ to 3.10.0
- Test count >= pre-migration baseline

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Run tests one final time to confirm baseline**

```bash
make test 2>&1 | tail -5
```

Record the final passing count in a comment or in `docs/superpowers/plans/2026-04-13-og-matplotlib-phase0.md` under a "Results" section at the bottom.

---

## Phase 0 Complete

After this plan:
- `python/matplotlib/` mirrors OG matplotlib 3.10.x Python structure
- All C extension imports resolve (stubs raise `NotImplementedError` on use)
- Non-rendering tests pass at baseline count
- Rendering tests fail gracefully with `NotImplementedError`

**Next steps (separate plans):**
- Phase 1: `crates/matplotlib-agg` — implement `RendererAgg` with `tiny-skia` + `fontdue`
- Phase 2: `crates/matplotlib-ft2font` — implement `FT2Font` with `fontdue`
- Phase 3: `crates/matplotlib-image`, `crates/matplotlib-contour`, `crates/matplotlib-qhull`
- Phase 4: PDF/PS backends
