//! Path translation: matplotlib.path.Path → tiny_skia::Path.
//!
//! matplotlib Path objects have:
//! - `vertices`: ndarray of shape (N, 2), float64
//! - `codes`: Optional ndarray of shape (N,), uint8, with values:
//!     STOP=0, MOVETO=1, LINETO=2, CURVE3=3, CURVE4=4, CLOSEPOLY=79
//!
//! When `codes` is None, all vertices are treated as an implicit
//! MOVETO (first) + LINETO (rest). This matches matplotlib.path.Path's
//! documented behavior.
//!
//! The affine transform is a matplotlib.transforms.Affine2DBase (or
//! similar) with a `get_matrix()` method returning a (3, 3) ndarray.

use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use tiny_skia::{Path, PathBuilder};

// matplotlib.path.Path code constants.
const STOP: u8 = 0;
const MOVETO: u8 = 1;
const LINETO: u8 = 2;
const CURVE3: u8 = 3;
const CURVE4: u8 = 4;
const CLOSEPOLY: u8 = 79;

/// A 3×3 affine transform stored row-major. Only the top 2 rows are
/// read; the third row is assumed to be [0, 0, 1].
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    pub a: f64, // m[0,0]
    pub b: f64, // m[0,1]
    pub c: f64, // m[0,2] (tx)
    pub d: f64, // m[1,0]
    pub e: f64, // m[1,1]
    pub f: f64, // m[1,2] (ty)
}

impl Affine {
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
        }
    }

    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.c,
            self.d * x + self.e * y + self.f,
        )
    }
}

/// Compose two 3×3 affines: result = left ∘ right. Equivalent to
/// applying `right` first then `left`. Both matrices have implicit
/// bottom row [0, 0, 1].
pub fn compose_affines(left: Affine, right: Affine) -> Affine {
    Affine {
        a: left.a * right.a + left.b * right.d,
        b: left.a * right.b + left.b * right.e,
        c: left.a * right.c + left.b * right.f + left.c,
        d: left.d * right.a + left.e * right.d,
        e: left.d * right.b + left.e * right.e,
        f: left.d * right.c + left.e * right.f + left.f,
    }
}

/// Read an affine transform from a Python object. Expects the object
/// to have a `get_matrix()` method returning a (3, 3) ndarray.
/// Falls back to identity if extraction fails.
pub fn extract_affine(obj: &Bound<'_, PyAny>) -> Affine {
    let Ok(mtx_obj) = obj.call_method0("get_matrix") else {
        return Affine::identity();
    };
    let Ok(arr) = mtx_obj.extract::<PyReadonlyArray2<f64>>() else {
        return Affine::identity();
    };
    let v = arr.as_array();
    if v.shape() != [3, 3] {
        return Affine::identity();
    }
    Affine {
        a: v[[0, 0]],
        b: v[[0, 1]],
        c: v[[0, 2]],
        d: v[[1, 0]],
        e: v[[1, 1]],
        f: v[[1, 2]],
    }
}

/// Read a matplotlib Path's vertices and codes arrays.
pub fn extract_path_verts_codes<'py>(
    path: &Bound<'py, PyAny>,
) -> PyResult<(
    PyReadonlyArray2<'py, f64>,
    Option<PyReadonlyArray1<'py, u8>>,
)> {
    let verts_obj = path.getattr("vertices")?;
    // Coerce to float64 contiguous array.
    let np = path.py().import("numpy")?;
    let verts_obj = np
        .call_method1("ascontiguousarray", (verts_obj,))?
        .call_method1("astype", ("float64",))?;
    let verts: PyReadonlyArray2<f64> = verts_obj.extract()?;

    let codes_obj = path.getattr("codes")?;
    let codes: Option<PyReadonlyArray1<u8>> = if codes_obj.is_none() {
        None
    } else {
        let coerced = np
            .call_method1("ascontiguousarray", (codes_obj,))?
            .call_method1("astype", ("uint8",))?;
        Some(coerced.extract()?)
    };

    Ok((verts, codes))
}

/// Translate a matplotlib Path to a tiny_skia::Path, applying the given
/// affine transform and a final y-flip (matplotlib origin is bottom-left,
/// tiny-skia / PNG origin is top-left).
///
/// When `snap_mode` is `Some(true)` / `Some(false)`, snapping is forced on /
/// off. When it is `None`, upstream-style auto snapping is applied: paths
/// made only of horizontal / vertical line segments are snapped, with the
/// exact snap offset depending on the rounded stroke width parity.
///
/// Returns `None` if the path has no drawable segments after translation.
pub fn path_to_tiny_skia(
    path: &Bound<'_, PyAny>,
    transform: Affine,
    canvas_height: f64,
    snap_mode: Option<bool>,
    stroke_width: f32,
) -> PyResult<Option<Path>> {
    let (verts_arr, codes_arr) = extract_path_verts_codes(path)?;
    let verts = verts_arr.as_array();
    let n = verts.nrows();
    if n == 0 {
        return Ok(None);
    }

    let total_vertices = n;
    let auto_should_snap = || -> bool {
        if total_vertices > 1024 {
            return false;
        }
        if let Some(codes) = &codes_arr {
            let codes = codes.as_array();
            let mut last = None::<(f64, f64)>;
            for i in 0..n {
                match codes[i] {
                    STOP => break,
                    MOVETO => {
                        last = Some((verts[[i, 0]], verts[[i, 1]]));
                    }
                    LINETO => {
                        let Some((x0, y0)) = last else {
                            last = Some((verts[[i, 0]], verts[[i, 1]]));
                            continue;
                        };
                        let x1 = verts[[i, 0]];
                        let y1 = verts[[i, 1]];
                        if (x0 - x1).abs() >= 1e-4 && (y0 - y1).abs() >= 1e-4 {
                            return false;
                        }
                        last = Some((x1, y1));
                    }
                    CURVE3 | CURVE4 => return false,
                    CLOSEPOLY => {}
                    _ => {}
                }
            }
            true
        } else {
            for i in 1..n {
                let x0 = verts[[i - 1, 0]];
                let y0 = verts[[i - 1, 1]];
                let x1 = verts[[i, 0]];
                let y1 = verts[[i, 1]];
                if (x0 - x1).abs() >= 1e-4 && (y0 - y1).abs() >= 1e-4 {
                    return false;
                }
            }
            true
        }
    };
    let snap_offset = match snap_mode {
        Some(true) => Some(if (stroke_width.round() as i32) % 2 != 0 {
            0.5
        } else {
            0.0
        }),
        Some(false) => None,
        None => {
            if auto_should_snap() {
                Some(if (stroke_width.round() as i32) % 2 != 0 {
                    0.5
                } else {
                    0.0
                })
            } else {
                None
            }
        }
    };

    let mut pb = PathBuilder::new();
    let mut pen_x: f64 = 0.0;
    let mut pen_y: f64 = 0.0;
    let mut has_move = false;

    let snap_coord = |v: f64| -> f64 {
        if let Some(offset) = snap_offset {
            (v + 0.5).floor() + offset
        } else {
            v
        }
    };

    let apply = |x: f64, y: f64| -> (f32, f32) {
        let (tx, ty) = transform.apply(x, y);
        // Y-flip: tiny-skia origin is top-left
        let pix_x = snap_coord(tx);
        let pix_y = snap_coord(canvas_height - ty);
        (pix_x as f32, pix_y as f32)
    };

    if let Some(codes) = codes_arr {
        let codes = codes.as_array();
        let mut i = 0usize;
        let mut pending_move = false;
        while i < n {
            let code = codes[i];
            match code {
                STOP => break,
                MOVETO => {
                    if verts[[i, 0]].is_nan() || verts[[i, 1]].is_nan() {
                        has_move = false;
                        pending_move = true;
                        i += 1;
                        continue;
                    }
                    let (x, y) = apply(verts[[i, 0]], verts[[i, 1]]);
                    pb.move_to(x, y);
                    pen_x = verts[[i, 0]];
                    pen_y = verts[[i, 1]];
                    has_move = true;
                    pending_move = false;
                    i += 1;
                }
                LINETO => {
                    if verts[[i, 0]].is_nan() || verts[[i, 1]].is_nan() {
                        has_move = false;
                        pending_move = true;
                        i += 1;
                        continue;
                    }
                    if !has_move || pending_move {
                        let (x, y) = apply(verts[[i, 0]], verts[[i, 1]]);
                        pb.move_to(x, y);
                        has_move = true;
                    } else {
                        let (x, y) = apply(verts[[i, 0]], verts[[i, 1]]);
                        pb.line_to(x, y);
                    }
                    pen_x = verts[[i, 0]];
                    pen_y = verts[[i, 1]];
                    pending_move = false;
                    i += 1;
                }
                CURVE3 => {
                    if i + 1 >= n {
                        break;
                    }
                    if verts[[i, 0]].is_nan()
                        || verts[[i, 1]].is_nan()
                        || verts[[i + 1, 0]].is_nan()
                        || verts[[i + 1, 1]].is_nan()
                    {
                        has_move = false;
                        pending_move = true;
                        i += 2;
                        continue;
                    }
                    if !has_move {
                        let (x0, y0) = apply(pen_x, pen_y);
                        pb.move_to(x0, y0);
                        has_move = true;
                    }
                    let (cx, cy) = apply(verts[[i, 0]], verts[[i, 1]]);
                    let (ex, ey) = apply(verts[[i + 1, 0]], verts[[i + 1, 1]]);
                    pb.quad_to(cx, cy, ex, ey);
                    pen_x = verts[[i + 1, 0]];
                    pen_y = verts[[i + 1, 1]];
                    i += 2;
                }
                CURVE4 => {
                    if i + 2 >= n {
                        break;
                    }
                    if verts[[i, 0]].is_nan()
                        || verts[[i, 1]].is_nan()
                        || verts[[i + 1, 0]].is_nan()
                        || verts[[i + 1, 1]].is_nan()
                        || verts[[i + 2, 0]].is_nan()
                        || verts[[i + 2, 1]].is_nan()
                    {
                        has_move = false;
                        pending_move = true;
                        i += 3;
                        continue;
                    }
                    if !has_move {
                        let (x0, y0) = apply(pen_x, pen_y);
                        pb.move_to(x0, y0);
                        has_move = true;
                    }
                    let (c1x, c1y) = apply(verts[[i, 0]], verts[[i, 1]]);
                    let (c2x, c2y) = apply(verts[[i + 1, 0]], verts[[i + 1, 1]]);
                    let (ex, ey) = apply(verts[[i + 2, 0]], verts[[i + 2, 1]]);
                    pb.cubic_to(c1x, c1y, c2x, c2y, ex, ey);
                    pen_x = verts[[i + 2, 0]];
                    pen_y = verts[[i + 2, 1]];
                    i += 3;
                }
                CLOSEPOLY => {
                    pb.close();
                    has_move = false;
                    pending_move = false;
                    i += 1;
                }
                _ => {
                    // Unknown code — skip
                    i += 1;
                }
            }
        }
    } else {
        // codes=None: implicit MOVETO(v[0]) + LINETO(v[1..])
        let mut has_move = false;
        let mut has_drawn = false;
        for i in 0..n {
            let vx = verts[[i, 0]];
            let vy = verts[[i, 1]];
            if vx.is_nan() || vy.is_nan() {
                has_move = false;
                continue;
            }
            let (x, y) = apply(vx, vy);
            if !has_move {
                pb.move_to(x, y);
                has_move = true;
                has_drawn = true;
            } else {
                pb.line_to(x, y);
                has_drawn = true;
            }
        }
        if !has_drawn {
            return Ok(None);
        }
    }

    Ok(pb.finish())
}
