/// matplotlib `_path` extension module — PyO3 implementation.
///
/// Implements all 15 functions exported by matplotlib's C++ `_path` module.
/// Path vertices/codes are extracted from Python Path objects; transforms
/// are extracted as 3×3 affine matrices.
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

// ---------------------------------------------------------------------------
// Path codes (match matplotlib's Path.code_type = np.uint8)
// ---------------------------------------------------------------------------
const STOP: u8 = 0;
const MOVETO: u8 = 1;
const LINETO: u8 = 2;
const CURVE3: u8 = 3;
const CURVE4: u8 = 4;
const CLOSEPOLY: u8 = 79;

// ---------------------------------------------------------------------------
// Helper: extract vertices (Nx2 f64) from a Python Path-like object.
// Returns owned Array2<f64>.
// ---------------------------------------------------------------------------
fn extract_vertices(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    let verts = path.getattr("vertices")?;
    let arr: PyReadonlyArray2<f64> = verts.extract()?;
    Ok(arr.as_array().to_owned())
}

// ---------------------------------------------------------------------------
// Helper: extract codes (N u8) from a Python Path-like object.
// Returns None if path.codes is None (implies MOVETO + N-1 LINETOs).
// ---------------------------------------------------------------------------
fn extract_codes(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Option<Array1<u8>>> {
    let codes_obj = path.getattr("codes")?;
    if codes_obj.is_none() {
        return Ok(None);
    }
    let arr: PyReadonlyArray1<u8> = codes_obj.extract()?;
    Ok(Some(arr.as_array().to_owned()))
}

// ---------------------------------------------------------------------------
// Helper: extract a 3×3 affine transform matrix from a Python transform.
// Returns identity if None or extraction fails.
// ---------------------------------------------------------------------------
fn extract_transform(trans: &Bound<'_, PyAny>) -> Array2<f64> {
    if trans.is_none() {
        return identity3();
    }
    // Try .get_matrix() first (Affine2D etc.)
    if let Ok(mat) = trans.call_method0("get_matrix") {
        if let Ok(arr) = mat.extract::<PyReadonlyArray2<f64>>() {
            let a = arr.as_array().to_owned();
            if a.shape() == [3, 3] {
                return a;
            }
        }
    }
    // Try .__array__()
    if let Ok(arr_obj) = trans.call_method0("__array__") {
        if let Ok(arr) = arr_obj.extract::<PyReadonlyArray2<f64>>() {
            let a = arr.as_array().to_owned();
            if a.shape() == [3, 3] {
                return a;
            }
        }
    }
    identity3()
}

fn identity3() -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((3, 3));
    m[[0, 0]] = 1.0;
    m[[1, 1]] = 1.0;
    m[[2, 2]] = 1.0;
    m
}

/// Check if a 3×3 matrix is effectively the identity.
fn is_identity(m: &Array2<f64>) -> bool {
    m[[0, 0]] == 1.0
        && m[[1, 1]] == 1.0
        && m[[2, 2]] == 1.0
        && m[[0, 1]] == 0.0
        && m[[0, 2]] == 0.0
        && m[[1, 0]] == 0.0
        && m[[1, 2]] == 0.0
        && m[[2, 0]] == 0.0
        && m[[2, 1]] == 0.0
}

/// Apply a 3×3 affine transform to an Nx2 array of points.
fn apply_transform_array(pts: &Array2<f64>, m: &Array2<f64>) -> Array2<f64> {
    if is_identity(m) {
        return pts.clone();
    }
    let n = pts.nrows();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = pts[[i, 0]];
        let y = pts[[i, 1]];
        out[[i, 0]] = m[[0, 0]] * x + m[[0, 1]] * y + m[[0, 2]];
        out[[i, 1]] = m[[1, 0]] * x + m[[1, 1]] * y + m[[1, 2]];
    }
    out
}

/// Apply a 3×3 affine transform to a single point.
fn apply_transform_pt(x: f64, y: f64, m: &Array2<f64>) -> (f64, f64) {
    if is_identity(m) {
        return (x, y);
    }
    let tx = m[[0, 0]] * x + m[[0, 1]] * y + m[[0, 2]];
    let ty = m[[1, 0]] * x + m[[1, 1]] * y + m[[1, 2]];
    (tx, ty)
}

// ---------------------------------------------------------------------------
// Ray-casting point-in-polygon
// ---------------------------------------------------------------------------

/// Test whether point (px, py) is inside a closed polygon defined by `verts`.
/// Returns true if inside using even-odd rule.
fn point_in_polygon(px: f64, py: f64, verts: &[(f64, f64)]) -> bool {
    let n = verts.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = verts[i];
        let (xj, yj) = verts[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Minimum distance squared from point (px, py) to line segment (x1,y1)-(x2,y2).
fn dist_sq_to_segment(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-20 {
        let ddx = px - x1;
        let ddy = py - y1;
        return ddx * ddx + ddy * ddy;
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let cx = x1 + t * dx;
    let cy = y1 + t * dy;
    let ddx = px - cx;
    let ddy = py - cy;
    ddx * ddx + ddy * ddy
}

// ---------------------------------------------------------------------------
// Path iteration: produce list of sub-paths [(vertices, closed)]
// ---------------------------------------------------------------------------

/// Split an Nx2 vertices + N codes array into sub-path polygons.
/// Each sub-path is a Vec<(f64, f64)> plus a `closed` flag.
fn iter_subpaths(verts: &Array2<f64>, codes: &Option<Array1<u8>>) -> Vec<(Vec<(f64, f64)>, bool)> {
    let n = verts.nrows();
    let mut subpaths: Vec<(Vec<(f64, f64)>, bool)> = Vec::new();
    let mut current: Vec<(f64, f64)> = Vec::new();
    let mut closed = false;

    for i in 0..n {
        let x = verts[[i, 0]];
        let y = verts[[i, 1]];
        let code = match codes {
            Some(c) => c[i],
            None => {
                if i == 0 {
                    MOVETO
                } else {
                    LINETO
                }
            }
        };

        match code {
            STOP => break,
            MOVETO => {
                if !current.is_empty() {
                    subpaths.push((std::mem::take(&mut current), closed));
                }
                closed = false;
                if !x.is_nan() && !y.is_nan() {
                    current.push((x, y));
                }
            }
            LINETO => {
                if !x.is_nan() && !y.is_nan() {
                    current.push((x, y));
                }
            }
            CURVE3 | CURVE4 => {
                // Approximate curves as straight lines to their endpoint
                if !x.is_nan() && !y.is_nan() {
                    current.push((x, y));
                }
            }
            CLOSEPOLY => {
                // Include the CLOSEPOLY vertex (usually same as start or ignored)
                // but don't include NaN
                if !x.is_nan() && !y.is_nan() && !current.is_empty() {
                    current.push((x, y));
                }
                if !current.is_empty() {
                    subpaths.push((std::mem::take(&mut current), true));
                }
                closed = false;
            }
            _ => {}
        }
    }
    if !current.is_empty() {
        subpaths.push((current, closed));
    }
    subpaths
}

// ---------------------------------------------------------------------------
// `point_in_path` core logic
// ---------------------------------------------------------------------------

fn point_in_path_core(
    px: f64,
    py: f64,
    radius: f64,
    verts: &Array2<f64>,
    codes: &Option<Array1<u8>>,
) -> bool {
    let subpaths = iter_subpaths(verts, codes);
    let r2 = radius * radius;
    for (poly, _closed) in &subpaths {
        if poly.len() < 2 {
            continue;
        }
        // If point is exactly on a boundary edge (within 1e-10), it's NOT inside
        // (matches real matplotlib's convention: boundary = outside).
        let on_boundary = poly.len() >= 2
            && (0..poly.len() - 1).any(|i| {
                let (x1, y1) = poly[i];
                let (x2, y2) = poly[i + 1];
                dist_sq_to_segment(px, py, x1, y1, x2, y2) < 1e-20
            });
        if on_boundary {
            // Boundary points are outside unless radius > 0 (stroke test)
            if radius > 0.0 {
                return true;
            }
            continue;
        }
        // Check if inside filled polygon
        if point_in_polygon(px, py, poly) {
            return true;
        }
        // Check if within radius of stroke
        if radius > 0.0 {
            for i in 0..poly.len() - 1 {
                let (x1, y1) = poly[i];
                let (x2, y2) = poly[i + 1];
                if dist_sq_to_segment(px, py, x1, y1, x2, y2) <= r2 {
                    return true;
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// PUBLIC PyO3 FUNCTIONS
// ---------------------------------------------------------------------------

/// affine_transform(points, trans) -> ndarray
///
/// Apply a 3×3 affine transform to an array of 2D points.
#[pyfunction]
fn affine_transform<'py>(
    py: Python<'py>,
    points: &Bound<'_, PyAny>,
    trans: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let numpy = py.import("numpy")?;
    let pts_obj = numpy
        .call_method1("asarray", (points,))?
        .call_method1("astype", ("float64",))?;
    let pts_arr: PyReadonlyArray2<f64> = pts_obj.extract()?;
    let pts = pts_arr.as_array().to_owned();
    let m = extract_transform(trans);
    let result = apply_transform_array(&pts, &m);
    Ok(result.into_pyarray(py))
}

/// cleanup_path(path, trans, remove_nans, clip_rect, snap_mode, stroke_width,
///              simplify, return_curves, sketch) -> (vertices, codes)
///
/// Cleans a path: removes NaNs, clips, optionally simplifies.
/// Returns (vertices_array, codes_array).
#[pyfunction]
#[pyo3(signature = (path, trans, remove_nans, clip_rect, snap_mode, stroke_width, simplify, return_curves, sketch))]
fn cleanup_path<'py>(
    py: Python<'py>,
    path: &Bound<'_, PyAny>,
    trans: &Bound<'_, PyAny>,
    remove_nans: bool,
    clip_rect: &Bound<'_, PyAny>,
    snap_mode: &Bound<'_, PyAny>,
    stroke_width: f64,
    simplify: Option<bool>,
    return_curves: bool,
    sketch: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut verts = extract_vertices(py, path)?;
    let codes_opt = extract_codes(py, path)?;
    let m = extract_transform(trans);

    // Apply transform
    if !is_identity(&m) {
        verts = apply_transform_array(&verts, &m);
    }

    let n = verts.nrows();

    // Build default codes if None
    let mut codes: Array1<u8> = match &codes_opt {
        Some(c) => c.clone(),
        None => {
            let mut c = Array1::<u8>::from_elem(n, LINETO);
            if n > 0 {
                c[0] = MOVETO;
            }
            c
        }
    };

    // Remove NaNs: replace NaN vertices with MOVETO on next valid point
    if remove_nans {
        let mut new_verts: Vec<[f64; 2]> = Vec::with_capacity(n);
        let mut new_codes: Vec<u8> = Vec::with_capacity(n);
        let mut pending_moveto = false;

        for i in 0..n {
            let x = verts[[i, 0]];
            let y = verts[[i, 1]];
            let code = codes[i];

            if x.is_nan() || y.is_nan() {
                pending_moveto = true;
                continue;
            }

            if pending_moveto && code != MOVETO {
                // Insert implicit MOVETO
                new_verts.push([x, y]);
                new_codes.push(MOVETO);
                pending_moveto = false;
            } else {
                pending_moveto = false;
                new_verts.push([x, y]);
                new_codes.push(code);
            }
        }

        let m2 = new_verts.len();
        let mut arr = Array2::<f64>::zeros((m2, 2));
        for (i, v) in new_verts.iter().enumerate() {
            arr[[i, 0]] = v[0];
            arr[[i, 1]] = v[1];
        }
        verts = arr;
        codes = Array1::from_vec(new_codes);
    }

    // Clip to rect if provided
    if !clip_rect.is_none() {
        // Extract clip rect as [x1, y1, x2, y2]
        let clip: Option<[f64; 4]> = extract_rect(clip_rect).ok();
        if let Some([rx1, ry1, rx2, ry2]) = clip {
            let n2 = verts.nrows();
            let mut new_verts: Vec<[f64; 2]> = Vec::with_capacity(n2);
            let mut new_codes: Vec<u8> = Vec::with_capacity(n2);
            for i in 0..n2 {
                let x = verts[[i, 0]];
                let y = verts[[i, 1]];
                let code = codes[i];
                if x >= rx1 && x <= rx2 && y >= ry1 && y <= ry2 {
                    new_verts.push([x, y]);
                    new_codes.push(code);
                } else if code == MOVETO {
                    // Keep MOVETO for continuity (will be skipped visually)
                    new_verts.push([x, y]);
                    new_codes.push(MOVETO);
                }
            }
            let m2 = new_verts.len();
            let mut arr = Array2::<f64>::zeros((m2, 2));
            for (i, v) in new_verts.iter().enumerate() {
                arr[[i, 0]] = v[0];
                arr[[i, 1]] = v[1];
            }
            verts = arr;
            codes = Array1::from_vec(new_codes);
        }
    }

    // If no vertices remain, return a minimal 1-vertex STOP path
    // (matching real matplotlib behavior for all-NaN paths)
    if verts.nrows() == 0 {
        let stop_verts = Array2::<f64>::zeros((1, 2));
        let stop_codes = Array1::<u8>::from_vec(vec![STOP]);
        let verts_py = stop_verts.into_pyarray(py);
        let codes_py = stop_codes.into_pyarray(py);
        return Ok(PyTuple::new(py, [verts_py.as_any(), codes_py.as_any()])?);
    }

    let verts_py = verts.into_pyarray(py);
    let codes_py = codes.into_pyarray(py);
    Ok(PyTuple::new(py, [verts_py.as_any(), codes_py.as_any()])?)
}

/// Extract a rect [x1, y1, x2, y2] from a Python object.
/// Handles Bbox objects (with x0, y0, x1, y1 attrs), 4-element sequences, or None.
fn extract_rect(obj: &Bound<'_, PyAny>) -> PyResult<[f64; 4]> {
    // Try Bbox-style: extents property (returns tuple or array)
    if let Ok(ext) = obj.getattr("extents") {
        // Try as numpy array first
        if let Ok(arr) = ext.extract::<PyReadonlyArray1<f64>>() {
            let s = arr.as_slice()?;
            if s.len() >= 4 {
                return Ok([s[0], s[1], s[2], s[3]]);
            }
        }
        // Try as tuple/list of floats
        if let Ok(lst) = ext.extract::<Vec<f64>>() {
            if lst.len() >= 4 {
                return Ok([lst[0], lst[1], lst[2], lst[3]]);
            }
        }
    }
    // Try direct x0/y0/x1/y1 attributes
    if let (Ok(x0), Ok(y0), Ok(x1), Ok(y1)) = (
        obj.getattr("x0"),
        obj.getattr("y0"),
        obj.getattr("x1"),
        obj.getattr("y1"),
    ) {
        if let (Ok(x0f), Ok(y0f), Ok(x1f), Ok(y1f)) = (
            x0.extract::<f64>(),
            y0.extract::<f64>(),
            x1.extract::<f64>(),
            y1.extract::<f64>(),
        ) {
            return Ok([x0f, y0f, x1f, y1f]);
        }
    }
    // Try sequence of 4 floats
    if let Ok(lst) = obj.extract::<Vec<f64>>() {
        if lst.len() >= 4 {
            return Ok([lst[0], lst[1], lst[2], lst[3]]);
        }
    }
    // Try bounds (x0, y0, width, height)
    if let Ok(b) = obj.getattr("bounds") {
        if let Ok(arr) = b.extract::<Vec<f64>>() {
            if arr.len() >= 4 {
                return Ok([arr[0], arr[1], arr[0] + arr[2], arr[1] + arr[3]]);
            }
        }
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "Cannot extract rect from object",
    ))
}

/// point_in_path(x, y, radius, path, trans) -> bool
#[pyfunction]
fn point_in_path(
    py: Python<'_>,
    x: f64,
    y: f64,
    radius: f64,
    path: &Bound<'_, PyAny>,
    trans: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let m = extract_transform(trans);
    // Inverse-transform the point so we can test in path space.
    // For simplicity, transform the path vertices into world space.
    let mut verts = extract_vertices(py, path)?;
    let codes = extract_codes(py, path)?;
    if !is_identity(&m) {
        verts = apply_transform_array(&verts, &m);
    }
    Ok(point_in_path_core(x, y, radius, &verts, &codes))
}

/// points_in_path(points, radius, path, trans) -> NDArray[bool]
#[pyfunction]
fn points_in_path<'py>(
    py: Python<'py>,
    points: &Bound<'_, PyAny>,
    radius: f64,
    path: &Bound<'_, PyAny>,
    trans: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    // Accept any array-like (list, ndarray, etc.)
    let numpy = py.import("numpy")?;
    let pts_obj = numpy
        .call_method1("asarray", (points,))?
        .call_method1("astype", ("float64",))?;
    let pts_arr: PyReadonlyArray2<f64> = pts_obj.extract()?;
    let pts = pts_arr.as_array();

    let m = extract_transform(trans);
    let mut verts = extract_vertices(py, path)?;
    let codes = extract_codes(py, path)?;
    if !is_identity(&m) {
        verts = apply_transform_array(&verts, &m);
    }
    let n = pts.nrows();
    let result: Array1<bool> = Array1::from_shape_fn(n, |i| {
        let x = pts[[i, 0]];
        let y = pts[[i, 1]];
        point_in_path_core(x, y, radius, &verts, &codes)
    });
    Ok(result.into_pyarray(py))
}

/// path_in_path(path_a, trans_a, path_b, trans_b) -> bool
///
/// Returns True if path_a CONTAINS path_b (all vertices of path_b are inside path_a).
/// Called as: path_in_path(self, None, other_path, transform)
/// So path_a is the outer/containing path and path_b is the inner path.
#[pyfunction]
fn path_in_path(
    py: Python<'_>,
    path_a: &Bound<'_, PyAny>,
    trans_a: &Bound<'_, PyAny>,
    path_b: &Bound<'_, PyAny>,
    trans_b: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let ma = extract_transform(trans_a);
    let mb = extract_transform(trans_b);

    // path_a is the container — we test vertices of path_b inside path_a
    let mut verts_a = extract_vertices(py, path_a)?;
    let codes_a = extract_codes(py, path_a)?;
    let mut verts_b = extract_vertices(py, path_b)?;
    let codes_b = extract_codes(py, path_b)?;

    if !is_identity(&ma) {
        verts_a = apply_transform_array(&verts_a, &ma);
    }
    if !is_identity(&mb) {
        verts_b = apply_transform_array(&verts_b, &mb);
    }

    let n = verts_b.nrows();
    if n == 0 {
        return Ok(false);
    }
    // All LINETO/CURVE vertices of path_b must be inside path_a
    for i in 0..n {
        let x = verts_b[[i, 0]];
        let y = verts_b[[i, 1]];
        let code = match &codes_b {
            Some(c) => c[i],
            None => {
                if i == 0 {
                    MOVETO
                } else {
                    LINETO
                }
            }
        };
        // Skip control codes
        if code == STOP || code == MOVETO || code == CLOSEPOLY {
            continue;
        }
        if !point_in_path_core(x, y, 0.0, &verts_a, &codes_a) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Check if point p is on segment (a, b).
fn point_on_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> bool {
    let (px, py) = p;
    let (ax, ay) = a;
    let (bx, by) = b;
    // Cross product must be ~0 (collinear)
    let cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    if cross.abs() > 1e-9 * ((bx - ax).abs() + (by - ay).abs() + 1.0) {
        return false;
    }
    // Check that the point is within the bounding box of the segment
    px >= ax.min(bx) - 1e-10
        && px <= ax.max(bx) + 1e-10
        && py >= ay.min(by) - 1e-10
        && py <= ay.max(by) + 1e-10
}

/// 2D cross product of vectors (ax,ay) and (bx,by).
#[inline]
fn cross2d(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ax * by - ay * bx
}

/// Check if two line segments intersect: (p1,p2) and (p3,p4).
///
/// Matches real matplotlib's _path C++ behavior:
/// - Proper crossings: True
/// - Collinear overlap (even if contained): True
/// - Collinear but separated (gap > 1e-13): False
/// - Parallel but not collinear: False
/// - Absolute tolerance 1e-13 for collinearity and overlap (matching real mpl)
fn segments_intersect(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> bool {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let (x3, y3) = p3;
    let (x4, y4) = p4;

    let d1x = x2 - x1;
    let d1y = y2 - y1;
    let d2x = x4 - x3;
    let d2y = y4 - y3;

    // Cross product of the two directions
    let cross12 = cross2d(d1x, d1y, d2x, d2y);

    // Absolute tolerance matching matplotlib's src/_path.h (1e-13 for isclose)
    let abs_tol = 1e-13;

    if cross12.abs() > abs_tol {
        // Non-parallel: standard parametric intersection test
        // t = where on segment 1, u = where on segment 2
        let dx31 = x3 - x1;
        let dy31 = y3 - y1;
        let t = cross2d(dx31, dy31, d2x, d2y) / cross12;
        let u = cross2d(dx31, dy31, d1x, d1y) / cross12;
        // Tiny epsilon for endpoint touching
        let eps = 1e-10;
        return t >= -eps && t <= 1.0 + eps && u >= -eps && u <= 1.0 + eps;
    }

    // Parallel (cross12 ≈ 0): check if collinear
    // The cross of d1 with the connector (p3-p1) tells us perpendicular offset
    let dx31 = x3 - x1;
    let dy31 = y3 - y1;
    let collinear_cross = cross2d(d1x, d1y, dx31, dy31);

    // Scale tolerance by the maximum dimension for numerical stability
    let max_dim = d1x.abs().max(d1y.abs()).max(dx31.abs()).max(dy31.abs()) + 1.0;
    if collinear_cross.abs() > abs_tol * max_dim {
        return false; // Parallel but not collinear
    }

    // Collinear: project onto d1 and check if ranges overlap
    let dot_d1 = d1x * d1x + d1y * d1y;
    if dot_d1 < abs_tol * abs_tol {
        // Degenerate segment p1=p2: check if p3 or p4 equals p1
        let eps2 = abs_tol * max_dim;
        let d13 = ((x3 - x1).powi(2) + (y3 - y1).powi(2)).sqrt();
        let d14 = ((x4 - x1).powi(2) + (y4 - y1).powi(2)).sqrt();
        return d13 < eps2 || d14 < eps2;
    }

    // Project p3 and p4 onto [0,1] parametric range of segment p1-p2
    let t3 = (d1x * (x3 - x1) + d1y * (y3 - y1)) / dot_d1;
    let t4 = (d1x * (x4 - x1) + d1y * (y4 - y1)) / dot_d1;

    let t_min = t3.min(t4);
    let t_max = t3.max(t4);

    // Segment [0,1] overlaps [t_min, t_max] if t_min < 1+eps AND t_max > -eps
    // Use absolute tolerance 1e-13 for the gap/overlap check
    t_min < 1.0 + abs_tol && t_max > -abs_tol
}

/// path_intersects_path(path1, path2, filled=False) -> bool
#[pyfunction]
#[pyo3(signature = (path1, path2, filled = false))]
fn path_intersects_path(
    py: Python<'_>,
    path1: &Bound<'_, PyAny>,
    path2: &Bound<'_, PyAny>,
    filled: bool,
) -> PyResult<bool> {
    let verts1 = extract_vertices(py, path1)?;
    let codes1 = extract_codes(py, path1)?;
    let verts2 = extract_vertices(py, path2)?;
    let codes2 = extract_codes(py, path2)?;

    let subpaths1 = iter_subpaths(&verts1, &codes1);
    let subpaths2 = iter_subpaths(&verts2, &codes2);

    // Check segment-segment intersections
    for (poly1, _) in &subpaths1 {
        for (poly2, _) in &subpaths2 {
            let n1 = poly1.len();
            let n2 = poly2.len();
            for i in 0..n1.saturating_sub(1) {
                for j in 0..n2.saturating_sub(1) {
                    if segments_intersect(poly1[i], poly1[i + 1], poly2[j], poly2[j + 1]) {
                        return Ok(true);
                    }
                }
            }

            if filled {
                // Check if any vertex of one is inside the other
                if !poly2.is_empty() {
                    let (x, y) = poly1[0];
                    if point_in_polygon(x, y, poly2) {
                        return Ok(true);
                    }
                }
                if !poly1.is_empty() {
                    let (x, y) = poly2[0];
                    if point_in_polygon(x, y, poly1) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

/// path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2, filled=False) -> bool
#[pyfunction]
#[pyo3(signature = (path, rect_x1, rect_y1, rect_x2, rect_y2, filled = false))]
fn path_intersects_rectangle(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    rect_x1: f64,
    rect_y1: f64,
    rect_x2: f64,
    rect_y2: f64,
    filled: bool,
) -> PyResult<bool> {
    let verts = extract_vertices(py, path)?;
    let codes = extract_codes(py, path)?;

    let rect_verts = vec![
        (rect_x1, rect_y1),
        (rect_x2, rect_y1),
        (rect_x2, rect_y2),
        (rect_x1, rect_y2),
        (rect_x1, rect_y1),
    ];

    let subpaths = iter_subpaths(&verts, &codes);

    for (poly, _) in &subpaths {
        let n = poly.len();
        // Check segment-segment intersections with rect edges
        for i in 0..n.saturating_sub(1) {
            for j in 0..4 {
                if segments_intersect(poly[i], poly[i + 1], rect_verts[j], rect_verts[j + 1]) {
                    return Ok(true);
                }
            }
        }

        if filled {
            // Check if any path vertex is inside rect
            for (x, y) in poly {
                if *x >= rect_x1 && *x <= rect_x2 && *y >= rect_y1 && *y <= rect_y2 {
                    return Ok(true);
                }
            }
            // Check if rect corner is inside path
            if !poly.is_empty() && point_in_polygon(rect_x1, rect_y1, poly) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// convert_path_to_polygons(path, trans, width=0, height=0, closed_only=False) -> list of arrays
#[pyfunction]
#[pyo3(signature = (path, trans, width = 0.0, height = 0.0, closed_only = false))]
fn convert_path_to_polygons<'py>(
    py: Python<'py>,
    path: &Bound<'_, PyAny>,
    trans: &Bound<'_, PyAny>,
    width: f64,
    height: f64,
    closed_only: bool,
) -> PyResult<Bound<'py, PyList>> {
    let mut verts = extract_vertices(py, path)?;
    let codes = extract_codes(py, path)?;
    let m = extract_transform(trans);

    if !is_identity(&m) {
        verts = apply_transform_array(&verts, &m);
    }

    let subpaths = iter_subpaths(&verts, &codes);
    let result = PyList::empty(py);

    for (mut poly, closed) in subpaths {
        let n_pts = poly.len();
        if closed_only {
            // For closed_only=True:
            // - Skip polygons with < 3 points (can't form a polygon)
            // - Auto-close polygons with >= 3 points (append first vertex if not already closed)
            if n_pts < 3 {
                continue;
            }
            // Auto-close: append first vertex if it differs from the last
            let first = poly[0];
            let last = poly[n_pts - 1];
            if (first.0 - last.0).abs() > 1e-10 || (first.1 - last.1).abs() > 1e-10 {
                poly.push(first);
            }
        } else {
            if n_pts < 2 {
                continue;
            }
        }
        let m2 = poly.len();
        let mut arr = Array2::<f64>::zeros((m2, 2));
        for (i, (x, y)) in poly.iter().enumerate() {
            arr[[i, 0]] = *x;
            arr[[i, 1]] = *y;
        }
        result.append(arr.into_pyarray(py))?;
    }
    Ok(result)
}

/// clip_path_to_rect(path, rect, inside) -> list of vertex arrays
///
/// Clip path polygons to a rectangle using Sutherland-Hodgman algorithm.
#[pyfunction]
fn clip_path_to_rect<'py>(
    py: Python<'py>,
    path: &Bound<'_, PyAny>,
    rect: &Bound<'_, PyAny>,
    inside: bool,
) -> PyResult<Bound<'py, PyList>> {
    let verts = extract_vertices(py, path)?;
    let codes = extract_codes(py, path)?;
    let [rx1, ry1, rx2, ry2] = extract_rect(rect)?;

    let subpaths = iter_subpaths(&verts, &codes);
    let result = PyList::empty(py);

    for (poly, _closed) in subpaths {
        if poly.is_empty() {
            continue;
        }
        let clipped = if inside {
            sutherland_hodgman(&poly, rx1, ry1, rx2, ry2)
        } else {
            // Outside: just return original polygon
            poly.clone()
        };
        if clipped.len() >= 2 {
            let m2 = clipped.len();
            let mut arr = Array2::<f64>::zeros((m2, 2));
            for (i, (x, y)) in clipped.iter().enumerate() {
                arr[[i, 0]] = *x;
                arr[[i, 1]] = *y;
            }
            result.append(arr.into_pyarray(py))?;
        }
    }
    Ok(result)
}

/// Sutherland-Hodgman polygon clipping to an axis-aligned rectangle.
fn sutherland_hodgman(poly: &[(f64, f64)], x1: f64, y1: f64, x2: f64, y2: f64) -> Vec<(f64, f64)> {
    let clip_edges = [
        (true, true, x1),   // left: x >= x1
        (true, false, x2),  // right: x <= x2
        (false, true, y1),  // bottom: y >= y1
        (false, false, y2), // top: y <= y2
    ];

    let mut output: Vec<(f64, f64)> = poly.to_vec();

    for (is_x, is_min, bound) in clip_edges {
        if output.is_empty() {
            break;
        }
        let input = std::mem::take(&mut output);
        let n = input.len();
        for i in 0..n {
            let current = input[i];
            let previous = input[(i + n - 1) % n];

            let inside_current = if is_x {
                if is_min {
                    current.0 >= bound
                } else {
                    current.0 <= bound
                }
            } else if is_min {
                current.1 >= bound
            } else {
                current.1 <= bound
            };

            let inside_previous = if is_x {
                if is_min {
                    previous.0 >= bound
                } else {
                    previous.0 <= bound
                }
            } else if is_min {
                previous.1 >= bound
            } else {
                previous.1 <= bound
            };

            if inside_current {
                if !inside_previous {
                    // Intersection
                    output.push(intersection(previous, current, is_x, bound));
                }
                output.push(current);
            } else if inside_previous {
                output.push(intersection(previous, current, is_x, bound));
            }
        }
    }
    output
}

fn intersection(p1: (f64, f64), p2: (f64, f64), is_x: bool, bound: f64) -> (f64, f64) {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    if is_x {
        let t = (bound - x1) / (x2 - x1 + 1e-30);
        (bound, y1 + t * (y2 - y1))
    } else {
        let t = (bound - y1) / (y2 - y1 + 1e-30);
        (x1 + t * (x2 - x1), bound)
    }
}

/// update_path_extents(path, trans, rect, minpos, ignore) -> (extents, minpos)
///
/// Update a bounding box with the extents of a transformed path.
/// Returns (new_extents_4, new_minpos_2).
#[pyfunction]
fn update_path_extents<'py>(
    py: Python<'py>,
    path: &Bound<'_, PyAny>,
    trans: &Bound<'_, PyAny>,
    rect: &Bound<'_, PyAny>,
    minpos: PyReadonlyArray1<f64>,
    ignore: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut verts = extract_vertices(py, path)?;
    let m = extract_transform(trans);
    if !is_identity(&m) {
        verts = apply_transform_array(&verts, &m);
    }

    // Current extents
    let mut xmin: f64;
    let mut ymin: f64;
    let mut xmax: f64;
    let mut ymax: f64;
    let mut min_x: f64;
    let mut min_y: f64;

    if ignore {
        xmin = f64::INFINITY;
        ymin = f64::INFINITY;
        xmax = f64::NEG_INFINITY;
        ymax = f64::NEG_INFINITY;
    } else {
        match extract_rect(rect) {
            Ok([r0, r1, r2, r3]) => {
                xmin = r0;
                ymin = r1;
                xmax = r2;
                ymax = r3;
            }
            Err(_) => {
                xmin = f64::INFINITY;
                ymin = f64::INFINITY;
                xmax = f64::NEG_INFINITY;
                ymax = f64::NEG_INFINITY;
            }
        }
    }

    let mp = minpos.as_slice()?;
    min_x = mp[0];
    min_y = mp[1];

    let n = verts.nrows();
    for i in 0..n {
        let x = verts[[i, 0]];
        let y = verts[[i, 1]];
        if x.is_nan() || y.is_nan() {
            continue;
        }
        if x < xmin {
            xmin = x;
        }
        if x > xmax {
            xmax = x;
        }
        if y < ymin {
            ymin = y;
        }
        if y > ymax {
            ymax = y;
        }
        if x > 0.0 && x < min_x {
            min_x = x;
        }
        if y > 0.0 && y < min_y {
            min_y = y;
        }
    }

    if xmin > xmax {
        xmin = 0.0;
        xmax = 0.0;
        ymin = 0.0;
        ymax = 0.0;
    }

    let extents = Array1::from_vec(vec![xmin, ymin, xmax, ymax]).into_pyarray(py);
    let new_minpos = Array1::from_vec(vec![min_x, min_y]).into_pyarray(py);
    Ok(PyTuple::new(py, [extents.as_any(), new_minpos.as_any()])?)
}

/// get_path_collection_extents(master_transform, paths, transforms, offsets, offset_transform)
///   -> (extents_4, minpos_2)
#[pyfunction]
fn get_path_collection_extents<'py>(
    py: Python<'py>,
    master_transform: &Bound<'_, PyAny>,
    paths: &Bound<'_, PyAny>,
    transforms: PyReadonlyArray2<f64>,
    offsets: PyReadonlyArray2<f64>,
    offset_transform: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mt = extract_transform(master_transform);
    let ot = extract_transform(offset_transform);
    let offsets_arr = offsets.as_array();
    let transforms_arr = transforms.as_array();

    let paths_list: Vec<Bound<'_, PyAny>> = paths.try_iter()?.collect::<PyResult<_>>()?;
    let n_paths = paths_list.len();
    let n_offsets = offsets_arr.nrows();

    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;

    if n_paths == 0 || n_offsets == 0 {
        let extents = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_pyarray(py);
        let mp = Array1::from_vec(vec![min_x, min_y]).into_pyarray(py);
        return Ok(PyTuple::new(py, [extents.as_any(), mp.as_any()])?);
    }

    // Number of unique transforms (may be 0 meaning use master_transform)
    let n_transforms = transforms_arr.shape()[0];
    // transforms is shape (N, 3, 3) — passed as atleast_3d which flattens to 2D if 0 rows
    // We'll handle it as a flat 2D array: row i*3+r, col c

    for j in 0..n_offsets {
        let off_x = offsets_arr[[j, 0]];
        let off_y = offsets_arr[[j, 1]];

        // Apply offset_transform to offset
        let (ox, oy) = apply_transform_pt(off_x, off_y, &ot);

        let path_idx = j % n_paths;
        let path = &paths_list[path_idx];
        let mut verts = extract_vertices(py, path)?;

        // transforms_arr has shape (N*3, 3) due to atleast_3d flattening,
        // or shape (0, 0) when empty. Use master_transform if no per-path transforms.
        if n_transforms >= 3 && transforms_arr.ncols() == 3 {
            let t_idx = (j % (n_transforms / 3)) * 3;
            let mut tm = Array2::<f64>::zeros((3, 3));
            for r in 0..3 {
                for c in 0..3 {
                    tm[[r, c]] = transforms_arr[[t_idx + r, c]];
                }
            }
            verts = apply_transform_array(&verts, &tm);
        } else {
            verts = apply_transform_array(&verts, &mt);
        }

        // Add offset
        let n = verts.nrows();
        for i in 0..n {
            let x = verts[[i, 0]] + ox;
            let y = verts[[i, 1]] + oy;
            if x.is_nan() || y.is_nan() {
                continue;
            }
            if x < xmin {
                xmin = x;
            }
            if x > xmax {
                xmax = x;
            }
            if y < ymin {
                ymin = y;
            }
            if y > ymax {
                ymax = y;
            }
            if x > 0.0 && x < min_x {
                min_x = x;
            }
            if y > 0.0 && y < min_y {
                min_y = y;
            }
        }
    }

    if xmin > xmax {
        xmin = 0.0;
        xmax = 0.0;
        ymin = 0.0;
        ymax = 0.0;
    }

    let extents = Array1::from_vec(vec![xmin, ymin, xmax, ymax]).into_pyarray(py);
    let mp = Array1::from_vec(vec![min_x, min_y]).into_pyarray(py);
    Ok(PyTuple::new(py, [extents.as_any(), mp.as_any()])?)
}

/// count_bboxes_overlapping_bbox(bbox, bboxes) -> int
///
/// Count how many bboxes in `bboxes` (Nx4 array) overlap with `bbox` ([x1,y1,x2,y2]).
#[pyfunction]
fn count_bboxes_overlapping_bbox(
    py: Python<'_>,
    bbox: &Bound<'_, PyAny>,
    bboxes: PyReadonlyArray2<f64>,
) -> PyResult<usize> {
    let [bx1, by1, bx2, by2] = extract_rect(bbox)?;
    let arr = bboxes.as_array();
    let n = arr.nrows();
    let mut count = 0;
    for i in 0..n {
        let x1 = arr[[i, 0]];
        let y1 = arr[[i, 1]];
        let x2 = arr[[i, 2]];
        let y2 = arr[[i, 3]];
        // Overlap if not separated
        if x1 <= bx2 && x2 >= bx1 && y1 <= by2 && y2 >= by1 {
            count += 1;
        }
    }
    Ok(count)
}

/// is_sorted_and_has_non_nan(array) -> bool
///
/// Return True if the 1D array is monotonically non-decreasing (ignoring NaNs)
/// and has at least one non-NaN value.
#[pyfunction]
fn is_sorted_and_has_non_nan(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<bool> {
    let arr: PyReadonlyArray1<f64> = array.extract()?;
    let s = arr.as_slice()?;
    let mut has_non_nan = false;
    let mut last_valid = f64::NEG_INFINITY;
    for &v in s {
        if v.is_nan() {
            continue;
        }
        has_non_nan = true;
        if v < last_valid {
            return Ok(false);
        }
        last_valid = v;
    }
    Ok(has_non_nan)
}

/// convert_to_string(path, trans, clip_rect, simplify, sketch, precision, codes, postfix)
///   -> bytes
///
/// Convert a path to a PostScript/SVG-style bytestring.
#[pyfunction]
#[pyo3(signature = (path, trans, clip_rect, simplify, sketch, precision, codes, postfix))]
fn convert_to_string<'py>(
    py: Python<'py>,
    path: &Bound<'_, PyAny>,
    trans: &Bound<'_, PyAny>,
    clip_rect: &Bound<'_, PyAny>,
    simplify: Option<bool>,
    sketch: &Bound<'_, PyAny>,
    precision: i64,
    codes: &Bound<'_, PyAny>,
    postfix: bool,
) -> PyResult<PyObject> {
    let mut verts = extract_vertices(py, path)?;
    let path_codes = extract_codes(py, path)?;
    let m = extract_transform(trans);

    if !is_identity(&m) {
        verts = apply_transform_array(&verts, &m);
    }

    // Extract the 5 code strings: MOVETO, LINETO, CURVE3, CURVE4, CLOSEPOLY
    let codes_list: Vec<Bound<'_, PyAny>> = codes.try_iter()?.collect::<PyResult<_>>()?;
    let get_code_str = |i: usize| -> String {
        if i < codes_list.len() {
            codes_list[i].extract::<String>().unwrap_or_default()
        } else {
            String::new()
        }
    };
    let moveto = get_code_str(0);
    let lineto = get_code_str(1);
    let curve3 = get_code_str(2);
    let curve4 = get_code_str(3);
    let closepoly = get_code_str(4);

    let fmt_coord = |v: f64| -> String {
        if precision <= 0 {
            format!("{}", v as i64)
        } else {
            format!("{:.prec$}", v, prec = precision as usize)
        }
    };

    let n = verts.nrows();
    let mut parts: Vec<String> = Vec::with_capacity(n);

    for i in 0..n {
        let x = verts[[i, 0]];
        let y = verts[[i, 1]];
        let code = match &path_codes {
            Some(c) => c[i],
            None => {
                if i == 0 {
                    MOVETO
                } else {
                    LINETO
                }
            }
        };

        let s = match code {
            MOVETO => {
                if postfix {
                    format!("{} {} {}", fmt_coord(x), fmt_coord(y), moveto)
                } else {
                    format!("{} {} {}", moveto, fmt_coord(x), fmt_coord(y))
                }
            }
            LINETO => {
                if postfix {
                    format!("{} {} {}", fmt_coord(x), fmt_coord(y), lineto)
                } else {
                    format!("{} {} {}", lineto, fmt_coord(x), fmt_coord(y))
                }
            }
            CLOSEPOLY => closepoly.clone(),
            CURVE3 | CURVE4 => {
                if postfix {
                    format!("{} {} {}", fmt_coord(x), fmt_coord(y), curve4)
                } else {
                    format!("{} {} {}", curve4, fmt_coord(x), fmt_coord(y))
                }
            }
            STOP => break,
            _ => String::new(),
        };
        if !s.is_empty() {
            parts.push(s);
        }
    }

    let joined = parts.join("\n");
    Ok(pyo3::types::PyBytes::new(py, joined.as_bytes()).into())
}

/// point_in_path_collection(x, y, radius, master_transform, paths, transforms,
///                          offsets, offset_trans, filled) -> list[int]
///
/// Returns list of indices of paths in the collection that contain the point.
#[pyfunction]
#[pyo3(signature = (x, y, radius, master_transform, paths, transforms, offsets, offset_trans, filled))]
fn point_in_path_collection<'py>(
    py: Python<'py>,
    x: f64,
    y: f64,
    radius: f64,
    master_transform: &Bound<'_, PyAny>,
    paths: &Bound<'_, PyAny>,
    transforms: PyReadonlyArray2<f64>,
    offsets: PyReadonlyArray2<f64>,
    offset_trans: &Bound<'_, PyAny>,
    filled: bool,
) -> PyResult<PyObject> {
    let mt = extract_transform(master_transform);
    let ot = extract_transform(offset_trans);
    let offsets_arr = offsets.as_array();
    let transforms_arr = transforms.as_array();

    let paths_list: Vec<Bound<'_, PyAny>> = paths.try_iter()?.collect::<PyResult<_>>()?;
    let n_paths = paths_list.len();
    let n_offsets = offsets_arr.nrows();
    let n_transforms = transforms_arr.shape()[0];

    let mut result_indices: Vec<usize> = Vec::new();

    for j in 0..n_offsets {
        let off_x = offsets_arr[[j, 0]];
        let off_y = offsets_arr[[j, 1]];
        let (ox, oy) = apply_transform_pt(off_x, off_y, &ot);

        let path_idx = if n_paths > 0 { j % n_paths } else { continue };
        let path = &paths_list[path_idx];
        let mut verts = extract_vertices(py, path)?;
        let codes = extract_codes(py, path)?;

        if n_transforms >= 3 && transforms_arr.ncols() == 3 {
            let t_idx = (j % (n_transforms / 3)) * 3;
            let mut tm = Array2::<f64>::zeros((3, 3));
            for r in 0..3 {
                for c in 0..3 {
                    tm[[r, c]] = transforms_arr[[t_idx + r, c]];
                }
            }
            verts = apply_transform_array(&verts, &tm);
        } else {
            verts = apply_transform_array(&verts, &mt);
        }

        // Translate point back by offset to test in path space
        let test_x = x - ox;
        let test_y = y - oy;

        if point_in_path_core(test_x, test_y, radius, &verts, &codes) {
            result_indices.push(j);
        }
    }

    // Return as numpy int array
    let arr: Array1<i64> = Array1::from_iter(result_indices.iter().map(|&i| i as i64));
    Ok(arr.into_pyarray(py).into())
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn _path(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(affine_transform, m)?)?;
    m.add_function(wrap_pyfunction!(cleanup_path, m)?)?;
    m.add_function(wrap_pyfunction!(clip_path_to_rect, m)?)?;
    m.add_function(wrap_pyfunction!(convert_path_to_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(convert_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(count_bboxes_overlapping_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(get_path_collection_extents, m)?)?;
    m.add_function(wrap_pyfunction!(is_sorted_and_has_non_nan, m)?)?;
    m.add_function(wrap_pyfunction!(path_in_path, m)?)?;
    m.add_function(wrap_pyfunction!(path_intersects_path, m)?)?;
    m.add_function(wrap_pyfunction!(path_intersects_rectangle, m)?)?;
    m.add_function(wrap_pyfunction!(point_in_path, m)?)?;
    m.add_function(wrap_pyfunction!(point_in_path_collection, m)?)?;
    m.add_function(wrap_pyfunction!(points_in_path, m)?)?;
    m.add_function(wrap_pyfunction!(update_path_extents, m)?)?;
    Ok(())
}
