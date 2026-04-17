//! matplotlib-qhull — Rust replacement for the matplotlib._qhull C extension.
//!
//! Exposes a single function:
//!   `delaunay(x, y, verbose) -> (triangles, neighbors)`
//!
//! `triangles` is an (ntri, 3) int32 array of point indices (CCW order).
//! `neighbors` is an (ntri, 3) int32 array where `neighbors[t, j]` is the
//! index of the triangle across the edge from `triangles[t, j]` to
//! `triangles[t, (j+1)%3]`, or -1 if that edge is on the boundary.
//!
//! This matches the Matplotlib contract documented in
//! `tri/_triangulation.py:212-214` and relied on by `TriRefiner`.
//!
//! Implementation uses the `delaunator` crate.

use delaunator::{triangulate, Point, EMPTY};
use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// `delaunay(x, y, verbose=False) -> (triangles, neighbors)`
///
/// Compute the Delaunay triangulation of the 2-D point set `(x[i], y[i])`.
///
/// Returns a 2-tuple of (ntri, 3) int32 numpy arrays:
///   - `triangles[t, :]`  — CCW point indices of triangle t
///   - `neighbors[t, j]`  — index of the triangle across the edge from
///                          triangles[t,j] to triangles[t,(j+1)%3],
///                          or -1 if that edge is on the convex hull
///
/// `verbose` is accepted for API compatibility; it has no effect.
#[pyfunction]
#[pyo3(name = "delaunay")]
// verbose is passed as sys.flags.verbose (an int, not a bool) by the caller
// in _triangulation.py; accept i32 to handle both 0/1 int and True/False bool.
#[pyo3(signature = (x, y, verbose = 0))]
fn delaunay_py<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    verbose: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let _ = verbose;

    // Accept any array-like for x and y, convert to Vec<f64>.
    let xv: Vec<f64> = x
        .call_method1("__iter__", ())?
        .try_iter()?
        .map(|v| v?.extract::<f64>())
        .collect::<PyResult<_>>()?;
    let yv: Vec<f64> = y
        .call_method1("__iter__", ())?
        .try_iter()?
        .map(|v| v?.extract::<f64>())
        .collect::<PyResult<_>>()?;

    let n = xv.len();
    if yv.len() != n {
        return Err(PyValueError::new_err(
            "delaunay: x and y must have the same length",
        ));
    }
    if n < 3 {
        return Err(PyValueError::new_err("delaunay: need at least 3 points"));
    }

    let points: Vec<Point> = xv
        .iter()
        .zip(yv.iter())
        .map(|(&xi, &yi)| Point { x: xi, y: yi })
        .collect();

    let result = triangulate(&points);
    let ntri = result.triangles.len() / 3;

    let mut tri_data = Array2::<i32>::zeros((ntri, 3));
    let mut nbr_data = Array2::<i32>::zeros((ntri, 3));

    for t in 0..ntri {
        tri_data[(t, 0)] = result.triangles[3 * t] as i32;
        tri_data[(t, 1)] = result.triangles[3 * t + 1] as i32;
        tri_data[(t, 2)] = result.triangles[3 * t + 2] as i32;

        // Delaunator halfedge layout for triangle t:
        //   3t+0: v0→v1  (from triangles[3t+0] to triangles[3t+1])
        //   3t+1: v1→v2  (from triangles[3t+1] to triangles[3t+2])
        //   3t+2: v2→v0  (from triangles[3t+2] to triangles[3t+0])
        //
        // Matplotlib contract: neighbors[t, j] = triangle across the edge
        // from triangles[t, j] to triangles[t, (j+1)%3].
        //
        // That is exactly the edge represented by halfedge 3t+j, so
        // neighbors[t, j] = halfedges[3t+j] / 3  (or -1 at boundary).
        for j in 0..3 {
            let h = 3 * t + j;
            let twin = result.halfedges[h];
            nbr_data[(t, j)] = if twin == EMPTY { -1 } else { (twin / 3) as i32 };
        }
    }

    let tri_py = tri_data.into_pyarray(py);
    let nbr_py = nbr_data.into_pyarray(py);
    PyTuple::new(py, [tri_py.as_any(), nbr_py.as_any()])
}

#[pymodule]
fn _qhull(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delaunay_py, m)?)?;
    Ok(())
}
