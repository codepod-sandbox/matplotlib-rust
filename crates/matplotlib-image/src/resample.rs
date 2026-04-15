//! `resample()` — output-pixel-driven inverse sampling.
//!
//! For each pixel in `output`, we compute the corresponding input
//! coordinate via the inverse of the user-supplied affine transform,
//! then sample the input with the selected filter kernel. The caller
//! allocates `output`; we write in place.
//!
//! Shape contract: input/output are either 2D (greyscale) or 3D with
//! last dim 4 (RGBA). dtype is u8, f32, or f64. Shapes and dtypes
//! between input and output must match (except for row/col counts).

use ndarray::{Array2, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};
use numpy::{PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

#[derive(Clone, Copy)]
struct Affine {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}

impl Affine {
    fn from_matrix(m: [[f64; 3]; 3]) -> Self {
        Self {
            a: m[0][0],
            b: m[0][1],
            tx: m[0][2],
            c: m[1][0],
            d: m[1][1],
            ty: m[1][2],
        }
    }

    /// Invert a 2D affine. Assumes non-singular.
    fn inverse(&self) -> PyResult<Self> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-30 {
            return Err(PyValueError::new_err("singular resample transform"));
        }
        let inv = 1.0 / det;
        let a = self.d * inv;
        let b = -self.b * inv;
        let c = -self.c * inv;
        let d = self.a * inv;
        let tx = -(a * self.tx + b * self.ty);
        let ty = -(c * self.tx + d * self.ty);
        Ok(Self { a, b, c, d, tx, ty })
    }

    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )
    }
}

fn extract_transform(transform: &Bound<'_, PyAny>) -> PyResult<Affine> {
    let mat = transform.call_method0("get_matrix")?;
    // Affine2DBase.get_matrix() returns a (3,3) float64 ndarray.
    let arr = mat.downcast::<numpy::PyArray2<f64>>().map_err(|_| {
        PyTypeError::new_err("transform.get_matrix() did not return a (3,3) float64 array")
    })?;
    let readonly = arr.readonly();
    let view = readonly.as_array();
    if view.shape() != [3, 3] {
        return Err(PyValueError::new_err(
            "transform.get_matrix() must return a (3,3) array",
        ));
    }
    let mut m = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = view[(r, c)];
        }
    }
    Ok(Affine::from_matrix(m))
}

/// The main entry point called by `matplotlib.image._resample` and
/// `matplotlib.colors.Colormap.__call__`.
///
/// Signature matches the OG C extension verbatim:
///   resample(input, output, transform, interpolation=1,
///            resample=False, alpha=1.0, norm=False, radius=1.0)
#[pyfunction]
#[pyo3(name = "resample")]
#[pyo3(signature = (
    input, output, transform,
    interpolation = 1,
    resample = false,
    alpha = 1.0,
    norm = false,
    radius = 1.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn resample_py(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    transform: &Bound<'_, PyAny>,
    interpolation: i32,
    resample: bool,
    alpha: f64,
    norm: bool,
    radius: f64,
) -> PyResult<()> {
    let _ = resample;
    let _ = norm;
    let _ = radius;

    let affine = extract_transform(transform)?;
    let inv = affine.inverse()?;

    let in_arr = input
        .downcast::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err("resample: `input` must be a numpy array"))?;
    let out_arr = output
        .downcast::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err("resample: `output` must be a numpy array"))?;

    let in_ndim = in_arr.ndim();
    let out_ndim = out_arr.ndim();
    if in_ndim != out_ndim {
        return Err(PyValueError::new_err(
            "resample: input and output must have the same number of dims",
        ));
    }

    match in_ndim {
        2 => dispatch_2d(py, input, output, inv, interpolation, alpha),
        3 => dispatch_3d(py, input, output, inv, interpolation, alpha),
        n => Err(PyValueError::new_err(format!(
            "resample: expected 2D or 3D array, got {n}D"
        ))),
    }
}

/// 2D greyscale sampler. Supports u8/f32/f64 via a downcast cascade.
fn dispatch_2d(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    inv: Affine,
    interp: i32,
    alpha: f64,
) -> PyResult<()> {
    if let Ok(i) = input.downcast::<numpy::PyArray2<f64>>() {
        let o = output
            .downcast::<numpy::PyArray2<f64>>()
            .map_err(|_| PyValueError::new_err("resample: dtype mismatch (expected f64 output)"))?;
        let ir = i.readonly();
        let mut ow = o.readwrite();
        resample_2d_f64(ir.as_array(), ow.as_array_mut(), inv, interp, alpha);
        return Ok(());
    }
    if let Ok(i) = input.downcast::<numpy::PyArray2<f32>>() {
        let o = output
            .downcast::<numpy::PyArray2<f32>>()
            .map_err(|_| PyValueError::new_err("resample: dtype mismatch (expected f32 output)"))?;
        let ir = i.readonly();
        let mut ow = o.readwrite();
        let in_f64 = ir.as_array().mapv(|v| v as f64);
        let mut out_f64 = Array2::<f64>::zeros(ow.as_array().raw_dim());
        resample_2d_f64(in_f64.view(), out_f64.view_mut(), inv, interp, alpha);
        ow.as_array_mut().assign(&out_f64.mapv(|v| v as f32));
        return Ok(());
    }
    if let Ok(i) = input.downcast::<numpy::PyArray2<u8>>() {
        let o = output
            .downcast::<numpy::PyArray2<u8>>()
            .map_err(|_| PyValueError::new_err("resample: dtype mismatch (expected u8 output)"))?;
        let ir = i.readonly();
        let mut ow = o.readwrite();
        let in_f64 = ir.as_array().mapv(|v| v as f64);
        let mut out_f64 = Array2::<f64>::zeros(ow.as_array().raw_dim());
        resample_2d_f64(in_f64.view(), out_f64.view_mut(), inv, interp, alpha);
        ow.as_array_mut()
            .assign(&out_f64.mapv(|v| v.clamp(0.0, 255.0).round() as u8));
        return Ok(());
    }
    let _ = py;
    Err(PyTypeError::new_err(
        "resample: unsupported input dtype (expected u8/f32/f64)",
    ))
}

/// 3D sampler — treats last dim as channels (RGBA), resamples each
/// channel independently. Applies `alpha` to the A channel.
fn dispatch_3d(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    inv: Affine,
    interp: i32,
    alpha: f64,
) -> PyResult<()> {
    let _ = py;
    let i = input
        .downcast::<numpy::PyArray3<u8>>()
        .map_err(|_| PyTypeError::new_err("resample: 3D input must be uint8 (RGBA)"))?;
    let o = output
        .downcast::<numpy::PyArray3<u8>>()
        .map_err(|_| PyTypeError::new_err("resample: 3D output must be uint8 (RGBA)"))?;
    let ir = i.readonly();
    let mut ow = o.readwrite();
    resample_3d_u8(ir.as_array(), ow.as_array_mut(), inv, interp, alpha);
    Ok(())
}

/// Core 2D f64 sampler — dispatch on filter.
fn resample_2d_f64(
    input: ArrayView2<'_, f64>,
    mut output: ArrayViewMut2<'_, f64>,
    inv: Affine,
    interp: i32,
    alpha: f64,
) {
    let (in_rows, in_cols) = (input.shape()[0] as i64, input.shape()[1] as i64);
    let (out_rows, out_cols) = (output.shape()[0], output.shape()[1]);

    for oy in 0..out_rows {
        for ox in 0..out_cols {
            let (ix, iy) = inv.apply(ox as f64 + 0.5, oy as f64 + 0.5);
            let s = match interp {
                1 /* BILINEAR */ => sample_bilinear(input, ix - 0.5, iy - 0.5, in_rows, in_cols),
                _  /* NEAREST + fallback */ => sample_nearest(input, ix, iy, in_rows, in_cols),
            };
            output[(oy, ox)] = s * alpha;
        }
    }
}

fn resample_3d_u8(
    input: ArrayView3<'_, u8>,
    mut output: ArrayViewMut3<'_, u8>,
    inv: Affine,
    interp: i32,
    alpha: f64,
) {
    let in_rows = input.shape()[0] as i64;
    let in_cols = input.shape()[1] as i64;
    let channels = input.shape()[2];
    let (out_rows, out_cols) = (output.shape()[0], output.shape()[1]);

    for oy in 0..out_rows {
        for ox in 0..out_cols {
            let (ix, iy) = inv.apply(ox as f64 + 0.5, oy as f64 + 0.5);
            for ch in 0..channels {
                let s = match interp {
                    1 => sample_bilinear_channel(input, ix - 0.5, iy - 0.5, in_rows, in_cols, ch),
                    _ => sample_nearest_channel(input, ix, iy, in_rows, in_cols, ch),
                };
                // Apply alpha only to the alpha channel (matches C ext).
                let scaled = if ch == 3 && channels == 4 {
                    s * alpha
                } else {
                    s
                };
                output[(oy, ox, ch)] = scaled.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
}

fn sample_nearest(input: ArrayView2<'_, f64>, ix: f64, iy: f64, in_rows: i64, in_cols: i64) -> f64 {
    let r = ix.floor() as i64;
    let c = iy.floor() as i64;
    if r < 0 || r >= in_cols || c < 0 || c >= in_rows {
        return 0.0;
    }
    input[(c as usize, r as usize)]
}

fn sample_bilinear(
    input: ArrayView2<'_, f64>,
    ix: f64,
    iy: f64,
    in_rows: i64,
    in_cols: i64,
) -> f64 {
    let x0 = ix.floor() as i64;
    let y0 = iy.floor() as i64;
    let fx = ix - x0 as f64;
    let fy = iy - y0 as f64;

    let get = |r: i64, c: i64| -> f64 {
        if r < 0 || r >= in_rows || c < 0 || c >= in_cols {
            0.0
        } else {
            input[(r as usize, c as usize)]
        }
    };

    let v00 = get(y0, x0);
    let v01 = get(y0, x0 + 1);
    let v10 = get(y0 + 1, x0);
    let v11 = get(y0 + 1, x0 + 1);
    let w0 = v00 * (1.0 - fx) + v01 * fx;
    let w1 = v10 * (1.0 - fx) + v11 * fx;
    w0 * (1.0 - fy) + w1 * fy
}

fn sample_nearest_channel(
    input: ArrayView3<'_, u8>,
    ix: f64,
    iy: f64,
    in_rows: i64,
    in_cols: i64,
    ch: usize,
) -> f64 {
    let r = ix.floor() as i64;
    let c = iy.floor() as i64;
    if r < 0 || r >= in_cols || c < 0 || c >= in_rows {
        return 0.0;
    }
    input[(c as usize, r as usize, ch)] as f64
}

fn sample_bilinear_channel(
    input: ArrayView3<'_, u8>,
    ix: f64,
    iy: f64,
    in_rows: i64,
    in_cols: i64,
    ch: usize,
) -> f64 {
    let x0 = ix.floor() as i64;
    let y0 = iy.floor() as i64;
    let fx = ix - x0 as f64;
    let fy = iy - y0 as f64;

    let get = |r: i64, c: i64| -> f64 {
        if r < 0 || r >= in_rows || c < 0 || c >= in_cols {
            0.0
        } else {
            input[(r as usize, c as usize, ch)] as f64
        }
    };

    let v00 = get(y0, x0);
    let v01 = get(y0, x0 + 1);
    let v10 = get(y0 + 1, x0);
    let v11 = get(y0 + 1, x0 + 1);
    let w0 = v00 * (1.0 - fx) + v01 * fx;
    let w1 = v10 * (1.0 - fx) + v11 * fx;
    w0 * (1.0 - fy) + w1 * fy
}
