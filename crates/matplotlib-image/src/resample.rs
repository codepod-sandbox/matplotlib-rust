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
//!
//! Alpha contract: `alpha` applies only to the alpha channel of 3D RGBA
//! inputs. For 2D scalar/mask inputs the caller does alpha composition
//! externally (see image.py:480–525); we do not touch `alpha` for 2D.

use ndarray::{Array2, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};
use numpy::{PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

// ── Affine helpers ────────────────────────────────────────────────────────────

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

    #[inline]
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )
    }
}

fn extract_transform(transform: &Bound<'_, PyAny>) -> PyResult<Affine> {
    let mat = transform.call_method0("get_matrix")?;
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

// ── Filter kernels ────────────────────────────────────────────────────────────

/// Modified Bessel function of the first kind I₀(x) — used by the Kaiser
/// window. Polynomial approximation (Abramowitz & Stegun 9.8.1, accuracy
/// ~1e-7 for all x ≥ 0).
fn bessel_i0(x: f64) -> f64 {
    if x < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + t2
            * (3.5156229
                + t2 * (3.0899424
                    + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    } else {
        let t = 3.75 / x;
        (x.exp() / x.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

/// Bessel function of the first kind J₁(x) — used by the Bessel/Jinc
/// filter. Polynomial approximation (A&S 9.4.3/9.4.6, accuracy ~1e-6).
fn bessel_j1(x: f64) -> f64 {
    if x.abs() <= 3.0 {
        let t = x / 3.0;
        let t2 = t * t;
        x * (0.5
            - t2 * (0.56249985
                + t2 * (0.21093573
                    - t2 * (0.03954289 + t2 * (0.00443319 - t2 * (0.00031761 + t2 * 0.00001109))))))
    } else {
        let ax = x.abs();
        let t = 3.0 / ax;
        let y = ax - 2.356194491;
        (2.0 / (std::f64::consts::PI * ax)).sqrt()
            * y.cos()
            * (0.99999999
                + t * (0.04687499 + t * (-0.02002690 + t * (0.00847222 + t * (-0.00239032)))))
    }
}

/// Mitchell–Netravali cubic filter for arbitrary B and C.
#[inline]
fn mitchell_bc(x: f64, b: f64, c: f64) -> f64 {
    let ax = x.abs();
    if ax < 1.0 {
        let ax2 = ax * ax;
        let ax3 = ax2 * ax;
        ((12.0 - 9.0 * b - 6.0 * c) * ax3 + (-18.0 + 12.0 * b + 6.0 * c) * ax2 + (6.0 - 2.0 * b))
            / 6.0
    } else if ax < 2.0 {
        let ax2 = ax * ax;
        let ax3 = ax2 * ax;
        ((-b - 6.0 * c) * ax3
            + (6.0 * b + 30.0 * c) * ax2
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Evaluate the 1-D kernel for the given filter constant at position `x`.
/// `radius` is the half-width for SINC/LANCZOS; other filters ignore it.
#[inline]
fn kernel_eval(interp: i32, x: f64, radius: f64) -> f64 {
    use std::f64::consts::PI;
    let ax = x.abs();
    match interp {
        // 2 = BICUBIC — Catmull-Rom (a = -0.5, i.e. Mitchell B=0, C=0.5)
        2 | 10 => {
            // CATROM is identical to BICUBIC
            if ax < 1.0 {
                let a = -0.5_f64;
                ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
            } else if ax < 2.0 {
                let a = -0.5_f64;
                (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a
            } else {
                0.0
            }
        }
        // 3 = SPLINE16 — piecewise cubic (Hermite spline)
        3 => {
            if ax < 1.0 {
                (2.0 * ax - 3.0) * ax * ax + 1.0
            } else if ax < 2.0 {
                let y = ax - 2.0;
                (-2.0 * y - 3.0) * y * y
            } else {
                0.0
            }
        }
        // 4 = SPLINE36 — 6-tap piecewise cubic
        4 => {
            if ax < 1.0 {
                ((13.0 / 11.0 * ax - 453.0 / 209.0) * ax - 3.0 / 209.0) * ax + 1.0
            } else if ax < 2.0 {
                ((-6.0 / 11.0 * (ax - 1.0) + 270.0 / 209.0) * (ax - 1.0) - 156.0 / 209.0)
                    * (ax - 1.0)
            } else if ax < 3.0 {
                ((1.0 / 11.0 * (ax - 2.0) - 45.0 / 209.0) * (ax - 2.0) + 26.0 / 209.0) * (ax - 2.0)
            } else {
                0.0
            }
        }
        // 5 = HANNING
        5 => {
            if ax >= 1.0 {
                0.0
            } else {
                0.5 + 0.5 * (PI * x).cos()
            }
        }
        // 6 = HAMMING
        6 => {
            if ax >= 1.0 {
                0.0
            } else {
                0.54 + 0.46 * (PI * x).cos()
            }
        }
        // 7 = HERMITE
        7 => {
            if ax >= 1.0 {
                0.0
            } else {
                (2.0 * ax - 3.0) * ax * ax + 1.0
            }
        }
        // 8 = KAISER — Kaiser-Bessel with alpha = 6.33
        8 => {
            if ax >= 1.0 {
                0.0
            } else {
                let alpha = 6.33_f64;
                let denom = bessel_i0(alpha);
                if denom == 0.0 {
                    0.0
                } else {
                    bessel_i0(alpha * (1.0 - ax * ax).sqrt()) / denom
                }
            }
        }
        // 9 = QUADRIC
        9 => {
            if ax < 0.5 {
                0.75 - ax * ax
            } else if ax < 1.5 {
                let y = ax - 1.5;
                0.5 * y * y
            } else {
                0.0
            }
        }
        // 11 = GAUSSIAN
        11 => {
            if ax >= 2.0 {
                0.0
            } else {
                (-2.0 * ax * ax).exp() * (2.0_f64 / std::f64::consts::PI).sqrt()
            }
        }
        // 12 = BESSEL (Jinc filter — J₁-based)
        12 => {
            const SUPPORT: f64 = 3.2383;
            if ax >= SUPPORT {
                0.0
            } else if ax < 1e-10 {
                1.0
            } else {
                // jinc(x) = J1(pi*x) / (pi*x/2)
                let px = PI * ax;
                bessel_j1(px) / (px / 2.0)
            }
        }
        // 13 = MITCHELL — B = 1/3, C = 1/3
        13 => mitchell_bc(x, 1.0 / 3.0, 1.0 / 3.0),
        // 14 = SINC — truncated sinc(pi*x), support = radius
        14 => {
            if ax >= radius {
                0.0
            } else if ax < 1e-10 {
                1.0
            } else {
                let px = PI * ax;
                px.sin() / px
            }
        }
        // 15 = LANCZOS — sinc(pi*x) * sinc(pi*x/a), support = radius
        15 => {
            if ax >= radius {
                0.0
            } else if ax < 1e-10 {
                1.0
            } else {
                let px = PI * ax;
                let win = px / radius;
                (px.sin() / px) * (win.sin() / win)
            }
        }
        // 16 = BLACKMAN — windowed-sinc with a Blackman window of half-width
        // `radius` (same category as SINC/LANCZOS; filterrad governs it per
        // image.py:858 / axes/_axes.py:5917).
        16 => {
            if ax >= radius {
                0.0
            } else if ax < 1e-10 {
                1.0
            } else {
                // Blackman-windowed sinc: sinc(pi*x) * W_blackman(x/radius)
                let px = PI * ax;
                let t = PI * ax / radius;
                let window = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
                (px.sin() / px) * window
            }
        }
        _ => 0.0,
    }
}

/// Integer ceiling of the filter half-support (the furthest tap we need
/// to evaluate in each axis).
fn filter_support(interp: i32, radius: f64) -> i64 {
    match interp {
        5 | 6 | 7 | 8 => 1,                   // HANNING/HAMMING/HERMITE/KAISER
        9 => 2,                               // QUADRIC: support 1.5 → ceil 2
        2 | 10 | 3 | 11 | 13 => 2,            // BICUBIC/CATROM/SPLINE16/GAUSSIAN/MITCHELL
        4 | 12 => 4,                          // SPLINE36/BESSEL: support 3+ → ceil 4
        14 | 15 | 16 => radius.ceil() as i64, // SINC/LANCZOS/BLACKMAN
        _ => 1,
    }
}

/// Windowed sample from a 2-D f64 view, normalized by the sum of weights.
/// `sx`, `sy` are the pixel-center coordinates in the source image.
fn sample_windowed(
    input: ArrayView2<'_, f64>,
    sx: f64,
    sy: f64,
    in_rows: i64,
    in_cols: i64,
    interp: i32,
    support: i64,
    radius: f64,
) -> f64 {
    let cx = sx.floor() as i64;
    let cy = sy.floor() as i64;
    let mut acc = 0.0_f64;
    let mut wsum = 0.0_f64;
    for ky in (cy - support + 1)..=(cy + support) {
        let wy = kernel_eval(interp, sy - ky as f64, radius);
        if wy == 0.0 {
            continue;
        }
        for kx in (cx - support + 1)..=(cx + support) {
            if kx < 0 || kx >= in_cols || ky < 0 || ky >= in_rows {
                continue;
            }
            let wx = kernel_eval(interp, sx - kx as f64, radius);
            if wx == 0.0 {
                continue;
            }
            let w = wx * wy;
            acc += input[(ky as usize, kx as usize)] * w;
            wsum += w;
        }
    }
    if wsum == 0.0 {
        0.0
    } else {
        acc / wsum
    }
}

/// Windowed sample from a 3-D u8 view for a single channel.
fn sample_windowed_channel(
    input: ArrayView3<'_, u8>,
    sx: f64,
    sy: f64,
    in_rows: i64,
    in_cols: i64,
    interp: i32,
    support: i64,
    radius: f64,
    ch: usize,
) -> f64 {
    let cx = sx.floor() as i64;
    let cy = sy.floor() as i64;
    let mut acc = 0.0_f64;
    let mut wsum = 0.0_f64;
    for ky in (cy - support + 1)..=(cy + support) {
        let wy = kernel_eval(interp, sy - ky as f64, radius);
        if wy == 0.0 {
            continue;
        }
        for kx in (cx - support + 1)..=(cx + support) {
            if kx < 0 || kx >= in_cols || ky < 0 || ky >= in_rows {
                continue;
            }
            let wx = kernel_eval(interp, sx - kx as f64, radius);
            if wx == 0.0 {
                continue;
            }
            let w = wx * wy;
            acc += input[(ky as usize, kx as usize, ch)] as f64 * w;
            wsum += w;
        }
    }
    if wsum == 0.0 {
        0.0
    } else {
        acc / wsum
    }
}

// ── Python entry point ────────────────────────────────────────────────────────

/// Signature matches the OG C extension verbatim:
///   resample(input, output, transform, interpolation=1,
///            resample=False, alpha=1.0, norm=False, radius=1.0)
///
/// `alpha` is applied only to the alpha channel of 3D RGBA outputs.
/// For 2D scalar inputs the Python caller handles alpha composition
/// outside this function (see image.py:480-525).
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
    let _ = norm; // always normalize windowed kernels at boundaries

    let affine = extract_transform(transform)?;
    let inv = affine.inverse()?;

    // resample=False: "only resample when the output image is larger than
    // the input image" (image.py:864 / axes/_axes.py:5921).  Detect
    // downsampling via the inverse-transform determinant: |det(inv)| > 1
    // means each output pixel maps to more than one source pixel.
    let effective_interp = if !resample {
        let det = (inv.a * inv.d - inv.b * inv.c).abs();
        if det > 1.0 {
            0 // NEAREST — do not filter when downsampling
        } else {
            interpolation
        }
    } else {
        interpolation
    };

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
        2 => dispatch_2d(py, input, output, inv, effective_interp, radius),
        3 => dispatch_3d(py, input, output, inv, effective_interp, alpha, radius),
        n => Err(PyValueError::new_err(format!(
            "resample: expected 2D or 3D array, got {n}D"
        ))),
    }
}

// ── Dispatch by dtype ─────────────────────────────────────────────────────────

/// 2D greyscale sampler. Does NOT apply `alpha` — 2D alpha composition
/// is done by the Python caller after this call returns.
fn dispatch_2d(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    inv: Affine,
    interp: i32,
    radius: f64,
) -> PyResult<()> {
    if let Ok(i) = input.downcast::<numpy::PyArray2<f64>>() {
        let o = output
            .downcast::<numpy::PyArray2<f64>>()
            .map_err(|_| PyValueError::new_err("resample: dtype mismatch (expected f64 output)"))?;
        let ir = i.readonly();
        let mut ow = o.readwrite();
        resample_2d_f64(ir.as_array(), ow.as_array_mut(), inv, interp, radius);
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
        resample_2d_f64(in_f64.view(), out_f64.view_mut(), inv, interp, radius);
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
        resample_2d_f64(in_f64.view(), out_f64.view_mut(), inv, interp, radius);
        ow.as_array_mut()
            .assign(&out_f64.mapv(|v| v.clamp(0.0, 255.0).round() as u8));
        return Ok(());
    }
    let _ = py;
    Err(PyTypeError::new_err(
        "resample: unsupported input dtype (expected u8/f32/f64)",
    ))
}

/// 3D RGBA sampler. Applies `alpha` to the alpha channel only.
fn dispatch_3d(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    inv: Affine,
    interp: i32,
    alpha: f64,
    radius: f64,
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
    resample_3d_u8(ir.as_array(), ow.as_array_mut(), inv, interp, alpha, radius);
    Ok(())
}

// ── Core samplers ─────────────────────────────────────────────────────────────

fn resample_2d_f64(
    input: ArrayView2<'_, f64>,
    mut output: ArrayViewMut2<'_, f64>,
    inv: Affine,
    interp: i32,
    radius: f64,
) {
    let (in_rows, in_cols) = (input.shape()[0] as i64, input.shape()[1] as i64);
    let (out_rows, out_cols) = (output.shape()[0], output.shape()[1]);
    let support = filter_support(interp, radius);

    for oy in 0..out_rows {
        for ox in 0..out_cols {
            // Map output pixel center to source coords; subtract 0.5 so
            // that pixel i in source has its center at integer i.
            let (ix, iy) = inv.apply(ox as f64 + 0.5, oy as f64 + 0.5);
            let sx = ix - 0.5;
            let sy = iy - 0.5;
            let s = match interp {
                0 => sample_nearest(input, ix, iy, in_rows, in_cols),
                1 => sample_bilinear(input, sx, sy, in_rows, in_cols),
                _ => sample_windowed(input, sx, sy, in_rows, in_cols, interp, support, radius),
            };
            // No alpha scaling for 2D — the Python image pipeline performs
            // alpha composition outside this function.
            output[(oy, ox)] = s;
        }
    }
}

fn resample_3d_u8(
    input: ArrayView3<'_, u8>,
    mut output: ArrayViewMut3<'_, u8>,
    inv: Affine,
    interp: i32,
    alpha: f64,
    radius: f64,
) {
    let in_rows = input.shape()[0] as i64;
    let in_cols = input.shape()[1] as i64;
    let channels = input.shape()[2];
    let (out_rows, out_cols) = (output.shape()[0], output.shape()[1]);
    let support = filter_support(interp, radius);

    for oy in 0..out_rows {
        for ox in 0..out_cols {
            let (ix, iy) = inv.apply(ox as f64 + 0.5, oy as f64 + 0.5);
            let sx = ix - 0.5;
            let sy = iy - 0.5;
            for ch in 0..channels {
                let s = match interp {
                    0 => sample_nearest_channel(input, ix, iy, in_rows, in_cols, ch),
                    1 => sample_bilinear_channel(input, sx, sy, in_rows, in_cols, ch),
                    _ => sample_windowed_channel(
                        input, sx, sy, in_rows, in_cols, interp, support, radius, ch,
                    ),
                };
                // Apply alpha only to the alpha channel (channel 3 in RGBA).
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

// ── Nearest / bilinear fast paths ─────────────────────────────────────────────

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
    sx: f64,
    sy: f64,
    in_rows: i64,
    in_cols: i64,
) -> f64 {
    let x0 = sx.floor() as i64;
    let y0 = sy.floor() as i64;
    let fx = sx - x0 as f64;
    let fy = sy - y0 as f64;

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
    sx: f64,
    sy: f64,
    in_rows: i64,
    in_cols: i64,
    ch: usize,
) -> f64 {
    let x0 = sx.floor() as i64;
    let y0 = sy.floor() as i64;
    let fx = sx - x0 as f64;
    let fy = sy - y0 as f64;

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
