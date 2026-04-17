//! matplotlib-image — Rust replacement for the matplotlib._image C
//! extension.
//!
//! Provides `resample(input, output, transform, interpolation, ...)`
//! and the 17 interpolation-mode module constants used by
//! `matplotlib.image` and `matplotlib.colors`.
//!
//! Per the Phase 2 directive, the Python stub at
//! python/matplotlib/_image.py is deleted when the 3A build lands.

use pyo3::prelude::*;

mod resample;

use resample::resample_py;

#[pymodule]
fn _image(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resample_py, m)?)?;

    // Filter constants — values match the OG C extension verbatim.
    // Used as dict keys in python/matplotlib/image.py:35 and as the
    // `interpolation` arg to `resample()`.
    m.add("NEAREST", 0_i32)?;
    m.add("BILINEAR", 1_i32)?;
    m.add("BICUBIC", 2_i32)?;
    m.add("SPLINE16", 3_i32)?;
    m.add("SPLINE36", 4_i32)?;
    m.add("HANNING", 5_i32)?;
    m.add("HAMMING", 6_i32)?;
    m.add("HERMITE", 7_i32)?;
    m.add("KAISER", 8_i32)?;
    m.add("QUADRIC", 9_i32)?;
    m.add("CATROM", 10_i32)?;
    m.add("GAUSSIAN", 11_i32)?;
    m.add("BESSEL", 12_i32)?;
    m.add("MITCHELL", 13_i32)?;
    m.add("SINC", 14_i32)?;
    m.add("LANCZOS", 15_i32)?;
    m.add("BLACKMAN", 16_i32)?;

    Ok(())
}
