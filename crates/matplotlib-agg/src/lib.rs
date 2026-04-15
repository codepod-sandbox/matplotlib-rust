//! matplotlib-agg — Rust implementation of _backend_agg (Phase 1).
//!
//! Replaces the Python no-op stub at
//! `python/matplotlib/backends/_backend_agg.py`. Provides the `RendererAgg`
//! class the OG `backend_agg.py` wrapper expects.
//!
//! The Rust class owns a `tiny_skia::Pixmap` as the draw surface and a
//! cached unpremultiplied rgba buffer exposed via the Python buffer protocol.
//!
//! Text glyph rasterization is NOT handled here; that's Phase 2 (ft2font).
//! This crate's `draw_text_image` blits whatever bitmap the passed object
//! provides. See `docs/superpowers/specs/2026-04-15-phase1-matplotlib-agg-design.md`.

use pyo3::prelude::*;

mod renderer;

use renderer::RendererAgg;

/// `get_hinting_flag()` — module-level function called by
/// `matplotlib.backends.backend_agg.get_hinting_flag` indirectly.
/// The value is passed to `FT2Font.set_text(flags=...)` which is a no-op
/// in Phase 1.
#[pyfunction]
fn get_hinting_flag() -> i32 {
    0
}

/// Python module entry point. The file is installed as
/// `python/matplotlib/backends/_backend_agg.so`, so the module name must be
/// `_backend_agg` for CPython's import machinery to find the init symbol.
#[pymodule]
fn _backend_agg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RendererAgg>()?;
    m.add_function(wrap_pyfunction!(get_hinting_flag, m)?)?;
    Ok(())
}
