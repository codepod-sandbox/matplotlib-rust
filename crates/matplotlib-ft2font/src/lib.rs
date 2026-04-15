//! matplotlib-ft2font — Rust replacement for the matplotlib.ft2font C
//! extension, backed by fontdue (rasterization) and ttf-parser (metadata
//! + outlines).
//!
//! Replaces the earlier Python stub at python/matplotlib/ft2font.py.
//! Per the "delete stubs once we have the real thing" directive, the
//! stub is removed when 2A lands.
//!
//! API surface is validated against every usage site in OG matplotlib
//! python code — see docs/superpowers/specs/2026-04-15-phase2-matplotlib-
//! ft2font-design.md.

use pyo3::prelude::*;

mod font;

use font::FT2Font;

/// Python module entry point. The built artifact is installed as
/// `python/matplotlib/ft2font.so` (or `.pyd`) and this function is
/// called by CPython's import machinery via the `PyInit_ft2font` symbol.
#[pymodule]
fn ft2font(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FT2Font>()?;
    m.add_class::<font::FT2Image>()?;
    m.add_class::<font::Glyph>()?;
    m.add_class::<font::Kerning>()?;
    m.add_class::<font::LoadFlags>()?;
    m.add_class::<font::FaceFlags>()?;
    m.add_class::<font::StyleFlags>()?;

    // Module-level constants that the current stub exposes.
    // Used by backend_agg.py and other OG code.
    m.add("__freetype_version__", "2.6.1")?;
    m.add("__freetype_build_type__", "fontdue")?;

    m.add("LOAD_DEFAULT", 0_i32)?;
    m.add("LOAD_NO_SCALE", 1_i32)?;
    m.add("LOAD_NO_HINTING", 2_i32)?;
    m.add("LOAD_RENDER", 4_i32)?;
    m.add("LOAD_NO_BITMAP", 8_i32)?;
    m.add("LOAD_VERTICAL_LAYOUT", 16_i32)?;
    m.add("LOAD_FORCE_AUTOHINT", 32_i32)?;
    m.add("LOAD_CROP_BITMAP", 64_i32)?;
    m.add("LOAD_PEDANTIC", 128_i32)?;
    m.add("LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH", 512_i32)?;
    m.add("LOAD_NO_RECURSE", 1024_i32)?;
    m.add("LOAD_IGNORE_TRANSFORM", 2048_i32)?;
    m.add("LOAD_MONOCHROME", 4096_i32)?;
    m.add("LOAD_LINEAR_DESIGN", 8192_i32)?;
    m.add("LOAD_NO_AUTOHINT", 32768_i32)?;

    m.add("KERNING_DEFAULT", 0_i32)?;
    m.add("KERNING_UNFITTED", 1_i32)?;
    m.add("KERNING_UNSCALED", 2_i32)?;

    m.add("FACE_FLAG_SCALABLE", 1_u32)?;
    m.add("FACE_FLAG_FIXED_SIZES", 2_u32)?;
    m.add("FACE_FLAG_FIXED_WIDTH", 4_u32)?;
    m.add("FACE_FLAG_SFNT", 8_u32)?;
    m.add("FACE_FLAG_HORIZONTAL", 16_u32)?;
    m.add("FACE_FLAG_VERTICAL", 32_u32)?;
    m.add("FACE_FLAG_KERNING", 64_u32)?;
    m.add("FACE_FLAG_FAST_GLYPHS", 128_u32)?;
    m.add("FACE_FLAG_MULTIPLE_MASTERS", 256_u32)?;
    m.add("FACE_FLAG_GLYPH_NAMES", 512_u32)?;
    m.add("FACE_FLAG_BOLD", 1_u32)?;
    m.add("FACE_FLAG_ITALIC", 2_u32)?;

    m.add("STYLE_FLAG_ITALIC", 1_u32)?;
    m.add("STYLE_FLAG_BOLD", 2_u32)?;

    Ok(())
}
