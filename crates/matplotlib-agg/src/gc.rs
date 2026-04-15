//! GraphicsContext translation: matplotlib GraphicsContextBase → tiny_skia.
//!
//! matplotlib's `GraphicsContextBase` exposes getters like `get_rgb()`,
//! `get_linewidth()`, `get_alpha()`, `get_clip_rectangle()`, etc. We read
//! the ones we need via Python attribute/method lookups on the gc object.
//!
//! Milestone 1A honors: rgb, linewidth, alpha, clip_rectangle. Caps,
//! joins, dashes, hatching, snapping, and clip_path are 1B.

use pyo3::prelude::*;
use tiny_skia::{Color, LineCap, LineJoin, Paint, Stroke};

/// Simplified view of a matplotlib GraphicsContext that matters for 1A.
#[derive(Clone, Debug)]
pub struct GcInfo {
    pub foreground: [f32; 4], // rgba, 0..=1
    pub alpha: f32,           // 0..=1
    pub linewidth: f32,       // in pixels (already converted from points)
    pub clip_rect: Option<[f32; 4]>, // (x, y, w, h) in display pixels
                              // 1B: cap, join, dashes, hatch
}

impl GcInfo {
    /// Read a GraphicsContext-like Python object. Fields that can't be
    /// extracted fall back to sensible defaults so that tests don't crash
    /// on unusual gc shapes.
    pub fn from_py(gc: &Bound<'_, PyAny>, dpi: f64) -> Self {
        let foreground = read_rgb(gc);
        let alpha = read_alpha(gc);
        let linewidth_pts = gc
            .call_method0("get_linewidth")
            .and_then(|v| v.extract::<f64>())
            .unwrap_or(1.0);
        let linewidth_px = (linewidth_pts * dpi / 72.0) as f32;

        let clip_rect = read_clip_rect(gc);

        Self {
            foreground,
            alpha: alpha as f32,
            linewidth: linewidth_px,
            clip_rect,
        }
    }

    /// Build a tiny_skia Paint for filling with the given RGBA color.
    pub fn make_fill_paint(&self, rgba: [f32; 4]) -> Paint<'static> {
        let mut p = Paint::default();
        let a = (rgba[3] * self.alpha).clamp(0.0, 1.0);
        let color = Color::from_rgba(
            rgba[0].clamp(0.0, 1.0),
            rgba[1].clamp(0.0, 1.0),
            rgba[2].clamp(0.0, 1.0),
            a,
        )
        .unwrap_or(Color::BLACK);
        p.set_color(color);
        p.anti_alias = true;
        p
    }

    /// Build a tiny_skia Paint for stroking with this gc's foreground.
    pub fn make_stroke_paint(&self) -> Paint<'static> {
        self.make_fill_paint(self.foreground)
    }

    pub fn make_stroke(&self) -> Stroke {
        Stroke {
            width: self.linewidth.max(0.0),
            miter_limit: 4.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            dash: None,
        }
    }
}

fn read_rgb(gc: &Bound<'_, PyAny>) -> [f32; 4] {
    // Try get_rgb() → 3-tuple or 4-tuple
    if let Ok(rgb_obj) = gc.call_method0("get_rgb") {
        if let Ok(tuple) = rgb_obj.extract::<(f64, f64, f64, f64)>() {
            return [
                tuple.0 as f32,
                tuple.1 as f32,
                tuple.2 as f32,
                tuple.3 as f32,
            ];
        }
        if let Ok(tuple) = rgb_obj.extract::<(f64, f64, f64)>() {
            return [tuple.0 as f32, tuple.1 as f32, tuple.2 as f32, 1.0];
        }
    }
    [0.0, 0.0, 0.0, 1.0]
}

fn read_alpha(gc: &Bound<'_, PyAny>) -> f64 {
    if let Ok(a) = gc.call_method0("get_alpha") {
        if let Ok(v) = a.extract::<f64>() {
            return v;
        }
    }
    1.0
}

fn read_clip_rect(gc: &Bound<'_, PyAny>) -> Option<[f32; 4]> {
    let rect_obj = gc.call_method0("get_clip_rectangle").ok()?;
    if rect_obj.is_none() {
        return None;
    }
    // matplotlib Bbox: (x0, y0) to (x1, y1), or bounds = (x, y, w, h)
    if let Ok(bounds_obj) = rect_obj.getattr("bounds") {
        if let Ok((x, y, w, h)) = bounds_obj.extract::<(f64, f64, f64, f64)>() {
            return Some([x as f32, y as f32, w as f32, h as f32]);
        }
    }
    None
}

/// Convert a Python "color or None" object to an RGBA array.
/// Used for `rgbFace` argument to draw_path.
pub fn extract_rgba_face(obj: &Bound<'_, PyAny>) -> Option<[f32; 4]> {
    if obj.is_none() {
        return None;
    }
    if let Ok((r, g, b, a)) = obj.extract::<(f64, f64, f64, f64)>() {
        return Some([r as f32, g as f32, b as f32, a as f32]);
    }
    if let Ok((r, g, b)) = obj.extract::<(f64, f64, f64)>() {
        return Some([r as f32, g as f32, b as f32, 1.0]);
    }
    None
}
