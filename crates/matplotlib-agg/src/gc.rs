//! GraphicsContext translation: matplotlib GraphicsContextBase → tiny_skia.
//!
//! matplotlib's `GraphicsContextBase` exposes getters like `get_rgb()`,
//! `get_linewidth()`, `get_alpha()`, `get_clip_rectangle()`, etc. We read
//! the ones we need via Python attribute/method lookups on the gc object.
//!
//! Milestone 1A: rgb, linewidth, alpha, clip_rectangle.
//! Milestone 1B.3: capstyle, joinstyle, dashes. Grid lines, dashed
//! annotations, rounded markers all depend on these.

use pyo3::prelude::*;
use tiny_skia::{Color, LineCap, LineJoin, Paint, Stroke, StrokeDash};

/// Simplified view of a matplotlib GraphicsContext that matters for 1A/1B.
#[derive(Clone, Debug)]
pub struct GcInfo {
    pub foreground: [f32; 4],        // rgba, 0..=1
    pub alpha: f32,                  // 0..=1
    pub linewidth: f32,              // in pixels (already converted from points)
    pub clip_rect: Option<[f32; 4]>, // (x, y, w, h) in display pixels

    pub line_cap: LineCap,
    pub line_join: LineJoin,
    pub miter_limit: f32,
    /// (offset, on/off array) in pixels. Empty array or None dashes = solid.
    pub dashes: Option<(f32, Vec<f32>)>,
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
        let line_cap = read_cap_style(gc);
        let line_join = read_join_style(gc);
        let dashes = read_dashes(gc, dpi);

        Self {
            foreground,
            alpha: alpha as f32,
            linewidth: linewidth_px,
            clip_rect,
            line_cap,
            line_join,
            miter_limit: 10.0,
            dashes,
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

    /// Build a tiny_skia Stroke honoring linewidth, caps, joins, and
    /// the gc's dash pattern (converted from points to pixels).
    pub fn make_stroke(&self) -> Stroke {
        let dash = self.dashes.as_ref().and_then(|(offset, arr)| {
            if arr.is_empty() {
                return None;
            }
            // tiny_skia's StrokeDash::new requires a Vec<f32> (alternating
            // on/off lengths) and an offset.
            StrokeDash::new(arr.clone(), *offset)
        });
        Stroke {
            width: self.linewidth.max(0.0),
            miter_limit: self.miter_limit,
            line_cap: self.line_cap,
            line_join: self.line_join,
            dash,
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

fn read_cap_style(gc: &Bound<'_, PyAny>) -> LineCap {
    if let Ok(v) = gc.call_method0("get_capstyle") {
        if let Ok(s) = v.extract::<String>() {
            return match s.as_str() {
                "butt" => LineCap::Butt,
                "round" => LineCap::Round,
                "projecting" | "square" => LineCap::Square,
                _ => LineCap::Butt,
            };
        }
    }
    LineCap::Butt
}

fn read_join_style(gc: &Bound<'_, PyAny>) -> LineJoin {
    if let Ok(v) = gc.call_method0("get_joinstyle") {
        if let Ok(s) = v.extract::<String>() {
            return match s.as_str() {
                "miter" => LineJoin::Miter,
                "round" => LineJoin::Round,
                "bevel" => LineJoin::Bevel,
                _ => LineJoin::Miter,
            };
        }
    }
    LineJoin::Miter
}

/// Read `gc.get_dashes()` → `(offset, on_off_list)` in POINTS, convert
/// to pixels using the renderer's dpi. matplotlib returns `(None, None)`
/// for solid lines, and `(offset, [on, off, ...])` for dashed.
fn read_dashes(gc: &Bound<'_, PyAny>, dpi: f64) -> Option<(f32, Vec<f32>)> {
    let tup = gc.call_method0("get_dashes").ok()?;
    let (offset_obj, seq_obj): (Option<f64>, Option<Vec<f64>>) = match tup.extract() {
        Ok(v) => v,
        Err(_) => return None,
    };
    let seq = seq_obj?;
    if seq.is_empty() {
        return None;
    }
    // tiny_skia wants at least 2 elements and even length; if odd,
    // matplotlib convention is to repeat the pattern. We pad by
    // appending the list to itself to get an even-length pattern.
    let mut arr: Vec<f32> = seq.iter().map(|v| (v * dpi / 72.0) as f32).collect();
    if arr.len() == 1 {
        // A single value N means "N on, N off".
        arr.push(arr[0]);
    } else if arr.len() % 2 != 0 {
        let tail = arr.clone();
        arr.extend(tail);
    }
    // Ensure all segments are positive (tiny_skia asserts this).
    if arr.iter().any(|&v| v <= 0.0) {
        return None;
    }
    let offset = (offset_obj.unwrap_or(0.0) * dpi / 72.0) as f32;
    Some((offset, arr))
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
