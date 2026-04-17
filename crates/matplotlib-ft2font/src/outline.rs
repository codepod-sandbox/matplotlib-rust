//! Outline extraction via ttf-parser. Implements the `OutlineBuilder`
//! trait to translate font glyph outlines into matplotlib `Path` codes.
//!
//! matplotlib `Path` codes (from python/matplotlib/path.py):
//!     STOP      = 0
//!     MOVETO    = 1
//!     LINETO    = 2
//!     CURVE3    = 3   // quadratic; 2 vertices: control, end
//!     CURVE4    = 4   // cubic;     3 vertices: ctrl1, ctrl2, end
//!     CLOSEPOLY = 79
//!
//! ttf-parser's OutlineBuilder yields move_to/line_to/quad_to/curve_to/
//! close in font units with a y-up baseline. We emit these directly;
//! matplotlib's text2path layer scales them based on pt size at render.

pub const MOVETO: u8 = 1;
pub const LINETO: u8 = 2;
pub const CURVE3: u8 = 3;
pub const CURVE4: u8 = 4;
pub const CLOSEPOLY: u8 = 79;

/// Accumulator for a glyph outline. Each `OutlineBuilder` callback
/// pushes vertices and codes into this struct.
pub struct OutlineCollector {
    pub vertices: Vec<(f64, f64)>,
    pub codes: Vec<u8>,
}

impl OutlineCollector {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            codes: Vec::new(),
        }
    }
}

impl ttf_parser::OutlineBuilder for OutlineCollector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.vertices.push((x as f64, y as f64));
        self.codes.push(MOVETO);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.vertices.push((x as f64, y as f64));
        self.codes.push(LINETO);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        // CURVE3 spans two vertices (control + end); both get code CURVE3.
        self.vertices.push((x1 as f64, y1 as f64));
        self.codes.push(CURVE3);
        self.vertices.push((x as f64, y as f64));
        self.codes.push(CURVE3);
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        // CURVE4 spans three vertices (ctrl1 + ctrl2 + end); all CURVE4.
        self.vertices.push((x1 as f64, y1 as f64));
        self.codes.push(CURVE4);
        self.vertices.push((x2 as f64, y2 as f64));
        self.codes.push(CURVE4);
        self.vertices.push((x as f64, y as f64));
        self.codes.push(CURVE4);
    }

    fn close(&mut self) {
        // CLOSEPOLY also takes a vertex slot; matplotlib convention is
        // (0, 0) — the actual close happens via the code, not the coord.
        self.vertices.push((0.0, 0.0));
        self.codes.push(CLOSEPOLY);
    }
}
