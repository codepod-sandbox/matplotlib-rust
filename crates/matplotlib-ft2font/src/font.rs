//! FT2Font pyclass + supporting types.

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// A single positioned glyph emitted by set_text, consumed by
/// draw_glyphs_to_bitmap. All coordinates are in pixels at the current
/// (ptsize, dpi); metrics are extracted from fontdue.
#[derive(Clone, Debug)]
struct Positioned {
    ch: char,
    /// Pen position in text-local pixel coords. `pen_x` advances
    /// left-to-right starting at 0; `pen_y` is the baseline.
    pen_x: f32,
    /// fontdue metrics for this glyph at the current size. Stored so
    /// draw_glyphs_to_bitmap doesn't have to re-measure.
    metrics: fontdue::Metrics,
}

#[pyclass(unsendable, module = "matplotlib.ft2font")]
pub struct FT2Font {
    /// Owned font file bytes — both fontdue and ttf-parser borrow into
    /// this. It must outlive every parsed view we hand out.
    font_data: Vec<u8>,
    /// fontdue rasterizer for glyph bitmaps.
    fontdue: fontdue::Font,

    /// Current size (points) + dpi, set via set_size.
    ptsize: f32,
    dpi: f32,

    /// Layout state from the last set_text call.
    current_text: String,
    #[allow(dead_code)]
    current_angle: f32,
    laid_out: Vec<Positioned>,

    /// Rasterized bitmap from draw_glyphs_to_bitmap. `get_image` returns
    /// a copy of this as a (H, W) uint8 ndarray.
    bitmap: ndarray::Array2<u8>,
    /// Bitmap top-left offset in pixmap coords, 26.6 subpixels. OG's
    /// backend_agg.py uses this to align the blit to the baseline.
    bitmap_offset: (i32, i32),

    /// Text metrics, all 26.6 fixed-point (multiply pixels by 64).
    width_26_6: f32,
    height_26_6: f32,
    descent_26_6: f32,

    /// Active glyph index for `get_path()`. Updated by `load_char` /
    /// `load_glyph`. matplotlib's text2path pipeline calls
    /// `font.load_char(ccode); font.get_path()` per character, expecting
    /// get_path() to return the outline of the most recently loaded glyph.
    current_glyph_index: u16,

    // ----- Public attributes ------
    #[pyo3(get)]
    fname: String,
    #[pyo3(get)]
    family_name: String,
    #[pyo3(get)]
    style_name: String,
    #[pyo3(get)]
    postscript_name: String,
    #[pyo3(get)]
    num_faces: u32,
    #[pyo3(get)]
    face_flags: u32,
    #[pyo3(get)]
    style_flags: u32,
    #[pyo3(get)]
    num_glyphs: u32,
    #[pyo3(get)]
    num_fixed_sizes: u32,
    #[pyo3(get)]
    num_charmaps: u32,
    #[pyo3(get)]
    scalable: bool,
    #[pyo3(get)]
    units_per_EM: u16,
    #[pyo3(get)]
    underline_position: i16,
    #[pyo3(get)]
    underline_thickness: i16,
    #[pyo3(get)]
    bbox: (i16, i16, i16, i16),
    #[pyo3(get)]
    ascender: i16,
    #[pyo3(get)]
    descender: i16,
    #[pyo3(get)]
    height: i16,
    #[pyo3(get)]
    max_advance_width: u16,
    #[pyo3(get)]
    max_advance_height: i16,
}

#[pymethods]
impl FT2Font {
    #[new]
    #[pyo3(signature = (filename, hinting_factor=8, *, _fallback_list=None, _kerning=false, _kerning_factor=None))]
    fn new(
        filename: &str,
        hinting_factor: u32,
        _fallback_list: Option<&Bound<'_, PyAny>>,
        _kerning: bool,
        _kerning_factor: Option<f32>,
    ) -> PyResult<Self> {
        let _ = hinting_factor;
        // Read font bytes.
        let font_data = std::fs::read(filename).map_err(|e| {
            pyo3::exceptions::PyOSError::new_err(format!("ft2font: failed to read {filename}: {e}"))
        })?;

        // Parse via fontdue for rasterization.
        let fontdue =
            fontdue::Font::from_bytes(font_data.as_slice(), fontdue::FontSettings::default())
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "ft2font: fontdue failed to parse {filename}: {e}"
                    ))
                })?;

        // Parse via ttf-parser for metadata + outlines. Face is
        // short-lived — we only use it here to extract stable fields.
        let face = ttf_parser::Face::parse(&font_data, 0).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ft2font: ttf-parser failed to parse {filename}: {e:?}"
            ))
        })?;

        // Extract metadata.
        let family_name = face
            .names()
            .into_iter()
            .find(|n| n.name_id == ttf_parser::name_id::FAMILY && n.is_unicode())
            .and_then(|n| n.to_string())
            .unwrap_or_default();
        let style_name = face
            .names()
            .into_iter()
            .find(|n| n.name_id == ttf_parser::name_id::SUBFAMILY && n.is_unicode())
            .and_then(|n| n.to_string())
            .unwrap_or_default();
        let postscript_name = face
            .names()
            .into_iter()
            .find(|n| n.name_id == ttf_parser::name_id::POST_SCRIPT_NAME && n.is_unicode())
            .and_then(|n| n.to_string())
            .unwrap_or_default();

        let units_per_em = face.units_per_em();
        let ascender = face.ascender();
        let descender = face.descender();
        let height_font = face.height();
        let global_bbox = face.global_bounding_box();
        let underline = face.underline_metrics().unwrap_or(ttf_parser::LineMetrics {
            position: -100,
            thickness: 50,
        });
        let num_glyphs = face.number_of_glyphs() as u32;

        let mut face_flags: u32 = 0;
        if face.is_regular() || face.is_bold() || face.is_italic() {
            face_flags |= 1; // SCALABLE-ish
        }
        face_flags |= 8; // SFNT (true for all TTF/OTF)
        face_flags |= 16; // HORIZONTAL
        if face.tables().kern.is_some() {
            face_flags |= 64; // KERNING
        }
        face_flags |= 512; // GLYPH_NAMES (ttf-parser supports glyph_name())

        let mut style_flags: u32 = 0;
        if face.is_italic() {
            style_flags |= 1;
        }
        if face.is_bold() {
            style_flags |= 2;
        }

        Ok(Self {
            font_data,
            fontdue,
            ptsize: 12.0,
            dpi: 72.0,
            current_text: String::new(),
            current_angle: 0.0,
            laid_out: Vec::new(),
            bitmap: ndarray::Array2::zeros((1, 1)),
            bitmap_offset: (0, 0),
            width_26_6: 0.0,
            height_26_6: 0.0,
            descent_26_6: 0.0,
            current_glyph_index: 0,

            fname: filename.to_string(),
            family_name,
            style_name,
            postscript_name,
            num_faces: 1,
            face_flags,
            style_flags,
            num_glyphs,
            num_fixed_sizes: 0,
            num_charmaps: 1,
            scalable: true,
            units_per_EM: units_per_em,
            underline_position: underline.position,
            underline_thickness: underline.thickness,
            bbox: (
                global_bbox.x_min,
                global_bbox.y_min,
                global_bbox.x_max,
                global_bbox.y_max,
            ),
            ascender,
            descender,
            height: height_font,
            max_advance_width: 0,
            max_advance_height: 0,
        })
    }

    // ----- lifecycle + size -----

    fn clear(&mut self) {
        self.current_text.clear();
        self.laid_out.clear();
        self.bitmap = ndarray::Array2::zeros((1, 1));
        self.bitmap_offset = (0, 0);
        self.width_26_6 = 0.0;
        self.height_26_6 = 0.0;
        self.descent_26_6 = 0.0;
    }

    fn set_size(&mut self, ptsize: f32, dpi: f32) {
        self.ptsize = ptsize;
        self.dpi = dpi;
    }

    // ----- placeholder stubs; real bitmap rendering lands in task #33 -----

    #[pyo3(signature = (s, angle=0.0, flags=0))]
    fn set_text(&mut self, s: &str, angle: f32, flags: i32) -> PyResult<()> {
        let _ = flags;
        self.current_text = s.to_string();
        self.current_angle = angle;
        self.laid_out.clear();

        // Lay out glyphs using fontdue metrics. Each glyph's horizontal
        // advance in pixels drives the pen. Kerning pairs (2C) will be
        // added later; for now the layout is monotonic advance.
        let px = self.px_size();
        let mut pen_x = 0.0_f32;
        let mut max_ascent = 0.0_f32;
        let mut max_descent = 0.0_f32;

        for ch in s.chars() {
            let metrics = self.fontdue.metrics(ch, px);
            // fontdue advance_width includes bearings; use it for pen.
            self.laid_out.push(Positioned { ch, pen_x, metrics });
            pen_x += metrics.advance_width;

            // Track ascent/descent from the raster bounds:
            //   ymin = baseline offset (negative = below baseline)
            //   ymin + height = top of glyph relative to baseline
            let top = metrics.ymin as f32 + metrics.height as f32;
            if top > max_ascent {
                max_ascent = top;
            }
            let bottom = metrics.ymin as f32;
            if bottom < -max_descent {
                max_descent = -bottom;
            }
        }

        // Bitmap metrics in 26.6 subpixels. get_width_height reports the
        // pixel-space size of the laid-out string; OG uses this to place
        // text in page coords.
        let total_h = (max_ascent + max_descent).max(px * 1.2);
        self.width_26_6 = pen_x * 64.0;
        self.height_26_6 = total_h * 64.0;
        self.descent_26_6 = max_descent * 64.0;
        Ok(())
    }

    fn get_width_height(&self) -> (f32, f32) {
        (self.width_26_6, self.height_26_6)
    }

    fn get_descent(&self) -> f32 {
        self.descent_26_6
    }

    /// Rasterize the laid-out glyphs into the internal bitmap. Accepts
    /// `antialiased` as either `bool` or truthy `int` — matplotlib's
    /// wrapper passes `gc.get_antialiased()` which is 0/1 in some paths.
    #[pyo3(signature = (antialiased=None))]
    fn draw_glyphs_to_bitmap(&mut self, antialiased: Option<&Bound<'_, PyAny>>) {
        let _ = antialiased; // fontdue is always AA; mono is a Phase 2C polish item
        if self.laid_out.is_empty() {
            self.bitmap = ndarray::Array2::zeros((1, 1));
            self.bitmap_offset = (0, 0);
            return;
        }

        // First pass: compute the tight bounding box of all laid-out
        // glyph rasters (in pixel coords, y-down pixmap space).
        //
        // fontdue's glyph metrics use y-up with baseline at 0:
        //   xmin = horizontal bearing (left edge of glyph)
        //   ymin = vertical bearing (negative = below baseline)
        //   width, height = raster dimensions
        //
        // For the bitmap we want y to grow downward, so we flip ymin:
        //   raster_top_y    = -(ymin + height)
        //   raster_bottom_y = -ymin
        //
        // The pen's baseline is at y = 0 in text-local coords; we
        // translate to bitmap coords by subtracting the min raster_top_y
        // we encounter (so the topmost glyph pixel sits at row 0).
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for g in &self.laid_out {
            if g.metrics.width == 0 || g.metrics.height == 0 {
                continue;
            }
            let gx = g.pen_x + g.metrics.xmin as f32;
            let gy = -(g.metrics.ymin as f32 + g.metrics.height as f32);
            let gx_end = gx + g.metrics.width as f32;
            let gy_end = gy + g.metrics.height as f32;
            if gx < min_x {
                min_x = gx;
            }
            if gy < min_y {
                min_y = gy;
            }
            if gx_end > max_x {
                max_x = gx_end;
            }
            if gy_end > max_y {
                max_y = gy_end;
            }
        }

        if !min_x.is_finite() || !min_y.is_finite() {
            // No drawable glyphs (e.g. whitespace-only string).
            self.bitmap = ndarray::Array2::zeros((1, 1));
            self.bitmap_offset = (0, 0);
            return;
        }

        let bitmap_w = (max_x - min_x).ceil().max(1.0) as usize;
        let bitmap_h = (max_y - min_y).ceil().max(1.0) as usize;
        let mut bitmap = ndarray::Array2::<u8>::zeros((bitmap_h, bitmap_w));

        // Second pass: rasterize each glyph and composite into the
        // shared bitmap. We add alpha (saturating) so overlapping
        // glyphs accumulate rather than overwrite — matches FreeType's
        // composition for connected scripts; for normal Latin text the
        // glyphs don't overlap.
        for g in &self.laid_out {
            if g.metrics.width == 0 || g.metrics.height == 0 {
                continue;
            }
            let (_m, pixels) = self.fontdue.rasterize(g.ch, self.px_size());
            let gw = g.metrics.width;
            let gh = g.metrics.height;
            let gx = g.pen_x + g.metrics.xmin as f32;
            let gy = -(g.metrics.ymin as f32 + g.metrics.height as f32);
            let dst_x = (gx - min_x).round() as i32;
            let dst_y = (gy - min_y).round() as i32;

            for sy in 0..gh {
                let dy = dst_y + sy as i32;
                if dy < 0 || dy as usize >= bitmap_h {
                    continue;
                }
                for sx in 0..gw {
                    let dx = dst_x + sx as i32;
                    if dx < 0 || dx as usize >= bitmap_w {
                        continue;
                    }
                    let src = pixels[sy * gw + sx];
                    if src == 0 {
                        continue;
                    }
                    let dst = &mut bitmap[(dy as usize, dx as usize)];
                    // Saturating max so overlapping glyphs produce the
                    // union rather than doubling alpha.
                    if src > *dst {
                        *dst = src;
                    }
                }
            }
        }

        self.bitmap = bitmap;
        // bitmap_offset is the pen-relative origin of the bitmap's
        // top-left corner, in 26.6 subpixels. OG's backend_agg.py:181-203
        // uses it to align the blit to the baseline after rotation.
        self.bitmap_offset = (
            (min_x * 64.0).round() as i32,
            // +min_y because we flipped earlier; this is the y offset
            // in pixmap-down coordinates from the baseline to the
            // bitmap's top row.
            (-min_y * 64.0).round() as i32,
        );
    }

    #[pyo3(signature = (image, x, y, glyph, antialiased=None))]
    fn draw_glyph_to_bitmap(
        &mut self,
        image: &Bound<'_, PyAny>,
        x: i32,
        y: i32,
        glyph: &Bound<'_, PyAny>,
        antialiased: Option<&Bound<'_, PyAny>>,
    ) {
        let _ = (image, x, y, glyph, antialiased);
    }

    fn get_image<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        self.bitmap.clone().into_pyarray(py)
    }

    fn get_bitmap_offset(&self) -> (i32, i32) {
        self.bitmap_offset
    }

    // ----- glyph lookup (stubs for 2C) -----

    fn get_num_glyphs(&self) -> u32 {
        self.num_glyphs
    }

    fn get_char_index(&self, codepoint: u32) -> u32 {
        if let Some(ch) = char::from_u32(codepoint) {
            self.fontdue.lookup_glyph_index(ch) as u32
        } else {
            0
        }
    }

    fn get_name_index(&self, _name: &str) -> u32 {
        0
    }

    fn get_glyph_name(&self, index: u32) -> String {
        format!("glyph_{index}")
    }

    fn get_charmap(&self) -> std::collections::HashMap<u32, u32> {
        std::collections::HashMap::new()
    }

    fn select_charmap(&mut self, _i: u32) {}
    fn set_charmap(&mut self, _i: u32) {}

    fn get_kerning(&self, _left: u32, _right: u32, _mode: i32) -> i32 {
        0
    }

    fn get_sfnt(&self) -> std::collections::HashMap<(u32, u32, u32, u32), Vec<u8>> {
        std::collections::HashMap::new()
    }

    fn get_sfnt_table(&self, _name: &str, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    fn get_ps_font_info(&self, py: Python<'_>) -> Py<PyAny> {
        // OG ft2font returns a 14-element tuple of font metadata; PyO3
        // tuple conversion tops out at 12 elements, so we build a
        // Python tuple manually.
        let items: Vec<PyObject> = vec![
            self.family_name
                .clone()
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into_any(),
            self.postscript_name
                .clone()
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into_any(),
            "Stub".into_pyobject(py).unwrap().unbind().into_any(),
            "Stub".into_pyobject(py).unwrap().unbind().into_any(),
            "Stub".into_pyobject(py).unwrap().unbind().into_any(),
            (self.units_per_EM as u32)
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
            0_i32.into_pyobject(py).unwrap().unbind().into_any(),
        ];
        pyo3::types::PyTuple::new(py, items)
            .unwrap()
            .unbind()
            .into_any()
    }

    /// Load a glyph by Unicode codepoint and remember it for `get_path`.
    /// Returns a Glyph with horizontal advance metrics in 26.6 subpixels.
    /// matplotlib's text2path pipeline calls
    ///     font.load_char(ord(ch), flags=...)
    ///     font.get_path()
    /// per character, so this method's primary job is to set the active
    /// glyph index that get_path will read.
    #[pyo3(signature = (codepoint, flags=0))]
    fn load_char(&mut self, codepoint: u32, flags: i32) -> Glyph {
        let _ = flags;
        let ch = char::from_u32(codepoint).unwrap_or('\0');
        let gid = self.fontdue.lookup_glyph_index(ch);
        self.current_glyph_index = gid;
        self.glyph_at(gid)
    }

    /// Load a glyph by index and remember it for `get_path`.
    #[pyo3(signature = (glyph_index, flags=0))]
    fn load_glyph(&mut self, glyph_index: u32, flags: i32) -> Glyph {
        let _ = flags;
        let gid = glyph_index as u16;
        self.current_glyph_index = gid;
        self.glyph_at(gid)
    }

    // ----- SVG path support (2B) -----

    /// Return a {char: FT2Font} mapping for the chars in `s`. matplotlib
    /// uses this to support per-character font fallback; for now we map
    /// every char to `self` (single-font fallback). Real fallback chains
    /// are 2C / Phase 3 territory.
    fn _get_fontmap<'py>(
        slf: PyRef<'py, Self>,
        s: &str,
    ) -> PyResult<std::collections::HashMap<String, Py<PyAny>>> {
        let py = slf.py();
        let self_obj: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        let mut map = std::collections::HashMap::new();
        for ch in s.chars() {
            let key = ch.to_string();
            if !map.contains_key(&key) {
                map.insert(key, self_obj.clone_ref(py));
            }
        }
        Ok(map)
    }

    /// Return the outline of the most recently loaded glyph as a
    /// (vertices, codes) tuple ready for matplotlib.path.Path. matplotlib
    /// text2path calls this immediately after `load_char` per character.
    ///
    /// Coordinates are in font units (y-up baseline). matplotlib's text2path
    /// layer applies its own scale based on pt size at render time.
    fn get_path<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, numpy::PyArray1<u8>>) {
        let mut collector = crate::outline::OutlineCollector::new();

        // Re-parse the face on demand. ttf-parser's Face borrows the
        // font_data buffer, but only for the duration of this call.
        if let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) {
            let _ = face.outline_glyph(
                ttf_parser::GlyphId(self.current_glyph_index),
                &mut collector,
            );
        }

        let n = collector.vertices.len();
        let mut verts = ndarray::Array2::<f64>::zeros((n, 2));
        for (i, (x, y)) in collector.vertices.iter().enumerate() {
            verts[(i, 0)] = *x;
            verts[(i, 1)] = *y;
        }
        let codes = ndarray::Array1::from_vec(collector.codes);
        (verts.into_pyarray(py), codes.into_pyarray(py))
    }
}

impl FT2Font {
    /// Current pixel size at (ptsize, dpi).
    #[inline]
    fn px_size(&self) -> f32 {
        self.ptsize * self.dpi / 72.0
    }

    /// Build a `Glyph` for the given glyph index, populating horizontal
    /// metrics from ttf-parser. All metrics are in 26.6 subpixels per
    /// matplotlib's ft2font convention.
    fn glyph_at(&self, gid: u16) -> Glyph {
        let face = match ttf_parser::Face::parse(&self.font_data, 0) {
            Ok(f) => f,
            Err(_) => return Glyph::default(),
        };
        let glyph_id = ttf_parser::GlyphId(gid);
        // horizontal_advance is in font units; convert to pixels then 26.6.
        let units_per_em = face.units_per_em() as f32;
        let px_per_unit = self.px_size() / units_per_em.max(1.0);
        let advance_px = face
            .glyph_hor_advance(glyph_id)
            .map(|a| a as f32 * px_per_unit)
            .unwrap_or(0.0);
        let bearing_px = face
            .glyph_hor_side_bearing(glyph_id)
            .map(|b| b as f32 * px_per_unit)
            .unwrap_or(0.0);
        let bbox = face.glyph_bounding_box(glyph_id);
        let (bx0, by0, bx1, by1) = if let Some(r) = bbox {
            (
                (r.x_min as f32 * px_per_unit * 64.0) as i32,
                (r.y_min as f32 * px_per_unit * 64.0) as i32,
                (r.x_max as f32 * px_per_unit * 64.0) as i32,
                (r.y_max as f32 * px_per_unit * 64.0) as i32,
            )
        } else {
            (0, 0, 0, 0)
        };
        let width = (bx1 - bx0).max(0);
        let height = (by1 - by0).max(0);
        Glyph {
            width,
            height,
            horiBearingX: (bearing_px * 64.0) as i32,
            horiBearingY: by1, // top of glyph above baseline
            horiAdvance: (advance_px * 64.0) as i32,
            linearHoriAdvance: (advance_px * 64.0) as i32,
            vertBearingX: 0,
            vertBearingY: 0,
            vertAdvance: 0,
            bbox: (bx0, by0, bx1, by1),
        }
    }
}

// ----- FT2Image: a simple u8 image buffer used by mathtext -----

#[pyclass(unsendable, module = "matplotlib.ft2font")]
pub struct FT2Image {
    data: ndarray::Array2<u8>,
}

#[pymethods]
impl FT2Image {
    #[new]
    fn new(width: u32, height: u32) -> Self {
        Self {
            data: ndarray::Array2::zeros((height.max(1) as usize, width.max(1) as usize)),
        }
    }

    /// Fill a rectangle (inclusive on all edges) with 255. Used by
    /// matplotlib._mathtext for caret bars and box decorations.
    fn draw_rect_filled(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        let (h, w) = (self.data.shape()[0], self.data.shape()[1]);
        let x0 = x0.max(0) as usize;
        let y0 = y0.max(0) as usize;
        let x1 = (x1.max(0) as usize).min(w.saturating_sub(1));
        let y1 = (y1.max(0) as usize).min(h.saturating_sub(1));
        if x1 < x0 || y1 < y0 {
            return;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.data[(y, x)] = 255;
            }
        }
    }

    /// Return the buffer as a (H, W) uint8 ndarray.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<&Bound<'_, PyAny>>,
    ) -> Bound<'py, PyArray2<u8>> {
        let _ = (dtype, copy);
        self.data.clone().into_pyarray(py)
    }

    #[getter]
    fn width(&self) -> u32 {
        self.data.shape()[1] as u32
    }

    #[getter]
    fn height(&self) -> u32 {
        self.data.shape()[0] as u32
    }
}

// ----- module-level pyclasses -----

#[pyclass(module = "matplotlib.ft2font")]
#[derive(Default, Clone, Copy)]
pub struct Glyph {
    #[pyo3(get)]
    pub width: i32,
    #[pyo3(get)]
    pub height: i32,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    pub horiBearingX: i32,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    pub horiBearingY: i32,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    pub horiAdvance: i32,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    pub linearHoriAdvance: i32,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    pub vertBearingX: i32,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    pub vertBearingY: i32,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    pub vertAdvance: i32,
    #[pyo3(get)]
    pub bbox: (i32, i32, i32, i32),
}

// Simple marker pyclasses exposing the same constants the .py stub did.
// These were referenced as `ft2font.LoadFlags.DEFAULT` etc. in some
// matplotlib code paths.

#[pyclass(module = "matplotlib.ft2font")]
pub struct LoadFlags;

#[pymethods]
impl LoadFlags {
    #[classattr]
    const DEFAULT: i32 = 0;
    #[classattr]
    const NO_SCALE: i32 = 1;
    #[classattr]
    const NO_HINTING: i32 = 2;
    #[classattr]
    const RENDER: i32 = 4;
    #[classattr]
    const NO_BITMAP: i32 = 8;
    #[classattr]
    const VERTICAL_LAYOUT: i32 = 16;
    #[classattr]
    const FORCE_AUTOHINT: i32 = 32;
    #[classattr]
    const CROP_BITMAP: i32 = 64;
    #[classattr]
    const PEDANTIC: i32 = 128;
    #[classattr]
    const IGNORE_GLOBAL_ADVANCE_WIDTH: i32 = 512;
    #[classattr]
    const NO_RECURSE: i32 = 1024;
    #[classattr]
    const IGNORE_TRANSFORM: i32 = 2048;
    #[classattr]
    const MONOCHROME: i32 = 4096;
    #[classattr]
    const LINEAR_DESIGN: i32 = 8192;
    #[classattr]
    const NO_AUTOHINT: i32 = 32768;
}

#[pyclass(module = "matplotlib.ft2font")]
pub struct Kerning;

#[pymethods]
impl Kerning {
    #[classattr]
    const DEFAULT: i32 = 0;
    #[classattr]
    const UNFITTED: i32 = 1;
    #[classattr]
    const UNSCALED: i32 = 2;
}

#[pyclass(module = "matplotlib.ft2font")]
pub struct FaceFlags;

#[pymethods]
impl FaceFlags {
    #[classattr]
    const SCALABLE: u32 = 1;
    #[classattr]
    const FIXED_SIZES: u32 = 2;
    #[classattr]
    const FIXED_WIDTH: u32 = 4;
    #[classattr]
    const SFNT: u32 = 8;
    #[classattr]
    const HORIZONTAL: u32 = 16;
    #[classattr]
    const VERTICAL: u32 = 32;
    #[classattr]
    const KERNING: u32 = 64;
    #[classattr]
    const FAST_GLYPHS: u32 = 128;
    #[classattr]
    const MULTIPLE_MASTERS: u32 = 256;
    #[classattr]
    const GLYPH_NAMES: u32 = 512;
    #[classattr]
    const EXTERNAL_STREAM: u32 = 1024;
    #[classattr]
    const HINTER: u32 = 2048;
    #[classattr]
    const CID_KEYED: u32 = 4096;
    #[classattr]
    const TRICKY: u32 = 8192;
    #[classattr]
    const COLOR: u32 = 16384;
}

#[pyclass(module = "matplotlib.ft2font")]
pub struct StyleFlags;

#[pymethods]
impl StyleFlags {
    #[classattr]
    const NORMAL: u32 = 0;
    #[classattr]
    const ITALIC: u32 = 1;
    #[classattr]
    const BOLD: u32 = 2;
}
