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

    /// Return the full codepoint → glyph-id map. ttf-parser's cmap
    /// iterator yields every mapped (codepoint, glyph_id) pair.
    fn get_charmap(&self) -> std::collections::HashMap<u32, u32> {
        let mut out = std::collections::HashMap::new();
        if let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) {
            if let Some(cmap) = face.tables().cmap {
                for sub in cmap.subtables {
                    sub.codepoints(|cp| {
                        if let Some(gid) = sub.glyph_index(cp) {
                            out.entry(cp).or_insert(gid.0 as u32);
                        }
                    });
                }
            }
        }
        out
    }

    fn select_charmap(&mut self, _i: u32) {}
    fn set_charmap(&mut self, _i: u32) {}

    /// Kerning between two glyph indices in 26.6 subpixels (mode is
    /// ignored — we always return design-unit kerning scaled to pixels).
    /// Uses ttf-parser's legacy `kern` table subtables. GPOS kerning
    /// requires full shaping and is out of scope for Phase 2.
    fn get_kerning(&self, left: u32, right: u32, _mode: i32) -> i32 {
        let face = match ttf_parser::Face::parse(&self.font_data, 0) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        let kern_table = match face.tables().kern {
            Some(k) => k,
            None => return 0,
        };
        let left_id = ttf_parser::GlyphId(left as u16);
        let right_id = ttf_parser::GlyphId(right as u16);
        let mut total_fu: i32 = 0;
        for sub in kern_table.subtables {
            if sub.horizontal && !sub.variable {
                if let Some(v) = sub.glyphs_kerning(left_id, right_id) {
                    total_fu += v as i32;
                }
            }
        }
        let upem = face.units_per_em() as f32;
        if upem <= 0.0 {
            return 0;
        }
        let px = total_fu as f32 * self.px_size() / upem;
        (px * 64.0) as i32
    }

    /// Return every name-table entry as
    ///   {(platform_id, encoding_id, language_id, name_id): bytes}.
    fn get_sfnt<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        let face = match ttf_parser::Face::parse(&self.font_data, 0) {
            Ok(f) => f,
            Err(_) => return Ok(dict),
        };
        if let Some(name_table) = face.tables().name {
            for name in name_table.names.into_iter() {
                let pid = platform_id_u16(name.platform_id) as u32;
                let eid = name.encoding_id as u32;
                let lid = name.language_id as u32;
                let nid = name.name_id as u32;
                let key = (pid, eid, lid, nid);
                dict.set_item(key, name.name)?;
            }
        }
        Ok(dict)
    }

    /// Return a dict representation of the named SFNT table. matplotlib
    /// touches a small handful of keys per table; we populate those
    /// from ttf-parser's parsed views. Unknown tables return None.
    fn get_sfnt_table(&self, name: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let face = match ttf_parser::Face::parse(&self.font_data, 0) {
            Ok(f) => f,
            Err(_) => return Ok(py.None()),
        };
        let tables = face.tables();
        let d = pyo3::types::PyDict::new(py);
        match name {
            "head" => {
                // _SfntHeadDict: version, fontRevision are 16.16
                // fixed-point exposed as (major, minor) tuples. created
                // / modified are LONGDATETIME (64-bit) split into
                // (hi, lo) 32-bit halves per the same convention.
                d.set_item("version", fixed_tuple_1616(0x10000))?;
                d.set_item("fontRevision", fixed_tuple_1616(0))?;
                d.set_item("checkSumAdjustment", 0_i32)?;
                d.set_item("magicNumber", 0x5F0F3CF5_u32)?;
                d.set_item("flags", 0_i32)?;
                d.set_item("unitsPerEm", face.units_per_em() as i32)?;
                d.set_item("created", (0_i32, 0_i32))?;
                d.set_item("modified", (0_i32, 0_i32))?;
                let bbox = face.global_bounding_box();
                d.set_item("xMin", bbox.x_min as i32)?;
                d.set_item("yMin", bbox.y_min as i32)?;
                d.set_item("xMax", bbox.x_max as i32)?;
                d.set_item("yMax", bbox.y_max as i32)?;
                let mut mac_style: u16 = 0;
                if face.is_bold() {
                    mac_style |= 0x01;
                }
                if face.is_italic() {
                    mac_style |= 0x02;
                }
                d.set_item("macStyle", mac_style as i32)?;
                d.set_item("lowestRecPPEM", 0_i32)?;
                d.set_item("fontDirectionHint", 2_i32)?;
                d.set_item("indexToLocFormat", 0_i32)?;
                d.set_item("glyphDataFormat", 0_i32)?;
                Ok(d.into_any().unbind())
            }
            "hhea" => {
                let h = tables.hhea;
                // _SfntHheaDict: version is 16.16 fixed-point tuple.
                d.set_item("version", fixed_tuple_1616(0x10000))?;
                d.set_item("ascent", h.ascender as i32)?;
                d.set_item("descent", h.descender as i32)?;
                d.set_item("lineGap", h.line_gap as i32)?;
                d.set_item("advanceWidthMax", 0_i32)?;
                d.set_item("minLeftBearing", 0_i32)?;
                d.set_item("minRightBearing", 0_i32)?;
                d.set_item("xMaxExtent", 0_i32)?;
                d.set_item("caretSlopeRise", 1_i32)?;
                d.set_item("caretSlopeRun", 0_i32)?;
                d.set_item("caretOffset", 0_i32)?;
                d.set_item("metricDataFormat", 0_i32)?;
                d.set_item("numOfLongHorMetrics", h.number_of_metrics as i32)?;
                Ok(d.into_any().unbind())
            }
            "OS/2" => {
                let os2 = match tables.os2 {
                    Some(t) => t,
                    None => return Ok(py.None()),
                };
                // _SfntOs2Dict: version is an int (not a tuple!).
                // We report version 4 (a real OS/2 version, not 0xFFFF
                // which matplotlib treats as "missing").
                d.set_item("version", 4_i32)?;
                d.set_item("xAvgCharWidth", 0_i32)?;
                d.set_item("usWeightClass", os2.weight().to_number() as i32)?;
                d.set_item("usWidthClass", os2.width().to_number() as i32)?;
                d.set_item("fsType", 0_i32)?;
                d.set_item("ySubscriptXSize", 0_i32)?;
                d.set_item("ySubscriptYSize", 0_i32)?;
                d.set_item("ySubscriptXOffset", 0_i32)?;
                d.set_item("ySubscriptYOffset", 0_i32)?;
                d.set_item("ySuperscriptXSize", 0_i32)?;
                d.set_item("ySuperscriptYSize", 0_i32)?;
                d.set_item("ySuperscriptXOffset", 0_i32)?;
                d.set_item("ySuperscriptYOffset", 0_i32)?;
                d.set_item("yStrikeoutSize", 0_i32)?;
                d.set_item("yStrikeoutPosition", 0_i32)?;
                d.set_item("sFamilyClass", 0_i32)?;
                // panose is raw 10 bytes; ulCharRange is a 4-tuple of
                // u32s (the Unicode range bits) per _SfntOs2Dict, NOT
                // bytes. achVendID is 4 raw bytes.
                d.set_item("panose", pyo3::types::PyBytes::new(py, &[0u8; 10]))?;
                d.set_item("ulCharRange", (0_u32, 0_u32, 0_u32, 0_u32))?;
                d.set_item("achVendID", pyo3::types::PyBytes::new(py, b"UKWN"))?;
                let mut fs_sel: u16 = 0;
                if face.is_italic() {
                    fs_sel |= 0x01;
                }
                if face.is_bold() {
                    fs_sel |= 0x20;
                }
                if !face.is_italic() && !face.is_bold() {
                    fs_sel |= 0x40;
                }
                d.set_item("fsSelection", fs_sel as i32)?;
                d.set_item("usFirstCharIndex", 0_i32)?;
                d.set_item("usLastCharIndex", 0xFFFF_i32)?;
                d.set_item("sTypoAscender", os2.typographic_ascender() as i32)?;
                d.set_item("sTypoDescender", os2.typographic_descender() as i32)?;
                d.set_item("sTypoLineGap", os2.typographic_line_gap() as i32)?;
                d.set_item("usWinAscent", os2.windows_ascender() as i32)?;
                d.set_item("usWinDescent", os2.windows_descender() as i32)?;
                Ok(d.into_any().unbind())
            }
            "post" => {
                // _SfntPostDict: format AND italicAngle are both 16.16
                // fixed-point exposed as (major, minor) tuples.
                // backend_pdf.py:1446 does `post['italicAngle'][1]`
                // which would TypeError on a float. matplotlib's OG C
                // ft2font returns the Fixed value split into signed
                // upper i16 + unsigned lower u16.
                d.set_item("format", fixed_tuple_1616(0x30000))?; // 3.0
                let italic_deg = face.italic_angle().unwrap_or(0.0);
                // Convert to 16.16 fixed-point then split.
                let fixed = (italic_deg * 65536.0) as i32;
                d.set_item("italicAngle", fixed_tuple_1616(fixed))?;
                d.set_item("underlinePosition", self.underline_position as i32)?;
                d.set_item("underlineThickness", self.underline_thickness as i32)?;
                d.set_item(
                    "isFixedPitch",
                    if face.is_monospaced() { 1_i32 } else { 0_i32 },
                )?;
                d.set_item("minMemType42", 0_i32)?;
                d.set_item("maxMemType42", 0_i32)?;
                d.set_item("minMemType1", 0_i32)?;
                d.set_item("maxMemType1", 0_i32)?;
                Ok(d.into_any().unbind())
            }
            "maxp" => {
                // _SfntMaxpDict: version is 16.16 fixed-point tuple.
                d.set_item("version", fixed_tuple_1616(0x10000))?;
                d.set_item("numGlyphs", face.number_of_glyphs() as i32)?;
                d.set_item("maxPoints", 0_i32)?;
                d.set_item("maxContours", 0_i32)?;
                d.set_item("maxComponentPoints", 0_i32)?;
                d.set_item("maxComponentContours", 0_i32)?;
                d.set_item("maxZones", 0_i32)?;
                d.set_item("maxTwilightPoints", 0_i32)?;
                d.set_item("maxStorage", 0_i32)?;
                d.set_item("maxFunctionDefs", 0_i32)?;
                d.set_item("maxInstructionDefs", 0_i32)?;
                d.set_item("maxStackElements", 0_i32)?;
                d.set_item("maxSizeOfInstructions", 0_i32)?;
                d.set_item("maxComponentElements", 0_i32)?;
                d.set_item("maxComponentDepth", 0_i32)?;
                Ok(d.into_any().unbind())
            }
            _ => Ok(py.None()),
        }
    }

    /// Return PostScript font info as a string-keyed dict.
    ///
    /// Despite what matplotlib's ft2font.pyi stub says (a 9-tuple), the
    /// real C ft2font extension returns a dict — font_manager.py:418
    /// does `font.get_ps_font_info()["weight"]`. The `.pyi` is stale.
    ///
    /// Keys mirror the FreeType `PS_FontInfoRec` struct. We populate
    /// what ttf-parser exposes from the `post` and `name` tables;
    /// unknown fields default to empty strings or zero.
    fn get_ps_font_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let face = ttf_parser::Face::parse(&self.font_data, 0).ok();
        let italic_angle = face
            .as_ref()
            .and_then(|f| f.italic_angle())
            .map(|a| a as i32)
            .unwrap_or(0);
        let is_fixed_pitch = face
            .as_ref()
            .map(|f| if f.is_monospaced() { 1 } else { 0 })
            .unwrap_or(0);
        let weight_name = face
            .as_ref()
            .and_then(|f| {
                if f.is_bold() {
                    Some("Bold")
                } else {
                    Some("Book")
                }
            })
            .unwrap_or("Unknown");

        let d = pyo3::types::PyDict::new(py);
        d.set_item("version", "")?;
        d.set_item("notice", "")?;
        d.set_item("full_name", self.postscript_name.clone())?;
        d.set_item("family_name", self.family_name.clone())?;
        d.set_item("weight", weight_name)?;
        d.set_item("italic_angle", italic_angle)?;
        d.set_item("is_fixed_pitch", is_fixed_pitch)?;
        d.set_item("underline_position", self.underline_position as i32)?;
        d.set_item("underline_thickness", self.underline_thickness as i32)?;
        Ok(d)
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

/// Split a 16.16 Fixed-point integer into the (major, minor) tuple
/// the OG C ft2font extension exposes for SFNT version / fontRevision
/// / italicAngle / format fields.
///
/// The upper 16 bits are a signed integer part; the lower 16 bits are
/// an unsigned fractional part. matplotlib's backend_pdf.py:1446
/// indexes italicAngle[1] (the lower u16), so returning a float here
/// would TypeError at runtime. See _SfntPostDict, _SfntHeadDict,
/// _SfntHheaDict, _SfntMaxpDict type stubs in ft2font.pyi.
fn fixed_tuple_1616(fixed: i32) -> (i16, u16) {
    let major = (fixed >> 16) as i16;
    let minor = (fixed & 0xFFFF) as u16;
    (major, minor)
}

/// Translate ttf-parser's PlatformId enum into the numeric value
/// stored in the TTF name table header (0=Unicode, 1=Macintosh,
/// 2=Iso, 3=Windows, 4=Custom).
fn platform_id_u16(pid: ttf_parser::PlatformId) -> u16 {
    match pid {
        ttf_parser::PlatformId::Unicode => 0,
        ttf_parser::PlatformId::Macintosh => 1,
        ttf_parser::PlatformId::Iso => 2,
        ttf_parser::PlatformId::Windows => 3,
        ttf_parser::PlatformId::Custom => 4,
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
            // horiAdvance uses 26.6 fixed-point (matplotlib divides by
            // 64 in backend_agg.py:222).
            horiAdvance: (advance_px * 64.0) as i32,
            // linearHoriAdvance uses 16.16 fixed-point — matplotlib's
            // _text_helpers.layout divides by 65536 (see
            // _text_helpers.py:81 `x += glyph.linearHoriAdvance / 65536`).
            // Using 26.6 here under-advances by 1024× and collapses
            // multi-glyph text layout in text2path / SVG output.
            linearHoriAdvance: (advance_px * 65536.0) as i32,
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
