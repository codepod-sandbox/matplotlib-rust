//! FT2Font pyclass + supporting types.

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// A single positioned glyph emitted by set_text, consumed by
/// draw_glyphs_to_bitmap. All coordinates are in pixels at the current
/// (ptsize, dpi); metrics are extracted from fontdue.
#[derive(Clone, Debug)]
struct Positioned {
    glyph_index: u16,
    /// Pen position in text-local pixel coords. `pen_x` advances
    /// left-to-right starting at 0; `pen_y` is the baseline.
    pen_x: f32,
    advance_width: f32,
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
    kerning: bool,
    kerning_factor: f32,

    /// Current size (points) + dpi, set via set_size.
    ptsize: f32,
    dpi: f32,

    /// Layout state from the last set_text call.
    current_text: String,
    #[allow(dead_code)]
    current_angle: f32,
    laid_out: Vec<Positioned>,
    loaded_glyphs: u32,

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
    has_current_glyph: bool,
    current_charmap_index: Option<usize>,
    fallback_list: Vec<Py<FT2Font>>,

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
    num_named_instances: u32,
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
        py: Python<'_>,
        filename: &Bound<'_, PyAny>,
        hinting_factor: u32,
        _fallback_list: Option<&Bound<'_, PyAny>>,
        _kerning: bool,
        _kerning_factor: Option<f32>,
    ) -> PyResult<Self> {
        let _ = hinting_factor;
        let filename = py
            .import("os")?
            .call_method1("fsdecode", (filename,))?
            .extract::<String>()?;
        let fallback_list = if let Some(fallbacks) = _fallback_list {
            if fallbacks.is_none() {
                Vec::new()
            } else {
                fallbacks
                    .try_iter()?
                    .map(|item| item?.extract::<Py<FT2Font>>())
                    .collect::<PyResult<Vec<_>>>()?
            }
        } else {
            Vec::new()
        };
        // Read font bytes.
        let font_data = std::fs::read(&filename).map_err(|e| {
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
        let num_charmaps = face.tables().cmap.map(|table| table.subtables.len() as u32).unwrap_or(0);
        let max_advance_width = (0..num_glyphs)
            .filter_map(|gid| face.glyph_hor_advance(ttf_parser::GlyphId(gid as u16)))
            .max()
            .unwrap_or(0);
        let max_advance_height = (0..num_glyphs)
            .filter_map(|gid| face.glyph_ver_advance(ttf_parser::GlyphId(gid as u16)))
            .max()
            .unwrap_or(height_font as u16) as i16;

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
            kerning: true,
            kerning_factor: _kerning_factor.unwrap_or(1.0),
            ptsize: 12.0,
            dpi: 72.0,
            current_text: String::new(),
            current_angle: 0.0,
            laid_out: Vec::new(),
            loaded_glyphs: 0,
            bitmap: ndarray::Array2::zeros((1, 1)),
            bitmap_offset: (0, 0),
            width_26_6: 0.0,
            height_26_6: 0.0,
            descent_26_6: 0.0,
            current_glyph_index: 0,
            has_current_glyph: false,
            current_charmap_index: None,
            fallback_list,

            fname: filename,
            family_name,
            style_name,
            postscript_name,
            num_faces: 1,
            num_named_instances: 0,
            face_flags,
            style_flags,
            num_glyphs,
            num_fixed_sizes: 0,
            num_charmaps,
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
            max_advance_width,
            max_advance_height,
        })
    }

    // ----- lifecycle + size -----

    fn clear(&mut self) {
        self.current_text.clear();
        self.laid_out.clear();
        self.loaded_glyphs = 0;
        self.bitmap = ndarray::Array2::zeros((1, 1));
        self.bitmap_offset = (0, 0);
        self.width_26_6 = 0.0;
        self.height_26_6 = 0.0;
        self.descent_26_6 = 0.0;
        self.current_charmap_index = None;
        self.has_current_glyph = false;
    }

    fn set_size(&mut self, ptsize: f32, dpi: f32) {
        self.ptsize = ptsize;
        self.dpi = dpi;
    }

    // ----- placeholder stubs; real bitmap rendering lands in task #33 -----

    #[pyo3(signature = (s, angle=0.0, flags=0, *, features=None, language=None))]
    fn set_text<'py>(
        &mut self,
        py: Python<'py>,
        s: &str,
        angle: f32,
        flags: i32,
        features: Option<&Bound<'_, PyAny>>,
        language: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let _ = flags;
        if let Some(features) = features {
            if !features.is_none() {
                if let Ok(iter) = features.try_iter() {
                    for item in iter {
                        let item = item?;
                        item.extract::<String>().map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err(
                                "features must be None or an iterable of strings",
                            )
                        })?;
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "features must be None or an iterable of strings",
                    ));
                }
            }
        }
        if let Some(language) = language {
            if !language.is_none()
                && language.extract::<String>().is_err()
            {
                if let Ok(iter) = language.try_iter() {
                    for item in iter {
                        let item = item?;
                        let tuple = item.extract::<(String, i32, i32)>().map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err(
                                "language must be None, a string, or an iterable of (str, int, int) tuples",
                            )
                        })?;
                        let _ = tuple;
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "language must be None, a string, or an iterable of (str, int, int) tuples",
                    ));
                }
            }
        }
        self.current_text = s.to_string();
        self.current_angle = angle;
        self.laid_out.clear();
        self.loaded_glyphs = 0;

        // Lay out glyphs using fontdue metrics. Each glyph's horizontal
        // advance in pixels drives the pen. Kerning pairs (2C) will be
        // added later; for now the layout is monotonic advance.
        let px = self.px_size();
        let mut pen_x = 0.0_f32;
        let mut max_ascent = 0.0_f32;
        let mut max_descent = 0.0_f32;
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;

        let mut prev_gid = None::<u16>;
        for ch in s.chars() {
            let gid = self.fontdue.lookup_glyph_index(ch);
            if self.kerning {
                if let Some(left_gid) = prev_gid {
                    let kern = self
                        .fontdue
                        .horizontal_kern_indexed(left_gid, gid, px)
                        .unwrap_or(0.0);
                    pen_x += kern * (if self.kerning_factor > 0.0 {
                        self.kerning_factor
                    } else {
                        1.0
                    });
                }
            }
            let metrics = self.fontdue.metrics_indexed(gid, px);
            let advance_width = metrics.advance_width;
            self.laid_out.push(Positioned {
                glyph_index: gid,
                pen_x,
                advance_width,
                metrics,
            });
            pen_x += advance_width;
            prev_gid = Some(gid);
            self.loaded_glyphs += 1;

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
            if metrics.width > 0 && metrics.height > 0 {
                let glyph_x = pen_x - advance_width + metrics.xmin as f32;
                let glyph_y = -(metrics.ymin as f32 + metrics.height as f32);
                if glyph_x < min_x {
                    min_x = glyph_x;
                }
                if glyph_y < min_y {
                    min_y = glyph_y;
                }
            }
        }

        if self.laid_out.is_empty() {
            self.width_26_6 = 0.0;
            self.height_26_6 = 0.0;
            self.descent_26_6 = 0.0;
            self.bitmap_offset = (0, 0);
            return Ok(ndarray::Array2::<i32>::zeros((0, 2)).into_pyarray(py));
        }

        // Bitmap metrics in 26.6 subpixels. get_width_height reports the
        // pixel-space size of the laid-out string; OG uses this to place
        // text in page coords.
        let total_h = (max_ascent + max_descent).max(px * 1.2);
        self.width_26_6 = pen_x * 64.0;
        self.height_26_6 = total_h * 64.0;
        self.descent_26_6 = max_descent * 64.0;
        if min_x.is_finite() && min_y.is_finite() {
            self.bitmap_offset = (
                (min_x.floor() * 64.0) as i32,
                (-min_y.floor() * 64.0) as i32,
            );
        } else {
            self.bitmap_offset = (0, 0);
        }
        let mut xys = ndarray::Array2::<i32>::zeros((self.laid_out.len(), 2));
        for (i, glyph) in self.laid_out.iter().enumerate() {
            xys[(i, 0)] = (glyph.pen_x * 64.0).round() as i32;
        }
        Ok(xys.into_pyarray(py))
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

        let min_x_floor = min_x.floor();
        let min_y_floor = min_y.floor();
        let max_x_ceil = max_x.ceil();
        let max_y_ceil = max_y.ceil();

        let bitmap_w = (max_x_ceil - min_x_floor).max(1.0) as usize;
        let bitmap_h = (max_y_ceil - min_y_floor).max(1.0) as usize;
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
            let (_m, pixels) = self.fontdue.rasterize_indexed(g.glyph_index, self.px_size());
            let gw = g.metrics.width;
            let gh = g.metrics.height;
            let gx = g.pen_x + g.metrics.xmin as f32;
            let gy = -(g.metrics.ymin as f32 + g.metrics.height as f32);
            let dst_x = gx.floor() as i32 - min_x_floor as i32;
            let dst_y = gy.floor() as i32 - min_y_floor as i32;

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
            (min_x_floor * 64.0) as i32,
            // +min_y because we flipped earlier; this is the y offset
            // in pixmap-down coordinates from the baseline to the
            // bitmap's top row.
            (-min_y_floor * 64.0) as i32,
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
        let _ = antialiased; // fontdue rasterization is antialiased.
        let Ok(mut image) = image.extract::<PyRefMut<'_, FT2Image>>() else {
            return;
        };
        let Ok(glyph) = glyph.extract::<PyRef<'_, Glyph>>() else {
            return;
        };
        let (metrics, pixels) = self
            .fontdue
            .rasterize_indexed(glyph.glyph_index as u16, glyph.pixel_size);
        if metrics.width == 0 || metrics.height == 0 {
            return;
        }

        let dst_x = x + metrics.xmin;
        let dst_y = y - (metrics.ymin + metrics.height as i32);
        let h = image.data.shape()[0] as i32;
        let w = image.data.shape()[1] as i32;

        for sy in 0..metrics.height {
            let dy = dst_y + sy as i32;
            if dy < 0 || dy >= h {
                continue;
            }
            for sx in 0..metrics.width {
                let dx = dst_x + sx as i32;
                if dx < 0 || dx >= w {
                    continue;
                }
                let src = pixels[sy * metrics.width + sx];
                if src == 0 {
                    continue;
                }
                let dst = &mut image.data[(dy as usize, dx as usize)];
                if src > *dst {
                    *dst = src;
                }
            }
        }
    }

    fn get_image<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        self.bitmap.clone().into_pyarray(py)
    }

    fn get_bitmap_offset(&self) -> (i32, i32) {
        self.bitmap_offset
    }

    // ----- glyph lookup (stubs for 2C) -----

    fn get_num_glyphs(&self) -> u32 {
        self.loaded_glyphs
    }

    fn get_char_index(&self, codepoint: u32) -> u32 {
        if codepoint == 0 {
            return 0;
        }
        let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) else {
            return 0;
        };
        if let Some(cmap) = face.tables().cmap {
            if let Some(index) = self.current_charmap_index {
                if let Some(sub) = cmap.subtables.get(index as u16) {
                    return sub.glyph_index(codepoint).map(|gid| gid.0 as u32).unwrap_or(0);
                }
            }
        }
        if let Some(ch) = char::from_u32(codepoint) {
            self.fontdue.lookup_glyph_index(ch) as u32
        } else {
            0
        }
    }

    fn get_name_index(&self, _name: &str) -> u32 {
        let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) else {
            return 0;
        };
        for gid in 0..face.number_of_glyphs() {
            let glyph_id = ttf_parser::GlyphId(gid);
            if face.glyph_name(glyph_id) == Some(_name) {
                return gid as u32;
            }
        }
        0
    }

    fn get_glyph_name(&self, index: u32) -> String {
        let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) else {
            return format!("glyph_{index}");
        };
        face.glyph_name(ttf_parser::GlyphId(index as u16))
            .map(str::to_string)
            .unwrap_or_else(|| {
                if index == 0 {
                    ".notdef".to_string()
                } else {
                    format!("glyph_{index}")
                }
            })
    }

    /// Return the full codepoint → glyph-id map. ttf-parser's cmap
    /// iterator yields every mapped (codepoint, glyph_id) pair.
    fn get_charmap(&self) -> std::collections::HashMap<u32, u32> {
        let mut out = std::collections::HashMap::new();
        if let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) {
            if let Some(cmap) = face.tables().cmap {
                if let Some(index) = self.current_charmap_index {
                    if let Some(sub) = cmap.subtables.get(index as u16) {
                        sub.codepoints(|cp| {
                            if cp == 0 {
                                return;
                            }
                            if let Some(gid) = sub.glyph_index(cp) {
                                out.entry(cp).or_insert(gid.0 as u32);
                            }
                        });
                    }
                } else {
                    for sub in cmap.subtables {
                        sub.codepoints(|cp| {
                            if cp == 0 {
                                return;
                            }
                            if let Some(gid) = sub.glyph_index(cp) {
                                out.entry(cp).or_insert(gid.0 as u32);
                            }
                        });
                    }
                }
            }
        }
        out
    }

    fn select_charmap(&mut self, i: u32) {
        let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) else {
            return;
        };
        let Some(cmap) = face.tables().cmap else {
            return;
        };
        match i {
            // FT_ENCODING_UNICODE
            0x756e6963 => {
                let mut best: Option<(usize, usize)> = None;
                for (idx, sub) in cmap.subtables.into_iter().enumerate() {
                    if !sub.is_unicode() {
                        continue;
                    }
                    let mut count = 0usize;
                    sub.codepoints(|cp| {
                        if let Some(gid) = sub.glyph_index(cp) {
                            let _ = gid;
                            count += 1;
                        }
                    });
                    if best.is_none_or(|(_, best_count)| count > best_count) {
                        best = Some((idx, count));
                    }
                }
                self.current_charmap_index = best.map(|(idx, _)| idx);
            }
            // FT_ENCODING_APPLE_ROMAN
            0x61726d6e => {
                self.current_charmap_index = cmap.subtables.into_iter().enumerate().find_map(
                    |(idx, sub)| {
                        (matches!(sub.platform_id, ttf_parser::PlatformId::Macintosh)
                            && sub.encoding_id == 0)
                            .then_some(idx)
                    },
                );
            }
            _ => {}
        }
    }

    fn set_charmap(&mut self, i: u32) {
        self.current_charmap_index = Some(i as usize);
    }

    /// Kerning between two glyph indices.
    ///
    /// - DEFAULT returns rounded pixel kerning in 26.6 subpixels.
    /// - UNFITTED returns unrounded scaled kerning in 26.6 subpixels.
    /// - UNSCALED returns raw font units.
    ///
    /// Uses ttf-parser's legacy `kern` table subtables. GPOS kerning
    /// requires full shaping and is out of scope for Phase 2.
    fn get_kerning(&self, left: u32, right: u32, mode: i32) -> i32 {
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
        match mode {
            2 => total_fu,
            1 => {
                let px = total_fu as f32 * self.px_size() / upem;
                (px * 64.0).floor() as i32
            }
            _ => {
                let px = total_fu as f32 * self.px_size() / upem;
                px.round() as i32 * 64
            }
        }
    }

    fn _layout<'py>(
        slf: Py<Self>,
        py: Python<'py>,
        s: &str,
        flags: i32,
    ) -> PyResult<Vec<LayoutItem>> {
        let _ = flags;
        let self_obj = slf.clone_ref(py);
        let slf_ref = slf.bind(py).borrow();
        let mut out = Vec::with_capacity(s.chars().count());
        for ch in s.chars() {
            let chosen = if slf_ref.get_char_index(ch as u32) != 0 {
                self_obj.clone_ref(py)
            } else {
                let mut chosen = None;
                for fallback in &slf_ref.fallback_list {
                    let fallback_ref = fallback.bind(py).borrow();
                    if fallback_ref.get_char_index(ch as u32) != 0 {
                        chosen = Some(fallback.clone_ref(py));
                        break;
                    }
                }
                chosen.unwrap_or_else(|| {
                    slf_ref.fallback_list
                        .last()
                        .map(|f| f.clone_ref(py))
                        .unwrap_or_else(|| self_obj.clone_ref(py))
                })
            };
            out.push(LayoutItem {
                char: ch.to_string(),
                ft_object: chosen,
            });
        }
        Ok(out)
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
                if name.name.is_empty() {
                    continue;
                }
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
        let raw_face = face.raw_face();
        let d = pyo3::types::PyDict::new(py);
        match name {
            "head" => {
                let Some(head) = raw_face.table(ttf_parser::Tag::from_bytes(b"head")) else {
                    return Ok(py.None());
                };
                d.set_item("version", fixed_tuple_1616(read_i32_be(head, 0).unwrap_or(0)))?;
                d.set_item("fontRevision", fixed_tuple_1616(read_i32_be(head, 4).unwrap_or(0)))?;
                d.set_item("checkSumAdjustment", read_i32_be(head, 8).unwrap_or(0))?;
                d.set_item("magicNumber", read_u32_be(head, 12).unwrap_or(0))?;
                d.set_item("flags", read_u16_be(head, 16).unwrap_or(0) as i32)?;
                d.set_item("unitsPerEm", read_u16_be(head, 18).unwrap_or(0) as i32)?;
                let created = read_u64_be(head, 20).unwrap_or(0);
                let modified = read_u64_be(head, 28).unwrap_or(0);
                d.set_item("created", ((created >> 32) as u32, created as u32))?;
                d.set_item("modified", ((modified >> 32) as u32, modified as u32))?;
                d.set_item("xMin", read_i16_be(head, 36).unwrap_or(0) as i32)?;
                d.set_item("yMin", read_i16_be(head, 38).unwrap_or(0) as i32)?;
                d.set_item("xMax", read_i16_be(head, 40).unwrap_or(0) as i32)?;
                d.set_item("yMax", read_i16_be(head, 42).unwrap_or(0) as i32)?;
                d.set_item("macStyle", read_u16_be(head, 44).unwrap_or(0) as i32)?;
                d.set_item("lowestRecPPEM", read_u16_be(head, 46).unwrap_or(0) as i32)?;
                d.set_item("fontDirectionHint", read_i16_be(head, 48).unwrap_or(0) as i32)?;
                d.set_item("indexToLocFormat", read_i16_be(head, 50).unwrap_or(0) as i32)?;
                d.set_item("glyphDataFormat", read_i16_be(head, 52).unwrap_or(0) as i32)?;
                Ok(d.into_any().unbind())
            }
            "hhea" => {
                let Some(hhea) = raw_face.table(ttf_parser::Tag::from_bytes(b"hhea")) else {
                    return Ok(py.None());
                };
                d.set_item("version", fixed_tuple_1616(read_i32_be(hhea, 0).unwrap_or(0)))?;
                d.set_item("ascent", read_i16_be(hhea, 4).unwrap_or(0) as i32)?;
                d.set_item("descent", read_i16_be(hhea, 6).unwrap_or(0) as i32)?;
                d.set_item("lineGap", read_i16_be(hhea, 8).unwrap_or(0) as i32)?;
                d.set_item("advanceWidthMax", read_u16_be(hhea, 10).unwrap_or(0) as i32)?;
                d.set_item("minLeftBearing", read_i16_be(hhea, 12).unwrap_or(0) as i32)?;
                d.set_item("minRightBearing", read_i16_be(hhea, 14).unwrap_or(0) as i32)?;
                d.set_item("xMaxExtent", read_i16_be(hhea, 16).unwrap_or(0) as i32)?;
                d.set_item("caretSlopeRise", read_i16_be(hhea, 18).unwrap_or(0) as i32)?;
                d.set_item("caretSlopeRun", read_i16_be(hhea, 20).unwrap_or(0) as i32)?;
                d.set_item("caretOffset", read_i16_be(hhea, 22).unwrap_or(0) as i32)?;
                d.set_item("metricDataFormat", read_i16_be(hhea, 32).unwrap_or(0) as i32)?;
                d.set_item("numOfLongHorMetrics", read_u16_be(hhea, 34).unwrap_or(0) as i32)?;
                Ok(d.into_any().unbind())
            }
            "OS/2" => {
                let Some(os2) = raw_face.table(ttf_parser::Tag::from_bytes(b"OS/2")) else {
                    return Ok(py.None());
                };
                let version = read_u16_be(os2, 0).unwrap_or(0) as i32;
                d.set_item("version", version)?;
                d.set_item("xAvgCharWidth", read_i16_be(os2, 2).unwrap_or(0) as i32)?;
                d.set_item("usWeightClass", read_u16_be(os2, 4).unwrap_or(0) as i32)?;
                d.set_item("usWidthClass", read_u16_be(os2, 6).unwrap_or(0) as i32)?;
                d.set_item("fsType", read_u16_be(os2, 8).unwrap_or(0) as i32)?;
                d.set_item("ySubscriptXSize", read_i16_be(os2, 10).unwrap_or(0) as i32)?;
                d.set_item("ySubscriptYSize", read_i16_be(os2, 12).unwrap_or(0) as i32)?;
                d.set_item("ySubscriptXOffset", read_i16_be(os2, 14).unwrap_or(0) as i32)?;
                d.set_item("ySubscriptYOffset", read_i16_be(os2, 16).unwrap_or(0) as i32)?;
                d.set_item("ySuperscriptXSize", read_i16_be(os2, 18).unwrap_or(0) as i32)?;
                d.set_item("ySuperscriptYSize", read_i16_be(os2, 20).unwrap_or(0) as i32)?;
                d.set_item("ySuperscriptXOffset", read_i16_be(os2, 22).unwrap_or(0) as i32)?;
                d.set_item("ySuperscriptYOffset", read_i16_be(os2, 24).unwrap_or(0) as i32)?;
                d.set_item("yStrikeoutSize", read_i16_be(os2, 26).unwrap_or(0) as i32)?;
                d.set_item("yStrikeoutPosition", read_i16_be(os2, 28).unwrap_or(0) as i32)?;
                d.set_item("sFamilyClass", read_i16_be(os2, 30).unwrap_or(0) as i32)?;
                d.set_item("panose", pyo3::types::PyBytes::new(py, os2.get(32..42).unwrap_or(&[])))?;
                d.set_item("ulUnicodeRange", (
                    read_u32_be(os2, 42).unwrap_or(0),
                    read_u32_be(os2, 46).unwrap_or(0),
                    read_u32_be(os2, 50).unwrap_or(0),
                    read_u32_be(os2, 54).unwrap_or(0),
                ))?;
                d.set_item("achVendID", pyo3::types::PyBytes::new(py, os2.get(58..62).unwrap_or(&[])))?;
                d.set_item("fsSelection", read_u16_be(os2, 62).unwrap_or(0) as i32)?;
                d.set_item("usFirstCharIndex", read_u16_be(os2, 64).unwrap_or(0) as i32)?;
                d.set_item("usLastCharIndex", read_u16_be(os2, 66).unwrap_or(0) as i32)?;
                d.set_item("sTypoAscender", read_i16_be(os2, 68).unwrap_or(0) as i32)?;
                d.set_item("sTypoDescender", read_i16_be(os2, 70).unwrap_or(0) as i32)?;
                d.set_item("sTypoLineGap", read_i16_be(os2, 72).unwrap_or(0) as i32)?;
                d.set_item("usWinAscent", read_u16_be(os2, 74).unwrap_or(0) as i32)?;
                d.set_item("usWinDescent", read_u16_be(os2, 76).unwrap_or(0) as i32)?;
                if version >= 1 {
                    d.set_item(
                        "ulCodePageRange",
                        (
                            read_u32_be(os2, 78).unwrap_or(0),
                            read_u32_be(os2, 82).unwrap_or(0),
                        ),
                    )?;
                }
                if version >= 2 {
                    d.set_item("sxHeight", read_i16_be(os2, 86).unwrap_or(0) as i32)?;
                    d.set_item("sCapHeight", read_i16_be(os2, 88).unwrap_or(0) as i32)?;
                    d.set_item("usDefaultChar", read_u16_be(os2, 90).unwrap_or(0) as i32)?;
                    d.set_item("usBreakChar", read_u16_be(os2, 92).unwrap_or(0) as i32)?;
                    d.set_item("usMaxContext", read_u16_be(os2, 94).unwrap_or(0) as i32)?;
                }
                Ok(d.into_any().unbind())
            }
            "post" => {
                let Some(post) = raw_face.table(ttf_parser::Tag::from_bytes(b"post")) else {
                    return Ok(py.None());
                };
                d.set_item("format", fixed_tuple_1616(read_i32_be(post, 0).unwrap_or(0)))?;
                d.set_item("italicAngle", fixed_tuple_1616(read_i32_be(post, 4).unwrap_or(0)))?;
                d.set_item("underlinePosition", read_i16_be(post, 8).unwrap_or(0) as i32)?;
                d.set_item("underlineThickness", read_i16_be(post, 10).unwrap_or(0) as i32)?;
                d.set_item("isFixedPitch", read_u32_be(post, 12).unwrap_or(0) as i32)?;
                d.set_item("minMemType42", read_u32_be(post, 16).unwrap_or(0) as i32)?;
                d.set_item("maxMemType42", read_u32_be(post, 20).unwrap_or(0) as i32)?;
                d.set_item("minMemType1", read_u32_be(post, 24).unwrap_or(0) as i32)?;
                d.set_item("maxMemType1", read_u32_be(post, 28).unwrap_or(0) as i32)?;
                Ok(d.into_any().unbind())
            }
            "maxp" => {
                let Some(maxp) = raw_face.table(ttf_parser::Tag::from_bytes(b"maxp")) else {
                    return Ok(py.None());
                };
                d.set_item("version", fixed_tuple_1616(read_i32_be(maxp, 0).unwrap_or(0)))?;
                d.set_item("numGlyphs", read_u16_be(maxp, 4).unwrap_or(0) as i32)?;
                d.set_item("maxPoints", read_u16_be(maxp, 6).unwrap_or(0) as i32)?;
                d.set_item("maxContours", read_u16_be(maxp, 8).unwrap_or(0) as i32)?;
                d.set_item("maxComponentPoints", read_u16_be(maxp, 10).unwrap_or(0) as i32)?;
                d.set_item("maxComponentContours", read_u16_be(maxp, 12).unwrap_or(0) as i32)?;
                d.set_item("maxZones", read_u16_be(maxp, 14).unwrap_or(0) as i32)?;
                d.set_item("maxTwilightPoints", read_u16_be(maxp, 16).unwrap_or(0) as i32)?;
                d.set_item("maxStorage", read_u16_be(maxp, 18).unwrap_or(0) as i32)?;
                d.set_item("maxFunctionDefs", read_u16_be(maxp, 20).unwrap_or(0) as i32)?;
                d.set_item("maxInstructionDefs", read_u16_be(maxp, 22).unwrap_or(0) as i32)?;
                d.set_item("maxStackElements", read_u16_be(maxp, 24).unwrap_or(0) as i32)?;
                d.set_item("maxSizeOfInstructions", read_u16_be(maxp, 26).unwrap_or(0) as i32)?;
                d.set_item("maxComponentElements", read_u16_be(maxp, 28).unwrap_or(0) as i32)?;
                d.set_item("maxComponentDepth", read_u16_be(maxp, 30).unwrap_or(0) as i32)?;
                Ok(d.into_any().unbind())
            }
            "pclt" => {
                let Some(pclt) = raw_face.table(ttf_parser::Tag::from_bytes(b"PCLT")) else {
                    return Ok(py.None());
                };
                d.set_item("version", fixed_tuple_1616(read_i32_be(pclt, 0).unwrap_or(0)))?;
                d.set_item("fontNumber", read_u32_be(pclt, 4).unwrap_or(0))?;
                d.set_item("pitch", read_u16_be(pclt, 8).unwrap_or(0) as i32)?;
                d.set_item("xHeight", read_u16_be(pclt, 10).unwrap_or(0) as i32)?;
                d.set_item("style", read_u16_be(pclt, 12).unwrap_or(0) as i32)?;
                d.set_item("typeFamily", read_u16_be(pclt, 14).unwrap_or(0) as i32)?;
                d.set_item("capHeight", read_u16_be(pclt, 16).unwrap_or(0) as i32)?;
                d.set_item("symbolSet", read_u16_be(pclt, 18).unwrap_or(0) as i32)?;
                d.set_item(
                    "typeFace",
                    pyo3::types::PyBytes::new(py, pclt.get(20..36).unwrap_or(&[])),
                )?;
                d.set_item(
                    "characterComplement",
                    pyo3::types::PyBytes::new(py, pclt.get(36..44).unwrap_or(&[])),
                )?;
                d.set_item("strokeWeight", pclt.get(50).copied().unwrap_or(0) as i8 as i32)?;
                d.set_item("widthType", pclt.get(51).copied().unwrap_or(0) as i8 as i32)?;
                d.set_item("serifStyle", pclt.get(52).copied().unwrap_or(0) as i8 as i32)?;
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
        self.has_current_glyph = true;
        self.loaded_glyphs += 1;
        self.glyph_at(gid)
    }

    /// Load a glyph by index and remember it for `get_path`.
    #[pyo3(signature = (glyph_index, flags=0))]
    fn load_glyph(&mut self, glyph_index: u32, flags: i32) -> Glyph {
        let _ = flags;
        let gid = glyph_index as u16;
        self.current_glyph_index = gid;
        self.has_current_glyph = true;
        self.loaded_glyphs += 1;
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
        if !self.has_current_glyph {
            return (
                ndarray::Array2::<f64>::zeros((0, 2)).into_pyarray(py),
                ndarray::Array1::<u8>::zeros(0).into_pyarray(py),
            );
        }
        let mut collector = crate::outline::OutlineCollector::new();
        let mut px_per_unit = 1.0_f64;

        // Re-parse the face on demand. ttf-parser's Face borrows the
        // font_data buffer, but only for the duration of this call.
        if let Ok(face) = ttf_parser::Face::parse(&self.font_data, 0) {
            px_per_unit = self.px_size() as f64 / f64::from(face.units_per_em().max(1));
            let _ = face.outline_glyph(
                ttf_parser::GlyphId(self.current_glyph_index),
                &mut collector,
            );
        }

        let n = collector.vertices.len();
        let mut verts = ndarray::Array2::<f64>::zeros((n, 2));
        for (i, (x, y)) in collector.vertices.iter().enumerate() {
            verts[(i, 0)] = *x * px_per_unit;
            verts[(i, 1)] = *y * px_per_unit;
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

fn read_u16_be(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_be_bytes(data.get(offset..offset + 2)?.try_into().ok()?))
}

fn read_i16_be(data: &[u8], offset: usize) -> Option<i16> {
    Some(i16::from_be_bytes(data.get(offset..offset + 2)?.try_into().ok()?))
}

fn read_u32_be(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(offset..offset + 4)?.try_into().ok()?))
}

fn read_i32_be(data: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_be_bytes(data.get(offset..offset + 4)?.try_into().ok()?))
}

fn read_u64_be(data: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_be_bytes(data.get(offset..offset + 8)?.try_into().ok()?))
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
            glyph_index: gid as u32,
            pixel_size: self.px_size(),
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

#[pyclass(unsendable, module = "matplotlib.ft2font")]
pub struct LayoutItem {
    #[pyo3(get)]
    char: String,
    #[pyo3(get)]
    ft_object: Py<FT2Font>,
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
    pub glyph_index: u32,
    pub pixel_size: f32,
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
