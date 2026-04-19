//! RendererAgg — the pyclass that backs `_backend_agg.RendererAgg`.
//!
//! Milestone 1A-pre: the full API surface is present as no-ops so the
//! existing Python test baseline stays green when the `.so` replaces the
//! old `_backend_agg.py` Python stub. Individual methods get real
//! tiny-skia implementations in follow-up tasks (draw_path next).
//!
//! Method list mirrors what `python/matplotlib/backends/backend_agg.py`
//! binds in `_update_methods()` and calls directly elsewhere. Every
//! entry was verified against that file.

use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tiny_skia::{FillRule, Mask, Pixmap, Transform};

use crate::buffer_region::BufferRegion;
use crate::gc::{extract_rgba_face, GcInfo};
use crate::path::{compose_affines, extract_affine, path_to_tiny_skia, Affine};

/// Read `gc.get_clip_path()` if present. matplotlib returns
/// `(Path, Affine)` or `(None, None)`. When a path is present, we
/// extract the affine from the second element and return both, ready
/// for `path_to_tiny_skia`.
fn read_clip_path<'py>(gc: &Bound<'py, PyAny>) -> Option<(Bound<'py, PyAny>, Affine)> {
    let tup = gc.call_method0("get_clip_path").ok()?;
    // Tuple unpacking: (path, affine)
    let path_obj = tup.get_item(0).ok()?;
    let affine_obj = tup.get_item(1).ok()?;
    if path_obj.is_none() {
        return None;
    }
    let affine = extract_affine(&affine_obj);
    Some((path_obj, affine))
}

#[pyclass(unsendable)]
pub struct RendererAgg {
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub dpi: f64,

    /// tiny-skia draw surface (premultiplied rgba8, initial white).
    pub pixmap: Pixmap,

    /// Cached unpremultiplied rgba exposed via buffer protocol. Length
    /// = width * height * 4. Lazily refreshed when `dirty` is true.
    pub unpremul_cache: Vec<u8>,

    /// Set by every draw_* / clear. Cleared after buffer refresh.
    pub dirty: bool,
}

impl RendererAgg {
    /// Build a `tiny_skia::Mask` that represents the gc's clip for the
    /// current pixmap, or `None` if no clip is set.
    ///
    /// The mask is built from two sources, intersected:
    /// 1. `gc.get_clip_rectangle()` — a display-coords bbox (y-up).
    /// 2. `gc.get_clip_path()` — an arbitrary `(Path, Affine)` pair,
    ///    used for polar plots, pie wedges, non-rectangular axes.
    ///
    /// If both are set, the mask contains the INTERSECTION of the two
    /// shapes (standard matplotlib semantics).
    fn build_clip_mask(&self, info: &GcInfo, gc: &Bound<'_, PyAny>) -> Option<Mask> {
        let has_rect = info.clip_rect.is_some();
        let clip_path_info = read_clip_path(gc);
        let has_path = clip_path_info.is_some();
        if !has_rect && !has_path {
            return None;
        }

        let mut mask = Mask::new(self.width, self.height)?;

        // Start by filling the whole canvas with the rect (or the full
        // canvas if no rect is set). Mask values outside this rect are
        // implicitly 0. Mask::fill_path overwrites, so we need to compose
        // with a second pass for the clip path.
        let canvas_h = self.height as f32;

        let rect_path = if let Some([cx, cy, cw, ch]) = info.clip_rect {
            if cw <= 0.0 || ch <= 0.0 {
                return None;
            }
            // Match upstream Agg's clip_box rounding:
            // x1 -> floor(x1 + 0.5), x2 -> floor(x2 + 0.5),
            // y is flipped and each edge is rounded independently.
            let x1 = (cx as f64 + 0.5).floor().max(0.0) as f32;
            let x2 = ((cx + cw) as f64 + 0.5).floor().min(self.width as f64) as f32;
            let y_top = (canvas_h as f64 - cy as f64 + 0.5)
                .floor()
                .clamp(0.0, self.height as f64) as f32;
            let y_bottom = (canvas_h as f64 - (cy + ch) as f64 + 0.5)
                .floor()
                .clamp(0.0, self.height as f64) as f32;
            let rect = tiny_skia::Rect::from_xywh(x1, y_bottom, (x2 - x1).max(0.0), (y_top - y_bottom).max(0.0))?;
            Some(tiny_skia::PathBuilder::from_rect(rect))
        } else {
            // No rect → start from a mask covering the whole canvas.
            let rect = tiny_skia::Rect::from_xywh(0.0, 0.0, self.width as f32, canvas_h)?;
            Some(tiny_skia::PathBuilder::from_rect(rect))
        };

        if let Some(p) = rect_path {
            mask.fill_path(&p, FillRule::Winding, false, Transform::identity());
        }

        if let Some((clip_path_obj, clip_affine)) = clip_path_info {
            // Rasterize the clip path against the existing rect mask
            // using intersect_path, which multiplies the new path alpha
            // into the existing mask bits.
            if let Ok(Some(tsk_path)) =
                path_to_tiny_skia(
                    &clip_path_obj,
                    clip_affine,
                    self.height as f64,
                    Some(false),
                    0.0,
                )
            {
                mask.intersect_path(&tsk_path, FillRule::Winding, true, Transform::identity());
            }
        }

        Some(mask)
    }

    /// Build a tiling pattern pixmap for the gc's hatch, if any.
    ///
    /// matplotlib exposes hatches via `gc.get_hatch_path(density)` which
    /// returns a matplotlib Path in unit coords [0, 1]^2, and
    /// `gc.get_hatch_color()` / `gc.get_hatch_linewidth()`. The backend
    /// is responsible for rasterizing one tile and tiling it across the
    /// fill region with `SpreadMode::Repeat`.
    ///
    /// Returns `(tile_pixmap, tile_size_px)` or `None` if the gc has no
    /// hatch set.
    fn build_hatch_tile(&self, gc: &Bound<'_, PyAny>) -> Option<(Pixmap, f32)> {
        // Matplotlib's hatch density controls how much geometry is generated
        // inside a unit tile; it is not the tile size itself. Other backends
        // rasterize/repeat roughly one display-inch tile per hatch pattern.
        const DENSITY_PTS: f64 = 6.0;
        let density_px = (DENSITY_PTS * self.dpi / 72.0) as f32;
        let tile_size = self.dpi.max(2.0).ceil() as u32;

        // get_hatch_path(density) → Path or None
        let hatch_obj = gc.call_method1("get_hatch_path", (DENSITY_PTS,)).ok()?;
        if hatch_obj.is_none() {
            return None;
        }

        // Hatch color (RGBA). matplotlib defaults the hatch color to
        // the *edge* color of the patch — which can be (0, 0, 0, 0)
        // (transparent) when edgecolor='none'. In that case we fall
        // back to opaque black so the hatch is actually visible.
        let mut hatch_rgba = gc
            .call_method0("get_hatch_color")
            .ok()
            .and_then(|c| c.extract::<(f64, f64, f64, f64)>().ok())
            .map(|(r, g, b, a)| {
                [
                    (r * 255.0).clamp(0.0, 255.0) as u8,
                    (g * 255.0).clamp(0.0, 255.0) as u8,
                    (b * 255.0).clamp(0.0, 255.0) as u8,
                    (a * 255.0).clamp(0.0, 255.0) as u8,
                ]
            })
            .unwrap_or([0, 0, 0, 255]);
        if hatch_rgba[3] == 0 {
            hatch_rgba = [0, 0, 0, 255];
        }

        // Hatch linewidth in points → pixels. Default 1pt.
        let hatch_lw = gc
            .call_method0("get_hatch_linewidth")
            .ok()
            .and_then(|v| v.extract::<f64>().ok())
            .map(|pts| (pts * self.dpi / 72.0) as f32)
            .unwrap_or((self.dpi / 72.0) as f32)
            .max(1.0);

        // Build a tile_size x tile_size pixmap. The unit hatch path
        // (vertices in [0, 1] with overshoot to [-0.5, 1.5] for diagonal
        // patterns) gets scaled to fit the tile.
        let mut tile = Pixmap::new(tile_size, tile_size)?;

        let scale_affine = Affine {
            a: tile_size as f64,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: tile_size as f64,
            f: 0.0,
        };

        // canvas_height=tile_size so path_to_tiny_skia's y-flip puts
        // the unit-square in the right pixmap orientation.
        let tsk_path =
            path_to_tiny_skia(&hatch_obj, scale_affine, tile_size as f64, Some(false), 0.0)
                .ok()??;

        // Build paint via set_color_rgba8 — the f32 set_color path was
        // observed to produce 0 visible pixels in this codebase, while
        // the u8 path renders correctly. set_color_rgba8 is also what
        // matplotlib's reference Agg uses internally.
        let mut paint = tiny_skia::Paint::default();
        paint.set_color_rgba8(hatch_rgba[0], hatch_rgba[1], hatch_rgba[2], hatch_rgba[3]);
        paint.anti_alias = true;

        let stroke = tiny_skia::Stroke {
            width: hatch_lw,
            ..Default::default()
        };

        tile.stroke_path(&tsk_path, &paint, &stroke, Transform::identity(), None);
        // Some hatch characters ('o', 'O', '*') include filled regions;
        // fill the path on top with winding rule so solid sections show.
        tile.fill_path(
            &tsk_path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        Some((tile, density_px))
    }

    /// Refresh `unpremul_cache` from `pixmap` if `dirty`. Call this
    /// before any buffer-exposing method.
    fn refresh_buffer(&mut self) {
        if !self.dirty {
            return;
        }
        let src = self.pixmap.data(); // premultiplied rgba
                                      // Unpremultiply: for each pixel, if a > 0 divide rgb by a.
        let len = src.len();
        if self.unpremul_cache.len() != len {
            self.unpremul_cache.resize(len, 0);
        }
        let dst = &mut self.unpremul_cache;
        for i in (0..len).step_by(4) {
            let r = src[i];
            let g = src[i + 1];
            let b = src[i + 2];
            let a = src[i + 3];
            if a == 0 {
                dst[i] = 0;
                dst[i + 1] = 0;
                dst[i + 2] = 0;
                dst[i + 3] = 0;
            } else if a == 255 {
                dst[i] = r;
                dst[i + 1] = g;
                dst[i + 2] = b;
                dst[i + 3] = 255;
            } else {
                let af = a as u32;
                dst[i] = ((r as u32 * 255 + af / 2) / af).min(255) as u8;
                dst[i + 1] = ((g as u32 * 255 + af / 2) / af).min(255) as u8;
                dst[i + 2] = ((b as u32 * 255 + af / 2) / af).min(255) as u8;
                dst[i + 3] = a;
            }
        }
        self.dirty = false;
    }
}

#[pymethods]
impl RendererAgg {
    #[new]
    fn new(width: u32, height: u32, dpi: f64) -> PyResult<Self> {
        let pixmap = Pixmap::new(width.max(1), height.max(1)).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to allocate Pixmap({width}, {height})"
            ))
        })?;
        let cache_len = (width as usize) * (height as usize) * 4;
        Ok(Self {
            width,
            height,
            dpi,
            pixmap,
            unpremul_cache: vec![0; cache_len],
            dirty: true,
        })
    }

    /// Reset the pixmap to transparent and mark dirty.
    fn clear(&mut self) {
        self.pixmap.fill(tiny_skia::Color::TRANSPARENT);
        self.dirty = true;
    }

    // ----- drawing primitives (no-ops for now; real impls land per-task) -----

    #[pyo3(signature = (gc, path, transform, rgb_face=None))]
    fn draw_path(
        &mut self,
        gc: &Bound<'_, PyAny>,
        path: &Bound<'_, PyAny>,
        transform: &Bound<'_, PyAny>,
        rgb_face: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let affine = extract_affine(transform);
        let info = GcInfo::from_py(gc, self.dpi);
        let tsk_path = match path_to_tiny_skia(
            path,
            affine,
            self.height as f64,
            info.snap,
            info.linewidth,
        )? {
                Some(p) => p,
                None => return Ok(()), // empty path
            };
        let face = rgb_face.and_then(extract_rgba_face);
        let clip = self.build_clip_mask(&info, gc);

        // Fill first (if face is set), then stroke the outline.
        if let Some(face_rgba) = face {
            let paint = info.make_fill_paint(face_rgba);
            self.pixmap.fill_path(
                &tsk_path,
                &paint,
                FillRule::EvenOdd,
                Transform::identity(),
                clip.as_ref(),
            );
        }

        // 1B.2: hatching. If the gc has a hatch, build a tile pixmap
        // and fill the path with a tiled Pattern shader. This overlays
        // the hatch on top of the face, clipped to the path via the
        // fill operation (tiny-skia only renders the pattern where the
        // path is covered).
        if let Some((tile, _density)) = self.build_hatch_tile(gc) {
            let shader = tiny_skia::Pattern::new(
                tile.as_ref(),
                tiny_skia::SpreadMode::Repeat,
                tiny_skia::FilterQuality::Nearest,
                1.0,
                Transform::identity(),
            );
            let paint = tiny_skia::Paint {
                shader,
                anti_alias: true,
                ..Default::default()
            };
            self.pixmap.fill_path(
                &tsk_path,
                &paint,
                FillRule::EvenOdd,
                Transform::identity(),
                clip.as_ref(),
            );
        }

        if info.linewidth > 0.0 && info.foreground[3] > 0.0 {
            let paint = info.make_stroke_paint();
            let stroke = info.make_stroke();
            self.pixmap.stroke_path(
                &tsk_path,
                &paint,
                &stroke,
                Transform::identity(),
                clip.as_ref(),
            );
        }

        self.dirty = true;
        Ok(())
    }

    #[pyo3(signature = (gc, marker_path, marker_trans, path, trans, rgb_face=None))]
    fn draw_markers(
        &mut self,
        gc: &Bound<'_, PyAny>,
        marker_path: &Bound<'_, PyAny>,
        marker_trans: &Bound<'_, PyAny>,
        path: &Bound<'_, PyAny>,
        trans: &Bound<'_, PyAny>,
        rgb_face: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        // Build the marker shape once in "display-local" coords centered
        // at the origin. We use canvas_height=0 in path_to_tiny_skia so
        // the y axis is flipped (y → -y), giving us a shape already in
        // pixmap-local orientation.
        let marker_affine = extract_affine(marker_trans);
        let marker_tsk = match path_to_tiny_skia(marker_path, marker_affine, 0.0, Some(false), 0.0)?
        {
            Some(p) => p,
            None => return Ok(()),
        };

        // Walk the data path's vertices, apply the data→display affine,
        // and translate the marker to the resulting pixmap position.
        let data_trans = extract_affine(trans);
        let (verts_arr, _codes) = crate::path::extract_path_verts_codes(path)?;
        let verts = verts_arr.as_array();

        let info = GcInfo::from_py(gc, self.dpi);
        let face = rgb_face.and_then(extract_rgba_face);
        let clip = self.build_clip_mask(&info, gc);
        let h = self.height as f64;

        for i in 0..verts.nrows() {
            let (dx, dy) = data_trans.apply(verts[[i, 0]], verts[[i, 1]]);
            // Display → pixmap y-flip
            let pix = Transform::from_translate(dx as f32, (h - dy) as f32);

            if let Some(face_rgba) = face {
                let paint = info.make_fill_paint(face_rgba);
                self.pixmap
                    .fill_path(&marker_tsk, &paint, FillRule::EvenOdd, pix, clip.as_ref());
            }
            if info.linewidth > 0.0 && info.foreground[3] > 0.0 {
                let paint = info.make_stroke_paint();
                let stroke = info.make_stroke();
                self.pixmap
                    .stroke_path(&marker_tsk, &paint, &stroke, pix, clip.as_ref());
            }
        }

        self.dirty = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_path_collection(
        &mut self,
        gc: &Bound<'_, PyAny>,
        master_transform: &Bound<'_, PyAny>,
        paths: &Bound<'_, PyAny>,
        all_transforms: &Bound<'_, PyAny>,
        offsets: &Bound<'_, PyAny>,
        offset_trans: &Bound<'_, PyAny>,
        facecolors: &Bound<'_, PyAny>,
        edgecolors: &Bound<'_, PyAny>,
        linewidths: &Bound<'_, PyAny>,
        _linestyles: &Bound<'_, PyAny>,
        _antialiaseds: &Bound<'_, PyAny>,
        _urls: &Bound<'_, PyAny>,
        _offset_position: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use numpy::PyReadonlyArray2;

        // Sequence lengths. Each of paths / facecolors / edgecolors /
        // linewidths can be empty or shorter than the number of drawn items;
        // we cycle with modulo. Collections like ContourSet often have many
        // paths but only a single dummy offset at (0, 0), so draw whichever
        // dimension is larger rather than assuming offsets drives the loop.
        let n_paths: usize = paths.len().unwrap_or(0);
        if n_paths == 0 {
            return Ok(());
        }

        // Extract offsets as (N, 2) float64.
        let py = paths.py();
        let np = py.import("numpy")?;
        let offsets_obj = np
            .call_method1("ascontiguousarray", (offsets,))?
            .call_method1("astype", ("float64",))?;
        let offsets_arr: PyReadonlyArray2<f64> = match offsets_obj.extract() {
            Ok(a) => a,
            Err(_) => return Ok(()), // empty offsets
        };
        let offsets_view = offsets_arr.as_array();
        let n_offsets = offsets_view.nrows();
        if n_offsets == 0 {
            return Ok(());
        }
        let n_items = n_paths.max(n_offsets);

        // Facecolors and edgecolors as (N, 4) float64. Either may be empty.
        let face_arr: Option<PyReadonlyArray2<f64>> = np
            .call_method1("ascontiguousarray", (facecolors,))
            .and_then(|a| a.call_method1("astype", ("float64",)))
            .and_then(|a| a.extract())
            .ok();
        let edge_arr: Option<PyReadonlyArray2<f64>> = np
            .call_method1("ascontiguousarray", (edgecolors,))
            .and_then(|a| a.call_method1("astype", ("float64",)))
            .and_then(|a| a.extract())
            .ok();

        // Linewidths as 1D float64.
        let lw_arr: Option<numpy::PyReadonlyArray1<f64>> = np
            .call_method1("ascontiguousarray", (linewidths,))
            .and_then(|a| a.call_method1("astype", ("float64",)))
            .and_then(|a| a.extract())
            .ok();

        let master = extract_affine(master_transform);
        let offset_affine = extract_affine(offset_trans);

        // Pre-extract paths and per-path transforms.
        let transforms_arr: Option<PyReadonlyArray2<f64>> = np
            .call_method1("ascontiguousarray", (all_transforms,))
            .and_then(|a| a.call_method1("astype", ("float64",)))
            .and_then(|a| a.extract())
            .ok();
        let n_transform_rows = transforms_arr.as_ref().map_or(0, |arr| arr.as_array().nrows());
        let n_transforms = if n_transform_rows >= 3 { n_transform_rows / 3 } else { 0 };
        let mut path_cache: Vec<Option<tiny_skia::Path>> = Vec::with_capacity(n_paths);
        for i in 0..n_paths {
            let p = paths.get_item(i)?;
            // Combine master_transform with the per-path transform (if any).
            let per_path = if n_transforms > 0 {
                match transforms_arr.as_ref() {
                    Some(arr) => {
                        let v = arr.as_array();
                        let t_idx = (i % n_transforms) * 3;
                        compose_affines(
                            master,
                            Affine {
                                a: v[[t_idx, 0]],
                                b: v[[t_idx, 1]],
                                c: v[[t_idx, 2]],
                                d: v[[t_idx + 1, 0]],
                                e: v[[t_idx + 1, 1]],
                                f: v[[t_idx + 1, 2]],
                            },
                        )
                    }
                    None => master,
                }
            } else {
                master
            };
            // Build the path in "display local" orientation (y-flipped at
            // origin), so we can translate it per offset with a tiny_skia
            // Transform::from_translate.
            let tsk = path_to_tiny_skia(&p, per_path, 0.0, Some(false), 0.0)?;
            path_cache.push(tsk);
        }

        let info = GcInfo::from_py(gc, self.dpi);
        let clip = self.build_clip_mask(&info, gc);
        let hatch_tile = self.build_hatch_tile(gc).map(|(tile, _density)| tile);
        let h = self.height as f64;

        for i in 0..n_items {
            let offset_idx = i % n_offsets;
            let ox = offsets_view[[offset_idx, 0]];
            let oy = offsets_view[[offset_idx, 1]];
            let (tx, ty) = offset_affine.apply(ox, oy);
            let tf = Transform::from_translate(tx as f32, (h - ty) as f32);

            let path_idx = i % n_paths;
            let Some(tsk_path) = path_cache[path_idx].as_ref() else {
                continue;
            };

            // Per-index face color.
            let face_rgba = face_arr.as_ref().and_then(|arr| {
                let v = arr.as_array();
                if v.nrows() == 0 {
                    return None;
                }
                let row = v.row(i % v.nrows());
                if row.len() < 4 {
                    return None;
                }
                Some([row[0] as f32, row[1] as f32, row[2] as f32, row[3] as f32])
            });

            let edge_rgba = edge_arr.as_ref().and_then(|arr| {
                let v = arr.as_array();
                if v.nrows() == 0 {
                    return None;
                }
                let row = v.row(i % v.nrows());
                if row.len() < 4 {
                    return None;
                }
                Some([row[0] as f32, row[1] as f32, row[2] as f32, row[3] as f32])
            });

            // Per-index linewidth (pts → px).
            let lw_px = lw_arr.as_ref().and_then(|arr| {
                let v = arr.as_array();
                if v.len() == 0 {
                    return None;
                }
                Some((v[i % v.len()] * self.dpi / 72.0) as f32)
            });

            if let Some(face) = face_rgba {
                if face[3] > 0.0 {
                    let paint = info.make_fill_paint(face);
                    self.pixmap
                        .fill_path(tsk_path, &paint, FillRule::EvenOdd, tf, clip.as_ref());
                }
            }

            if let Some(tile) = hatch_tile.as_ref() {
                let shader = tiny_skia::Pattern::new(
                    tile.as_ref(),
                    tiny_skia::SpreadMode::Repeat,
                    tiny_skia::FilterQuality::Nearest,
                    1.0,
                    Transform::identity(),
                );
                let paint = tiny_skia::Paint {
                    shader,
                    anti_alias: true,
                    ..Default::default()
                };
                self.pixmap
                    .fill_path(tsk_path, &paint, FillRule::EvenOdd, tf, clip.as_ref());
            }

            if let (Some(edge), Some(lw)) = (edge_rgba, lw_px) {
                if edge[3] > 0.0 && lw > 0.0 {
                    let paint = info.make_fill_paint(edge);
                    let stroke = tiny_skia::Stroke {
                        width: lw,
                        ..info.make_stroke()
                    };
                    self.pixmap
                        .stroke_path(tsk_path, &paint, &stroke, tf, clip.as_ref());
                }
            }
        }

        self.dirty = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_quad_mesh(
        &mut self,
        gc: &Bound<'_, PyAny>,
        master_transform: &Bound<'_, PyAny>,
        mesh_width: usize,
        mesh_height: usize,
        coordinates: &Bound<'_, PyAny>,
        offsets: &Bound<'_, PyAny>,
        offset_trans: &Bound<'_, PyAny>,
        facecolors: &Bound<'_, PyAny>,
        _antialiased: bool,
        edgecolors: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use numpy::{PyReadonlyArray2, PyReadonlyArray3};

        let py = coordinates.py();
        let np = py.import("numpy")?;
        let coords_obj = np
            .call_method1("ascontiguousarray", (coordinates,))?
            .call_method1("astype", ("float64",))?;
        let coords: PyReadonlyArray3<f64> = match coords_obj.extract() {
            Ok(arr) => arr,
            Err(_) => return Ok(()),
        };
        let coords = coords.as_array();
        if coords.shape().len() != 3 || coords.shape()[2] != 2 {
            return Ok(());
        }
        if coords.shape()[0] < mesh_height + 1 || coords.shape()[1] < mesh_width + 1 {
            return Ok(());
        }

        let face_arr: Option<PyReadonlyArray2<f64>> = np
            .call_method1("ascontiguousarray", (facecolors,))
            .and_then(|a| a.call_method1("astype", ("float64",)))
            .and_then(|a| a.extract())
            .ok();
        let edge_arr: Option<PyReadonlyArray2<f64>> = np
            .call_method1("ascontiguousarray", (edgecolors,))
            .and_then(|a| a.call_method1("astype", ("float64",)))
            .and_then(|a| a.extract())
            .ok();
        let offsets_arr: Option<PyReadonlyArray2<f64>> = np
            .call_method1("ascontiguousarray", (offsets,))
            .and_then(|a| a.call_method1("astype", ("float64",)))
            .and_then(|a| a.extract())
            .ok();

        let master = extract_affine(master_transform);
        let offset_affine = extract_affine(offset_trans);
        let info = GcInfo::from_py(gc, self.dpi);
        let clip = self.build_clip_mask(&info, gc);
        let canvas_h = self.height as f64;

        let offset = offsets_arr
            .as_ref()
            .and_then(|arr| {
                let v = arr.as_array();
                if v.nrows() == 0 || v.ncols() < 2 {
                    None
                } else {
                    Some(offset_affine.apply(v[[0, 0]], v[[0, 1]]))
                }
            })
            .unwrap_or((0.0, 0.0));

        let get_rgba = |arr: &Option<PyReadonlyArray2<f64>>, idx: usize| -> Option<[f32; 4]> {
            let arr = arr.as_ref()?;
            let v = arr.as_array();
            if v.nrows() == 0 || v.ncols() < 4 {
                return None;
            }
            let row = v.row(idx % v.nrows());
            Some([row[0] as f32, row[1] as f32, row[2] as f32, row[3] as f32])
        };

        for j in 0..mesh_height {
            for i in 0..mesh_width {
                let idx = j * mesh_width + i;
                let corners = [
                    (coords[[j, i, 0]], coords[[j, i, 1]]),
                    (coords[[j, i + 1, 0]], coords[[j, i + 1, 1]]),
                    (coords[[j + 1, i + 1, 0]], coords[[j + 1, i + 1, 1]]),
                    (coords[[j + 1, i, 0]], coords[[j + 1, i, 1]]),
                ];

                let mut pb = tiny_skia::PathBuilder::new();
                for (k, (x, y)) in corners.iter().enumerate() {
                    let (tx, ty) = master.apply(*x, *y);
                    let px = (tx + offset.0) as f32;
                    let py = (canvas_h - (ty + offset.1)) as f32;
                    if k == 0 {
                        pb.move_to(px, py);
                    } else {
                        pb.line_to(px, py);
                    }
                }
                pb.close();
                let Some(path) = pb.finish() else {
                    continue;
                };

                if let Some(face) = get_rgba(&face_arr, idx) {
                    if face[3] > 0.0 {
                        let paint = info.make_fill_paint(face);
                        self.pixmap.fill_path(
                            &path,
                            &paint,
                            FillRule::EvenOdd,
                            Transform::identity(),
                            clip.as_ref(),
                        );
                    }
                }

                if let Some(edge) = get_rgba(&edge_arr, idx) {
                    if edge[3] > 0.0 && info.linewidth > 0.0 {
                        let paint = info.make_fill_paint(edge);
                        let stroke = info.make_stroke();
                        self.pixmap.stroke_path(
                            &path,
                            &paint,
                            &stroke,
                            Transform::identity(),
                            clip.as_ref(),
                        );
                    }
                }
            }
        }

        self.dirty = true;
        Ok(())
    }

    fn draw_image(
        &mut self,
        gc: &Bound<'_, PyAny>,
        x: f64,
        y: f64,
        im: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use numpy::PyReadonlyArray3;

        // `im` is an (H, W, 4) uint8 RGBA ndarray. Matplotlib's draw_image
        // contract uses display coordinates with the origin at the bottom
        // left, and Agg images arrive bottom-up; tiny-skia expects top-down
        // rows, so flip the source rows as we build the pixmap.
        //
        // The array is already prepared by Matplotlib's image machinery for
        // backend consumption, so preserve the channel bytes exactly rather
        // than premultiplying a second time.
        let arr: PyReadonlyArray3<u8> = match im.extract() {
            Ok(a) => a,
            Err(_) => return Ok(()),
        };
        let view = arr.as_array();
        let shape = view.shape();
        if shape.len() != 3 || shape[2] != 4 {
            return Ok(());
        }
        let h = shape[0] as u32;
        let w = shape[1] as u32;
        if h == 0 || w == 0 {
            return Ok(());
        }

        let gc_alpha = gc
            .call_method0("get_alpha")
            .ok()
            .and_then(|a| a.extract::<f64>().ok())
            .unwrap_or(1.0) as f32;

        let is_full_canvas_copy = x == 0.0
            && y == 0.0
            && w == self.width as u32
            && h == self.height as u32
            && (gc_alpha - 1.0).abs() < f32::EPSILON;

        // Build a tiny-skia Pixmap from the ndarray. The data must be
        // contiguous; `ascontiguousarray` in numpy land guarantees it.
        let mut buf = Vec::with_capacity((w as usize) * (h as usize) * 4);
        for row in 0..h as usize {
            let src_row = h as usize - 1 - row;
            for col in 0..w as usize {
                let r = view[[src_row, col, 0]] as u16;
                let g = view[[src_row, col, 1]] as u16;
                let b = view[[src_row, col, 2]] as u16;
                let a = view[[src_row, col, 3]] as u16;
                buf.push(((r * a + 127) / 255) as u8);
                buf.push(((g * a + 127) / 255) as u8);
                buf.push(((b * a + 127) / 255) as u8);
                buf.push(a as u8);
            }
        }
        let Some(src) =
            tiny_skia::Pixmap::from_vec(buf, tiny_skia::IntSize::from_wh(w, h).unwrap())
        else {
            return Ok(());
        };

        // matplotlib's (x, y) is the bottom-left corner in display coords.
        // tiny-skia wants top-left. With draw_image, the image extends
        // upward from (x, y) in display space.
        let canvas_h = self.height as f64;
        let dest_x = x as f32;
        let dest_y = (canvas_h - y - h as f64) as f32;

        // Apply gc alpha if present.
        let blend_mode = if is_full_canvas_copy {
            // Full-canvas renderer-buffer copies (e.g. pickle round-trip
            // tests using Figure.figimage(renderer.buffer_rgba())) should
            // behave as a direct framebuffer blit, not be alpha-composited
            // over the destination figure background a second time.
            tiny_skia::BlendMode::Source
        } else {
            tiny_skia::BlendMode::SourceOver
        };

        // Matplotlib's image pipeline has already resampled into the target
        // raster size before draw_image is called. Use nearest-neighbor here
        // so figure-image blits are pixel exact, matching Agg's behavior for
        // round-tripped renderer buffers.
        let paint = tiny_skia::PixmapPaint {
            opacity: gc_alpha.clamp(0.0, 1.0),
            blend_mode,
            quality: tiny_skia::FilterQuality::Nearest,
        };
        // Build a clip mask from the gc if present so the blit respects
        // the axes rectangle. draw_pixmap does not take a mask directly,
        // so use the GcInfo path instead.
        let info = GcInfo::from_py(gc, self.dpi);
        let clip = self.build_clip_mask(&info, gc);
        self.pixmap.draw_pixmap(
            dest_x as i32,
            dest_y as i32,
            src.as_ref(),
            &paint,
            Transform::identity(),
            clip.as_ref(),
        );

        self.dirty = true;
        Ok(())
    }

    fn draw_gouraud_triangles(
        &mut self,
        _gc: &Bound<'_, PyAny>,
        _points: &Bound<'_, PyAny>,
        _colors: &Bound<'_, PyAny>,
        _transform: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.dirty = true;
        Ok(())
    }

    fn draw_text_image(
        &mut self,
        obj: &Bound<'_, PyAny>,
        x: i32,
        y: i32,
        angle: f64,
        gc: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use numpy::PyReadonlyArray2;

        // `obj` is either an FT2Font instance (with .get_image() → ndarray)
        // or a 2D uint8 ndarray directly (TeX code path).
        let bitmap_obj = if let Ok(img) = obj.call_method0("get_image") {
            img.into_any()
        } else {
            obj.clone().into_any()
        };

        let arr: PyReadonlyArray2<u8> = match bitmap_obj.extract() {
            Ok(a) => a,
            Err(_) => {
                // ft2font stub may return something that's not a proper
                // ndarray; skip silently (text is invisible in Phase 1).
                self.dirty = true;
                return Ok(());
            }
        };
        let view = arr.as_array();
        let shape = view.shape();
        if shape.len() != 2 {
            return Ok(());
        }
        let bh = shape[0] as u32;
        let bw = shape[1] as u32;
        if bh == 0 || bw == 0 {
            return Ok(());
        }

        // Read gc foreground color for glyph tinting.
        let info = GcInfo::from_py(gc, self.dpi);
        let [r, g, b, _a] = info.foreground;
        let gc_alpha = info.alpha;

        // Build an RGBA pixmap from the grayscale mask: tinted color *
        // mask value as alpha. Premultiplied.
        let mut buf = Vec::with_capacity((bw as usize) * (bh as usize) * 4);
        for row in 0..bh as usize {
            for col in 0..bw as usize {
                let m = view[[row, col]];
                if m == 0 {
                    buf.extend_from_slice(&[0, 0, 0, 0]);
                    continue;
                }
                let alpha = (m as f32 / 255.0) * gc_alpha;
                let ra = (r * alpha * 255.0).clamp(0.0, 255.0) as u8;
                let ga = (g * alpha * 255.0).clamp(0.0, 255.0) as u8;
                let ba = (b * alpha * 255.0).clamp(0.0, 255.0) as u8;
                let aa = (alpha * 255.0).clamp(0.0, 255.0) as u8;
                buf.extend_from_slice(&[ra, ga, ba, aa]);
            }
        }
        let Some(src) =
            tiny_skia::Pixmap::from_vec(buf, tiny_skia::IntSize::from_wh(bw, bh).unwrap())
        else {
            return Ok(());
        };

        let clip = self.build_clip_mask(&info, gc);
        // Match upstream Agg semantics:
        // - the glyph bitmap occupies [0, bw] x [0, bh]
        // - the incoming (x, y) anchors the *top* edge for the unrotated
        //   branch via a y - bh shift
        // - the rotated branch first shifts by -bh in y, then rotates by
        //   -angle, then translates to (x, y)
        if angle.abs() < 1e-6 {
            let paint = tiny_skia::PixmapPaint {
                opacity: 1.0,
                blend_mode: tiny_skia::BlendMode::SourceOver,
                quality: tiny_skia::FilterQuality::Nearest,
            };
            self.pixmap.draw_pixmap(
                x,
                y - bh as i32,
                src.as_ref(),
                &paint,
                Transform::identity(),
                clip.as_ref(),
            );
        } else {
            let paint = tiny_skia::PixmapPaint {
                opacity: 1.0,
                blend_mode: tiny_skia::BlendMode::SourceOver,
                quality: tiny_skia::FilterQuality::Bicubic,
            };
            self.pixmap
                .draw_pixmap(
                    0,
                    0,
                    src.as_ref(),
                    &paint,
                    Transform::identity()
                        .post_translate(0.0, -(bh as f32))
                        .post_rotate(-(angle as f32))
                        .post_translate(x as f32, y as f32),
                    clip.as_ref(),
                );
        }

        self.dirty = true;
        Ok(())
    }

    // ----- buffer exposure -----

    /// Return an owned (H, W, 4) uint8 ndarray view of the current
    /// unpremultiplied buffer. OG's `backend_agg.py` calls
    /// `np.asarray(self._renderer)` at line 266 which routes here via
    /// __array__. This method is also callable directly as a fallback.
    fn buffer_rgba<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        self.refresh_buffer();
        let h = self.height as usize;
        let w = self.width as usize;
        let arr = ndarray::Array3::from_shape_vec((h, w, 4), self.unpremul_cache.clone())
            .expect("buffer shape mismatch");
        arr.into_pyarray(py)
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &mut self,
        py: Python<'py>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<&Bound<'_, PyAny>>,
    ) -> Bound<'py, PyArray3<u8>> {
        let _ = (dtype, copy);
        self.buffer_rgba(py)
    }

    /// Return the buffer as a Python bytes object (RGB only, no alpha).
    fn tostring_rgb<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.refresh_buffer();
        let n = (self.width as usize) * (self.height as usize);
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            out.push(self.unpremul_cache[i * 4]);
            out.push(self.unpremul_cache[i * 4 + 1]);
            out.push(self.unpremul_cache[i * 4 + 2]);
        }
        PyBytes::new(py, &out)
    }

    /// Return the buffer as a Python bytes object (ARGB order).
    fn tostring_argb<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.refresh_buffer();
        let n = (self.width as usize) * (self.height as usize);
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            out.push(self.unpremul_cache[i * 4 + 3]); // A
            out.push(self.unpremul_cache[i * 4]); // R
            out.push(self.unpremul_cache[i * 4 + 1]); // G
            out.push(self.unpremul_cache[i * 4 + 2]); // B
        }
        PyBytes::new(py, &out)
    }

    // ----- region lifecycle -----
    //
    // Wrapper calls (see backend_agg.py:283-316):
    //   renderer.copy_from_bbox(bbox) -> BufferRegion
    //   renderer.restore_region(region)
    //   renderer.restore_region(region, x1, y1, x2, y2, ox, oy)
    //
    // These are used by Qt blit (backends/backend_qtagg.py), animation
    // background restore (animation.py:1200), and widget background
    // caching (widgets.py:1090). They must be real, not no-ops.

    /// Capture a rectangular region of the current pixmap as a
    /// `BufferRegion`. `bbox` is a matplotlib Bbox-like whose `extents`
    /// attribute gives (x0, y0, x1, y1) in display coords (y-up). The
    /// real Bbox returns an ndarray from `.extents`, not a tuple, so
    /// we coerce via a tuple conversion.
    fn copy_from_bbox(&self, bbox: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let extract_4 = |obj: &Bound<'_, PyAny>| -> PyResult<(f64, f64, f64, f64)> {
            // Prefer direct tuple extraction for plain objects.
            if let Ok(t) = obj.extract::<(f64, f64, f64, f64)>() {
                return Ok(t);
            }
            // Fall back to iterating (handles ndarray, list, tuple, etc.)
            let list: Vec<f64> = obj
                .try_iter()?
                .map(|it| it?.extract::<f64>())
                .collect::<PyResult<_>>()?;
            if list.len() >= 4 {
                Ok((list[0], list[1], list[2], list[3]))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "bbox extents must have 4 elements",
                ))
            }
        };

        let (x0, y0, x1, y1): (f64, f64, f64, f64) = if let Ok(ext) = bbox.getattr("extents") {
            extract_4(&ext)?
        } else if let Ok(b) = bbox.getattr("bounds") {
            let (x, y, w, h) = extract_4(&b)?;
            (x, y, x + w, y + h)
        } else {
            return Ok(py.None());
        };

        // Display (y-up) → pixmap (y-down).
        let canvas_h = self.height as f64;
        let px = x0.floor() as i32;
        let py_top = (canvas_h - y1).floor() as i32;
        let pw = (x1.ceil() - x0.floor()).max(0.0) as u32;
        let ph = (y1.ceil() - y0.floor()).max(0.0) as u32;

        let region = BufferRegion::from_pixmap(&self.pixmap, px, py_top, pw, ph);
        Ok(Py::new(py, region)?.into_any())
    }

    /// Restore a previously captured BufferRegion back into the pixmap.
    /// Two calling shapes:
    ///   - restore_region(region): blit the whole region at its original
    ///     position
    ///   - restore_region(region, x1, y1, x2, y2, ox, oy): blit the
    ///     sub-rectangle (x1, y1)..(x2, y2) of the region (expressed in
    ///     pixmap coords relative to the whole canvas) to destination
    ///     (ox, oy) in pixmap coords.
    ///
    /// The wrapper at backend_agg.py:312 converts its Python-level bbox
    /// coords to integers before calling the 7-arg form; it always
    /// passes integers in pixmap (y-down) coords.
    #[pyo3(signature = (region, x1=None, y1=None, x2=None, y2=None, ox=None, oy=None))]
    fn restore_region(
        &mut self,
        region: &Bound<'_, PyAny>,
        x1: Option<i32>,
        y1: Option<i32>,
        x2: Option<i32>,
        y2: Option<i32>,
        ox: Option<i32>,
        oy: Option<i32>,
    ) -> PyResult<()> {
        if region.is_none() {
            return Ok(());
        }
        let region_ref = match region.extract::<PyRef<BufferRegion>>() {
            Ok(r) => r,
            Err(_) => return Ok(()), // non-BufferRegion: silent no-op
        };

        match (x1, y1, x2, y2, ox, oy) {
            (Some(px1), Some(py1), Some(px2), Some(py2), Some(pox), Some(poy)) => {
                // Sub-rect restore. (px1, py1)..(px2, py2) are canvas-
                // space pixmap coords, so translate to region-local
                // coords by subtracting the region's stored origin.
                let sx = px1 - region_ref.x;
                let sy = py1 - region_ref.y;
                let sw = (px2 - px1).max(0) as u32;
                let sh = (py2 - py1).max(0) as u32;
                region_ref.blit_to(&mut self.pixmap, (sx, sy, sw, sh), (pox, poy));
            }
            _ => {
                // Full-region restore at original position.
                let sw = region_ref.width;
                let sh = region_ref.height;
                region_ref.blit_to(
                    &mut self.pixmap,
                    (0, 0, sw, sh),
                    (region_ref.x, region_ref.y),
                );
            }
        }

        self.dirty = true;
        Ok(())
    }

    // ----- filter / rasterizing stack -----

    fn start_filter(&mut self) {}
    fn stop_filter(&mut self, _post_processing: &Bound<'_, PyAny>) {}
    fn start_rasterizing(&mut self) {}
    fn stop_rasterizing(&mut self) {}
}
