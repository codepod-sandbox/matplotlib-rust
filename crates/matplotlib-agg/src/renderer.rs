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
use tiny_skia::Pixmap;

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

    #[pyo3(signature = (_gc, _path, _transform, _rgb_face=None))]
    fn draw_path(
        &mut self,
        _gc: &Bound<'_, PyAny>,
        _path: &Bound<'_, PyAny>,
        _transform: &Bound<'_, PyAny>,
        _rgb_face: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.dirty = true;
        Ok(())
    }

    #[pyo3(signature = (_gc, _marker_path, _marker_trans, _path, _trans, _rgb_face=None))]
    fn draw_markers(
        &mut self,
        _gc: &Bound<'_, PyAny>,
        _marker_path: &Bound<'_, PyAny>,
        _marker_trans: &Bound<'_, PyAny>,
        _path: &Bound<'_, PyAny>,
        _trans: &Bound<'_, PyAny>,
        _rgb_face: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.dirty = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_path_collection(
        &mut self,
        _gc: &Bound<'_, PyAny>,
        _master_transform: &Bound<'_, PyAny>,
        _paths: &Bound<'_, PyAny>,
        _all_transforms: &Bound<'_, PyAny>,
        _offsets: &Bound<'_, PyAny>,
        _offset_trans: &Bound<'_, PyAny>,
        _facecolors: &Bound<'_, PyAny>,
        _edgecolors: &Bound<'_, PyAny>,
        _linewidths: &Bound<'_, PyAny>,
        _linestyles: &Bound<'_, PyAny>,
        _antialiaseds: &Bound<'_, PyAny>,
        _urls: &Bound<'_, PyAny>,
        _offset_position: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.dirty = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_quad_mesh(
        &mut self,
        _gc: &Bound<'_, PyAny>,
        _master_transform: &Bound<'_, PyAny>,
        _mesh_width: usize,
        _mesh_height: usize,
        _coordinates: &Bound<'_, PyAny>,
        _offsets: &Bound<'_, PyAny>,
        _offset_trans: &Bound<'_, PyAny>,
        _facecolors: &Bound<'_, PyAny>,
        _antialiased: bool,
        _edgecolors: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.dirty = true;
        Ok(())
    }

    fn draw_image(
        &mut self,
        _gc: &Bound<'_, PyAny>,
        _x: f64,
        _y: f64,
        _im: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
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
        _obj: &Bound<'_, PyAny>,
        _x: i32,
        _y: i32,
        _angle: f64,
        _gc: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
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

    fn __array__<'py>(
        &mut self,
        py: Python<'py>,
        _dtype: Option<&Bound<'_, PyAny>>,
        _copy: Option<&Bound<'_, PyAny>>,
    ) -> Bound<'py, PyArray3<u8>> {
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
    // Wrapper calls:
    //   renderer.copy_from_bbox(bbox) -> region
    //   renderer.restore_region(region)  OR
    //   renderer.restore_region(region, x1, y1, x2, y2, ox, oy)
    //
    // For Milestone 1A these are no-ops returning None / sentinels. The
    // Python wrapper doesn't touch the region object directly in the
    // common savefig path.

    fn copy_from_bbox(&self, _bbox: &Bound<'_, PyAny>, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[pyo3(signature = (_region, _x1=None, _y1=None, _x2=None, _y2=None, _ox=None, _oy=None))]
    fn restore_region(
        &mut self,
        _region: &Bound<'_, PyAny>,
        _x1: Option<i32>,
        _y1: Option<i32>,
        _x2: Option<i32>,
        _y2: Option<i32>,
        _ox: Option<i32>,
        _oy: Option<i32>,
    ) -> PyResult<()> {
        Ok(())
    }

    // ----- filter / rasterizing stack -----

    fn start_filter(&mut self) {}
    fn stop_filter(&mut self, _post_processing: &Bound<'_, PyAny>) {}
    fn start_rasterizing(&mut self) {}
    fn stop_rasterizing(&mut self) {}
}
