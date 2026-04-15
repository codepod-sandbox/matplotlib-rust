//! BufferRegion — a saved snapshot of a rectangular region of the
//! RendererAgg's pixmap, used by Qt blit, animation background
//! restoration, and widget background caching.
//!
//! Matches the OG `matplotlib._backend_agg.BufferRegion` contract:
//! `copy_from_bbox(bbox)` returns one of these, and
//! `restore_region(region[, x1, y1, x2, y2, ox, oy])` blits it back.
//!
//! Exposes:
//! - `get_extents() -> (x0, y0, x1, y1)`
//! - `.width`, `.height`
//! - `__array__(dtype=None, copy=None) -> numpy ndarray` (H, W, 4) uint8
//! - `__getbuffer__` / `__releasebuffer__` — real Python buffer
//!   protocol, so `memoryview(region)` produces a (H, W, 4) view with
//!   format 'B'. This is what `backend_qtagg.py:50` and
//!   `backend_gtk3agg.py:42` rely on.

use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::os::raw::c_int;

#[pyclass(unsendable, module = "matplotlib.backends._backend_agg")]
pub struct BufferRegion {
    /// Premultiplied rgba8 pixel data for the captured region. Flat
    /// (height * width * 4) bytes, row-major.
    pub data: Vec<u8>,
    /// Pixmap-coord x origin of the region (top-left).
    pub x: i32,
    /// Pixmap-coord y origin of the region (top-left).
    pub y: i32,
    pub width: u32,
    pub height: u32,

    // Shape and strides are stored as struct fields so their addresses
    // are stable for the lifetime of the BufferRegion object. The
    // Py_buffer machinery holds raw pointers into these arrays; they
    // must not be on the stack or in a temporary.
    shape: [ffi::Py_ssize_t; 3],
    strides: [ffi::Py_ssize_t; 3],
}

#[pymethods]
impl BufferRegion {
    /// Matches OG matplotlib's `BufferRegion.get_extents()` — returns
    /// (x0, y0, x1, y1) in pixmap coords.
    fn get_extents(&self) -> (i32, i32, i32, i32) {
        (
            self.x,
            self.y,
            self.x + self.width as i32,
            self.y + self.height as i32,
        )
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    /// numpy array protocol: `np.asarray(region)` → (H, W, 4) uint8 view.
    /// The returned array is an owned copy of the region's data so it
    /// remains valid after the BufferRegion is collected. Callers can
    /// still mutate the array; it does not aliase the region's buffer.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<&Bound<'_, PyAny>>,
    ) -> Bound<'py, PyArray3<u8>> {
        let _ = (dtype, copy);
        let h = self.height as usize;
        let w = self.width as usize;
        let total = h * w * 4;
        // Build a fresh (H, W, 4) ndarray. If the region is empty fall
        // back to a zero-size array with the expected trailing dim.
        if total == 0 {
            return ndarray::Array3::<u8>::zeros((h.max(0), w.max(0), 4)).into_pyarray(py);
        }
        let arr =
            ndarray::Array3::from_shape_vec((h, w, 4), self.data.clone()).expect("shape mismatch");
        arr.into_pyarray(py)
    }

    /// Python buffer protocol: `memoryview(region)` → 3-D B-format view
    /// of shape (H, W, 4), strides (W*4, 4, 1), readonly.
    ///
    /// Safety: view's buf, shape, and strides point into `slf`'s heap-
    /// owned fields. The view holds a reference to `slf` via `(*view).obj`
    /// (incremented here, decremented automatically by pyo3 after
    /// __releasebuffer__), so the backing storage outlives the view.
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("null Py_buffer view"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err(
                "BufferRegion is read-only; writable buffer requested",
            ));
        }

        let total = (slf.width as ffi::Py_ssize_t) * (slf.height as ffi::Py_ssize_t) * 4;

        unsafe {
            // Zero out the view first (C API best practice).
            (*view).buf = if slf.data.is_empty() {
                // Empty region: provide a non-null but 0-length pointer.
                // Using the address of `slf.width` as a stand-in is safe
                // because len=0 means nothing is read.
                (&slf.width as *const u32) as *mut c_void
            } else {
                slf.data.as_ptr() as *mut c_void
            };

            // Reference the BufferRegion so Python keeps it alive until
            // the memoryview is dropped.
            let obj_ptr = slf.as_ptr();
            ffi::Py_INCREF(obj_ptr);
            (*view).obj = obj_ptr;

            (*view).len = total;
            (*view).readonly = 1;
            (*view).itemsize = 1;

            // Format string 'B' = unsigned byte. Static null-terminated.
            static FORMAT: &[u8] = b"B\0";
            (*view).format = FORMAT.as_ptr() as *mut _;

            (*view).ndim = 3;
            // Point into the struct-owned arrays; they outlive the view
            // because (*view).obj keeps slf alive.
            (*view).shape = slf.shape.as_ptr() as *mut ffi::Py_ssize_t;
            (*view).strides = slf.strides.as_ptr() as *mut ffi::Py_ssize_t;
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
        }
        Ok(())
    }

    /// Release hook. pyo3 automatically decrements `(*view).obj`'s
    /// refcount after this function returns, so there is nothing to
    /// do here — the shape/strides arrays are owned by `slf` and
    /// freed when the BufferRegion is collected.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

impl BufferRegion {
    /// Create a new BufferRegion by copying a rectangle out of a source
    /// pixmap. Clamps the rectangle to the pixmap bounds.
    pub fn from_pixmap(src: &tiny_skia::Pixmap, x: i32, y: i32, w: u32, h: u32) -> Self {
        let src_w = src.width() as i32;
        let src_h = src.height() as i32;

        // Clamp to pixmap bounds.
        let x0 = x.max(0).min(src_w);
        let y0 = y.max(0).min(src_h);
        let x1 = (x + w as i32).max(0).min(src_w);
        let y1 = (y + h as i32).max(0).min(src_h);
        let cw = (x1 - x0).max(0) as u32;
        let ch = (y1 - y0).max(0) as u32;

        if cw == 0 || ch == 0 {
            return Self::empty(x0, y0);
        }

        let stride = (src_w as usize) * 4;
        let src_bytes = src.data();
        let mut data = Vec::with_capacity((cw as usize) * (ch as usize) * 4);
        for row in 0..ch as usize {
            let src_row = y0 as usize + row;
            let start = src_row * stride + (x0 as usize) * 4;
            let end = start + (cw as usize) * 4;
            data.extend_from_slice(&src_bytes[start..end]);
        }

        Self::with_data(data, x0, y0, cw, ch)
    }

    fn with_data(data: Vec<u8>, x: i32, y: i32, w: u32, h: u32) -> Self {
        let ws = w as ffi::Py_ssize_t;
        let hs = h as ffi::Py_ssize_t;
        Self {
            data,
            x,
            y,
            width: w,
            height: h,
            shape: [hs, ws, 4],
            strides: [ws * 4, 4, 1],
        }
    }

    fn empty(x: i32, y: i32) -> Self {
        Self::with_data(Vec::new(), x, y, 0, 0)
    }

    /// Blit a sub-rectangle of this region back into a destination pixmap.
    /// `sub` is (sub_x, sub_y, sub_w, sub_h) in region-local coords;
    /// `dst_pos` is the pixmap destination top-left corner in pixmap coords.
    pub fn blit_to(
        &self,
        dst: &mut tiny_skia::Pixmap,
        sub: (i32, i32, u32, u32),
        dst_pos: (i32, i32),
    ) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let (sx, sy, sw, sh) = sub;
        let (dx, dy) = dst_pos;

        let region_w = self.width as i32;
        let region_h = self.height as i32;

        // Clip sub to region bounds.
        let src_x0 = sx.max(0).min(region_w);
        let src_y0 = sy.max(0).min(region_h);
        let src_x1 = (sx + sw as i32).max(0).min(region_w);
        let src_y1 = (sy + sh as i32).max(0).min(region_h);
        if src_x0 >= src_x1 || src_y0 >= src_y1 {
            return;
        }

        // Clip destination to pixmap bounds.
        let dst_w = dst.width() as i32;
        let dst_h = dst.height() as i32;
        let dst_x0 = dx.max(0);
        let dst_y0 = dy.max(0);
        let skip_x = dst_x0 - dx;
        let skip_y = dst_y0 - dy;
        let copy_w = ((src_x1 - src_x0) - skip_x).min(dst_w - dst_x0).max(0);
        let copy_h = ((src_y1 - src_y0) - skip_y).min(dst_h - dst_y0).max(0);
        if copy_w <= 0 || copy_h <= 0 {
            return;
        }

        let region_stride = (self.width as usize) * 4;
        let dst_stride = (dst_w as usize) * 4;
        // Copy rows directly. Data is already premultiplied, which is
        // what the destination pixmap expects. This is a raw overwrite,
        // NOT an alpha-blend; matches OG's BufferRegion behavior (the
        // region fully replaces the destination).
        let dst_data = dst.data_mut();
        for row in 0..copy_h as usize {
            let src_row = (src_y0 + skip_y) as usize + row;
            let src_col = (src_x0 + skip_x) as usize;
            let src_off = src_row * region_stride + src_col * 4;
            let dst_row = dst_y0 as usize + row;
            let dst_col = dst_x0 as usize;
            let dst_off = dst_row * dst_stride + dst_col * 4;
            let len = (copy_w as usize) * 4;
            dst_data[dst_off..dst_off + len].copy_from_slice(&self.data[src_off..src_off + len]);
        }
    }
}
