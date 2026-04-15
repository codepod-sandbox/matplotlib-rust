//! BufferRegion — a saved snapshot of a rectangular region of the
//! RendererAgg's pixmap, used by Qt blit, animation background
//! restoration, and widget background caching.
//!
//! Matches the OG `matplotlib._backend_agg.BufferRegion` contract:
//! `copy_from_bbox(bbox)` returns one of these, and
//! `restore_region(region[, x1, y1, x2, y2, ox, oy])` blits it back.

use pyo3::prelude::*;

#[pyclass(unsendable, module = "matplotlib.backends._backend_agg")]
pub struct BufferRegion {
    /// Premultiplied rgba8 pixel data for the captured region.
    pub data: Vec<u8>,
    /// Pixmap-coord x origin of the region (top-left).
    pub x: i32,
    /// Pixmap-coord y origin of the region (top-left).
    pub y: i32,
    pub width: u32,
    pub height: u32,
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
            return Self {
                data: Vec::new(),
                x: x0,
                y: y0,
                width: 0,
                height: 0,
            };
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

        Self {
            data,
            x: x0,
            y: y0,
            width: cw,
            height: ch,
        }
    }

    /// Blit a sub-rectangle of this region back into a destination pixmap.
    /// `sub` is (sub_x, sub_y, sub_w, sub_h) in region-local coords;
    /// `dst` is the pixmap destination top-left corner in pixmap coords.
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
