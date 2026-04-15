"""Stub for matplotlib._backend_agg.

Phase 1 target: replaced by crates/matplotlib-agg (tiny-skia + fontdue).

For now this is a *no-op renderer* — methods satisfy the RendererBase
interface and return sensible defaults so that `fig.canvas.draw()` can
run without crashing. Text metrics return fixed estimates (6px * len(s),
12px height) since ft2font is not available (Phase 2). Draw methods are
no-ops; `buffer_rgba()` returns a blank image, `tostring_rgb()` returns
all-zero bytes.

This does NOT produce rasterized output. Tests that need actual pixel
content should continue to skip as Phase 1 proper. But tests that only
need `canvas.draw()` to complete (e.g. to trigger tick formatters) now
pass.
"""

import numpy as np

from matplotlib.backend_bases import RendererBase


def get_hinting_flag():
    return 0


class RendererAgg(RendererBase):
    """No-op RendererAgg — satisfies the API without actually rendering."""

    lock = None  # For thread safety (compat)

    def __init__(self, width, height, dpi):
        super().__init__()
        self.dpi = dpi
        self.width = int(width)
        self.height = int(height)
        self._renderer = self  # OG code uses self._renderer
        self._filter_renderers = []

    # -- drawing (no-ops) ----------------------------------------------------
    def draw_path(self, gc, path, transform, rgbFace=None):
        pass

    def draw_markers(self, gc, marker_path, marker_trans, path, trans,
                     rgbFace=None):
        pass

    def draw_path_collection(self, gc, master_transform, paths,
                             all_transforms, offsets, offset_trans, facecolors,
                             edgecolors, linewidths, linestyles, antialiaseds,
                             urls, offset_position):
        pass

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        pass

    def draw_gouraud_triangle(self, gc, points, colors, transform):
        pass

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        pass

    def draw_image(self, gc, x, y, im):
        pass

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        pass

    def draw_text_image(self, font_or_image, x, y, angle, gc):
        pass

    # -- text metrics (fake but sensible) -----------------------------------
    def get_text_width_height_descent(self, s, prop, ismath):
        if s is None:
            return 0.0, 0.0, 0.0
        try:
            n = len(str(s))
        except Exception:
            n = 0
        try:
            size = float(prop.get_size_in_points()) if prop is not None else 10
        except Exception:
            size = 10
        # Rough estimate: avg char width 0.6 * font size, line height 1.2
        width = n * size * 0.6
        height = size * 1.2
        descent = size * 0.2
        return width, height, descent

    def flipy(self):
        return True

    def get_canvas_width_height(self):
        return self.width, self.height

    def option_image_nocomposite(self):
        return False

    def option_scale_image(self):
        return True

    def points_to_pixels(self, points):
        return float(points) * self.dpi / 72.0

    def get_image_magnification(self):
        return 1.0

    # -- rasterized output (blank) ------------------------------------------
    def clear(self):
        pass

    def tostring_rgb(self):
        return bytes(self.width * self.height * 3)

    def tostring_argb(self):
        return bytes(self.width * self.height * 4)

    def buffer_rgba(self):
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def copy_from_bbox(self, bbox):
        return None

    def restore_region(self, region, bbox=None, xy=None):
        pass

    def start_filter(self):
        pass

    def stop_filter(self, filter_func):
        pass

    def start_rasterizing(self):
        pass

    def stop_rasterizing(self):
        pass
