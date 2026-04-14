"""Stub for matplotlib._backend_agg.

Replaced by crates/matplotlib-agg (tiny-skia + fontdue) in Phase 1.
"""


def get_hinting_flag():
    return 0


class RendererAgg:
    """Stub RendererAgg — raises NotImplementedError on all drawing calls."""

    def __init__(self, width, height, dpi):
        self.width = int(width)
        self.height = int(height)
        self.dpi = dpi

    def get_image_magnification(self):
        return 1.0

    def clear(self):
        pass

    def draw_path(self, gc, path, transform, rgbFace=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_markers(self, gc, marker_path, marker_trans, path, trans,
                     rgbFace=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_path_collection(self, gc, master_transform, paths,
                             all_transforms, offsets, offset_trans, facecolors,
                             edgecolors, linewidths, linestyles, antialiaseds,
                             urls, offset_position):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_gouraud_triangle(self, gc, points, colors, transform):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_image(self, gc, x, y, im):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def tostring_rgb(self):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def tostring_argb(self):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def buffer_rgba(self):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def copy_from_bbox(self, bbox):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")

    def restore_region(self, region, bbox=None, xy=None):
        raise NotImplementedError("_backend_agg not yet implemented (Phase 1)")
