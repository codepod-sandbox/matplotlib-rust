"""Pure-Python stub for matplotlib._path (used by RustPython, which cannot load .so).

CPython loads the real _path.so C extension. This stub is only used when the
.so is unavailable (e.g. RustPython smoke tests).
"""
import numpy as np


def affine_transform(points, matrix):
    pts = np.asarray(points, dtype=float)
    m = np.asarray(matrix, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.ones((pts.shape[0], 1))
    h = np.hstack([pts, ones])
    return (h @ m[:2, :].T + m[2, :2]).reshape(-1, 2) if pts.shape[1] == 2 else pts


def count_bboxes_overlapping_bbox(bbox, bboxes):
    return 0


def update_path_extents(path, transform, rect, minpos, ignore):
    return rect, minpos, False


def cleanup_path(path, transform, remove_nans, clip_rect, snap_mode,
                 stroke_width, simplify, return_curves, sketch):
    verts = np.asarray(path.vertices, dtype=float)
    codes = path.codes
    return verts, codes


def clip_path_to_rect(path, rect, inside):
    return [(path, None)]


def convert_path_to_polygons(path, transform=None, width=0, height=0,
                              closed_only=False):
    return []


def convert_to_string(path, transform, clip_rect, simplify, sketch,
                      default_linewidth, line_cap, line_join, close_paths,
                      israw=False):
    return b""


def get_path_collection_extents(master_transform, paths, transforms,
                                offsets, offset_transform):
    return (0.0, 0.0, 1.0, 1.0)


def is_sorted_and_has_non_nan(array):
    return False


def path_in_path(path_a, transform_a, path_b, transform_b, radius=0.0):
    return False


def path_intersects_path(path1, path2, transform1=None, transform2=None,
                         filled=False):
    return False


def path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2,
                               filled=False):
    return False


def point_in_path(x, y, radius, path, transform):
    return False


def point_in_path_collection(x, y, radius, master_transform, paths,
                              transforms, offsets, offset_trans,
                              filled, offset_position):
    return []


def points_in_path(points, radius, path, transform):
    pts = np.asarray(points)
    return np.zeros(len(pts), dtype=bool)
