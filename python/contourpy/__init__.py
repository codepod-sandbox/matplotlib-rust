"""Small contourpy compatibility layer for local Matplotlib tests.

This is not a real contourpy implementation. It only provides the narrow
generator contract that ``matplotlib.contour`` uses in this repo today.
"""

from __future__ import annotations

import numpy as np

__version__ = "1.3.0"


class CoordinateType:
    Separate = 0
    SeparateCode = 1
    ChunkCombinedArray = 2
    ChunkCombinedCodesOffsets = 3
    ChunkCombinedOffset = 4
    ChunkCombinedNan = 5


class FillType:
    OuterCode = 0
    OuterOffset = 1
    ChunkCombinedCode = 2
    ChunkCombinedOffset = 3
    ChunkCombinedCodeOffset = 4
    ChunkCombinedOffsetOffset = 5


class LineType:
    Separate = 0
    SeparateCode = 1
    ChunkCombinedArray = 2
    ChunkCombinedOffset = 3
    ChunkCombinedNan = 4


def _normalize_xy(x, y, z):
    z_arr = np.ma.asarray(z, dtype=float)
    if z_arr.ndim != 2:
        raise TypeError("z must be a 2D array")

    if x is None or y is None:
        ny, nx = z_arr.shape
        x_arr, y_arr = np.meshgrid(np.arange(nx, dtype=float),
                                   np.arange(ny, dtype=float))
    else:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

    return x_arr, y_arr, z_arr


def _rect_path(x0, x1, y0, y1):
    vertices = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
        dtype=float,
    )
    codes = np.array([1, 2, 2, 2, 79], dtype=np.uint8)
    return vertices, codes


def _clamp01(value):
    return min(1.0, max(0.0, float(value)))


def _edge_point(p0, p1, z0, z1, level):
    if z1 == z0:
        t = 0.5
    else:
        t = (float(level) - float(z0)) / (float(z1) - float(z0))
    t = _clamp01(t)
    return np.array([
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1]),
    ], dtype=float)


def _cell_segments(points, values, level):
    edges = []
    edge_defs = ((0, 1), (1, 2), (2, 3), (3, 0))
    for i0, i1 in edge_defs:
        z0 = float(values[i0])
        z1 = float(values[i1])
        if ((z0 <= level < z1) or (z1 <= level < z0)
                or (level == z1 and z0 != z1)):
            edges.append(_edge_point(points[i0], points[i1], z0, z1, level))

    unique_edges = []
    for point in edges:
        if not any(np.allclose(point, existing) for existing in unique_edges):
            unique_edges.append(point)
    edges = unique_edges

    if len(edges) == 2:
        if np.allclose(edges[0], edges[1]):
            return []
        return [edges]
    if len(edges) == 4:
        segments = []
        for pair in ((edges[0], edges[1]), (edges[2], edges[3])):
            if not np.allclose(pair[0], pair[1]):
                segments.append([pair[0], pair[1]])
        return segments
    return []


def _clip_polygon(points, values, threshold, keep_above):
    clipped_points = []
    clipped_values = []
    n = len(points)
    for idx in range(n):
        p0 = points[idx]
        p1 = points[(idx + 1) % n]
        z0 = float(values[idx])
        z1 = float(values[(idx + 1) % n])
        inside0 = z0 >= threshold if keep_above else z0 <= threshold
        inside1 = z1 >= threshold if keep_above else z1 <= threshold

        if inside0 and inside1:
            clipped_points.append(np.array(p1, dtype=float))
            clipped_values.append(z1)
        elif inside0 and not inside1:
            ip = _edge_point(p0, p1, z0, z1, threshold)
            clipped_points.append(ip)
            clipped_values.append(float(threshold))
        elif not inside0 and inside1:
            ip = _edge_point(p0, p1, z0, z1, threshold)
            clipped_points.append(ip)
            clipped_values.append(float(threshold))
            clipped_points.append(np.array(p1, dtype=float))
            clipped_values.append(z1)

    return clipped_points, clipped_values


def _cell_filled_polygon(points, values, lower, upper):
    poly_points = [np.array(p, dtype=float) for p in points]
    poly_values = [float(v) for v in values]

    poly_points, poly_values = _clip_polygon(poly_points, poly_values, lower, True)
    if len(poly_points) < 3:
        return None
    poly_points, poly_values = _clip_polygon(poly_points, poly_values, upper, False)
    if len(poly_points) < 3:
        return None

    vertices = np.vstack(poly_points)
    if np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        return None

    return vertices


def _points_close(p0, p1):
    return np.allclose(p0, p1)


def _merge_segments(segments):
    paths = [[np.array(seg[0], dtype=float), np.array(seg[1], dtype=float)] for seg in segments]
    changed = True
    while changed:
        changed = False
        for i in range(len(paths)):
            if changed:
                break
            for j in range(i + 1, len(paths)):
                a = paths[i]
                b = paths[j]
                if _points_close(a[-1], b[0]):
                    paths[i] = a + b[1:]
                elif _points_close(a[-1], b[-1]):
                    paths[i] = a + list(reversed(b[:-1]))
                elif _points_close(a[0], b[-1]):
                    paths[i] = b[:-1] + a
                elif _points_close(a[0], b[0]):
                    paths[i] = list(reversed(b[1:])) + a
                else:
                    continue
                del paths[j]
                changed = True
                break
    return [np.vstack(path) for path in paths]


def _point_key(point, ndigits=12):
    return tuple(np.round(np.asarray(point, dtype=float), ndigits))


def _clean_polygon_vertices(polygon):
    points = [np.array(p, dtype=float) for p in polygon]
    cleaned = []
    for point in points:
        if not cleaned or not _points_close(cleaned[-1], point):
            cleaned.append(point)
    if len(cleaned) > 1 and _points_close(cleaned[0], cleaned[-1]):
        cleaned.pop()
    return cleaned


def _merge_polygons(polygons):
    edge_counts = {}
    edge_points = {}
    for polygon in polygons:
        points = _clean_polygon_vertices(polygon)
        if len(points) < 3:
            continue
        for i, p0 in enumerate(points):
            p1 = points[(i + 1) % len(points)]
            k0 = _point_key(p0)
            k1 = _point_key(p1)
            if k0 == k1:
                continue
            edge_key = tuple(sorted((k0, k1)))
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
            edge_points[edge_key] = (np.array(k0, dtype=float), np.array(k1, dtype=float))

    adjacency = {}
    for edge_key, count in edge_counts.items():
        if count != 1:
            continue
        p0, p1 = edge_points[edge_key]
        k0 = _point_key(p0)
        k1 = _point_key(p1)
        adjacency.setdefault(k0, set()).add(k1)
        adjacency.setdefault(k1, set()).add(k0)

    loops = []
    visited = set()
    for start in list(adjacency):
        if start in visited:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)
        if not component:
            continue

        start_node = min(component)
        polygon = [start_node]
        prev = None
        current = start_node
        while True:
            neighbors = adjacency[current] - ({prev} if prev is not None else set())
            if not neighbors:
                break
            if len(polygon) > 1 and start_node in neighbors:
                next_node = start_node
            else:
                next_node = sorted(neighbors)[0]
            polygon.append(next_node)
            prev, current = current, next_node
            if current == start_node:
                break
        if len(polygon) >= 4 and polygon[-1] == start_node:
            loops.append(np.array(polygon, dtype=float))

    return loops if loops else polygons


class _ContourGenerator:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.xmin = float(np.nanmin(x))
        self.xmax = float(np.nanmax(x))
        self.ymin = float(np.nanmin(y))
        self.ymax = float(np.nanmax(y))
        finite = z.compressed() if np.ma.isMaskedArray(z) else np.asarray(z).ravel()
        finite = finite[np.isfinite(finite)]
        self.zmin = float(np.min(finite)) if finite.size else 0.0
        self.zmax = float(np.max(finite)) if finite.size else 1.0

    def _normalize_level(self, level):
        if self.zmax <= self.zmin:
            return 0.5
        return _clamp01((float(level) - self.zmin) / (self.zmax - self.zmin))

    def create_contour(self, level):
        if not np.isfinite(level):
            return [], []

        segments = []
        ny, nx = self.z.shape
        for j in range(ny - 1):
            for i in range(nx - 1):
                points = [
                    np.array([self.x[j, i], self.y[j, i]], dtype=float),
                    np.array([self.x[j, i + 1], self.y[j, i + 1]], dtype=float),
                    np.array([self.x[j + 1, i + 1], self.y[j + 1, i + 1]], dtype=float),
                    np.array([self.x[j + 1, i], self.y[j + 1, i]], dtype=float),
                ]
                values = [
                    self.z[j, i],
                    self.z[j, i + 1],
                    self.z[j + 1, i + 1],
                    self.z[j + 1, i],
                ]
                segments.extend(_cell_segments(points, values, level))

        vertices = _merge_segments(segments)
        codes = [np.array([1, 2], dtype=np.uint8) for _ in vertices]
        codes = [np.concatenate(([1], np.full(len(v) - 1, 2, dtype=np.uint8))) for v in vertices]
        return vertices, codes

    def create_filled_contour(self, lower, upper):
        if not (np.isfinite(lower) and np.isfinite(upper)):
            return [], []
        if upper < lower:
            lower, upper = upper, lower

        polygons = []
        ny, nx = self.z.shape
        for j in range(ny - 1):
            for i in range(nx - 1):
                points = [
                    np.array([self.x[j, i], self.y[j, i]], dtype=float),
                    np.array([self.x[j, i + 1], self.y[j, i + 1]], dtype=float),
                    np.array([self.x[j + 1, i + 1], self.y[j + 1, i + 1]], dtype=float),
                    np.array([self.x[j + 1, i], self.y[j + 1, i]], dtype=float),
                ]
                values = [
                    self.z[j, i],
                    self.z[j, i + 1],
                    self.z[j + 1, i + 1],
                    self.z[j + 1, i],
                ]
                polygon = _cell_filled_polygon(points, values, lower, upper)
                if polygon is not None:
                    polygons.append(polygon)

        merged_polygons = _merge_polygons(polygons)

        vertices = []
        codes = []
        for polygon in merged_polygons:
            if len(polygon) > 1 and _points_close(polygon[0], polygon[-1]):
                polygon = polygon[:-1]
            closed = np.vstack([polygon, polygon[0]])
            code = np.full(len(closed), 2, dtype=np.uint8)
            code[0] = 1
            code[-1] = 79
            vertices.append(closed)
            codes.append(code)
        return vertices, codes


def contour_generator(x=None, y=None, z=None, name="serial",
                      corner_mask=None, line_type=None, fill_type=None,
                      chunk_size=None, chunk_count=None,
                      total_chunk_count=None, quad_as_tri=False,
                      z_interp=None, thread_count=0):
    if z is None:
        raise TypeError("contour_generator() missing required argument: 'z'")
    if line_type not in (None, LineType.SeparateCode):
        raise NotImplementedError("only LineType.SeparateCode is supported")
    if fill_type not in (None, FillType.OuterCode):
        raise NotImplementedError("only FillType.OuterCode is supported")

    x_arr, y_arr, z_arr = _normalize_xy(x, y, z)
    return _ContourGenerator(x_arr, y_arr, z_arr)
