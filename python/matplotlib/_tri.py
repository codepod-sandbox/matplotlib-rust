import numpy as np

from matplotlib.path import Path


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

    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return polygons

    loops = []
    visited_edges = set()
    for start in sorted(adjacency):
        for next_node in sorted(adjacency[start]):
            edge = frozenset((start, next_node))
            if edge in visited_edges:
                continue

            polygon = [start]
            prev = start
            current = next_node
            visited_edges.add(edge)

            while True:
                polygon.append(current)
                if current == start:
                    break

                neighbors = adjacency[current]
                candidates = sorted(neighbors - {prev})
                if len(candidates) != 1:
                    return polygons

                next_point = candidates[0]
                edge = frozenset((current, next_point))
                if edge in visited_edges:
                    return polygons

                visited_edges.add(edge)
                prev, current = current, next_point

            if len(polygon) >= 4:
                loops.append(np.array(polygon, dtype=float))

    return loops if loops else polygons


def _endpoint_key(point):
    point = np.asarray(point, dtype=float)
    return (point[0], -point[1])


def _normalize_polyline(path):
    path = np.asarray(path, dtype=float)
    if _endpoint_key(path[0]) < _endpoint_key(path[-1]):
        path = path[::-1]
    return path


def _polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _normalize_polygon(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) > 1 and _points_close(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if _polygon_area(vertices) < 0:
        vertices = vertices[::-1]
    start = max(range(len(vertices)), key=lambda i: _endpoint_key(vertices[i]))
    return np.roll(vertices, -start, axis=0)


class Triangulation:
    def __init__(self, *args):
        if len(args) < 3:
            raise TypeError("__init__(): incompatible constructor arguments.")
        x, y, triangles, *rest = args
        mask = rest[0] if len(rest) > 0 else ()
        edges = rest[1] if len(rest) > 1 else ()
        neighbors = rest[2] if len(rest) > 2 else ()
        not_delaunay = rest[3] if len(rest) > 3 else False

        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if self.x.ndim != 1 or self.y.ndim != 1 or len(self.x) != len(self.y):
            raise ValueError("x and y must be 1D arrays of the same length")

        self.triangles = np.asarray(triangles, dtype=np.int32)
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must be a 2D array of shape (?,3)")

        if mask is None:
            raise ValueError("mask must be a 1D array with the same length as the triangles array")
        if len(mask) == 0:
            self.mask = None
        else:
            self.mask = np.asarray(mask, dtype=bool)
            if self.mask.ndim != 1 or len(self.mask) != len(self.triangles):
                raise ValueError("mask must be a 1D array with the same length as the triangles array")

        if edges is None:
            raise ValueError("edges must be a 2D array with shape (?,2)")
        if len(edges) == 0:
            self._edges = None
        else:
            self._edges = np.asarray(edges, dtype=np.int32)
            if self._edges.ndim != 2 or self._edges.shape[1] != 2:
                raise ValueError("edges must be a 2D array with shape (?,2)")

        if neighbors is None:
            raise ValueError("neighbors must be a 2D array with the same shape as the triangles array")
        if len(neighbors) == 0:
            self._neighbors = None
        else:
            self._neighbors = np.asarray(neighbors, dtype=np.int32)
            if self._neighbors.shape != self.triangles.shape:
                raise ValueError("neighbors must be a 2D array with the same shape as the triangles array")
        self.is_delaunay = not not_delaunay

    def _masked_triangles(self):
        if self.mask is None:
            return self.triangles
        return self.triangles[~self.mask]

    def get_edges(self):
        if self._edges is None:
            edge_set = set()
            for tri in self._masked_triangles():
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    edge_set.add(tuple(sorted((int(a), int(b)), reverse=True)))
            if edge_set:
                self._edges = np.asarray(sorted(edge_set), dtype=np.int32)
            else:
                self._edges = np.empty((0, 2), dtype=np.int32)
        return self._edges

    def get_neighbors(self):
        if self._neighbors is None:
            tris = self.triangles
            neighbors = np.full((len(tris), 3), -1, dtype=np.int32)
            edge_map = {}
            if self.mask is None:
                active = np.ones(len(tris), dtype=bool)
            else:
                active = ~self.mask
            for tri_idx, tri in enumerate(tris):
                if not active[tri_idx]:
                    continue
                for edge_idx, (a, b) in enumerate(((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))):
                    key = tuple(sorted((int(a), int(b))))
                    prev = edge_map.get(key)
                    if prev is None:
                        edge_map[key] = (tri_idx, edge_idx)
                    else:
                        other_tri_idx, other_edge_idx = prev
                        neighbors[tri_idx, edge_idx] = other_tri_idx
                        neighbors[other_tri_idx, other_edge_idx] = tri_idx
            self._neighbors = neighbors
        return self._neighbors

    def set_mask(self, mask):
        if mask is None:
            raise ValueError("mask must be a 1D array with the same length as the triangles array")
        if len(mask) == 0:
            self.mask = None
        else:
            self.mask = np.asarray(mask, dtype=bool)
            if self.mask.ndim != 1 or len(self.mask) != len(self.triangles):
                raise ValueError("mask must be a 1D array with the same length as the triangles array")
        self._edges = None
        self._neighbors = None

    def calculate_plane_coefficients(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 1 or len(z) != len(self.x):
            raise ValueError("z must be a 1D array with the same length as the triangulation x and y arrays")
        coeffs = np.zeros((len(self.triangles), 3), dtype=np.float64)
        for i, tri in enumerate(self.triangles):
            x = self.x[tri]
            y = self.y[tri]
            zz = z[tri]
            A = np.column_stack([x, y, np.ones(3)])
            try:
                coeffs[i] = np.linalg.solve(A, zz)
            except np.linalg.LinAlgError:
                coeffs[i] = np.linalg.lstsq(A, zz, rcond=None)[0]
        return coeffs


class TriContourGenerator:
    def __init__(self, *args):
        if len(args) < 2:
            raise TypeError("__init__(): incompatible constructor arguments.")
        triangulation, z = args
        self._tri = triangulation
        self._z = np.asarray(z, dtype=np.float64)
        if self._z.ndim != 1 or len(self._z) != len(self._tri.x):
            raise ValueError("z must be a 1D array with the same length as the x and y arrays")

    @staticmethod
    def _interp(p0, p1, z0, z1, level):
        if z1 == z0:
            return None
        t = (level - z0) / (z1 - z0)
        if 0.0 <= t <= 1.0:
            return p0 + t * (p1 - p0)
        return None

    def create_contour(self, level):
        segments = []
        for tri in self._tri._masked_triangles():
            pts = np.column_stack([self._tri.x[tri], self._tri.y[tri]])
            z = self._z[tri]
            intersections = []
            for i0, i1 in ((0, 1), (1, 2), (2, 0)):
                p = self._interp(pts[i0], pts[i1], z[i0], z[i1], level)
                if p is not None:
                    intersections.append(p)
            unique = []
            for p in intersections:
                if not any(np.allclose(p, q) for q in unique):
                    unique.append(p)
            if len(unique) == 2:
                segments.append(np.asarray(unique, dtype=np.float64))
        verts = [_normalize_polyline(path) for path in _merge_segments(segments)]
        codes = []
        for path in verts:
            code = np.full(len(path), Path.LINETO, dtype=np.uint8)
            code[0] = Path.MOVETO
            if len(path) > 2 and _points_close(path[0], path[-1]):
                code[-1] = Path.CLOSEPOLY
            codes.append(code)
        return verts, codes

    @staticmethod
    def _clip_polygon(poly, level, keep_above):
        if not poly:
            return []
        result = []
        prev = poly[-1]
        prev_inside = prev[2] >= level if keep_above else prev[2] <= level
        for curr in poly:
            curr_inside = curr[2] >= level if keep_above else curr[2] <= level
            if curr_inside != prev_inside:
                z0, z1 = prev[2], curr[2]
                if z1 != z0:
                    t = (level - z0) / (z1 - z0)
                    xy = prev[:2] + t * (curr[:2] - prev[:2])
                    result.append(np.array([xy[0], xy[1], level], dtype=np.float64))
            if curr_inside:
                result.append(curr)
            prev = curr
            prev_inside = curr_inside
        return result

    def create_filled_contour(self, lower, upper):
        if lower >= upper:
            raise ValueError("filled contour levels must be increasing")
        polygons = []
        for tri in self._tri._masked_triangles():
            poly = [
                np.array([self._tri.x[idx], self._tri.y[idx], self._z[idx]], dtype=np.float64)
                for idx in tri
            ]
            poly = self._clip_polygon(poly, lower, keep_above=True)
            poly = self._clip_polygon(poly, upper, keep_above=False)
            if len(poly) >= 3:
                xy = np.asarray([p[:2] for p in poly], dtype=np.float64)
                polygons.append(xy)
        verts = []
        codes = []
        for xy in _merge_polygons(polygons):
            xy = _normalize_polygon(xy)
            xy = np.vstack([xy, xy[0]])
            code = np.full(len(xy), Path.LINETO, dtype=np.uint8)
            code[0] = Path.MOVETO
            code[-1] = Path.CLOSEPOLY
            verts.append(xy)
            codes.append(code)
        return verts, codes


class TrapezoidMapTriFinder:
    def __init__(self, *args):
        if len(args) < 1:
            raise TypeError("__init__(): incompatible constructor arguments.")
        triangulation, = args
        self._triangulation = triangulation

    def initialize(self):
        return None

    @staticmethod
    def _point_in_triangle(px, py, tri_xy):
        x1, y1 = tri_xy[0]
        x2, y2 = tri_xy[1]
        x3, y3 = tri_xy[2]
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if det == 0:
            return False
        a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
        b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
        c = 1.0 - a - b
        tol = 1e-12
        return a >= -tol and b >= -tol and c >= -tol

    def find_many(self, x, y):
        xs = np.asarray(x, dtype=np.float64)
        ys = np.asarray(y, dtype=np.float64)
        if xs.shape != ys.shape:
            raise ValueError("x and y must be array-like with same shape")
        result = np.full(xs.shape, -1, dtype=np.int32)

        triangles = self._triangulation.triangles
        if self._triangulation.mask is None:
            active = np.arange(len(triangles), dtype=np.int32)
        else:
            active = np.nonzero(~self._triangulation.mask)[0].astype(np.int32)

        for i, (px, py) in enumerate(zip(xs, ys)):
            for tri_idx in active:
                tri = triangles[tri_idx]
                tri_xy = np.column_stack([
                    self._triangulation.x[tri],
                    self._triangulation.y[tri],
                ])
                if self._point_in_triangle(px, py, tri_xy):
                    result[i] = tri_idx
                    break
        return result

    def get_tree_stats(self):
        return {}

    def print_tree(self):
        return None
