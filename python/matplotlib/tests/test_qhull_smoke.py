"""Smoke tests for the Rust matplotlib._qhull extension (Phase 3B).

Pins the contract that `matplotlib.tri._triangulation` relies on:
`delaunay(x, y, verbose)` returns `(triangles, neighbors)` as (ntri, 3)
int32 arrays with valid indices and the correct neighbor convention.
"""

import numpy as np
import pytest

from matplotlib import _qhull


def test_delaunay_function_exists():
    assert hasattr(_qhull, "delaunay"), "_qhull.delaunay not found"
    assert callable(_qhull.delaunay)


def test_delaunay_square_shape():
    # 4 corner points → 2 triangles, so (2, 3) arrays.
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tri, nbr = _qhull.delaunay(x, y, False)
    assert tri.shape == (2, 3), f"triangles shape {tri.shape!r} != (2, 3)"
    assert nbr.shape == (2, 3), f"neighbors shape {nbr.shape!r} != (2, 3)"


def test_delaunay_dtype():
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    tri, nbr = _qhull.delaunay(x, y)
    assert tri.dtype == np.int32, f"triangles dtype {tri.dtype!r} != int32"
    assert nbr.dtype == np.int32, f"neighbors dtype {nbr.dtype!r} != int32"


def test_delaunay_indices_in_range():
    # Every triangle index must be a valid point index.
    rng = np.random.default_rng(42)
    n = 20
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    tri, _ = _qhull.delaunay(x, y)
    assert tri.min() >= 0
    assert tri.max() < n


def test_delaunay_no_degenerate_triangles():
    # No triangle should have a repeated vertex index.
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, 30)
    y = rng.uniform(0, 1, 30)
    tri, _ = _qhull.delaunay(x, y)
    for row in tri:
        assert len(set(row.tolist())) == 3, f"degenerate triangle: {row}"


def test_delaunay_neighbors_in_range():
    # Neighbor values must be -1 (boundary) or a valid triangle index.
    rng = np.random.default_rng(13)
    x = rng.uniform(0, 5, 15)
    y = rng.uniform(0, 5, 15)
    tri, nbr = _qhull.delaunay(x, y)
    ntri = tri.shape[0]
    assert ((nbr >= -1) & (nbr < ntri)).all(), "neighbors out of range"


def test_delaunay_neighbor_symmetry():
    # If triangle s is a neighbor of triangle t, then t must be a neighbor
    # of s — adjacency must be symmetric.
    rng = np.random.default_rng(99)
    x = rng.uniform(0, 4, 12)
    y = rng.uniform(0, 4, 12)
    tri, nbr = _qhull.delaunay(x, y)
    ntri = tri.shape[0]
    for t in range(ntri):
        for j in range(3):
            s = nbr[t, j]
            if s == -1:
                continue
            # s must list t as one of its neighbors.
            assert t in nbr[s], (
                f"asymmetric neighbors: triangle {t} lists {s} as neighbor "
                f"but {s}'s neighbors are {nbr[s].tolist()}"
            )


def test_delaunay_neighbor_edge_convention():
    # Pin the exact Matplotlib contract (_triangulation.py:212-214):
    #   neighbors[t, j] = triangle across edge from triangles[t, j]
    #                     to triangles[t, (j+1)%3]
    #
    # For each neighbor pair (t, j) → s, the shared edge must be the
    # edge {triangles[t, j], triangles[t, (j+1)%3]}.  The neighbor s
    # must contain both of those vertex indices (possibly in reverse order).
    # This test would fail with the "opposite vertex" slot mapping because
    # that maps slot j to the edge opposite vertex j (a different edge).
    rng = np.random.default_rng(17)
    x = rng.uniform(0, 3, 14)
    y = rng.uniform(0, 3, 14)
    tri, nbr = _qhull.delaunay(x, y)
    ntri = tri.shape[0]
    for t in range(ntri):
        for j in range(3):
            s = int(nbr[t, j])
            if s == -1:
                continue
            # Edge on triangle t that should be shared with s.
            va = int(tri[t, j])
            vb = int(tri[t, (j + 1) % 3])
            shared = {va, vb}
            neighbor_verts = set(tri[s].tolist())
            assert shared.issubset(neighbor_verts), (
                f"neighbors[{t}, {j}]={s}: expected shared edge "
                f"{{{va}, {vb}}} but triangle {s} has vertices "
                f"{neighbor_verts}"
            )


def test_delaunay_triangle_single():
    # Exactly 3 points → exactly 1 triangle.
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    tri, nbr = _qhull.delaunay(x, y)
    assert tri.shape == (1, 3)
    assert nbr.shape == (1, 3)
    # Single triangle has no neighbors.
    assert (nbr == -1).all()


def test_delaunay_verbose_flag_accepted():
    # verbose is passed as sys.flags.verbose (int 0/1) in production.
    # Both int and bool must be accepted; it's a no-op in the Rust impl.
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    tri, _ = _qhull.delaunay(x, y, 1)
    assert tri.shape[1] == 3


def test_delaunay_triangulation_integration():
    # Verify that matplotlib.tri.Triangulation.__init__ completes successfully
    # using our _qhull extension and that basic attributes are correct.
    #
    # Note: the `.neighbors` property triggers get_cpp_triangulation() which
    # requires `_tri` (Phase 4), so we check `_neighbors` directly instead.
    from matplotlib.tri._triangulation import Triangulation

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 25)
    y = rng.uniform(0, 1, 25)

    # Capture the delaunay result before Triangulation.__init__ discards it
    # via set_mask(None) resetting _neighbors.
    from matplotlib import _qhull
    tri_arr, nbr_arr = _qhull.delaunay(x, y, 0)

    triang = Triangulation(x, y)
    assert triang.is_delaunay
    assert triang.triangles.shape[1] == 3
    # Confirm the returned neighbors array is valid (covers the call contract).
    assert nbr_arr.shape == tri_arr.shape
    assert nbr_arr.dtype == np.int32
