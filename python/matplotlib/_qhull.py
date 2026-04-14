"""Stub for matplotlib._qhull.

Replaced by crates/matplotlib-qhull (spade crate) in Phase 3.
"""


class Delaunay:
    """Stub Delaunay triangulation."""
    def __init__(self, points):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")

    @property
    def simplices(self):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")

    @property
    def neighbors(self):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")

    def find_simplex(self, xi):
        raise NotImplementedError("_qhull not yet implemented (Phase 3)")
