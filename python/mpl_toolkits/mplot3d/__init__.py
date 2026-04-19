from .axes3d import Axes3D

try:
    from matplotlib import projections as _projections
except Exception:
    _projections = None
else:
    if hasattr(_projections, "register_projection"):
        _projections.register_projection(Axes3D)

__all__ = ['Axes3D']
