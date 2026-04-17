"""Minimal matplotlib.style package compatibility layer."""

from .core import STYLE_BLACKLIST, available, context, use

__all__ = ["STYLE_BLACKLIST", "available", "context", "use"]
