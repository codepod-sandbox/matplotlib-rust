"""Stub for contourpy — contour line/polygon generation.

Replaced by actual contourpy or crates/matplotlib-contour in Phase 3.
See https://contourpy.readthedocs.io/
"""

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


def contour_generator(x=None, y=None, z=None, name="serial",
                      corner_mask=None, line_type=None, fill_type=None,
                      chunk_size=None, chunk_count=None,
                      total_chunk_count=None, quad_as_tri=False,
                      z_interp=None, thread_count=0):
    """Stub — raises NotImplementedError (Phase 3)."""
    raise NotImplementedError("contourpy not yet implemented (Phase 3)")
