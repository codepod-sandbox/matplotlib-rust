"""
matplotlib._pil_backend — render a Figure to PNG bytes via PIL.
"""

import io
from matplotlib.colors import to_rgb as _to_rgb_float
from matplotlib.backend_bases import RendererBase


def _to_rgb_255(color):
    """Convert a colour to an (r, g, b) int tuple (0-255) for PIL."""
    r, g, b = _to_rgb_float(color)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


class RendererPIL(RendererBase):
    """PIL/Pillow renderer that draws onto an RGB image."""

    def __init__(self, width, height, dpi):
        super().__init__(width, height, dpi)
        from PIL import Image, ImageDraw
        self._img = Image.new('RGB', (width, height), (255, 255, 255))
        self._draw = ImageDraw.Draw(self._img)

    def draw_line(self, xdata, ydata, color, linewidth, linestyle):
        col = _to_rgb_255(color)
        lw = max(1, int(linewidth))
        for i in range(len(xdata) - 1):
            self._draw.line(
                [(int(xdata[i]), int(ydata[i])),
                 (int(xdata[i + 1]), int(ydata[i + 1]))],
                fill=col, width=lw
            )

    def draw_markers(self, xdata, ydata, color, size):
        col = _to_rgb_255(color)
        r = max(1, int(size))
        for i in range(len(xdata)):
            cx, cy = int(xdata[i]), int(ydata[i])
            self._draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=col)

    def draw_rect(self, x, y, width, height, stroke, fill):
        fill_col = _to_rgb_255(fill) if fill else None
        outline_col = _to_rgb_255(stroke) if stroke else None
        self._draw.rectangle(
            [(int(x), int(y)), (int(x + width), int(y + height))],
            fill=fill_col, outline=outline_col
        )

    def draw_circle(self, cx, cy, r, color):
        col = _to_rgb_255(color)
        self._draw.ellipse(
            [(int(cx - r), int(cy - r)), (int(cx + r), int(cy + r))],
            fill=col
        )

    def draw_polygon(self, points, color, alpha):
        col = _to_rgb_255(color)
        pts = [(int(x), int(y)) for x, y in points]
        if len(pts) >= 3:
            # PIL in RustPython lacks polygon(); approximate with lines
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                self._draw.line([p1, p2], fill=col, width=1)

    def draw_text(self, x, y, text, fontsize, color, ha):
        col = _to_rgb_255(color)
        self._draw.text((int(x), int(y)), str(text), fill=col)

    def set_clip_rect(self, x, y, width, height):
        # PIL doesn't support clipping natively; store for potential future use
        self._clip = (x, y, width, height)

    def clear_clip(self):
        self._clip = None

    def get_result(self):
        buf = io.BytesIO()
        self._img.save(buf, format='PNG')
        return buf.getvalue()


