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

    def draw_line(self, xdata, ydata, color, linewidth, linestyle, opacity=1.0):
        col = _to_rgb_255(color)
        lw = max(1, int(linewidth))
        for i in range(len(xdata) - 1):
            self._draw.line(
                [(int(xdata[i]), int(ydata[i])),
                 (int(xdata[i + 1]), int(ydata[i + 1]))],
                fill=col, width=lw
            )

    def draw_markers(self, xdata, ydata, color, size, marker='o'):
        col = _to_rgb_255(color)
        r = max(1, int(size))
        for i in range(len(xdata)):
            cx, cy = int(xdata[i]), int(ydata[i])
            if marker in ('o', '.'):
                self._draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=col)
            elif marker == 's':
                pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
                self._draw.polygon(pts, fill=col)
            elif marker == '^':
                pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
                self._draw.polygon(pts, fill=col)
            elif marker == 'v':
                pts = [(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)]
                self._draw.polygon(pts, fill=col)
            elif marker == 'D':
                pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
                self._draw.polygon(pts, fill=col)
            elif marker == '+':
                hw = max(1, r // 4)
                pts_v = [(cx - hw, cy - r), (cx + hw, cy - r), (cx + hw, cy + r), (cx - hw, cy + r)]
                pts_h = [(cx - r, cy - hw), (cx + r, cy - hw), (cx + r, cy + hw), (cx - r, cy + hw)]
                self._draw.polygon(pts_v, fill=col)
                self._draw.polygon(pts_h, fill=col)
            elif marker == 'x':
                self._draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=col, width=2)
                self._draw.line([(cx + r, cy - r), (cx - r, cy + r)], fill=col, width=2)
            elif marker == '*':
                hw = max(1, r // 4)
                pts_v = [(cx - hw, cy - r), (cx + hw, cy - r), (cx + hw, cy + r), (cx - hw, cy + r)]
                pts_h = [(cx - r, cy - hw), (cx + r, cy - hw), (cx + r, cy + hw), (cx - r, cy + hw)]
                self._draw.polygon(pts_v, fill=col)
                self._draw.polygon(pts_h, fill=col)
                self._draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=col, width=2)
                self._draw.line([(cx + r, cy - r), (cx - r, cy + r)], fill=col, width=2)
            else:
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

    def draw_ellipse(self, cx, cy, rx, ry, angle, facecolor, edgecolor, alpha):
        import math
        fc = _to_rgb_255(facecolor) if facecolor and facecolor != 'none' else None
        ec = _to_rgb_255(edgecolor) if edgecolor and edgecolor != 'none' else None

        if angle == 0:
            # Use PIL's native ellipse for non-rotated ellipses
            bbox = [(int(cx - rx), int(cy - ry)), (int(cx + rx), int(cy + ry))]
            if fc is not None:
                self._draw.ellipse(bbox, fill=fc)
            if ec is not None:
                self._draw.ellipse(bbox, outline=ec)
        else:
            # Approximate rotated ellipse with 36-point polygon
            pts = []
            angle_rad = math.radians(angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            for i in range(36):
                t = 2 * math.pi * i / 36
                x = rx * math.cos(t)
                y = ry * math.sin(t)
                px = cx + x * cos_a - y * sin_a
                py = cy - (x * sin_a + y * cos_a)  # negate: screen y-down
                pts.append((int(px), int(py)))
            if fc is not None:
                self._draw.polygon(pts, fill=fc)
            if ec is not None:
                for i in range(len(pts)):
                    self._draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=ec, width=1)

    def draw_wedge(self, cx, cy, r, start_angle, end_angle, color):
        import math
        col = _to_rgb_255(color)
        sweep = end_angle - start_angle
        if abs(sweep) >= 360:
            self.draw_circle(cx, cy, r, color)
            return
        n_segments = max(16, int(abs(sweep) / 2))
        pts = [(int(cx), int(cy))]
        for i in range(n_segments + 1):
            angle_deg = start_angle + sweep * i / n_segments
            angle_rad = math.radians(angle_deg)
            x = cx + r * math.cos(angle_rad)
            y = cy - r * math.sin(angle_rad)  # negate: screen y-axis is down
            pts.append((int(x), int(y)))
        self._draw.polygon(pts, fill=col)

    def draw_polygon(self, points, color, alpha):
        col = _to_rgb_255(color)
        pts = [(int(x), int(y)) for x, y in points]
        if len(pts) >= 3:
            self._draw.polygon(pts, fill=col)

    def draw_text(self, x, y, text, fontsize, color, ha):
        col = _to_rgb_255(color)
        self._draw.text((int(x), int(y)), str(text), fill=col)

    def draw_arrow(self, x1, y1, x2, y2, arrowstyle, color, linewidth):
        """Draw arrow: line + arrowhead polygon."""
        import math
        col = _to_rgb_255(color)
        lw = max(1, int(linewidth))

        # Draw the shaft
        self._draw.line([(int(x1), int(y1)), (int(x2), int(y2))],
                        fill=col, width=lw)

        has_end = arrowstyle in ('->', '<->', 'fancy')
        has_start = arrowstyle in ('<-', '<->')

        head_len = max(8, lw * 4)
        head_w = max(4, lw * 2)

        def _arrowhead(tx, ty, fx, fy):
            """Draw a filled triangle arrowhead at (tx,ty) pointing away from (fx,fy)."""
            dx = tx - fx
            dy = ty - fy
            length = math.hypot(dx, dy)
            if length < 1e-6:
                return
            ux, uy = dx / length, dy / length
            px, py = -uy, ux  # perpendicular
            tip = (int(tx), int(ty))
            base1 = (int(tx - ux * head_len + px * head_w),
                     int(ty - uy * head_len + py * head_w))
            base2 = (int(tx - ux * head_len - px * head_w),
                     int(ty - uy * head_len - py * head_w))
            self._draw.line([tip, base1], fill=col, width=lw)
            self._draw.line([tip, base2], fill=col, width=lw)
            self._draw.line([base1, base2], fill=col, width=lw)

        if has_end:
            _arrowhead(x2, y2, x1, y1)
        if has_start:
            _arrowhead(x1, y1, x2, y2)

    def draw_image(self, x, y, width, height, rgba_array):
        """Draw an image into the canvas using nearest-neighbor scaling."""
        rows = len(rgba_array)
        cols = len(rgba_array[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return

        disp_w = max(1, int(width))
        disp_h = max(1, int(height))
        ox = int(x)
        oy = int(y)

        img_w = self._img.width
        img_h = self._img.height

        # Nearest-neighbor scale from (cols, rows) to (disp_w, disp_h)
        for dy in range(disp_h):
            src_row = int(dy * rows / disp_h)
            if src_row >= rows:
                src_row = rows - 1
            row = rgba_array[src_row]
            for dx in range(disp_w):
                src_col = int(dx * cols / disp_w)
                if src_col >= cols:
                    src_col = cols - 1
                px = row[src_col]
                px_x = ox + dx
                px_y = oy + dy
                if 0 <= px_x < img_w and 0 <= px_y < img_h:
                    self._img.putpixel((px_x, px_y), (int(px[0]), int(px[1]), int(px[2])))

    def set_clip_rect(self, x, y, width, height):
        # PIL doesn't support clipping natively; store for potential future use
        self._clip = (x, y, width, height)

    def clear_clip(self):
        self._clip = None

    def get_result(self):
        buf = io.BytesIO()
        self._img.save(buf, format='PNG')
        return buf.getvalue()


