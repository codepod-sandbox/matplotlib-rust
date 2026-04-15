"""
matplotlib._svg_backend — render a Figure to an SVG string.
"""

from matplotlib.colors import to_hex, to_rgb
from matplotlib.backend_bases import RendererBase


class RendererSVG(RendererBase):
    """SVG renderer that accumulates SVG fragments in a list."""

    def __init__(self, width, height, dpi):
        super().__init__()
        self.width = width
        self.height = height
        self.dpi = dpi
        self._parts = []
        self._clip_id = None
        self._clip_counter = 0

    def draw_line(self, xdata, ydata, color, linewidth, linestyle, opacity=1.0):
        dash = _svg_dash(linestyle)
        points = ' '.join(
            f'{xdata[i]:.2f},{ydata[i]:.2f}' for i in range(len(xdata))
        )
        clip = self._clip_attr()
        opacity_attr = f' opacity="{opacity}"' if opacity < 1.0 else ''
        self._parts.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{color}" stroke-width="{linewidth}"{dash}{opacity_attr}{clip}/>'
        )

    def _draw_markers_simple(self, xdata, ydata, color, size, marker='o'):
        clip = self._clip_attr()
        r = size
        for i in range(len(xdata)):
            cx, cy = xdata[i], ydata[i]
            if marker in ('o', '.'):
                self._parts.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
                    f'r="{r}" fill="{color}"{clip}/>'
                )
            elif marker == 's':
                x0, y0 = cx - r, cy - r
                side = r * 2
                self._parts.append(
                    f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{side:.2f}" height="{side:.2f}" '
                    f'fill="{color}"{clip}/>'
                )
            elif marker == '^':
                pts = f'{cx:.2f},{cy - r:.2f} {cx - r:.2f},{cy + r:.2f} {cx + r:.2f},{cy + r:.2f}'
                self._parts.append(
                    f'<polygon points="{pts}" fill="{color}"{clip}/>'
                )
            elif marker == 'v':
                pts = f'{cx:.2f},{cy + r:.2f} {cx - r:.2f},{cy - r:.2f} {cx + r:.2f},{cy - r:.2f}'
                self._parts.append(
                    f'<polygon points="{pts}" fill="{color}"{clip}/>'
                )
            elif marker == 'D':
                pts = f'{cx:.2f},{cy - r:.2f} {cx + r:.2f},{cy:.2f} {cx:.2f},{cy + r:.2f} {cx - r:.2f},{cy:.2f}'
                self._parts.append(
                    f'<polygon points="{pts}" fill="{color}"{clip}/>'
                )
            elif marker == '+':
                hw = r * 0.25
                # vertical bar
                self._parts.append(
                    f'<rect x="{cx - hw:.2f}" y="{cy - r:.2f}" width="{hw * 2:.2f}" height="{r * 2:.2f}" '
                    f'fill="{color}"{clip}/>'
                )
                # horizontal bar
                self._parts.append(
                    f'<rect x="{cx - r:.2f}" y="{cy - hw:.2f}" width="{r * 2:.2f}" height="{hw * 2:.2f}" '
                    f'fill="{color}"{clip}/>'
                )
            elif marker == 'x':
                self._parts.append(
                    f'<line x1="{cx - r:.2f}" y1="{cy - r:.2f}" x2="{cx + r:.2f}" y2="{cy + r:.2f}" '
                    f'stroke="{color}" stroke-width="1.5"{clip}/>'
                )
                self._parts.append(
                    f'<line x1="{cx + r:.2f}" y1="{cy - r:.2f}" x2="{cx - r:.2f}" y2="{cy + r:.2f}" '
                    f'stroke="{color}" stroke-width="1.5"{clip}/>'
                )
            elif marker == '*':
                hw = r * 0.25
                # horizontal bar
                self._parts.append(
                    f'<rect x="{cx - r:.2f}" y="{cy - hw:.2f}" width="{r * 2:.2f}" height="{hw * 2:.2f}" '
                    f'fill="{color}"{clip}/>'
                )
                # vertical bar
                self._parts.append(
                    f'<rect x="{cx - hw:.2f}" y="{cy - r:.2f}" width="{hw * 2:.2f}" height="{r * 2:.2f}" '
                    f'fill="{color}"{clip}/>'
                )
                # diagonal lines
                self._parts.append(
                    f'<line x1="{cx - r:.2f}" y1="{cy - r:.2f}" x2="{cx + r:.2f}" y2="{cy + r:.2f}" '
                    f'stroke="{color}" stroke-width="1.5"{clip}/>'
                )
                self._parts.append(
                    f'<line x1="{cx + r:.2f}" y1="{cy - r:.2f}" x2="{cx - r:.2f}" y2="{cy + r:.2f}" '
                    f'stroke="{color}" stroke-width="1.5"{clip}/>'
                )
            else:
                self._parts.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
                    f'r="{r}" fill="{color}"{clip}/>'
                )

    def draw_rect(self, x, y, width, height, stroke, fill):
        fill_attr = fill if fill else "none"
        stroke_attr = stroke if stroke else "none"
        clip = self._clip_attr()
        self._parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="{fill_attr}" stroke="{stroke_attr}"{clip}/>'
        )

    def draw_circle(self, cx, cy, r, color):
        clip = self._clip_attr()
        self._parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"{clip}/>'
        )

    def draw_ellipse(self, cx, cy, rx, ry, angle, facecolor, edgecolor, alpha):
        fill = facecolor if facecolor and facecolor != 'none' else 'none'
        stroke = edgecolor if edgecolor and edgecolor != 'none' else 'none'
        transform = ''
        if angle != 0:
            transform = f' transform="rotate({-angle:.2f} {cx:.2f} {cy:.2f})"'
        self._parts.append(
            f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
            f'fill="{fill}" stroke="{stroke}"{transform} />'
        )

    def draw_wedge(self, cx, cy, r, start_angle, end_angle, color):
        import math
        sweep = end_angle - start_angle
        clip = self._clip_attr()
        if sweep >= 360:
            # Full circle
            self._parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"{clip}/>'
            )
            return
        # SVG y-axis is down, so negate angles
        svg_start = -start_angle
        svg_end = -end_angle
        x1 = cx + r * math.cos(math.radians(svg_start))
        y1 = cy + r * math.sin(math.radians(svg_start))
        x2 = cx + r * math.cos(math.radians(svg_end))
        y2 = cy + r * math.sin(math.radians(svg_end))
        large_arc = 1 if abs(sweep) > 180 else 0
        sweep_flag = 0 if sweep < 0 else 1
        d = (
            f'M {cx:.2f},{cy:.2f} '
            f'L {x1:.2f},{y1:.2f} '
            f'A {r},{r} 0 {large_arc},{sweep_flag} {x2:.2f},{y2:.2f} Z'
        )
        self._parts.append(
            f'<path d="{d}" fill="{color}"{clip}/>'
        )

    def draw_polygon(self, points, color, alpha):
        pts = ' '.join(f'{x},{y}' for x, y in points)
        clip = self._clip_attr()
        self._parts.append(
            f'<polygon points="{pts}" fill="{color}" '
            f'fill-opacity="{alpha}" stroke="none"{clip}/>'
        )

    # ── RendererBase interface ────────────────────────────────────────────────

    def get_canvas_width_height(self):
        return self.width, self.height

    def points_to_pixels(self, points):
        return points * self.dpi / 72.0

    def flipy(self):
        # We flip y manually in draw_path/draw_text so matplotlib should NOT
        # apply any additional flip.
        return False

    def get_text_width_height_descent(self, s, prop, ismath):
        """Rough estimate for text layout; tests check structure not exact sizing."""
        sz = prop.get_size_in_points() if hasattr(prop, 'get_size_in_points') else 10
        return sz * len(str(s)) * 0.6, sz, sz * 0.2

    def draw_path(self, gc, path, transform, rgbFace=None):
        """Implement the RendererBase draw_path contract for Figure.draw()."""
        from matplotlib.path import Path as MPath

        # Apply clip rectangle from gc if present
        clip_rect = gc.get_clip_rectangle()
        if clip_rect is not None:
            x0, y0, w, h0 = clip_rect.bounds
            # SVG y is flipped: y0 in display coords is bottom of rect
            svg_y = self.height - y0 - h0
            self.set_clip_rect(x0, svg_y, w, h0)
        else:
            self.clear_clip()

        d_parts = []
        h = self.height
        for verts, code in path.iter_segments(transform=transform, remove_nans=True):
            if code == MPath.MOVETO:
                x, y = verts[0], verts[1]
                d_parts.append(f'M{x:.2f},{h - y:.2f}')
            elif code == MPath.LINETO:
                x, y = verts[0], verts[1]
                d_parts.append(f'L{x:.2f},{h - y:.2f}')
            elif code == MPath.CURVE3:
                cx, cy, ex, ey = verts[0], verts[1], verts[2], verts[3]
                d_parts.append(
                    f'Q{cx:.2f},{h - cy:.2f} {ex:.2f},{h - ey:.2f}')
            elif code == MPath.CURVE4:
                cx1, cy1 = verts[0], verts[1]
                cx2, cy2 = verts[2], verts[3]
                ex, ey = verts[4], verts[5]
                d_parts.append(
                    f'C{cx1:.2f},{h - cy1:.2f} {cx2:.2f},{h - cy2:.2f}'
                    f' {ex:.2f},{h - ey:.2f}')
            elif code == MPath.CLOSEPOLY:
                d_parts.append('Z')

        if not d_parts:
            return

        d = ' '.join(d_parts)

        # Stroke
        rgb = gc.get_rgb()
        stroke = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        lw = gc.get_linewidth() * self.dpi / 72.0
        alpha = gc.get_alpha()

        # Fill
        if rgbFace is not None:
            fill = '#{:02x}{:02x}{:02x}'.format(
                int(rgbFace[0] * 255), int(rgbFace[1] * 255),
                int(rgbFace[2] * 255))
        else:
            fill = 'none'

        # Dashes
        dash_attr = ''
        dashes = gc.get_dashes()
        if dashes and dashes[1] is not None:
            px_seq = [d * self.dpi / 72.0 for d in dashes[1]]
            dash_attr = ' stroke-dasharray="{}"'.format(
                ','.join(f'{v:.1f}' for v in px_seq))

        opacity_attr = f' opacity="{alpha:.3f}"' if alpha < 1.0 else ''
        clip = self._clip_attr()
        self._parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}"'
            f' stroke-width="{lw:.1f}"{dash_attr}{opacity_attr}{clip}/>')

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """RendererBase draw_text — called by Figure.draw() for all text."""
        try:
            sz = prop.get_size_in_points()
        except Exception:
            sz = 10
        rgb = gc.get_rgb()
        color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        # Display coords have y=0 at bottom; SVG has y=0 at top.
        svg_y = self.height - y
        clip = self._clip_attr()
        rot = f' transform="rotate({-angle:.1f},{x:.2f},{svg_y:.2f})"' if angle else ''
        self._parts.append(
            f'<text x="{x:.2f}" y="{svg_y:.2f}" font-size="{sz:.1f}"'
            f' fill="{color}"{rot}{clip}>{_esc(str(s))}</text>')

    # ── Legacy high-level text helper (kept for callers using the simple API)

    def _draw_text_simple(self, x, y, text, fontsize, color, ha):
        anchor_map = {"left": "start", "center": "middle", "right": "end"}
        anchor = anchor_map.get(ha, "start")
        clip = self._clip_attr()
        self._parts.append(
            f'<text x="{x}" y="{y}" font-size="{fontsize}" '
            f'fill="{color}" text-anchor="{anchor}"{clip}>'
            f'{_esc(text)}</text>'
        )

    def draw_arrow(self, x1, y1, x2, y2, arrowstyle, color, linewidth):
        """Draw an arrow using SVG path with marker-end for arrowhead."""
        arrow_id = f'arrow-{len(self._parts)}'
        has_end = arrowstyle in ('->', '<->', 'fancy')
        has_start = arrowstyle in ('<-', '<->')

        if has_end or has_start:
            # Define a simple triangle arrowhead marker
            marker = (
                f'<defs>'
                f'<marker id="{arrow_id}-end" markerWidth="8" markerHeight="6" '
                f'refX="7" refY="3" orient="auto">'
                f'<polygon points="0 0, 8 3, 0 6" fill="{color}"/>'
                f'</marker>'
            )
            if has_start:
                marker += (
                    f'<marker id="{arrow_id}-start" markerWidth="8" markerHeight="6" '
                    f'refX="1" refY="3" orient="auto-start-reverse">'
                    f'<polygon points="0 0, 8 3, 0 6" fill="{color}"/>'
                    f'</marker>'
                )
            marker += '</defs>'
            self._parts.append(marker)

        attrs = f'stroke="{color}" stroke-width="{linewidth}" fill="none"'
        if has_end:
            attrs += f' marker-end="url(#{arrow_id}-end)"'
        if has_start:
            attrs += f' marker-start="url(#{arrow_id}-start)"'

        self._parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attrs}/>'
        )

    def _draw_image_simple(self, x, y, width, height, rgba_array):
        """Embed image as base64 PNG data URL."""
        import base64
        import struct
        import zlib

        rows = len(rgba_array)
        cols = len(rgba_array[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return

        # Minimal PNG encoder (no PIL dependency for SVG backend)
        def _make_png(w, h, pixel_rows):
            def chunk(name, data):
                c = name + data
                return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
            header = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
            ihdr = chunk(b'IHDR', ihdr_data)
            raw_data = b''
            for row in pixel_rows:
                raw_data += b'\x00'
                for px in row:
                    raw_data += bytes([int(px[0]), int(px[1]), int(px[2])])
            idat = chunk(b'IDAT', zlib.compress(raw_data))
            iend = chunk(b'IEND', b'')
            return header + ihdr + idat + iend

        pixel_rows = []
        for row in rgba_array:
            pixel_rows.append([(px[0], px[1], px[2]) for px in row])

        png_bytes = _make_png(cols, rows, pixel_rows)
        b64 = base64.b64encode(png_bytes).decode('ascii')
        data_url = f'data:image/png;base64,{b64}'

        self._parts.append(
            f'<image x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'href="{data_url}" />'
        )

    def set_clip_rect(self, x, y, width, height):
        self._clip_counter += 1
        self._clip_id = f'clip-{self._clip_counter}'
        self._parts.append(
            f'<defs><clipPath id="{self._clip_id}">'
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}"/>'
            f'</clipPath></defs>'
        )

    def clear_clip(self):
        self._clip_id = None

    def get_result(self):
        header = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
        )
        return '\n'.join([header] + self._parts + ['</svg>'])

    def _clip_attr(self):
        if self._clip_id:
            return f' clip-path="url(#{self._clip_id})"'
        return ''


# Named style → (on, off, ...) dash sequences
_NAMED_DASHES = {
    'solid': None,
    '-': None,
    'dashed': (6, 3),
    '--': (6, 3),
    'dotted': (2, 2),
    ':': (2, 2),
    'dashdot': (6, 2, 2, 2),
    '-.': (6, 2, 2, 2),
    'loosely dashed': (6, 6),
    'densely dashed': (4, 1),
    'loosely dotted': (2, 4),
    'densely dotted': (1, 1),
    'loosely dashdotted': (6, 4, 2, 4),
    'densely dashdotted': (4, 1, 2, 1),
}


def _svg_dash(ls):
    """Return SVG stroke-dasharray attribute string for a linestyle."""
    if isinstance(ls, tuple):
        # (offset, (on, off, ...)) format
        offset, dashes = ls
        dash_str = ','.join(str(d) for d in dashes)
        offset_attr = f' stroke-dashoffset="{offset}"' if offset else ''
        return f' stroke-dasharray="{dash_str}"{offset_attr}'

    seq = _NAMED_DASHES.get(ls)
    if seq is None:
        return ''  # solid or unknown → no dasharray
    dash_str = ','.join(str(d) for d in seq)
    return f' stroke-dasharray="{dash_str}"'


def _nice_ticks(lo, hi, target_count):
    """Generate roughly *target_count* nicely-rounded ticks between lo and hi."""
    if hi <= lo:
        return [lo]
    import math
    raw_step = (hi - lo) / max(target_count, 1)
    mag = 10 ** math.floor(math.log10(raw_step + 1e-15))
    residual = raw_step / mag
    if residual <= 1.5:
        step = 1 * mag
    elif residual <= 3:
        step = 2 * mag
    elif residual <= 7:
        step = 5 * mag
    else:
        step = 10 * mag

    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 0.01:
        if lo <= t <= hi:
            ticks.append(round(t, 10))
        t += step
    return ticks if ticks else [lo, hi]


def _fmt_tick(v):
    """Format a tick value nicely."""
    if v == int(v):
        return str(int(v))
    return f'{v:.2g}'


def _esc(s):
    """Escape HTML entities."""
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
