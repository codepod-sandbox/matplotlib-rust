"""
matplotlib._svg_backend — render a Figure to an SVG string.
"""

from matplotlib.colors import to_hex, to_rgb
from matplotlib.backend_bases import RendererBase


class RendererSVG(RendererBase):
    """SVG renderer that accumulates SVG fragments in a list."""

    def __init__(self, width, height, dpi):
        super().__init__(width, height, dpi)
        self._parts = []
        self._clip_id = None
        self._clip_counter = 0

    def draw_line(self, xdata, ydata, color, linewidth, linestyle):
        dash = _svg_dash(linestyle)
        points = ' '.join(
            f'{xdata[i]:.2f},{ydata[i]:.2f}' for i in range(len(xdata))
        )
        clip = self._clip_attr()
        self._parts.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{color}" stroke-width="{linewidth}"{dash}{clip}/>'
        )

    def draw_markers(self, xdata, ydata, color, size):
        clip = self._clip_attr()
        for i in range(len(xdata)):
            self._parts.append(
                f'<circle cx="{xdata[i]:.2f}" cy="{ydata[i]:.2f}" '
                f'r="{size}" fill="{color}"{clip}/>'
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
        large_arc = 1 if sweep > 180 else 0
        sweep_flag = 1  # CW in SVG screen coords = CCW in math coords
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

    def draw_text(self, x, y, text, fontsize, color, ha):
        anchor_map = {"left": "start", "center": "middle", "right": "end"}
        anchor = anchor_map.get(ha, "start")
        clip = self._clip_attr()
        self._parts.append(
            f'<text x="{x}" y="{y}" font-size="{fontsize}" '
            f'fill="{color}" text-anchor="{anchor}"{clip}>'
            f'{_esc(text)}</text>'
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


def _svg_dash(ls):
    """Return SVG stroke-dasharray attribute string for a linestyle."""
    if ls == '--' or ls == 'dashed':
        return ' stroke-dasharray="6,3"'
    elif ls == ':' or ls == 'dotted':
        return ' stroke-dasharray="2,2"'
    elif ls == '-.' or ls == 'dashdot':
        return ' stroke-dasharray="6,2,2,2"'
    return ''


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
