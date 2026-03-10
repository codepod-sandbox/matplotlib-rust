"""
matplotlib.backend_bases — abstract renderer and layout helpers.
"""


class AxesLayout:
    """Stores plot area geometry and provides data-to-pixel transforms."""

    def __init__(self, plot_x, plot_y, plot_w, plot_h,
                 xmin, xmax, ymin, ymax):
        self.plot_x = plot_x
        self.plot_y = plot_y
        self.plot_w = plot_w
        self.plot_h = plot_h
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def sx(self, v):
        """Map data x to pixel x."""
        return self.plot_x + (v - self.xmin) / (self.xmax - self.xmin) * self.plot_w

    def sy(self, v):
        """Map data y to pixel y (inverted: ymax at top, ymin at bottom)."""
        return self.plot_y + self.plot_h - (v - self.ymin) / (self.ymax - self.ymin) * self.plot_h


class RendererBase:
    """Abstract base class for renderers."""

    def __init__(self, width, height, dpi):
        self.width = width
        self.height = height
        self.dpi = dpi

    def draw_line(self, xdata, ydata, color, linewidth, linestyle):
        raise NotImplementedError

    def draw_markers(self, xdata, ydata, color, size):
        raise NotImplementedError

    def draw_rect(self, x, y, width, height, stroke, fill):
        raise NotImplementedError

    def draw_circle(self, cx, cy, r, color):
        raise NotImplementedError

    def draw_wedge(self, cx, cy, r, start_angle, end_angle, color):
        raise NotImplementedError

    def draw_polygon(self, points, color, alpha):
        raise NotImplementedError

    def draw_text(self, x, y, text, fontsize, color, ha):
        raise NotImplementedError

    def set_clip_rect(self, x, y, width, height):
        raise NotImplementedError

    def clear_clip(self):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError
