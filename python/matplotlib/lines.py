"""matplotlib.lines --- Line2D artist for line/marker plots."""

from matplotlib.artist import Artist
from matplotlib.colors import to_hex, to_rgba


class Line2D(Artist):
    """A line and/or marker artist for 2-D data."""

    zorder = 2

    def __init__(self, xdata, ydata, color=None, linewidth=None,
                 linestyle=None, marker=None, label=None, **kwargs):
        super().__init__()

        self._xdata = list(xdata)
        self._ydata = list(ydata)

        self._color = color if color is not None else 'C0'
        self._linewidth = linewidth if linewidth is not None else 1.5
        self._linestyle = linestyle if linestyle is not None else '-'
        self._marker = marker if marker is not None else 'None'

        if label is not None:
            self.set_label(label)

        # Extra kwargs
        self._markersize = kwargs.get('markersize', kwargs.get('ms', 6.0))
        self._fillstyle = kwargs.get('fillstyle', 'full')
        self._drawstyle = kwargs.get('drawstyle', 'default')
        self._markeredgecolor = kwargs.get('markeredgecolor', kwargs.get('mec', None))
        self._markerfacecolor = kwargs.get('markerfacecolor', kwargs.get('mfc', None))
        self._markeredgewidth = kwargs.get('markeredgewidth', kwargs.get('mew', None))
        self._markevery = kwargs.get('markevery', None)
        self._antialiased = kwargs.get('antialiased', kwargs.get('aa', True))
        self._solid_capstyle = kwargs.get('solid_capstyle', None)
        self._solid_joinstyle = kwargs.get('solid_joinstyle', None)
        self._dash_capstyle = kwargs.get('dash_capstyle', None)
        self._dash_joinstyle = kwargs.get('dash_joinstyle', None)

    # --- xdata ---
    def get_xdata(self):
        return list(self._xdata)

    def set_xdata(self, x):
        self._xdata = list(x)

    # --- ydata ---
    def get_ydata(self):
        return list(self._ydata)

    def set_ydata(self, y):
        self._ydata = list(y)

    # --- data ---
    def get_data(self):
        return self.get_xdata(), self.get_ydata()

    def set_data(self, x, y):
        self.set_xdata(x)
        self.set_ydata(y)

    # --- color ---
    def get_color(self):
        return self._color

    def set_color(self, color):
        # Validate that color is a recognized color spec
        to_rgba(color)
        self._color = color

    set_c = set_color

    # --- linewidth ---
    def get_linewidth(self):
        return self._linewidth

    def set_linewidth(self, w):
        self._linewidth = w

    set_lw = set_linewidth

    # --- linestyle ---
    _valid_linestyles = {'-', 'solid', '--', 'dashed', '-.', 'dashdot',
                         ':', 'dotted', 'None', 'none', ''}

    def get_linestyle(self):
        return self._linestyle

    def set_linestyle(self, ls):
        if ls not in self._valid_linestyles:
            raise ValueError(
                f"'{ls}' is not a valid value for linestyle; supported "
                f"values are '-', '--', '-.', ':', 'None', 'none', '', "
                f"'solid', 'dashed', 'dashdot', 'dotted'")
        self._linestyle = ls

    set_ls = set_linestyle

    # --- marker ---
    def get_marker(self):
        return self._marker

    def set_marker(self, marker):
        self._marker = marker

    # --- markersize ---
    def get_markersize(self):
        return self._markersize

    def set_markersize(self, sz):
        self._markersize = sz

    set_ms = set_markersize

    # --- fillstyle ---
    def get_fillstyle(self):
        return self._fillstyle

    def set_fillstyle(self, fs):
        self._fillstyle = fs

    # --- markeredgecolor ---
    def get_markeredgecolor(self):
        if self._markeredgecolor is not None:
            return self._markeredgecolor
        return self._color

    def set_markeredgecolor(self, color):
        self._markeredgecolor = color

    set_mec = set_markeredgecolor

    # --- markerfacecolor ---
    def get_markerfacecolor(self):
        if self._markerfacecolor is not None:
            return self._markerfacecolor
        return self._color

    def set_markerfacecolor(self, color):
        self._markerfacecolor = color

    set_mfc = set_markerfacecolor

    # --- markerfacecoloralt ---
    def get_markerfacecoloralt(self):
        return getattr(self, '_markerfacecoloralt', 'none')

    def set_markerfacecoloralt(self, color):
        self._markerfacecoloralt = color

    set_mfcalt = set_markerfacecoloralt

    # --- markeredgewidth ---
    def get_markeredgewidth(self):
        if self._markeredgewidth is not None:
            return self._markeredgewidth
        return 1.0

    def set_markeredgewidth(self, w):
        self._markeredgewidth = w

    set_mew = set_markeredgewidth

    # --- markevery ---
    def get_markevery(self):
        return getattr(self, '_markevery', None)

    def set_markevery(self, every):
        self._markevery = every

    # --- antialiased ---
    def get_antialiased(self):
        return getattr(self, '_antialiased', True)

    def set_antialiased(self, b):
        self._antialiased = b

    set_aa = set_antialiased

    # --- solid_capstyle ---
    def get_solid_capstyle(self):
        return getattr(self, '_solid_capstyle', None)

    def set_solid_capstyle(self, s):
        self._solid_capstyle = s

    # --- solid_joinstyle ---
    def get_solid_joinstyle(self):
        return getattr(self, '_solid_joinstyle', None)

    def set_solid_joinstyle(self, s):
        self._solid_joinstyle = s

    # --- dash_capstyle ---
    def get_dash_capstyle(self):
        return getattr(self, '_dash_capstyle', None)

    def set_dash_capstyle(self, s):
        self._dash_capstyle = s

    # --- dash_joinstyle ---
    def get_dash_joinstyle(self):
        return getattr(self, '_dash_joinstyle', None)

    def set_dash_joinstyle(self, s):
        self._dash_joinstyle = s

    # --- path ---
    def get_path(self):
        """Return a path-like object with vertices."""
        return _LinePath(list(zip(self._xdata, self._ydata)))

    # --- xydata ---
    def get_xydata(self):
        """Return data as list of (x, y) tuples."""
        return list(zip(self._xdata, self._ydata))

    # --- drawstyle ---
    _valid_drawstyles = {'default', 'steps', 'steps-pre', 'steps-mid',
                         'steps-post'}

    def get_drawstyle(self):
        return self._drawstyle

    def set_drawstyle(self, ds):
        if ds not in self._valid_drawstyles:
            raise ValueError(
                f"'{ds}' is not a valid value for drawstyle; supported "
                f"values are 'default', 'steps', 'steps-pre', "
                f"'steps-mid', 'steps-post'")
        self._drawstyle = ds

    # --- draw (new renderer architecture) ---
    def draw(self, renderer, layout):
        """Draw this line onto the renderer."""
        if not self.get_visible():
            return
        x_px = [layout.sx(v) for v in self._xdata]
        y_px = [layout.sy(v) for v in self._ydata]
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        color = to_hex(self._color)

        # Draw line (skip if no-line style)
        if (self._linestyle not in ('None', 'none', '')
                and len(x_px) >= 2):
            renderer.draw_line(x_px, y_px, color,
                               float(self._linewidth), self._linestyle)

        # Draw markers
        if self._marker and self._marker not in ('None', 'none', ''):
            renderer.draw_markers(x_px, y_px, color,
                                  float(self._markersize))

    # --- class-level valid linestyles dict (upstream compat) ---
    lineStyles = {
        '-': '_draw_solid',
        '--': '_draw_dashed',
        '-.': '_draw_dash_dot',
        ':': '_draw_dotted',
        'None': '_draw_nothing',
        'none': '_draw_nothing',
        ' ': '_draw_nothing',
        '': '_draw_nothing',
    }

    markers = {
        '.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down',
        '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right',
        '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right',
        '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star',
        'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x',
        'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline',
        'P': 'plus_filled', 'X': 'x_filled',
        'None': 'nothing', 'none': 'nothing', ' ': 'nothing', '': 'nothing',
    }

    filled_markers = frozenset({
        'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H',
        'D', 'd', 'P', 'X',
    })

    zorder = 2


class _LinePath:
    """Lightweight path-like object returned by Line2D.get_path()."""

    def __init__(self, vertices):
        self.vertices = vertices

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)


# Module-level aliases for upstream compatibility
lineStyles = Line2D.lineStyles
lineMarkers = Line2D.markers

