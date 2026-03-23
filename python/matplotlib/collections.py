"""matplotlib.collections --- Collection and PathCollection artists."""

from matplotlib.artist import Artist
from matplotlib.colors import to_hex, to_rgba


class Collection(Artist):
    """Base class for collections of similar artists."""

    zorder = 1

    def __init__(self, **kwargs):
        super().__init__()
        self._edgecolors = []
        self._facecolors = []
        self._linewidths = [1.0]
        self._linestyles = ['solid']
        self._offsets = []
        if kwargs:
            self.set(**kwargs)

    def get_edgecolors(self):
        return list(self._edgecolors)

    def get_edgecolor(self):
        return list(self._edgecolors)

    def set_edgecolor(self, colors):
        if colors is None:
            self._edgecolors = []
        elif isinstance(colors, str):
            self._edgecolors = [colors]
        else:
            self._edgecolors = list(colors)

    def set_edgecolors(self, colors):
        self.set_edgecolor(colors)

    def get_facecolors(self):
        return list(self._facecolors)

    def get_facecolor(self):
        return list(self._facecolors)

    def set_facecolor(self, colors):
        if colors is None:
            self._facecolors = []
        elif isinstance(colors, str):
            self._facecolors = [colors]
        else:
            self._facecolors = list(colors)

    def set_facecolors(self, colors):
        self.set_facecolor(colors)

    def get_linewidths(self):
        return list(self._linewidths)

    def get_linewidth(self):
        return list(self._linewidths)

    def set_linewidth(self, lw):
        if hasattr(lw, '__iter__'):
            self._linewidths = list(lw)
        else:
            self._linewidths = [lw]

    def set_linewidths(self, lw):
        self.set_linewidth(lw)

    def get_linestyles(self):
        return list(self._linestyles)

    def get_linestyle(self):
        return list(self._linestyles)

    def set_linestyle(self, ls):
        if isinstance(ls, str):
            self._linestyles = [ls]
        elif hasattr(ls, '__iter__'):
            self._linestyles = list(ls)
        else:
            self._linestyles = [ls]

    def set_linestyles(self, ls):
        self.set_linestyle(ls)

    def get_offsets(self):
        return list(self._offsets)

    def set_offsets(self, offsets):
        self._offsets = list(offsets)

    def get_paths(self):
        """Return paths for the collection."""
        return getattr(self, '_paths', [])

    def set_paths(self, paths):
        """Set paths for the collection."""
        self._paths = list(paths)

    def get_array(self):
        """Return the data array."""
        return getattr(self, '_array', None)

    def set_array(self, A):
        """Set the data array."""
        self._array = A

    def set_color(self, c):
        """Set both facecolor and edgecolor."""
        self.set_facecolor(c)
        self.set_edgecolor(c)


class PathCollection(Collection):
    """A collection of paths with offsets, used by scatter()."""

    def __init__(self, offsets=None, sizes=None, facecolors=None,
                 edgecolors=None, label=None, marker='o', **kwargs):
        super().__init__(**kwargs)

        self._offsets = list(offsets) if offsets is not None else []
        self._sizes = list(sizes) if sizes is not None else [20.0]
        self._facecolors = list(facecolors) if facecolors is not None else []
        self._edgecolors = list(edgecolors) if edgecolors is not None else []
        self._marker = marker

        if label is not None:
            self.set_label(label)

    # --- sizes ---
    def get_sizes(self):
        return list(self._sizes)

    def set_sizes(self, sizes):
        self._sizes = list(sizes)

    # --- draw (new renderer architecture) ---
    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        if not self._offsets:
            return
        import math
        color = to_hex(self._facecolors[0]) if self._facecolors else to_hex('C0')
        s = self._sizes[0] if self._sizes else 20.0
        r = max(1.0, math.sqrt(s) / 2)
        marker = self._marker
        xdata = [layout.sx(pt[0]) for pt in self._offsets]
        ydata = [layout.sy(pt[1]) for pt in self._offsets]
        renderer.draw_markers(xdata, ydata, color, r, marker)


class LineCollection(Collection):
    """A collection of line segments.

    Each segment is a list of (x, y) points.
    """

    zorder = 2

    def __init__(self, segments=None, linewidths=None, colors=None,
                 linestyles=None, label=None, **kwargs):
        super().__init__(**kwargs)
        self._segments = [list(seg) for seg in segments] if segments else []
        if linewidths is not None:
            if hasattr(linewidths, '__iter__'):
                self._linewidths = list(linewidths)
            else:
                self._linewidths = [linewidths]
        if colors is not None:
            if isinstance(colors, str):
                self._edgecolors = [colors]
            else:
                self._edgecolors = list(colors)
        if linestyles is not None:
            if isinstance(linestyles, str):
                self._linestyles = [linestyles]
            else:
                self._linestyles = list(linestyles)
        if label is not None:
            self.set_label(label)

    def get_segments(self):
        """Return a copy of the segments."""
        return [list(seg) for seg in self._segments]

    def set_segments(self, segments):
        """Set the segments."""
        self._segments = [list(seg) for seg in segments] if segments else []

    set_verts = set_segments

    def set_paths(self, paths):  # type: ignore[override]
        self.set_segments(paths)

    def get_paths(self):
        """Return paths (segments) for the collection."""
        return self.get_segments()

    def get_color(self):
        """Return the edge colors (line colors)."""
        return list(self._edgecolors) if self._edgecolors else ['black']

    def set_color(self, c):
        """Set line colors."""
        if isinstance(c, str):
            self._edgecolors = [c]
        else:
            self._edgecolors = list(c)

    def get_colors(self):
        return self.get_color()

    def set_colors(self, c):
        self.set_color(c)

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        for seg in self._segments:
            if len(seg) < 2:
                continue
            x_px = [layout.sx(pt[0]) for pt in seg]
            y_px = [layout.sy(pt[1]) for pt in seg]
            color = to_hex(self._edgecolors[0]) if self._edgecolors else '#000000'
            lw = self._linewidths[0] if self._linewidths else 1.0
            renderer.draw_line(x_px, y_px, color, float(lw), '-')


class PolyCollection(Collection):
    """A collection of polygons."""

    def __init__(self, verts=None, **kwargs):
        super().__init__(**kwargs)
        self._verts = [list(v) for v in verts] if verts else []

    def get_verts(self):
        return [list(v) for v in self._verts]

    def set_verts(self, verts):
        self._verts = [list(v) for v in verts] if verts else []


class EventCollection(Collection):
    """A collection of discrete events, drawn as line segments."""

    def __init__(self, positions=None, orientation='horizontal',
                 lineoffset=0, linelength=1, linewidth=None,
                 color=None, linestyle='solid', **kwargs):
        super().__init__(**kwargs)
        self._positions = list(positions) if positions is not None else []
        self._orientation = orientation
        self._lineoffset = lineoffset
        self._linelength = linelength
        if linewidth is not None:
            self._linewidths = [linewidth] if not hasattr(linewidth, '__iter__') else list(linewidth)
        if color is not None:
            if isinstance(color, str):
                self._edgecolors = [color]
            else:
                self._edgecolors = list(color)
        if linestyle is not None:
            self._linestyles = [linestyle]

    def get_positions(self):
        return list(self._positions)

    def set_positions(self, positions):
        self._positions = list(positions) if positions is not None else []

    def get_orientation(self):
        return self._orientation

    def set_orientation(self, orientation):
        self._orientation = orientation

    def get_lineoffset(self):
        return self._lineoffset

    def set_lineoffset(self, lineoffset):
        self._lineoffset = lineoffset

    def get_linelength(self):
        return self._linelength

    def set_linelength(self, linelength):
        self._linelength = linelength

