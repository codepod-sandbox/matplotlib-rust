"""matplotlib.collections --- Collection and PathCollection artists."""

import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import to_hex, to_rgba

_LINESTYLE_ALIASES = {
    '-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted',
    'solid': 'solid', 'dashed': 'dashed', 'dashdot': 'dashdot', 'dotted': 'dotted',
}


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
        self._capstyle = None
        self._joinstyle = None
        self.norm = None
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

    def get_capstyle(self):
        return self._capstyle

    def set_capstyle(self, cs):
        self._capstyle = cs

    def get_joinstyle(self):
        return self._joinstyle

    def set_joinstyle(self, js):
        self._joinstyle = js

    def set_linestyle(self, ls):
        if isinstance(ls, str):
            if ls not in _LINESTYLE_ALIASES and not isinstance(ls, tuple):
                raise ValueError(f"Do not know how to convert {ls!r} to dashes")
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
        """Set the data array (copied)."""
        if isinstance(A, str):
            raise TypeError(f"Image data of dtype {type(A)!r} cannot be converted "
                            "to float")
        self._array = np.array(A, copy=True) if A is not None else None

    def set_color(self, c):
        """Set both facecolor and edgecolor."""
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def get_cmap(self):
        """Return the colormap."""
        return getattr(self, '_cmap', None)

    def set_cmap(self, cmap):
        """Set the colormap, resolving string names to colormap objects."""
        if isinstance(cmap, str):
            from matplotlib.cm import get_cmap as _get_cmap
            cmap = _get_cmap(cmap)
        self._cmap = cmap

    def get_clim(self):
        """Return (vmin, vmax) of the norm."""
        norm = getattr(self, 'norm', None)
        if norm is not None:
            return norm.vmin, norm.vmax
        return getattr(self, '_vmin', None), getattr(self, '_vmax', None)

    def set_clim(self, vmin=None, vmax=None):
        """Set the norm limits (vmin, vmax)."""
        # Accept set_clim((vmin, vmax)) or set_clim(vmin, vmax)
        if vmax is None and hasattr(vmin, '__len__'):
            vmin, vmax = vmin
        if not hasattr(self, 'norm') or self.norm is None:
            from matplotlib.colors import Normalize
            self.norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            if vmin is not None:
                self.norm.vmin = vmin
            if vmax is not None:
                self.norm.vmax = vmax

    def get_norm(self):
        """Return the normalization instance."""
        return getattr(self, 'norm', None)

    def set_norm(self, norm):
        """Set the normalization instance."""
        self.norm = norm


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

    def get_offsets(self):
        """Return offsets as a numpy array of shape (N, 2)."""
        import numpy as np
        if not self._offsets:
            return np.zeros((0, 2))
        return np.array(self._offsets)

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
        """Return line colors as RGBA tuples."""
        colors = self._edgecolors if self._edgecolors else ['black']
        return [to_rgba(c) for c in colors]

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
        # Extract facecolors/edgecolors before passing to super
        facecolors = kwargs.pop('facecolors', None)
        edgecolors = kwargs.pop('edgecolors', None)
        super().__init__(**kwargs)
        self._verts = [list(v) for v in verts] if verts else []
        if facecolors is not None:
            self.set_facecolor(facecolors)
        if edgecolors is not None:
            self.set_edgecolor(edgecolors)

    def get_verts(self):
        return [list(v) for v in self._verts]

    def set_verts(self, verts):
        self._verts = [list(v) for v in verts] if verts else []

    def draw(self, renderer, layout):
        """Draw each polygon in the collection."""
        if not self.get_visible():
            return
        alpha = self.get_alpha() if self.get_alpha() is not None else 1.0
        for i, verts in enumerate(self._verts):
            if len(verts) < 3:
                continue
            # Get face color for this polygon
            if self._facecolors:
                fc_idx = i % len(self._facecolors)
                fc = self._facecolors[fc_idx]
                if isinstance(fc, str) and fc.lower() == 'none':
                    fc_hex = 'none'
                else:
                    fc_hex = to_hex(fc)
            else:
                fc_hex = 'none'
            if fc_hex == 'none':
                continue
            # Convert verts to pixel coordinates
            pts = []
            for pt in verts:
                if hasattr(pt, '__len__') and len(pt) >= 2:
                    pts.append((layout.sx(pt[0]), layout.sy(pt[1])))
            if len(pts) >= 3:
                renderer.draw_polygon(pts, fc_hex, alpha)

    def get_facecolor(self):
        """Return face colors as RGBA tuples."""
        if not self._facecolors:
            return []
        result = []
        for c in self._facecolors:
            if isinstance(c, str) and c.lower() == 'none':
                result.append((0.0, 0.0, 0.0, 0.0))
            else:
                result.append(to_rgba(c))
        return result

    def get_facecolors(self):
        return self.get_facecolor()


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

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = value

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

    def is_horizontal(self):
        return self._orientation.lower() in ('horizontal', 'none')

    def switch_orientation(self):
        self._orientation = ('vertical' if self.is_horizontal()
                             else 'horizontal')

    def add_positions(self, position):
        """Add a single position."""
        positions = self._positions
        positions.append(position)
        self._positions = positions

    def extend_positions(self, positions):
        """Add multiple positions."""
        self._positions = list(self._positions) + list(positions)

    def append_positions(self, position):
        """Append a single position (alias for add_positions)."""
        self._positions = list(self._positions) + [position]

    def get_color(self):
        return list(self._edgecolors)

    def get_colors(self):
        return list(self._edgecolors)

    def get_segments(self):
        """Return line segments for all positions."""
        positions = self._positions
        ll = self._linelength
        lo = self._lineoffset
        segs = []
        for p in positions:
            if self.is_horizontal():
                segs.append(np.array([[p, lo + ll / 2], [p, lo - ll / 2]]))
            else:
                segs.append(np.array([[lo + ll / 2, p], [lo - ll / 2, p]]))
        return segs


class EllipseCollection(Collection):
    """A collection of ellipses."""

    def __init__(self, widths, heights, angles, units='points',
                 offsets=None, offset_transform=None, **kwargs):
        super().__init__(**kwargs)
        self._widths = np.array(widths).ravel() * 0.5
        self._heights = np.array(heights).ravel() * 0.5
        self._angles = np.deg2rad(np.array(angles).ravel())
        self._units = units
        if offsets is not None:
            self._offsets = np.asarray(offsets)
        self._offset_transform = offset_transform

    def get_widths(self):
        return self._widths * 2

    def get_heights(self):
        return self._heights * 2

    def get_angles(self):
        return np.rad2deg(self._angles)

    def set_widths(self, widths):
        self._widths = np.array(widths).ravel() * 0.5

    def set_heights(self, heights):
        self._heights = np.array(heights).ravel() * 0.5

    def set_angles(self, angles):
        self._angles = np.deg2rad(np.array(angles).ravel())


class QuadMesh(Collection):
    """A mesh of quads for pcolormesh.

    Stores the data array and mesh parameters, with get/set methods
    for colormap, norm, and data array.
    """

    def __init__(self, C, x0=0, x1=1, y0=0, y1=1,
                 cmap=None, norm=None, vmin=None, vmax=None, **kwargs):
        super().__init__(**kwargs)
        self._C = C
        self._x0 = x0
        self._x1 = x1
        self._y0 = y0
        self._y1 = y1
        self._cmap = cmap
        self._norm = norm
        self._vmin = vmin
        self._vmax = vmax

    def get_array(self):
        """Return the data array."""
        return self._C

    def set_array(self, A):
        """Set the data array."""
        self._C = A

    def get_cmap(self):
        """Return the colormap."""
        return self._cmap

    def set_cmap(self, cmap):
        """Set the colormap, resolving string names to colormap objects."""
        if isinstance(cmap, str):
            from matplotlib.cm import get_cmap as _get_cmap
            cmap = _get_cmap(cmap)
        self._cmap = cmap

    def get_clim(self):
        """Return (vmin, vmax)."""
        norm = getattr(self, 'norm', None)
        if norm is not None:
            return norm.vmin, norm.vmax
        return self._vmin, self._vmax

    def set_clim(self, vmin=None, vmax=None):
        """Set the colormap limits."""
        if vmax is None and hasattr(vmin, '__len__'):
            vmin, vmax = vmin
        if not hasattr(self, 'norm') or self.norm is None:
            from matplotlib.colors import Normalize
            self.norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            if vmin is not None:
                self.norm.vmin = vmin
            if vmax is not None:
                self.norm.vmax = vmax

    def get_norm(self):
        """Return the normalization."""
        return getattr(self, 'norm', self._norm)

    def set_norm(self, norm):
        """Set the normalization."""
        self.norm = norm

    def draw(self, renderer, layout):
        """Draw this quad mesh (stub)."""
        pass


class CircleCollection(Collection):
    """A collection of circles, used for scatter-like plots."""

    def __init__(self, sizes, **kwargs):
        """
        Parameters
        ----------
        sizes : array-like of float
            Diameters (in points^2) of circles.
        """
        super().__init__(**kwargs)
        import numpy as np
        self._sizes = np.asarray(sizes, dtype=float)

    def get_sizes(self):
        return list(self._sizes)

    def set_sizes(self, sizes):
        import numpy as np
        self._sizes = np.asarray(sizes, dtype=float)

    def draw(self, renderer, layout):
        pass  # handled by renderer


class RegularPolyCollection(Collection):
    """A collection of regular polygons."""

    def __init__(self, numsides, sizes=(1,), **kwargs):
        """
        Parameters
        ----------
        numsides : int
            Number of sides of the polygon.
        sizes : array-like
            Area of each polygon in points^2.
        """
        super().__init__(**kwargs)
        self._numsides = numsides
        import numpy as np
        self._sizes = np.asarray(sizes, dtype=float)

    def get_numsides(self):
        return self._numsides

    def get_sizes(self):
        return list(self._sizes)

    def set_sizes(self, sizes):
        import numpy as np
        self._sizes = np.asarray(sizes, dtype=float)

    def draw(self, renderer, layout):
        pass


class BrokenBarHCollection(PolyCollection):
    """A collection of horizontal bars with gaps."""

    def __init__(self, xranges, yrange, **kwargs):
        """
        Parameters
        ----------
        xranges : sequence of (xmin, xwidth) pairs
        yrange : (ymin, ywidth) pair
        """
        import numpy as np
        ymin, ywidth = yrange
        ymax = ymin + ywidth
        verts = []
        for xmin, xwidth in xranges:
            xmax = xmin + xwidth
            verts.append(np.array([[xmin, ymin], [xmax, ymin],
                                    [xmax, ymax], [xmin, ymax]]))
        self._xranges = list(xranges)
        self._yrange = yrange
        super().__init__(verts, **kwargs)

    @classmethod
    def span_where(cls, x, ymin, ymax, where, **kwargs):
        """Create BrokenBarHCollection from boolean mask."""
        import numpy as np
        x = np.asarray(x)
        where = np.asarray(where, dtype=bool)
        xranges = []
        i = 0
        while i < len(x):
            if where[i]:
                j = i
                while j < len(x) and where[j]:
                    j += 1
                xranges.append((x[i], x[j-1] - x[i]))
                i = j
            else:
                i += 1
        return cls(xranges, (ymin, ymax - ymin), **kwargs)
