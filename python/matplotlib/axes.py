"""
matplotlib.axes --- Axes class that stores plot elements.
"""

import math

builtins_range = range  # alias so we can use 'range' as a parameter name

from matplotlib.colors import DEFAULT_CYCLE, to_hex, parse_fmt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, Wedge
from matplotlib.collections import PathCollection, LineCollection, Collection
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.text import Text, Annotation
from matplotlib.backend_bases import AxesLayout
from matplotlib._svg_backend import _nice_ticks, _fmt_tick, _esc


class Axes:
    """A single set of axes in a Figure."""

    def __init__(self, fig, position):
        self.figure = fig
        self._position = position
        self._title = ''
        self._xlabel = ''
        self._ylabel = ''
        self._xlim = None
        self._ylim = None
        self._xticks = None
        self._yticks = None
        self._xticklabels = None
        self._yticklabels = None
        self._grid = False
        self._legend = False
        self._color_idx = 0
        self._facecolor = 'white'
        self._prop_cycle = None  # custom property cycle
        self._prop_cycle_idx = 0

        # Typed artist lists
        self.lines = []
        self.collections = []
        self.patches = []
        self.containers = []
        self.texts = []
        self.images = []

        # Axis state
        self._x_inverted = False
        self._y_inverted = False
        self._xscale = 'linear'
        self._yscale = 'linear'
        self._aspect = 'auto'

        # Shared axes
        self._shared_x = []  # list of axes sharing x-limits
        self._shared_y = []  # list of axes sharing y-limits

        # Tick/label visibility (for label_outer)
        self._xticklabels_visible = True
        self._yticklabels_visible = True
        self._xlabel_visible = True
        self._ylabel_visible = True

        # Navigate state (for interactive backends)
        self._navigate = True
        self._navigable = True

    def _next_color(self):
        if self._prop_cycle is not None:
            colors = self._prop_cycle
            c = colors[self._prop_cycle_idx % len(colors)]
            self._prop_cycle_idx += 1
            return c
        c = DEFAULT_CYCLE[self._color_idx % len(DEFAULT_CYCLE)]
        self._color_idx += 1
        return c

    def set_prop_cycle(self, *args, **kwargs):
        """Set the property cycle for this Axes.

        Parameters
        ----------
        *args : cycler or list of colors or None
            If a single iterable of color strings, use as the color cycle.
            If None, reset to default.
        **kwargs : dict
            If 'color' keyword is given, use that as the color cycle.
        """
        if len(args) == 1 and args[0] is None:
            self._prop_cycle = None
            self._prop_cycle_idx = 0
            self._color_idx = 0
            return

        if 'color' in kwargs:
            colors = kwargs['color']
            if isinstance(colors, str):
                colors = [colors]
            self._prop_cycle = list(colors)
            self._prop_cycle_idx = 0
            return

        if len(args) == 1:
            arg = args[0]
            if hasattr(arg, '__iter__') and not isinstance(arg, str):
                # Could be a cycler or list of colors
                if hasattr(arg, 'by_key'):
                    # It's a cycler object
                    keys = arg.by_key()
                    if 'color' in keys:
                        self._prop_cycle = list(keys['color'])
                    else:
                        self._prop_cycle = None
                else:
                    self._prop_cycle = list(arg)
                self._prop_cycle_idx = 0
                return

        if len(args) >= 2:
            # (key, values) form: set_prop_cycle('color', [...])
            key = args[0]
            values = args[1]
            if key == 'color':
                self._prop_cycle = list(values)
                self._prop_cycle_idx = 0
                return

    # ------------------------------------------------------------------
    # Plot types
    # ------------------------------------------------------------------

    def plot(self, *args, **kwargs):
        """Line plot.

        Accepts:
        - ``plot(y)``
        - ``plot(x, y)``
        - ``plot(x, y, fmt)``
        - ``plot(x1, y1, fmt1, x2, y2, fmt2, ...)`` (multi-line)
        - ``plot(x1, y1, x2, y2, ...)``
        """
        # Check for multi-group calls
        groups = _parse_plot_args_multi(args)

        all_lines = []
        for x, y, fmt in groups:
            color_fmt, marker, linestyle = parse_fmt(fmt)
            color = kwargs.get('color') or kwargs.get('c')
            if color is None:
                color = color_fmt
            if color is None:
                color = self._next_color()
            color = to_hex(color)
            label = kwargs.get('label')
            linewidth = kwargs.get('linewidth', kwargs.get('lw', 1.5))
            if linestyle is None:
                linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
            # Forward extra kwargs
            mkw = {}
            for mk in ('markersize', 'ms', 'markeredgecolor', 'mec',
                        'markerfacecolor', 'mfc', 'markeredgewidth', 'mew',
                        'markevery', 'fillstyle', 'drawstyle',
                        'antialiased', 'aa',
                        'solid_capstyle', 'solid_joinstyle',
                        'dash_capstyle', 'dash_joinstyle'):
                if mk in kwargs:
                    mkw[mk] = kwargs[mk]

            # Resolve marker: explicit kwarg overrides fmt
            final_marker = marker
            if 'marker' in kwargs:
                final_marker = kwargs['marker']
            if final_marker is None:
                final_marker = 'None'

            # Create Line2D artist
            line = Line2D(
                x, y,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                marker=final_marker,
                label=label,
                **mkw,
            )
            line.axes = self
            line.figure = self.figure

            # Store in typed list
            self.lines.append(line)
            all_lines.append(line)

        return all_lines

    def scatter(self, x, y, s=20, c=None, marker=None, cmap=None,
                norm=None, vmin=None, vmax=None, alpha=None,
                edgecolors=None, **kwargs):
        """Scatter plot.

        Parameters
        ----------
        c : color or array-like of float
            If a sequence of numbers, they are mapped through *cmap*.
        cmap : str or Colormap, optional
        norm : Normalize, optional
        vmin, vmax : float, optional
        """
        from matplotlib.cm import get_cmap, ScalarMappable
        from matplotlib.colors import Normalize as Norm, to_rgba

        label = kwargs.get('label')
        color_kw = kwargs.get('color')

        x_list = list(x)
        y_list = list(y)
        offsets = list(zip(x_list, y_list))

        # Validate sizes
        if isinstance(s, str):
            raise ValueError("'s' must be a float or array-like of float, "
                             "not a string")
        if hasattr(s, '__iter__'):
            sizes = list(s)
            if len(sizes) != len(x_list):
                raise ValueError("s must be the same size as x and y")
        else:
            sizes = [s]

        # Resolve colors
        facecolors = []
        mappable = None
        c_is_array = False

        if c is not None:
            # Check if c is array of numeric values for colormap mapping
            if hasattr(c, '__iter__') and not isinstance(c, str):
                c_list = list(c)
                if c_list and isinstance(c_list[0], (int, float)):
                    c_is_array = True
                    # Map through colormap
                    if cmap is None:
                        cmap = 'viridis'
                    cmap_obj = get_cmap(cmap) if isinstance(cmap, str) else cmap
                    if norm is None:
                        _vmin = vmin if vmin is not None else min(c_list)
                        _vmax = vmax if vmax is not None else max(c_list)
                        norm_obj = Norm(vmin=_vmin, vmax=_vmax)
                    else:
                        norm_obj = norm
                    for val in c_list:
                        nv = norm_obj(val)
                        rgba = cmap_obj(nv)
                        facecolors.append(to_hex(rgba))
                    # Create ScalarMappable for colorbar support
                    mappable = ScalarMappable(norm=norm_obj, cmap=cmap_obj)
                    mappable.set_array(c_list)
                else:
                    # List of color specs
                    facecolors = [to_hex(ci) for ci in c_list]
            else:
                facecolors = [to_hex(c)]
        elif color_kw is not None:
            facecolors = [to_hex(color_kw)]
        else:
            facecolors = [to_hex(self._next_color())]

        # Create PathCollection
        pc = PathCollection(
            offsets=offsets,
            sizes=sizes,
            facecolors=facecolors,
            label=label,
        )
        if edgecolors is not None:
            if isinstance(edgecolors, str):
                pc.set_edgecolor([edgecolors])
            else:
                pc.set_edgecolor(list(edgecolors))
        if alpha is not None:
            pc.set_alpha(alpha)
        pc.axes = self
        pc.figure = self.figure

        # Attach ScalarMappable properties for colorbar
        if mappable is not None:
            pc._norm = mappable._norm
            pc._cmap = mappable._cmap
            pc._A = mappable._A
            pc.get_array = lambda: pc._A
            pc.get_clim = lambda: (pc._norm.vmin, pc._norm.vmax)
            pc.set_clim = lambda vmin=None, vmax=None: None

        # Store in typed list
        self.collections.append(pc)

        return pc

    def bar(self, x, height, width=0.8, **kwargs):
        """Bar chart."""
        # Color handling: facecolor takes precedence over color
        facecolor = kwargs.get('facecolor')
        edgecolor = kwargs.get('edgecolor')
        color = kwargs.get('color')
        alpha = kwargs.get('alpha')
        label = kwargs.get('label')
        bottom = kwargs.get('bottom', 0)

        if facecolor is None:
            if color is not None:
                facecolor = color
            else:
                facecolor = self._next_color()

        # Convert facecolor to hex for backend (handle 'none')
        if isinstance(facecolor, str) and facecolor.lower() == 'none':
            fc_hex = '#000000'  # transparent will be handled by alpha
        else:
            fc_hex = to_hex(facecolor)

        if edgecolor is None:
            edgecolor = 'black'

        # Convert x and height to lists (broadcast scalar height)
        if isinstance(x, (str, int, float)):
            x_vals = [x]
        else:
            x_vals = list(x)
        if not hasattr(height, '__iter__'):
            h_vals = [height] * len(x_vals)
        else:
            h_vals = list(height)

        # Handle label: list vs scalar
        if isinstance(label, list):
            if len(label) != len(x_vals):
                raise ValueError(
                    f"'label' must have the same length as 'x' "
                    f"({len(label)} != {len(x_vals)})")
            bar_labels = label
            container_label = '_nolegend_'
        else:
            bar_labels = ['_nolegend_'] * len(x_vals)
            container_label = label

        # Handle bottom as list or scalar
        if hasattr(bottom, '__iter__'):
            b_vals = list(bottom)
        else:
            b_vals = [bottom] * len(x_vals)

        # Handle width as list or scalar
        if hasattr(width, '__iter__'):
            w_vals = list(width)
        else:
            w_vals = [width] * len(x_vals)

        # Map string x values to numeric positions
        if x_vals and isinstance(x_vals[0], str):
            x_numeric = list(range(len(x_vals)))
        else:
            x_numeric = x_vals

        # Create Rectangle patches for each bar
        rect_patches = []
        for i in range(len(x_vals)):
            x_center = x_numeric[i]
            h = h_vals[i]
            b = b_vals[i]
            w = w_vals[i] if i < len(w_vals) else w_vals[-1]
            rect = Rectangle(
                (x_center - w / 2, b),
                w,
                h,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
            if alpha is not None:
                rect.set_alpha(alpha)
            rect.set_label(bar_labels[i])
            rect.axes = self
            rect.figure = self.figure
            self.patches.append(rect)
            rect_patches.append(rect)

        # Create BarContainer
        bc = BarContainer(rect_patches, label=container_label)
        self.containers.append(bc)

        return bc

    def hist(self, x, bins=10, range=None, density=False, weights=None,
             cumulative=False, bottom=None, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, log=False, color=None,
             label=None, stacked=False, **kwargs):
        """Histogram --- compute bins, store as bar chart.

        Parameters
        ----------
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}
        stacked : bool
        range : (float, float), optional
        """
        # Backwards compat: pull from kwargs if passed there
        if label is None:
            label = kwargs.get('label')
        if color is None:
            color = kwargs.get('color')
        if not density:
            density = kwargs.get('density', False)

        # Check for list-of-lists (multiple datasets)
        multi = False
        datasets = []
        if hasattr(x, '__iter__') and not isinstance(x, str):
            x_check = list(x)
            if x_check and hasattr(x_check[0], '__iter__') and not isinstance(x_check[0], str):
                multi = True
                datasets = [list(ds) for ds in x_check]

        if not multi:
            datasets = [list(x)]

        # Resolve bins
        all_data = []
        for ds in datasets:
            all_data.extend(ds)

        if isinstance(bins, str) and bins == 'auto':
            bins = _auto_bins(all_data)

        # Handle empty data
        if not all_data:
            c = color or self._next_color()
            c = to_hex(c)
            n_bins = bins if isinstance(bins, int) else 10
            counts = [0] * n_bins
            edges = list(builtins_range(n_bins + 1))
            bc = BarContainer([], label=label)
            self.containers.append(bc)
            if multi:
                return [counts for _ in datasets], edges, bc
            return counts, edges, bc

        # Compute bin edges
        if isinstance(bins, (list, tuple)):
            edges = list(bins)
            n_bins = len(edges) - 1
        else:
            n_bins = bins
            if range is not None:
                lo, hi = float(range[0]), float(range[1])
            else:
                lo = min(all_data)
                hi = max(all_data)
            if lo == hi:
                hi = lo + 1
            bin_width = (hi - lo) / n_bins
            edges = [lo + i * bin_width for i in builtins_range(n_bins + 1)]

        # Compute counts for each dataset
        all_counts = []
        for ds in datasets:
            counts = [0] * n_bins
            bw = edges[-1] - edges[0]
            for v in ds:
                if v < edges[0] or v > edges[-1]:
                    continue
                idx = int((v - edges[0]) / (bw / n_bins))
                if idx >= n_bins:
                    idx = n_bins - 1
                if weights is not None:
                    # Find index in original ds
                    pass
                counts[idx] += 1
            all_counts.append(counts)

        # Density normalization
        if density:
            for ci in builtins_range(len(all_counts)):
                n = sum(all_counts[ci])
                if n > 0:
                    for bi in builtins_range(n_bins):
                        bw = edges[bi + 1] - edges[bi]
                        all_counts[ci][bi] = all_counts[ci][bi] / (n * bw) if bw > 0 else 0

        # Cumulative
        if cumulative:
            for ci in builtins_range(len(all_counts)):
                for bi in builtins_range(1, n_bins):
                    all_counts[ci][bi] += all_counts[ci][bi - 1]

        # Stacked: accumulate counts
        if stacked and len(all_counts) > 1:
            for ci in builtins_range(1, len(all_counts)):
                for bi in builtins_range(n_bins):
                    all_counts[ci][bi] += all_counts[ci - 1][bi]

        # Determine colors
        if color is None:
            colors_list = [to_hex(self._next_color()) for _ in datasets]
        elif isinstance(color, (list, tuple)) and len(color) == len(datasets):
            colors_list = [to_hex(c) for c in color]
        else:
            colors_list = [to_hex(color)] * len(datasets)

        # Build visual elements
        rect_patches = []
        all_lines = []

        if histtype in ('bar', 'barstacked'):
            rw = rwidth if rwidth is not None else 0.9
            for di in builtins_range(len(datasets)):
                counts = all_counts[di]
                clr = colors_list[di]
                for bi in builtins_range(n_bins):
                    bw = (edges[bi + 1] - edges[bi]) * rw
                    center = (edges[bi] + edges[bi + 1]) / 2
                    bot = 0
                    if stacked and di > 0:
                        bot = all_counts[di - 1][bi]
                        h = counts[bi] - bot
                    else:
                        h = counts[bi]
                    rect = Rectangle(
                        (center - bw / 2, bot), bw, h,
                        facecolor=clr, edgecolor='black',
                    )
                    rect.axes = self
                    rect.figure = self.figure
                    self.patches.append(rect)
                    rect_patches.append(rect)

        elif histtype == 'step':
            for di in builtins_range(len(datasets)):
                counts = all_counts[di]
                clr = colors_list[di]
                xs, ys = [], []
                for bi in builtins_range(n_bins):
                    xs.extend([edges[bi], edges[bi + 1]])
                    ys.extend([counts[bi], counts[bi]])
                line = Line2D(xs, ys, color=clr,
                              linewidth=kwargs.get('linewidth', kwargs.get('lw', 1.5)))
                line.axes = self
                line.figure = self.figure
                self.lines.append(line)
                all_lines.append(line)

        elif histtype == 'stepfilled':
            for di in builtins_range(len(datasets)):
                counts = all_counts[di]
                clr = colors_list[di]
                verts = []
                verts.append((edges[0], 0))
                for bi in builtins_range(n_bins):
                    verts.append((edges[bi], counts[bi]))
                    verts.append((edges[bi + 1], counts[bi]))
                verts.append((edges[-1], 0))
                poly = Polygon(verts, facecolor=clr, edgecolor='black')
                poly.axes = self
                poly.figure = self.figure
                self.patches.append(poly)
                rect_patches.append(poly)

        lbl = label if not multi else (label[0] if isinstance(label, list) else label)
        bc = BarContainer(rect_patches, label=lbl)
        self.containers.append(bc)

        if multi:
            return all_counts, edges, bc
        return all_counts[0], edges, bc

    def barh(self, y, width, height=0.8, **kwargs):
        """Horizontal bar chart."""
        color = kwargs.get('color') or self._next_color()
        color = to_hex(color)
        label = kwargs.get('label')
        y_vals = list(y)
        w_vals = list(width)

        rect_patches = []
        for i in range(len(y_vals)):
            y_center = y_vals[i]
            w = w_vals[i]
            rect = Rectangle(
                (0, y_center - height / 2), w, height,
                facecolor=color, edgecolor='black',
            )
            rect.axes = self
            rect.figure = self.figure
            self.patches.append(rect)
            rect_patches.append(rect)

        bc = BarContainer(rect_patches, label=label)
        self.containers.append(bc)

        return bc

    def errorbar(self, x, y, yerr=None, xerr=None, **kwargs):
        """Error bar plot."""
        color = kwargs.get('color') or self._next_color()
        color = to_hex(color)
        label = kwargs.get('label')
        fmt = kwargs.get('fmt', '')

        x_list = list(x)
        y_list = list(y)

        # Create the data line (suppressed when fmt='none' or 'None')
        if fmt.lower() == 'none':
            data_line = None
        else:
            data_line = Line2D(x_list, y_list, color=color, label=label)
            data_line.axes = self
            data_line.figure = self.figure
            self.lines.append(data_line)

        # Create ErrorbarContainer
        has_yerr = yerr is not None
        has_xerr = xerr is not None
        ec = ErrorbarContainer(
            (data_line, [], []),
            has_xerr=has_xerr,
            has_yerr=has_yerr,
            label=label,
        )
        self.containers.append(ec)

        # Store error data on the container for rendering
        ec._yerr_data = (x_list, y_list, yerr) if yerr is not None else None
        ec._xerr_data = (x_list, y_list, xerr) if xerr is not None else None

        return ec

    def fill_between(self, x, y1, y2=0, **kwargs):
        """Fill between two curves."""
        # Validate 2D inputs
        _validate_1d(x, 'x')
        _validate_1d(y1, 'y1')
        if hasattr(y2, '__iter__'):
            _validate_1d(y2, 'y2')

        color = kwargs.get('color') or self._next_color()
        color = to_hex(color)
        label = kwargs.get('label')
        alpha = kwargs.get('alpha', 0.5)

        x_list = list(x)
        y1_list = list(y1)
        y2_list = list(y2) if hasattr(y2, '__iter__') else [y2] * len(x_list)

        # Build polygon: forward along y1, backward along y2
        verts = []
        for i in range(len(x_list)):
            verts.append((x_list[i], y1_list[i]))
        for i in range(len(x_list) - 1, -1, -1):
            verts.append((x_list[i], y2_list[i]))

        poly = Polygon(verts, facecolor=color, edgecolor='none')
        poly.set_alpha(alpha)
        if label:
            poly.set_label(label)
        poly.axes = self
        poly.figure = self.figure
        self.patches.append(poly)

        return poly

    def fill_betweenx(self, y, x1, x2=0, **kwargs):
        """Fill between two curves in the x-direction."""
        # Validate 2D inputs
        _validate_1d(y, 'y')
        _validate_1d(x1, 'x1')
        if hasattr(x2, '__iter__'):
            _validate_1d(x2, 'x2')

        color = kwargs.get('color') or self._next_color()
        color = to_hex(color)
        label = kwargs.get('label')
        alpha = kwargs.get('alpha', 0.5)

        y_list = list(y)
        x1_list = list(x1)
        x2_list = list(x2) if hasattr(x2, '__iter__') else [x2] * len(y_list)

        verts = []
        for i in range(len(y_list)):
            verts.append((x1_list[i], y_list[i]))
        for i in range(len(y_list) - 1, -1, -1):
            verts.append((x2_list[i], y_list[i]))

        poly = Polygon(verts, facecolor=color, edgecolor='none')
        poly.set_alpha(alpha)
        if label:
            poly.set_label(label)
        poly.axes = self
        poly.figure = self.figure
        self.patches.append(poly)

        return poly

    def axhline(self, y=0, **kwargs):
        """Add a horizontal line across the axes."""
        color = kwargs.get('color') or kwargs.get('c', 'black')
        color = to_hex(color)
        linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
        linewidth = kwargs.get('linewidth', kwargs.get('lw', 1.0))
        label = kwargs.get('label')

        # Create a Line2D (with sentinel x-data for axhline)
        line = Line2D(
            [0], [y],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
        line._spanning = 'horizontal'
        line.axes = self
        line.figure = self.figure
        self.lines.append(line)

        return line

    def axvline(self, x=0, **kwargs):
        """Add a vertical line across the axes."""
        color = kwargs.get('color') or kwargs.get('c', 'black')
        color = to_hex(color)
        linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
        linewidth = kwargs.get('linewidth', kwargs.get('lw', 1.0))
        label = kwargs.get('label')

        # Create a Line2D (with sentinel y-data for axvline)
        line = Line2D(
            [x], [0],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
        line._spanning = 'vertical'
        line.axes = self
        line.figure = self.figure
        self.lines.append(line)

        return line

    def hlines(self, y, xmin, xmax, **kwargs):
        """Plot horizontal lines at each *y* from *xmin* to *xmax*."""
        color = kwargs.get('colors', kwargs.get('color', 'black'))
        linewidth = kwargs.get('linewidth', kwargs.get('lw', 1.0))
        linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
        label = kwargs.get('label')

        if not hasattr(y, '__iter__'):
            y = [y]
        y_list = list(y)

        if not hasattr(xmin, '__iter__'):
            xmin = [xmin] * len(y_list)
        if not hasattr(xmax, '__iter__'):
            xmax = [xmax] * len(y_list)
        xmin_list = list(xmin)
        xmax_list = list(xmax)

        lines = []
        for i, yv in enumerate(y_list):
            line = Line2D([xmin_list[i], xmax_list[i]], [yv, yv],
                          color=to_hex(color), linewidth=linewidth,
                          linestyle=linestyle)
            if label and i == 0:
                line.set_label(label)
            else:
                line.set_label('_nolegend_')
            line.axes = self
            line.figure = self.figure
            self.lines.append(line)
            lines.append(line)
        return lines

    def vlines(self, x, ymin, ymax, **kwargs):
        """Plot vertical lines at each *x* from *ymin* to *ymax*."""
        color = kwargs.get('colors', kwargs.get('color', 'black'))
        linewidth = kwargs.get('linewidth', kwargs.get('lw', 1.0))
        linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
        label = kwargs.get('label')

        if not hasattr(x, '__iter__'):
            x = [x]
        x_list = list(x)

        if not hasattr(ymin, '__iter__'):
            ymin = [ymin] * len(x_list)
        if not hasattr(ymax, '__iter__'):
            ymax = [ymax] * len(x_list)
        ymin_list = list(ymin)
        ymax_list = list(ymax)

        lines = []
        for i, xv in enumerate(x_list):
            line = Line2D([xv, xv], [ymin_list[i], ymax_list[i]],
                          color=to_hex(color), linewidth=linewidth,
                          linestyle=linestyle)
            if label and i == 0:
                line.set_label(label)
            else:
                line.set_label('_nolegend_')
            line.axes = self
            line.figure = self.figure
            self.lines.append(line)
            lines.append(line)
        return lines

    def step(self, x, y, where='pre', **kwargs):
        """Step plot."""
        x_list = list(x)
        y_list = list(y)
        n = len(x_list)
        if n < 2:
            return self.plot(x_list, y_list, **kwargs)

        if where == 'pre':
            xs, ys = [x_list[0]], [y_list[0]]
            for i in range(1, n):
                xs.extend([x_list[i], x_list[i]])
                ys.extend([y_list[i - 1], y_list[i]])
        elif where == 'post':
            xs, ys = [x_list[0]], [y_list[0]]
            for i in range(1, n):
                xs.extend([x_list[i - 1], x_list[i]])
                ys.extend([y_list[i], y_list[i]])
        elif where == 'mid':
            xs, ys = [x_list[0]], [y_list[0]]
            for i in range(1, n):
                mid = (x_list[i - 1] + x_list[i]) / 2
                xs.extend([mid, mid, x_list[i]])
                ys.extend([y_list[i - 1], y_list[i], y_list[i]])
        else:
            raise ValueError(
                f"'where' must be 'pre', 'post', or 'mid', not {where!r}")

        return self.plot(xs, ys, **kwargs)

    def stairs(self, values, edges=None, **kwargs):
        """Staircase plot for pre-binned data."""
        vals = list(values)
        n = len(vals)
        if edges is None:
            edg = list(range(n + 1))
        else:
            edg = list(edges)

        xs, ys = [], []
        for i in range(n):
            xs.extend([edg[i], edg[i + 1]])
            ys.extend([vals[i], vals[i]])

        color = kwargs.pop('color', None) or self._next_color()
        line = Line2D(xs, ys,
                      color=color,
                      linewidth=kwargs.pop('linewidth', kwargs.pop('lw', 1.5)),
                      linestyle=kwargs.pop('linestyle', '-'),
                      label=kwargs.pop('label', None))
        line.axes = self
        line.figure = self.figure
        self.lines.append(line)
        return line

    def stackplot(self, x, *args, labels=None, colors=None, baseline='zero',
                  **kwargs):
        """Stacked area plot.

        Parameters
        ----------
        baseline : {'zero', 'sym', 'wiggle', 'weighted_wiggle'}
        """
        x_list = list(x)
        ys = [list(a) for a in args]
        n = len(x_list)
        m = len(ys)

        if labels is None:
            labels = ['_nolegend_'] * m
        if colors is None:
            colors = [self._next_color() for _ in ys]
        else:
            colors = [to_hex(c) for c in colors]

        alpha = kwargs.get('alpha', 0.5)

        # Compute baselines
        if baseline == 'zero':
            base = [0.0] * n
        elif baseline == 'sym':
            # Symmetric: center the stack around zero
            totals = [sum(ys[k][j] for k in range(m)) for j in range(n)]
            base = [-t / 2.0 for t in totals]
        elif baseline == 'wiggle':
            # ThemeRiver / wiggle: minimize derivative of baseline
            base = [0.0] * n
            if m > 0:
                for j in range(n):
                    s = sum(ys[k][j] for k in range(m))
                    base[j] = -s / 2.0
        elif baseline == 'weighted_wiggle':
            base = [0.0] * n
            if m > 0:
                for j in range(n):
                    s = sum(ys[k][j] for k in range(m))
                    base[j] = -s / 2.0
        else:
            raise ValueError(f"Unknown baseline: {baseline!r}")

        cur_base = list(base)
        polys = []
        for i, y_data in enumerate(ys):
            top = [cur_base[j] + y_data[j] for j in range(n)]
            verts = []
            for j in range(n):
                verts.append((x_list[j], top[j]))
            for j in range(n - 1, -1, -1):
                verts.append((x_list[j], cur_base[j]))

            poly = Polygon(verts, facecolor=colors[i], edgecolor='none')
            poly.set_alpha(alpha)
            poly.set_label(labels[i])
            poly.axes = self
            poly.figure = self.figure
            self.patches.append(poly)
            polys.append(poly)
            cur_base = top

        return polys

    def stem(self, *args, linefmt=None, markerfmt=None, basefmt=None,
             bottom=0, label=None, orientation='vertical', **kwargs):
        """Stem plot (lollipop chart)."""
        if len(args) == 0:
            raise TypeError(
                "stem() requires at least 1 positional argument, "
                "but 0 were given")

        # Parse positional args: stem(y), stem(x, y), stem(y, fmt),
        # stem(x, y, fmt)
        if len(args) == 1:
            y_list = list(args[0])
            x_list = list(range(len(y_list)))
        elif len(args) == 2:
            if isinstance(args[1], str):
                # stem(y, fmt)
                y_list = list(args[0])
                x_list = list(range(len(y_list)))
                if linefmt is None:
                    linefmt = args[1]
            else:
                x_list = list(args[0])
                y_list = list(args[1])
        elif len(args) == 3:
            x_list = list(args[0])
            y_list = list(args[1])
            if isinstance(args[2], str) and linefmt is None:
                linefmt = args[2]
        else:
            raise TypeError(
                f"stem() takes 1-3 positional args, got {len(args)}")

        # Validate data is 1-D
        if hasattr(args[0], 'ndim'):
            if args[0].ndim > 1:
                raise ValueError("x or y must be 1-D")
        elif hasattr(args[0], '__iter__') and not isinstance(args[0], str):
            d = list(args[0])
            if d and hasattr(d[0], '__iter__') and not isinstance(d[0], str):
                raise ValueError("x or y must be 1-D")
        if len(args) >= 2 and not isinstance(args[1], str):
            if hasattr(args[1], 'ndim'):
                if args[1].ndim > 1:
                    raise ValueError("x or y must be 1-D")
            elif hasattr(args[1], '__iter__') and not isinstance(args[1], str):
                d = list(args[1])
                if d and hasattr(d[0], '__iter__') and not isinstance(d[0], str):
                    raise ValueError("x or y must be 1-D")

        # Parse line format string for color
        line_color = None
        if linefmt:
            fmt_color, _, fmt_ls = parse_fmt(linefmt)
            if fmt_color:
                line_color = to_hex(fmt_color)

        color = line_color or self._next_color()

        # Determine marker color and style from markerfmt
        marker_color = color
        marker_marker = 'o'
        if markerfmt is not None:
            if markerfmt in ('', ' '):
                marker_marker = 'None'
            else:
                mfmt_color, mfmt_marker, _ = parse_fmt(markerfmt)
                if mfmt_color:
                    marker_color = to_hex(mfmt_color)
                if mfmt_marker:
                    marker_marker = mfmt_marker
        # If no markerfmt color, inherit from line color
        if markerfmt is not None and marker_color == color:
            mfmt_color, _, _ = parse_fmt(markerfmt) if markerfmt not in ('', ' ') else (None, None, None)
            if not mfmt_color:
                marker_color = color

        is_horiz = orientation == 'horizontal'

        stemlines = []
        for i in range(len(x_list)):
            if is_horiz:
                sl = Line2D([bottom, y_list[i]], [x_list[i], x_list[i]],
                            color=color, linewidth=1.0, linestyle='-')
            else:
                sl = Line2D([x_list[i], x_list[i]], [bottom, y_list[i]],
                            color=color, linewidth=1.0, linestyle='-')
            sl.set_label('_nolegend_')
            sl.axes = self
            sl.figure = self.figure
            self.lines.append(sl)
            stemlines.append(sl)

        if is_horiz:
            markerline = Line2D(y_list, x_list, color=marker_color,
                                linewidth=0, linestyle='None',
                                marker=marker_marker)
        else:
            markerline = Line2D(x_list, y_list, color=marker_color,
                                linewidth=0, linestyle='None',
                                marker=marker_marker)
        markerline.set_label('_nolegend_')
        markerline.axes = self
        markerline.figure = self.figure
        self.lines.append(markerline)

        if is_horiz:
            baseline = Line2D([bottom, bottom], [min(x_list), max(x_list)],
                              color='C3', linewidth=1.0, linestyle='-')
        else:
            baseline = Line2D([min(x_list), max(x_list)], [bottom, bottom],
                              color='C3', linewidth=1.0, linestyle='-')
        baseline.set_label('_nolegend_')
        baseline.axes = self
        baseline.figure = self.figure
        self.lines.append(baseline)

        # Store color on stemlines collection for markerfmt test access
        class _StemlineCollection:
            """Lightweight proxy so stemlines.get_color() works."""
            def __init__(self, lines, color):
                self._lines = lines
                self._color = color
            def get_color(self):
                return self._color
            def __len__(self):
                return len(self._lines)
            def __iter__(self):
                return iter(self._lines)

        stemlines_coll = _StemlineCollection(stemlines, color)

        sc = StemContainer((markerline, stemlines_coll, baseline), label=label)
        self.containers.append(sc)
        return sc

    def pie(self, x, labels=None, colors=None, explode=None,
            autopct=None, startangle=0, counterclock=True, **kwargs):
        """Pie chart."""
        vals = list(x)
        # Validate: no negative values
        for v in vals:
            if v < 0:
                raise ValueError(
                    "Wedge sizes must be non-negative; got negative value(s)")
        total = sum(vals)
        if total == 0:
            raise ValueError("All wedge sizes are zero")

        n = len(vals)
        if colors is None:
            colors = [DEFAULT_CYCLE[i % len(DEFAULT_CYCLE)] for i in range(n)]
        if labels is None:
            labels = [None] * n
        if explode is None:
            explode = [0.0] * n

        self.set_aspect('equal')

        cx, cy = 0.0, 0.0
        radius = 1.0

        wedges = []
        texts = []
        autotexts = [] if autopct else None

        angle = startangle
        for i in range(n):
            frac = vals[i] / total
            sweep = frac * 360.0
            if not counterclock:
                sweep = -sweep

            theta1 = angle
            theta2 = angle + sweep

            # Explode offset
            if explode[i] != 0:
                mid_angle = math.radians((theta1 + theta2) / 2)
                dx = explode[i] * math.cos(mid_angle)
                dy = explode[i] * math.sin(mid_angle)
            else:
                dx, dy = 0, 0

            w = Wedge((cx + dx, cy + dy), radius, theta1, theta2,
                      facecolor=colors[i], edgecolor='white')
            w.axes = self
            w.figure = self.figure
            self.patches.append(w)
            wedges.append(w)

            # Label text at 1.2 * radius
            if labels[i] is not None:
                mid_angle = math.radians((theta1 + theta2) / 2)
                lx = cx + dx + 1.2 * radius * math.cos(mid_angle)
                ly = cy + dy + 1.2 * radius * math.sin(mid_angle)
                ha = 'left' if math.cos(mid_angle) >= 0 else 'right'
                t = Text(lx, ly, labels[i], ha=ha, va='center', fontsize=11)
                t.axes = self
                t.figure = self.figure
                self.texts.append(t)
                texts.append(t)

            # Autopct text at 0.6 * radius
            if autopct is not None:
                pct = frac * 100
                mid_angle = math.radians((theta1 + theta2) / 2)
                px = cx + dx + 0.6 * radius * math.cos(mid_angle)
                py = cy + dy + 0.6 * radius * math.sin(mid_angle)
                pct_text = autopct % pct
                at = Text(px, py, pct_text, ha='center', va='center',
                          fontsize=10)
                at.axes = self
                at.figure = self.figure
                self.texts.append(at)
                autotexts.append(at)

            angle = theta2

        if autopct is not None:
            return wedges, texts, autotexts
        return wedges, texts

    def boxplot(self, x, vert=True, widths=None, showfliers=True,
                showmeans=False, **kwargs):
        """Box-and-whisker plot."""
        if not x:
            return {'boxes': [], 'medians': [], 'whiskers': [],
                    'caps': [], 'fliers': [], 'means': []}
        # Normalize input: always a list of datasets
        if not hasattr(x[0], '__iter__'):
            datasets = [list(x)]
        else:
            datasets = [list(d) for d in x]

        n = len(datasets)
        if widths is None:
            widths = [0.5] * n
        elif not hasattr(widths, '__iter__'):
            widths = [widths] * n

        result = {
            'boxes': [],
            'medians': [],
            'whiskers': [],
            'caps': [],
            'fliers': [],
            'means': [],
        }

        for i, data in enumerate(datasets):
            pos = i + 1
            w = widths[i]
            sorted_data = sorted(data)

            q1 = _percentile(sorted_data, 25)
            med = _median(sorted_data)
            q3 = _percentile(sorted_data, 75)
            iqr = q3 - q1

            whisker_lo = q1 - 1.5 * iqr
            whisker_hi = q3 + 1.5 * iqr
            actual_lo = min(v for v in sorted_data if v >= whisker_lo)
            actual_hi = max(v for v in sorted_data if v <= whisker_hi)

            if vert:
                # Box
                box = Rectangle((pos - w / 2, q1), w, q3 - q1,
                                facecolor='white', edgecolor='black')
                box.set_label('_nolegend_')
                box.axes = self
                box.figure = self.figure
                self.patches.append(box)
                result['boxes'].append(box)

                # Median line
                med_line = Line2D([pos - w / 2, pos + w / 2], [med, med],
                                  color='orange', linewidth=2.0)
                med_line.set_label('_nolegend_')
                med_line.axes = self
                med_line.figure = self.figure
                self.lines.append(med_line)
                result['medians'].append(med_line)

                # Whiskers
                lo_whisker = Line2D([pos, pos], [actual_lo, q1],
                                    color='black', linewidth=1.0, linestyle='--')
                lo_whisker.set_label('_nolegend_')
                lo_whisker.axes = self
                lo_whisker.figure = self.figure
                self.lines.append(lo_whisker)

                hi_whisker = Line2D([pos, pos], [q3, actual_hi],
                                    color='black', linewidth=1.0, linestyle='--')
                hi_whisker.set_label('_nolegend_')
                hi_whisker.axes = self
                hi_whisker.figure = self.figure
                self.lines.append(hi_whisker)
                result['whiskers'].extend([lo_whisker, hi_whisker])

                # Caps
                lo_cap = Line2D([pos - w / 4, pos + w / 4],
                                [actual_lo, actual_lo],
                                color='black', linewidth=1.0)
                lo_cap.set_label('_nolegend_')
                lo_cap.axes = self
                lo_cap.figure = self.figure
                self.lines.append(lo_cap)

                hi_cap = Line2D([pos - w / 4, pos + w / 4],
                                [actual_hi, actual_hi],
                                color='black', linewidth=1.0)
                hi_cap.set_label('_nolegend_')
                hi_cap.axes = self
                hi_cap.figure = self.figure
                self.lines.append(hi_cap)
                result['caps'].extend([lo_cap, hi_cap])

                # Fliers
                if showfliers:
                    flier_pts = [v for v in sorted_data
                                 if v < actual_lo or v > actual_hi]
                    if flier_pts:
                        flier_x = [pos] * len(flier_pts)
                        pc = PathCollection(
                            offsets=list(zip(flier_x, flier_pts)),
                            sizes=[20], facecolors=['black'])
                        pc.set_label('_nolegend_')
                        pc.axes = self
                        pc.figure = self.figure
                        self.collections.append(pc)
                        result['fliers'].append(pc)

                # Means
                if showmeans:
                    mean_val = sum(data) / len(data)
                    mean_pc = PathCollection(
                        offsets=[(pos, mean_val)],
                        sizes=[50], facecolors=['green'])
                    mean_pc.set_label('_nolegend_')
                    mean_pc.axes = self
                    mean_pc.figure = self.figure
                    self.collections.append(mean_pc)
                    result['means'].append(mean_pc)

            else:
                # Horizontal boxplot: swap x and y
                box = Rectangle((q1, pos - w / 2), q3 - q1, w,
                                facecolor='white', edgecolor='black')
                box.set_label('_nolegend_')
                box.axes = self
                box.figure = self.figure
                self.patches.append(box)
                result['boxes'].append(box)

                med_line = Line2D([med, med], [pos - w / 2, pos + w / 2],
                                  color='orange', linewidth=2.0)
                med_line.set_label('_nolegend_')
                med_line.axes = self
                med_line.figure = self.figure
                self.lines.append(med_line)
                result['medians'].append(med_line)

                lo_whisker = Line2D([actual_lo, q1], [pos, pos],
                                    color='black', linewidth=1.0, linestyle='--')
                lo_whisker.set_label('_nolegend_')
                lo_whisker.axes = self
                lo_whisker.figure = self.figure
                self.lines.append(lo_whisker)

                hi_whisker = Line2D([q3, actual_hi], [pos, pos],
                                    color='black', linewidth=1.0, linestyle='--')
                hi_whisker.set_label('_nolegend_')
                hi_whisker.axes = self
                hi_whisker.figure = self.figure
                self.lines.append(hi_whisker)
                result['whiskers'].extend([lo_whisker, hi_whisker])

                lo_cap = Line2D([actual_lo, actual_lo],
                                [pos - w / 4, pos + w / 4],
                                color='black', linewidth=1.0)
                lo_cap.set_label('_nolegend_')
                lo_cap.axes = self
                lo_cap.figure = self.figure
                self.lines.append(lo_cap)

                hi_cap = Line2D([actual_hi, actual_hi],
                                [pos - w / 4, pos + w / 4],
                                color='black', linewidth=1.0)
                hi_cap.set_label('_nolegend_')
                hi_cap.axes = self
                hi_cap.figure = self.figure
                self.lines.append(hi_cap)
                result['caps'].extend([lo_cap, hi_cap])

                if showfliers:
                    flier_pts = [v for v in sorted_data
                                 if v < actual_lo or v > actual_hi]
                    if flier_pts:
                        flier_y = [pos] * len(flier_pts)
                        pc = PathCollection(
                            offsets=list(zip(flier_pts, flier_y)),
                            sizes=[20], facecolors=['black'])
                        pc.set_label('_nolegend_')
                        pc.axes = self
                        pc.figure = self.figure
                        self.collections.append(pc)
                        result['fliers'].append(pc)

        return result

    def violinplot(self, dataset, positions=None, vert=True, widths=0.5,
                   showmeans=False, showmedians=False, showextrema=True,
                   **kwargs):
        """Violin plot."""
        # Check for empty dataset (handle numpy arrays and lists)
        try:
            is_empty = len(dataset) == 0
        except TypeError:
            is_empty = False
        if is_empty:
            return {'bodies': [], 'cmeans': [], 'cmedians': [],
                    'cmins': [], 'cmaxes': [], 'cbars': []}
        # Normalize: always list of datasets
        if not hasattr(dataset[0], '__iter__'):
            datasets = [list(dataset)]
        else:
            datasets = [list(d) for d in dataset]

        n = len(datasets)
        if positions is None:
            positions = list(range(1, n + 1))

        if not hasattr(widths, '__iter__'):
            widths = [widths] * n

        result = {
            'bodies': [],
            'cmeans': [],
            'cmedians': [],
            'cmins': [],
            'cmaxes': [],
            'cbars': [],
        }

        for i, data in enumerate(datasets):
            pos = positions[i]
            w = widths[i]
            color = self._next_color()

            kde_pos, kde_dens = _gaussian_kde(data, n_points=50)
            if not kde_dens:
                continue

            max_d = max(kde_dens) if kde_dens else 1.0
            scale = (w / 2) / max_d if max_d > 0 else 1.0

            if vert:
                verts = []
                for j in range(len(kde_pos)):
                    verts.append((pos + kde_dens[j] * scale, kde_pos[j]))
                for j in range(len(kde_pos) - 1, -1, -1):
                    verts.append((pos - kde_dens[j] * scale, kde_pos[j]))
            else:
                verts = []
                for j in range(len(kde_pos)):
                    verts.append((kde_pos[j], pos + kde_dens[j] * scale))
                for j in range(len(kde_pos) - 1, -1, -1):
                    verts.append((kde_pos[j], pos - kde_dens[j] * scale))

            poly = Polygon(verts, facecolor=color, edgecolor='black')
            poly.set_alpha(0.5)
            poly.set_label('_nolegend_')
            poly.axes = self
            poly.figure = self.figure
            self.patches.append(poly)
            result['bodies'].append(poly)

            sorted_data = sorted(data)
            data_min = sorted_data[0]
            data_max = sorted_data[-1]
            data_mean = sum(data) / len(data)
            data_med = _median(data)

            if showextrema:
                if vert:
                    bar = Line2D([pos, pos], [data_min, data_max],
                                 color='black', linewidth=1.0)
                    bar.set_label('_nolegend_')
                    bar.axes = self
                    bar.figure = self.figure
                    self.lines.append(bar)
                    result['cbars'].append(bar)

                    min_line = Line2D([pos - w / 4, pos + w / 4],
                                     [data_min, data_min],
                                     color='black', linewidth=1.0)
                    min_line.set_label('_nolegend_')
                    min_line.axes = self
                    min_line.figure = self.figure
                    self.lines.append(min_line)
                    result['cmins'].append(min_line)

                    max_line = Line2D([pos - w / 4, pos + w / 4],
                                     [data_max, data_max],
                                     color='black', linewidth=1.0)
                    max_line.set_label('_nolegend_')
                    max_line.axes = self
                    max_line.figure = self.figure
                    self.lines.append(max_line)
                    result['cmaxes'].append(max_line)
                else:
                    bar = Line2D([data_min, data_max], [pos, pos],
                                 color='black', linewidth=1.0)
                    bar.set_label('_nolegend_')
                    bar.axes = self
                    bar.figure = self.figure
                    self.lines.append(bar)
                    result['cbars'].append(bar)

                    min_line = Line2D([data_min, data_min],
                                     [pos - w / 4, pos + w / 4],
                                     color='black', linewidth=1.0)
                    min_line.set_label('_nolegend_')
                    min_line.axes = self
                    min_line.figure = self.figure
                    self.lines.append(min_line)
                    result['cmins'].append(min_line)

                    max_line = Line2D([data_max, data_max],
                                     [pos - w / 4, pos + w / 4],
                                     color='black', linewidth=1.0)
                    max_line.set_label('_nolegend_')
                    max_line.axes = self
                    max_line.figure = self.figure
                    self.lines.append(max_line)
                    result['cmaxes'].append(max_line)

            if showmeans:
                if vert:
                    m = Line2D([pos - w / 4, pos + w / 4],
                               [data_mean, data_mean],
                               color='red', linewidth=1.5)
                else:
                    m = Line2D([data_mean, data_mean],
                               [pos - w / 4, pos + w / 4],
                               color='red', linewidth=1.5)
                m.set_label('_nolegend_')
                m.axes = self
                m.figure = self.figure
                self.lines.append(m)
                result['cmeans'].append(m)

            if showmedians:
                if vert:
                    m = Line2D([pos - w / 4, pos + w / 4],
                               [data_med, data_med],
                               color='blue', linewidth=1.5)
                else:
                    m = Line2D([data_med, data_med],
                               [pos - w / 4, pos + w / 4],
                               color='blue', linewidth=1.5)
                m.set_label('_nolegend_')
                m.axes = self
                m.figure = self.figure
                self.lines.append(m)
                result['cmedians'].append(m)

        return result

    def text(self, x, y, s, **kwargs):
        """Add text to the axes."""
        t = Text(x, y, str(s), **kwargs)
        t.axes = self
        t.figure = self.figure
        self.texts.append(t)

        return t

    def annotate(self, text, xy, xytext=None, arrowprops=None, **kwargs):
        """Add an annotation to the axes."""
        ann = Annotation(text, xy, xytext=xytext, arrowprops=arrowprops,
                         **kwargs)
        ann.axes = self
        ann.figure = self.figure
        self.texts.append(ann)

        return ann

    # ------------------------------------------------------------------
    # Convenience scale methods
    # ------------------------------------------------------------------

    def loglog(self, *args, **kwargs):
        """Log-log plot: set both axes to log scale, then plot."""
        self.set_xscale('log')
        self.set_yscale('log')
        return self.plot(*args, **kwargs)

    def semilogx(self, *args, **kwargs):
        """Semi-log plot: set x-axis to log scale, then plot."""
        self.set_xscale('log')
        return self.plot(*args, **kwargs)

    def semilogy(self, *args, **kwargs):
        """Semi-log plot: set y-axis to log scale, then plot."""
        self.set_yscale('log')
        return self.plot(*args, **kwargs)

    # ------------------------------------------------------------------
    # Margins
    # ------------------------------------------------------------------

    def margins(self, *args, **kwargs):
        """Set or get margins.

        - ``margins()`` returns (xmargin, ymargin).
        - ``margins(m)`` sets both margins.
        - ``margins(x=mx, y=my)`` sets individually.
        """
        if not args and not kwargs:
            return (getattr(self, '_xmargin', 0.05),
                    getattr(self, '_ymargin', 0.05))
        if args:
            if len(args) == 1:
                self._xmargin = args[0]
                self._ymargin = args[0]
            elif len(args) == 2:
                self._xmargin = args[0]
                self._ymargin = args[1]
        if 'x' in kwargs:
            self._xmargin = kwargs['x']
        if 'y' in kwargs:
            self._ymargin = kwargs['y']

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def set_xbound(self, lower=None, upper=None):
        """Set the x-axis bounds (always ordered lower < upper)."""
        if lower is not None and upper is not None:
            if lower > upper:
                lower, upper = upper, lower
        self.set_xlim(lower, upper)

    def get_xbound(self):
        """Return (lower, upper) for x-axis, always lower <= upper."""
        lo, hi = self.get_xlim()
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    def set_ybound(self, lower=None, upper=None):
        """Set the y-axis bounds (always ordered lower < upper)."""
        if lower is not None and upper is not None:
            if lower > upper:
                lower, upper = upper, lower
        self.set_ylim(lower, upper)

    def get_ybound(self):
        """Return (lower, upper) for y-axis, always lower <= upper."""
        lo, hi = self.get_ylim()
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    # ------------------------------------------------------------------
    # Artist management
    # ------------------------------------------------------------------

    def add_artist(self, artist):
        """Add an Artist to the axes.

        The artist is placed in the appropriate typed list based on its type.
        """
        from matplotlib.lines import Line2D as _Line2D
        from matplotlib.collections import Collection as _Coll
        from matplotlib.text import Text as _Text

        artist.axes = self
        artist.figure = self.figure

        if isinstance(artist, _Line2D):
            self.lines.append(artist)
        elif isinstance(artist, _Coll):
            self.collections.append(artist)
        elif isinstance(artist, _Text):
            self.texts.append(artist)
        else:
            # Default: try patches
            self.patches.append(artist)
        return artist

    def add_line(self, line):
        """Add a Line2D to the axes."""
        line.axes = self
        line.figure = self.figure
        self.lines.append(line)
        return line

    def add_patch(self, patch):
        """Add a Patch to the axes."""
        patch.axes = self
        patch.figure = self.figure
        self.patches.append(patch)
        return patch

    def add_collection(self, collection, autolim=True):
        """Add a Collection to the axes."""
        collection.axes = self
        collection.figure = self.figure
        self.collections.append(collection)
        return collection

    def add_container(self, container):
        """Add a Container to the axes."""
        self.containers.append(container)
        return container

    def add_image(self, image):
        """Add an AxesImage to the axes."""
        image.axes = self
        image.figure = self.figure
        self.images.append(image)
        return image

    # --- Getters for typed lists ---

    def get_lines(self):
        """Return a list of lines in the axes."""
        return list(self.lines)

    def get_images(self):
        """Return a list of images in the axes."""
        return list(self.images)

    # --- relim / autoscale ---

    def relim(self, visible_only=False):
        """Recompute the auto-limits from data (no-op: limits are always auto-computed)."""
        # In our implementation, limits are always computed from data.
        # Clear any manual limits so they get recomputed.
        pass

    def autoscale(self, enable=True, axis='both', tight=None):
        """Auto-scale the axes based on data.

        Parameters
        ----------
        enable : bool, default True
        axis : {'both', 'x', 'y'}
        tight : bool, optional
        """
        if enable:
            if axis in ('both', 'x'):
                self._xlim = None
            if axis in ('both', 'y'):
                self._ylim = None

    def autoscale_view(self, tight=None, scalex=True, scaley=True):
        """Autoscale the view based on data limits."""
        if scalex:
            self._xlim = None
        if scaley:
            self._ylim = None

    def can_pan(self):
        """Return whether the axes can be panned."""
        return True

    def can_zoom(self):
        """Return whether the axes can be zoomed."""
        return True

    def has_data(self):
        """Return whether the axes has any data."""
        return bool(self.lines or self.collections or self.patches or self.images)

    def _remove_artist(self, artist):
        """Remove an artist from this axes' typed lists."""
        if artist in self.lines:
            self.lines.remove(artist)
        elif artist in self.collections:
            self.collections.remove(artist)
        elif artist in self.patches:
            self.patches.remove(artist)
        elif artist in self.texts:
            self.texts.remove(artist)
        artist.axes = None
        artist.figure = None

    def get_legend_handles_labels(self):
        """Return (handles, labels) for all artists with non-underscore labels."""
        handles = []
        labels = []

        # Collect from all artist lists
        all_artists = (
            list(self.lines) +
            list(self.collections) +
            list(self.containers) +
            list(self.patches)
        )

        seen_labels = set()
        for artist in all_artists:
            lbl = artist.get_label() if hasattr(artist, 'get_label') else ''
            if lbl and not lbl.startswith('_') and lbl not in seen_labels:
                handles.append(artist)
                labels.append(lbl)
                seen_labels.add(lbl)

        return handles, labels

    # ------------------------------------------------------------------
    # Labels / config
    # ------------------------------------------------------------------

    def set_title(self, s):
        self._title = s

    def get_title(self):
        return self._title

    def set_xlabel(self, s):
        self._xlabel = s

    def get_xlabel(self):
        return self._xlabel

    def set_ylabel(self, s):
        self._ylabel = s

    def get_ylabel(self):
        return self._ylabel

    def set_xlim(self, left=None, right=None, _propagating=False):
        # Validate: reject NaN or Inf
        for val, name in [(left, 'left'), (right, 'right')]:
            if val is not None:
                if math.isnan(val):
                    raise ValueError(
                        f"Axis limits cannot be NaN: {name}={val}")
                if math.isinf(val):
                    raise ValueError(
                        f"Axis limits cannot be Inf: {name}={val}")
        self._xlim = (left, right)
        # Propagate to shared axes
        if not _propagating:
            for other in self._shared_x:
                if other is not self:
                    other.set_xlim(left, right, _propagating=True)

    def get_xlim(self):
        if self._xlim is not None and self._xlim[0] is not None and self._xlim[1] is not None:
            lo, hi = self._xlim
            if self._x_inverted:
                return (hi, lo)
            return (lo, hi)
        # Auto-calculate from data
        lo, hi = self._auto_xlim()
        if self._x_inverted:
            return (hi, lo)
        return (lo, hi)

    def set_ylim(self, bottom=None, top=None, _propagating=False):
        for val, name in [(bottom, 'bottom'), (top, 'top')]:
            if val is not None:
                if math.isnan(val):
                    raise ValueError(
                        f"Axis limits cannot be NaN: {name}={val}")
                if math.isinf(val):
                    raise ValueError(
                        f"Axis limits cannot be Inf: {name}={val}")
        self._ylim = (bottom, top)
        if not _propagating:
            for other in self._shared_y:
                if other is not self:
                    other.set_ylim(bottom, top, _propagating=True)

    def get_ylim(self):
        if self._ylim is not None and self._ylim[0] is not None and self._ylim[1] is not None:
            lo, hi = self._ylim
            if self._y_inverted:
                return (hi, lo)
            return (lo, hi)
        # Auto-calculate from data
        lo, hi = self._auto_ylim()
        if self._y_inverted:
            return (hi, lo)
        return (lo, hi)

    def _auto_xlim(self):
        """Auto-calculate x limits from data in lines, collections, and patches.

        If this axes shares the x-axis with others, collect data from all.
        """
        # Collect from all shared axes (including self)
        axes_to_check = self._shared_x if self._shared_x else [self]
        xs = []
        for ax in axes_to_check:
            xs.extend(ax._collect_x_data())
        if not xs:
            return (0.0, 1.0)
        return (min(xs), max(xs))

    def _collect_x_data(self):
        """Collect all x data points from this axes' artists."""
        xs = []
        for line in self.lines:
            spanning = getattr(line, '_spanning', None)
            if spanning == 'horizontal':
                continue  # axhline doesn't contribute to x range
            xs.extend(line.get_xdata())
        for coll in self.collections:
            for pt in coll.get_offsets():
                xs.append(pt[0])
        for patch in self.patches:
            if hasattr(patch, '_xy') and hasattr(patch, '_width'):
                xs.append(patch._xy[0])
                xs.append(patch._xy[0] + patch._width)
            elif hasattr(patch, '_xy') and isinstance(patch._xy, list):
                for pt in patch._xy:
                    xs.append(pt[0])
            elif hasattr(patch, '_center') and hasattr(patch, '_r'):
                xs.append(patch._center[0] - patch._r)
                xs.append(patch._center[0] + patch._r)
        return xs

    def _auto_ylim(self):
        """Auto-calculate y limits from data in lines, collections, and patches.

        If this axes shares the y-axis with others, collect data from all.
        """
        axes_to_check = self._shared_y if self._shared_y else [self]
        ys = []
        for ax in axes_to_check:
            ys.extend(ax._collect_y_data())
        if not ys:
            return (0.0, 1.0)
        return (min(ys), max(ys))

    def _collect_y_data(self):
        """Collect all y data points from this axes' artists."""
        ys = []
        for line in self.lines:
            spanning = getattr(line, '_spanning', None)
            if spanning == 'vertical':
                continue  # axvline doesn't contribute to y range
            ys.extend(line.get_ydata())
        for coll in self.collections:
            for pt in coll.get_offsets():
                ys.append(pt[1])
        for patch in self.patches:
            if hasattr(patch, '_xy') and hasattr(patch, '_height'):
                ys.append(patch._xy[1])
                ys.append(patch._xy[1] + patch._height)
            elif hasattr(patch, '_xy') and isinstance(patch._xy, list):
                for pt in patch._xy:
                    ys.append(pt[1])
            elif hasattr(patch, '_center') and hasattr(patch, '_r'):
                ys.append(patch._center[1] - patch._r)
                ys.append(patch._center[1] + patch._r)
        # Bars typically start from 0
        if ys and any(hasattr(p, '_height') for p in self.patches):
            ys.append(0)
        return ys

    def set_xticks(self, ticks, labels=None, **kwargs):
        self._xticks = list(ticks)
        if labels is not None:
            self._xticklabels = list(labels)

    def get_xticks(self):
        return self._xticks if self._xticks is not None else []

    def set_yticks(self, ticks, labels=None, **kwargs):
        self._yticks = list(ticks)
        if labels is not None:
            self._yticklabels = list(labels)

    def get_yticks(self):
        return self._yticks if self._yticks is not None else []

    def get_xticklabels(self):
        """Return the x-axis tick labels."""
        if self._xticklabels is not None:
            return list(self._xticklabels)
        return []

    def set_xticklabels(self, labels, **kwargs):
        self._xticklabels = list(labels)

    def get_yticklabels(self):
        """Return the y-axis tick labels."""
        if self._yticklabels is not None:
            return list(self._yticklabels)
        return []

    def set_yticklabels(self, labels, **kwargs):
        self._yticklabels = list(labels)

    def tick_params(self, axis='both', **kwargs):
        """Change the appearance of ticks, tick labels, and gridlines.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default 'both'
            Which axis to apply to.
        **kwargs
            Tick parameters: direction, length, width, color, pad,
            labelsize, labelcolor, labeltop, labelbottom, labelleft,
            labelright, labelrotation, grid_color, grid_alpha,
            grid_linewidth, grid_linestyle, bottom, top, left, right,
            which ('major', 'minor', 'both').
        """
        if not hasattr(self, '_tick_params'):
            self._tick_params = {'x': {}, 'y': {}}

        if axis == 'both':
            self._tick_params['x'].update(kwargs)
            self._tick_params['y'].update(kwargs)
        elif axis == 'x':
            self._tick_params['x'].update(kwargs)
        elif axis == 'y':
            self._tick_params['y'].update(kwargs)

    def get_tick_params(self, axis='x'):
        """Return the tick parameters for *axis*."""
        if not hasattr(self, '_tick_params'):
            self._tick_params = {'x': {}, 'y': {}}
        return dict(self._tick_params.get(axis, {}))

    def minorticks_on(self):
        """Turn on minor ticks (no-op)."""
        pass

    def minorticks_off(self):
        """Turn off minor ticks (no-op)."""
        pass

    def set_visible(self, b):
        """Set whether the axes is visible."""
        self._visible = b

    def get_visible(self):
        """Return whether the axes is visible."""
        return getattr(self, '_visible', True)

    def legend(self, *args, **kwargs):
        """Add a legend to the axes.

        Returns a Legend object with get_texts(), get_title(), set_title().
        """
        from matplotlib.legend import Legend

        if len(args) > 2:
            raise TypeError(
                f"legend() takes at most 2 positional arguments "
                f"({len(args)} given)")

        handles = None
        labels = None

        if len(args) == 2:
            handles, labels = args
        elif len(args) == 1:
            # Single arg: list of labels
            labels = list(args[0])

        if handles is None:
            handles = kwargs.pop('handles', None)
        if labels is None:
            labels = kwargs.pop('labels', None)

        # Collect from lines/patches/collections if not provided
        if handles is None and labels is None:
            h, l = self.get_legend_handles_labels()
            handles = h
            labels = l
        elif handles is not None and labels is None:
            labels = [h.get_label() for h in handles]
        elif labels is not None and handles is None:
            h, _ = self.get_legend_handles_labels()
            handles = h[:len(labels)]

        loc = kwargs.pop('loc', 'best')
        title = kwargs.pop('title', '')

        leg = Legend(self, handles or [], labels or [],
                     loc=loc, title=title, **kwargs)
        self._legend_obj = leg
        self._legend = True
        return leg

    def grid(self, visible=True, **kwargs):
        self._grid = visible

    # ------------------------------------------------------------------
    # Axis inversion
    # ------------------------------------------------------------------

    def invert_xaxis(self):
        """Invert the x-axis."""
        self._x_inverted = not self._x_inverted

    def invert_yaxis(self):
        """Invert the y-axis."""
        self._y_inverted = not self._y_inverted

    def xaxis_inverted(self):
        """Return whether the x-axis is inverted."""
        return self._x_inverted

    def yaxis_inverted(self):
        """Return whether the y-axis is inverted."""
        return self._y_inverted

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------

    def set_xscale(self, scale):
        """Set the x-axis scale (e.g. 'linear', 'log')."""
        self._xscale = scale

    def set_yscale(self, scale):
        """Set the y-axis scale (e.g. 'linear', 'log')."""
        self._yscale = scale

    def get_xscale(self):
        """Return the x-axis scale."""
        return self._xscale

    def get_yscale(self):
        """Return the y-axis scale."""
        return self._yscale

    # ------------------------------------------------------------------
    # Aspect
    # ------------------------------------------------------------------

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        """Set the axes aspect ratio."""
        self._aspect = aspect
        if adjustable is not None:
            self._adjustable = adjustable
        if anchor is not None:
            self._anchor = anchor

    def get_aspect(self):
        """Return the axes aspect ratio."""
        return self._aspect

    def set_adjustable(self, adjustable, share=False):
        """Set how the axes adjusts to achieve the required aspect."""
        self._adjustable = adjustable

    def get_adjustable(self):
        """Return the adjustable setting."""
        return getattr(self, '_adjustable', 'box')

    def set_anchor(self, anchor, share=False):
        """Set the anchor position."""
        self._anchor = anchor

    def get_anchor(self):
        """Return the anchor position."""
        return getattr(self, '_anchor', 'C')

    def set_box_aspect(self, aspect=None):
        """Set the box aspect ratio (height/width)."""
        self._box_aspect = aspect

    def get_box_aspect(self):
        """Return the box aspect ratio."""
        return getattr(self, '_box_aspect', None)

    # ------------------------------------------------------------------
    # Axis utility
    # ------------------------------------------------------------------

    def axis(self, option=None):
        """Set or get axis properties.

        Parameters
        ----------
        option : str, optional
            - 'square': set equal aspect with matched limits
            - 'equal': set equal aspect
            - 'off'/'on': toggle visibility
        """
        if option is None:
            xlim = self.get_xlim()
            ylim = self.get_ylim()
            return xlim + ylim
        if option == 'square':
            self.set_aspect('equal')
            xlim = self.get_xlim()
            ylim = self.get_ylim()
            lo = min(xlim[0], ylim[0])
            hi = max(xlim[1], ylim[1])
            self.set_xlim(lo, hi)
            self.set_ylim(lo, hi)
        elif option == 'equal':
            self.set_aspect('equal')
        elif option == 'off':
            pass  # axes visibility not implemented yet
        elif option == 'on':
            pass

    # ------------------------------------------------------------------
    # Label outer
    # ------------------------------------------------------------------

    def label_outer(self):
        """Only show outer labels and tick labels for subplots.

        Hides x-axis labels/ticks if this is not a bottom-row subplot,
        and y-axis labels/ticks if this is not a left-column subplot.
        """
        pos = self._position
        # Determine grid position from (nrows, ncols, index) tuple
        if isinstance(pos, tuple) and len(pos) == 3:
            nrows, ncols, index = pos
            row = (index - 1) // ncols
            col = (index - 1) % ncols
            is_bottom = (row == nrows - 1)
            is_left = (col == 0)
        else:
            # Non-grid axes — keep all labels
            is_bottom = True
            is_left = True

        if not is_bottom:
            self._xticklabels_visible = False
            self._xlabel_visible = False

        if not is_left:
            self._yticklabels_visible = False
            self._ylabel_visible = False

    # ------------------------------------------------------------------
    # Twin axes
    # ------------------------------------------------------------------

    def twinx(self):
        """Create a twin Axes sharing the x-axis."""
        ax2 = Axes(self.figure, self._position)
        self.figure._axes.append(ax2)
        # Share x-axis
        shared = self._shared_x if self._shared_x else [self]
        shared.append(ax2)
        for a in shared:
            a._shared_x = shared
        # Copy current x-limits
        if self._xlim is not None:
            ax2._xlim = self._xlim
        return ax2

    def twiny(self):
        """Create a twin Axes sharing the y-axis."""
        ax2 = Axes(self.figure, self._position)
        self.figure._axes.append(ax2)
        # Share y-axis
        shared = self._shared_y if self._shared_y else [self]
        shared.append(ax2)
        for a in shared:
            a._shared_y = shared
        # Copy current y-limits
        if self._ylim is not None:
            ax2._ylim = self._ylim
        return ax2

    # ------------------------------------------------------------------
    # Batch setter
    # ------------------------------------------------------------------

    def set(self, **kwargs):
        """Batch property setter."""
        for k, v in kwargs.items():
            setter = getattr(self, f'set_{k}', None)
            if setter is not None:
                if isinstance(v, (tuple, list)) and k in ('xlim', 'ylim'):
                    setter(*v)
                else:
                    setter(v)

    # ------------------------------------------------------------------
    # Renderer draw path
    # ------------------------------------------------------------------

    def _compute_layout(self, fig_w_px, fig_h_px):
        ml, mr, mt, mb = 70, 20, 40, 50
        plot_x = ml
        plot_y = mt
        plot_w = fig_w_px - ml - mr
        plot_h = fig_h_px - mt - mb
        if plot_w <= 0 or plot_h <= 0:
            return None

        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()

        # Padding (same as current backends)
        dx = (xmax - xmin) or 1.0
        dy = (ymax - ymin) or 1.0
        xmin -= dx * 0.05
        xmax += dx * 0.05
        ymin -= dy * 0.05
        ymax += dy * 0.05

        return AxesLayout(plot_x, plot_y, plot_w, plot_h, xmin, xmax, ymin, ymax)

    def draw(self, renderer):
        layout = self._compute_layout(renderer.width, renderer.height)
        if layout is None:
            return

        px, py, pw, ph = layout.plot_x, layout.plot_y, layout.plot_w, layout.plot_h

        # Frame border
        renderer.draw_rect(px, py, pw, ph, '#000000', 'none')

        # Grid
        if self._grid:
            xticks = _nice_ticks(layout.xmin, layout.xmax, 8)
            yticks = _nice_ticks(layout.ymin, layout.ymax, 6)
            for t in xticks:
                tx = layout.sx(t)
                if px < tx < px + pw:
                    renderer.draw_line([tx, tx], [py, py + ph],
                                       '#dddddd', 0.5, '--')
            for t in yticks:
                ty = layout.sy(t)
                if py < ty < py + ph:
                    renderer.draw_line([px, px + pw], [ty, ty],
                                       '#dddddd', 0.5, '--')

        # Tick marks + labels
        xticks = _nice_ticks(layout.xmin, layout.xmax, 8)
        yticks = _nice_ticks(layout.ymin, layout.ymax, 6)
        for t in xticks:
            tx = layout.sx(t)
            if px <= tx <= px + pw:
                renderer.draw_line([tx, tx], [py + ph, py + ph + 5],
                                   '#000000', 1.0, '-')
                if self._xticklabels_visible:
                    renderer.draw_text(tx, py + ph + 18, _fmt_tick(t),
                                       11, '#333333', 'center')
        for t in yticks:
            ty = layout.sy(t)
            if py <= ty <= py + ph:
                renderer.draw_line([px - 5, px], [ty, ty],
                                   '#000000', 1.0, '-')
                if self._yticklabels_visible:
                    renderer.draw_text(px - 8, ty + 4, _fmt_tick(t),
                                       11, '#333333', 'right')

        # Clip for data area
        renderer.set_clip_rect(px, py, pw, ph)

        # Collect + sort all artists by zorder
        all_artists = []
        for line in self.lines:
            all_artists.append(line)
        for patch in self.patches:
            all_artists.append(patch)
        for coll in self.collections:
            all_artists.append(coll)
        for txt in self.texts:
            all_artists.append(txt)
        all_artists.sort(key=lambda a: a.get_zorder())

        for artist in all_artists:
            spanning = getattr(artist, '_spanning', None)
            if spanning:
                continue  # drawn separately below
            if hasattr(artist, 'draw') and callable(artist.draw):
                artist.draw(renderer, layout)

        # Spanning lines (axhline/axvline)
        for line in self.lines:
            if not line.get_visible():
                continue
            spanning = getattr(line, '_spanning', None)
            if spanning == 'horizontal':
                y_val = line._ydata[0]
                py_val = layout.sy(y_val)
                color = to_hex(line._color)
                renderer.draw_line([float(px), float(px + pw)], [py_val, py_val],
                                   color, float(line._linewidth), line._linestyle)
            elif spanning == 'vertical':
                x_val = line._xdata[0]
                px_val = layout.sx(x_val)
                color = to_hex(line._color)
                renderer.draw_line([px_val, px_val], [float(py), float(py + ph)],
                                   color, float(line._linewidth), line._linestyle)

        # Errorbar whiskers
        for container in self.containers:
            yerr_data = getattr(container, '_yerr_data', None)
            if yerr_data:
                x_list, y_list, yerr = yerr_data
                yerr_list = list(yerr) if hasattr(yerr, '__iter__') else [yerr] * len(x_list)
                color = '#000000'
                if hasattr(container, 'lines') and container.lines[0]:
                    color = to_hex(container.lines[0]._color)
                for i in range(len(x_list)):
                    err = yerr_list[i] if i < len(yerr_list) else yerr_list[-1]
                    cx = layout.sx(x_list[i])
                    y_lo = layout.sy(y_list[i] - err)
                    y_hi = layout.sy(y_list[i] + err)
                    renderer.draw_line([cx, cx], [y_lo, y_hi], color, 1.0, '-')
                    cap = 3
                    renderer.draw_line([cx - cap, cx + cap], [y_lo, y_lo], color, 1.0, '-')
                    renderer.draw_line([cx - cap, cx + cap], [y_hi, y_hi], color, 1.0, '-')

            xerr_data = getattr(container, '_xerr_data', None)
            if xerr_data:
                x_list, y_list, xerr = xerr_data
                xerr_list = list(xerr) if hasattr(xerr, '__iter__') else [xerr] * len(x_list)
                color = '#000000'
                if hasattr(container, 'lines') and container.lines[0]:
                    color = to_hex(container.lines[0]._color)
                for i in range(len(x_list)):
                    err = xerr_list[i] if i < len(xerr_list) else xerr_list[-1]
                    cy = layout.sy(y_list[i])
                    x_lo = layout.sx(x_list[i] - err)
                    x_hi = layout.sx(x_list[i] + err)
                    renderer.draw_line([x_lo, x_hi], [cy, cy], color, 1.0, '-')
                    cap = 3
                    renderer.draw_line([x_lo, x_lo], [cy - cap, cy + cap], color, 1.0, '-')
                    renderer.draw_line([x_hi, x_hi], [cy - cap, cy + cap], color, 1.0, '-')

        renderer.clear_clip()

        # Title
        if self._title:
            renderer.draw_text(px + pw / 2, py - 10, self._title,
                               14, '#000000', 'center')

        # Axis labels
        if self._xlabel and self._xlabel_visible:
            renderer.draw_text(px + pw / 2, renderer.height - 5,
                               self._xlabel, 12, '#333333', 'center')
        if self._ylabel and self._ylabel_visible:
            ty = py + ph / 2
            renderer.draw_text(15, ty, self._ylabel, 12, '#333333', 'center')

        # Legend
        if self._legend:
            self._draw_legend(renderer, px + pw - 10, py + 10)

    def _draw_legend(self, renderer, right_x, top_y):
        handles, labels = self.get_legend_handles_labels()
        if not labels:
            return
        lw = 120
        lh = len(labels) * 20 + 10
        lx = right_x - lw
        ly = top_y
        renderer.draw_rect(lx, ly, lw, lh, '#999999', '#ffffff')
        for i, (handle, label) in enumerate(zip(handles, labels)):
            iy = ly + 15 + i * 20
            color = '#000000'
            if hasattr(handle, 'get_color'):
                try:
                    color = to_hex(handle.get_color())
                except Exception:
                    pass
            renderer.draw_line([lx + 5, lx + 25], [iy, iy], color, 2.0, '-')
            renderer.draw_text(lx + 30, iy + 4, label, 11, '#333333', 'left')

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self):
        """Remove this axes from its figure."""
        if self.figure is not None:
            self.figure.delaxes(self)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def cla(self):
        """Clear the axes."""
        self._title = ''
        self._xlabel = ''
        self._ylabel = ''
        self._xlim = None
        self._ylim = None
        self._xticks = None
        self._yticks = None
        self._xticklabels = None
        self._yticklabels = None
        self._grid = False
        self._legend = False
        self._color_idx = 0
        self._facecolor = 'white'
        # Clear typed artist lists
        self.lines.clear()
        self.collections.clear()
        self.patches.clear()
        self.containers.clear()
        self.texts.clear()
        self.images.clear()
        # Reset axis state
        self._x_inverted = False
        self._y_inverted = False
        self._xscale = 'linear'
        self._yscale = 'linear'
        self._aspect = 'auto'
        # Note: do NOT reset _shared_x/_shared_y — shared axes persist across cla()
        self._xticklabels_visible = True
        self._yticklabels_visible = True
        self._xlabel_visible = True
        self._ylabel_visible = True
        # Reset tick params
        self._tick_params = {'x': {}, 'y': {}}

    def clear(self):
        self.cla()

    # ------------------------------------------------------------------
    # axhspan / axvspan
    # ------------------------------------------------------------------

    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
        """Add a horizontal span (rectangle) across the axes.

        Parameters
        ----------
        ymin, ymax : float
            Y-coordinates of the span in data coordinates.
        xmin, xmax : float, default 0..1
            X-coordinates of the span in axes fraction (0..1).
        **kwargs
            Additional kwargs passed to Rectangle (facecolor, alpha, etc.).
        """
        facecolor = kwargs.pop('facecolor', kwargs.pop('fc', None))
        if facecolor is None:
            facecolor = self._next_color()
        edgecolor = kwargs.pop('edgecolor', kwargs.pop('ec', 'none'))
        alpha = kwargs.pop('alpha', 0.5)
        label = kwargs.pop('label', '_nolegend_')

        rect = Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor=facecolor, edgecolor=edgecolor,
        )
        rect.set_alpha(alpha)
        rect.set_label(label)
        rect._spanning = 'hspan'
        rect.axes = self
        rect.figure = self.figure
        self.patches.append(rect)
        return rect

    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """Add a vertical span (rectangle) across the axes.

        Parameters
        ----------
        xmin, xmax : float
            X-coordinates of the span in data coordinates.
        ymin, ymax : float, default 0..1
            Y-coordinates of the span in axes fraction (0..1).
        **kwargs
            Additional kwargs passed to Rectangle (facecolor, alpha, etc.).
        """
        facecolor = kwargs.pop('facecolor', kwargs.pop('fc', None))
        if facecolor is None:
            facecolor = self._next_color()
        edgecolor = kwargs.pop('edgecolor', kwargs.pop('ec', 'none'))
        alpha = kwargs.pop('alpha', 0.5)
        label = kwargs.pop('label', '_nolegend_')

        rect = Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor=facecolor, edgecolor=edgecolor,
        )
        rect.set_alpha(alpha)
        rect.set_label(label)
        rect._spanning = 'vspan'
        rect.axes = self
        rect.figure = self.figure
        self.patches.append(rect)
        return rect

    # ------------------------------------------------------------------
    # bar_label
    # ------------------------------------------------------------------

    def bar_label(self, container, labels=None, fmt='%g', label_type='edge',
                  padding=0, **kwargs):
        """Add labels to bars in *container*.

        Parameters
        ----------
        container : BarContainer
        labels : list of str, optional
        fmt : str or callable
        label_type : {'edge', 'center'}
        padding : float or array-like
        """
        bars = container.patches

        # Validate fmt
        if not isinstance(fmt, (str, type(lambda: None))) and not callable(fmt):
            raise TypeError("fmt must be a str or callable, not %s" %
                            type(fmt).__name__)

        # Validate padding
        if hasattr(padding, '__iter__'):
            padding_list = list(padding)
            if len(padding_list) != len(bars):
                raise ValueError(
                    f"padding must be of length {len(bars)}, "
                    f"but has length {len(padding_list)}")
        else:
            padding_list = [padding] * len(bars)

        annotations = []
        for i, bar in enumerate(bars):
            # Determine the data value
            if labels is not None:
                text = str(labels[i])
            else:
                # Determine value from bar geometry
                # For vertical bars, value = height; for horizontal, value = width
                x0, y0 = bar.get_xy()
                w = bar.get_width()
                h = bar.get_height()

                # Detect orientation: vertical if abs(h) >= abs(w) or width ~ standard bar width
                is_vertical = True
                if hasattr(container, '_orientation'):
                    is_vertical = container._orientation == 'vertical'
                else:
                    # Heuristic: if all bars have x-extent < y-extent, horizontal
                    # Check if bar was created by barh (width > height typically)
                    if abs(w) > abs(h) and y0 != 0:
                        is_vertical = False

                if is_vertical:
                    value = h
                else:
                    value = w

                # Format the value
                if math.isnan(value):
                    text = ''
                elif isinstance(fmt, str):
                    if '%' in fmt:
                        text = fmt % value
                    else:
                        text = format(value, fmt)
                elif callable(fmt):
                    text = fmt(value)
                else:
                    text = str(value)

            pad = padding_list[i]
            x0, y0 = bar.get_xy()
            w = bar.get_width()
            h = bar.get_height()

            if label_type == 'center':
                # Place at center of bar in axes coordinates (0.5, 0.5)
                ann = Annotation(
                    text, xy=(0.5, 0.5),
                    ha='center', va='center',
                )
                ann.xyann = (pad, pad)
            else:
                # 'edge' — place at the end of the bar
                # Detect orientation
                is_vertical = abs(h) >= abs(w) or (y0 == 0 and x0 != 0) or (abs(w) < 2)
                # Better heuristic: check if this came from bar() or barh()
                # bar(): xy = (x - w/2, bottom), height = value
                # barh(): xy = (0, y - h/2), width = value
                if x0 == 0 or (abs(w) > abs(h) and abs(h) < 2):
                    # Horizontal bar (barh)
                    is_vertical = False

                if is_vertical:
                    x_pos = x0 + w / 2
                    h_val = h if not math.isnan(h) else 0
                    if h_val >= 0 or math.isnan(h):
                        y_pos = y0 + h_val
                        va = 'bottom'
                    else:
                        y_pos = y0 + h_val
                        va = 'top'
                    ann = Annotation(text, xy=(x_pos, y_pos),
                                     ha='center', va=va)
                    ann.xyann = (0, pad)
                else:
                    w_val = w if not math.isnan(w) else 0
                    y_pos = y0 + h / 2
                    if w_val >= 0 or math.isnan(w):
                        x_pos = x0 + w_val
                        ha = 'left'
                    else:
                        x_pos = x0 + w_val
                        ha = 'right'
                    ann = Annotation(text, xy=(x_pos, y_pos),
                                     ha=ha, va='center')
                    ann.xyann = (pad, 0)

            ann.axes = self
            ann.figure = self.figure
            self.texts.append(ann)
            annotations.append(ann)

        return annotations

    # ------------------------------------------------------------------
    # Facecolor
    # ------------------------------------------------------------------

    def set_facecolor(self, color):
        """Set the axes background color."""
        self._facecolor = color

    def get_facecolor(self):
        """Return the axes background color as an RGBA tuple."""
        from matplotlib.colors import to_rgba
        return to_rgba(self._facecolor)

    # alias used by upstream
    set_fc = set_facecolor
    get_fc = get_facecolor

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def get_position(self):
        """Return the axes position.

        Returns a Bbox-like object with x0, y0, width, height attributes.
        For subplot positions (nrows, ncols, index), computes the position.
        """
        pos = self._position
        if isinstance(pos, tuple) and len(pos) == 3:
            nrows, ncols, index = pos
            row = (index - 1) // ncols
            col = (index - 1) % ncols
            w = 1.0 / ncols
            h = 1.0 / nrows
            x0 = col * w
            y0 = 1.0 - (row + 1) * h
            return _BboxLike(x0, y0, w, h)
        if isinstance(pos, (tuple, list)) and len(pos) == 4:
            return _BboxLike(pos[0], pos[1], pos[2], pos[3])
        return _BboxLike(0, 0, 1, 1)

    def set_position(self, pos):
        """Set the axes position."""
        if isinstance(pos, (tuple, list)) and len(pos) == 4:
            self._position = tuple(pos)
        elif hasattr(pos, 'bounds'):
            self._position = pos.bounds

    # ------------------------------------------------------------------
    # Navigate
    # ------------------------------------------------------------------

    def set_navigate(self, b):
        """Set whether the axes responds to navigation commands."""
        self._navigate = b

    def get_navigate(self):
        """Return whether the axes responds to navigation commands."""
        return self._navigate

    def get_navigate_mode(self):
        """Return the navigation mode."""
        return getattr(self, '_navigate_mode', None)

    def set_navigate_mode(self, b):
        """Set the navigation mode."""
        self._navigate_mode = b

    def format_coord(self, x, y):
        """Return a format string for the (x, y) coordinate."""
        return f'x={x:g}, y={y:g}'

    def get_frame_on(self):
        """Return whether the axes frame is drawn."""
        return getattr(self, '_frame_on', True)

    def set_frame_on(self, b):
        """Set whether the axes frame is drawn."""
        self._frame_on = b

    def get_axisbelow(self):
        """Return whether axis ticks are below artists."""
        return getattr(self, '_axisbelow', True)

    def set_axisbelow(self, b):
        """Set whether axis ticks are below artists."""
        self._axisbelow = b

    # ------------------------------------------------------------------
    # imshow
    # ------------------------------------------------------------------

    def imshow(self, X, cmap=None, norm=None, aspect=None, interpolation=None,
               alpha=None, vmin=None, vmax=None, origin=None, extent=None,
               **kwargs):
        """Display data as an image.

        Parameters
        ----------
        X : array-like
            The image data. Can be 2D (grayscale) or 3D (RGB/RGBA).
        extent : (x0, x1, y0, y1), optional
            The bounding box in data coordinates.
        """
        from matplotlib.image import AxesImage

        # Determine shape from numpy or list
        if hasattr(X, 'shape') and hasattr(X, 'ndim'):
            shape = X.shape
            nrows = shape[0]
            ncols = shape[1] if len(shape) > 1 else 1
            data = X.tolist() if hasattr(X, 'tolist') else X
        elif hasattr(X, 'tolist'):
            data = X.tolist()
            nrows = len(data)
            ncols = len(data[0]) if nrows > 0 and hasattr(data[0], '__len__') else 1
        else:
            data = [list(row) if hasattr(row, '__iter__') else [row] for row in X]
            nrows = len(data)
            ncols = len(data[0]) if nrows > 0 else 0

        if extent is not None:
            x0, x1, y0, y1 = extent
        else:
            x0, x1 = -0.5, ncols - 0.5
            y0, y1 = nrows - 0.5, -0.5

        im = AxesImage(self, data=data, extent=(x0, x1, y0, y1),
                        cmap=cmap, norm=norm, interpolation=interpolation)
        if alpha is not None:
            im.set_alpha(alpha)

        label = kwargs.get('label')
        if label is not None:
            im.set_label(label)

        im.axes = self
        im.figure = self.figure
        self.images.append(im)

        # Set aspect to 'equal' by default for imshow (like upstream)
        if aspect is None:
            self.set_aspect('equal')
        else:
            self.set_aspect(aspect)

        return im

    # ------------------------------------------------------------------
    # pcolormesh (simplified)
    # ------------------------------------------------------------------

    def pcolormesh(self, *args, **kwargs):
        """Simplified pcolormesh: create a Rectangle covering the data range."""
        if len(args) == 1:
            C = args[0]
            if hasattr(C, 'shape') and hasattr(C, 'ndim') and C.ndim >= 2:
                nrows, ncols = C.shape[0], C.shape[1]
            elif hasattr(C, 'tolist'):
                data = C.tolist()
                nrows = len(data)
                ncols = len(data[0]) if nrows > 0 and hasattr(data[0], '__len__') else 1
            else:
                data = list(C)
                nrows = len(data)
                ncols = len(data[0]) if nrows > 0 and hasattr(data[0], '__len__') else 1
            x0, x1 = 0, ncols
            y0, y1 = 0, nrows
        elif len(args) == 3:
            X, Y, C = args
            x_flat = list(X.flat) if hasattr(X, 'flat') else (list(X) if not hasattr(X[0], '__iter__') else [v for row in X for v in row])
            y_flat = list(Y.flat) if hasattr(Y, 'flat') else (list(Y) if not hasattr(Y[0], '__iter__') else [v for row in Y for v in row])
            x0, x1 = min(x_flat), max(x_flat)
            y0, y1 = min(y_flat), max(y_flat)
            if hasattr(C, 'shape') and hasattr(C, 'ndim') and C.ndim >= 2:
                nrows, ncols = C.shape[0], C.shape[1]
            else:
                nrows = len(list(C))
                ncols = 1
        else:
            raise TypeError(f"pcolormesh takes 1 or 3 positional args, got {len(args)}")

        cmap = kwargs.get('cmap')
        color = kwargs.get('color') or self._next_color()
        color = to_hex(color)

        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                          facecolor=color, edgecolor='none')
        rect.axes = self
        rect.figure = self.figure
        self.patches.append(rect)
        return rect

    # ------------------------------------------------------------------
    # table
    # ------------------------------------------------------------------

    def table(self, **kwargs):
        """Add a table to the axes.

        Returns a Table object.
        """
        from matplotlib.table import table as _table_func
        tbl = _table_func(self, **kwargs)
        return tbl

    # ------------------------------------------------------------------
    # contour / contourf (stub)
    # ------------------------------------------------------------------

    def contour(self, *args, **kwargs):
        """Stub contour that stores data but doesn't compute contour lines."""
        return _ContourStub(self, args, kwargs)

    def contourf(self, *args, **kwargs):
        """Stub contourf that stores data but doesn't compute contour fills."""
        return _ContourStub(self, args, kwargs)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self):
        """Return a string representation of the Axes."""
        parts = []
        label = getattr(self, '_label_str', '')
        if label:
            parts.append(f"label='{label}'")
        if self._title:
            parts.append(f"title={{'center': '{self._title}'}}")
        if self._xlabel:
            parts.append(f"xlabel='{self._xlabel}'")
        if self._ylabel:
            parts.append(f"ylabel='{self._ylabel}'")
        inner = ', '.join(parts)
        if inner:
            return f"<Axes: {inner}>"
        return "<Axes: >"

    def set_label(self, s):
        """Set the axes label (for repr, not axis label)."""
        self._label_str = str(s) if s is not None else ''

    def get_label(self):
        """Get the axes label."""
        return getattr(self, '_label_str', '')

    # ------------------------------------------------------------------
    # get_children
    # ------------------------------------------------------------------

    def get_children(self):
        """Return a list of all artists contained in the Axes."""
        children = []
        children.extend(self.lines)
        children.extend(self.patches)
        children.extend(self.texts)
        children.extend(self.collections)
        children.extend(self.images)
        return children

    def findobj(self, match=None, include_self=True):
        """Find artist objects matching a criterion.

        Parameters
        ----------
        match : None, class, or callable
            If None, match all artists.
            If a class, match isinstance.
            If callable, match where callable returns True.
        include_self : bool
            Whether to include self.
        """
        result = []
        if include_self:
            if match is None:
                result.append(self)
            elif isinstance(match, type) and isinstance(self, match):
                result.append(self)
            elif callable(match) and not isinstance(match, type) and match(self):
                result.append(self)
        for child in self.get_children():
            if hasattr(child, 'findobj'):
                result.extend(child.findobj(match, include_self=True))
            else:
                if match is None:
                    result.append(child)
                elif isinstance(match, type) and isinstance(child, match):
                    result.append(child)
                elif callable(match) and not isinstance(match, type) and match(child):
                    result.append(child)
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _percentile(data, pct):
    """Simple percentile calculation (linear interpolation)."""
    sorted_d = sorted(data)
    n = len(sorted_d)
    if n == 0:
        return 0
    if n == 1:
        return sorted_d[0]
    k = (n - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_d[-1]
    return sorted_d[f] + (k - f) * (sorted_d[c] - sorted_d[f])


def _median(data):
    """Simple median calculation."""
    return _percentile(data, 50)


def _gaussian_kde(data, n_points=100):
    """Simple Gaussian kernel density estimate using Silverman's rule."""
    n = len(data)
    if n == 0:
        return [], []

    mean = sum(data) / n
    var = sum((v - mean) ** 2 for v in data) / n
    std = math.sqrt(var) if var > 0 else 1.0

    bw = 1.06 * std * (n ** -0.2) if std > 0 else 1.0

    lo = min(data) - 3 * bw
    hi = max(data) + 3 * bw
    step = (hi - lo) / (n_points - 1) if n_points > 1 else 1.0
    positions = [lo + i * step for i in range(n_points)]

    densities = []
    coeff = 1.0 / (n * bw * math.sqrt(2 * math.pi))
    for p in positions:
        total = 0.0
        for d in data:
            z = (p - d) / bw
            total += math.exp(-0.5 * z * z)
        densities.append(total * coeff)

    return positions, densities


def _parse_plot_args(args):
    """Parse positional args for plot(): (y,), (x, y), (x, y, fmt)."""
    fmt = ''
    if len(args) == 1:
        y = list(args[0])
        x = list(range(len(y)))
    elif len(args) >= 2:
        first, second = args[0], args[1]
        if isinstance(second, str):
            # plot(y, fmt)
            y = list(first)
            x = list(range(len(y)))
            fmt = second
        else:
            x = list(first)
            y = list(second)
            if len(x) != len(y):
                raise ValueError(
                    f"x and y must have same first dimension, "
                    f"but have shapes ({len(x)},) and ({len(y)},)")
            if len(args) >= 3 and isinstance(args[2], str):
                fmt = args[2]
    else:
        x, y = [], []
    return x, y, fmt


def _is_data_like(arg):
    """Return True if arg looks like array data (not a format string)."""
    if isinstance(arg, str):
        return False
    if hasattr(arg, '__iter__'):
        return True
    return isinstance(arg, (int, float))


def _parse_plot_args_multi(args):
    """Parse multi-group plot() args: x1,y1,[fmt1,] x2,y2,[fmt2,] ...

    Returns list of (x, y, fmt) tuples.
    """
    if len(args) == 0:
        return [([], [], '')]

    # Quick check: if only 1-3 args and they match simple forms, use old parser
    if len(args) <= 3:
        x, y, fmt = _parse_plot_args(args)
        return [(x, y, fmt)]

    # Multi-group parsing
    groups = []
    i = 0
    remaining = list(args)

    while i < len(remaining):
        # Try to consume one group: x, y, [fmt]
        if i >= len(remaining):
            break

        arg0 = remaining[i]

        # Single array left: plot(y)
        if i == len(remaining) - 1 and _is_data_like(arg0):
            y = list(arg0)
            x = list(range(len(y)))
            groups.append((x, y, ''))
            break

        # Two items: could be (y, fmt), (x, y), or start of longer group
        if i + 1 < len(remaining):
            arg1 = remaining[i + 1]

            if isinstance(arg1, str):
                # (y, fmt)
                y = list(arg0)
                x = list(range(len(y)))
                groups.append((x, y, arg1))
                i += 2
                continue
            elif _is_data_like(arg1):
                # (x, y, ...) — check if next is fmt
                x = list(arg0)
                y = list(arg1)
                if len(x) != len(y):
                    raise ValueError(
                        f"x and y must have same first dimension, "
                        f"but have shapes ({len(x)},) and ({len(y)},)")
                if i + 2 < len(remaining) and isinstance(remaining[i + 2], str):
                    fmt = remaining[i + 2]
                    groups.append((x, y, fmt))
                    i += 3
                else:
                    groups.append((x, y, ''))
                    i += 2
                continue

        # Single data item
        if _is_data_like(arg0):
            y = list(arg0)
            x = list(range(len(y)))
            groups.append((x, y, ''))
            i += 1
        else:
            i += 1

    if not groups:
        return [([], [], '')]
    return groups


def _auto_bins(data):
    """Compute automatic bin count using Sturges' rule."""
    n = len(data)
    if n == 0:
        return 10
    import math as _math
    return max(1, int(_math.ceil(_math.log2(n) + 1)))


def _validate_1d(data, name):
    """Raise ValueError if data has ndim > 1 (is a 2D array)."""
    if hasattr(data, 'ndim'):
        if data.ndim > 1:
            raise ValueError(
                f"'{name}' must be 1D, but has ndim={data.ndim}")
    elif hasattr(data, '__iter__') and not isinstance(data, str):
        # Check if it looks like a list of lists
        data_list = list(data)
        if data_list and hasattr(data_list[0], '__iter__') and not isinstance(data_list[0], str):
            raise ValueError(
                f"'{name}' must be 1D, but appears to be 2D")


class _BboxLike:
    """Lightweight Bbox stand-in returned by Axes.get_position()."""

    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x0 + width
        self.y1 = y0 + height
        self.width = width
        self.height = height
        self.bounds = (x0, y0, width, height)
        self.p0 = (x0, y0)
        self.p1 = (x0 + width, y0 + height)

    def __iter__(self):
        return iter(self.bounds)

    def __repr__(self):
        return (f"Bbox([[{self.x0}, {self.y0}], "
                f"[{self.x1}, {self.y1}]])")


class _ContourStub:
    """Minimal stub for contour/contourf return values."""

    def __init__(self, ax, args, kwargs):
        self.ax = ax
        self.levels = kwargs.get('levels', [])
        self.collections = []
