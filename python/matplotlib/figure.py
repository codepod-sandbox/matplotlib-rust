"""
matplotlib.figure — Figure class.
"""

import math

from matplotlib.axes import Axes
from matplotlib.gridspec import SubplotSpec
from matplotlib.text import Text


class _AlignGroup:
    """Stub for align_xlabels/align_ylabels grouping."""

    def __init__(self):
        self._groups = []

    def join(self, ax, *args):
        """Join axes into the group."""
        pass

    def get_siblings(self, ax):
        """Return sibling axes in the same group."""
        return [ax]


class _FigureCanvas:
    """Minimal canvas stub for Figure."""
    def __init__(self, figure):
        self.figure = figure
    def draw(self):
        """Trigger tick formatter updates on all axes (needed for tests)."""
        if self.figure is None:
            return
        for ax in getattr(self.figure, '_axes', []):
            for axis in [getattr(ax, 'xaxis', None), getattr(ax, 'yaxis', None)]:
                if axis is None:
                    continue
                try:
                    locs = list(axis._major_locator())
                except Exception:
                    continue
                fmt = axis._major_formatter
                if hasattr(fmt, 'set_locs') and locs:
                    try:
                        fmt.set_locs(locs)
                    except Exception:
                        pass
    def draw_idle(self):
        self.draw()

    def is_saving(self):
        """Return whether the canvas is currently saving."""
        return False

    def get_renderer(self):
        """Return a renderer (stub)."""
        return None

    @property
    def toolbar(self):
        return None

    def mpl_disconnect(self, cid):
        pass

    def mpl_connect(self, s, func):
        return 0


def _validate_figsize(w, h):
    for val, name in [(w, 'width'), (h, 'height')]:
        if math.isnan(val):
            raise ValueError(f"figure size must be finite, not {name}={val}")
        if math.isinf(val):
            raise ValueError(f"figure size must be finite, not {name}={val}")
        if val <= 0:
            raise ValueError(f"figure size must be positive, not {name}={val}")


class Figure:
    """Top-level container for a matplotlib plot."""

    def __init__(self, figsize=None, dpi=100, facecolor=None, edgecolor=None, **kwargs):
        self.figsize = figsize or (6.4, 4.8)
        _validate_figsize(self.figsize[0], self.figsize[1])
        self.dpi = dpi
        self._facecolor = facecolor if facecolor is not None else 'white'
        self._edgecolor = edgecolor if edgecolor is not None else 'white'
        self._axes = []
        self._suptitle = None
        self._supxlabel = None
        self._supylabel = None
        self._label = ''
        self.number = None
        self.stale = True
        self.texts = []
        self._current_ax = None
        self.canvas = _FigureCanvas(self)
        self.suppressComposite = None
        self._setup_transforms()

    def _setup_transforms(self):
        """Set up figure-level transform stubs required by _AxesBase."""
        from matplotlib.transforms import (
            BboxTransformTo, Bbox, IdentityTransform, Affine2D
        )
        # transSubfigure: maps [0,1]x[0,1] subfigure coords to display coords
        # For our purposes, a simple identity-like transform suffices.
        w, h = self.figsize
        dpi = self.dpi
        self.bbox_inches = Bbox([[0, 0], [w, h]])
        self.bbox = Bbox([[0, 0], [w * dpi, h * dpi]])
        self.transSubfigure = BboxTransformTo(self.bbox)
        self.transFigure = self.transSubfigure
        # dpi_scale_trans: scale from inches to display points
        self.dpi_scale_trans = Affine2D().scale(dpi)
        # align label groups for align_xlabels/align_ylabels support
        self._align_label_groups = {'x': _AlignGroup(), 'y': _AlignGroup(),
                                     'title': _AlignGroup()}

    # ------------------------------------------------------------------
    # Axes management
    # ------------------------------------------------------------------

    def add_subplot(self, *args, **kwargs):
        """Add an Axes to the figure.

        Accepts:
            add_subplot(nrows, ncols, index)
            add_subplot(SubplotSpec)
            add_subplot(existing_axes)
            add_subplot(NCI)  -- 3-digit integer
        """
        # Accept an existing Axes instance
        if len(args) == 1 and isinstance(args[0], Axes):
            ax = args[0]
            if ax in self._axes:
                self._current_ax = ax
                return ax
            self._axes.append(ax)
            self._current_ax = ax
            return ax

        if len(args) == 1 and isinstance(args[0], SubplotSpec):
            ss = args[0]
            gs = ss.get_gridspec()
            nrows, ncols = gs.nrows, gs.ncols
            # Compute flat index from rowspan/colspan
            r0, r1 = ss.rowspan[0], ss.rowspan[1]
            c0, c1 = ss.colspan[0], ss.colspan[1]
            index = r0 * ncols + c0 + 1
            pos = (nrows, ncols, index)
        elif len(args) == 1 and isinstance(args[0], int) and args[0] >= 100:
            # 3-digit form: 211 -> (2, 1, 1)
            n = args[0]
            nrows, ncols, index = n // 100, (n % 100) // 10, n % 10
            if nrows < 1 or ncols < 1:
                raise ValueError(
                    f"Number of rows and columns must be > 0, got {nrows} and {ncols}"
                )
            if index < 1 or index > nrows * ncols:
                raise ValueError(
                    f"Subplot index {index} is out of range 1..{nrows * ncols}"
                )
            pos = (nrows, ncols, index)
        elif len(args) == 3:
            nrows, ncols, index = args
            if nrows < 1 or ncols < 1:
                raise ValueError(
                    f"Number of rows and columns must be > 0, got {nrows} and {ncols}"
                )
            if index < 1 or index > nrows * ncols:
                raise ValueError(
                    f"Subplot index {index} is out of range 1..{nrows * ncols}"
                )
            pos = (nrows, ncols, index)
        elif len(args) == 0:
            pos = (1, 1, 1)
        else:
            nrows = args[0] if len(args) > 0 else kwargs.get('nrows', 1)
            ncols = args[1] if len(args) > 1 else kwargs.get('ncols', 1)
            index = args[2] if len(args) > 2 else kwargs.get('index', 1)
            pos = (nrows, ncols, index)

        # Pass nrows, ncols, index as separate positional args so that
        # _AxesBase.__init__ can call SubplotSpec._from_subplot_args(fig, args)
        nrows, ncols, index = pos
        ax = Axes(self, nrows, ncols, index, **kwargs)
        self._axes.append(ax)
        self._current_ax = ax
        return ax

    def add_axes(self, rect=None, **kwargs):
        """Add an Axes at position *rect* [left, bottom, width, height].

        If *rect* is an Axes instance, add it (or re-activate it).
        If *rect* is None, defaults to [0, 0, 1, 1].
        """
        if isinstance(rect, Axes):
            ax = rect
            if ax in self._axes:
                self._current_ax = ax
                return ax
            self._axes.append(ax)
            self._current_ax = ax
            return ax
        if rect is None:
            rect = [0, 0, 1, 1]
        ax = Axes(self, tuple(rect))
        self._axes.append(ax)
        self._current_ax = ax
        return ax

    def gca(self):
        """Get current axes, or create one if none exist."""
        if self._current_ax is not None and self._current_ax in self._axes:
            return self._current_ax
        if not self._axes:
            return self.add_subplot(1, 1, 1)
        return self._axes[-1]

    def sca(self, ax):
        """Set the current axes to *ax*.

        Sets _current_ax without reordering the axes list.
        """
        self._current_ax = ax

    def delaxes(self, ax):
        """Remove the Axes *ax* from this figure."""
        if ax in self._axes:
            self._axes.remove(ax)

    def get_axes(self):
        """Return a list of Axes in this figure."""
        return list(self._axes)

    @property
    def axes(self):
        return list(self._axes)

    # ------------------------------------------------------------------
    # Figure appearance
    # ------------------------------------------------------------------

    def get_facecolor(self):
        """Return the figure background color as an RGBA tuple."""
        from matplotlib.colors import to_rgba
        fc = getattr(self, '_facecolor', 'white')
        return to_rgba(fc)

    def set_facecolor(self, color):
        """Set the figure background color."""
        self._facecolor = color

    def get_edgecolor(self):
        """Return the figure edge color as an RGBA tuple."""
        from matplotlib.colors import to_rgba
        ec = getattr(self, '_edgecolor', 'white')
        return to_rgba(ec)

    def set_edgecolor(self, color):
        """Set the figure edge color."""
        self._edgecolor = color

    # ------------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------------

    def suptitle(self, t, **kwargs):
        """Set a centered suptitle for the figure.

        Returns the Text object.
        """
        self._suptitle = t
        txt = Text(0.5, 0.98, t, **kwargs)
        self.texts.append(txt)
        self.stale = True
        return txt

    def get_suptitle(self):
        """Return the figure suptitle string, or '' if not set."""
        return self._suptitle if self._suptitle is not None else ''

    def supxlabel(self, t, **kwargs):
        """Set the supxlabel for the figure."""
        self._supxlabel = t
        self.stale = True
        return t

    def get_supxlabel(self):
        """Return the figure supxlabel string, or '' if not set."""
        return self._supxlabel if self._supxlabel is not None else ''

    def supylabel(self, t, **kwargs):
        """Set the supylabel for the figure."""
        self._supylabel = t
        self.stale = True
        return t

    def get_supylabel(self):
        """Return the figure supylabel string, or '' if not set."""
        return self._supylabel if self._supylabel is not None else ''

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def set_size_inches(self, w, h=None):
        """Set the figure size in inches.

        Accepts ``set_size_inches(w, h)`` or ``set_size_inches((w, h))``.
        """
        if h is None:
            # w is a (w, h) tuple
            w, h = w
        _validate_figsize(float(w), float(h))
        self.figsize = (float(w), float(h))
        self.stale = True

    def get_size_inches(self):
        """Return the figure size as ``(width, height)`` in inches."""
        return tuple(self.figsize)

    def set_figwidth(self, val):
        """Set the figure width in inches."""
        self.figsize = (float(val), self.figsize[1])
        self.stale = True

    def set_figheight(self, val):
        """Set the figure height in inches."""
        self.figsize = (self.figsize[0], float(val))
        self.stale = True

    def get_figwidth(self):
        """Return the figure width in inches."""
        return self.figsize[0]

    def get_figheight(self):
        """Return the figure height in inches."""
        return self.figsize[1]

    # ------------------------------------------------------------------
    # DPI
    # ------------------------------------------------------------------

    def get_dpi(self):
        """Return the figure dpi."""
        return self.dpi

    def set_dpi(self, val):
        """Set the figure dpi."""
        self.dpi = val
        self.stale = True

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------

    def get_label(self):
        """Return the figure label."""
        return self._label

    def set_label(self, label):
        """Set the figure label."""
        self._label = str(label)

    # ------------------------------------------------------------------
    # Layout / clearing
    # ------------------------------------------------------------------

    def tight_layout(self, **kwargs):
        """Adjust subplot parameters for a tight layout.

        No-op placeholder in this implementation.
        """
        pass

    def text(self, x, y, s, **kwargs):
        """Add text to the figure at position (x, y).

        Returns the Text object.
        """
        txt = Text(x, y, s, **kwargs)
        self.texts.append(txt)
        return txt

    def clear(self):
        """Clear the figure — remove all axes and reset suptitle."""
        self._axes.clear()
        self._suptitle = None
        self.texts = []
        self._current_ax = None
        self.stale = True

    def clf(self):
        """Clear the figure (alias for :meth:`clear`)."""
        self.clear()

    def legend(self, *args, **kwargs):
        """Add a legend to the figure.

        Collects legend handles/labels from all axes and stores them.
        Returns a Legend object.
        """
        from matplotlib.legend import Legend

        self._has_legend = True
        # Collect from all axes
        handles = kwargs.pop('handles', None)
        labels = kwargs.pop('labels', None)

        if len(args) == 2:
            handles, labels = args
        elif len(args) == 1:
            labels = list(args[0])

        if handles is None or labels is None:
            h_all = []
            l_all = []
            seen = set()
            for ax in self._axes:
                h, l = ax.get_legend_handles_labels()
                for hi, li in zip(h, l):
                    if li not in seen:
                        h_all.append(hi)
                        l_all.append(li)
                        seen.add(li)
            if handles is None:
                handles = h_all
            if labels is None:
                labels = l_all

        self._legend_handles = handles
        self._legend_labels = labels

        leg = Legend(self, handles, labels, **kwargs)
        self._legend_obj = leg
        return leg

    def draw_without_rendering(self):
        """No-op placeholder for layout engine compatibility."""
        pass

    def subplots(self, nrows=1, ncols=1, **kwargs):
        """Add a set of subplots to this figure.

        Returns
        -------
        ax or array of ax
        """
        import numpy as np
        sharex = kwargs.pop('sharex', False)
        sharey = kwargs.pop('sharey', False)
        squeeze = kwargs.pop('squeeze', True)

        if nrows == 1 and ncols == 1:
            ax = self.add_subplot(1, 1, 1)
            if squeeze:
                return ax
            else:
                return np.array([[ax]])

        all_axes = []
        axes_grid = []
        for r in range(nrows):
            row = []
            for c in range(ncols):
                ax = self.add_subplot(nrows, ncols, r * ncols + c + 1)
                row.append(ax)
                all_axes.append(ax)
            axes_grid.append(row)

        if sharex and len(all_axes) > 1:
            ref = all_axes[0]
            for ax in all_axes[1:]:
                try:
                    ax.sharex(ref)
                except Exception:
                    ax._shared_x = all_axes
        if sharey and len(all_axes) > 1:
            ref = all_axes[0]
            for ax in all_axes[1:]:
                try:
                    ax.sharey(ref)
                except Exception:
                    ax._shared_y = all_axes

        axes_arr = np.array(axes_grid)  # shape (nrows, ncols)

        if not squeeze:
            return axes_arr

        # squeeze: remove axes with length 1
        if nrows == 1 and ncols == 1:
            return axes_arr[0, 0]
        elif nrows == 1:
            return axes_arr[0, :]  # 1D array of ncols axes
        elif ncols == 1:
            return axes_arr[:, 0]  # 1D array of nrows axes
        else:
            return axes_arr  # 2D array

    def get_children(self):
        """Return list of children artists."""
        children = list(self._axes)
        children.extend(self.texts)
        return children

    def get_constrained_layout(self):
        """Return whether constrained layout is active."""
        return getattr(self, '_constrained_layout', False)

    def set_constrained_layout(self, constrained):
        """Set constrained layout."""
        self._constrained_layout = constrained

    def get_tight_layout(self):
        """Return whether tight layout is active."""
        return getattr(self, '_tight_layout', False)

    def set_tight_layout(self, tight):
        """Set tight layout."""
        self._tight_layout = tight

    def align_xlabels(self, axs=None):
        """Align x-axis labels (no-op)."""
        pass

    def align_ylabels(self, axs=None):
        """Align y-axis labels (no-op)."""
        pass

    def align_labels(self, axs=None):
        """Align axis labels (no-op)."""
        pass

    def colorbar(self, mappable, ax=None, cax=None, **kwargs):
        """Add a colorbar to the figure."""
        from matplotlib.colorbar import Colorbar
        cbar_ax = self.add_subplot(111) if cax is None else cax
        cbar = Colorbar(cbar_ax, mappable, **kwargs)
        cbar.ax = cbar_ax
        # Set the ylim from the mappable's norm if available
        norm = getattr(mappable, 'norm', None)
        if norm is not None:
            vmin = getattr(norm, 'vmin', None)
            vmax = getattr(norm, 'vmax', None)
            if vmin is not None and vmax is not None:
                cbar_ax.set_ylim(vmin, vmax)
        return cbar

    def add_gridspec(self, nrows=1, ncols=1, **kwargs):
        """Add a GridSpec to the figure."""
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows, ncols, figure=self, **kwargs)
        return gs

    def set_layout_engine(self, layout=None, **kwargs):
        """Set the layout engine (no-op stub)."""
        self._layout_engine = layout

    def get_layout_engine(self):
        """Get the layout engine."""
        return getattr(self, '_layout_engine', None)

    # ------------------------------------------------------------------
    # Renderer draw path
    # ------------------------------------------------------------------

    def draw(self, renderer):
        # White background
        renderer.draw_rect(0, 0, renderer.width, renderer.height, 'none', '#ffffff')

        # Draw all axes
        for ax in self._axes:
            ax.draw(renderer)

        # Suptitle
        if self._suptitle:
            renderer.draw_text(renderer.width / 2, 20, self._suptitle,
                               14, '#000000', 'center')

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def savefig(self, fname, *, format=None, dpi=None, **kwargs):
        """Save figure to *fname*.  Format inferred from extension if not given.

        *fname* may be a file path (str) or a file-like object (BytesIO/StringIO).
        """
        import io as _io
        dpi = dpi or self.dpi
        # Determine format
        if format is None:
            if isinstance(fname, str):
                if fname.lower().endswith('.png'):
                    format = 'png'
                else:
                    format = 'svg'
            else:
                format = 'png'  # default for file-like objects

        w_px = int(self.figsize[0] * dpi)
        h_px = int(self.figsize[1] * dpi)

        if format == 'png':
            from matplotlib._pil_backend import RendererPIL
            renderer = RendererPIL(w_px, h_px, dpi)
        else:
            from matplotlib._svg_backend import RendererSVG
            renderer = RendererSVG(w_px, h_px, dpi)

        self.draw(renderer)
        result = renderer.get_result()

        # Write to file path or file-like object
        if hasattr(fname, 'write'):
            if isinstance(result, bytes):
                fname.write(result)
            else:
                fname.write(result)
        elif isinstance(result, bytes):
            with open(fname, 'wb') as f:
                f.write(result)
        else:
            with open(fname, 'w') as f:
                f.write(result)

    def to_svg(self, dpi=None):
        """Render the figure to an SVG string and return it."""
        dpi = dpi or self.dpi
        w_px = int(self.figsize[0] * dpi)
        h_px = int(self.figsize[1] * dpi)
        from matplotlib._svg_backend import RendererSVG
        renderer = RendererSVG(w_px, h_px, dpi)
        self.draw(renderer)
        return renderer.get_result()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        w, h = self.figsize
        # Format dimensions similarly to real matplotlib: WxH
        # Use integer representation when values are whole numbers
        w_str = str(int(w * self.dpi))
        h_str = str(int(h * self.dpi))
        n = len(self._axes)
        return f'<Figure size {w_str}x{h_str} with {n} Axes>'
