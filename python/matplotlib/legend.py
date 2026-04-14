# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
"""
matplotlib.legend --- Legend artist.
"""

from matplotlib.text import Text
from matplotlib.colors import to_hex
from matplotlib import _docstring
from matplotlib.artist import Artist

# Register docstring interpolation keys used by axes/_axes.py
_legend_kw_axes_st = """
loc : str or pair of floats, default: :rc:`legend.loc`
    The location of the legend.
"""
_docstring.interpd.register(_legend_kw_axes=_legend_kw_axes_st)
_docstring.interpd.register(_legend_kw_figure=_legend_kw_axes_st)
_docstring.interpd.register(_legend_kw_doc=_legend_kw_axes_st)
_docstring.interpd.register(_legend_kw_set_loc_doc=_legend_kw_axes_st)


_LOC_MAP = {
    0: 'best', 1: 'upper right', 2: 'upper left',
    3: 'lower left', 4: 'lower right', 5: 'right',
    6: 'center left', 7: 'center right',
    8: 'lower center', 9: 'upper center', 10: 'center',
}
_ROW_H = 20   # pixels per legend row
_PAD = 8      # padding inside box
_SWATCH_W = 22  # width of colour swatch
_MIN_W = 80   # minimum legend box width


class LegendText:
    """Minimal text-like proxy for a legend entry label."""
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text

    def set_text(self, s):
        self._text = s


class Legend(Artist):
    """A legend for an Axes.

    Parameters
    ----------
    parent : Axes
        The parent axes.
    handles : list of Artist
        Legend handles.
    labels : list of str
        Legend labels.
    loc : str or int, default 'best'
        Legend location.
    title : str, default ''
        Legend title.
    """

    # Location codes matching upstream matplotlib
    codes = {
        'best': 0,
        'upper right': 1,
        'upper left': 2,
        'lower left': 3,
        'lower right': 4,
        'right': 5,
        'center left': 6,
        'center right': 7,
        'lower center': 8,
        'upper center': 9,
        'center': 10,
    }

    def __init__(self, parent, handles, labels, *,
                 loc='best', ncol=1, ncols=None,
                 bbox_to_anchor=None, framealpha=0.8,
                 frameon=None, fontsize=None, title_fontsize=None,
                 title=None, **kwargs):
        super().__init__()
        self._parent = parent
        self._ax = parent
        self._handles = list(handles) if handles else []
        self._labels = list(labels) if labels else []
        self._loc = _LOC_MAP.get(loc, loc) if isinstance(loc, int) else loc
        self._ncol = ncols if ncols is not None else ncol
        self._bbox_to_anchor = bbox_to_anchor
        self._framealpha = framealpha
        self._frameon = frameon if frameon is not None else True
        self._fontsize = fontsize or 11
        self._title_fontsize = title_fontsize
        self._title_text = LegendText(title) if title else LegendText('')
        self._title = Text(0, 0, title or '')
        self._visible = True
        self._draggable = False
        self._shadow = kwargs.get('shadow', False)
        self._fancybox = kwargs.get('fancybox', None)
        self._edgecolor = kwargs.get('edgecolor', None)
        self._facecolor = kwargs.get('facecolor', None)

        # Build LegendText list for get_texts() API
        self._texts = [LegendText(lbl) for lbl in labels]
        # Build Text objects for each label (alternate API)
        self._text_objects = [Text(0, 0, l) for l in self._labels]

        # _loc_real for upstream compatibility
        self._loc_real = self._loc

    # --- API ---
    def get_texts(self):
        """Return a list of LegendText/Text objects for the labels."""
        return self._texts

    def get_title(self):
        """Return the title Text object."""
        return self._title_text

    def set_title(self, title, prop=None):
        """Set the legend title."""
        self._title = Text(0, 0, title)
        self._title_text = LegendText(title or '')

    def get_loc(self):
        return self._loc

    def set_loc(self, loc):
        """Set the legend location."""
        self._loc = loc

    def get_ncol(self):
        return self._ncol

    @property
    def _ncols(self):
        return self._ncol

    def get_handles(self):
        return list(self._handles)

    def get_frame(self):
        """Return the legend frame (a stub Rectangle)."""
        from matplotlib.patches import Rectangle
        return Rectangle((0, 0), 1, 1)

    def get_lines(self):
        """Return legend line handles."""
        from matplotlib.lines import Line2D
        return [h for h in self._handles if isinstance(h, Line2D)]

    def get_patches(self):
        """Return legend patch handles."""
        from matplotlib.patches import Patch
        return [h for h in self._handles if isinstance(h, Patch)]

    def get_visible(self):
        return self._visible

    def set_visible(self, b):
        self._visible = b

    def set_draggable(self, state, use_blit=False, update='loc'):
        """Enable or disable dragging."""
        self._draggable = state

    def get_draggable(self):
        return self._draggable

    def remove(self):
        """Remove the legend."""
        if hasattr(self._parent, '_legend_obj'):
            self._parent._legend_obj = None
            self._parent._legend = False

    def get_frame_on(self):
        return self._frameon

    def set_frame_on(self, b):
        self._frameon = b

    @property
    def legendHandles(self):
        """Deprecated: use legend_handles instead."""
        return self._handles

    @property
    def legend_handles(self):
        return self._handles

    def get_label(self):
        return getattr(self, '_label', '')

    def set_label(self, s):
        self._label = str(s) if s is not None else ''

    def get_shadow(self):
        return self._shadow

    def set_shadow(self, shadow):
        self._shadow = shadow

    def get_fancybox(self):
        return self._fancybox

    def set_fancybox(self, b):
        self._fancybox = b

    def get_framealpha(self):
        return self._framealpha

    def set_framealpha(self, alpha):
        self._framealpha = alpha

    def get_edgecolor(self):
        return self._edgecolor

    def set_edgecolor(self, color):
        self._edgecolor = color

    def get_facecolor(self):
        return self._facecolor

    def set_facecolor(self, color):
        self._facecolor = color

    # --- Geometry ---
    def _box_size(self):
        """Compute (width, height) of the legend box in pixels."""
        n = len(self._labels)
        if n == 0:
            return (0, 0)
        title_h = _ROW_H if self._title_text.get_text() else 0
        nrows = -(-n // self._ncol)  # ceiling division
        h = title_h + nrows * _ROW_H + _PAD
        # estimate width: swatch + longest label text
        max_len = max((len(lbl) for lbl in self._labels), default=0)
        col_w = _SWATCH_W + max_len * 6 + _PAD  # ~6px per char
        w = max(_MIN_W, col_w * self._ncol + _PAD)
        return (w, h)

    def _box_origin(self, layout):
        """Compute (lx, ly) top-left of legend box using loc."""
        px, py = layout.plot_x, layout.plot_y
        pw, ph = layout.plot_w, layout.plot_h
        w, h = self._box_size()

        if self._bbox_to_anchor is not None:
            bta = self._bbox_to_anchor
            if len(bta) >= 2:
                return (px + bta[0] * pw - w, py + bta[1] * ph - h)

        margin = 5
        loc = self._loc if self._loc != 'best' else 'upper right'

        if loc in ('upper right', 'right'):
            return (px + pw - w - margin, py + margin)
        elif loc == 'upper left':
            return (px + margin, py + margin)
        elif loc == 'lower left':
            return (px + margin, py + ph - h - margin)
        elif loc == 'lower right':
            return (px + pw - w - margin, py + ph - h - margin)
        elif loc == 'upper center':
            return (px + pw / 2 - w / 2, py + margin)
        elif loc == 'lower center':
            return (px + pw / 2 - w / 2, py + ph - h - margin)
        elif loc == 'center left':
            return (px + margin, py + ph / 2 - h / 2)
        elif loc == 'center right':
            return (px + pw - w - margin, py + ph / 2 - h / 2)
        elif loc == 'center':
            return (px + pw / 2 - w / 2, py + ph / 2 - h / 2)
        return (px + pw - w - margin, py + margin)  # fallback upper right

    def draw(self, renderer, layout):
        """Draw legend onto renderer."""
        if not self._labels:
            return

        w, h = self._box_size()
        lx, ly = self._box_origin(layout)

        # Frame
        renderer.draw_rect(lx, ly, w, h, '#999999', '#ffffff')

        # Title
        row = 0
        title = self._title_text.get_text()
        if title:
            renderer.draw_text(
                lx + _PAD, ly + _PAD + row * _ROW_H + 13,
                title, self._fontsize + 1, '#000000', 'left',
            )
            row += 1

        # Entries (simple single-column for now; ncol support is additive)
        for i, (handle, label) in enumerate(
                zip(self._handles, self._labels)):
            col = i % self._ncol
            r = row + i // self._ncol
            ey = ly + _PAD + r * _ROW_H + _ROW_H // 2
            ex = lx + _PAD + col * (w // self._ncol)

            # Colour swatch (line or patch)
            color = '#555555'
            if hasattr(handle, 'get_color'):
                try:
                    color = to_hex(handle.get_color())
                except Exception:
                    pass
            elif hasattr(handle, 'get_facecolor'):
                try:
                    fc = handle.get_facecolor()
                    if len(fc) == 4:
                        color = to_hex(fc[:3])
                except Exception:
                    pass

            renderer.draw_line(
                [ex, ex + _SWATCH_W - 4], [ey, ey],
                color, 2.0, '-', opacity=1.0,
            )
            renderer.draw_text(
                ex + _SWATCH_W, ey + 4,
                self._texts[i].get_text(),
                self._fontsize, '#333333', 'left',
            )

    def __repr__(self):
        return f"<Legend>"


def _get_legend_handles(axs, legend_handler_map=None):
    """Return handles (artists) that can appear in legend from *axs*."""
    handles_seen = set()
    for ax in axs:
        artists = []
        if hasattr(ax, 'get_lines'):
            artists.extend(ax.get_lines())
        if hasattr(ax, 'collections'):
            artists.extend(ax.collections)
        if hasattr(ax, 'patches'):
            artists.extend(ax.patches)
        for artist in artists:
            if id(artist) not in handles_seen:
                handles_seen.add(id(artist))
                yield artist


def _get_legend_handles_labels(axs, legend_handler_map=None):
    """Return handles and labels for legend."""
    handles = []
    labels = []
    for handle in _get_legend_handles(axs, legend_handler_map):
        if hasattr(handle, 'get_label'):
            label = handle.get_label()
            if label and not label.startswith('_'):
                handles.append(handle)
                labels.append(label)
    return handles, labels


def _parse_legend_args(axs, *args, handles=None, labels=None, **kwargs):
    """Parse legend arguments from axes or figure."""
    # If both handles and labels provided as kwargs
    if handles is not None and labels is not None:
        return list(handles), list(labels), kwargs

    # If handles only
    if handles is not None and labels is None:
        labels = [h.get_label() for h in handles
                  if hasattr(h, 'get_label')]
        return list(handles), labels, kwargs

    # If labels only
    if labels is not None and handles is None:
        # Try to get handles from axs
        all_handles = []
        for ax in axs:
            if hasattr(ax, 'get_lines'):
                all_handles.extend(ax.get_lines())
            if hasattr(ax, 'patches'):
                all_handles.extend(ax.patches)
        handles = all_handles[:len(labels)] if all_handles else []
        return list(handles), list(labels), kwargs

    # Parse positional args
    if len(args) == 0:
        # Auto-detect from axs
        all_handles, all_labels = [], []
        for ax in axs:
            if hasattr(ax, 'get_legend_handles_labels'):
                h, l = ax.get_legend_handles_labels()
                all_handles.extend(h)
                all_labels.extend(l)
            else:
                if hasattr(ax, 'get_lines'):
                    for line in ax.get_lines():
                        lbl = line.get_label() if hasattr(line, 'get_label') else ''
                        if lbl and not lbl.startswith('_'):
                            all_handles.append(line)
                            all_labels.append(lbl)
        return all_handles, all_labels, kwargs
    elif len(args) == 1:
        # labels only
        labels = args[0]
        all_handles = []
        for ax in axs:
            if hasattr(ax, 'get_lines'):
                all_handles.extend(ax.get_lines())
        handles = all_handles[:len(labels)] if all_handles else []
        return list(handles), list(labels), kwargs
    else:
        # handles, labels
        return list(args[0]), list(args[1]), kwargs
