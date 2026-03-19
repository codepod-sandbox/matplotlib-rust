"""
matplotlib.legend --- Legend artist.
"""

from matplotlib.text import Text


class Legend:
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

    def __init__(self, parent, handles, labels, loc='best', title='',
                 frameon=None, fontsize=None, title_fontsize=None,
                 ncol=1, ncols=None, **kwargs):
        self._parent = parent
        self._handles = list(handles) if handles else []
        self._labels = list(labels) if labels else []
        self._loc = loc
        self._title = Text(0, 0, title or '')
        self._frameon = frameon if frameon is not None else True
        self._fontsize = fontsize
        self._title_fontsize = title_fontsize
        self._ncol = ncols if ncols is not None else ncol
        self._visible = True
        self._draggable = False
        self._shadow = kwargs.get('shadow', False)
        self._fancybox = kwargs.get('fancybox', None)
        self._framealpha = kwargs.get('framealpha', None)
        self._edgecolor = kwargs.get('edgecolor', None)
        self._facecolor = kwargs.get('facecolor', None)

        # Build Text objects for each label
        self._text_objects = [Text(0, 0, l) for l in self._labels]

    def get_texts(self):
        """Return a list of Text objects for the labels."""
        return list(self._text_objects)

    def get_title(self):
        """Return the title Text object."""
        return self._title

    def set_title(self, title, prop=None):
        """Set the legend title."""
        self._title = Text(0, 0, title)

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

    def get_loc(self):
        return self._loc

    def set_loc(self, loc):
        """Set the legend location."""
        self._loc = loc

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

    @property
    def _ncols(self):
        return self._ncol

    def __repr__(self):
        return f"<Legend>"
