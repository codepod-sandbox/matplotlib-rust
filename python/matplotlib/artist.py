"""matplotlib.artist --- base class for all visual objects."""
import functools


def allow_rasterization(draw):
    """Decorator to allow rasterization for an artist's draw method."""
    @functools.wraps(draw)
    def draw_wrapper(artist, renderer):
        return draw(artist, renderer)
    draw_wrapper._supports_rasterization = True
    return draw_wrapper


class Artist:
    """Base class for objects that render into a Figure."""

    zorder = 0

    def __init__(self):
        self._visible = True
        self._alpha = None
        self._label = ''
        self._zorder = self.__class__.zorder
        self._clip_on = True
        self.figure = None
        self._axes = None
        self._stale = True

    @property
    def axes(self):
        """The `~.axes.Axes` instance the artist resides in, or *None*."""
        return self._axes

    @axes.setter
    def axes(self, new_axes):
        self._axes = new_axes

    def get_visible(self):
        return self._visible

    def set_visible(self, b):
        self._visible = b

    def get_clip_on(self):
        return self._clip_on

    def set_clip_on(self, b):
        self._clip_on = bool(b)

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, alpha):
        self._alpha = alpha

    def get_label(self):
        return self._label

    def set_label(self, s):
        self._label = str(s) if s is not None else '_nolegend_'

    def get_zorder(self):
        return self._zorder

    def set_zorder(self, level):
        self._zorder = level

    def remove(self):
        """Remove this artist from its figure/axes."""
        if hasattr(self, '_remove_method') and self._remove_method is not None:
            self._remove_method(self)
        elif self.axes is not None and hasattr(self.axes, '_remove_artist'):
            self.axes._remove_artist(self)

    def set(self, **kwargs):
        """Batch property setter."""
        for k, v in kwargs.items():
            setter = getattr(self, f'set_{k}', None)
            if setter:
                setter(v)

    def update(self, props):
        """Update properties from a dictionary.

        Equivalent to calling ``set(**props)``.
        """
        if props is None:
            return
        self.set(**props)

    def properties(self):
        """Return a dict of all gettable properties and their values."""
        result = {}
        for attr in dir(self):
            if attr.startswith('get_') and callable(getattr(self, attr)):
                key = attr[4:]
                try:
                    result[key] = getattr(self, attr)()
                except Exception:
                    pass
        return result

    def get_clip_on(self):
        """Return whether clipping is enabled."""
        return getattr(self, '_clip_on', True)

    def set_clip_on(self, b):
        """Set whether clipping is enabled."""
        self._clip_on = b

    def get_clip_box(self):
        """Return the clip box."""
        return getattr(self, '_clip_box', None)

    def set_clip_box(self, clipbox):
        """Set the clip box."""
        self._clip_box = clipbox

    def get_clip_path(self):
        """Return the clip path."""
        return getattr(self, '_clip_path', None)

    def set_clip_path(self, path, transform=None):
        """Set the clip path."""
        self._clip_path = path

    def get_transform(self):
        """Return the artist transform."""
        return getattr(self, '_transform', None)

    def set_transform(self, t):
        """Set the artist transform."""
        self._transform = t

    def get_animated(self):
        """Return whether the artist is animated."""
        return getattr(self, '_animated', False)

    def set_animated(self, b):
        """Set whether the artist is animated."""
        self._animated = b

    def get_rasterized(self):
        """Return whether the artist is rasterized."""
        return getattr(self, '_rasterized', None)

    def set_rasterized(self, b):
        """Set whether the artist is rasterized."""
        self._rasterized = b

    def get_sketch_params(self):
        """Return sketch params (scale, length, randomness)."""
        return getattr(self, '_sketch_params', None)

    def set_sketch_params(self, scale=None, length=None, randomness=None):
        """Set sketch params."""
        if scale is None:
            self._sketch_params = None
        else:
            self._sketch_params = (scale, length or 128, randomness or 16)

    def get_snap(self):
        """Return the snap setting."""
        return getattr(self, '_snap', None)

    def set_snap(self, snap):
        """Set the snap setting."""
        self._snap = snap

    def get_path_effects(self):
        """Return list of path effects."""
        return getattr(self, '_path_effects', [])

    def set_path_effects(self, path_effects):
        """Set path effects."""
        self._path_effects = list(path_effects)

    def get_url(self):
        """Return the URL for this artist."""
        return getattr(self, '_url', None)

    def set_url(self, url):
        """Set the URL for this artist."""
        self._url = url

    def get_gid(self):
        """Return the group id."""
        return getattr(self, '_gid', None)

    def set_gid(self, gid):
        """Set the group id."""
        self._gid = gid

    def get_in_layout(self):
        """Return whether artist is included in layout calculations."""
        return getattr(self, '_in_layout', True)

    def set_in_layout(self, in_layout):
        """Set whether artist is included in layout calculations."""
        self._in_layout = in_layout

    def get_picker(self):
        """Return the picker."""
        return getattr(self, '_picker', None)

    def set_picker(self, picker):
        """Set the picker."""
        self._picker = picker

    def pickable(self):
        """Return whether the artist is pickable."""
        return self.get_picker() is not None

    def get_agg_filter(self):
        """Return the agg_filter."""
        return getattr(self, '_agg_filter', None)

    def set_agg_filter(self, filter_func):
        """Set the agg_filter."""
        self._agg_filter = filter_func

    def get_contains(self):
        """Return the contains test function."""
        return getattr(self, '_contains', None)

    def set_contains(self, picker):
        """Set the contains test function."""
        self._contains = picker

    def convert_xunits(self, x):
        """Convert *x* using the unit type of the xaxis."""
        ax = getattr(self, 'axes', None)
        if ax is None:
            return x
        xaxis = getattr(ax, 'xaxis', None)
        if xaxis is None:
            return x
        return xaxis.convert_units(x)

    def convert_yunits(self, y):
        """Convert *y* using the unit type of the yaxis."""
        ax = getattr(self, 'axes', None)
        if ax is None:
            return y
        yaxis = getattr(ax, 'yaxis', None)
        if yaxis is None:
            return y
        return yaxis.convert_units(y)

    class _StickyEdges:
        """Simple container for sticky_edges x/y lists."""
        def __init__(self):
            self.x = []
            self.y = []

    @property
    def sticky_edges(self):
        """x and y sticky edge lists for autoscaling."""
        if not hasattr(self, '_sticky_edges'):
            self._sticky_edges = Artist._StickyEdges()
        return self._sticky_edges

    def _internal_update(self, kwargs):
        """Update artist properties from kwargs dict."""
        for k, v in kwargs.items():
            setter = getattr(self, f'set_{k}', None)
            if setter is not None:
                setter(v)
            else:
                raise AttributeError(
                    f"{type(self).__name__}.set() got an unexpected keyword "
                    f"argument {k!r}")
        return self

    def get_mouseover(self):
        """Return whether the artist responds to mouse-over."""
        return getattr(self, '_mouseover', False)

    def set_mouseover(self, val):
        """Set whether the artist responds to mouse-over."""
        self._mouseover = bool(val)

    def get_clip_path(self):
        """Return the clip path."""
        return getattr(self, '_clip_path', None)

    def set_clip_path(self, path, transform=None):
        """Set the clip path."""
        self._clip_path = path

    def get_path_effects(self):
        """Return path effects."""
        return getattr(self, '_path_effects', [])

    def set_path_effects(self, effects):
        """Set path effects."""
        self._path_effects = effects

    def get_sketch_params(self):
        """Return sketch params."""
        return getattr(self, '_sketch', None)

    def set_sketch_params(self, scale=0.1, length=0.1, randomness=0.1):
        """Set sketch params."""
        self._sketch = (scale, length, randomness)

    def get_snap(self):
        """Return whether snapping is enabled."""
        return getattr(self, '_snap', None)

    def set_snap(self, snap):
        """Set whether snapping is enabled."""
        self._snap = snap

    def get_agg_filter(self):
        """Return the agg filter."""
        return getattr(self, '_agg_filter', None)

    def set_agg_filter(self, filter_func):
        """Set the agg filter."""
        self._agg_filter = filter_func

    def get_rasterized(self):
        """Return whether the artist is rasterized."""
        return getattr(self, '_rasterized', False)

    def set_rasterized(self, val):
        """Set whether the artist is rasterized."""
        self._rasterized = bool(val)

    def get_in_layout(self):
        """Return whether the artist is included in layout calculations."""
        return getattr(self, '_in_layout', True)

    def set_in_layout(self, val):
        """Set whether the artist is included in layout calculations."""
        self._in_layout = bool(val)

    def get_transform(self):
        """Return the transform."""
        return getattr(self, '_transform', None)

    def get_figure(self, root=True):
        """Return the figure.

        Parameters
        ----------
        root : bool, default True
            If True, return the root-level Figure. If False, return the
            immediate parent Figure (may be a subfigure).
        """
        return self.figure

    def set_figure(self, fig):
        """Set the figure."""
        self.figure = fig

    def is_transform_set(self):
        """Return whether a transform has been set."""
        return hasattr(self, '_transform') and self._transform is not None

    @property
    def stale(self):
        """Whether the artist needs to be redrawn."""
        return self._stale

    @stale.setter
    def stale(self, val):
        self._stale = val

    def pchanged(self):
        """Fire event when property changes."""
        self._stale = True

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
        # Check children
        if hasattr(self, 'get_children'):
            for child in self.get_children():
                result.extend(child.findobj(match, include_self=True))
        return result


def kwdoc(artist):
    """Stub kwdoc for _docstring.interpd compatibility."""
    return ''


class ArtistInspector:
    """Stub ArtistInspector."""

    def __init__(self, o):
        pass

    def pprint_setters(self, leadingspace=2):
        return []

    def pprint_setters_rest(self, leadingspace=4):
        return []
