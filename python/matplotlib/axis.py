# Copyright (c) 2024 CodePod Contributors
# BSD 3-Clause License (see LICENSE at repository root)
"""matplotlib.axis — XAxis and YAxis wrappers."""

GRIDLINE_INTERPOLATION_STEPS = 180

from matplotlib.ticker import (
    AutoLocator, ScalarFormatter,
    FixedLocator, FixedFormatter,
    NullLocator, NullFormatter,
)
from matplotlib.scale import LinearScale
from matplotlib.artist import Artist
from matplotlib.cbook import CallbackRegistry


class _TickerPair:
    """Holds a (locator, formatter) pair for major or minor ticks."""

    def __init__(self, locator, formatter):
        self.locator = locator
        self.formatter = formatter


class Axis(Artist):
    """Base class for a single axis (x or y)."""

    def __init__(self, axes=None, *, clear=True):
        super().__init__()
        self._major = _TickerPair(AutoLocator(), ScalarFormatter())
        self._minor = _TickerPair(NullLocator(), NullFormatter())
        self._ticks = None   # None means "use locator"; list means "fixed"
        self.axes = axes
        self._label = None
        self._scale = LinearScale(self)
        self.callbacks = CallbackRegistry(signals=["units"])
        self._converter = None
        self.units = None
        # label attribute as a Text-like stub
        from matplotlib.text import Text
        self.label = Text(text='')
        self.offsetText = Text(text='')

    def clear(self):
        """Clear/reset this axis."""
        try:
            self._scale = LinearScale(self)
        except Exception:
            pass
        self._major = _TickerPair(AutoLocator(), ScalarFormatter())
        self._minor = _TickerPair(NullLocator(), NullFormatter())
        self._ticks = None

    def grid(self, visible=None, which='major', **kwargs):
        """Set grid visibility for this axis."""
        pass

    def set_label_position(self, position):
        """Set label position ('top'/'bottom' or 'left'/'right')."""
        self._label_position = position

    def get_label_position(self):
        return getattr(self, '_label_position', 'bottom')

    def set_tick_params(self, which='major', **kwargs):
        """Set tick parameters."""
        pass

    def get_major_ticks(self, numticks=None):
        """Return major tick objects."""
        return []

    def get_minor_ticks(self, numticks=None):
        """Return minor tick objects."""
        return []

    def set_label_text(self, label, fontdict=None, **kwargs):
        """Set axis label text."""
        self._label_text = label
        self.label.set_text(label)
        return self.label

    def get_label_text(self):
        return getattr(self, '_label_text', '')

    def set_units(self, u):
        """Set axis units."""
        self._units = u

    def get_units(self):
        return getattr(self, '_units', None)

    def convert_units(self, x):
        """Convert units to float."""
        return x

    def have_units(self):
        return False

    def update_units(self, data):
        """Update units for the axis based on data."""
        pass

    def _update_ticks(self):
        """Update tick locations and labels."""
        pass

    def get_offset_text(self):
        """Return the offset text artist."""
        from matplotlib.text import Text
        return getattr(self, '_offset_text', Text(text=''))

    def set_offset_text(self, text):
        """Set offset text."""
        self._offset_text_str = text

    def minorticks_on(self):
        """Enable minor ticks on this Axis."""
        from matplotlib.ticker import AutoMinorLocator
        self._minor.locator = AutoMinorLocator()

    def minorticks_off(self):
        """Disable minor ticks on this Axis."""
        self._minor.locator = NullLocator()

    def set_inverted(self, inverted):
        """Set whether the axis is inverted."""
        self._inverted = bool(inverted)

    def cla(self):
        """Clear the axis (alias for clear)."""
        self.clear()

    def get_label(self):
        """Return axis label (Text or str)."""
        return getattr(self, '_label_text', '')

    def set_minor_formatter(self, formatter):
        self._minor.formatter = formatter

    def register_axis(self, axis):
        """Register this spine with an axis."""
        self._axis = axis

    def get_transform(self):
        """Return the scale transform for this Axis."""
        return self._scale.get_transform()

    # --- scale ---
    def get_scale(self):
        """Return the scale name of this Axis (e.g. 'linear', 'log')."""
        if hasattr(self._scale, 'name'):
            return self._scale.name
        return 'linear'

    def set_scale(self, scale):
        """Set scale object and update default locator/formatter."""
        from matplotlib.scale import LogScale, SymmetricalLogScale
        from matplotlib.ticker import (LogLocator, LogFormatter,
                                        SymmetricalLogLocator)
        self._scale = scale
        if isinstance(scale, LogScale):
            self._major.locator = LogLocator(base=scale.base)
            self._major.formatter = LogFormatter(base=scale.base)
        elif isinstance(scale, SymmetricalLogScale):
            self._major.locator = SymmetricalLogLocator(
                base=scale.base, linthresh=scale.linthresh)
            self._major.formatter = ScalarFormatter()
        else:
            # LinearScale or FuncScale
            self._major.locator = AutoLocator()
            self._major.formatter = ScalarFormatter()

    # --- major ---
    def get_major_locator(self):
        return self._major.locator

    def set_major_locator(self, locator):
        self._major.locator = locator
        self._major.formatter = ScalarFormatter()
        self._ticks = None  # clear fixed ticks

    def get_major_formatter(self):
        return self._major.formatter

    def set_major_formatter(self, formatter):
        self._major.formatter = formatter

    # --- minor ---
    def get_minor_locator(self):
        return self._minor.locator

    def set_minor_locator(self, locator):
        self._minor.locator = locator

    def get_minor_formatter(self):
        return self._minor.formatter

    def set_minor_formatter(self, formatter):
        self._minor.formatter = formatter

    # --- explicit ticks ---
    def set_ticks(self, ticks, labels=None, **kwargs):
        """Set the ticks for this Axis using set_ticks."""
        ticks = list(ticks)
        self._ticks = ticks
        self._major.locator = FixedLocator(ticks)
        if labels is not None:
            self._major.formatter = FixedFormatter(list(labels))
        else:
            self._major.formatter = ScalarFormatter()

    def get_ticks(self):
        """Return the fixed tick list, or [] if using auto locator."""
        return list(self._ticks) if self._ticks is not None else []

    def tick_values(self, vmin, vmax):
        """Ask the locator for tick positions in [vmin, vmax]."""
        return self._major.locator.tick_values(vmin, vmax)

    def format_ticks(self, values):
        """Format a sequence of tick values using the major formatter."""
        fmt = self._major.formatter
        vals_list = list(values)
        # ScalarFormatter (and some others) need set_locs + a view interval
        # to choose the right number of significant digits.
        if hasattr(fmt, 'set_locs'):
            if fmt.axis is None:
                fmt.create_dummy_axis()
            if vals_list:
                vmin, vmax = min(vals_list), max(vals_list)
                fmt.axis.set_view_interval(vmin, vmax)
                fmt.axis.set_data_interval(vmin, vmax)
            fmt.set_locs(vals_list)  # always call, even with empty list
        return [fmt(v, i) for i, v in enumerate(vals_list)]

    # --- methods required by _AxesBase._axis_method_wrapper ---

    def get_gridlines(self):
        """Return the gridlines of this Axis as a list of Line2D."""
        return []

    def get_ticklines(self, minor=False):
        """Return tick line artists for this Axis."""
        return []

    def _get_autoscale_on(self):
        """Get whether autoscaling is enabled for this Axis."""
        return getattr(self, '_autoscale_on', True)

    def _set_autoscale_on(self, b):
        """Set whether autoscaling is enabled for this Axis."""
        self._autoscale_on = b

    def get_inverted(self):
        """Return whether this Axis is inverted."""
        return getattr(self, '_inverted', False)

    def _set_axes_scale(self, value, **kwargs):
        """Set the scale of this Axis by name."""
        from matplotlib.scale import (LinearScale, LogScale,
                                       SymmetricalLogScale)
        scale_map = {
            'linear': LinearScale,
            'log': LogScale,
            'symlog': SymmetricalLogScale,
        }
        cls = scale_map.get(value)
        if cls is not None:
            self.set_scale(cls(self))

    def get_ticklocs(self, minor=False):
        """Return tick locations for this Axis."""
        return self.get_ticks()

    def get_majorticklabels(self):
        """Return major tick label artists for this Axis."""
        return []

    def get_majorticklocs(self):
        """Return locations of major ticks for this Axis."""
        return self.get_ticks()

    def get_minorticklabels(self):
        """Return minor tick label artists for this Axis."""
        return []

    def get_minorticklocs(self):
        """Return locations of minor ticks for this Axis."""
        return []

    def get_ticklabels(self, minor=False, which=None):
        """Return tick labels for this Axis."""
        return []

    def set_ticklabels(self, labels, *, minor=False, **kwargs):
        """Set the text values of the tick labels for this Axis.

        Use Axis.set_ticks to set tick positions and labels simultaneously.
        """
        pass

    def axis_date(self, tz=None):
        """Set up this Axis to use dates."""
        pass

    def _set_lim(self, v0, v1, *, auto=False, emit=True):
        """Set axis limits (stub for _AxesBase compatibility)."""
        self._view_interval = (v0, v1)

    def set_view_interval(self, vmin, vmax, ignore=False):
        """Set view interval."""
        self._view_interval = (vmin, vmax)

    def get_view_interval(self):
        """Get view interval."""
        return getattr(self, '_view_interval', (0, 1))

    def set_data_interval(self, vmin, vmax, ignore=False):
        """Set data interval."""
        self._data_interval = (vmin, vmax)

    def get_data_interval(self):
        """Get data interval."""
        return getattr(self, '_data_interval', (0, 1))


class XAxis(Axis):
    """X-axis."""

    def tick_top(self):
        """Move ticks and ticklabels to the top."""
        self.set_label_position('top')

    def tick_bottom(self):
        """Move ticks and ticklabels to the bottom."""
        self.set_label_position('bottom')

    def set_ticks_position(self, position):
        """Set the ticks position (top/bottom/both/default/none)."""
        self._ticks_position = position


class YAxis(Axis):
    """Y-axis."""

    def tick_left(self):
        """Move ticks and ticklabels to the left."""
        self.set_label_position('left')

    def tick_right(self):
        """Move ticks and ticklabels to the right."""
        self.set_label_position('right')

    def set_ticks_position(self, position):
        """Set the ticks position (left/right/both/default/none)."""
        self._ticks_position = position

    def set_offset_position(self, position):
        """Set the position of the offset text for this YAxis ('left' or 'right')."""
        self._offset_position = position

    def get_offset_position(self):
        """Get the position of the offset text ('left' or 'right')."""
        return getattr(self, '_offset_position', 'left')

