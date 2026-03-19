"""
matplotlib.ticker --- Tick locators and formatters.

Provides Locator and Formatter classes for axis tick generation and labelling.
"""

import math


# ===================================================================
# Formatter base class and subclasses
# ===================================================================

class Formatter:
    """Base class for tick formatters."""

    locs = []

    def __call__(self, x, pos=None):
        """Return the format for tick value *x* at position *pos*."""
        return self.format_data(x)

    def format_data(self, value):
        return str(value)

    def format_data_short(self, value):
        return self.format_data(value)

    def set_locs(self, locs):
        self.locs = list(locs)

    def get_offset(self):
        return ''

    def fix_minus(self, s):
        """Replace hyphens with unicode minus signs."""
        return s.replace('-', '\u2212')


class NullFormatter(Formatter):
    """A formatter that returns empty strings."""

    def __call__(self, x, pos=None):
        return ''


class FixedFormatter(Formatter):
    """A formatter that uses a fixed list of strings.

    Parameters
    ----------
    seq : list of str
        Strings for successive ticks.
    """

    def __init__(self, seq):
        self.seq = list(seq)
        self.offset_string = ''

    def __call__(self, x, pos=None):
        if pos is not None and pos < len(self.seq):
            return self.seq[pos]
        return ''

    def get_offset(self):
        return self.offset_string

    def set_offset_string(self, ofs):
        self.offset_string = ofs


class FuncFormatter(Formatter):
    """A formatter that uses a user-supplied function.

    Parameters
    ----------
    func : callable
        ``func(x, pos) -> str``
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, x, pos=None):
        return self.func(x, pos)


class FormatStrFormatter(Formatter):
    """A formatter that uses a format string.

    Parameters
    ----------
    fmt : str
        A %-style format string, e.g. ``'%1.2f'``.
    """

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        return self.fmt % x


class StrMethodFormatter(Formatter):
    """A formatter using str.format().

    Parameters
    ----------
    fmt : str
        A str.format()-style format string, e.g. ``'{x:.2f}'``.
    """

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        return self.fmt.format(x=x, pos=pos)


class ScalarFormatter(Formatter):
    """Default formatter for scalars: auto-pick fixed or scientific notation."""

    def __init__(self, useOffset=True, useMathText=False, useLocale=False):
        self.useOffset = useOffset
        self.useMathText = useMathText
        self.useLocale = useLocale
        self.offset = 0
        self._scientific = False
        self._powerlimits = (-3, 4)
        self.orderOfMagnitude = 0

    def __call__(self, x, pos=None):
        if x == int(x) and abs(x) < 1e15:
            return f'{int(x)}'
        return f'{x:g}'

    def set_scientific(self, b):
        self._scientific = b

    def set_powerlimits(self, lims):
        self._powerlimits = lims

    def get_offset(self):
        return ''


class LogFormatter(Formatter):
    """Formatter for log axes."""

    def __init__(self, base=10.0, labelOnlyBase=False):
        self.base = base
        self.labelOnlyBase = labelOnlyBase

    def __call__(self, x, pos=None):
        if x <= 0:
            return ''
        exponent = math.log(x) / math.log(self.base)
        is_integer_exp = abs(exponent - round(exponent)) < 1e-9
        if self.labelOnlyBase and not is_integer_exp:
            return ''
        if is_integer_exp:
            return f'$10^{{{int(round(exponent))}}}$'
        return f'{x:g}'


class PercentFormatter(Formatter):
    """Format values as percentages.

    Parameters
    ----------
    xmax : float
        Data value that corresponds to 100%.
    decimals : int or None
        Number of decimal places.
    symbol : str
        Percent symbol.
    """

    def __init__(self, xmax=100, decimals=None, symbol='%', is_latex=False):
        self.xmax = xmax
        self.decimals = decimals
        self.symbol = symbol
        self.is_latex = is_latex

    def __call__(self, x, pos=None):
        pct = x / self.xmax * 100.0
        if self.decimals is not None:
            s = f'{pct:.{self.decimals}f}'
        else:
            s = f'{pct:g}'
        return s + self.symbol


# ===================================================================
# Locator base class and subclasses
# ===================================================================

class Locator:
    """Base class for tick locators."""

    MAXTICKS = 1000

    def __call__(self):
        """Return a list of tick locations."""
        return self.tick_values(0, 1)

    def tick_values(self, vmin, vmax):
        """Return tick values between *vmin* and *vmax*."""
        raise NotImplementedError

    def set_params(self, **kwargs):
        """Set parameters. Subclasses override."""
        pass

    def view_limits(self, dmin, dmax):
        """Return adjusted view limits."""
        return (dmin, dmax)

    def raise_if_exceeds(self, locs):
        if len(locs) > self.MAXTICKS:
            raise RuntimeError(
                f"Locator attempting to generate {len(locs)} ticks "
                f"([{locs[0]}, ..., {locs[-1]}]), which exceeds "
                f"Locator.MAXTICKS ({self.MAXTICKS}).")
        return locs

    @property
    def numticks(self):
        return getattr(self, '_numticks', self.MAXTICKS)

    @numticks.setter
    def numticks(self, val):
        self._numticks = val


class NullLocator(Locator):
    """No ticks."""

    def __call__(self):
        return []

    def tick_values(self, vmin, vmax):
        return []


class FixedLocator(Locator):
    """Tick at fixed positions.

    Parameters
    ----------
    locs : list of float
        Tick positions.
    nbins : int or None
        Maximum number of ticks. If set, subsample.
    """

    def __init__(self, locs, nbins=None):
        self._locs = list(locs)
        self.nbins = nbins

    def __call__(self):
        return list(self._locs)

    def tick_values(self, vmin, vmax):
        ticks = [t for t in self._locs if vmin <= t <= vmax]
        if self.nbins is not None and len(ticks) > self.nbins + 1:
            step = max(1, len(ticks) // (self.nbins + 1))
            ticks = ticks[::step]
        return ticks

    def set_params(self, nbins=None):
        if nbins is not None:
            self.nbins = nbins


class IndexLocator(Locator):
    """Place ticks every *base* data units, offset by *offset*.

    Parameters
    ----------
    base : float
    offset : float
    """

    def __init__(self, base, offset):
        self._base = base
        self._offset = offset

    def tick_values(self, vmin, vmax):
        start = self._offset + self._base * math.ceil((vmin - self._offset) / self._base)
        ticks = []
        t = start
        while t <= vmax:
            ticks.append(t)
            t += self._base
        return ticks

    def set_params(self, base=None, offset=None):
        if base is not None:
            self._base = base
        if offset is not None:
            self._offset = offset


class LinearLocator(Locator):
    """Place *numticks* evenly spaced ticks.

    Parameters
    ----------
    numticks : int or None
    """

    def __init__(self, numticks=None):
        self._numticks = numticks if numticks is not None else 11

    def tick_values(self, vmin, vmax):
        n = max(2, self._numticks)
        step = (vmax - vmin) / (n - 1) if n > 1 else 1
        return [vmin + i * step for i in range(n)]

    def set_params(self, numticks=None):
        if numticks is not None:
            self._numticks = numticks


class MultipleLocator(Locator):
    """Place ticks at multiples of a given base.

    Parameters
    ----------
    base : float
    """

    def __init__(self, base=1.0):
        self._base = float(base)

    def tick_values(self, vmin, vmax):
        if self._base == 0:
            return []
        lo = math.floor(vmin / self._base)
        hi = math.ceil(vmax / self._base)
        ticks = [self._base * i for i in range(lo, hi + 1)]
        return [t for t in ticks if vmin - 1e-10 <= t <= vmax + 1e-10]

    def set_params(self, base=None):
        if base is not None:
            self._base = float(base)

    def view_limits(self, dmin, dmax):
        return (math.floor(dmin / self._base) * self._base,
                math.ceil(dmax / self._base) * self._base)


class MaxNLocator(Locator):
    """Find up to *nbins* + 1 nice tick locations.

    Uses a simplified version of the extended Wilkinson algorithm.

    Parameters
    ----------
    nbins : int or 'auto'
        Maximum number of intervals (ticks - 1). Default 10.
    steps : list of float, optional
        Sequence of nice step sizes (ascending, must include 1 and 10).
    integer : bool
        If True, ticks are forced to integer values.
    symmetric : bool
        If True, autoscale to be symmetric about zero.
    prune : str or None
        'lower', 'upper', 'both', or None.
    min_n_ticks : int
        Minimum number of ticks.
    """

    default_params = dict(
        nbins=10,
        steps=None,
        integer=False,
        symmetric=False,
        prune=None,
        min_n_ticks=2,
    )

    def __init__(self, nbins=None, **kwargs):
        params = dict(self.default_params)
        if nbins is not None:
            params['nbins'] = nbins
        params.update(kwargs)
        self._nbins = params['nbins']
        self._steps = params['steps'] or [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
        self._integer = params['integer']
        self._symmetric = params['symmetric']
        self._prune = params['prune']
        self._min_n_ticks = params['min_n_ticks']

    def set_params(self, **kwargs):
        if 'nbins' in kwargs:
            self._nbins = kwargs['nbins']
        if 'steps' in kwargs:
            self._steps = kwargs['steps']
        if 'integer' in kwargs:
            self._integer = kwargs['integer']
        if 'symmetric' in kwargs:
            self._symmetric = kwargs['symmetric']
        if 'prune' in kwargs:
            self._prune = kwargs['prune']
        if 'min_n_ticks' in kwargs:
            self._min_n_ticks = kwargs['min_n_ticks']

    def tick_values(self, vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            if vmin == 0:
                return [-0.5, 0.0, 0.5]
            vmin = vmin - abs(vmin) * 0.1
            vmax = vmax + abs(vmax) * 0.1

        nbins = self._nbins
        if nbins == 'auto':
            nbins = 9  # reasonable default

        raw_step = (vmax - vmin) / nbins
        # Find the magnitude
        if raw_step == 0:
            return [vmin]
        mag = 10 ** math.floor(math.log10(raw_step))

        # Find the best step
        best = None
        for step in self._steps:
            s = step * mag
            if self._integer and s < 1:
                continue
            if s >= raw_step * 0.9:
                best = s
                break
        if best is None:
            best = self._steps[-1] * mag

        if self._integer:
            best = max(1, math.ceil(best))

        # Generate ticks
        lo = math.floor(vmin / best) * best
        hi = math.ceil(vmax / best) * best
        # Avoid floating point issues
        n = int(round((hi - lo) / best)) + 1
        ticks = [lo + i * best for i in range(n)]

        # Round to avoid floating point artifacts
        ticks = [round(t, 12) for t in ticks]

        # Prune
        if self._prune == 'lower' and len(ticks) > 1:
            ticks = ticks[1:]
        elif self._prune == 'upper' and len(ticks) > 1:
            ticks = ticks[:-1]
        elif self._prune == 'both' and len(ticks) > 2:
            ticks = ticks[1:-1]

        return ticks

    def __call__(self):
        return self.tick_values(0, 1)

    def view_limits(self, dmin, dmax):
        if self._symmetric:
            maxabs = max(abs(dmin), abs(dmax))
            dmin, dmax = -maxabs, maxabs
        ticks = self.tick_values(dmin, dmax)
        if ticks:
            return (ticks[0], ticks[-1])
        return (dmin, dmax)


class AutoLocator(MaxNLocator):
    """Automatically find nice tick locations.

    Like MaxNLocator with nbins='auto'.
    """

    def __init__(self):
        super().__init__(nbins='auto')


class AutoMinorLocator(Locator):
    """Automatically find minor tick locations.

    Parameters
    ----------
    n : int or None
        Number of subdivisions per major tick interval.
    """

    def __init__(self, n=None):
        self.ndivs = n

    def tick_values(self, vmin, vmax):
        # Simplified: divide the range into many small ticks
        n = self.ndivs if self.ndivs is not None else 4
        step = (vmax - vmin) / (n * 10)
        if step == 0:
            return []
        ticks = []
        t = vmin
        while t <= vmax + step * 0.5:
            ticks.append(round(t, 12))
            t += step
        return ticks


class LogLocator(Locator):
    """Place ticks at powers of a given base.

    Parameters
    ----------
    base : float
        Log base. Default 10.
    subs : sequence of float
        Subdivision factors within each decade. Default (1.0,).
    numticks : int or 'auto'
    """

    def __init__(self, base=10.0, subs=(1.0,), numticks=None):
        self._base = float(base)
        self._subs = tuple(subs) if subs is not None else (1.0,)
        self._numticks = numticks

    def set_params(self, base=None, subs=None, numticks=None):
        if base is not None:
            self._base = float(base)
        if subs is not None:
            self._subs = tuple(subs)
        if numticks is not None:
            self._numticks = numticks

    def tick_values(self, vmin, vmax):
        if vmin <= 0:
            vmin = 1e-10
        if vmax <= vmin:
            vmax = vmin * 10

        log_base = math.log(self._base)
        lo = math.floor(math.log(vmin) / log_base)
        hi = math.ceil(math.log(vmax) / log_base)

        ticks = []
        for exp in range(lo - 1, hi + 2):
            for sub in self._subs:
                tick = sub * self._base ** exp
                if vmin * 0.99 <= tick <= vmax * 1.01:
                    ticks.append(tick)

        return sorted(set(ticks))

    def __call__(self):
        return self.tick_values(1, 1000)


class SymmetricalLogLocator(Locator):
    """Symmetric log locator for symlog scale."""

    def __init__(self, base=10.0, linthresh=1.0, subs=None):
        self._base = float(base)
        self._linthresh = float(linthresh)
        self._subs = tuple(subs) if subs is not None else (1.0,)

    def tick_values(self, vmin, vmax):
        ticks = [0.0]
        # Positive side
        if vmax > 0:
            log_loc = LogLocator(base=self._base, subs=self._subs)
            lo = max(self._linthresh, abs(vmin) if vmin > 0 else self._linthresh)
            pos_ticks = log_loc.tick_values(lo, max(lo, vmax))
            ticks.extend(pos_ticks)
        # Negative side
        if vmin < 0:
            log_loc = LogLocator(base=self._base, subs=self._subs)
            lo = max(self._linthresh, abs(vmax) if vmax < 0 else self._linthresh)
            neg_ticks = log_loc.tick_values(lo, max(lo, abs(vmin)))
            ticks.extend([-t for t in neg_ticks])
        return sorted(set(ticks))

    def set_params(self, base=None, linthresh=None, subs=None):
        if base is not None:
            self._base = float(base)
        if linthresh is not None:
            self._linthresh = float(linthresh)
        if subs is not None:
            self._subs = tuple(subs)
