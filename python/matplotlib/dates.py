"""Stub matplotlib.dates module."""

from matplotlib import ticker


class DateLocator:
    """Stub DateLocator."""
    pass


class AutoDateLocator(DateLocator):
    """Stub AutoDateLocator."""
    pass


class DateFormatter(ticker.Formatter):
    """Stub DateFormatter."""

    def __init__(self, fmt, tz=None, usetex=None):
        self.fmt = fmt

    def __call__(self, x, pos=0):
        return ""


class AutoDateFormatter(ticker.Formatter):
    """Stub AutoDateFormatter."""

    def __init__(self, locator=None, tz=None, defaultfmt='%Y'):
        self.defaultfmt = defaultfmt

    def __call__(self, x, pos=0):
        return ""


class ConciseDateFormatter(ticker.Formatter):
    def __init__(self, locator, tz=None, formats=None, offset_formats=None,
                 show_offset=True, usetex=None): pass

    def __call__(self, x, pos=0):
        return ""


class num2date:
    """Stub num2date."""

    def __new__(cls, x, tz=None):
        return x


def date2num(d):
    """Stub date2num."""
    import numpy as np
    return np.asarray(d, dtype=float)


class DateConverter:
    """Stub DateConverter."""
    pass


class RRuleLocator(DateLocator):
    def __init__(self, o, tz=None): pass


class WeekdayLocator(RRuleLocator):
    def __init__(self, byweekday=1, interval=1, tz=None): pass


class DayLocator(RRuleLocator):
    def __init__(self, bymonthday=None, interval=1, tz=None): pass


class HourLocator(RRuleLocator):
    def __init__(self, byhour=None, interval=1, tz=None): pass


class MinuteLocator(RRuleLocator):
    def __init__(self, byminute=None, interval=1, tz=None): pass


class SecondLocator(RRuleLocator):
    def __init__(self, bysecond=None, interval=1, tz=None): pass


class MonthLocator(RRuleLocator):
    def __init__(self, bymonth=None, bymonthday=1, interval=1, tz=None): pass


class YearLocator(DateLocator):
    def __init__(self, base=1, month=1, day=1, tz=None): pass


class MicrosecondLocator(DateLocator):
    def __init__(self, interval=1, tz=None): pass


MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)
