"""Stub matplotlib.dates module."""


class DateLocator:
    """Stub DateLocator."""
    pass


class AutoDateLocator(DateLocator):
    """Stub AutoDateLocator."""
    pass


class DateFormatter:
    """Stub DateFormatter."""

    def __init__(self, fmt, tz=None):
        self.fmt = fmt


class AutoDateFormatter:
    """Stub AutoDateFormatter."""

    def __init__(self, locator=None, tz=None, defaultfmt='%Y'):
        self.defaultfmt = defaultfmt


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
