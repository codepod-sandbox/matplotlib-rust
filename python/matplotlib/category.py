"""
Stub matplotlib.category module.

Provides minimal category unit converter registration so that OG axes/_axes.py
can import this module as a side effect without error.
"""

import logging
import numpy as np

from matplotlib import _api, units

_log = logging.getLogger(__name__)


class StrCategoryLocator(object):
    """Stub tick locator for categorical data."""

    def __init__(self, units_mapping):
        self._units = units_mapping

    def tick_values(self, vmin, vmax):
        return list(self._units.values())

    def __call__(self):
        return self.tick_values(0, len(self._units))


class StrCategoryFormatter(object):
    """Stub tick formatter for categorical data."""

    def __init__(self, units_mapping):
        self._units = {v: k for k, v in units_mapping.items()}

    def __call__(self, x, pos=None):
        return self._units.get(int(x), '')


class UnitData:
    """Stub UnitData for categorical axes."""

    def __init__(self, data=None):
        self._mapping = {}
        if data is not None:
            self.update(data)

    def update(self, data):
        for val in data:
            if val not in self._mapping:
                self._mapping[val] = len(self._mapping)


class StrCategoryConverter(units.ConversionInterface):
    """Stub converter for string category data."""

    @staticmethod
    def convert(value, unit, axis):
        if isinstance(value, str):
            if unit is None:
                unit = UnitData([value])
            unit.update([value])
            return unit._mapping.get(value, 0)
        try:
            values = np.asarray(value)
        except Exception:
            return 0
        if values.dtype.kind in ('U', 'S', 'O'):
            if unit is None:
                unit = UnitData(values)
            unit.update(values)
            return np.array([unit._mapping.get(v, 0) for v in values],
                            dtype=float)
        return values

    @staticmethod
    def axisinfo(unit, axis):
        if unit is None:
            return units.AxisInfo()
        loc = StrCategoryLocator(unit._mapping)
        fmt = StrCategoryFormatter(unit._mapping)
        return units.AxisInfo(majloc=loc, majfmt=fmt)

    @staticmethod
    def default_units(data, axis):
        axis.unit_data = UnitData(np.atleast_1d(data))
        return axis.unit_data
