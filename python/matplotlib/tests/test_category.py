"""Category tests imported from upstream matplotlib."""

import numpy as np
import pytest

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat


class TestUnitData:
    @pytest.mark.parametrize(
        "data, locs",
        [
            (["hello world"], [0]),
            (["Здравствуйте мир"], [0]),
            (['A', "np.nan", 'B', "3.14", "мир"], [0, 1, 2, 3, 4]),
        ],
    )
    def test_unit(self, data, locs):
        unit = cat.UnitData(data)
        assert list(unit._mapping.keys()) == data
        assert list(unit._mapping.values()) == locs

    def test_update(self):
        unit = cat.UnitData(['a', 'd'])
        unit.update(['b', 'd', 'e'])
        assert list(unit._mapping.keys()) == ['a', 'd', 'b', 'e']
        assert list(unit._mapping.values()) == [0, 1, 2, 3]

    @pytest.mark.parametrize(
        "bad_data",
        [3.14, np.nan, [3.14, 12], ["A", 2]],
    )
    def test_non_string_fails(self, bad_data):
        with pytest.raises(TypeError):
            cat.UnitData(bad_data)

    @pytest.mark.parametrize(
        "bad_data",
        [3.14, np.nan, [3.14, 12], ["A", 2]],
    )
    def test_non_string_update_fails(self, bad_data):
        unit = cat.UnitData()
        with pytest.raises(TypeError):
            unit.update(bad_data)


class FakeAxis:
    def __init__(self, units):
        self.units = units


class TestStrCategoryConverter:
    @pytest.fixture(autouse=True)
    def mock_axis(self):
        self.cc = cat.StrCategoryConverter()
        self.unit = cat.UnitData()
        self.ax = FakeAxis(self.unit)

    @pytest.mark.parametrize(
        "vals",
        [
            ["Здравствуйте мир"],
            ["hello world"],
            ['a', 'b', 'c'],
            ["1", "2"],
        ],
    )
    def test_convert(self, vals):
        np.testing.assert_allclose(
            self.cc.convert(vals, self.ax.units, self.ax),
            range(len(vals)),
        )

    @pytest.mark.parametrize("value", ["hi", "мир"])
    def test_convert_one_string(self, value):
        assert self.cc.convert(value, self.unit, self.ax) == 0

    @pytest.mark.parametrize(
        "bad_vals",
        [[3.14, 'A', np.inf], ['42', 42]],
    )
    def test_convert_fail(self, bad_vals):
        with pytest.raises(TypeError):
            self.cc.convert(bad_vals, self.unit, self.ax)

    def test_axisinfo(self):
        axis = self.cc.axisinfo(self.unit, self.ax)
        assert isinstance(axis.majloc, cat.StrCategoryLocator)
        assert isinstance(axis.majfmt, cat.StrCategoryFormatter)

    def test_default_units(self):
        assert isinstance(self.cc.default_units(["a"], self.ax), cat.UnitData)


PLOT_LIST = [Axes.scatter, Axes.plot, Axes.bar]


def axis_test(axis, labels):
    ticks = list(range(len(labels)))
    np.testing.assert_array_equal(axis.get_majorticklocs(), ticks)
    graph_labels = [axis.major.formatter(i, i) for i in ticks]
    assert graph_labels == [cat.StrCategoryFormatter._text(l) for l in labels]
    assert list(axis.units._mapping.keys()) == [l for l in labels]
    assert list(axis.units._mapping.values()) == ticks


class TestPlotBytes:
    @pytest.mark.parametrize(
        "plotter",
        PLOT_LIST,
    )
    @pytest.mark.parametrize(
        "bdata",
        [
            ['a', 'b', 'c'],
            [b'a', b'b', b'c'],
            np.array([b'a', b'b', b'c']),
        ],
    )
    def test_plot_bytes(self, plotter, bdata):
        ax = plt.figure().subplots()
        counts = np.array([4, 6, 5])
        plotter(ax, bdata, counts)
        axis_test(ax.xaxis, bdata)


class TestPlotNumlike:
    @pytest.mark.parametrize("plotter", PLOT_LIST)
    @pytest.mark.parametrize(
        "ndata",
        [
            ['1', '11', '3'],
            np.array(['1', '11', '3']),
            [b'1', b'11', b'3'],
            np.array([b'1', b'11', b'3']),
        ],
    )
    def test_plot_numlike(self, plotter, ndata):
        ax = plt.figure().subplots()
        counts = np.array([4, 6, 5])
        plotter(ax, ndata, counts)
        axis_test(ax.xaxis, ndata)
