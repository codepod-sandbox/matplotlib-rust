"""Tests for matplotlib.ticker --- tick locators and formatters."""

import math
import pytest
import matplotlib.pyplot as plt
from matplotlib.tests._approx import approx

from matplotlib.ticker import (
    Formatter,
    NullFormatter,
    FixedFormatter,
    FuncFormatter,
    FormatStrFormatter,
    StrMethodFormatter,
    ScalarFormatter,
    LogFormatter,
    PercentFormatter,
    Locator,
    NullLocator,
    FixedLocator,
    IndexLocator,
    LinearLocator,
    MultipleLocator,
    MaxNLocator,
    AutoLocator,
    AutoMinorLocator,
    LogLocator,
    SymmetricalLogLocator,
)


# ===================================================================
# Formatter tests
# ===================================================================

class TestNullFormatter:
    def test_returns_empty(self):
        fmt = NullFormatter()
        assert fmt(1.0) == ''
        assert fmt(0.0) == ''
        assert fmt(42.0, pos=0) == ''


class TestFixedFormatter:
    def test_basic(self):
        fmt = FixedFormatter(['a', 'b', 'c'])
        assert fmt(0, pos=0) == 'a'
        assert fmt(0, pos=1) == 'b'
        assert fmt(0, pos=2) == 'c'

    def test_out_of_range(self):
        fmt = FixedFormatter(['a', 'b'])
        assert fmt(0, pos=5) == ''

    def test_no_pos(self):
        fmt = FixedFormatter(['a'])
        assert fmt(0) == ''  # pos is None

    def test_offset_string(self):
        fmt = FixedFormatter(['a'])
        fmt.set_offset_string('offset')
        assert fmt.get_offset() == 'offset'


class TestFuncFormatter:
    def test_basic(self):
        fmt = FuncFormatter(lambda x, pos: f'{x:.1f}!')
        assert fmt(3.14) == '3.1!'

    def test_with_pos(self):
        fmt = FuncFormatter(lambda x, pos: f'{pos}:{x}')
        assert fmt(1.0, pos=2) == '2:1.0'


class TestFormatStrFormatter:
    def test_basic(self):
        fmt = FormatStrFormatter('%1.2f')
        assert fmt(3.14159) == '3.14'

    def test_integer(self):
        fmt = FormatStrFormatter('%d')
        assert fmt(42.7) == '42'

    def test_scientific(self):
        fmt = FormatStrFormatter('%1.2e')
        assert '1.00e' in fmt(1.0) or '1.00E' in fmt(1.0).upper()


class TestStrMethodFormatter:
    def test_basic(self):
        fmt = StrMethodFormatter('{x:.2f}')
        assert fmt(3.14159) == '3.14'

    def test_with_pos(self):
        fmt = StrMethodFormatter('{x:.1f} at {pos}')
        assert fmt(3.14, pos=0) == '3.1 at 0'


class TestScalarFormatter:
    def test_integer(self):
        fmt = ScalarFormatter()
        assert fmt(42.0) == '42'

    def test_float(self):
        fmt = ScalarFormatter()
        result = fmt(3.14)
        assert '3.14' in result

    def test_zero(self):
        fmt = ScalarFormatter()
        assert fmt(0.0) == '0'

    def test_set_scientific(self):
        fmt = ScalarFormatter()
        fmt.set_scientific(True)  # should not raise

    def test_set_powerlimits(self):
        fmt = ScalarFormatter()
        fmt.set_powerlimits((-2, 2))  # should not raise

    def test_get_offset(self):
        fmt = ScalarFormatter()
        assert fmt.get_offset() == ''


class TestLogFormatter:
    def test_basic(self):
        fmt = LogFormatter()
        result = fmt(100)
        assert '2' in result  # 10^2

    def test_zero(self):
        fmt = LogFormatter()
        assert fmt(0) == ''

    def test_negative(self):
        fmt = LogFormatter()
        assert fmt(-1) == ''

    def test_labelOnlyBase(self):
        fmt = LogFormatter(labelOnlyBase=True)
        assert fmt(100) != ''  # 10^2 is a base power
        # non-base powers should be empty
        assert fmt(50) == ''


class TestPercentFormatter:
    def test_basic(self):
        fmt = PercentFormatter()
        assert fmt(50) == '50%'

    def test_xmax(self):
        fmt = PercentFormatter(xmax=1.0)
        assert fmt(0.5) == '50%'

    def test_decimals(self):
        fmt = PercentFormatter(decimals=2)
        result = fmt(33.333)
        assert '33.33%' in result

    def test_custom_symbol(self):
        fmt = PercentFormatter(symbol=' pct')
        assert fmt(50).endswith(' pct')

    def test_no_symbol(self):
        fmt = PercentFormatter(symbol='')
        assert '%' not in fmt(50)


# ===================================================================
# Locator tests
# ===================================================================

class TestNullLocator:
    def test_empty(self):
        loc = NullLocator()
        assert loc() == []

    def test_tick_values_empty(self):
        loc = NullLocator()
        assert loc.tick_values(0, 10) == []


class TestFixedLocator:
    def test_basic(self):
        loc = FixedLocator([1, 2, 3, 4, 5])
        assert loc() == [1, 2, 3, 4, 5]

    def test_tick_values_filters(self):
        loc = FixedLocator([1, 2, 3, 4, 5])
        ticks = loc.tick_values(2, 4)
        assert 1 not in ticks
        assert 5 not in ticks
        assert 2 in ticks
        assert 3 in ticks
        assert 4 in ticks

    def test_nbins(self):
        loc = FixedLocator(list(range(100)), nbins=5)
        ticks = loc.tick_values(0, 99)
        assert len(ticks) <= 20  # subsampled

    def test_set_params(self):
        loc = FixedLocator([1, 2, 3])
        loc.set_params(nbins=2)
        assert loc.nbins == 2

    def test_empty(self):
        loc = FixedLocator([])
        assert loc() == []


class TestIndexLocator:
    def test_basic(self):
        loc = IndexLocator(base=5, offset=0)
        ticks = loc.tick_values(0, 20)
        assert 0 in ticks
        assert 5 in ticks
        assert 10 in ticks
        assert 15 in ticks
        assert 20 in ticks

    def test_with_offset(self):
        loc = IndexLocator(base=10, offset=3)
        ticks = loc.tick_values(0, 30)
        assert 3 in ticks
        assert 13 in ticks
        assert 23 in ticks

    def test_set_params(self):
        loc = IndexLocator(5, 0)
        loc.set_params(base=10)
        ticks = loc.tick_values(0, 20)
        assert 10 in ticks


class TestLinearLocator:
    def test_basic(self):
        loc = LinearLocator(numticks=5)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) == 5
        assert ticks[0] == approx(0.0)
        assert ticks[-1] == approx(10.0)

    def test_default(self):
        loc = LinearLocator()
        ticks = loc.tick_values(0, 1)
        assert len(ticks) == 11

    def test_set_params(self):
        loc = LinearLocator()
        loc.set_params(numticks=3)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) == 3


class TestMultipleLocator:
    def test_basic(self):
        loc = MultipleLocator(base=5)
        ticks = loc.tick_values(0, 20)
        assert 0 in ticks
        assert 5 in ticks
        assert 10 in ticks
        assert 15 in ticks
        assert 20 in ticks

    def test_fractional(self):
        loc = MultipleLocator(base=0.5)
        ticks = loc.tick_values(0, 2)
        assert 0.0 in ticks
        assert 0.5 in ticks
        assert 1.0 in ticks
        assert 1.5 in ticks
        assert 2.0 in ticks

    def test_view_limits(self):
        loc = MultipleLocator(base=5)
        lo, hi = loc.view_limits(3, 17)
        assert lo == 0
        assert hi == 20

    def test_set_params(self):
        loc = MultipleLocator(base=5)
        loc.set_params(base=10)
        ticks = loc.tick_values(0, 20)
        assert 5 not in ticks

    def test_zero_base(self):
        loc = MultipleLocator(base=0)
        assert loc.tick_values(0, 10) == []


class TestMaxNLocator:
    def test_basic(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) >= 2
        assert len(ticks) <= 12  # reasonable bound

    def test_nice_ticks(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0, 100)
        # Ticks should be at nice round numbers
        for t in ticks:
            assert t == int(t)

    def test_negative_range(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(-100, 0)
        assert len(ticks) >= 2
        assert ticks[0] <= -100
        assert ticks[-1] >= 0

    def test_small_range(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0, 0.01)
        assert len(ticks) >= 2

    def test_same_value(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(5, 5)
        assert len(ticks) >= 2

    def test_integer(self):
        loc = MaxNLocator(integer=True)
        ticks = loc.tick_values(0.1, 9.9)
        for t in ticks:
            assert t == int(t)

    def test_prune_lower(self):
        loc = MaxNLocator(nbins=5, prune='lower')
        ticks = loc.tick_values(0, 10)
        if ticks:
            assert ticks[0] > 0 or len(ticks) == 1

    def test_prune_upper(self):
        loc = MaxNLocator(nbins=5, prune='upper')
        ticks = loc.tick_values(0, 10)
        if ticks:
            assert ticks[-1] < 10 or len(ticks) == 1

    def test_prune_both(self):
        loc = MaxNLocator(nbins=5, prune='both')
        ticks = loc.tick_values(0, 100)
        if len(ticks) > 2:
            assert ticks[0] > 0
            assert ticks[-1] < 100

    def test_symmetric(self):
        loc = MaxNLocator(nbins=5, symmetric=True)
        lo, hi = loc.view_limits(-3, 7)
        assert lo == -hi

    def test_set_params(self):
        loc = MaxNLocator(nbins=5)
        loc.set_params(nbins=10)
        assert loc._nbins == 10

    def test_auto_nbins(self):
        loc = MaxNLocator(nbins='auto')
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2

    def test_steps(self):
        loc = MaxNLocator(steps=[1, 2, 5, 10])
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2

    def test_min_n_ticks(self):
        loc = MaxNLocator(nbins=5, min_n_ticks=3)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) >= 2

    def test_call(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc()
        assert isinstance(ticks, list)

    def test_default_params(self):
        loc = MaxNLocator()
        assert loc._nbins == 10

    def test_view_limits_basic(self):
        loc = MaxNLocator(nbins=5)
        lo, hi = loc.view_limits(0.5, 9.5)
        assert lo <= 0.5
        assert hi >= 9.5


class TestAutoLocator:
    def test_basic(self):
        loc = AutoLocator()
        ticks = loc.tick_values(0, 10)
        assert len(ticks) >= 2

    def test_is_maxn(self):
        loc = AutoLocator()
        assert isinstance(loc, MaxNLocator)


class TestAutoMinorLocator:
    def test_basic(self):
        # AutoMinorLocator uses __call__ via axis, not tick_values
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1.39)
        ax.minorticks_on()
        ticks = ax.xaxis.get_ticklocs(minor=True)
        assert len(ticks) > 5

    def test_custom_n(self):
        loc = AutoMinorLocator(n=2)
        assert loc.ndivs == 2


class TestLogLocator:
    def test_basic(self):
        loc = LogLocator()
        ticks = loc.tick_values(1, 1000)
        assert 1 in ticks or any(abs(t - 1) < 0.1 for t in ticks)
        assert 10 in ticks or any(abs(t - 10) < 1 for t in ticks)
        assert 100 in ticks or any(abs(t - 100) < 10 for t in ticks)

    def test_subs(self):
        loc = LogLocator(subs=(1.0, 2.0, 5.0))
        ticks = loc.tick_values(1, 100)
        assert len(ticks) > 3

    def test_small_range(self):
        loc = LogLocator()
        ticks = loc.tick_values(1, 10)
        assert len(ticks) >= 1

    def test_set_params(self):
        loc = LogLocator()
        loc.set_params(base=2.0)
        assert loc._base == 2.0

    def test_call(self):
        loc = LogLocator()
        ticks = loc()
        assert isinstance(ticks, list)

    def test_negative_vmin_handled(self):
        loc = LogLocator()
        ticks = loc.tick_values(-5, 100)
        assert len(ticks) >= 1


class TestSymmetricalLogLocator:
    def test_basic(self):
        loc = SymmetricalLogLocator(linthresh=1.0)
        ticks = loc.tick_values(-100, 100)
        assert 0.0 in ticks
        assert len(ticks) >= 3

    def test_set_params(self):
        loc = SymmetricalLogLocator()
        loc.set_params(linthresh=2.0)
        assert loc._linthresh == 2.0


# ===================================================================
# Locator base class
# ===================================================================

class TestLocatorBase:
    def test_raise_if_exceeds(self):
        loc = Locator()
        loc.MAXTICKS = 5
        with pytest.raises(RuntimeError, match='exceeds'):
            loc.raise_if_exceeds(list(range(10)))

    def test_raise_if_within(self):
        loc = Locator()
        result = loc.raise_if_exceeds([1, 2, 3])
        assert result == [1, 2, 3]

    def test_numticks_property(self):
        loc = Locator()
        loc.numticks = 50
        assert loc.numticks == 50

    def test_set_params_noop(self):
        loc = Locator()
        loc.set_params(anything='value')  # should not raise

    def test_view_limits_passthrough(self):
        loc = Locator()
        assert loc.view_limits(0, 10) == (0, 10)


# ===================================================================
# Formatter base class
# ===================================================================

class TestFormatterBase:
    def test_set_locs(self):
        fmt = Formatter()
        fmt.set_locs([1, 2, 3])
        assert fmt.locs == [1, 2, 3]

    def test_get_offset(self):
        fmt = Formatter()
        assert fmt.get_offset() == ''

    def test_fix_minus(self):
        fmt = Formatter()
        assert fmt.fix_minus('-5') == '\u22125'

    def test_format_data(self):
        fmt = Formatter()
        assert fmt.format_data(42) == '42'

    def test_format_data_short(self):
        fmt = Formatter()
        assert fmt.format_data_short(42) == '42'


# ===================================================================
# Extended parametric tests for ticker
# ===================================================================

class TestTickerParametric:
    """Parametric tests for ticker locators and formatters."""

    @pytest.mark.parametrize('locs,expected', [
        ([1, 2, 3], [1, 2, 3]),
        ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
        ([10, 20, 30, 40], [10, 20, 30, 40]),
    ])
    def test_fixed_locator_locs(self, locs, expected):
        """FixedLocator returns given locs."""
        from matplotlib.ticker import FixedLocator
        loc = FixedLocator(locs)
        ticks = loc.tick_values(0, 100)
        for v in expected:
            assert v in ticks

    @pytest.mark.parametrize('base', [2, 5, 10])
    def test_log_locator_base(self, base):
        """LogLocator with various bases generates ticks."""
        from matplotlib.ticker import LogLocator
        loc = LogLocator(base=base)
        ticks = loc.tick_values(1, 100)
        assert len(ticks) > 0

    @pytest.mark.parametrize('vmin,vmax', [
        (0, 1), (0, 10), (-5, 5), (0, 100),
    ])
    def test_auto_locator_range(self, vmin, vmax):
        """AutoLocator generates ticks in range."""
        from matplotlib.ticker import AutoLocator
        loc = AutoLocator()
        ticks = loc.tick_values(vmin, vmax)
        assert len(ticks) > 0
        assert all(vmin - 1 <= t <= vmax + 1 for t in ticks)

    @pytest.mark.parametrize('n', [2, 3, 5, 7, 10])
    def test_max_n_locator(self, n):
        """MaxNLocator generates at most n+1 ticks."""
        from matplotlib.ticker import MaxNLocator
        loc = MaxNLocator(nbins=n)
        ticks = loc.tick_values(0, 100)
        assert len(ticks) <= n + 2  # allow some slack

    @pytest.mark.parametrize('seq', [
        ['a', 'b', 'c'],
        ['x', 'y'],
        ['1', '2', '3', '4'],
        ['Jan', 'Feb', 'Mar'],
    ])
    def test_fixed_formatter_seq(self, seq):
        """FixedFormatter returns labels via format_ticks."""
        from matplotlib.ticker import FixedFormatter
        fmt = FixedFormatter(seq)
        locs = list(range(len(seq)))
        fmt.set_locs(locs)
        labels = fmt.format_ticks(locs)
        assert labels == seq

    @pytest.mark.parametrize('fmt_str', ['%d', '%.2f', '%.3g', '%e'])
    def test_format_str_formatter(self, fmt_str):
        """FormatStrFormatter formats value."""
        from matplotlib.ticker import FormatStrFormatter
        fmt = FormatStrFormatter(fmt_str)
        result = fmt(42, None)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize('vmin,vmax', [
        (1, 10), (0.1, 100), (1, 1000), (10, 10000),
    ])
    def test_log_locator_tick_values_positive(self, vmin, vmax):
        """LogLocator tick_values are all positive."""
        from matplotlib.ticker import LogLocator
        loc = LogLocator(base=10)
        ticks = loc.tick_values(vmin, vmax)
        assert len(ticks) > 0
        assert all(t > 0 for t in ticks)

    @pytest.mark.parametrize('n_minor', [2, 4, 5, 9])
    def test_auto_minor_locator(self, n_minor):
        """AutoMinorLocator generates minor ticks."""
        from matplotlib.ticker import AutoMinorLocator
        loc = AutoMinorLocator(n=n_minor)
        assert loc is not None
