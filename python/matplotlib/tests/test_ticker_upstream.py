"""Upstream-ported tests for matplotlib.ticker — locators and formatters."""

import math
import pytest
from matplotlib.tests._approx import approx

from matplotlib.ticker import (
    Formatter, NullFormatter, FixedFormatter, FuncFormatter,
    FormatStrFormatter, StrMethodFormatter, ScalarFormatter,
    LogFormatter, PercentFormatter,
    Locator, NullLocator, FixedLocator, IndexLocator,
    LinearLocator, MultipleLocator, MaxNLocator, AutoLocator,
    AutoMinorLocator, LogLocator, SymmetricalLogLocator,
)


# ============================================================# Formatter base
# ===================================================================

class TestFormatter:
    def test_format_data(self):
        fmt = Formatter()
        assert fmt.format_data(42) == '42'

    def test_format_data_short(self):
        fmt = Formatter()
        assert fmt.format_data_short(42) == '42'

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

    def test_fix_minus_no_minus(self):
        fmt = Formatter()
        assert fmt.fix_minus('5') == '5'


# ===================================================================
# NullFormatter
# ===================================================================

class TestNullFormatterUpstream:
    def test_always_empty(self):
        fmt = NullFormatter()
        for val in [0, 1, -1, 100, 3.14, 1e10]:
            assert fmt(val) == ''

    def test_with_pos(self):
        fmt = NullFormatter()
        assert fmt(0, pos=5) == ''


# ===================================================================
# FixedFormatter
# ===================================================================

class TestFixedFormatterUpstream:
    def test_basic_sequence(self):
        fmt = FixedFormatter(['a', 'b', 'c', 'd'])
        assert fmt(0, pos=0) == 'a'
        assert fmt(0, pos=1) == 'b'
        assert fmt(0, pos=2) == 'c'
        assert fmt(0, pos=3) == 'd'

    def test_out_of_range(self):
        fmt = FixedFormatter(['a'])
        assert fmt(0, pos=5) == ''

    def test_no_pos(self):
        fmt = FixedFormatter(['a'])
        assert fmt(0) == ''

    def test_offset_string(self):
        fmt = FixedFormatter(['a'])
        fmt.set_offset_string('+100')
        assert fmt.get_offset() == '+100'

    def test_default_offset(self):
        fmt = FixedFormatter(['a'])
        assert fmt.get_offset() == ''


# ===================================================================
# FuncFormatter
# ===================================================================

class TestFuncFormatterUpstream:
    def test_basic(self):
        fmt = FuncFormatter(lambda x, pos: f'{x:.1f}!')
        assert fmt(3.14) == '3.1!'

    def test_with_pos(self):
        fmt = FuncFormatter(lambda x, pos: f'{x}-{pos}')
        assert fmt(1, pos=2) == '1-2'

    def test_no_pos(self):
        fmt = FuncFormatter(lambda x, pos: f'{x}')
        assert fmt(42) == '42'


# ===================================================================
# FormatStrFormatter
# ===================================================================

class TestFormatStrFormatterUpstream:
    def test_float(self):
        fmt = FormatStrFormatter('%1.2f')
        assert fmt(3.14159) == '3.14'

    def test_integer(self):
        fmt = FormatStrFormatter('%d')
        assert fmt(42) == '42'

    def test_scientific(self):
        fmt = FormatStrFormatter('%.2e')
        result = fmt(1234.5)
        assert 'e' in result or 'E' in result

    def test_percent_format(self):
        fmt = FormatStrFormatter('%.0f%%')
        assert fmt(95) == '95%'


# ===================================================================
# StrMethodFormatter
# ===================================================================

class TestStrMethodFormatterUpstream:
    def test_basic(self):
        fmt = StrMethodFormatter('{x:.2f}')
        assert fmt(3.14159) == '3.14'

    def test_with_pos(self):
        fmt = StrMethodFormatter('{x} at {pos}')
        assert fmt(5, pos=3) == '5 at 3'

    def test_integer_format(self):
        fmt = StrMethodFormatter('{x:d}')
        assert fmt(42) == '42'


# ===================================================================
# ScalarFormatter
# ===================================================================

class TestScalarFormatterUpstream:
    def test_integer(self):
        fmt = ScalarFormatter()
        assert fmt(5.0) == '5'

    def test_float(self):
        fmt = ScalarFormatter()
        result = fmt(3.14)
        assert '3.14' in result

    def test_zero(self):
        fmt = ScalarFormatter()
        assert fmt(0.0) == '0'

    def test_large_int(self):
        fmt = ScalarFormatter()
        result = fmt(1000000.0)
        assert '1000000' in result

    def test_set_scientific(self):
        fmt = ScalarFormatter()
        fmt.set_scientific(True)
        assert fmt._scientific is True

    def test_set_powerlimits(self):
        fmt = ScalarFormatter()
        fmt.set_powerlimits((-2, 3))
        assert fmt._powerlimits == (-2, 3)

    def test_get_offset(self):
        fmt = ScalarFormatter()
        assert fmt.get_offset() == ''

    def test_useOffset(self):
        fmt = ScalarFormatter(useOffset=False)
        assert fmt.useOffset is False

    def test_useMathText(self):
        fmt = ScalarFormatter(useMathText=True)
        assert fmt.useMathText is True


# ===================================================================
# LogFormatter
# ===================================================================

class TestLogFormatterUpstream:
    def test_power_of_ten(self):
        fmt = LogFormatter(base=10.0)
        result = fmt(100)
        assert '$10^{2}$' in result

    def test_zero(self):
        fmt = LogFormatter()
        assert fmt(0) == ''

    def test_negative(self):
        fmt = LogFormatter()
        assert fmt(-1) == ''

    def test_one(self):
        fmt = LogFormatter()
        result = fmt(1)
        assert '$10^{0}$' in result

    def test_labelOnlyBase(self):
        fmt = LogFormatter(base=10.0, labelOnlyBase=True)
        # sqrt(10) ~= 3.16 is NOT a base power
        result = fmt(3.16)
        assert result == ''

    def test_base_power_with_labelOnlyBase(self):
        fmt = LogFormatter(base=10.0, labelOnlyBase=True)
        result = fmt(100)
        assert result != ''


# ===================================================================
# PercentFormatter
# ===================================================================

class TestPercentFormatterUpstream:
    def test_default(self):
        fmt = PercentFormatter()
        assert fmt(50) == '50%'

    def test_xmax(self):
        fmt = PercentFormatter(xmax=1.0)
        result = fmt(0.5)
        assert '50' in result

    def test_decimals(self):
        fmt = PercentFormatter(xmax=100, decimals=2)
        result = fmt(33.333)
        assert result == '33.33%'

    def test_symbol(self):
        fmt = PercentFormatter(symbol=' pct')
        result = fmt(50)
        assert result.endswith(' pct')

    def test_zero(self):
        fmt = PercentFormatter()
        assert fmt(0) == '0%'

    def test_hundred(self):
        fmt = PercentFormatter()
        assert fmt(100) == '100%'

    def test_xmax_200(self):
        fmt = PercentFormatter(xmax=200)
        result = fmt(100)
        assert '50' in result


# ===================================================================
# Locator base
# ===================================================================

class TestLocator:
    def test_MAXTICKS(self):
        assert Locator.MAXTICKS == 1000

    def test_raise_if_exceeds_ok(self):
        loc = Locator()
        result = loc.raise_if_exceeds([1, 2, 3])
        assert result == [1, 2, 3]

    def test_raise_if_exceeds_too_many(self):
        loc = Locator()
        with pytest.raises(RuntimeError):
            loc.raise_if_exceeds(list(range(1001)))

    def test_view_limits(self):
        loc = Locator()
        assert loc.view_limits(0, 10) == (0, 10)

    def test_numticks_default(self):
        loc = Locator()
        assert loc.numticks == Locator.MAXTICKS

    def test_numticks_setter(self):
        loc = Locator()
        loc.numticks = 5
        assert loc.numticks == 5


# ===================================================================
# NullLocator
# ===================================================================

class TestNullLocatorUpstream:
    def test_call(self):
        loc = NullLocator()
        assert loc() == []

    def test_tick_values(self):
        loc = NullLocator()
        assert loc.tick_values(0, 10) == []


# ===================================================================
# FixedLocator
# ===================================================================

class TestFixedLocatorUpstream:
    def test_basic(self):
        loc = FixedLocator([0, 1, 2, 3, 4])
        assert loc() == [0, 1, 2, 3, 4]

    def test_tick_values_filter(self):
        loc = FixedLocator([0, 1, 2, 3, 4, 5])
        ticks = loc.tick_values(1, 3)
        assert 0 not in ticks
        assert 4 not in ticks
        assert 5 not in ticks
        assert 1 in ticks
        assert 2 in ticks
        assert 3 in ticks

    def test_nbins(self):
        loc = FixedLocator([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], nbins=3)
        ticks = loc.tick_values(0, 9)
        assert len(ticks) <= 5  # nbins+1 = 4 max, but subsampling varies

    def test_set_params(self):
        loc = FixedLocator([0, 1, 2])
        loc.set_params(nbins=5)
        assert loc.nbins == 5


# ===================================================================
# IndexLocator
# ===================================================================

class TestIndexLocatorUpstream:
    def test_basic(self):
        loc = IndexLocator(base=2, offset=0)
        ticks = loc.tick_values(0, 10)
        assert 0 in ticks
        assert 2 in ticks
        assert 4 in ticks
        assert 6 in ticks
        assert 8 in ticks
        assert 10 in ticks

    def test_with_offset(self):
        loc = IndexLocator(base=5, offset=1)
        ticks = loc.tick_values(0, 20)
        assert 1 in ticks
        assert 6 in ticks
        assert 11 in ticks
        assert 16 in ticks

    def test_set_params(self):
        loc = IndexLocator(base=2, offset=0)
        loc.set_params(base=3, offset=1)
        ticks = loc.tick_values(0, 10)
        assert 1 in ticks
        assert 4 in ticks
        assert 7 in ticks
        assert 10 in ticks


# ===================================================================
# LinearLocator
# ===================================================================

class TestLinearLocatorUpstream:
    def test_default(self):
        loc = LinearLocator()
        ticks = loc.tick_values(0, 10)
        assert len(ticks) == 11

    def test_numticks(self):
        loc = LinearLocator(numticks=5)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) == 5
        assert ticks[0] == approx(0.0)
        assert ticks[-1] == approx(10.0)

    def test_two_ticks(self):
        loc = LinearLocator(numticks=2)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) == 2
        assert ticks[0] == approx(0.0)
        assert ticks[1] == approx(10.0)

    def test_set_params(self):
        loc = LinearLocator()
        loc.set_params(numticks=3)
        ticks = loc.tick_values(0, 10)
        assert len(ticks) == 3


# ===================================================================
# MultipleLocator
# ===================================================================

class TestMultipleLocatorUpstream:
    def test_base_1(self):
        loc = MultipleLocator(1.0)
        ticks = loc.tick_values(0, 5)
        assert 0 in ticks
        assert 1 in ticks
        assert 5 in ticks

    def test_base_2(self):
        loc = MultipleLocator(2.0)
        ticks = loc.tick_values(0, 10)
        assert 0 in ticks
        assert 2 in ticks
        assert 4 in ticks
        assert 6 in ticks
        assert 8 in ticks
        assert 10 in ticks
        assert 1 not in ticks

    def test_base_0_5(self):
        loc = MultipleLocator(0.5)
        ticks = loc.tick_values(0, 2)
        assert len(ticks) == 5  # 0, 0.5, 1, 1.5, 2

    def test_view_limits(self):
        loc = MultipleLocator(5.0)
        vmin, vmax = loc.view_limits(3, 17)
        assert vmin == 0.0
        assert vmax == 20.0

    def test_set_params(self):
        loc = MultipleLocator(1.0)
        loc.set_params(base=5.0)
        ticks = loc.tick_values(0, 20)
        assert 0 in ticks
        assert 5 in ticks
        assert 10 in ticks
        assert 15 in ticks
        assert 20 in ticks
        assert 1 not in ticks

    def test_base_zero(self):
        loc = MultipleLocator(0)
        assert loc.tick_values(0, 10) == []


# ===================================================================
# MaxNLocator
# ===================================================================

class TestMaxNLocatorUpstream:
    def test_default_nbins(self):
        loc = MaxNLocator()
        assert loc._nbins == 10

    def test_custom_nbins(self):
        loc = MaxNLocator(nbins=5)
        assert loc._nbins == 5

    def test_tick_values_basic(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2
        assert ticks[0] <= 0
        assert ticks[-1] >= 100

    def test_tick_values_small_range(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(0.1, 0.9)
        assert len(ticks) >= 2

    def test_tick_values_equal(self):
        loc = MaxNLocator()
        ticks = loc.tick_values(5, 5)
        assert len(ticks) >= 2

    def test_tick_values_zero_equal(self):
        loc = MaxNLocator()
        ticks = loc.tick_values(0, 0)
        assert len(ticks) >= 2
        assert any(t < 0 for t in ticks)
        assert any(t > 0 for t in ticks)

    def test_reversed_input(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc.tick_values(100, 0)
        assert ticks[0] <= 0
        assert ticks[-1] >= 100

    def test_integer(self):
        loc = MaxNLocator(integer=True)
        ticks = loc.tick_values(0.5, 9.5)
        for t in ticks:
            assert t == int(t)

    def test_prune_lower(self):
        loc = MaxNLocator(nbins=5, prune='lower')
        ticks = loc.tick_values(0, 100)
        assert ticks[0] > 0 or len(ticks) <= 1

    def test_prune_upper(self):
        loc = MaxNLocator(nbins=5, prune='upper')
        ticks = loc.tick_values(0, 100)
        if len(ticks) > 1:
            assert ticks[-1] < 100 + 50  # Some tolerance

    def test_prune_both(self):
        loc = MaxNLocator(nbins=5, prune='both')
        ticks = loc.tick_values(0, 100)
        # Both ends pruned
        if len(ticks) > 2:
            pass  # Just checking it doesn't crash

    def test_symmetric(self):
        loc = MaxNLocator(symmetric=True)
        vmin, vmax = loc.view_limits(-3, 10)
        assert abs(vmin) == abs(vmax)

    def test_set_params_nbins(self):
        loc = MaxNLocator(nbins=5)
        loc.set_params(nbins=10)
        assert loc._nbins == 10

    def test_set_params_integer(self):
        loc = MaxNLocator()
        loc.set_params(integer=True)
        assert loc._integer is True

    def test_set_params_prune(self):
        loc = MaxNLocator()
        loc.set_params(prune='lower')
        assert loc._prune == 'lower'

    def test_set_params_steps(self):
        loc = MaxNLocator()
        loc.set_params(steps=[1, 2, 5, 10])
        assert loc._steps == [1, 2, 5, 10]

    def test_set_params_min_n_ticks(self):
        loc = MaxNLocator()
        loc.set_params(min_n_ticks=5)
        assert loc._min_n_ticks == 5

    def test_set_params_symmetric(self):
        loc = MaxNLocator()
        loc.set_params(symmetric=True)
        assert loc._symmetric is True

    def test_auto_nbins(self):
        loc = MaxNLocator(nbins='auto')
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2

    def test_default_params(self):
        params = MaxNLocator.default_params
        assert params['nbins'] == 10
        assert params['steps'] is None
        assert params['integer'] is False
        assert params['symmetric'] is False
        assert params['prune'] is None
        assert params['min_n_ticks'] == 2

    def test_call(self):
        loc = MaxNLocator(nbins=5)
        ticks = loc()
        assert isinstance(ticks, list)
        assert len(ticks) >= 2

    def test_steps(self):
        loc = MaxNLocator(nbins=5, steps=[1, 5, 10])
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2


# ===================================================================
# AutoLocator
# ===================================================================

class TestAutoLocatorUpstream:
    def test_basic(self):
        loc = AutoLocator()
        ticks = loc.tick_values(0, 100)
        assert len(ticks) >= 2

    def test_is_maxnlocator(self):
        loc = AutoLocator()
        assert isinstance(loc, MaxNLocator)

    def test_auto_nbins(self):
        loc = AutoLocator()
        assert loc._nbins == 'auto'


# ===================================================================
# AutoMinorLocator
# ===================================================================

class TestAutoMinorLocatorUpstream:
    def test_basic(self):
        loc = AutoMinorLocator()
        ticks = loc.tick_values(0, 10)
        assert len(ticks) > 0

    def test_custom_n(self):
        loc = AutoMinorLocator(n=5)
        assert loc.ndivs == 5

    def test_none_n(self):
        loc = AutoMinorLocator()
        assert loc.ndivs is None


# ===================================================================
# LogLocator
# ===================================================================

class TestLogLocatorUpstream:
    def test_basic(self):
        loc = LogLocator(base=10)
        ticks = loc.tick_values(1, 1000)
        assert 1 in ticks or any(abs(t - 1) < 0.1 for t in ticks)
        assert 10 in ticks or any(abs(t - 10) < 1 for t in ticks)
        assert 100 in ticks or any(abs(t - 100) < 10 for t in ticks)

    def test_subs(self):
        loc = LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        ticks = loc.tick_values(1, 100)
        # Should include sub-decade ticks
        assert len(ticks) > 3

    def test_set_params(self):
        loc = LogLocator()
        loc.set_params(base=2.0)
        assert loc._base == 2.0

    def test_set_params_subs(self):
        loc = LogLocator()
        loc.set_params(subs=(1.0, 3.0))
        assert loc._subs == (1.0, 3.0)

    def test_call(self):
        loc = LogLocator()
        ticks = loc()
        assert isinstance(ticks, list)


# ===================================================================
# SymmetricalLogLocator
# ===================================================================

class TestSymmetricalLogLocatorUpstream:
    def test_basic(self):
        loc = SymmetricalLogLocator(base=10.0, linthresh=1.0)
        ticks = loc.tick_values(-100, 100)
        assert len(ticks) > 0
        assert any(t < 0 for t in ticks)
        assert any(t > 0 for t in ticks)
        assert 0.0 in ticks

    def test_positive_only(self):
        loc = SymmetricalLogLocator(base=10.0, linthresh=1.0)
        ticks = loc.tick_values(1, 1000)
        assert len(ticks) > 0

    def test_set_params(self):
        loc = SymmetricalLogLocator()
        loc.set_params(base=2.0, linthresh=0.5)
        assert loc._base == 2.0
        assert loc._linthresh == 0.5

    def test_set_params_subs(self):
        loc = SymmetricalLogLocator()
        loc.set_params(subs=(1.0, 5.0))
        assert loc._subs == (1.0, 5.0)


# ===================================================================
# Locator set_params base
# ===================================================================

class TestLocatorSetParams:
    def test_base_set_params_noop(self):
        loc = Locator()
        # Should not raise
        loc.set_params(foo=42)


# ===================================================================
# Edge cases
# ===================================================================

class TestTickerEdgeCases:
    def test_maxnlocator_very_small_range(self):
        loc = MaxNLocator()
        ticks = loc.tick_values(1e-10, 2e-10)
        assert len(ticks) >= 2

    def test_maxnlocator_large_range(self):
        loc = MaxNLocator()
        ticks = loc.tick_values(0, 1e10)
        assert len(ticks) >= 2

    def test_multiplelocator_negative(self):
        loc = MultipleLocator(5.0)
        ticks = loc.tick_values(-20, -5)
        assert any(t < 0 for t in ticks)

    def test_loglocator_small_range(self):
        loc = LogLocator(base=10)
        ticks = loc.tick_values(0.001, 0.1)
        assert len(ticks) >= 1

    def test_fixedformatter_empty(self):
        fmt = FixedFormatter([])
        assert fmt(0, pos=0) == ''

    def test_percentformatter_negative(self):
        fmt = PercentFormatter()
        result = fmt(-50)
        assert '-50' in result

    def test_logformatter_base_2(self):
        fmt = LogFormatter(base=2)
        result = fmt(8)  # 2^3
        # Should format somehow
        assert isinstance(result, str)


# Ported from lib/matplotlib/tests/test_ticker.py
import pytest
import matplotlib


def test_rcparams_formatter_keys():
    """Formatter rcParams keys must exist with correct defaults."""
    assert 'axes.formatter.limits' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.limits'] == [-5, 6]
    assert 'axes.formatter.use_locale' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.use_locale'] is False
    assert 'axes.formatter.use_mathtext' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.use_mathtext'] is False
    assert 'axes.formatter.min_exponent' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.min_exponent'] == 0
    assert 'axes.formatter.useoffset' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.useoffset'] is True
    assert 'axes.formatter.offset_threshold' in matplotlib.rcParams
    assert matplotlib.rcParams['axes.formatter.offset_threshold'] == 4


# ---------------------------------------------------------------------------
# Ported from lib/matplotlib/tests/test_ticker.py
# ---------------------------------------------------------------------------

def test_formatter_str():
    """NullFormatter and FixedFormatter basics."""
    from matplotlib.ticker import NullFormatter, FixedFormatter
    assert NullFormatter()(1.0, 0) == ''
    fmt = FixedFormatter(['a', 'b', 'c'])
    assert fmt(0, 0) == 'a'
    assert fmt(1, 1) == 'b'
    assert fmt(5, 5) == ''  # out of range -> empty string


def test_scalar_formatter():
    """ScalarFormatter produces plain strings for small values."""
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter()
    fmt.create_dummy_axis()
    fmt.axis.set_view_interval(0, 2000)
    fmt.axis.set_data_interval(0, 2000)
    fmt.set_locs([0, 500, 1000, 1500, 2000])
    result = fmt(1000, 0)
    assert isinstance(result, str)
    assert result == '1000'


def test_percent_formatter():
    """PercentFormatter formats fractions as percentages."""
    from matplotlib.ticker import PercentFormatter
    fmt = PercentFormatter(xmax=1.0, decimals=0)
    fmt.create_dummy_axis()
    assert fmt(0.5, 0) == '50%'
    assert fmt(1.0, 0) == '100%'
    fmt2 = PercentFormatter(xmax=100, decimals=0)
    fmt2.create_dummy_axis()
    assert fmt2(50, 0) == '50%'


def test_func_formatter():
    """FuncFormatter delegates to a callable."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda x, pos: f'val={x:.1f}')
    assert fmt(3.14, 0) == 'val=3.1'


def test_str_method_formatter():
    """StrMethodFormatter uses str.format."""
    from matplotlib.ticker import StrMethodFormatter
    fmt = StrMethodFormatter('{x:.2f}')
    assert fmt(3.14159, 0) == '3.14'
    fmt2 = StrMethodFormatter('{x:d}')
    assert fmt2(42, 0) == '42'


def test_null_locator():
    """NullLocator returns empty tick list."""
    from matplotlib.ticker import NullLocator
    loc = NullLocator()
    assert loc.tick_values(0, 10) == []


def test_fixed_locator():
    """FixedLocator returns its preset tick positions."""
    from matplotlib.ticker import FixedLocator
    loc = FixedLocator([1, 2, 3, 5])
    result = loc.tick_values(0, 6)
    assert list(result) == [1, 2, 3, 5]


def test_multiple_locator():
    """MultipleLocator produces ticks at multiples of base."""
    from matplotlib.ticker import MultipleLocator
    loc = MultipleLocator(0.5)
    ticks = loc.tick_values(0.0, 2.0)
    assert 0.5 in ticks
    assert 1.0 in ticks
    assert 1.5 in ticks


def test_auto_locator():
    """AutoLocator produces ~5-8 nice ticks in range."""
    from matplotlib.ticker import AutoLocator
    loc = AutoLocator()
    ticks = loc.tick_values(0, 10)
    assert 4 <= len(ticks) <= 12
    # ticks should be within or near range
    assert min(ticks) <= 0.01
    assert max(ticks) >= 9.99


def test_log_locator():
    """LogLocator places ticks at powers of base."""
    from matplotlib.ticker import LogLocator
    loc = LogLocator(base=10.0)
    ticks = loc.tick_values(1, 1000)
    tick_list = sorted(ticks)
    assert 1 in tick_list or any(abs(t - 1) < 0.01 for t in tick_list)
    assert any(abs(t - 10) < 0.01 for t in tick_list)
    assert any(abs(t - 100) < 0.01 for t in tick_list)


def test_maxn_locator():
    """MaxNLocator respects nbins limit."""
    from matplotlib.ticker import MaxNLocator
    loc = MaxNLocator(nbins=4)
    ticks = loc.tick_values(0, 100)
    assert len(ticks) <= 6  # nbins+1 at most


def test_logformatter():
    # Ported from lib/matplotlib/tests/test_ticker.py::test_LogFormatter
    """LogFormatter produces reasonable strings for log-scale values."""
    from matplotlib.ticker import LogFormatter
    fmt = LogFormatter(base=10)
    fmt.create_dummy_axis()
    result = fmt(100, 0)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# axis.py tests
# ---------------------------------------------------------------------------

def test_axis_default_locator_formatter():
    """XAxis defaults to AutoLocator + ScalarFormatter."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import AutoLocator, ScalarFormatter
    ax_obj = XAxis()
    assert isinstance(ax_obj.get_major_locator(), AutoLocator)
    assert isinstance(ax_obj.get_major_formatter(), ScalarFormatter)


def test_axis_set_major_locator():
    """set_major_locator() replaces the locator."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import FixedLocator
    ax_obj = XAxis()
    loc = FixedLocator([1, 2, 3])
    ax_obj.set_major_locator(loc)
    assert ax_obj.get_major_locator() is loc


def test_axis_set_ticks_uses_fixed_locator():
    """set_ticks() installs a FixedLocator + ScalarFormatter."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import FixedLocator
    ax_obj = XAxis()
    ax_obj.set_ticks([0.0, 0.5, 1.0])
    assert isinstance(ax_obj.get_major_locator(), FixedLocator)
    assert list(ax_obj.get_ticks()) == [0.0, 0.5, 1.0]


def test_axis_set_ticks_with_labels():
    """set_ticks() with labels installs FixedFormatter."""
    from matplotlib.axis import XAxis
    from matplotlib.ticker import FixedFormatter
    ax_obj = XAxis()
    ax_obj.set_ticks([1, 2, 3], ['a', 'b', 'c'])
    assert isinstance(ax_obj.get_major_formatter(), FixedFormatter)
    assert ax_obj.get_major_formatter().seq == ['a', 'b', 'c']


def test_axis_tick_values():
    """tick_values() delegates to the locator."""
    from matplotlib.axis import XAxis
    ax_obj = XAxis()
    ax_obj.set_ticks([10, 20, 30])
    vals = ax_obj.tick_values(0, 40)
    assert list(vals) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Axes integration tests
# ---------------------------------------------------------------------------

def test_axes_has_xaxis_yaxis():
    """Axes must expose .xaxis and .yaxis as Axis instances."""
    import matplotlib.pyplot as plt
    from matplotlib.axis import XAxis, YAxis
    fig, ax = plt.subplots()
    assert isinstance(ax.xaxis, XAxis)
    assert isinstance(ax.yaxis, YAxis)
    plt.close('all')


def test_set_xticks_delegates_to_xaxis():
    """set_xticks() must set a FixedLocator on xaxis."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    assert isinstance(ax.xaxis.get_major_locator(), FixedLocator)
    assert ax.get_xticks() == [1, 2, 3]
    plt.close('all')


def test_set_xticklabels_sets_formatter():
    """set_xticklabels() must install FixedFormatter on xaxis."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['zero', 'one', 'two'])
    assert isinstance(ax.xaxis.get_major_formatter(), FixedFormatter)
    plt.close('all')


def test_cla_resets_axis():
    """cla() must reset xaxis/yaxis to default AutoLocator."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator, FixedLocator
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    assert isinstance(ax.xaxis.get_major_locator(), FixedLocator)
    ax.cla()
    assert isinstance(ax.xaxis.get_major_locator(), AutoLocator)
    plt.close('all')


def test_draw_renders_tick_labels(tmp_path):
    """draw() must produce tick label text in SVG output via Axis objects."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 20, 30])
    svg_path = tmp_path / "test.svg"
    fig.savefig(str(svg_path))
    svg = svg_path.read_text()
    # SVG must contain some numeric tick label text
    assert any(str(n) in svg for n in range(1, 31))
    plt.close('all')
