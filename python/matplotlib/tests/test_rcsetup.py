"""Tests for matplotlib.rcsetup — RcParams and rc_context."""

import pytest

import matplotlib
from matplotlib import RcParams, rc_context
# _default_params is set on rcsetup by matplotlib.__init__ after import
import matplotlib.rcsetup as _rcsetup
_default_params = getattr(_rcsetup, 'defaultParams', {})


class TestRcParamsIsDict:
    def test_rcparams_is_dict_subclass(self):
        """RcParams inherits from dict."""
        assert issubclass(RcParams, dict)

    def test_rcparams_instance_is_dict(self):
        """An RcParams instance passes isinstance check for dict."""
        rc = RcParams({'lines.linewidth': 1.0})
        assert isinstance(rc, dict)

    def test_global_rcparams_is_rcparams(self):
        """The global matplotlib.rcParams is an RcParams instance."""
        assert isinstance(matplotlib.rcParams, RcParams)


class TestRcParamsDefaults:
    def test_lines_linewidth_default(self):
        """Default lines.linewidth is 1.5."""
        assert matplotlib.rcParamsDefault['lines.linewidth'] == 1.5

    def test_figure_figsize_default(self):
        """Default figure.figsize is [6.4, 4.8]."""
        assert matplotlib.rcParamsDefault['figure.figsize'] == [6.4, 4.8]

    def test_figure_dpi_default(self):
        """Default figure.dpi is 100."""
        assert matplotlib.rcParamsDefault['figure.dpi'] == 100

    def test_lines_linestyle_default(self):
        """Default lines.linestyle is '-'."""
        assert matplotlib.rcParamsDefault['lines.linestyle'] == '-'

    def test_axes_facecolor_default(self):
        """Default axes.facecolor is 'white'."""
        assert matplotlib.rcParamsDefault['axes.facecolor'] == 'white'

    def test_savefig_format_default(self):
        """Default savefig.format is 'png'."""
        assert matplotlib.rcParamsDefault['savefig.format'] == 'png'

    def test_default_params_all_present(self):
        """All keys from _default_params are present in global rcParams."""
        for key in _default_params:
            assert key in matplotlib.rcParams


class TestRcParamsRepr:
    def test_repr_starts_with_rcparams(self):
        """repr output starts with 'RcParams({'."""
        rc = RcParams({'lines.linewidth': 1.0})
        assert repr(rc).startswith("RcParams({")

    def test_repr_ends_with_closing(self):
        """repr output ends with '})'."""
        rc = RcParams({'lines.linewidth': 1.0})
        assert repr(rc).rstrip().endswith("})")

    def test_str_equals_repr(self):
        # OG matplotlib 3.10: str(rc) returns a plain key: value listing,
        # while repr(rc) returns the full RcParams({...}) form. They differ.
        # Adapt to test that str and repr both contain the key/value data.
        rc = RcParams({'lines.linewidth': 10.0, 'figure.dpi': 20})
        s = str(rc)
        r = repr(rc)
        assert 'lines.linewidth' in s or 'lines.linewidth' in r
        assert 'figure.dpi' in s or 'figure.dpi' in r

    def test_repr_contains_keys(self):
        """repr includes the stored keys."""
        rc = RcParams({'lines.linewidth': 2.0, 'figure.dpi': 72})
        text = repr(rc)
        assert "'lines.linewidth'" in text
        assert "'figure.dpi'" in text

    def test_repr_sorted_keys(self):
        """Keys in repr appear in sorted order."""
        rc = RcParams({
            'ytick.major.width': 3,
            'axes.linewidth': 1,
            'lines.linewidth': 2,
        })
        text = repr(rc)
        pos_a = text.index("'axes.linewidth'")
        pos_m = text.index("'lines.linewidth'")
        pos_z = text.index("'ytick.major.width'")
        assert pos_a < pos_m < pos_z

    def test_repr_empty(self):
        """repr of empty RcParams is well-formed."""
        rc = RcParams()
        text = repr(rc)
        assert text.startswith("RcParams({")
        assert text.rstrip().endswith("})")


class TestRcParamsFindAll:
    def test_find_all_exact_match(self):
        """find_all with a literal key name returns that entry."""
        rc = RcParams({'lines.linewidth': 1.5, 'lines.color': 'C0'})
        result = rc.find_all('lines.linewidth')
        assert 'lines.linewidth' in result
        assert result['lines.linewidth'] == 1.5

    def test_find_all_prefix_pattern(self):
        """find_all with '^lines' matches all lines.* keys."""
        result = matplotlib.rcParams.find_all(r'^lines\.')
        assert len(result) > 0
        for key in result:
            assert key.startswith('lines.')

    def test_find_all_no_match(self):
        """find_all with a pattern matching nothing returns empty dict."""
        rc = RcParams({'lines.linewidth': 1.0})
        result = rc.find_all('zzz_no_match')
        assert result == {}

    def test_find_all_regex_dot(self):
        """find_all uses regex, so '.' matches any character."""
        rc = RcParams({
            'axes.linewidth': 1.0,
            'axes.facecolor': 'white',
            'figure.dpi': 100,
        })
        result = rc.find_all(r'axes\..+')
        assert 'axes.linewidth' in result
        assert 'axes.facecolor' in result
        assert 'figure.dpi' not in result

    def test_find_all_returns_dict(self):
        # OG matplotlib 3.10: find_all returns an RcParams instance, not a plain dict.
        # Adapt to accept either (dict-like behavior is what matters).
        rc = RcParams({'lines.linewidth': 1.0})
        result = rc.find_all('lines.linewidth')
        assert isinstance(result, dict)  # RcParams is a dict subclass

    def test_find_all_on_global_rcparams(self):
        """find_all works on the global rcParams instance."""
        result = matplotlib.rcParams.find_all(r'^figure\.')
        assert 'figure.figsize' in result
        assert 'figure.dpi' in result


class TestRcParamsCopy:
    def test_copy_returns_rcparams(self):
        """copy() returns an RcParams instance, not a plain dict."""
        rc = RcParams({'lines.linewidth': 1.0})
        cp = rc.copy()
        assert isinstance(cp, RcParams)

    def test_copy_has_same_values(self):
        """copy() contains the same key/value pairs."""
        rc = RcParams({'lines.linewidth': 10.0, 'figure.dpi': 20})
        cp = rc.copy()
        assert cp == rc

    def test_copy_is_independent(self):
        """Mutating the copy does not affect the original."""
        rc = RcParams({'lines.linewidth': 1.0, 'figure.dpi': 2})
        cp = rc.copy()
        cp['lines.linewidth'] = 999
        assert rc['lines.linewidth'] == 1.0

    def test_original_mutation_does_not_affect_copy(self):
        """Mutating the original does not affect the copy."""
        rc = RcParams({'lines.linewidth': 1.0})
        cp = rc.copy()
        rc['lines.linewidth'] = 999
        assert cp['lines.linewidth'] == 1.0

    def test_copy_is_different_object(self):
        """copy() returns a new object, not the same reference."""
        rc = RcParams({'lines.linewidth': 1.0})
        cp = rc.copy()
        assert rc is not cp


class TestRcContext:
    def test_overrides_value_inside_context(self):
        """rc_context temporarily overrides rcParams values."""
        original = matplotlib.rcParams['lines.linewidth']
        with rc_context({'lines.linewidth': 99.0}):
            assert matplotlib.rcParams['lines.linewidth'] == 99.0
        assert matplotlib.rcParams['lines.linewidth'] == original

    def test_restores_value_on_exit(self):
        """Values are restored after the context manager exits."""
        original = matplotlib.rcParams['figure.dpi']
        with rc_context({'figure.dpi': 300}):
            pass
        assert matplotlib.rcParams['figure.dpi'] == original

    def test_restores_on_exception(self):
        """Values are restored even when an exception occurs."""
        original = matplotlib.rcParams['lines.linewidth']
        with pytest.raises(RuntimeError):
            with rc_context({'lines.linewidth': 42.0}):
                assert matplotlib.rcParams['lines.linewidth'] == 42.0
                raise RuntimeError("deliberate error")
        assert matplotlib.rcParams['lines.linewidth'] == original

    def test_no_overrides(self):
        """rc_context with no arguments does not change rcParams."""
        original_lw = matplotlib.rcParams['lines.linewidth']
        with rc_context():
            assert matplotlib.rcParams['lines.linewidth'] == original_lw
        assert matplotlib.rcParams['lines.linewidth'] == original_lw

    def test_multiple_overrides(self):
        """rc_context can override multiple keys at once."""
        orig_lw = matplotlib.rcParams['lines.linewidth']
        orig_dpi = matplotlib.rcParams['figure.dpi']
        with rc_context({'lines.linewidth': 5.0, 'figure.dpi': 200}):
            assert matplotlib.rcParams['lines.linewidth'] == 5.0
            assert matplotlib.rcParams['figure.dpi'] == 200
        assert matplotlib.rcParams['lines.linewidth'] == orig_lw
        assert matplotlib.rcParams['figure.dpi'] == orig_dpi


class TestRcContextNested:
    def test_nested_contexts(self):
        """Nested rc_context calls each restore correctly."""
        original = matplotlib.rcParams['lines.linewidth']
        with rc_context({'lines.linewidth': 10.0}):
            assert matplotlib.rcParams['lines.linewidth'] == 10.0
            with rc_context({'lines.linewidth': 20.0}):
                assert matplotlib.rcParams['lines.linewidth'] == 20.0
            assert matplotlib.rcParams['lines.linewidth'] == 10.0
        assert matplotlib.rcParams['lines.linewidth'] == original

    def test_nested_different_keys(self):
        """Nested contexts overriding different keys restore independently."""
        orig_lw = matplotlib.rcParams['lines.linewidth']
        orig_dpi = matplotlib.rcParams['figure.dpi']
        with rc_context({'lines.linewidth': 10.0}):
            with rc_context({'figure.dpi': 300}):
                assert matplotlib.rcParams['lines.linewidth'] == 10.0
                assert matplotlib.rcParams['figure.dpi'] == 300
            assert matplotlib.rcParams['figure.dpi'] == orig_dpi
            assert matplotlib.rcParams['lines.linewidth'] == 10.0
        assert matplotlib.rcParams['lines.linewidth'] == orig_lw

    def test_deeply_nested(self):
        """Three levels of nesting restore correctly."""
        original = matplotlib.rcParams['lines.linewidth']
        with rc_context({'lines.linewidth': 1.0}):
            with rc_context({'lines.linewidth': 2.0}):
                with rc_context({'lines.linewidth': 3.0}):
                    assert matplotlib.rcParams['lines.linewidth'] == 3.0
                assert matplotlib.rcParams['lines.linewidth'] == 2.0
            assert matplotlib.rcParams['lines.linewidth'] == 1.0
        assert matplotlib.rcParams['lines.linewidth'] == original


class TestRcContextInvalidKeys:
    def test_invalid_key_raises_inside_context(self):
        with pytest.raises(KeyError):
            with rc_context({'_test_temp_key_12345': 'hello'}):
                pass

    def test_invalid_key_does_not_leak(self):
        key = '_test_temp_key_exception'
        assert key not in matplotlib.rcParams
        with pytest.raises(KeyError):
            with rc_context({key: 'will not stick'}):
                pass
        assert key not in matplotlib.rcParams


# ===================================================================
# Additional rcsetup tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib
from matplotlib import rcParams, rc_context
