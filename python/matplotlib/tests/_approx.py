"""Tolerance comparison helper that avoids numpy/pytest.approx issues."""


def approx(expected, abs=1e-7, rel=1e-7):
    """Return an _Approx object for use with == comparisons.

    This avoids pytest.approx issues with numpy.bool_ in some environments.
    """
    return _Approx(expected, abs=abs, rel=rel)


class _Approx:
    def __init__(self, expected, abs=1e-7, rel=1e-7):
        self.expected = expected
        self._abs = abs
        self._rel = rel

    def _close(self, actual, expected):
        try:
            a = float(actual)
            e = float(expected)
        except (TypeError, ValueError):
            return NotImplemented
        tol = max(self._abs, self._rel * max(1.0, builtins_abs(e)))
        return builtins_abs(a - e) <= tol

    def __eq__(self, other):
        # Handle tuple/list comparison
        if isinstance(self.expected, (tuple, list)):
            if not isinstance(other, (tuple, list)):
                return NotImplemented
            if len(other) != len(self.expected):
                return False
            for a, e in zip(other, self.expected):
                result = self._close(a, e)
                if result is NotImplemented:
                    return NotImplemented
                if not result:
                    return False
            return True

        return self._close(other, self.expected)

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __repr__(self):
        return f"approx({self.expected!r})"


# Save builtins.abs since we shadow it with our parameter name
import builtins as _builtins
builtins_abs = _builtins.abs
