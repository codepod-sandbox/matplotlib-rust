"""
matplotlib.cycler --- Minimal cycler implementation for property cycling.

This provides a lightweight version of the cycler library's functionality,
supporting color cycling and basic iteration.
"""


class Cycler:
    """A composable cycle of properties.

    Parameters
    ----------
    key : str
        Property name (e.g. 'color').
    values : iterable
        Values to cycle through.
    """

    def __init__(self, key=None, values=None):
        if key is not None and values is not None:
            self._keys = [key]
            self._values = [{key: v} for v in values]
        else:
            self._keys = []
            self._values = []

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    def __add__(self, other):
        """Concatenate two cyclers."""
        result = Cycler()
        result._keys = list(set(self._keys + other._keys))
        result._values = list(self._values) + list(other._values)
        return result

    def __mul__(self, other):
        """Outer product of two cyclers."""
        if isinstance(other, int):
            result = Cycler()
            result._keys = list(self._keys)
            result._values = list(self._values) * other
            return result
        result = Cycler()
        result._keys = list(set(self._keys + other._keys))
        result._values = []
        for a in self._values:
            for b in other._values:
                merged = {}
                merged.update(a)
                merged.update(b)
                result._values.append(merged)
        return result

    def __repr__(self):
        return f"cycler({self._keys}, {self._values})"

    def by_key(self):
        """Return a dict mapping keys to lists of values."""
        result = {}
        for key in self._keys:
            result[key] = [d.get(key) for d in self._values]
        return result

    @property
    def keys(self):
        return set(self._keys)


def cycler(key_or_dict=None, values=None, **kwargs):
    """Create a Cycler from a key and values.

    Parameters
    ----------
    key_or_dict : str or dict
        Property name or dict of property lists.
    values : iterable, optional
        Values if key_or_dict is a string.
    **kwargs
        Alternative: keyword arguments.

    Returns
    -------
    Cycler
    """
    if isinstance(key_or_dict, str) and values is not None:
        return Cycler(key_or_dict, values)

    if isinstance(key_or_dict, dict):
        result = None
        for key, vals in key_or_dict.items():
            c = Cycler(key, vals)
            if result is None:
                result = c
            else:
                result = result + c
        return result if result is not None else Cycler()

    if kwargs:
        result = None
        for key, vals in kwargs.items():
            c = Cycler(key, vals)
            if result is None:
                result = c
            else:
                result = result + c
        return result if result is not None else Cycler()

    return Cycler()
