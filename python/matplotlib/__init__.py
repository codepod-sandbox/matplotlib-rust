"""
matplotlib — plotting library for codepod.

SVG primary output with optional PIL/PNG backend.
"""

__version__ = "3.8.0"

import os as _os
import pathlib as _pathlib

# Data path stub — no bundled data in WASM sandbox
_DATA_PATH = _pathlib.Path(_os.path.dirname(_os.path.abspath(__file__)), "_data")


def get_data_path():
    """Return path to matplotlib data directory (stub for WASM sandbox)."""
    return str(_DATA_PATH)


from matplotlib.rcsetup import RcParams, _default_params, rc_context
from matplotlib import style
from matplotlib.cm import _colormaps as colormaps
from matplotlib.colors import _color_sequences as color_sequences

# Global configuration parameters
rcParams = RcParams(_default_params)

# Interactive mode flag
_interactive = False


def is_interactive():
    """Return whether interactive mode is enabled."""
    return _interactive


def _val_or_rc(val, rc_name):
    """If *val* is None, return ``rcParams[rc_name]``, otherwise return val."""
    return val if val is not None else rcParams[rc_name]


def rc(group, **kwargs):
    """Set rcParams for a *group* of parameters.

    Parameters
    ----------
    group : str
        The group name, e.g. ``'lines'``, ``'axes'``, ``'figure'``.
    **kwargs
        Key/value pairs to set.  Each key is prefixed with
        ``group + '.'`` to form the full rcParams key.

    Example
    -------
    >>> import matplotlib
    >>> matplotlib.rc('lines', linewidth=2.0, linestyle='-')
    """
    for k, v in kwargs.items():
        key = f'{group}.{k}'
        rcParams[key] = v


def _replacer(data, value):
    """Either returns ``data[value]`` or passes ``data`` back."""
    try:
        if isinstance(value, str):
            value = data[value]
    except Exception:
        pass
    try:
        from matplotlib.cbook import sanitize_sequence
        return sanitize_sequence(value)
    except Exception:
        return value


def _label_from_arg(y, default_name):
    try:
        return y.name
    except AttributeError:
        if isinstance(default_name, str):
            return default_name
    return None


def _add_data_doc(docstring, replace_names):
    """Add documentation for a *data* field to the given docstring."""
    if (docstring is None
            or replace_names is not None and len(replace_names) == 0):
        return docstring
    import inspect
    docstring = inspect.cleandoc(docstring)
    data_doc = (
        "If given, all parameters also accept a string ``s``, which is "
        "interpreted as ``data[s]`` if ``s`` is a key in ``data``."
        if replace_names is None else
        "If given, the following parameters also accept a string ``s``, "
        "which is interpreted as ``data[s]`` if ``s`` is a key in "
        f"``data``:\n\n    {', '.join(map('*{}*'.format, replace_names))}"
    )
    return docstring + "\n\nParameters\n----------\ndata : indexable object, optional\n    " + data_doc


def _preprocess_data(func=None, *, replace_names=None, label_namer=None):
    """
    A decorator to add a 'data' kwarg to a function.

    When applied::

        @_preprocess_data()
        def func(ax, *args, **kwargs): ...

    the signature is modified to ``decorated(ax, *args, data=None, **kwargs)``
    """
    import functools
    import inspect
    from inspect import Parameter

    if func is None:
        return functools.partial(
            _preprocess_data,
            replace_names=replace_names, label_namer=label_namer)

    sig = inspect.signature(func)
    varargs_name = None
    varkwargs_name = None
    arg_names = []
    params = list(sig.parameters.values())
    for p in params:
        if p.kind is Parameter.VAR_POSITIONAL:
            varargs_name = p.name
        elif p.kind is Parameter.VAR_KEYWORD:
            varkwargs_name = p.name
        else:
            arg_names.append(p.name)
    data_param = Parameter("data", Parameter.KEYWORD_ONLY, default=None)
    if varkwargs_name:
        params.insert(-1, data_param)
    else:
        params.append(data_param)
    new_sig = sig.replace(parameters=params)
    arg_names = arg_names[1:]  # remove the first "ax" / self arg

    @functools.wraps(func)
    def inner(ax, *args, data=None, **kwargs):
        if data is None:
            try:
                from matplotlib.cbook import sanitize_sequence
                return func(
                    ax,
                    *map(sanitize_sequence, args),
                    **{k: sanitize_sequence(v) for k, v in kwargs.items()})
            except Exception:
                return func(ax, *args, **kwargs)

        bound = new_sig.bind(ax, *args, **kwargs)
        auto_label = (
            (bound.arguments.get(label_namer) or bound.kwargs.get(label_namer))
            if label_namer is not None else None
        )

        for k, v in bound.arguments.items():
            if k == varkwargs_name:
                for k1, v1 in v.items():
                    if replace_names is None or k1 in replace_names:
                        v[k1] = _replacer(data, v1)
            elif k == varargs_name:
                if replace_names is None:
                    bound.arguments[k] = tuple(_replacer(data, v1) for v1 in v)
            else:
                if replace_names is None or k in replace_names:
                    bound.arguments[k] = _replacer(data, v)

        new_args = bound.args
        new_kwargs = bound.kwargs

        args_and_kwargs = {**bound.arguments, **bound.kwargs}
        if label_namer and "label" not in args_and_kwargs:
            new_kwargs["label"] = _label_from_arg(
                args_and_kwargs.get(label_namer), auto_label)

        return func(*new_args, **new_kwargs)

    inner.__doc__ = _add_data_doc(inner.__doc__, replace_names)
    inner.__signature__ = new_sig  # type: ignore[attr-defined]
    return inner
