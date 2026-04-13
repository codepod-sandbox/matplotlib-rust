"""
CodePod-specific extensions that supplement the real matplotlib API.

This module holds utilities that our axes.py and pyplot.py depend on
but that don't exist in upstream matplotlib. As we progressively fork
upstream modules, imports here keep the custom code working.
"""

# Default colour cycle (matplotlib C0-C9, matches rcParams['axes.prop_cycle'])
DEFAULT_CYCLE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# -----------------------------------------------------------------------
# Format-string parser
# -----------------------------------------------------------------------

_COLOR_CHARS = set('bgrcmykw')

_MARKER_CHARS = {
    'o': 'circle',
    's': 'square',
    '^': 'triangle_up',
    'v': 'triangle_down',
    '<': 'triangle_left',
    '>': 'triangle_right',
    '1': 'tri_down',
    '2': 'tri_up',
    '3': 'tri_left',
    '4': 'tri_right',
    '+': 'plus',
    'x': 'x',
    'd': 'thin_diamond',
    'D': 'diamond',
    '|': 'vline',
    '_': 'hline',
    'p': 'pentagon',
    'h': 'hexagon1',
    'H': 'hexagon2',
    '*': 'star',
    '.': 'point',
}

_LINE_CHARS = {
    '-': 'solid',
    '--': 'dashed',
    ':': 'dotted',
    '-.': 'dashdot',
}


def parse_fmt(fmt):
    """Parse a matplotlib format string, return ``(color, marker, linestyle)``.

    Parameters
    ----------
    fmt : str
        A format string like ``'ro-'``, ``'b--'``, ``'g^:'``.

    Returns
    -------
    tuple
        ``(color, marker, linestyle)`` — each is a string or None.
    """
    color = None
    marker = None
    linestyle = None
    if not fmt:
        return color, marker, linestyle
    i = 0
    while i < len(fmt):
        ch = fmt[i]
        if ch in _COLOR_CHARS and color is None:
            color = ch
        elif ch in _MARKER_CHARS and marker is None:
            marker = ch
        elif ch == '-' and i + 1 < len(fmt) and fmt[i + 1] in ('-', '.'):
            linestyle = fmt[i:i + 2]
            i += 1
        elif ch in ('-', ':', '.'):
            if linestyle is None:
                linestyle = ch
        i += 1
    return color, marker, linestyle
