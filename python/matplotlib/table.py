"""
matplotlib.table --- Table artist for displaying tabular data on axes.
"""

from matplotlib.artist import Artist
from matplotlib.text import Text
from matplotlib.patches import Rectangle


class Cell:
    """A single cell in a Table."""

    def __init__(self, xy, width, height, text='', loc='right',
                 facecolor='white', edgecolor='black'):
        self._xy = xy
        self._width = width
        self._height = height
        self._text = Text(xy[0] + width / 2, xy[1] + height / 2, text)
        self._loc = loc
        self._facecolor = facecolor
        self._edgecolor = edgecolor
        self._visible = True

    def get_text(self):
        return self._text

    def set_text_props(self, **kwargs):
        for k, v in kwargs.items():
            setter = getattr(self._text, f'set_{k}', None)
            if setter:
                setter(v)

    def get_width(self):
        return self._width

    def set_width(self, w):
        self._width = w

    def get_height(self):
        return self._height

    def set_height(self, h):
        self._height = h

    def get_facecolor(self):
        return self._facecolor

    def set_facecolor(self, color):
        self._facecolor = color

    def get_edgecolor(self):
        return self._edgecolor

    def set_edgecolor(self, color):
        self._edgecolor = color

    def get_loc(self):
        return self._loc

    def set_loc(self, loc):
        self._loc = loc

    def get_celld(self):
        return self

    def visible_edges(self):
        return 'open' if not self._visible else 'closed'

    def set_visible(self, b):
        self._visible = b

    def get_visible(self):
        return self._visible

    @property
    def PAD(self):
        return 0.1

    def __repr__(self):
        return f"Cell(xy={self._xy}, text='{self._text.get_text()}')"


class Table(Artist):
    """A table of cells on an Axes.

    Parameters
    ----------
    ax : Axes
        The parent axes.
    loc : str, default 'bottom'
        Table location.
    """

    FONTSIZE = 10
    AXESPAD = 0.02

    codes = {
        'best': 0,
        'upper right': 1,
        'upper left': 2,
        'lower left': 3,
        'lower right': 4,
        'center left': 5,
        'center right': 6,
        'center': 7,
        'upper center': 8,
        'lower center': 9,
        'bottom': 17,
        'top': 16,
        'left': 15,
        'right': 14,
    }

    def __init__(self, ax, loc='bottom', bbox=None, **kwargs):
        super().__init__()
        self.axes = ax
        self._loc = loc
        self._bbox = bbox
        self._cells = {}
        self._edges = 'closed'
        self._fontsize = kwargs.get('fontsize', self.FONTSIZE)
        self._visible = True

    def add_cell(self, row, col, width=1, height=1, text='', loc='right',
                 facecolor='white', edgecolor='black'):
        """Add a cell to the table at (row, col)."""
        cell = Cell((col * width, -row * height), width, height,
                    text=text, loc=loc,
                    facecolor=facecolor, edgecolor=edgecolor)
        self._cells[(row, col)] = cell
        return cell

    def __setitem__(self, position, cell):
        self._cells[position] = cell

    def __getitem__(self, position):
        return self._cells[position]

    def __contains__(self, position):
        return position in self._cells

    def get_celld(self):
        """Return the dict of cells."""
        return dict(self._cells)

    def get_children(self):
        return list(self._cells.values())

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value):
        self._edges = value

    def set_fontsize(self, size):
        self._fontsize = size
        for cell in self._cells.values():
            cell.get_text().set_fontsize(size)

    def get_fontsize(self):
        return self._fontsize

    def scale(self, xscale, yscale):
        """Scale column widths and row heights."""
        for cell in self._cells.values():
            cell.set_width(cell.get_width() * xscale)
            cell.set_height(cell.get_height() * yscale)

    def auto_set_font_size(self, value=True):
        """Auto-set font size (no-op in this implementation)."""
        self._auto_font = value

    def auto_set_column_width(self, col):
        """Auto-set column width (no-op in this implementation)."""
        pass

    def get_window_extent(self, renderer=None):
        """Return the bounding box."""
        from matplotlib.transforms import Bbox
        if not self._cells:
            return Bbox.unit()
        xs = []
        ys = []
        for cell in self._cells.values():
            xs.extend([cell._xy[0], cell._xy[0] + cell._width])
            ys.extend([cell._xy[1], cell._xy[1] + cell._height])
        return Bbox.from_extents(min(xs), min(ys), max(xs), max(ys))

    def __repr__(self):
        return f"<Table with {len(self._cells)} cells>"


def table(ax, cellText=None, cellColours=None, cellLoc='right',
          colWidths=None, rowLabels=None, rowColours=None, rowLoc='left',
          colLabels=None, colColours=None, colLoc='center',
          loc='bottom', bbox=None, edges='closed', **kwargs):
    """Add a table to an axes.

    Parameters
    ----------
    ax : Axes
    cellText : list of list of str
        The texts to place into the table cells.
    colLabels : list of str
        Column header labels.
    rowLabels : list of str
        Row labels.
    loc : str
        Table location.

    Returns
    -------
    Table
    """
    tbl = Table(ax, loc=loc, bbox=bbox, **kwargs)
    tbl.edges = edges

    if cellText is None:
        cellText = []

    nrows = len(cellText)
    ncols = max((len(row) for row in cellText), default=0)
    if colLabels:
        ncols = max(ncols, len(colLabels))

    if colWidths is None:
        colWidths = [1.0 / max(ncols, 1)] * ncols

    # Column labels (row index -1 or use row 0 offset)
    offset = 0
    if colLabels:
        for j, label in enumerate(colLabels):
            fc = colColours[j] if colColours and j < len(colColours) else 'lightgray'
            tbl.add_cell(0, j, width=colWidths[j] if j < len(colWidths) else 1.0 / ncols,
                        height=0.05, text=str(label), loc=colLoc,
                        facecolor=fc)
        offset = 1

    # Data cells
    for i, row in enumerate(cellText):
        for j, val in enumerate(row):
            fc = 'white'
            if cellColours and i < len(cellColours) and j < len(cellColours[i]):
                fc = cellColours[i][j]
            tbl.add_cell(i + offset, j,
                        width=colWidths[j] if j < len(colWidths) else 1.0 / ncols,
                        height=0.05, text=str(val), loc=cellLoc,
                        facecolor=fc)

    # Row labels
    if rowLabels:
        for i, label in enumerate(rowLabels):
            fc = rowColours[i] if rowColours and i < len(rowColours) else 'lightgray'
            tbl.add_cell(i + offset, -1,
                        width=1.0 / max(ncols, 1), height=0.05,
                        text=str(label), loc=rowLoc,
                        facecolor=fc)

    tbl.axes = ax
    return tbl
