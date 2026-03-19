"""matplotlib.gridspec --- GridSpec for advanced subplot layouts."""


class SubplotSpec:
    """Specification for the location of a subplot in a GridSpec."""

    def __init__(self, gridspec, rowspan, colspan):
        self._gridspec = gridspec
        self.rowspan = rowspan  # (start, stop)
        self.colspan = colspan  # (start, stop)

    def get_gridspec(self):
        return self._gridspec

    @property
    def num1(self):
        """Flat index of the start cell."""
        gs = self._gridspec
        return self.rowspan[0] * gs.ncols + self.colspan[0]

    @property
    def num2(self):
        """Flat index of the end cell (inclusive)."""
        gs = self._gridspec
        return (self.rowspan[1] - 1) * gs.ncols + (self.colspan[1] - 1)

    def get_position(self, figure=None):
        """Return the position as (x0, y0, width, height)."""
        gs = self._gridspec
        nrows, ncols = gs.nrows, gs.ncols
        r0, r1 = self.rowspan
        c0, c1 = self.colspan
        w = (c1 - c0) / ncols
        h = (r1 - r0) / nrows
        x0 = c0 / ncols
        y0 = 1.0 - r1 / nrows
        return (x0, y0, w, h)

    def __repr__(self):
        return (f"SubplotSpec({self.rowspan}, {self.colspan})")


class GridSpec:
    """A grid layout to place subplots within a figure.

    Usage::

        gs = GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1:3])
        ax3 = fig.add_subplot(gs[1, :])
    """

    def __init__(self, nrows, ncols, figure=None, **kwargs):
        self.nrows = nrows
        self.ncols = ncols
        self.figure = figure
        self._hspace = kwargs.get('hspace', None)
        self._wspace = kwargs.get('wspace', None)
        self._width_ratios = kwargs.get('width_ratios', None)
        self._height_ratios = kwargs.get('height_ratios', None)
        self._left = kwargs.get('left', None)
        self._right = kwargs.get('right', None)
        self._top = kwargs.get('top', None)
        self._bottom = kwargs.get('bottom', None)

    def __getitem__(self, key):
        """Return a SubplotSpec for the given grid position.

        Supports integer indexing and slicing:
            gs[0]        -> flat index (row-major)
            gs[0, 0]     -> single cell
            gs[0, :]     -> full row
            gs[:, 0]     -> full column
            gs[0:2, 0:2] -> block
            gs[0:4]      -> flat slice (row-major)
        """
        if isinstance(key, int):
            # Flat integer index
            row = key // self.ncols
            col = key % self.ncols
            return SubplotSpec(self, (row, row + 1), (col, col + 1))

        if isinstance(key, slice):
            # Flat slice - convert to row/col spans
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.nrows * self.ncols
            r0 = start // self.ncols
            c0 = start % self.ncols
            r1 = (stop - 1) // self.ncols + 1
            c1 = self.ncols  # full width for flat slices
            return SubplotSpec(self, (r0, r1), (c0 if c0 == 0 else 0, c1))

        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("GridSpec index must be a 2-tuple (row, col) or integer")

        row_key, col_key = key

        # Normalize row
        if isinstance(row_key, int):
            if row_key < 0:
                row_key = self.nrows + row_key
            rowspan = (row_key, row_key + 1)
        elif isinstance(row_key, slice):
            start = row_key.start if row_key.start is not None else 0
            stop = row_key.stop if row_key.stop is not None else self.nrows
            if start < 0:
                start = self.nrows + start
            if stop < 0:
                stop = self.nrows + stop
            rowspan = (start, stop)
        else:
            raise IndexError(f"Invalid row index: {row_key}")

        # Normalize col
        if isinstance(col_key, int):
            if col_key < 0:
                col_key = self.ncols + col_key
            colspan = (col_key, col_key + 1)
        elif isinstance(col_key, slice):
            start = col_key.start if col_key.start is not None else 0
            stop = col_key.stop if col_key.stop is not None else self.ncols
            if start < 0:
                start = self.ncols + start
            if stop < 0:
                stop = self.ncols + stop
            colspan = (start, stop)
        else:
            raise IndexError(f"Invalid col index: {col_key}")

        return SubplotSpec(self, rowspan, colspan)

    def get_subplot_params(self):
        """Return subplot parameters."""
        return _SubplotParams(
            left=self._left, right=self._right,
            top=self._top, bottom=self._bottom,
            hspace=self._hspace, wspace=self._wspace,
        )

    def get_geometry(self):
        """Return (nrows, ncols)."""
        return (self.nrows, self.ncols)

    def get_width_ratios(self):
        return self._width_ratios

    def get_height_ratios(self):
        return self._height_ratios

    def update(self, **kwargs):
        """Update subplot parameters."""
        for k, v in kwargs.items():
            setattr(self, f'_{k}', v)

    def tight_layout(self, figure=None, **kwargs):
        """No-op."""
        pass

    def subplots(self, **kwargs):
        """Create subplots for this GridSpec (requires figure)."""
        if self.figure is None:
            raise ValueError("GridSpec must have a figure to call subplots()")
        axes = []
        for r in range(self.nrows):
            row_axes = []
            for c in range(self.ncols):
                ss = self[r, c]
                ax = self.figure.add_subplot(ss)
                row_axes.append(ax)
            axes.append(row_axes)
        if self.nrows == 1 and self.ncols == 1:
            return axes[0][0]
        if self.nrows == 1:
            return axes[0]
        if self.ncols == 1:
            return [row[0] for row in axes]
        return axes

    def __repr__(self):
        return f"GridSpec({self.nrows}, {self.ncols})"


class GridSpecFromSubplotSpec(GridSpec):
    """A GridSpec created from a SubplotSpec (nested grids)."""

    def __init__(self, nrows, ncols, subplot_spec, **kwargs):
        super().__init__(nrows, ncols, **kwargs)
        self._subplot_spec = subplot_spec


class _SubplotParams:
    """Subplot parameters."""

    def __init__(self, left=None, right=None, top=None, bottom=None,
                 hspace=None, wspace=None):
        self.left = left or 0.125
        self.right = right or 0.9
        self.top = top or 0.88
        self.bottom = bottom or 0.11
        self.hspace = hspace or 0.2
        self.wspace = wspace or 0.2
