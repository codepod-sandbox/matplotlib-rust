"""matplotlib.image --- Image artists (AxesImage, etc.)."""

from matplotlib.artist import Artist


class AxesImage(Artist):
    """An image attached to an Axes, returned by imshow()."""

    zorder = 0

    def __init__(self, ax, data=None, extent=None, cmap=None, norm=None,
                 interpolation=None, **kwargs):
        super().__init__()
        self.axes = ax
        self._data = data
        self._extent = extent  # (x0, x1, y0, y1)
        self._cmap = cmap
        self._norm = norm
        self._interpolation = interpolation
        self._clim = None

        if kwargs:
            self.set(**kwargs)

    # --- data ---
    def get_array(self):
        """Return the image data."""
        return self._data

    def set_array(self, A):
        """Set the image data."""
        if hasattr(A, 'tolist'):
            self._data = A.tolist()
        else:
            self._data = A

    def set_data(self, A):
        """Set the image data (alias for set_array)."""
        self.set_array(A)

    def get_size(self):
        """Return (nrows, ncols) of the image data."""
        if self._data is None:
            return (0, 0)
        nrows = len(self._data)
        ncols = len(self._data[0]) if nrows > 0 else 0
        return (nrows, ncols)

    @property
    def shape(self):
        """Return the shape of the data."""
        return self.get_size()

    # --- extent ---
    def get_extent(self):
        """Return the image extent as (x0, x1, y0, y1)."""
        return self._extent

    def set_extent(self, extent):
        """Set the image extent as (x0, x1, y0, y1)."""
        self._extent = tuple(extent)

    # --- clim ---
    def set_clim(self, vmin=None, vmax=None):
        """Set the color limits."""
        if vmin is not None and vmax is None and hasattr(vmin, '__iter__'):
            vmin, vmax = vmin
        self._clim = (vmin, vmax)

    def get_clim(self):
        """Return the color limits."""
        if self._clim is not None:
            return self._clim
        # Auto from data
        if self._data is not None:
            flat = []
            for row in self._data:
                if hasattr(row, '__iter__') and not isinstance(row, str):
                    for val in row:
                        if isinstance(val, (int, float)):
                            flat.append(val)
                elif isinstance(row, (int, float)):
                    flat.append(row)
            if flat:
                return (min(flat), max(flat))
        return (0, 1)

    # --- cmap ---
    def get_cmap(self):
        return self._cmap

    def set_cmap(self, cmap):
        self._cmap = cmap

    # --- norm ---
    def get_norm(self):
        return self._norm

    def set_norm(self, norm):
        self._norm = norm

    # --- interpolation ---
    def get_interpolation(self):
        return self._interpolation

    def set_interpolation(self, interpolation):
        self._interpolation = interpolation

    def draw(self, renderer, layout):
        if not self.get_visible():
            return
        if self._data is None:
            return

        import matplotlib.cm as _cm

        rows = self._data
        if not rows:
            return

        first_row = rows[0] if rows else []
        if not first_row:
            return
        first_cell = first_row[0] if first_row else None
        is_scalar = first_cell is not None and not hasattr(first_cell, '__len__')

        if is_scalar:
            cmap_name = self._cmap if self._cmap else 'viridis'
            colormap = _cm.get_cmap(cmap_name)
            vmin, vmax = self.get_clim()
            rng = (vmax - vmin) if vmax != vmin else 1.0
            rgba_array = []
            for row in rows:
                rgba_row = []
                for v in row:
                    t = max(0.0, min(1.0, (float(v) - vmin) / rng))
                    rgba = colormap(t)
                    rgba_row.append((
                        int(rgba[0] * 255), int(rgba[1] * 255),
                        int(rgba[2] * 255), int(rgba[3] * 255)
                    ))
                rgba_array.append(rgba_row)
        else:
            rgba_array = []
            for row in rows:
                rgba_row = []
                for px in row:
                    px = list(px)
                    if len(px) == 3:
                        rgba_row.append((int(px[0]), int(px[1]), int(px[2]), 255))
                    else:
                        rgba_row.append((int(px[0]), int(px[1]), int(px[2]), int(px[3])))
                rgba_array.append(rgba_row)

        # Map image extent to pixel coordinates
        if self._extent is not None and layout is not None:
            x0, x1, y0, y1 = self._extent
            px0 = layout.sx(x0)
            px1 = layout.sx(x1)
            py0 = layout.sy(y0)
            py1 = layout.sy(y1)
            img_x = min(px0, px1)
            img_y = min(py0, py1)
            img_w = abs(px1 - px0)
            img_h = abs(py1 - py0)
        else:
            # Fill the plot area
            img_x = layout.plot_x if layout else 0
            img_y = layout.plot_y if layout else 0
            img_w = layout.plot_w if layout else renderer.width
            img_h = layout.plot_h if layout else renderer.height

        renderer.draw_image(img_x, img_y, img_w, img_h, rgba_array)
