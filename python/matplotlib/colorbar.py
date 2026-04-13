"""Stub for matplotlib.colorbar for RustPython/WASM sandbox."""


class Colorbar:
    """Minimal Colorbar stub."""

    def __init__(self, ax, mappable=None, **kwargs):
        self.ax = ax
        self.mappable = mappable
        self._ticks = []

    def set_label(self, label, **kwargs):
        pass

    def set_ticks(self, ticks, **kwargs):
        self._ticks = list(ticks) if ticks is not None else []

    def get_ticks(self):
        return self._ticks

    def set_ticklabels(self, labels, **kwargs):
        pass

    def remove(self):
        pass


def make_axes(parents, **kwargs):
    return parents, {}


def make_axes_gridspec(parent, **kwargs):
    return parent, {}


# Alias for backwards compatibility
ColorbarBase = Colorbar
