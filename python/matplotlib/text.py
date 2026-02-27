"""matplotlib.text --- Text and Annotation artists."""

from matplotlib.artist import Artist
from matplotlib.patches import Patch


class Text(Artist):
    """An artist for rendering text."""

    zorder = 3

    def __init__(self, x=0, y=0, text='', **kwargs):
        super().__init__()
        self._x = x
        self._y = y
        self._text = str(text)

        # fontsize: accept 'fontsize' or 'size'
        self._fontsize = kwargs.get('fontsize', kwargs.get('size', 12.0))

        # fontweight: accept 'fontweight' or 'weight'
        self._fontweight = kwargs.get('fontweight', kwargs.get('weight', 'normal'))

        # horizontal alignment: accept 'ha' or 'horizontalalignment'
        self._ha = kwargs.get('ha', kwargs.get('horizontalalignment', 'left'))

        # vertical alignment: accept 'va' or 'verticalalignment'
        self._va = kwargs.get('va', kwargs.get('verticalalignment', 'baseline'))

    # --- text ---
    def get_text(self):
        return self._text

    def set_text(self, s):
        self._text = str(s)

    # --- fontsize ---
    def get_fontsize(self):
        return self._fontsize

    def set_fontsize(self, size):
        self._fontsize = size

    set_size = set_fontsize

    # --- fontweight ---
    def get_weight(self):
        return self._fontweight

    def set_weight(self, weight):
        self._fontweight = weight

    def get_fontweight(self):
        return self._fontweight

    def set_fontweight(self, weight):
        self._fontweight = weight

    # --- horizontal alignment ---
    def get_horizontalalignment(self):
        return self._ha

    def set_horizontalalignment(self, align):
        self._ha = align

    set_ha = set_horizontalalignment

    # --- vertical alignment ---
    def get_verticalalignment(self):
        return self._va

    def set_verticalalignment(self, align):
        self._va = align

    set_va = set_verticalalignment

    # --- position ---
    def get_position(self):
        return (self._x, self._y)

    def set_position(self, xy):
        self._x, self._y = xy


class Annotation(Text):
    """A Text with an optional arrow pointing to a data coordinate."""

    def __init__(self, text, xy, xytext=None, arrowprops=None, **kwargs):
        # xy is the point being annotated
        self.xy = tuple(xy)
        # xytext is where the text is placed; defaults to xy
        self.xytext = tuple(xytext) if xytext is not None else self.xy

        # Initialise Text at the xytext position
        super().__init__(x=self.xytext[0], y=self.xytext[1], text=text,
                         **kwargs)

        # arrow_patch: None when no arrowprops, otherwise a Patch placeholder
        if arrowprops is not None:
            self.arrow_patch = Patch()
        else:
            self.arrow_patch = None
