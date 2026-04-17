"""
Upstream matplotlib tests imported from lib/matplotlib/tests/test_text.py.

These tests are copied (or minimally adapted) from the real matplotlib test
suite to validate compatibility of our Text implementation.
"""

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import matplotlib
from matplotlib.text import Text, Annotation


# ===================================================================
# Rotation (6 tests — direct imports)
# ===================================================================

def test_get_rotation_string():
    assert Text(rotation='horizontal').get_rotation() == 0.
    assert Text(rotation='vertical').get_rotation() == 90.


def test_get_rotation_float():
    for i in [15., 16.70, 77.4]:
        assert Text(rotation=i).get_rotation() == i


def test_get_rotation_int():
    for i in [67, 16, 41]:
        assert Text(rotation=i).get_rotation() == float(i)


def test_get_rotation_raises():
    with pytest.raises(ValueError):
        Text(rotation='hozirontal')


def test_get_rotation_none():
    assert Text(rotation=None).get_rotation() == 0.0


def test_get_rotation_mod360():
    for i, j in zip([360., 377., 720+177.2], [0., 17., 177.2]):
        assert_almost_equal(Text(rotation=i).get_rotation(), j)


# ===================================================================
# Antialiased (2 tests — direct import + adapted)
# ===================================================================

def test_get_set_antialiased():
    txt = Text(.5, .5, "foo\nbar")
    assert txt._antialiased == matplotlib.rcParams['text.antialiased']
    assert txt.get_antialiased() == matplotlib.rcParams['text.antialiased']

    txt.set_antialiased(True)
    assert txt._antialiased is True
    assert txt.get_antialiased() == txt._antialiased

    txt.set_antialiased(False)
    assert txt._antialiased is False
    assert txt.get_antialiased() == txt._antialiased


def test_annotation_antialiased():
    annot = Annotation("foo\nbar", (.5, .5), antialiased=True)
    assert annot._antialiased is True
    assert annot.get_antialiased() == annot._antialiased

    annot2 = Annotation("foo\nbar", (.5, .5), antialiased=False)
    assert annot2._antialiased is False
    assert annot2.get_antialiased() == annot2._antialiased

    annot3 = Annotation("foo\nbar", (.5, .5), antialiased=False)
    annot3.set_antialiased(True)
    assert annot3.get_antialiased() is True
    assert annot3._antialiased is True

    annot4 = Annotation("foo\nbar", (.5, .5))
    assert annot4._antialiased == matplotlib.rcParams['text.antialiased']


# ===================================================================
# Angle-based alignment helpers (2 tests — direct imports)
# ===================================================================

def test_ha_for_angle():
    text_instance = Text()
    angles = np.arange(0, 360.1, 0.1)
    for angle in angles:
        alignment = text_instance._ha_for_angle(angle)
        assert alignment in ['center', 'left', 'right']


def test_va_for_angle():
    text_instance = Text()
    angles = np.arange(0, 360.1, 0.1)
    for angle in angles:
        alignment = text_instance._va_for_angle(angle)
        assert alignment in ['center', 'top', 'baseline', 'bottom']


# ===========================================================================
# Newly ported upstream tests (2026-03-19)
# Source: https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/tests/test_text.py
# ===========================================================================

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# test_invalid_color (upstream)
# ---------------------------------------------------------------------------
def test_invalid_color():
    """Upstream: invalid color raises ValueError."""
    with pytest.raises(ValueError):
        plt.text(0.5, 0.5, "foo", color="foobar")


# ---------------------------------------------------------------------------
# test_text_set_position (upstream-inspired)
# ---------------------------------------------------------------------------
def test_text_set_position():
    """Test Text.set_position / get_position round-trip."""
    t = Text(1, 2, "hello")
    assert t.get_position() == (1, 2)
    t.set_position((3, 4))
    assert t.get_position() == (3, 4)


# ---------------------------------------------------------------------------
# test_text_fontsize (upstream-inspired)
# ---------------------------------------------------------------------------
def test_text_fontsize():
    """Test fontsize get/set."""
    t = Text(0, 0, "hello", fontsize=20)
    assert t.get_fontsize() == 20
    t.set_fontsize(14)
    assert t.get_fontsize() == 14


# ---------------------------------------------------------------------------
# test_text_fontweight (upstream-inspired)
# ---------------------------------------------------------------------------
def test_text_fontweight():
    """Test fontweight get/set."""
    t = Text(0, 0, "hello", fontweight='bold')
    assert t.get_fontweight() == 'bold'
    t.set_fontweight('normal')
    assert t.get_fontweight() == 'normal'


# ---------------------------------------------------------------------------
# test_text_alignment (upstream-inspired)
# ---------------------------------------------------------------------------
def test_text_alignment():
    """Test horizontal and vertical alignment."""
    t = Text(0, 0, "hello", ha='center', va='top')
    assert t.get_horizontalalignment() == 'center'
    assert t.get_verticalalignment() == 'top'
    t.set_ha('right')
    t.set_va('bottom')
    assert t.get_horizontalalignment() == 'right'
    assert t.get_verticalalignment() == 'bottom'


# ---------------------------------------------------------------------------
# test_annotation_basic (upstream-inspired)
# ---------------------------------------------------------------------------
def test_annotation_basic():
    """Test Annotation creation and properties."""
    from matplotlib.text import Annotation
    ann = Annotation("test", xy=(1, 2), xytext=(3, 4))
    assert ann.xy == (1, 2)
    assert ann.xytext == (3, 4)
    assert ann.get_text() == "test"


# ---------------------------------------------------------------------------
# test_annotation_default_xytext (upstream-inspired)
# ---------------------------------------------------------------------------
def test_annotation_default_xytext():
    """Annotation defaults xytext to xy if not given."""
    from matplotlib.text import Annotation
    ann = Annotation("test", xy=(5, 6))
    assert ann.xytext == (5, 6)


# ===========================================================================
# Newly ported upstream tests (2026-03-19, batch 2)
# ===========================================================================


def test_text_constructor_kwargs():
    """Text constructor accepts common keyword arguments."""
    from matplotlib.text import Text
    t = Text(1.0, 2.0, 'hello', fontsize=14, fontweight='bold',
             ha='center', va='top', rotation=45, color='red')
    assert t.get_text() == 'hello'
    assert t.get_fontsize() == 14
    assert t.get_fontweight() == 'bold'
    assert t.get_horizontalalignment() == 'center'
    assert t.get_verticalalignment() == 'top'
    assert t.get_rotation() == 45.0


def test_text_set_text():
    """Text.set_text updates the text string."""
    from matplotlib.text import Text
    t = Text(0, 0, 'original')
    assert t.get_text() == 'original'
    t.set_text('updated')
    assert t.get_text() == 'updated'


def test_text_position_roundtrip():
    """Text position can be set and retrieved."""
    from matplotlib.text import Text
    t = Text(1, 2, 'pos test')
    assert t.get_position() == (1, 2)
    t.set_position((3, 4))
    assert t.get_position() == (3, 4)


def test_text_visibility():
    """Text visibility can be toggled."""
    from matplotlib.text import Text
    t = Text(0, 0, 'vis test')
    assert t.get_visible() is True
    t.set_visible(False)
    assert t.get_visible() is False


def test_text_zorder():
    """Text has default zorder of 3."""
    from matplotlib.text import Text
    t = Text(0, 0, 'z test')
    assert t.get_zorder() == 3


def test_annotation_with_arrow():
    """Annotation with arrowprops creates an arrow_patch."""
    from matplotlib.text import Annotation
    ann = Annotation("arr", xy=(0, 0), xytext=(1, 1), arrowprops={})
    assert ann.arrow_patch is not None


def test_annotation_without_arrow():
    """Annotation without arrowprops has arrow_patch=None."""
    from matplotlib.text import Annotation
    ann = Annotation("no arr", xy=(0, 0), xytext=(1, 1))
    assert ann.arrow_patch is None


def test_text_axes_integration():
    """Text added via ax.text is in ax.texts."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, 'integrated')
    assert t in ax.texts
    assert t.get_text() == 'integrated'


def test_annotate_axes_integration():
    """Annotation added via ax.annotate is in ax.texts."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ann = ax.annotate('note', xy=(0, 0), xytext=(1, 1))
    assert ann in ax.texts
    assert ann.get_text() == 'note'


# ===================================================================
# Text basic properties
# ===================================================================

def test_text_default_position():
    """Text default position is (0, 0)."""
    t = Text()
    assert t.get_position() == (0, 0)


def test_text_set_position():
    """Text.set_position changes position."""
    t = Text(0, 0, 'hello')
    t.set_position((5, 10))
    assert t.get_position() == (5, 10)


def test_text_get_text():
    """Text.get_text returns the string."""
    t = Text(0, 0, 'hello world')
    assert t.get_text() == 'hello world'


def test_text_set_text():
    """Text.set_text changes the string."""
    t = Text(0, 0, 'old')
    t.set_text('new')
    assert t.get_text() == 'new'


def test_text_fontsize():
    """Text fontsize property."""
    t = Text(0, 0, 'sized', fontsize=14)
    assert t.get_fontsize() == 14


def test_text_set_fontsize():
    """Text.set_fontsize changes the size."""
    t = Text()
    t.set_fontsize(20)
    assert t.get_fontsize() == 20


def test_text_ha():
    """Text horizontal alignment."""
    t = Text(0, 0, 'aligned', ha='center')
    assert t.get_ha() == 'center'


def test_text_va():
    """Text vertical alignment."""
    t = Text(0, 0, 'aligned', va='top')
    assert t.get_va() == 'top'


def test_text_set_ha():
    """Text.set_ha changes alignment."""
    t = Text()
    t.set_ha('right')
    assert t.get_ha() == 'right'


def test_text_set_va():
    """Text.set_va changes alignment."""
    t = Text()
    t.set_va('bottom')
    assert t.get_va() == 'bottom'


def test_text_visible():
    """Text visibility."""
    t = Text()
    assert t.get_visible() is True
    t.set_visible(False)
    assert t.get_visible() is False


def test_text_alpha():
    """Text alpha."""
    t = Text()
    assert t.get_alpha() is None
    t.set_alpha(0.5)
    assert t.get_alpha() == 0.5


def test_text_label():
    """Text label."""
    t = Text()
    t.set_label('my text')
    assert t.get_label() == 'my text'


def test_text_color():
    """Text color."""
    t = Text(0, 0, 'colored', color='red')
    assert t.get_color() == 'red'


def test_text_set_color():
    """Text.set_color changes color."""
    t = Text()
    t.set_color('blue')
    assert t.get_color() == 'blue'


# ===================================================================
# Annotation properties
# ===================================================================

def test_annotation_xy():
    """Annotation stores xy."""
    ann = Annotation('test', xy=(1, 2))
    assert ann.xy == (1, 2)


def test_annotation_xytext():
    """Annotation stores xytext."""
    ann = Annotation('test', xy=(1, 2), xytext=(3, 4))
    assert ann.xyann == (3, 4)


def test_annotation_is_text():
    """Annotation is a subclass of Text."""
    ann = Annotation('test', xy=(0, 0))
    assert isinstance(ann, Text)


def test_annotation_get_text():
    """Annotation.get_text works."""
    ann = Annotation('hello', xy=(0, 0))
    assert ann.get_text() == 'hello'


def test_annotation_set_text():
    """Annotation.set_text changes text."""
    ann = Annotation('old', xy=(0, 0))
    ann.set_text('new')
    assert ann.get_text() == 'new'


# ===================================================================
# Text extended property tests
# ===================================================================

class TestTextExtendedProperties:
    def test_fontstyle_default(self):
        t = Text()
        assert t.get_fontstyle() == 'normal'

    def test_set_fontstyle(self):
        t = Text()
        t.set_fontstyle('italic')
        assert t.get_fontstyle() == 'italic'

    def test_fontfamily_default(self):
        # OG get_fontfamily() returns a list like ['sans-serif']
        t = Text()
        family = t.get_fontfamily()
        assert isinstance(family, list)
        assert len(family) > 0

    def test_set_fontfamily(self):
        t = Text()
        t.set_fontfamily('monospace')
        # OG returns a list, not a bare string
        family = t.get_fontfamily()
        assert isinstance(family, list)
        assert 'monospace' in family

    def test_fontname_default(self):
        # OG get_fontname() returns a string (may be empty if no font finalized)
        t = Text()
        name = t.get_fontname()
        assert isinstance(name, str)

    def test_fontvariant_default(self):
        t = Text()
        assert t.get_fontvariant() == 'normal'

    def test_set_fontvariant(self):
        t = Text()
        t.set_fontvariant('small-caps')
        assert t.get_fontvariant() == 'small-caps'

    def test_fontstretch_default(self):
        t = Text()
        stretch = t.get_fontstretch()
        assert stretch is not None

    def test_set_fontstretch(self):
        t = Text()
        t.set_fontstretch('condensed')
        assert t.get_fontstretch() == 'condensed'

    def test_stretch_alias(self):
        t = Text()
        t.set_stretch('expanded')
        assert t.get_stretch() == 'expanded'

    def test_wrap_default_false(self):
        t = Text()
        assert t.get_wrap() is False

    def test_set_wrap(self):
        t = Text()
        t.set_wrap(True)
        assert t.get_wrap() is True

    def test_usetex_default_false(self):
        t = Text()
        assert t.get_usetex() is False  # default is False

    def test_set_usetex(self):
        t = Text()
        t.set_usetex(True)
        assert t.get_usetex() is True

    def test_math_fontfamily_default(self):
        t = Text()
        mff = t.get_math_fontfamily()
        assert mff is not None

    def test_set_math_fontfamily(self):
        t = Text()
        t.set_math_fontfamily('stix')
        assert t.get_math_fontfamily() == 'stix'

    def test_position_default(self):
        t = Text()
        pos = t.get_position()
        assert pos == (0, 0)

    def test_set_position(self):
        t = Text()
        t.set_position((3.0, 4.0))
        assert t.get_position() == (3.0, 4.0)

    def test_weight_alias(self):
        t = Text()
        t.set_weight('bold')
        assert t.get_weight() == 'bold'
        assert t.get_fontweight() == 'bold'

    def test_repr(self):
        t = Text(1, 2, 'hello')
        r = repr(t)
        assert 'hello' in r or 'Text' in r


# ===================================================================
# Text integration with axes
# ===================================================================

class TestTextAxesIntegration:
    def test_ax_text_default_color(self):
        """ax.text with explicit color stores it."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'test', color='black')
        color = t.get_color()
        assert color == 'black'
        plt.close('all')

    def test_ax_text_with_rotation(self):
        """ax.text with rotation stores the rotation."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'rot', rotation=45)
        assert t.get_rotation() == 45.0
        plt.close('all')

    def test_ax_text_ha_center(self):
        """ax.text with ha='center' stores alignment."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'centered', ha='center')
        assert t.get_horizontalalignment() == 'center'
        plt.close('all')

    def test_ax_text_va_top(self):
        """ax.text with va='top' stores alignment."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'top', va='top')
        assert t.get_verticalalignment() == 'top'
        plt.close('all')

    def test_ax_text_fontsize_constructor(self):
        """ax.text with fontsize stores it."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'sized', fontsize=18)
        assert t.get_fontsize() == 18
        plt.close('all')

    def test_ax_text_visible_toggle(self):
        """ax.text visible can be toggled after creation."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'hidden')
        t.set_visible(False)
        assert t.get_visible() is False
        plt.close('all')

    def test_multiple_ax_texts(self):
        """Multiple ax.text calls add to texts list."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t1 = ax.text(0.1, 0.1, 'a')
        t2 = ax.text(0.5, 0.5, 'b')
        t3 = ax.text(0.9, 0.9, 'c')
        assert t1 in ax.texts
        assert t2 in ax.texts
        assert t3 in ax.texts
        plt.close('all')


# ===================================================================
# Text rotation properties (extended)
# ===================================================================

class TestTextRotationExtended:
    def test_rotation_45(self):
        """Text rotation 45 is stored exactly."""
        t = Text(0, 0, 'r', rotation=45)
        assert t.get_rotation() == 45.0

    def test_rotation_90(self):
        """Text rotation 90 is stored exactly."""
        t = Text(0, 0, 'r', rotation=90)
        assert t.get_rotation() == 90.0

    def test_rotation_180(self):
        """Text rotation 180 is stored exactly."""
        t = Text(0, 0, 'r', rotation=180)
        assert t.get_rotation() == 180.0

    def test_rotation_270(self):
        """Text rotation 270 mod 360 is 270."""
        t = Text(0, 0, 'r', rotation=270)
        assert t.get_rotation() == 270.0

    def test_set_rotation_updates(self):
        """set_rotation updates get_rotation."""
        t = Text(0, 0, 'r', rotation=0)
        t.set_rotation(30)
        assert t.get_rotation() == 30.0

    def test_ha_for_angle_0(self):
        """_ha_for_angle(0) returns a valid alignment."""
        t = Text()
        ha = t._ha_for_angle(0)
        assert ha in ('center', 'left', 'right')

    def test_ha_for_angle_90(self):
        """_ha_for_angle(90) returns a valid alignment."""
        t = Text()
        ha = t._ha_for_angle(90)
        assert ha in ('center', 'left', 'right')

    def test_va_for_angle_0(self):
        """_va_for_angle(0) is 'center'."""
        t = Text()
        assert t._va_for_angle(0) == 'center'

    def test_ha_for_angle_45_is_valid(self):
        """_ha_for_angle(45) is a valid alignment."""
        t = Text()
        ha = t._ha_for_angle(45)
        assert ha in ('center', 'left', 'right')

    def test_va_for_angle_45_is_valid(self):
        """_va_for_angle(45) is a valid alignment."""
        t = Text()
        va = t._va_for_angle(45)
        assert va in ('center', 'top', 'baseline')


# ===================================================================
# Annotation extended
# ===================================================================

class TestAnnotationExtendedProperties:
    def test_annotation_arrowprops_stores_arrow_patch(self):
        """Annotation with arrowprops creates arrow_patch."""
        from matplotlib.text import Annotation
        ann = Annotation('note', xy=(0, 0), xytext=(1, 1),
                         arrowprops={'arrowstyle': '->'})
        assert ann.arrow_patch is not None

    def test_annotation_no_arrowprops_null_arrow_patch(self):
        """Annotation without arrowprops has arrow_patch=None."""
        from matplotlib.text import Annotation
        ann = Annotation('note', xy=(0, 0))
        assert ann.arrow_patch is None

    def test_annotation_set_position(self):
        """Annotation.set_position changes position."""
        from matplotlib.text import Annotation
        ann = Annotation('note', xy=(1, 2))
        ann.set_position((5, 6))
        pos = ann.get_position()
        assert pos == (5, 6)

    def test_annotation_rotation(self):
        """Annotation supports rotation."""
        from matplotlib.text import Annotation
        ann = Annotation('note', xy=(0, 0), rotation=30)
        assert ann.get_rotation() == 30.0

    def test_annotation_fontsize_constructor(self):
        """Annotation fontsize from constructor."""
        from matplotlib.text import Annotation
        ann = Annotation('note', xy=(0, 0), fontsize=16)
        assert ann.get_fontsize() == 16

    def test_annotation_is_subclass_text(self):
        """Annotation is a subclass of Text."""
        from matplotlib.text import Annotation, Text
        ann = Annotation('test', xy=(0, 0))
        assert isinstance(ann, Text)

    def test_annotation_zorder_default(self):
        """Annotation zorder is 3 (same as Text)."""
        from matplotlib.text import Annotation
        ann = Annotation('test', xy=(0, 0))
        assert ann.get_zorder() == 3


# ===================================================================
# Extended parametric tests for text (upstream-style)
# ===================================================================


class TestTextProperties:
    """Tests for Text property get/set."""

    def test_text_fontsize(self):
        from matplotlib.text import Text
        t = Text(0, 0, 'hello')
        t.set_fontsize(16)
        assert t.get_fontsize() == 16

    def test_text_color(self):
        from matplotlib.text import Text
        t = Text(0, 0, 'hello')
        t.set_color('blue')
        assert t.get_color() == 'blue'

    def test_text_alpha(self):
        from matplotlib.text import Text
        t = Text(0, 0, 'hello')
        t.set_alpha(0.7)
        assert abs(t.get_alpha() - 0.7) < 1e-10

    def test_text_visible(self):
        from matplotlib.text import Text
        t = Text(0, 0, 'hello')
        t.set_visible(False)
        assert not t.get_visible()

    def test_text_ha_va(self):
        from matplotlib.text import Text
        t = Text(0, 0, 'hello')
        t.set_ha('right')
        t.set_va('top')
        assert t.get_ha() == 'right'
        assert t.get_va() == 'top'

    @pytest.mark.parametrize('ha', ['left', 'center', 'right'])
    def test_text_ha_parametric(self, ha):
        from matplotlib.text import Text
        t = Text(0, 0, 'test')
        t.set_ha(ha)
        assert t.get_ha() == ha

    @pytest.mark.parametrize('va', ['top', 'center', 'bottom', 'baseline'])
    def test_text_va_parametric(self, va):
        from matplotlib.text import Text
        t = Text(0, 0, 'test')
        t.set_va(va)
        assert t.get_va() == va


class TestTextInAxes:
    """Tests for text within an axes context."""

    def test_ax_text_in_texts(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'hello')
        assert t in ax.texts
        plt.close('all')

    def test_ax_text_content_in_svg(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'UNIQUE_SVG_TEXT_ABC')
        svg = fig.to_svg()
        assert 'UNIQUE_SVG_TEXT_ABC' in svg
        plt.close('all')

    def test_ax_title_in_svg(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title('MY_TITLE_XYZ')
        svg = fig.to_svg()
        assert 'MY_TITLE_XYZ' in svg
        plt.close('all')

    def test_multiple_texts_in_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(5):
            ax.text(i * 0.2, 0.5, f'text{i}')
        assert len(ax.texts) == 5
        plt.close('all')

    @pytest.mark.parametrize('fontsize', [8, 10, 12, 14])
    def test_text_fontsize_in_axes(self, fontsize):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'test', fontsize=fontsize)
        assert t.get_fontsize() == fontsize
        plt.close('all')
