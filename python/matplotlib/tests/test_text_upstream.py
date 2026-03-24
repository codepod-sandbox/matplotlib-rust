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
        assert alignment in ['center', 'top', 'baseline']


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
        t = Text()
        family = t.get_fontfamily()
        assert family is None  # default is None (no family specified)

    def test_set_fontfamily(self):
        t = Text()
        t.set_fontfamily('monospace')
        assert t.get_fontfamily() == 'monospace'

    def test_fontname_default(self):
        t = Text()
        name = t.get_fontname()
        assert name is None  # default is None

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
