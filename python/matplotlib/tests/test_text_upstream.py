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
