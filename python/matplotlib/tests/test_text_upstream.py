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
