"""Extended tests for matplotlib.text --- new fontfamily, fontstyle, repr, etc."""

import pytest

from matplotlib.text import Text, Annotation


class TestTextFontFamily:
    def test_default_fontfamily(self):
        t = Text(0, 0, 'hello')
        assert t.get_fontfamily() is None

    def test_set_fontfamily(self):
        t = Text(0, 0, 'hello')
        t.set_fontfamily('serif')
        assert t.get_fontfamily() == 'serif'

    def test_init_fontfamily(self):
        t = Text(0, 0, 'hello', fontfamily='monospace')
        assert t.get_fontfamily() == 'monospace'

    def test_init_family_alias(self):
        t = Text(0, 0, 'hello', family='sans-serif')
        assert t.get_fontfamily() == 'sans-serif'

    def test_get_family_alias(self):
        t = Text(0, 0, 'hello')
        t.set_fontfamily('cursive')
        assert t.get_family() == 'cursive'

    def test_set_family_alias(self):
        t = Text(0, 0, 'hello')
        t.set_family('fantasy')
        assert t.get_fontfamily() == 'fantasy'

    def test_fontname_init(self):
        t = Text(0, 0, 'hello', fontname='Arial')
        assert t.get_fontfamily() == 'Arial'

    def test_get_fontname(self):
        t = Text(0, 0, 'hello')
        t.set_fontfamily('Helvetica')
        assert t.get_fontname() == 'Helvetica'


class TestTextFontStyle:
    def test_default_fontstyle(self):
        t = Text(0, 0, 'hello')
        assert t.get_fontstyle() == 'normal'

    def test_set_fontstyle_italic(self):
        t = Text(0, 0, 'hello')
        t.set_fontstyle('italic')
        assert t.get_fontstyle() == 'italic'

    def test_set_fontstyle_oblique(self):
        t = Text(0, 0, 'hello')
        t.set_fontstyle('oblique')
        assert t.get_fontstyle() == 'oblique'

    def test_set_fontstyle_normal(self):
        t = Text(0, 0, 'hello')
        t.set_fontstyle('italic')
        t.set_fontstyle('normal')
        assert t.get_fontstyle() == 'normal'

    def test_set_fontstyle_invalid(self):
        t = Text(0, 0, 'hello')
        with pytest.raises(ValueError, match='fontstyle'):
            t.set_fontstyle('bold')

    def test_init_fontstyle(self):
        t = Text(0, 0, 'hello', fontstyle='italic')
        assert t.get_fontstyle() == 'italic'

    def test_init_style_alias(self):
        t = Text(0, 0, 'hello', style='oblique')
        assert t.get_fontstyle() == 'oblique'

    def test_get_style_alias(self):
        t = Text(0, 0, 'hello')
        t.set_fontstyle('italic')
        assert t.get_style() == 'italic'

    def test_set_style_alias(self):
        t = Text(0, 0, 'hello')
        t.set_style('italic')
        assert t.get_fontstyle() == 'italic'


class TestTextRepr:
    def test_basic_repr(self):
        t = Text(1, 2, 'hello')
        r = repr(t)
        assert 'Text' in r
        assert '1' in r
        assert '2' in r
        assert 'hello' in r

    def test_empty_text_repr(self):
        t = Text(0, 0, '')
        r = repr(t)
        assert 'Text' in r

    def test_repr_with_special_chars(self):
        t = Text(0, 0, "it's a 'test'")
        r = repr(t)
        assert 'Text' in r


class TestTextExtended:
    def test_get_size_alias(self):
        t = Text(0, 0, 'hello', fontsize=14)
        assert t.get_size() == 14

    def test_fontvariant_default(self):
        t = Text(0, 0, 'hello')
        assert t.get_fontvariant() == 'normal'

    def test_set_fontvariant(self):
        t = Text(0, 0, 'hello')
        t.set_fontvariant('small-caps')
        assert t.get_fontvariant() == 'small-caps'

    def test_fontstretch_default(self):
        t = Text(0, 0, 'hello')
        assert t.get_fontstretch() == 'normal'

    def test_set_fontstretch(self):
        t = Text(0, 0, 'hello')
        t.set_fontstretch('condensed')
        assert t.get_fontstretch() == 'condensed'

    def test_get_stretch_alias(self):
        t = Text(0, 0, 'hello')
        t.set_fontstretch('expanded')
        assert t.get_stretch() == 'expanded'

    def test_set_stretch_alias(self):
        t = Text(0, 0, 'hello')
        t.set_stretch('ultra-condensed')
        assert t.get_fontstretch() == 'ultra-condensed'

    def test_wrap_default(self):
        t = Text(0, 0, 'hello')
        assert t.get_wrap() is False

    def test_set_wrap(self):
        t = Text(0, 0, 'hello')
        t.set_wrap(True)
        assert t.get_wrap() is True

    def test_usetex_default(self):
        t = Text(0, 0, 'hello')
        assert t.get_usetex() is False

    def test_set_usetex(self):
        t = Text(0, 0, 'hello')
        t.set_usetex(True)
        assert t.get_usetex() is True

    def test_math_fontfamily_default(self):
        t = Text(0, 0, 'hello')
        assert t.get_math_fontfamily() == 'dejavusans'

    def test_set_math_fontfamily(self):
        t = Text(0, 0, 'hello')
        t.set_math_fontfamily('cm')
        assert t.get_math_fontfamily() == 'cm'


class TestTextSet:
    """Test the batch set() method with new properties."""

    def test_set_fontfamily(self):
        t = Text(0, 0, 'hello')
        t.set(fontfamily='serif')
        assert t.get_fontfamily() == 'serif'

    def test_set_fontstyle(self):
        t = Text(0, 0, 'hello')
        t.set(fontstyle='italic')
        assert t.get_fontstyle() == 'italic'

    def test_set_multiple(self):
        t = Text(0, 0, 'hello')
        t.set(fontfamily='serif', fontstyle='italic', fontsize=20)
        assert t.get_fontfamily() == 'serif'
        assert t.get_fontstyle() == 'italic'
        assert t.get_fontsize() == 20


class TestAnnotationExtended:
    def test_annotation_has_repr(self):
        ann = Annotation('note', xy=(1, 2))
        # Annotation inherits from Text, which has __repr__
        r = repr(ann)
        assert 'Text' in r

    def test_annotation_fontfamily(self):
        ann = Annotation('note', xy=(1, 2), fontfamily='monospace')
        assert ann.get_fontfamily() == 'monospace'

    def test_annotation_fontstyle(self):
        ann = Annotation('note', xy=(1, 2), fontstyle='italic')
        assert ann.get_fontstyle() == 'italic'


# ===================================================================
# Additional text parametric tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.text import Text


class TestTextParametric:
    """Parametric tests for Text properties."""

    @pytest.mark.parametrize('fontsize', [6, 8, 10, 12, 14, 16, 18, 24])
    def test_text_fontsize(self, fontsize):
        """Text fontsize can be set to various sizes."""
        t = Text(0, 0, 'test')
        t.set_fontsize(fontsize)
        assert t.get_fontsize() == fontsize

    @pytest.mark.parametrize('rotation', [0, 30, 45, 90, 180, 270])
    def test_text_rotation(self, rotation):
        """Text rotation is settable."""
        t = Text(0, 0, 'test')
        t.set_rotation(rotation)
        assert abs(t.get_rotation() - rotation) < 1e-10

    @pytest.mark.parametrize('ha', ['left', 'center', 'right'])
    def test_text_horizontalalignment(self, ha):
        """Text horizontalalignment is settable."""
        t = Text(0, 0, 'test')
        t.set_horizontalalignment(ha)
        assert t.get_horizontalalignment() == ha

    @pytest.mark.parametrize('va', ['top', 'center', 'bottom', 'baseline'])
    def test_text_verticalalignment(self, va):
        """Text verticalalignment is settable."""
        t = Text(0, 0, 'test')
        t.set_verticalalignment(va)
        assert t.get_verticalalignment() == va

    @pytest.mark.parametrize('alpha', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_text_alpha(self, alpha):
        """Text alpha is settable."""
        t = Text(0, 0, 'test')
        t.set_alpha(alpha)
        assert abs(t.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('zorder', [1, 2, 3, 5, 10])
    def test_text_zorder(self, zorder):
        """Text zorder is settable."""
        t = Text(0, 0, 'test')
        t.set_zorder(zorder)
        assert t.get_zorder() == zorder

    @pytest.mark.parametrize('text', ['hello', 'world', 'matplotlib', ''])
    def test_text_content(self, text):
        """Text stores string content."""
        t = Text(0, 0, text)
        assert t.get_text() == text

    @pytest.mark.parametrize('x,y', [(0, 0), (1, 2), (-3, 4), (0.5, 0.5)])
    def test_text_position(self, x, y):
        """Text position is stored correctly."""
        t = Text(x, y, 'test')
        pos = t.get_position()
        assert abs(pos[0] - x) < 1e-10
        assert abs(pos[1] - y) < 1e-10


class TestTextAxesParametric:
    """Parametric tests for ax.text behavior."""

    @pytest.mark.parametrize('text', ['label1', 'label2', 'My Label'])
    def test_ax_text_content(self, text):
        """ax.text stores text content."""
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, text)
        assert t.get_text() == text
        plt.close('all')

    @pytest.mark.parametrize('fontsize', [8, 10, 12, 14])
    def test_ax_text_fontsize(self, fontsize):
        """ax.text fontsize is set correctly."""
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'test', fontsize=fontsize)
        assert t.get_fontsize() == fontsize
        plt.close('all')

    @pytest.mark.parametrize('ha', ['left', 'center', 'right'])
    def test_ax_text_ha(self, ha):
        """ax.text horizontal alignment is stored."""
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'test', ha=ha)
        assert t.get_horizontalalignment() == ha
        plt.close('all')
