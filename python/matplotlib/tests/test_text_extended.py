"""Extended tests for matplotlib.text --- new fontfamily, fontstyle, repr, etc."""

import pytest

import matplotlib
from matplotlib.text import Text, Annotation


class TestTextFontFamily:
    def test_default_fontfamily(self):
        # OG: get_fontfamily() returns a list (e.g. ['sans-serif'])
        t = Text(0, 0, 'hello')
        fam = t.get_fontfamily()
        assert fam is not None
        assert len(fam) > 0  # OG default is ['sans-serif']

    def test_set_fontfamily(self):
        t = Text(0, 0, 'hello')
        t.set_fontfamily('serif')
        assert 'serif' in t.get_fontfamily()

    def test_init_fontfamily(self):
        t = Text(0, 0, 'hello', fontfamily='monospace')
        assert 'monospace' in t.get_fontfamily()

    def test_init_family_alias(self):
        t = Text(0, 0, 'hello', family='sans-serif')
        assert 'sans-serif' in t.get_fontfamily()

    def test_get_family_alias(self):
        t = Text(0, 0, 'hello')
        t.set_fontfamily('cursive')
        assert 'cursive' in t.get_family()

    def test_set_family_alias(self):
        t = Text(0, 0, 'hello')
        t.set_family('fantasy')
        assert 'fantasy' in t.get_fontfamily()

    def test_fontname_init(self):
        t = Text(0, 0, 'hello', fontname='Arial')
        assert 'Arial' in t.get_fontfamily()

    def test_get_fontname(self):
        t = Text(0, 0, 'hello')
        t.set_fontfamily('Helvetica')
        assert isinstance(t.get_fontname(), str)
        assert t.get_fontname()


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
        # OG error message says 'style', not 'fontstyle'
        with pytest.raises(ValueError, match='style'):
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
        assert t.get_math_fontfamily() == matplotlib.rcParams['mathtext.fontset']

    def test_set_math_fontfamily(self):
        t = Text(0, 0, 'hello')
        t.set_math_fontfamily('cm')
        assert t.get_math_fontfamily() == 'cm'


class TestTextSet:
    """Test the batch set() method with new properties."""

    def test_set_fontfamily(self):
        t = Text(0, 0, 'hello')
        t.set(fontfamily='serif')
        assert 'serif' in t.get_fontfamily()

    def test_set_fontstyle(self):
        t = Text(0, 0, 'hello')
        t.set(fontstyle='italic')
        assert t.get_fontstyle() == 'italic'

    def test_set_multiple(self):
        t = Text(0, 0, 'hello')
        t.set(fontfamily='serif', fontstyle='italic', fontsize=20)
        assert 'serif' in t.get_fontfamily()
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
        assert 'monospace' in ann.get_fontfamily()

    def test_annotation_fontstyle(self):
        ann = Annotation('note', xy=(1, 2), fontstyle='italic')
        assert ann.get_fontstyle() == 'italic'


# ===================================================================
# Additional text parametric tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.text import Text
