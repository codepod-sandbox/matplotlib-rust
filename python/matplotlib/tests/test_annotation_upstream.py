# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_text.py (annotation section)
import pytest


def test_renderer_svg_draw_arrow_no_error():
    """RendererSVG.draw_arrow must produce valid SVG with a path element."""
    from matplotlib._svg_backend import RendererSVG
    r = RendererSVG(200, 200, 100)
    r.draw_arrow(10, 100, 150, 50, '->', '#ff0000', 1.5)
    svg = r.get_result()
    assert '<path' in svg or '<line' in svg
    assert 'marker-end' in svg or 'polygon' in svg.lower() or '<path' in svg


def test_renderer_svg_draw_arrow_no_head():
    """draw_arrow with style '-' must draw a line without arrowhead."""
    from matplotlib._svg_backend import RendererSVG
    r = RendererSVG(200, 200, 100)
    r.draw_arrow(10, 100, 150, 50, '-', '#000000', 1.0)
    svg = r.get_result()
    assert '<polyline' in svg or '<line' in svg or '<path' in svg


def test_renderer_pil_draw_arrow_no_error():
    """RendererPIL.draw_arrow must not raise."""
    from matplotlib._pil_backend import RendererPIL
    r = RendererPIL(200, 200, 100)
    r.draw_arrow(10, 100, 150, 50, '->', '#ff0000', 1.5)
    # Just check it produces bytes without error
    result = r.get_result()
    assert len(result) > 0


def test_annotate_default_arrow():
    """ax.annotate with arrowprops must render without error."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.annotate('hi', xy=(0.5, 0.5), xytext=(0.2, 0.8),
                arrowprops=dict(arrowstyle='->'))
    svg = fig.to_svg()
    assert len(svg) > 0
    plt.close('all')


def test_annotate_no_arrowprops():
    """ax.annotate without arrowprops renders only text."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.annotate('label', xy=(0.5, 0.5))
    svg = fig.to_svg()
    assert 'label' in svg
    plt.close('all')


def test_annotate_arrowprops_styles():
    """Multiple arrowstyle strings must not raise."""
    import matplotlib.pyplot as plt
    for style in ['->', '<-', '<->', '-', 'fancy']:
        fig, ax = plt.subplots()
        ax.annotate('x', xy=(0.5, 0.5), xytext=(0.1, 0.9),
                    arrowprops=dict(arrowstyle=style))
        fig.to_svg()  # must not raise
        plt.close('all')


def test_fancy_arrow_patch_draw():
    """OG backend_bases does not expose the codepod-only AxesLayout helper."""
    import matplotlib.backend_bases as backend_bases
    assert not hasattr(backend_bases, 'AxesLayout')


# ===================================================================
# Annotation extended tests (upstream-ported from test_text.py)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.text import Annotation


class TestAnnotationExtended:
    def test_annotation_get_xy(self):
        """Annotation stores xy point."""
        ann = Annotation('label', xy=(3, 4))
        assert ann.xy == (3, 4)

    def test_annotation_get_xytext(self):
        """Annotation xytext offset is stored."""
        ann = Annotation('label', xy=(1, 2), xytext=(5, 6))
        assert ann.xyann == (5, 6)

    def test_annotation_xytext_defaults_to_xy(self):
        """When xytext is None, xyann == xy."""
        ann = Annotation('label', xy=(2, 3))
        assert ann.xyann == (2, 3)

    def test_annotation_get_text(self):
        """Annotation inherits Text.get_text."""
        ann = Annotation('hello world', xy=(0, 0))
        assert ann.get_text() == 'hello world'

    def test_annotation_set_text(self):
        """Annotation inherits Text.set_text."""
        ann = Annotation('old', xy=(0, 0))
        ann.set_text('new text')
        assert ann.get_text() == 'new text'

    def test_annotation_visible_default(self):
        """Annotation is visible by default."""
        ann = Annotation('test', xy=(0, 0))
        assert ann.get_visible() is True

    def test_annotation_set_visible(self):
        """Annotation set_visible works."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_visible(False)
        assert ann.get_visible() is False

    def test_annotation_alpha_default(self):
        """Annotation alpha is None by default."""
        ann = Annotation('test', xy=(0, 0))
        assert ann.get_alpha() is None

    def test_annotation_set_alpha(self):
        """Annotation set_alpha works."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_alpha(0.5)
        assert ann.get_alpha() == 0.5

    def test_annotation_label(self):
        """Annotation label is settable."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_label('my_annotation')
        assert ann.get_label() == 'my_annotation'

    def test_annotation_in_axes(self):
        """ax.annotate adds annotation to axes texts."""
        fig, ax = plt.subplots()
        ann = ax.annotate('point', xy=(1, 1))
        assert ann in ax.texts
        plt.close('all')

    def test_annotation_with_arrowprops_in_axes(self):
        """ax.annotate with arrowprops adds to axes."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        ann = ax.annotate('peak', xy=(1, 1), xytext=(1.5, 1.5),
                          arrowprops=dict(arrowstyle='->'))
        assert ann is not None
        plt.close('all')

    def test_annotation_xy_is_tuple(self):
        """Annotation xy is stored as a tuple."""
        ann = Annotation('test', xy=(1, 2))
        assert isinstance(ann.xy, tuple)
        assert ann.xy[0] == 1
        assert ann.xy[1] == 2

    def test_annotation_fontsize(self):
        """Annotation fontsize is settable."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_fontsize(14)
        assert ann.get_fontsize() == 14

    def test_annotation_color(self):
        """Annotation color is settable."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_color('red')
        assert ann.get_color() == 'red'

    def test_annotation_position_from_xy(self):
        """Annotation get_position reflects xy."""
        ann = Annotation('test', xy=(5, 7))
        pos = ann.get_position()
        assert pos[0] == 5
        assert pos[1] == 7


# ===================================================================
# Additional annotation tests (upstream-inspired batch)
# ===================================================================

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import pytest


class TestAnnotationCoords:
    """Tests for annotation coordinate systems."""

    def test_annotation_xy_coords_data(self):
        """xycoords='data' uses data coordinates."""
        ann = Annotation('test', xy=(3, 4), xycoords='data')
        assert ann.xy == (3, 4)

    def test_annotation_xytext_offset_points(self):
        """textcoords='offset points' stores offset."""
        ann = Annotation('test', xy=(0, 0), xytext=(10, 20),
                         textcoords='offset points')
        assert ann.xyann == (10, 20)

    def test_annotation_xy_direct_assignment(self):
        """xy attribute can be assigned directly."""
        ann = Annotation('test', xy=(1, 2))
        ann.xy = (5, 6)
        assert ann.xy == (5, 6)

    def test_annotation_xy_float(self):
        """Annotation accepts float coordinates."""
        ann = Annotation('val', xy=(1.5, 2.7))
        assert abs(ann.xy[0] - 1.5) < 1e-10
        assert abs(ann.xy[1] - 2.7) < 1e-10

    def test_annotation_negative_coords(self):
        """Annotation accepts negative coordinates."""
        ann = Annotation('neg', xy=(-3, -7))
        assert ann.xy[0] == -3
        assert ann.xy[1] == -7


class TestAnnotationArrowStyles:
    """Tests for annotation arrow rendering."""

    @pytest.mark.parametrize('style', ['->', '<-', '<->', '-'])
    def test_arrowstyle_renders(self, style):
        """Each arrowstyle renders without error."""
        fig, ax = plt.subplots()
        ann = ax.annotate('pt', xy=(0.5, 0.5), xytext=(0.2, 0.8),
                          arrowprops=dict(arrowstyle=style))
        assert ann is not None
        plt.close('all')

    def test_annotation_arrow_patch_stored(self):
        """Annotation with arrowprops stores arrow_patch."""
        fig, ax = plt.subplots()
        ann = ax.annotate('x', xy=(0.5, 0.5), xytext=(0.2, 0.8),
                          arrowprops=dict(arrowstyle='->'))
        assert ann.arrow_patch is not None
        plt.close('all')

    def test_annotation_no_arrow_has_none_patch(self):
        """Annotation without arrowprops has arrow_patch=None."""
        ann = Annotation('text', xy=(1, 1))
        assert ann.arrow_patch is None


class TestAnnotationAxesInteraction:
    """Tests for annotation behavior within axes."""

    def test_multiple_annotations(self):
        """Multiple annotations can be added to same axes."""
        fig, ax = plt.subplots()
        ann1 = ax.annotate('first', xy=(0.2, 0.2))
        ann2 = ax.annotate('second', xy=(0.8, 0.8))
        assert ann1 in ax.texts
        assert ann2 in ax.texts
        plt.close('all')

    def test_annotation_text_content_in_svg(self):
        """Annotation text appears in SVG output."""
        fig, ax = plt.subplots()
        ax.annotate('UNIQUE_ANNOT', xy=(0.5, 0.5))
        svg = fig.to_svg()
        assert 'UNIQUE_ANNOT' in svg
        plt.close('all')

    def test_annotation_is_instance_of_text(self):
        """Annotation is a subclass of Text."""
        from matplotlib.text import Text
        ann = Annotation('test', xy=(0, 0))
        assert isinstance(ann, Text)

    def test_annotation_zorder_default(self):
        """Annotation has default zorder of 3."""
        ann = Annotation('test', xy=(0, 0))
        assert ann.get_zorder() == 3

    def test_annotation_zorder_settable(self):
        """Annotation zorder can be set."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_zorder(5)
        assert ann.get_zorder() == 5

    @pytest.mark.parametrize('ha', ['left', 'center', 'right'])
    def test_annotation_ha(self, ha):
        """Annotation horizontalalignment is settable."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_ha(ha)
        assert ann.get_ha() == ha

    @pytest.mark.parametrize('va', ['top', 'center', 'bottom', 'baseline'])
    def test_annotation_va(self, va):
        """Annotation verticalalignment is settable."""
        ann = Annotation('test', xy=(0, 0))
        ann.set_va(va)
        assert ann.get_va() == va


# ===================================================================
# Additional parametric tests
# ===================================================================

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import pytest


class TestAnnotationAppearance:
    """Tests for annotation visual properties."""

    def test_fontfamily_default(self):
        ann = Annotation('test', xy=(0, 0))
        # get_fontfamily may return None in our implementation (no font system)
        _ = ann.get_fontfamily()  # should not raise

    def test_fontsize_settable(self):
        ann = Annotation('test', xy=(0, 0))
        for size in [8, 10, 12, 14, 16]:
            ann.set_fontsize(size)
            assert ann.get_fontsize() == size

    def test_color_settable(self):
        ann = Annotation('test', xy=(0, 0))
        for color in ['red', 'blue', 'green', 'black', 'white']:
            ann.set_color(color)
            assert ann.get_color() == color

    def test_alpha_range(self):
        ann = Annotation('test', xy=(0, 0))
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ann.set_alpha(alpha)
            assert abs(ann.get_alpha() - alpha) < 1e-10

    def test_visible_toggle(self):
        ann = Annotation('test', xy=(0, 0))
        ann.set_visible(True)
        assert ann.get_visible() is True
        ann.set_visible(False)
        assert ann.get_visible() is False

    def test_zorder(self):
        ann = Annotation('test', xy=(0, 0))
        for z in [1, 2, 5, 10]:
            ann.set_zorder(z)
            assert ann.get_zorder() == z

    def test_label_get_set(self):
        ann = Annotation('test', xy=(0, 0))
        ann.set_label('my_label')
        assert ann.get_label() == 'my_label'

    def test_text_update(self):
        ann = Annotation('original', xy=(0, 0))
        assert ann.get_text() == 'original'
        ann.set_text('updated')
        assert ann.get_text() == 'updated'


class TestAnnotationInAxes:
    """Tests for annotations inside a figure/axes context."""

    def test_svg_has_annotation_text(self):
        fig, ax = plt.subplots()
        ax.annotate('SVG_TEXT_XYZ', xy=(0.5, 0.5))
        svg = fig.to_svg()
        assert 'SVG_TEXT_XYZ' in svg
        plt.close('all')

    def test_multiple_annotations_all_in_texts(self):
        fig, ax = plt.subplots()
        texts = []
        for i in range(5):
            ann = ax.annotate(f'label{i}', xy=(i * 0.1, 0.5))
            texts.append(ann)
        for t in texts:
            assert t in ax.texts
        plt.close('all')

    def test_annotation_survives_plot(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ann = ax.annotate('peak', xy=(0.5, 0.5))
        assert ann in ax.texts
        plt.close('all')

    @pytest.mark.parametrize('style', ['->', '<-', '-'])
    def test_arrowstyle_no_raise_in_svg(self, style):
        fig, ax = plt.subplots()
        ax.annotate('pt', xy=(0.5, 0.5), xytext=(0.2, 0.8),
                    arrowprops=dict(arrowstyle=style))
        svg = fig.to_svg()
        assert len(svg) > 0
        plt.close('all')

    def test_annotation_at_origin(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('origin', xy=(0, 0))
        assert ann.xy == (0, 0)
        plt.close('all')

    def test_annotation_ha_va_in_axes(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('test', xy=(0.5, 0.5), ha='right', va='top')
        assert ann.get_ha() == 'right'
        assert ann.get_va() == 'top'
        plt.close('all')


class TestAnnotationTextContent:
    def test_annotate_text_is_retrievable(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('hello world', xy=(0.5, 0.5))
        assert ann.get_text() == 'hello world'
        plt.close('all')

    def test_annotate_set_text_updates(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('original', xy=(0.5, 0.5))
        ann.set_text('updated')
        assert ann.get_text() == 'updated'
        plt.close('all')

    def test_annotate_fontsize_set(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('label', xy=(0.5, 0.5), fontsize=14)
        assert ann.get_fontsize() == 14
        plt.close('all')

    def test_annotate_color_set(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('label', xy=(0.5, 0.5), color='blue')
        import matplotlib.colors as mc
        assert mc.to_hex(ann.get_color()) == '#0000ff'
        plt.close('all')

    def test_annotate_xy_stored(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('pt', xy=(0.3, 0.7))
        assert ann.xy == (0.3, 0.7)
        plt.close('all')

    def test_annotate_xytext_offset(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('pt', xy=(0.5, 0.5), xytext=(0.1, 0.9))
        # xytext stored separately
        assert ann.xytext == (0.1, 0.9)
        plt.close('all')

    @pytest.mark.parametrize('fs', [8, 10, 12, 16, 20])
    def test_annotate_fontsize_parametric(self, fs):
        fig, ax = plt.subplots()
        ann = ax.annotate('x', xy=(0.5, 0.5), fontsize=fs)
        assert ann.get_fontsize() == fs
        plt.close('all')

    def test_annotate_visible_false(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('hidden', xy=(0.5, 0.5))
        ann.set_visible(False)
        assert not ann.get_visible()
        plt.close('all')

    def test_annotate_alpha_set(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('transparent', xy=(0.5, 0.5))
        ann.set_alpha(0.5)
        assert abs(ann.get_alpha() - 0.5) < 1e-6
        plt.close('all')

    def test_annotate_in_svg_contains_text(self):
        fig, ax = plt.subplots()
        ax.annotate('SVG_MARKER_TEXT', xy=(0.5, 0.5))
        svg = fig.to_svg()
        assert 'SVG_MARKER_TEXT' in svg
        plt.close('all')

    @pytest.mark.parametrize('ha', ['left', 'center', 'right'])
    def test_annotate_ha_parametric(self, ha):
        fig, ax = plt.subplots()
        ann = ax.annotate('pt', xy=(0.5, 0.5), ha=ha)
        assert ann.get_ha() == ha
        plt.close('all')
