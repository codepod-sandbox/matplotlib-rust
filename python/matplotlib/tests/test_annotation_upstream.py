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
    """FancyArrowPatch.draw must call renderer.draw_arrow."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.backend_bases import AxesLayout
    from matplotlib.scale import LinearScale

    class MockRenderer:
        def __init__(self):
            self.draw_arrow_calls = []
        def draw_arrow(self, *args, **kwargs):
            self.draw_arrow_calls.append(args)

    layout = AxesLayout(0, 0, 100, 100, 0, 1, 0, 1,
                        LinearScale(), LinearScale())
    renderer = MockRenderer()
    patch = FancyArrowPatch((0.1, 0.2), (0.8, 0.7),
                             arrowstyle='->', color='red', linewidth=1.5)
    patch.draw(renderer, layout)
    assert len(renderer.draw_arrow_calls) == 1


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
