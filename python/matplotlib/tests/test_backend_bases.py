"""
Tests for backend_bases (RendererBase, AxesLayout) and the new
RendererSVG / RendererPIL classes.
"""

import re
import pytest


# ===================================================================
# TestRendererBase
# ===================================================================

class TestRendererBase:
    """RendererBase stores dimensions and all methods raise NotImplementedError."""

    def test_init_stores_dimensions(self):
        from matplotlib.backend_bases import RendererBase
        r = RendererBase(800, 600, 100)
        assert r.width == 800
        assert r.height == 600
        assert r.dpi == 100

    @pytest.mark.parametrize("method", [
        "draw_line",
        "draw_rect",
        "draw_circle",
        "draw_polygon",
        "set_clip_rect",
        "clear_clip",
        "get_result",
    ])
    def test_stub_specific_methods_absent(self, method):
        from matplotlib.backend_bases import RendererBase
        r = RendererBase(800, 600, 100)
        assert not hasattr(r, method)

def test_axeslayout_absent():
    import matplotlib.backend_bases as backend_bases
    assert not hasattr(backend_bases, "AxesLayout")


# ===================================================================
# TestRendererSVG
# ===================================================================

class TestRendererSVG:
    """RendererSVG produces correct SVG elements."""

    def _make_renderer(self, w=800, h=600, dpi=100):
        from matplotlib._svg_backend import RendererSVG
        return RendererSVG(w, h, dpi)

    def test_get_result_is_valid_svg(self):
        r = self._make_renderer()
        svg = r.get_result()
        assert svg.startswith('<svg ')
        assert svg.strip().endswith('</svg>')
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_draw_line_produces_polyline(self):
        r = self._make_renderer()
        r.draw_line([0, 100], [0, 100], "#ff0000", 2.0, "-")
        svg = r.get_result()
        assert '<polyline' in svg
        assert 'stroke="#ff0000"' in svg

    def test_draw_line_dashed(self):
        r = self._make_renderer()
        r.draw_line([0, 100], [0, 100], "#000", 1.5, "--")
        svg = r.get_result()
        assert 'stroke-dasharray' in svg

    def test_draw_line_dotted(self):
        r = self._make_renderer()
        r.draw_line([0, 100], [0, 100], "#000", 1.5, ":")
        svg = r.get_result()
        assert 'stroke-dasharray="2,2"' in svg

    def test_draw_markers_produces_circles(self):
        r = self._make_renderer()
        r._draw_markers_simple([10, 20, 30], [40, 50, 60], "#00ff00", 4)
        svg = r.get_result()
        assert svg.count('<circle') == 3

    def test_draw_rect_produces_rect(self):
        r = self._make_renderer()
        r.draw_rect(10, 20, 100, 50, "#0000ff", None)
        svg = r.get_result()
        assert '<rect ' in svg
        assert 'x="10"' in svg

    def test_draw_rect_with_fill(self):
        r = self._make_renderer()
        r.draw_rect(10, 20, 100, 50, "#000", "#ff0000")
        svg = r.get_result()
        assert 'fill="#ff0000"' in svg

    def test_draw_circle_produces_circle(self):
        r = self._make_renderer()
        r.draw_circle(50, 60, 10, "#f00")
        svg = r.get_result()
        assert '<circle' in svg
        assert 'cx="50"' in svg
        assert 'cy="60"' in svg
        assert 'r="10"' in svg

    def test_draw_polygon_produces_polygon(self):
        r = self._make_renderer()
        r.draw_polygon([(0, 0), (100, 0), (50, 50)], "#00f", 0.5)
        svg = r.get_result()
        assert '<polygon' in svg
        assert 'fill-opacity="0.5"' in svg

    def test_draw_text_produces_text(self):
        r = self._make_renderer()
        r._draw_text_simple(100, 200, "Hello World", 14, "#000", "left")
        svg = r.get_result()
        assert '<text' in svg
        assert 'Hello World' in svg
        assert 'text-anchor="start"' in svg

    def test_draw_text_center_anchor(self):
        r = self._make_renderer()
        r._draw_text_simple(100, 200, "Centered", 14, "#000", "center")
        svg = r.get_result()
        assert 'text-anchor="middle"' in svg

    def test_draw_text_right_anchor(self):
        r = self._make_renderer()
        r._draw_text_simple(100, 200, "Right", 14, "#000", "right")
        svg = r.get_result()
        assert 'text-anchor="end"' in svg

    def test_draw_text_escapes_html(self):
        r = self._make_renderer()
        r._draw_text_simple(100, 200, "<b>bold</b>", 14, "#000", "left")
        svg = r.get_result()
        assert '&lt;b&gt;bold&lt;/b&gt;' in svg

    def test_clip_rect_and_clear(self):
        r = self._make_renderer()
        r.set_clip_rect(10, 20, 100, 80)
        r.draw_line([0, 50], [0, 50], "#000", 1, "-")
        svg_clipped = r.get_result()
        assert '<clipPath' in svg_clipped
        assert 'clip-path="url(#' in svg_clipped

    def test_clear_clip_removes_clipping(self):
        r = self._make_renderer()
        r.set_clip_rect(10, 20, 100, 80)
        r.draw_line([0, 50], [0, 50], "#000", 1, "-")
        r.clear_clip()
        r.draw_line([0, 50], [0, 50], "#f00", 1, "-")
        svg = r.get_result()
        # Second polyline should NOT have clip-path
        polylines = re.findall(r'<polyline[^/]*/>', svg)
        assert len(polylines) == 2
        assert 'clip-path' in polylines[0]
        assert 'clip-path' not in polylines[1]

    def test_multiple_draw_calls_accumulate(self):
        r = self._make_renderer()
        r.draw_line([0, 1], [0, 1], "#000", 1, "-")
        r.draw_line([2, 3], [2, 3], "#f00", 1, "-")
        r._draw_markers_simple([10], [10], "#0f0", 3)
        svg = r.get_result()
        assert svg.count('<polyline') == 2
        assert svg.count('<circle') == 1

    def test_draw_path_clip_does_not_leak_to_draw_text(self):
        """draw_path with a gc clip must not bleed into the next draw_text call."""
        import unittest.mock as mock
        from matplotlib.path import Path as MPath
        import numpy as np
        r = self._make_renderer()
        # Build a trivial clipped gc
        gc_with_clip = mock.Mock()
        from matplotlib.transforms import Bbox
        gc_with_clip.get_clip_rectangle.return_value = Bbox([[10, 20], [110, 100]])
        gc_with_clip.get_rgb.return_value = (0, 0, 0, 1)
        gc_with_clip.get_linewidth.return_value = 1.0
        gc_with_clip.get_alpha.return_value = 1.0
        gc_with_clip.get_dashes.return_value = (0, None)
        path = MPath(np.array([[50, 50], [100, 50]]),
                     np.array([MPath.MOVETO, MPath.LINETO]))
        from matplotlib.transforms import IdentityTransform
        r.draw_path(gc_with_clip, path, IdentityTransform())
        # Now draw_text with a gc that has NO clip rectangle.
        gc_no_clip = mock.Mock()
        gc_no_clip.get_clip_rectangle.return_value = None
        gc_no_clip.get_rgb.return_value = (0, 0, 0, 1)
        gc_no_clip.get_alpha.return_value = 1.0
        prop = mock.Mock()
        prop.get_size_in_points.return_value = 12
        r.draw_text(gc_no_clip, 10, 10, "hello", prop, 0)
        svg = r.get_result()
        # The <text> element must NOT carry a clip-path attribute.
        text_tags = re.findall(r'<text[^>]*>', svg)
        assert text_tags, "Expected at least one <text> tag"
        assert 'clip-path' not in text_tags[-1], (
            f"draw_text inherited clip from earlier draw_path: {text_tags[-1]}")

    def test_draw_text_rgba_color_produces_opacity(self):
        """ax.text(color=(r,g,b,a)) without set_alpha stores alpha in gc.get_rgb()[3].
        Both sources must be multiplied to get effective opacity."""
        import unittest.mock as mock
        r = self._make_renderer()
        gc = mock.Mock()
        gc.get_clip_rectangle.return_value = None
        # Color with 30% alpha, no artist-level alpha set (get_alpha returns None).
        gc.get_rgb.return_value = (1.0, 0.0, 0.0, 0.3)
        gc.get_alpha.return_value = None
        prop = mock.Mock()
        prop.get_size_in_points.return_value = 12
        r.draw_text(gc, 50, 50, "faded", prop, 0)
        svg = r.get_result()
        assert 'opacity="0.300"' in svg, (
            f"Expected opacity from rgb[3]=0.3 with gc.get_alpha()=None, got: {svg}")

    def test_draw_text_combined_alpha(self):
        """Both gc.get_alpha() and rgb[3] must be multiplied for effective opacity."""
        import unittest.mock as mock
        r = self._make_renderer()
        gc = mock.Mock()
        gc.get_clip_rectangle.return_value = None
        gc.get_rgb.return_value = (1.0, 0.0, 0.0, 0.5)  # color alpha = 0.5
        gc.get_alpha.return_value = 0.4                  # artist alpha = 0.4
        prop = mock.Mock()
        prop.get_size_in_points.return_value = 12
        r.draw_text(gc, 50, 50, "combined", prop, 0)
        svg = r.get_result()
        # effective = 0.5 * 0.4 = 0.2
        assert 'opacity="0.200"' in svg, (
            f"Expected effective opacity 0.5*0.4=0.2, got: {svg}")

    def test_draw_path_fill_alpha_emits_fill_opacity(self):
        """rgbFace with alpha < 1 must produce fill-opacity attribute in SVG."""
        import unittest.mock as mock
        from matplotlib.path import Path as MPath
        import numpy as np
        r = self._make_renderer()
        gc = mock.Mock()
        gc.get_clip_rectangle.return_value = None
        gc.get_rgb.return_value = (0, 0, 0, 1)
        gc.get_linewidth.return_value = 1.0
        gc.get_alpha.return_value = 1.0
        gc.get_dashes.return_value = (0, None)
        path = MPath(np.array([[10, 10], [50, 10], [30, 50], [10, 10]]),
                     np.array([MPath.MOVETO, MPath.LINETO,
                                MPath.LINETO, MPath.CLOSEPOLY]))
        from matplotlib.transforms import IdentityTransform
        # Face with 50 % alpha
        r.draw_path(gc, path, IdentityTransform(), rgbFace=(1.0, 0.0, 0.0, 0.5))
        svg = r.get_result()
        assert 'fill-opacity="0.500"' in svg, (
            f"Expected fill-opacity for semi-transparent face, got: {svg}")


# ===================================================================
# TestRendererPIL
# ===================================================================

class TestRendererPIL:
    """RendererPIL produces valid PNG output."""

    def _make_renderer(self, w=200, h=150, dpi=100):
        from matplotlib._pil_backend import RendererPIL
        return RendererPIL(w, h, dpi)

    def test_init_creates_image(self):
        r = self._make_renderer()
        assert r.width == 200
        assert r.height == 150

    def test_get_result_returns_png_bytes(self):
        r = self._make_renderer()
        result = r.get_result()
        assert isinstance(result, bytes)
        assert result[:4] == b'\x89PNG'

    def test_draw_line_no_crash(self):
        r = self._make_renderer()
        r.draw_line([10, 100], [10, 100], "#ff0000", 2.0, "-")
        result = r.get_result()
        assert result[:4] == b'\x89PNG'

    def test_draw_markers_no_crash(self):
        r = self._make_renderer()
        r._draw_markers_simple([10, 50, 90], [10, 80, 40], "#00ff00", 4)
        result = r.get_result()
        assert result[:4] == b'\x89PNG'

    def test_draw_rect_no_crash(self):
        r = self._make_renderer()
        r.draw_rect(10, 20, 80, 50, "#0000ff", None)
        result = r.get_result()
        assert result[:4] == b'\x89PNG'

    def test_draw_rect_with_fill(self):
        r = self._make_renderer()
        r.draw_rect(10, 20, 80, 50, "#000", "#ff0000")
        result = r.get_result()
        assert result[:4] == b'\x89PNG'

    def test_draw_circle_no_crash(self):
        r = self._make_renderer()
        r.draw_circle(50, 50, 20, "#f00")
        result = r.get_result()
        assert result[:4] == b'\x89PNG'

    def test_draw_polygon_no_crash(self):
        r = self._make_renderer()
        r.draw_polygon([(10, 10), (100, 10), (50, 80)], "#00f", 0.5)
        result = r.get_result()
        assert result[:4] == b'\x89PNG'

    def test_draw_text_no_crash(self):
        r = self._make_renderer()
        r._draw_text_simple(50, 75, "Hello", 12, "#000", "left")
        result = r.get_result()
        assert result[:4] == b'\x89PNG'

    def test_multiple_draws(self):
        r = self._make_renderer()
        r.draw_line([10, 100], [10, 100], "#f00", 2, "-")
        r._draw_markers_simple([50], [50], "#0f0", 5)
        r.draw_rect(10, 10, 50, 50, "#00f", "#00f")
        result = r.get_result()
        assert result[:4] == b'\x89PNG'


import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def test_axeslayout_absent_for_stub_draw_tests():
    import matplotlib.backend_bases as backend_bases
    assert not hasattr(backend_bases, "AxesLayout")


def test_figure_draw_still_renders_without_axeslayout(tmp_path):
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1])
    path = str(tmp_path / 'test.svg')
    fig.savefig(path)
    with open(path) as f:
        content = f.read()
    assert '<svg' in content
    assert '<path' in content or '<polyline' in content


# ===================================================================
# Extended parametric tests for backend_bases
# ===================================================================

import pytest as _pytest_bb
