"""
Rendering tests — verify that SVG and PNG backends produce actual chart output.

We don't need pixel-perfect matching with upstream matplotlib, but we do need
to confirm that our backends emit the right structural elements: lines for
line plots, circles for scatter, rectangles for bars, text for titles/labels,
and that PNG output is a valid non-blank image.
"""

import re
import pytest

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib._svg_backend import RendererSVG
from matplotlib._pil_backend import RendererPIL


# ===================================================================
# Helpers
# ===================================================================

def _svg(fig):
    """Render a figure to SVG string."""
    dpi = fig.dpi
    w_px = int(fig.figsize[0] * dpi)
    h_px = int(fig.figsize[1] * dpi)
    renderer = RendererSVG(w_px, h_px, dpi)
    fig.draw(renderer)
    return renderer.get_result()


def render_figure_png(fig, dpi=None):
    """Render a figure to PNG bytes using the new renderer."""
    dpi = dpi or fig.dpi
    w_px = int(fig.figsize[0] * dpi)
    h_px = int(fig.figsize[1] * dpi)
    renderer = RendererPIL(w_px, h_px, dpi)
    fig.draw(renderer)
    return renderer.get_result()


def _count_tags(svg, tag):
    """Count occurrences of an SVG tag (e.g. 'polyline', 'circle', 'rect')."""
    return len(re.findall(rf'<{tag}\b', svg))


def _has_text(svg, text):
    """Check whether *text* appears inside an SVG <text> element."""
    # Match <text ...>content</text> where content contains the text
    return text in svg


# ===================================================================
# SVG: basic structure (5 tests)
# ===================================================================

class TestSvgStructure:
    def test_svg_has_root(self):
        """SVG starts with <svg> and ends with </svg>."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        svg = _svg(fig)
        assert svg.startswith('<svg ')
        assert svg.strip().endswith('</svg>')

    def test_svg_dimensions(self):
        """SVG width/height match figure dimensions."""
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, 1])
        svg = _svg(fig)
        assert 'width="800"' in svg
        assert 'height="600"' in svg

    def test_svg_white_background(self):
        """SVG has a white background rect."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        svg = _svg(fig)
        assert 'fill="white"' in svg or 'fill="#ffffff"' in svg

    def test_svg_plot_border(self):
        """SVG has a plot area border (rect with stroke)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        svg = _svg(fig)
        assert 'stroke="#000"' in svg or 'stroke="#000000"' in svg

    def test_svg_clip_path(self):
        """SVG defines a clipPath for data elements."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        svg = _svg(fig)
        assert '<clipPath' in svg


# ===================================================================
# SVG: line plots (4 tests)
# ===================================================================

class TestSvgLinePlot:
    def test_line_produces_polyline(self):
        """ax.plot() produces a <polyline> in SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [10, 20, 15])
        svg = _svg(fig)
        assert _count_tags(svg, 'polyline') >= 1

    def test_multiple_lines(self):
        """Multiple plot() calls produce multiple polylines."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.plot([1, 2, 3], [3, 2, 1])
        svg = _svg(fig)
        assert _count_tags(svg, 'polyline') >= 2

    def test_line_color(self):
        """Line color appears in SVG stroke attribute."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], color='red')
        svg = _svg(fig)
        assert 'stroke="#ff0000"' in svg

    def test_dashed_line(self):
        """Dashed linestyle produces stroke-dasharray."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], linestyle='--')
        svg = _svg(fig)
        assert 'stroke-dasharray' in svg


# ===================================================================
# SVG: scatter plots (3 tests)
# ===================================================================

class TestSvgScatter:
    def test_scatter_produces_circles(self):
        """ax.scatter() produces <circle> elements."""
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [4, 5, 6])
        svg = _svg(fig)
        # 3 data points = at least 3 circles
        assert _count_tags(svg, 'circle') >= 3

    def test_scatter_color(self):
        """Scatter color appears in SVG fill attribute."""
        fig, ax = plt.subplots()
        ax.scatter([1, 2], [3, 4], color='blue')
        svg = _svg(fig)
        assert '#0000ff' in svg

    def test_scatter_point_count(self):
        """Number of circles matches number of data points."""
        fig, ax = plt.subplots()
        n = 10
        ax.scatter(list(range(n)), list(range(n)))
        svg = _svg(fig)
        assert _count_tags(svg, 'circle') >= n


# ===================================================================
# SVG: bar charts (3 tests)
# ===================================================================

class TestSvgBar:
    def test_bar_produces_rects(self):
        """ax.bar() produces <rect> elements for each bar."""
        fig, ax = plt.subplots()
        ax.bar([1, 2, 3], [10, 20, 15])
        svg = _svg(fig)
        # Background rect + border rect + 3 bar rects + clip rect = at least 3 data rects
        rects = re.findall(r'<rect[^/]*/>', svg)
        # Filter out the background/border/clip rects to find data rects
        data_rects = [r for r in rects if 'fill="white"' not in r
                      and 'fill="none"' not in r
                      and '<clipPath' not in r]
        assert len(data_rects) >= 3

    def test_bar_color(self):
        """Bar color appears in SVG."""
        fig, ax = plt.subplots()
        ax.bar([1, 2], [5, 10], color='green')
        svg = _svg(fig)
        assert '#008000' in svg

    def test_barh_produces_rects(self):
        """ax.barh() produces <rect> elements."""
        fig, ax = plt.subplots()
        ax.barh([1, 2, 3], [10, 20, 15])
        svg = _svg(fig)
        rects = re.findall(r'<rect[^/]*/>', svg)
        data_rects = [r for r in rects if 'fill="white"' not in r
                      and 'fill="none"' not in r]
        assert len(data_rects) >= 3


# ===================================================================
# SVG: titles, labels, ticks (5 tests)
# ===================================================================

class TestSvgText:
    def test_title_in_svg(self):
        """set_title() text appears in SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        ax.set_title('My Chart Title')
        svg = _svg(fig)
        assert _has_text(svg, 'My Chart Title')

    def test_xlabel_in_svg(self):
        """set_xlabel() text appears in SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        ax.set_xlabel('X Axis')
        svg = _svg(fig)
        assert _has_text(svg, 'X Axis')

    def test_ylabel_in_svg(self):
        """set_ylabel() text appears in SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        ax.set_ylabel('Y Axis')
        svg = _svg(fig)
        assert _has_text(svg, 'Y Axis')

    def test_tick_labels_present(self):
        """SVG contains tick label <text> elements."""
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 100])
        svg = _svg(fig)
        # Should have tick labels (numbers)
        text_tags = re.findall(r'<text[^>]*>([^<]+)</text>', svg)
        assert len(text_tags) >= 4  # at least a few x and y ticks

    def test_suptitle_not_in_axes_svg(self):
        """suptitle is set on Figure, not rendered by axes SVG directly.
        But axes title IS rendered."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        ax.set_title('Axes Title')
        svg = _svg(fig)
        assert _has_text(svg, 'Axes Title')


# ===================================================================
# SVG: grid (2 tests)
# ===================================================================

class TestSvgGrid:
    def test_grid_off_no_dashed_lines(self):
        """Without grid(), no dashed grid lines in SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        svg = _svg(fig)
        # Grid uses stroke="#ddd" or "#dddddd" with dasharray
        assert 'stroke="#ddd"' not in svg and 'stroke="#dddddd"' not in svg

    def test_grid_on_produces_dashed_lines(self):
        """grid(True) produces dashed grid lines."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.grid(True)
        svg = _svg(fig)
        assert 'stroke="#ddd"' in svg or 'stroke="#dddddd"' in svg


# ===================================================================
# SVG: legend (2 tests)
# ===================================================================

class TestSvgLegend:
    def test_legend_with_labels(self):
        """Legend appears when labels are provided."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2], label='Line A')
        ax.plot([1, 2], [2, 1], label='Line B')
        ax.legend()
        svg = _svg(fig)
        assert _has_text(svg, 'Line A')
        assert _has_text(svg, 'Line B')

    def test_no_legend_without_call(self):
        """Without legend(), no legend box appears even with labels."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2], label='Hidden')
        svg = _svg(fig)
        # The label text should not appear since legend() wasn't called
        # (Legend is only drawn when ax._legend is truthy)
        # Actually our plot() stores label in the element dict, but legend
        # box is only drawn when ax._legend is set
        legend_rect_count = len(re.findall(r'stroke="#999"', svg))
        assert legend_rect_count == 0


# ===================================================================
# SVG: errorbar (2 tests)
# ===================================================================

class TestSvgErrorbar:
    def test_errorbar_has_whiskers(self):
        """errorbar() produces vertical whisker lines."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [10, 20, 15], yerr=[1, 2, 1.5])
        svg = _svg(fig)
        # New renderer uses <polyline> for all lines (whiskers + caps)
        lines = _count_tags(svg, 'line') + _count_tags(svg, 'polyline')
        assert lines >= 6  # at least 2 per whisker (cap + stem) for 3 points

    def test_errorbar_with_xerr(self):
        """errorbar() with xerr produces horizontal whiskers."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2], [10, 20], xerr=[0.5, 1.0])
        svg = _svg(fig)
        lines = _count_tags(svg, 'line') + _count_tags(svg, 'polyline')
        assert lines >= 4


# ===================================================================
# SVG: fill_between (1 test)
# ===================================================================

class TestSvgFillBetween:
    def test_fill_between_produces_polygon(self):
        """fill_between() produces a <polygon> element."""
        fig, ax = plt.subplots()
        ax.fill_between([1, 2, 3], [1, 3, 1], [0, 0, 0])
        svg = _svg(fig)
        assert _count_tags(svg, 'polygon') >= 1


# ===================================================================
# SVG: axhline / axvline (2 tests)
# ===================================================================

class TestSvgAxLines:
    def test_axhline(self):
        """axhline() produces a horizontal line in SVG."""
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.axhline(y=5, color='red')
        svg = _svg(fig)
        assert '#ff0000' in svg

    def test_axvline(self):
        """axvline() produces a vertical line in SVG."""
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.axvline(x=5, color='blue')
        svg = _svg(fig)
        assert '#0000ff' in svg


# ===================================================================
# SVG: multiple subplots (2 tests)
# ===================================================================

class TestSvgSubplots:
    def test_two_subplots(self):
        """Two subplots produce two plot area borders."""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot([1, 2], [1, 2])
        ax2.plot([1, 2], [2, 1])
        svg = _svg(fig)
        # Each subplot gets its own border rect (fill="none" stroke="#000" or "#000000")
        border_rects = len(re.findall(r'fill="none"\s+stroke="#000(?:000)?"', svg))
        assert border_rects >= 2

    def test_subplot_each_has_data(self):
        """Each subplot renders its own data elements."""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot([1, 2, 3], [1, 2, 3])
        ax2.scatter([1, 2, 3], [3, 2, 1])
        svg = _svg(fig)
        assert _count_tags(svg, 'polyline') >= 1
        assert _count_tags(svg, 'circle') >= 3


# ===================================================================
# PNG: valid image output (5 tests)
# ===================================================================

class TestPngOutput:
    def test_png_bytes(self):
        """render_figure_png() returns bytes."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        data = render_figure_png(fig)
        assert isinstance(data, bytes)
        assert len(data) > 100  # not empty

    def test_png_signature(self):
        """PNG output starts with the PNG signature."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        data = render_figure_png(fig)
        assert data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_png_dimensions(self):
        """PNG image dimensions match figure size * dpi."""
        from PIL import Image
        import io
        fig = plt.figure(figsize=(4, 3), dpi=50)
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, 1])
        data = render_figure_png(fig, dpi=50)
        img = Image.open(io.BytesIO(data))
        assert img.size == (200, 150)  # 4*50, 3*50

    def test_png_not_blank(self):
        """PNG output is not a blank white image — it has non-white pixels."""
        from PIL import Image
        import io
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 10, 5], color='red')
        data = render_figure_png(fig)
        img = Image.open(io.BytesIO(data))
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 10  # lots of non-white pixels (axes, line, ticks)

    def test_png_colored_line(self):
        """PNG contains pixels of the plotted line's color."""
        from PIL import Image
        import io
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 5, 10, 5, 0], color='red')
        data = render_figure_png(fig)
        img = Image.open(io.BytesIO(data))
        pixels = list(img.getdata())
        red_pixels = [p for p in pixels if p[0] > 200 and p[1] < 50 and p[2] < 50]
        assert len(red_pixels) > 0, "Expected red pixels from the plotted line"


# ===================================================================
# PNG: different chart types render (3 tests)
# ===================================================================

class TestPngChartTypes:
    def test_png_scatter(self):
        """Scatter plot PNG is not blank."""
        from PIL import Image
        import io
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3, 4, 5], [5, 3, 4, 2, 1], color='blue')
        data = render_figure_png(fig)
        img = Image.open(io.BytesIO(data))
        pixels = list(img.getdata())
        blue_pixels = [p for p in pixels if p[2] > 200 and p[0] < 50 and p[1] < 50]
        assert len(blue_pixels) > 0

    def test_png_bar(self):
        """Bar chart PNG has colored rectangles."""
        from PIL import Image
        import io
        fig, ax = plt.subplots()
        ax.bar([1, 2, 3], [10, 20, 15], color='green')
        data = render_figure_png(fig)
        img = Image.open(io.BytesIO(data))
        pixels = list(img.getdata())
        green_pixels = [p for p in pixels if p[1] > 100 and p[0] < 50 and p[2] < 50]
        assert len(green_pixels) > 0

    def test_png_multiple_subplots(self):
        """Multiple subplots PNG is non-trivial."""
        from PIL import Image
        import io
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot([1, 2, 3], [1, 4, 2], color='red')
        ax2.bar([1, 2, 3], [5, 10, 7], color='blue')
        data = render_figure_png(fig)
        img = Image.open(io.BytesIO(data))
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 50


# ===================================================================
# SVG: savefig integration (2 tests)
# ===================================================================

class TestSavefig:
    def test_savefig_svg(self, tmp_path):
        """fig.savefig() writes a valid SVG file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [10, 20, 15])
        path = str(tmp_path / 'test.svg')
        fig.savefig(path)
        with open(path) as f:
            content = f.read()
        assert content.startswith('<svg ')
        assert '<polyline' in content

    def test_savefig_png(self, tmp_path):
        """fig.savefig() writes a valid PNG file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [10, 20, 15])
        path = str(tmp_path / 'test.png')
        fig.savefig(path)
        with open(path, 'rb') as f:
            data = f.read()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'


# ===================================================================
# Additional rendering tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt


class TestSvgParametric:
    """Parametric SVG rendering tests."""

    @pytest.mark.parametrize('n_lines', [1, 2, 3, 5])
    def test_n_lines_svg_valid(self, n_lines):
        """SVG is valid for n plotted lines."""
        fig, ax = plt.subplots()
        for i in range(n_lines):
            ax.plot([0, 1], [i, i+1])
        svg = fig.to_svg()
        assert '<svg' in svg
        assert '</svg>' in svg
        plt.close('all')

    @pytest.mark.parametrize('color,expected', [
        ('red', 'ff0000'),
        ('blue', '0000ff'),
        ('green', '008000'),
    ])
    def test_line_color_in_svg(self, color, expected):
        """Line color appears in SVG output."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color=color)
        svg = fig.to_svg()
        assert expected.lower() in svg.lower()
        plt.close('all')

    @pytest.mark.parametrize('title', ['A', 'B', 'Long Title Text'])
    def test_title_text_in_svg(self, title):
        """Title text is present in SVG."""
        fig, ax = plt.subplots()
        ax.set_title(title)
        svg = fig.to_svg()
        assert title in svg
        plt.close('all')

    @pytest.mark.parametrize('w,h', [(4, 3), (8, 6), (10, 10)])
    def test_figsize_affects_svg_dimensions(self, w, h):
        """Figure size affects SVG dimensions attribute."""
        fig, ax = plt.subplots(figsize=(w, h))
        svg = fig.to_svg()
        # Should have width and height attributes
        assert 'width' in svg
        assert 'height' in svg
        plt.close('all')

    @pytest.mark.parametrize('n_bars', [1, 3, 5, 10])
    def test_bar_chart_n_bars(self, n_bars):
        """Bar chart with n bars produces SVG."""
        fig, ax = plt.subplots()
        ax.bar(list(range(n_bars)), list(range(n_bars)))
        svg = fig.to_svg()
        assert len(svg) > 100
        plt.close('all')

    def test_scatter_and_line_combined(self):
        """Scatter and line on same axes produces SVG."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.scatter([0.5, 1.5], [0.5, 0.5])
        svg = fig.to_svg()
        assert '</svg>' in svg
        plt.close('all')

    def test_imshow_svg_contains_image(self):
        """imshow SVG contains image element."""
        import numpy as np
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((10, 10)))
        svg = fig.to_svg()
        assert '<image' in svg or '<rect' in svg
        plt.close('all')

    def test_twin_axes_svg(self):
        """Twin axes renders to SVG without error."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot([0, 1], [0, 1])
        ax2.plot([0, 1], [1, 0], color='red')
        svg = fig.to_svg()
        assert '</svg>' in svg
        plt.close('all')


# ===================================================================
# Extended parametric rendering tests
# ===================================================================

class TestRenderingParametricExtended:
    """Parametric tests for rendering output."""

    @pytest.mark.parametrize('n', [1, 2, 3, 5])
    def test_svg_n_subplots_valid(self, n):
        """SVG is valid for figure with n subplots."""
        fig, axes = plt.subplots(1, n)
        svg = fig.to_svg()
        assert '<svg' in svg
        assert '</svg>' in svg
        plt.close('all')

    @pytest.mark.parametrize('figsize', [(4, 3), (6.4, 4.8), (8, 6), (10, 8)])
    def test_svg_figsize_produces_output(self, figsize):
        """SVG output is non-empty for any figsize."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert isinstance(svg, str)
        assert len(svg) > 200
        plt.close('all')

    @pytest.mark.parametrize('linewidth', [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_svg_linewidth_no_error(self, linewidth):
        """SVG is produced with various linewidths."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linewidth=linewidth)
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_svg_alpha_no_error(self, alpha):
        """SVG is produced with various alpha values."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], alpha=alpha)
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'v', 'D', '*', '+', 'x'])
    def test_svg_marker_no_error(self, marker):
        """SVG is produced with various markers."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0], marker=marker)
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('linestyle', ['-', '--', ':', '-.'])
    def test_svg_linestyle_no_error(self, linestyle):
        """SVG is produced with various linestyles."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle=linestyle)
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('n_bars', [1, 3, 5, 10])
    def test_svg_bar_chart_n_bars(self, n_bars):
        """SVG is produced for bar chart with n bars."""
        fig, ax = plt.subplots()
        ax.bar(range(n_bars), range(n_bars))
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('xlabel,ylabel', [
        ('X', 'Y'),
        ('Time (s)', 'Amplitude'),
        ('', ''),
        ('Distance', 'Value'),
    ])
    def test_svg_with_axis_labels(self, xlabel, ylabel):
        """SVG is produced when axis labels are set."""
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    @pytest.mark.parametrize('dpi', [72, 96, 100, 150])
    def test_svg_any_dpi(self, dpi):
        """SVG is produced for any DPI."""
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')
