"""Tests for rendering fixes and improvements (sub-project B).

Baseline: 868 passed, 0 failed. Any regression is a bug.

NOTE: All tests in this module require Phase 1 (_backend_agg) or Phase 2 (ft2font).
"""
import io
import numpy as np
import pytest



def test_polygon_fill_png():
    """fill_between() must produce solid fill pixels, not a hollow outline."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    ax.fill_between([0, 1], [0, 0], [1, 1], color='red')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = np.asarray(img).reshape(-1, 3)
    red_pixels = sum(1 for r, g, b in pixels if r > 200 and g < 50)
    assert red_pixels > len(pixels) * 0.05, \
        f"Expected filled red region, got {red_pixels}/{len(pixels)} red pixels"
    plt.close(fig)


def test_pie_fill_png():
    """Pie chart slices must be filled, not wireframe."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    ax.pie([1, 2, 3], colors=['red', 'green', 'blue'])
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = np.asarray(img).reshape(-1, 3)
    saturated = sum(
        1 for r, g, b in pixels
        if (r > 180 and g < 80 and b < 80)
        or (g > 180 and r < 80 and b < 80)
        or (b > 180 and r < 80 and g < 80)
    )
    assert saturated > len(pixels) * 0.03, \
        f"Expected filled pie slices, got {saturated}/{len(pixels)} saturated pixels"
    plt.close(fig)


def test_markers_square_svg():
    """scatter with marker='s' produces <rect> elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3], marker='s', color='blue')
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert '<rect' in svg, \
        f"Expected <rect> elements for square markers, got SVG without <rect>"
    plt.close(fig)


def test_markers_triangle_svg():
    """scatter with marker='^' produces marker elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3], marker='^', color='green')
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    # Real matplotlib SVG backend uses <path> for triangle markers; our custom
    # renderer uses <polygon>.  Accept either.
    assert '<polygon' in svg or '<path' in svg, \
        f"Expected <polygon> or <path> elements for triangle markers"
    plt.close(fig)


def test_markers_square_png():
    """scatter with marker='s' produces non-white pixels in correct positions."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    ax.scatter([0.5], [0.5], marker='s', s=200, color='blue')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = np.asarray(img).reshape(-1, 3)
    blue_pixels = sum(1 for r, g, b in pixels if b > 150 and r < 100 and g < 100)
    assert blue_pixels > 5, \
        f"Expected blue square marker pixels, got {blue_pixels}"
    plt.close(fig)


def test_imshow_svg():
    """imshow() produces a base64 data URL image embedded in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    data = [[0, 128, 255], [64, 192, 32]]
    ax.imshow(data, cmap='viridis')
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert 'data:image/png;base64,' in svg, \
        "Expected base64 PNG embedded in SVG for imshow()"
    plt.close(fig)


def test_imshow_png():
    """imshow() with RGB array renders correct pixel colors."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1, 1), dpi=10)
    data = [
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]],
    ]
    ax.imshow(data)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    pixels = np.asarray(img).reshape(-1, 3)
    has_red = any(r > 200 and g < 50 for r, g, b in pixels)
    has_green = any(g > 200 and r < 50 for r, g, b in pixels)
    has_blue = any(b > 200 and r < 50 for r, g, b in pixels)
    assert has_red, "Expected red pixel from imshow"
    assert has_green, "Expected green pixel from imshow"
    assert has_blue, "Expected blue pixel from imshow"
    plt.close(fig)


# ===================================================================
# Additional rendering tests (upstream-inspired)
# ===================================================================

def test_line_svg_has_polyline():
    """A simple plot produces a polyline in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    svg = fig.to_svg()
    assert '<polyline' in svg or 'polyline' in svg
    plt.close('all')


def test_scatter_svg_has_path():
    """Scatter plot produces path elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([0, 1, 2], [0, 1, 0])
    svg = fig.to_svg()
    assert '<circle' in svg or 'circle' in svg or '<path' in svg
    plt.close('all')


def test_text_svg_has_text():
    """Text added to axes appears in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'hello')
    svg = fig.to_svg()
    assert 'hello' in svg
    plt.close('all')


def test_title_svg():
    """Axes title appears in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title('My Title')
    svg = fig.to_svg()
    assert 'My Title' in svg
    plt.close('all')


def test_xlabel_svg():
    """X-axis label appears in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlabel('X Axis')
    svg = fig.to_svg()
    assert 'X Axis' in svg
    plt.close('all')


def test_ylabel_svg():
    """Y-axis label appears in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylabel('Y Axis')
    svg = fig.to_svg()
    assert 'Y Axis' in svg
    plt.close('all')


def test_legend_svg():
    """Legend label appears in SVG output."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label='data series')
    ax.legend()
    svg = fig.to_svg()
    assert 'data series' in svg
    plt.close('all')


def test_bar_svg_has_rect():
    """Bar chart produces rect elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [4, 5, 6])
    svg = fig.to_svg()
    assert '<rect' in svg
    plt.close('all')


def test_dashed_line_svg():
    """Dashed line produces stroke-dasharray in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--')
    svg = fig.to_svg()
    assert 'stroke-dasharray' in svg
    plt.close('all')


def test_colored_line_svg():
    """Colored line has stroke attribute in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color='red')
    svg = fig.to_svg()
    # Red is #ff0000 or rgb(255,0,0)
    assert 'stroke' in svg
    plt.close('all')


def test_svg_contains_svg_tag():
    """Every figure SVG output contains svg element."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    svg = fig.to_svg()
    assert '<svg' in svg
    plt.close('all')


def test_multiple_lines_svg():
    """Multiple lines produce multiple polylines in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label='line1')
    ax.plot([0, 1], [1, 0], label='line2')
    svg = fig.to_svg()
    # Should have at least 2 polylines
    count = svg.count('<polyline')
    assert count >= 2
    plt.close('all')


# ===================================================================
# Additional SVG rendering tests
# ===================================================================

def test_figure_suptitle_svg():
    """Suptitle appears in SVG output."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.suptitle('Main Title')
    svg = fig.to_svg()
    assert 'Main Title' in svg
    plt.close('all')


def test_fill_between_svg():
    """fill_between produces path elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.fill_between([0, 1, 2], [0, 1, 0], [1, 2, 1], color='blue')
    svg = fig.to_svg()
    assert '<path' in svg or '<polygon' in svg
    plt.close('all')


def test_scatter_svg_has_circles():
    """Scatter with default marker produces circle-like elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([0.5], [0.5], s=100, color='red')
    svg = fig.to_svg()
    # should contain some rendering element
    assert '<circle' in svg or '<path' in svg or '<use' in svg
    plt.close('all')


def test_axhline_svg():
    """axhline produces a horizontal line in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.axhline(y=0.5)
    svg = fig.to_svg()
    assert '<polyline' in svg or '<line' in svg or '<path' in svg
    plt.close('all')


def test_axvline_svg():
    """axvline produces a vertical line in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.axvline(x=0.5)
    svg = fig.to_svg()
    assert '<polyline' in svg or '<line' in svg or '<path' in svg
    plt.close('all')


def test_annotate_svg_has_text():
    """annotate text appears in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.annotate('note', xy=(0.5, 0.5))
    svg = fig.to_svg()
    assert 'note' in svg
    plt.close('all')


def test_svg_valid_structure():
    """SVG output is well-formed (starts with <svg and ends with </svg>)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    svg = fig.to_svg()
    assert '<svg' in svg
    assert '</svg>' in svg
    plt.close('all')


def test_imshow_svg_has_image_tag():
    """imshow in SVG uses image element."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow([[1, 2], [3, 4]], cmap='gray')
    svg = fig.to_svg()
    assert '<image' in svg or 'image' in svg.lower()
    plt.close('all')


def test_line_color_red_svg():
    """Red line contains red color code in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color='red')
    svg = fig.to_svg()
    # ff0000 or rgb(255 or #ff0000
    assert 'ff0000' in svg.lower() or '#f00' in svg.lower() or 'rgb(255' in svg
    plt.close('all')


def test_empty_plot_svg():
    """Empty axes produce valid SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    svg = fig.to_svg()
    assert '<svg' in svg
    assert len(svg) > 100
    plt.close('all')


def test_errorbar_svg():
    """errorbar produces path elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.5, 0.5])
    svg = fig.to_svg()
    assert '<path' in svg or '<polyline' in svg
    plt.close('all')


def test_hist_svg():
    """Histogram produces rect elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist([1, 2, 2, 3, 3, 3], bins=3)
    svg = fig.to_svg()
    assert '<rect' in svg
    plt.close('all')


def test_hlines_svg():
    """hlines produces line elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hlines([0, 1, 2], 0, 1)
    svg = fig.to_svg()
    assert '<polyline' in svg or '<line' in svg or '<path' in svg
    plt.close('all')


def test_vlines_svg():
    """vlines produces line elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.vlines([0, 1, 2], 0, 1)
    svg = fig.to_svg()
    assert '<polyline' in svg or '<line' in svg or '<path' in svg
    plt.close('all')


def test_pie_svg_has_path():
    """Pie chart produces path elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.pie([1, 2, 3])
    svg = fig.to_svg()
    assert '<path' in svg
    plt.close('all')


def test_step_plot_svg():
    """Step plot produces polyline in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.step([0, 1, 2, 3], [0, 1, 0, 1])
    svg = fig.to_svg()
    assert '<polyline' in svg or '<path' in svg
    plt.close('all')


# ===================================================================
# Additional rendering tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt


class TestSVGRendering:
    """Tests for SVG output content."""

    def test_bar_chart_svg_has_rects(self):
        fig, ax = plt.subplots()
        ax.bar([1, 2, 3], [4, 5, 6])
        svg = fig.to_svg()
        assert '<rect' in svg or '<path' in svg
        plt.close('all')

    def test_scatter_svg(self):
        import numpy as np
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [4, 5, 6])
        svg = fig.to_svg()
        assert len(svg) > 100
        plt.close('all')

    def test_fill_between_svg(self):
        import numpy as np
        fig, ax = plt.subplots()
        ax.fill_between([0, 1, 2], [0, 0, 0], [1, 2, 1])
        svg = fig.to_svg()
        assert len(svg) > 100
        plt.close('all')

    def test_text_svg_has_text(self):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'HELLO_WORLD')
        svg = fig.to_svg()
        assert 'HELLO_WORLD' in svg
        plt.close('all')

    def test_xlabel_svg(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X Label ABC')
        svg = fig.to_svg()
        assert 'X Label ABC' in svg
        plt.close('all')

    def test_ylabel_svg(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Y Label DEF')
        svg = fig.to_svg()
        assert 'Y Label DEF' in svg
        plt.close('all')

    def test_title_svg(self):
        fig, ax = plt.subplots()
        ax.set_title('My Title XYZ')
        svg = fig.to_svg()
        assert 'My Title XYZ' in svg
        plt.close('all')

    @pytest.mark.parametrize('n', [1, 3, 5])
    def test_multiple_lines_svg(self, n):
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        svg = fig.to_svg()
        assert svg.count('<polyline') >= n or '<path' in svg
        plt.close('all')

    def test_empty_axes_svg_valid(self):
        fig, ax = plt.subplots()
        svg = fig.to_svg()
        assert svg.startswith('<?xml') or svg.startswith('<svg') or '<svg' in svg
        plt.close('all')

    def test_legend_svg_has_text(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='my line')
        ax.legend()
        svg = fig.to_svg()
        assert 'my line' in svg
        plt.close('all')


class TestSVGContentDetails:
    def test_xlabel_in_svg(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('Time (s)')
        svg = fig.to_svg()
        assert 'Time (s)' in svg
        plt.close('all')

    def test_ylabel_in_svg(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Amplitude')
        svg = fig.to_svg()
        assert 'Amplitude' in svg
        plt.close('all')

    def test_imshow_svg_not_empty(self):
        import numpy as np
        fig, ax = plt.subplots()
        data = np.zeros((4, 4))
        ax.imshow(data)
        svg = fig.to_svg()
        assert len(svg) > 100
        plt.close('all')

    def test_pie_svg_not_empty(self):
        fig, ax = plt.subplots()
        ax.pie([30, 40, 30], labels=['A', 'B', 'C'])
        svg = fig.to_svg()
        assert len(svg) > 100
        plt.close('all')

    def test_pie_svg_contains_labels(self):
        fig, ax = plt.subplots()
        ax.pie([30, 40, 30], labels=['Alpha', 'Beta', 'Gamma'])
        svg = fig.to_svg()
        assert 'Alpha' in svg
        assert 'Beta' in svg
        plt.close('all')

    def test_histogram_svg_not_empty(self):
        import numpy as np
        fig, ax = plt.subplots()
        ax.hist(np.random.randn(50), bins=10)
        svg = fig.to_svg()
        assert len(svg) > 100
        plt.close('all')

    def test_svg_contains_svg_tag(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert '<svg' in svg
        plt.close('all')

    def test_svg_well_formed_closes_svg(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        svg = fig.to_svg()
        assert '</svg>' in svg
        plt.close('all')

    @pytest.mark.parametrize('label', ['series A', 'test data', 'line 1'])
    def test_label_appears_in_legend_svg(self, label):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label=label)
        ax.legend()
        svg = fig.to_svg()
        assert label in svg
        plt.close('all')

    def test_two_axes_svg_not_empty(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot([0, 1], [0, 1])
        ax2.plot([0, 1], [1, 0])
        svg = fig.to_svg()
        assert len(svg) > 200
        plt.close('all')

    def test_suptitle_in_svg(self):
        fig, ax = plt.subplots()
        fig.suptitle('Overall Title')
        svg = fig.to_svg()
        assert 'Overall Title' in svg
        plt.close('all')
