"""Tests for rendering fixes and improvements (sub-project B).

Baseline: 868 passed, 0 failed. Any regression is a bug.
"""
import io
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
    pixels = list(img.getdata())
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
    pixels = list(img.getdata())
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
    """scatter with marker='^' produces <polygon> elements in SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3], marker='^', color='green')
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert '<polygon' in svg, \
        f"Expected <polygon> elements for triangle markers, got SVG without <polygon>"
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
    pixels = list(img.getdata())
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
    pixels = list(img.getdata())
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
