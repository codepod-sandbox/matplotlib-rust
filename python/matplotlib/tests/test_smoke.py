"""Smoke tests for pre-push hook — fast, no parametrize."""
import pytest


def test_import_matplotlib():
    import matplotlib
    assert matplotlib.__name__ == 'matplotlib'


def test_import_pyplot():
    import matplotlib.pyplot as plt
    assert hasattr(plt, 'subplots')


def test_figure_creation():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    assert fig is not None
    plt.close(fig)


def test_subplots():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    assert fig is not None
    assert ax is not None
    plt.close("all")


def test_plot_line():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1, 2], [0, 1, 4])
    assert len(ax.lines) == 1
    plt.close("all")


def test_set_title():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("Test")
    assert ax.get_title() == "Test"
    plt.close("all")


def test_set_xlim():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    assert ax.get_xlim() == (0, 10)
    plt.close("all")


def test_bar():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [4, 5, 6])
    assert len(ax.patches) == 3
    plt.close("all")


def test_scatter():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    plt.close("all")


def test_legend():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line1")
    ax.legend()
    plt.close("all")


def test_colors():
    import matplotlib.colors as mcolors
    assert mcolors.to_rgba('red') == (1.0, 0.0, 0.0, 1.0)


def test_colormap():
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    assert cmap is not None


def test_patches():
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    assert r.get_width() == 1
    assert r.get_height() == 1


def test_text():
    from matplotlib.text import Text
    t = Text(0, 0, 'hello')
    assert t.get_text() == 'hello'


def test_lines():
    from matplotlib.lines import Line2D
    line = Line2D([0, 1], [0, 1])
    assert line.get_linewidth() > 0


def test_collections():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([1], [1])
    assert len(ax.collections) == 1
    plt.close("all")


def test_cycler():
    from matplotlib.cycler import cycler
    c = cycler(color=['r', 'g', 'b'])
    assert len(c) == 3


def test_multiple_subplots():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2)
    assert len(axes) == 2
    assert len(axes[0]) == 2
    plt.close("all")


@pytest.mark.skip(reason="Phase 2: savefig SVG requires ft2font")
def test_save_svg():
    import matplotlib.pyplot as plt
    import io
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    svg = buf.getvalue()
    assert '<svg' in svg
    plt.close("all")


@pytest.mark.skip(reason="Phase 1: savefig PNG requires _backend_agg")
def test_save_png():
    import matplotlib.pyplot as plt
    import io
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    data = buf.getvalue()
    assert data[:4] == b'\x89PNG'
    plt.close("all")
