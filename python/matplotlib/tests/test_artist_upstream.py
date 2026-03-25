"""
Upstream matplotlib tests for the Artist base class.
"""

import pytest

from matplotlib.artist import Artist


def test_artist_default_visible():
    """Artist is visible by default."""
    a = Artist()
    assert a.get_visible() is True


def test_artist_set_visible():
    """set_visible changes visibility."""
    a = Artist()
    a.set_visible(False)
    assert a.get_visible() is False
    a.set_visible(True)
    assert a.get_visible() is True


def test_artist_default_alpha():
    """Artist alpha is None by default."""
    a = Artist()
    assert a.get_alpha() is None


def test_artist_set_alpha():
    """set_alpha changes alpha."""
    a = Artist()
    a.set_alpha(0.5)
    assert a.get_alpha() == 0.5


def test_artist_default_label():
    """Artist label is empty by default."""
    a = Artist()
    assert a.get_label() == ''


def test_artist_set_label():
    """set_label changes label."""
    a = Artist()
    a.set_label('test')
    assert a.get_label() == 'test'


def test_artist_set_label_none():
    """set_label(None) sets to '_nolegend_'."""
    a = Artist()
    a.set_label(None)
    assert a.get_label() == '_nolegend_'


def test_artist_default_zorder():
    """Artist default zorder is 0."""
    a = Artist()
    assert a.get_zorder() == 0


def test_artist_set_zorder():
    """set_zorder changes zorder."""
    a = Artist()
    a.set_zorder(5)
    assert a.get_zorder() == 5


def test_artist_stale():
    """Artist starts stale."""
    a = Artist()
    assert a._stale is True


def test_artist_figure_none():
    """Artist figure is None by default."""
    a = Artist()
    assert a.figure is None


def test_artist_axes_none():
    """Artist axes is None by default."""
    a = Artist()
    assert a.axes is None


def test_artist_set_batch():
    """Artist.set() batch setter."""
    a = Artist()
    a.set(visible=False, alpha=0.3, label='batch')
    assert a.get_visible() is False
    assert a.get_alpha() == 0.3
    assert a.get_label() == 'batch'


def test_artist_remove_from_axes():
    """Artist.remove removes from axes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    ax.add_artist(r)
    assert r in ax.patches
    r.remove()
    assert r not in ax.patches
    assert r.axes is None
    assert r.figure is None
# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from lib/matplotlib/tests/test_artist.py and test_lines.py
import pytest
import matplotlib.pyplot as plt


def test_artist_clip_on_default_true():
    """Artist.clip_on must default to True."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    assert line.get_clip_on() is True


def test_artist_set_clip_on():
    """set_clip_on(False) must turn off clipping."""
    from matplotlib.patches import Rectangle
    r = Rectangle((0, 0), 1, 1)
    r.set_clip_on(False)
    assert r.get_clip_on() is False


def test_artist_alpha():
    """set_alpha / get_alpha round-trip."""
    from matplotlib.lines import Line2D
    line = Line2D([0], [0])
    assert line.get_alpha() is None  # default
    line.set_alpha(0.5)
    assert line.get_alpha() == 0.5


def test_zorder_defaults():
    """Line2D zorder=2, Patch zorder=1, Text zorder=3."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    assert Line2D([0], [0]).get_zorder() == 2
    assert Patch().get_zorder() == 1
    assert Text(0, 0, 'x').get_zorder() == 3


def test_zorder_draw_order_in_svg():
    """Artists with lower zorder must appear earlier in SVG."""
    fig, ax = plt.subplots()
    # Default: patches zorder=1 before lines zorder=2
    ax.bar([1], [1])  # adds a Rectangle (zorder=1)
    ax.plot([0, 2], [0, 2], label='line')  # adds a Line2D (zorder=2)
    svg = fig.to_svg()
    # polyline (line) must appear after rect (bar) in SVG
    rect_pos = svg.find('<rect')
    line_pos = svg.find('<polyline')
    assert rect_pos < line_pos, "Patch (zorder=1) must appear before Line2D (zorder=2)"
    plt.close('all')


def test_alpha_in_svg():
    """A line with alpha=0.5 must store alpha on Line2D and produce opacity in SVG."""
    fig, ax = plt.subplots()
    line, = ax.plot([1, 2], [1, 2], alpha=0.5)
    # Verify alpha is stored on the artist
    assert line.get_alpha() == 0.5
    # Verify opacity appears in SVG output
    svg = fig.to_svg()
    assert 'opacity="0.5"' in svg or 'opacity="0.5' in svg
    plt.close('all')


def test_linestyle_tuple_format():
    """Linestyle as (offset, (on, off)) tuple must appear in SVG."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], linestyle=(0, (3, 5)))
    svg = fig.to_svg()
    assert 'stroke-dasharray' in svg
    plt.close('all')


def test_linestyle_named_solid():
    """linestyle='solid' must produce no stroke-dasharray."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], linestyle='solid')
    svg = fig.to_svg()
    # solid lines produce no dasharray
    assert 'stroke-dasharray' not in svg or svg.count('stroke-dasharray') == 0
    plt.close('all')


def test_linestyle_loosely_dashed():
    """Named extended linestyle 'loosely dashed' must produce dasharray."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], linestyle='loosely dashed')
    svg = fig.to_svg()
    assert 'stroke-dasharray' in svg
    plt.close('all')


# ===================================================================
# Artist extended property tests
# ===================================================================

class TestArtistExtendedProperties:
    def test_clip_box_default_none(self):
        a = Artist()
        assert a.get_clip_box() is None

    def test_set_clip_box(self):
        a = Artist()
        box = object()
        a.set_clip_box(box)
        assert a.get_clip_box() is box

    def test_clip_path_default_none(self):
        a = Artist()
        assert a.get_clip_path() is None

    def test_set_clip_path(self):
        a = Artist()
        path = object()
        a.set_clip_path(path)
        assert a.get_clip_path() is path

    def test_transform_default_none(self):
        a = Artist()
        assert a.get_transform() is None

    def test_set_transform(self):
        a = Artist()
        t = object()
        a.set_transform(t)
        assert a.get_transform() is t

    def test_is_transform_set_false_by_default(self):
        a = Artist()
        assert not a.is_transform_set()

    def test_is_transform_set_after_setting(self):
        a = Artist()
        a.set_transform(object())
        assert a.is_transform_set()

    def test_animated_default_false(self):
        a = Artist()
        assert a.get_animated() is False

    def test_set_animated(self):
        a = Artist()
        a.set_animated(True)
        assert a.get_animated() is True

    def test_rasterized_default_none(self):
        a = Artist()
        assert a.get_rasterized() is None

    def test_set_rasterized(self):
        a = Artist()
        a.set_rasterized(True)
        assert a.get_rasterized() is True

    def test_sketch_params_default_none(self):
        a = Artist()
        assert a.get_sketch_params() is None

    def test_set_sketch_params(self):
        a = Artist()
        a.set_sketch_params(scale=1.0)
        params = a.get_sketch_params()
        assert params is not None
        assert params[0] == 1.0

    def test_set_sketch_params_none_clears(self):
        a = Artist()
        a.set_sketch_params(scale=1.0)
        a.set_sketch_params(None)
        assert a.get_sketch_params() is None

    def test_sketch_params_with_length_randomness(self):
        a = Artist()
        a.set_sketch_params(scale=2.0, length=64, randomness=8)
        scale, length, randomness = a.get_sketch_params()
        assert scale == 2.0
        assert length == 64
        assert randomness == 8

    def test_snap_default_none(self):
        a = Artist()
        assert a.get_snap() is None

    def test_set_snap(self):
        a = Artist()
        a.set_snap(True)
        assert a.get_snap() is True

    def test_path_effects_default_empty(self):
        a = Artist()
        assert a.get_path_effects() == []

    def test_set_path_effects(self):
        a = Artist()
        effects = [object(), object()]
        a.set_path_effects(effects)
        assert len(a.get_path_effects()) == 2

    def test_url_default_none(self):
        a = Artist()
        assert a.get_url() is None

    def test_set_url(self):
        a = Artist()
        a.set_url('https://example.com')
        assert a.get_url() == 'https://example.com'

    def test_gid_default_none(self):
        a = Artist()
        assert a.get_gid() is None

    def test_set_gid(self):
        a = Artist()
        a.set_gid('my-id')
        assert a.get_gid() == 'my-id'

    def test_in_layout_default_true(self):
        a = Artist()
        assert a.get_in_layout() is True

    def test_set_in_layout(self):
        a = Artist()
        a.set_in_layout(False)
        assert a.get_in_layout() is False

    def test_picker_default_none(self):
        a = Artist()
        assert a.get_picker() is None

    def test_set_picker(self):
        a = Artist()
        a.set_picker(True)
        assert a.get_picker() is True

    def test_pickable_false_by_default(self):
        a = Artist()
        assert not a.pickable()

    def test_pickable_true_after_set_picker(self):
        a = Artist()
        a.set_picker(lambda *args: True)
        assert a.pickable()

    def test_agg_filter_default_none(self):
        a = Artist()
        assert a.get_agg_filter() is None

    def test_set_agg_filter(self):
        a = Artist()
        f = lambda x: x
        a.set_agg_filter(f)
        assert a.get_agg_filter() is f

    def test_contains_default_none(self):
        a = Artist()
        assert a.get_contains() is None

    def test_set_contains(self):
        a = Artist()
        fn = lambda event: True
        a.set_contains(fn)
        assert a.get_contains() is fn

    def test_figure_default_none(self):
        a = Artist()
        assert a.get_figure() is None

    def test_set_figure(self):
        a = Artist()
        fig = object()
        a.set_figure(fig)
        assert a.get_figure() is fig

    def test_pchanged_sets_stale(self):
        a = Artist()
        a._stale = False
        a.pchanged()
        assert a.stale is True

    def test_stale_setter(self):
        a = Artist()
        a.stale = False
        assert a.stale is False
        a.stale = True
        assert a.stale is True


# ===================================================================
# Additional artist parametric tests (upstream-inspired batch)
# ===================================================================

import pytest
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


class TestArtistParametric:
    """Parametric tests for Artist base properties."""

    @pytest.mark.parametrize('alpha', [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_line_alpha_roundtrip(self, alpha):
        """Line2D alpha is settable and retrievable."""
        line = Line2D([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('zorder', [0, 1, 2, 3, 4, 5, 10, 100])
    def test_line_zorder_roundtrip(self, zorder):
        """Line2D zorder is settable and retrievable."""
        line = Line2D([0, 1], [0, 1])
        line.set_zorder(zorder)
        assert line.get_zorder() == zorder

    @pytest.mark.parametrize('visible', [True, False])
    def test_line_visible_roundtrip(self, visible):
        """Line2D visibility is settable."""
        line = Line2D([0, 1], [0, 1])
        line.set_visible(visible)
        assert line.get_visible() is visible

    @pytest.mark.parametrize('label', ['line_a', 'series_1', '', 'My Line'])
    def test_line_label_roundtrip(self, label):
        """Line2D label is settable and retrievable."""
        line = Line2D([0, 1], [0, 1], label=label)
        assert line.get_label() == label

    @pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
    def test_rect_alpha_roundtrip(self, alpha):
        """Rectangle alpha is settable."""
        r = Rectangle((0, 0), 1, 1)
        r.set_alpha(alpha)
        assert abs(r.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('zorder', [0, 1, 2, 5])
    def test_rect_zorder_roundtrip(self, zorder):
        """Rectangle zorder is settable."""
        r = Rectangle((0, 0), 1, 1)
        r.set_zorder(zorder)
        assert r.get_zorder() == zorder

    @pytest.mark.parametrize('visible', [True, False])
    def test_rect_visible_roundtrip(self, visible):
        """Rectangle visibility is settable."""
        r = Rectangle((0, 0), 1, 1)
        r.set_visible(visible)
        assert r.get_visible() is visible

    @pytest.mark.parametrize('gid', ['rect_1', 'shape_a', 'my_patch'])
    def test_artist_gid_roundtrip(self, gid):
        """Artist gid is settable."""
        r = Rectangle((0, 0), 1, 1)
        r.set_gid(gid)
        assert r.get_gid() == gid

    @pytest.mark.parametrize('url', [
        'https://example.com', 'http://test.org', '/local/path'
    ])
    def test_artist_url_roundtrip(self, url):
        """Artist URL is settable."""
        line = Line2D([0, 1], [0, 1])
        line.set_url(url)
        assert line.get_url() == url

    @pytest.mark.parametrize('clip', [True, False])
    def test_artist_clip_on_roundtrip(self, clip):
        """Artist clip_on is settable."""
        line = Line2D([0, 1], [0, 1])
        line.set_clip_on(clip)
        assert line.get_clip_on() == clip


# ===================================================================
# More parametric tests for artist (upstream-style)
# ===================================================================

class TestArtistUpstreamParametric:
    """Parametric tests for artist properties."""

    @pytest.mark.parametrize('alpha', [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_artist_alpha(self, alpha):
        """Artist alpha is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10

    @pytest.mark.parametrize('visible', [True, False])
    def test_artist_visible(self, visible):
        """Artist visibility is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_visible(visible)
        assert line.get_visible() == visible

    @pytest.mark.parametrize('zorder', [0, 1, 2, 5, 10, 100])
    def test_artist_zorder(self, zorder):
        """Artist zorder is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_zorder(zorder)
        assert line.get_zorder() == zorder

    @pytest.mark.parametrize('label', ['line1', '', 'series_a', 'my_data'])
    def test_artist_label(self, label):
        """Artist label is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1], label=label)
        assert line.get_label() == label

    @pytest.mark.parametrize('color', ['red', 'blue', 'green', '#ff0000', 'black'])
    def test_line_color(self, color):
        """Line2D color is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_color(color)
        assert line.get_color() == color

    @pytest.mark.parametrize('lw', [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_line_linewidth(self, lw):
        """Line2D linewidth is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_linewidth(lw)
        assert abs(line.get_linewidth() - lw) < 1e-10

    @pytest.mark.parametrize('marker', ['o', 's', '^', 'v', 'D', '+', 'x', '*'])
    def test_line_marker(self, marker):
        """Line2D marker is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_marker(marker)
        assert line.get_marker() == marker

    @pytest.mark.parametrize('clip', [True, False])
    def test_clip_on(self, clip):
        """Artist clip_on is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_clip_on(clip)
        assert line.get_clip_on() == clip

    @pytest.mark.parametrize('url', ['http://example.com', '', 'https://test.org'])
    def test_url(self, url):
        """Artist url is stored."""
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0, 1])
        line.set_url(url)
        assert line.get_url() == url
