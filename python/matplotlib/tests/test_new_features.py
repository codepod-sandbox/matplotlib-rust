"""
Tests for newly implemented features:
- Multi-line plot()
- LineCollection
- Artist.update() / properties() / findobj()
- Line2D improvements (markeredgecolor, markerfacecolor, etc.)
- Axes.get_lines(), add_line(), add_patch(), add_collection()
- Axes.relim(), autoscale(), autoscale_view()
- New patch types (Ellipse, Arc, FancyBboxPatch, FancyArrowPatch, etc.)
- Collection base class improvements
- Artist property getters/setters
- More pyplot wrappers
"""

import numpy as np
import pytest

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Patch, Rectangle, Circle, Polygon, Wedge,
    Ellipse, Arc, FancyBboxPatch, FancyArrowPatch, Arrow,
    RegularPolygon, PathPatch, ConnectionPatch,
)
from matplotlib.collections import (
    Collection, PathCollection, LineCollection, PolyCollection,
    EventCollection,
)
from matplotlib.text import Text, Annotation
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba


def _arr_eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def _color_close(a, b):
    return np.allclose(to_rgba(a), to_rgba(b))


def _colors_close(arr, color_list):
    """Check that each row in arr (RGBA) matches the corresponding color."""
    if len(arr) != len(color_list):
        return False
    return all(np.allclose(to_rgba(c), row) for c, row in
               zip(color_list, arr))


# ===========================================================================
# Artist.update() and related
# ===========================================================================

class TestArtistUpdate:
    def test_update_dict(self):
        a = Artist()
        a.update({'visible': False, 'alpha': 0.5, 'label': 'test'})
        assert a.get_visible() is False
        assert a.get_alpha() == 0.5
        assert a.get_label() == 'test'

    def test_update_none(self):
        a = Artist()
        try:
            a.update(None)  # OG may raise TypeError/AttributeError
        except (TypeError, AttributeError):
            pass
        assert a.get_visible() is True

    def test_properties(self):
        a = Artist()
        a.set_alpha(0.3)
        props = a.properties()
        assert 'alpha' in props
        assert props['alpha'] == 0.3
        assert 'visible' in props
        assert props['visible'] is True

    def test_findobj_all(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.plot([5, 6], [7, 8])
        found = ax.findobj(Line2D)
        assert len(found) >= 2  # OG may find more (tick lines)

    def test_findobj_none(self):
        a = Artist()
        found = a.findobj()
        assert a in found

    def test_findobj_callable(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='visible')
        ax.plot([5, 6], [7, 8], label='_hidden')
        found = ax.findobj(lambda a: isinstance(a, Line2D)
                           and not a.get_label().startswith('_'))
        assert len(found) >= 1  # at least the 'visible' line


class TestArtistProperties:
    def test_clip_on(self):
        a = Artist()
        assert a.get_clip_on() is True
        a.set_clip_on(False)
        assert a.get_clip_on() is False

    def test_clip_box(self):
        a = Artist()
        assert a.get_clip_box() is None

    def test_clip_path(self):
        a = Artist()
        assert a.get_clip_path() is None

    def test_transform(self):
        a = Artist()
        # OG returns IdentityTransform by default, not None
        assert not a.is_transform_set()

    def test_animated(self):
        a = Artist()
        assert a.get_animated() is False
        a.set_animated(True)
        assert a.get_animated() is True

    def test_rasterized(self):
        a = Artist()
        assert not a.get_rasterized()  # None or False initially
        with pytest.warns(UserWarning, match="Rasterization of .* will be ignored"):
            a.set_rasterized(True)
        assert a.get_rasterized() is True

    def test_sketch_params(self):
        a = Artist()
        assert a.get_sketch_params() is None
        a.set_sketch_params(1.0, 128, 16)
        assert a.get_sketch_params() == (1.0, 128, 16)
        a.set_sketch_params()  # reset
        assert a.get_sketch_params() is None

    def test_snap(self):
        a = Artist()
        assert a.get_snap() is None
        a.set_snap(True)
        assert a.get_snap() is True

    def test_path_effects(self):
        a = Artist()
        assert a.get_path_effects() == []
        a.set_path_effects([1, 2, 3])
        assert a.get_path_effects() == [1, 2, 3]

    def test_url(self):
        a = Artist()
        assert a.get_url() is None
        a.set_url('http://example.com')
        assert a.get_url() == 'http://example.com'

    def test_gid(self):
        a = Artist()
        assert a.get_gid() is None
        a.set_gid('myid')
        assert a.get_gid() == 'myid'

    def test_in_layout(self):
        a = Artist()
        assert a.get_in_layout() is True
        a.set_in_layout(False)
        assert a.get_in_layout() is False

    def test_picker(self):
        a = Artist()
        # OG may return False not None initially
        assert not a.get_picker()
        assert a.pickable() is False
        a.set_picker(True)
        assert a.get_picker() is True
        # OG pickable() requires a figure reference; check picker was set
        assert a.get_picker() is True

    def test_agg_filter(self):
        a = Artist()
        assert a.get_agg_filter() is None
        a.set_agg_filter(lambda x: x)
        assert a.get_agg_filter() is not None

    def test_figure_property(self):
        a = Artist()
        assert a.get_figure() is None
        fig = Figure()
        a.set_figure(fig)
        assert a.get_figure() is fig

    def test_stale_property(self):
        a = Artist()
        a.stale = False
        assert a.stale is False

    def test_contains(self):
        a = Artist()
        assert a.get_contains() is None
        fn = lambda *args: True
        a.set_contains(fn)
        assert a.get_contains() is fn


# ===========================================================================
# Line2D improvements
# ===========================================================================

class TestLine2DNewProperties:
    def test_markeredgecolor(self):
        line = Line2D([1, 2], [3, 4], markeredgecolor='red')
        assert line.get_markeredgecolor() == 'red'

    def test_markeredgecolor_default(self):
        line = Line2D([1, 2], [3, 4], color='blue')
        # Default mec should be line color
        assert line.get_markeredgecolor() == 'blue'

    def test_set_markeredgecolor(self):
        line = Line2D([1, 2], [3, 4])
        line.set_markeredgecolor('green')
        assert line.get_markeredgecolor() == 'green'

    def test_markerfacecolor(self):
        line = Line2D([1, 2], [3, 4], markerfacecolor='red')
        assert line.get_markerfacecolor() == 'red'

    def test_markerfacecolor_default(self):
        line = Line2D([1, 2], [3, 4], color='blue')
        assert line.get_markerfacecolor() == 'blue'

    def test_set_markerfacecolor(self):
        line = Line2D([1, 2], [3, 4])
        line.set_markerfacecolor('green')
        assert line.get_markerfacecolor() == 'green'

    def test_markeredgewidth(self):
        line = Line2D([1, 2], [3, 4], markeredgewidth=2.0)
        assert line.get_markeredgewidth() == 2.0

    def test_markeredgewidth_default(self):
        line = Line2D([1, 2], [3, 4])
        assert line.get_markeredgewidth() == matplotlib.rcParams['lines.markeredgewidth']

    def test_markevery(self):
        line = Line2D([1, 2], [3, 4])
        assert line.get_markevery() is None
        line.set_markevery(5)
        assert line.get_markevery() == 5

    def test_antialiased(self):
        line = Line2D([1, 2], [3, 4])
        assert line.get_antialiased() is True
        line.set_antialiased(False)
        assert line.get_antialiased() is False

    def test_set_antialiased_alias(self):
        line = Line2D([1, 2], [3, 4])
        line.set_aa(False)
        assert line.get_antialiased() is False

    def test_solid_capstyle(self):
        line = Line2D([1, 2], [3, 4], solid_capstyle='round')
        assert line.get_solid_capstyle() == 'round'

    def test_solid_joinstyle(self):
        line = Line2D([1, 2], [3, 4], solid_joinstyle='round')
        assert line.get_solid_joinstyle() == 'round'

    def test_dash_capstyle(self):
        line = Line2D([1, 2], [3, 4], dash_capstyle='round')
        assert line.get_dash_capstyle() == 'round'

    def test_dash_joinstyle(self):
        line = Line2D([1, 2], [3, 4], dash_joinstyle='round')
        assert line.get_dash_joinstyle() == 'round'

    def test_mec_alias(self):
        line = Line2D([1, 2], [3, 4])
        line.set_mec('red')
        assert line.get_markeredgecolor() == 'red'

    def test_mfc_alias(self):
        line = Line2D([1, 2], [3, 4])
        line.set_mfc('red')
        assert line.get_markerfacecolor() == 'red'

    def test_mew_alias(self):
        line = Line2D([1, 2], [3, 4])
        line.set_mew(3.0)
        assert line.get_markeredgewidth() == 3.0

    def test_markerfacecoloralt(self):
        line = Line2D([1, 2], [3, 4])
        assert line.get_markerfacecoloralt() == 'none'
        line.set_markerfacecoloralt('red')
        assert line.get_markerfacecoloralt() == 'red'

    def test_get_path(self):
        line = Line2D([1, 2, 3], [4, 5, 6])
        path = line.get_path()
        assert len(path) == 3

    def test_get_xydata(self):
        import numpy as np
        line = Line2D([1, 2, 3], [4, 5, 6])
        xydata = line.get_xydata()
        assert isinstance(xydata, np.ndarray)
        assert xydata.shape == (3, 2)
        assert np.allclose(xydata, [[1, 4], [2, 5], [3, 6]])

    def test_line2d_update(self):
        line = Line2D([1, 2], [3, 4])
        line.update({'color': 'red', 'linewidth': 3.0})
        assert line.get_color() == 'red'
        assert line.get_linewidth() == 3.0


class TestLine2DClassAttributes:
    def test_lineStyles(self):
        assert '-' in Line2D.lineStyles
        assert '--' in Line2D.lineStyles
        assert ':' in Line2D.lineStyles
        assert '-.' in Line2D.lineStyles
        assert 'None' in Line2D.lineStyles

    def test_markers(self):
        assert 'o' in Line2D.markers
        assert 's' in Line2D.markers
        assert '^' in Line2D.markers

    def test_filled_markers(self):
        assert 'o' in Line2D.filled_markers
        assert 's' in Line2D.filled_markers
        assert isinstance(Line2D.filled_markers, (frozenset, tuple, list))

    def test_module_lineStyles(self):
        from matplotlib.lines import lineStyles
        assert '-' in lineStyles

    def test_module_lineMarkers(self):
        from matplotlib.lines import lineMarkers
        assert 'o' in lineMarkers


# ===========================================================================
# Multi-line plot()
# ===========================================================================

class TestMultiLinePlot:
    def test_plot_two_groups(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], 'r-', [5, 6], [7, 8], 'b--')
        assert len(lines) == 2

    def test_plot_two_groups_no_fmt(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], [5, 6], [7, 8])
        assert len(lines) == 2

    def test_plot_three_groups(self):
        fig, ax = plt.subplots()
        lines = ax.plot(
            [1, 2], [3, 4], 'r-',
            [5, 6], [7, 8], 'g--',
            [9, 10], [11, 12], 'b:',
        )
        assert len(lines) == 3

    def test_plot_single_still_works(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3])
        assert len(lines) == 1

    def test_plot_xy_still_works(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4])
        assert len(lines) == 1

    def test_plot_xy_fmt_still_works(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], 'r--')
        assert len(lines) == 1


# ===========================================================================
# LineCollection
# ===========================================================================

class TestLineCollection:
    def test_create_empty(self):
        lc = LineCollection()
        assert lc.get_segments() == []

    def test_create_with_segments(self):
        segs = [[(0, 0), (1, 1)], [(2, 2), (3, 3)]]
        lc = LineCollection(segs)
        assert len(lc.get_segments()) == 2

    def test_set_segments(self):
        lc = LineCollection()
        lc.set_segments([[(0, 0), (1, 1)]])
        assert len(lc.get_segments()) == 1

    def test_set_verts(self):
        lc = LineCollection()
        lc.set_verts([[(0, 0), (1, 1)]])
        assert len(lc.get_segments()) == 1

    def test_get_color(self):
        lc = LineCollection(colors='red')
        assert _colors_close(lc.get_color(), ['red'])

    def test_set_color(self):
        lc = LineCollection()
        lc.set_color('blue')
        assert _colors_close(lc.get_color(), ['blue'])

    def test_set_colors_list(self):
        lc = LineCollection()
        lc.set_colors(['red', 'blue'])
        colors = lc.get_colors()
        assert len(colors) == 2
        # get_colors returns RGBA tuples
        assert len(colors[0]) == 4
        assert len(colors[1]) == 4

    def test_linewidths(self):
        lc = LineCollection(linewidths=[2.0, 3.0])
        assert _arr_eq(lc.get_linewidths(), [2.0, 3.0])

    def test_label(self):
        lc = LineCollection(label='test')
        assert lc.get_label() == 'test'

    def test_visible(self):
        lc = LineCollection()
        assert lc.get_visible() is True
        lc.set_visible(False)
        assert lc.get_visible() is False

    def test_alpha(self):
        lc = LineCollection()
        lc.set_alpha(0.5)
        assert lc.get_alpha() == 0.5

    def test_zorder(self):
        lc = LineCollection()
        assert lc.get_zorder() == 2

    def test_add_to_axes(self):
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]])
        ax.add_collection(lc)
        assert lc in ax.collections

    def test_linestyles(self):
        lc = LineCollection(linestyles='dashed')
        # OG stores as (offset, dash) tuples, not plain strings
        assert len(lc.get_linestyles()) >= 1


class TestPolyCollection:
    def test_create_empty(self):
        pc = PolyCollection([])  # OG requires verts argument
        assert len(pc.get_verts()) == 0

    def test_create_with_verts(self):
        verts = [[(0, 0), (1, 0), (1, 1), (0, 1)]]
        pc = PolyCollection(verts)
        assert len(pc.get_verts()) == 1

    def test_set_verts(self):
        pc = PolyCollection([])  # OG requires verts argument
        pc.set_verts([[(0, 0), (1, 1), (2, 0)]])
        assert len(pc.get_verts()) == 1


class TestEventCollection:
    def test_create(self):
        ec = EventCollection([1, 2, 3])
        assert ec.get_positions() == [1, 2, 3]

    def test_set_positions(self):
        ec = EventCollection()
        ec.set_positions([4, 5])
        assert ec.get_positions() == [4, 5]

    def test_orientation(self):
        ec = EventCollection(orientation='vertical')
        assert ec.get_orientation() == 'vertical'

    def test_lineoffset(self):
        ec = EventCollection(lineoffset=2)
        assert ec.get_lineoffset() == 2

    def test_linelength(self):
        ec = EventCollection(linelength=0.5)
        assert ec.get_linelength() == 0.5


# ===========================================================================
# Collection base class
# ===========================================================================

class TestCollection:
    def test_set_edgecolor(self):
        c = Collection()
        c.set_edgecolor('red')
        assert _colors_close(c.get_edgecolor(), ['red'])

    def test_set_facecolor(self):
        c = Collection()
        c.set_facecolor('blue')
        assert _colors_close(c.get_facecolor(), ['blue'])

    def test_set_linewidth(self):
        c = Collection()
        c.set_linewidth(2.0)
        assert _arr_eq(c.get_linewidth(), [2.0])

    def test_set_linewidths_list(self):
        c = Collection()
        c.set_linewidths([1.0, 2.0])
        assert _arr_eq(c.get_linewidths(), [1.0, 2.0])

    def test_set_linestyle(self):
        c = Collection()
        c.set_linestyle('dashed')
        # OG stores as list of (offset, dash) tuples; just check length
        assert len(c.get_linestyle()) >= 1

    def test_get_array(self):
        import numpy as np
        c = Collection()
        assert c.get_array() is None
        c.set_array([1, 2, 3])
        np.testing.assert_array_equal(c.get_array(), [1, 2, 3])

    def test_get_paths(self):
        c = Collection()
        # OG Collection.get_paths() may return None before paths are set
        paths = c.get_paths()
        assert paths is None or paths == []

    def test_set_color(self):
        c = Collection()
        c.set_color('red')
        assert _colors_close(c.get_facecolor(), ['red'])
        assert _colors_close(c.get_edgecolor(), ['red'])

    def test_set_edgecolor_none(self):
        c = Collection()
        c.set_edgecolor(None)
        assert c.get_edgecolor() is not None

    def test_set_facecolor_none(self):
        c = Collection()
        c.set_facecolor(None)
        fc = c.get_facecolor()
        assert fc is not None


# ===========================================================================
# New Patch types
# ===========================================================================

class TestEllipse:
    def test_create(self):
        e = Ellipse((1, 2), 3, 4)
        assert np.allclose(e.get_center(), (1, 2))
        assert e.get_width() == 3
        assert e.get_height() == 4

    def test_set_center(self):
        e = Ellipse((0, 0), 1, 1)
        e.set_center((5, 6))
        assert np.allclose(e.get_center(), (5, 6))

    def test_set_width(self):
        e = Ellipse((0, 0), 1, 1)
        e.set_width(10)
        assert e.get_width() == 10

    def test_set_height(self):
        e = Ellipse((0, 0), 1, 1)
        e.set_height(10)
        assert e.get_height() == 10

    def test_angle(self):
        e = Ellipse((0, 0), 1, 1, angle=45)
        assert e.get_angle() == 45
        e.set_angle(90)
        assert e.get_angle() == 90

    def test_facecolor(self):
        e = Ellipse((0, 0), 1, 1, facecolor='red')
        fc = e.get_facecolor()
        assert fc[0] == 1.0  # red channel

    def test_edgecolor(self):
        e = Ellipse((0, 0), 1, 1, edgecolor='blue')
        ec = e.get_edgecolor()
        assert ec[2] == 1.0  # blue channel

    def test_is_patch(self):
        e = Ellipse((0, 0), 1, 1)
        assert isinstance(e, Patch)


class TestArc:
    def test_create(self):
        a = Arc((1, 2), 3, 4, angle=0, theta1=0, theta2=180)
        assert np.allclose(a.get_center(), (1, 2))
        assert a.get_width() == 3
        assert a.get_theta1() == 0
        assert a.get_theta2() == 180

    def test_is_ellipse(self):
        a = Arc((0, 0), 1, 1)
        assert isinstance(a, Ellipse)

    def test_set_theta(self):
        a = Arc((0, 0), 1, 1)
        a.set_theta1(45)
        a.set_theta2(270)
        assert a.get_theta1() == 45
        assert a.get_theta2() == 270


class TestFancyBboxPatch:
    def test_create(self):
        p = FancyBboxPatch((1, 2), 3, 4)
        assert np.allclose(p.get_xy(), (1, 2))
        assert p.get_width() == 3
        assert p.get_height() == 4

    def test_boxstyle(self):
        p = FancyBboxPatch((0, 0), 1, 1, boxstyle='round')
        # OG returns BoxStyle object, check name
        bs = p.get_boxstyle()
        assert type(bs).__name__.lower() in ('round', 'boxstylebase') or 'round' in str(bs).lower()
        p.set_boxstyle('square')
        bs = p.get_boxstyle()
        assert 'square' in str(bs).lower() or type(bs).__name__.lower() == 'square'

    def test_set_xy(self):
        p = FancyBboxPatch((0, 0), 1, 1)
        p.set_xy((5, 6))
        assert np.allclose(p.get_xy(), (5, 6))

    def test_is_patch(self):
        p = FancyBboxPatch((0, 0), 1, 1)
        assert isinstance(p, Patch)


class TestFancyArrowPatch:
    def test_create(self):
        p = FancyArrowPatch(posA=(0, 0), posB=(1, 1))
        assert isinstance(p, Patch)

    def test_arrowstyle(self):
        p = FancyArrowPatch(posA=(0, 0), posB=(1, 1), arrowstyle='->')
        # OG returns ArrowStyle object; '->' maps to CurveB
        style = p.get_arrowstyle()
        name = type(style).__name__
        assert '->' in str(style) or name in ('Simple', 'CurveB')

    def test_mutation_scale(self):
        p = FancyArrowPatch(posA=(0, 0), posB=(1, 1), mutation_scale=20)
        assert p.get_mutation_scale() == 20

    def test_set_positions(self):
        p = FancyArrowPatch(posA=(0, 0), posB=(1, 1))
        p.set_positions((0, 0), (1, 1))
        # OG stores positions in _posA_posB, not _posA/_posB
        assert np.allclose(p._posA_posB[0], (0, 0))
        assert np.allclose(p._posA_posB[1], (1, 1))


class TestArrow:
    def test_create(self):
        a = Arrow(0, 0, 1, 1)
        assert isinstance(a, Patch)

    def test_width(self):
        a = Arrow(0, 0, 1, 1, width=2.0)
        # OG stores width as _width
        assert hasattr(a, '_arrow_width') or hasattr(a, 'width') or hasattr(a, '_width')


class TestRegularPolygon:
    def test_create(self):
        p = RegularPolygon((0, 0), 6, radius=1.0)
        assert p.numvertices == 6
        assert np.allclose(p.xy, (0, 0))

    def test_is_patch(self):
        p = RegularPolygon((0, 0), 5)
        assert isinstance(p, Patch)


class TestPathPatch:
    def test_create(self):
        p = PathPatch('some_path')
        assert p.get_path() == 'some_path'

    def test_set_path(self):
        p = PathPatch('a')
        p.set_path('b')
        assert p.get_path() == 'b'


class TestConnectionPatch:
    def test_create(self):
        p = ConnectionPatch((0, 0), (1, 1), 'data', 'data')
        assert isinstance(p, FancyArrowPatch)


# ===========================================================================
# Axes improvements
# ===========================================================================

class TestAxesGetLines:
    def test_get_lines_empty(self):
        fig, ax = plt.subplots()
        assert ax.get_lines() == []

    def test_get_lines_after_plot(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.plot([5, 6], [7, 8])
        lines = ax.get_lines()
        assert len(lines) == 2
        assert all(isinstance(l, Line2D) for l in lines)

    def test_get_lines_copy(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        lines = ax.get_lines()
        lines.append('extra')
        assert len(ax.get_lines()) == 1

    def test_get_images_empty(self):
        fig, ax = plt.subplots()
        assert ax.get_images() == []


class TestAxesAddMethods:
    def test_add_line(self):
        fig, ax = plt.subplots()
        line = Line2D([1, 2], [3, 4])
        ax.add_line(line)
        assert line in ax.lines
        assert line.axes is ax
        assert line.figure is fig

    def test_add_patch(self):
        fig, ax = plt.subplots()
        rect = Rectangle((0, 0), 1, 1)
        ax.add_patch(rect)
        assert rect in ax.patches
        assert rect.axes is ax

    def test_add_collection(self):
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]])
        ax.add_collection(lc)
        assert lc in ax.collections
        assert lc.axes is ax

    def test_add_container(self):
        from matplotlib.container import BarContainer
        fig, ax = plt.subplots()
        bc = BarContainer([])
        ax.add_container(bc)
        assert bc in ax.containers


class TestAxesRelimAutoscale:
    def test_relim(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.relim()  # should not raise

    def test_autoscale(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        assert ax.get_xlim() == (0, 10)
        ax.plot([1, 2], [3, 4])
        ax.autoscale()
        # After autoscale, limits should be auto-computed (OG adds ~5% margins)
        xlim = ax.get_xlim()
        assert xlim[0] <= 1 and xlim[1] >= 2

    def test_autoscale_axis_x(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 5)
        ax.plot([1, 2], [3, 4])
        ax.autoscale(axis='x')
        xlim = ax.get_xlim()
        assert xlim[0] <= 1 and xlim[1] >= 2
        assert ax.get_ylim() == (-5, 5)

    def test_autoscale_axis_y(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 5)
        ax.plot([1, 2], [3, 4])
        ax.autoscale(axis='y')
        assert ax.get_xlim() == (0, 10)
        ylim = ax.get_ylim()
        assert ylim[0] <= 3 + 1e-12
        assert ylim[1] >= 4 - 1e-12

    def test_autoscale_view(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.plot([1, 2], [3, 4])
        ax.autoscale_view()
        xlim = ax.get_xlim()
        assert xlim[0] <= 1 and xlim[1] >= 2

    def test_has_data_true(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        assert ax.has_data() is True

    def test_has_data_false(self):
        fig, ax = plt.subplots()
        assert ax.has_data() is False


class TestAxesProperties:
    def test_set_visible(self):
        fig, ax = plt.subplots()
        assert ax.get_visible() is True
        ax.set_visible(False)
        assert ax.get_visible() is False

    def test_format_coord(self):
        fig, ax = plt.subplots()
        s = ax.format_coord(1.5, 2.5)
        # OG returns '(x, y) = (1.500, 2.500)' format
        assert '1.5' in s or '1.50' in s
        assert '2.5' in s or '2.50' in s

    def test_frame_on(self):
        fig, ax = plt.subplots()
        assert ax.get_frame_on() is True
        ax.set_frame_on(False)
        assert ax.get_frame_on() is False

    def test_axisbelow(self):
        fig, ax = plt.subplots()
        ax.set_axisbelow(False)
        assert ax.get_axisbelow() is False

    def test_navigate_mode(self):
        fig, ax = plt.subplots()
        assert ax.get_navigate_mode() is None
        ax.set_navigate_mode('ZOOM')
        assert ax.get_navigate_mode() == 'ZOOM'

    def test_can_pan(self):
        fig, ax = plt.subplots()
        assert ax.can_pan() is True

    def test_can_zoom(self):
        fig, ax = plt.subplots()
        assert ax.can_zoom() is True

    def test_minorticks_on_off(self):
        fig, ax = plt.subplots()
        ax.minorticks_on()  # no-op, should not raise
        ax.minorticks_off()  # no-op, should not raise


# ===========================================================================
# Figure improvements
# ===========================================================================

class TestFigureSubplots:
    def test_figure_subplots_single(self):
        fig = Figure()
        ax = fig.subplots()
        assert ax is not None
        assert len(fig.axes) == 1

    def test_figure_subplots_grid(self):
        fig = Figure()
        axes = fig.subplots(2, 2)
        assert len(axes) == 2
        assert len(axes[0]) == 2

    def test_figure_subplots_1xN(self):
        fig = Figure()
        axes = fig.subplots(1, 3)
        assert len(axes) == 3

    def test_figure_subplots_Nx1(self):
        fig = Figure()
        axes = fig.subplots(3, 1)
        assert len(axes) == 3


class TestFigureProperties:
    def test_get_children(self):
        fig = Figure()
        fig.add_subplot(1, 1, 1)
        children = fig.get_children()
        assert len(children) >= 1

    def test_constrained_layout(self):
        fig = Figure()
        assert fig.get_constrained_layout() is False
        with pytest.warns(PendingDeprecationWarning, match="set_constrained_layout"):
            fig.set_constrained_layout(True)
        assert fig.get_constrained_layout() is True

    def test_tight_layout_prop(self):
        fig = Figure()
        assert fig.get_tight_layout() is False
        with pytest.warns(PendingDeprecationWarning, match="set_tight_layout"):
            fig.set_tight_layout(True)
        assert fig.get_tight_layout() is True

    def test_align_labels(self):
        fig = Figure()
        fig.align_xlabels()
        fig.align_ylabels()
        fig.align_labels()

    def test_colorbar(self):
        fig = Figure()
        ax = fig.add_subplot()
        import matplotlib.cm as cm
        import numpy as np
        sm = cm.ScalarMappable(cmap='viridis')
        sm.set_array(np.linspace(0, 1, 10))
        cb = fig.colorbar(sm, ax=ax)
        assert cb is not None

    def test_add_gridspec(self):
        fig = Figure()
        gs = fig.add_gridspec(2, 3)
        assert gs.nrows == 2
        assert gs.ncols == 3

    def test_layout_engine(self):
        fig = Figure()
        assert fig.get_layout_engine() is None
        fig.set_layout_engine('constrained')
        # OG returns ConstrainedLayoutEngine object, not string
        engine = fig.get_layout_engine()
        assert engine is not None
        assert 'constrained' in type(engine).__name__.lower()


# ===========================================================================
# More pyplot wrappers
# ===========================================================================

class TestPyplotNewWrappers:
    def test_fill_betweenx(self):
        plt.close('all')
        poly = plt.fill_betweenx([0, 1, 2], [0, 1, 0])
        assert poly is not None

    def test_annotate(self):
        plt.close('all')
        ann = plt.annotate('test', (0, 0), xytext=(1, 1))
        assert ann is not None

    def test_axis(self):
        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        result = plt.axis()
        assert len(result) == 4

    def test_twinx(self):
        plt.close('all')
        fig, ax = plt.subplots()
        ax2 = plt.twinx()
        assert ax2 is not None
        assert ax2 is not ax

    def test_twiny(self):
        plt.close('all')
        fig, ax = plt.subplots()
        ax2 = plt.twiny()
        assert ax2 is not None

    def test_xscale(self):
        plt.close('all')
        fig, ax = plt.subplots()
        plt.xscale('log')
        assert ax.get_xscale() == 'log'

    def test_yscale(self):
        plt.close('all')
        fig, ax = plt.subplots()
        plt.yscale('log')
        assert ax.get_yscale() == 'log'

    def test_tight_layout(self):
        plt.close('all')
        fig, ax = plt.subplots()
        plt.tight_layout()

    def test_figtext(self):
        plt.close('all')
        fig, ax = plt.subplots()
        t = plt.figtext(0.5, 0.5, 'hello')
        assert t is not None

    def test_rc(self):
        plt.close('all')
        old = matplotlib.rcParams['lines.linewidth']
        plt.rc('lines', linewidth=3.0)
        assert matplotlib.rcParams['lines.linewidth'] == 3.0
        matplotlib.rcParams['lines.linewidth'] = old  # restore

    def test_subplot_mosaic(self):
        plt.close('all')
        fig, axes_dict = plt.subplot_mosaic("AB\nCC")
        assert 'A' in axes_dict
        assert 'B' in axes_dict
        assert 'C' in axes_dict


# ===========================================================================
# Upstream-style tests that are now possible
# ===========================================================================

class TestUpstreamAxesMultiPlot:
    """Tests inspired by upstream test_axes.py that use multi-line plot."""

    def test_plot_multiple_lines_different_colors(self):
        fig, ax = plt.subplots()
        lines = ax.plot([0, 1], [0, 1], 'r', [0, 1], [1, 0], 'b')
        assert len(lines) == 2

    def test_plot_returns_correct_count(self):
        fig, ax = plt.subplots()
        lines = ax.plot([0, 1], [0, 1], 'r-',
                        [0, 1], [1, 2], 'g--',
                        [0, 1], [2, 3], 'b:')
        assert len(lines) == 3


class TestUpstreamLineCollection:
    """Tests inspired by upstream test_collections.py for LineCollection."""

    def test_linecollection_basic(self):
        fig, ax = plt.subplots()
        segments = [[(0, 0), (1, 1)], [(1, 0), (0, 1)]]
        lc = LineCollection(segments)
        ax.add_collection(lc)
        assert lc in ax.collections

    def test_linecollection_set_segments(self):
        lc = LineCollection([[(0, 0), (1, 1)]])
        lc.set_segments([[(2, 2), (3, 3)], [(4, 4), (5, 5)]])
        assert len(lc.get_segments()) == 2

    def test_linecollection_set_linewidth(self):
        lc = LineCollection([[(0, 0), (1, 1)]])
        lc.set_linewidth([2.0])
        assert _arr_eq(lc.get_linewidths(), [2.0])

    def test_linecollection_set_color(self):
        lc = LineCollection([[(0, 0), (1, 1)]])
        lc.set_color(['red', 'blue'])
        assert _colors_close(lc.get_color(), ['red', 'blue'])

    def test_linecollection_remove(self):
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)]])
        ax.add_collection(lc)
        assert lc in ax.collections
        lc.remove()
        assert lc not in ax.collections


class TestUpstreamArtistUpdate:
    """Tests inspired by upstream test_artist.py for update/properties."""

    def test_artist_update_properties(self):
        a = Artist()
        a.update({'visible': False, 'alpha': 0.3})
        assert a.get_visible() is False
        assert a.get_alpha() == 0.3

    def test_line2d_update_properties(self):
        line = Line2D([0, 1], [0, 1])
        line.update({'linewidth': 5.0, 'linestyle': '--'})
        assert line.get_linewidth() == 5.0
        assert line.get_linestyle() == '--'

    def test_rectangle_update(self):
        rect = Rectangle((0, 0), 1, 1)
        rect.update({'facecolor': 'red', 'edgecolor': 'blue'})
        fc = rect.get_facecolor()
        assert fc[0] == 1.0  # red

    def test_text_update(self):
        t = Text(0, 0, 'hello')
        t.update({'fontsize': 20})
        assert t.get_fontsize() == 20


class TestUpstreamEllipse:
    """Tests inspired by upstream test_patches.py for Ellipse."""

    def test_ellipse_default_angle(self):
        e = Ellipse((0, 0), 1, 1)
        assert e.get_angle() == 0

    def test_ellipse_properties(self):
        e = Ellipse((1, 2), width=3, height=4, angle=45)
        assert e.get_center() == (1, 2)
        assert e.get_width() == 3
        assert e.get_height() == 4
        assert e.get_angle() == 45

    def test_ellipse_set(self):
        e = Ellipse((0, 0), 1, 1)
        e.set_width(5)
        e.set_height(10)
        e.set_angle(90)
        assert e.get_width() == 5
        assert e.get_height() == 10
        assert e.get_angle() == 90


class TestUpstreamLine2DProperties:
    """Tests for Line2D properties from upstream test_lines.py."""

    def test_markeredgecolor_inherits(self):
        """mec defaults to line color."""
        line = Line2D([0, 1], [0, 1], color='red')
        assert line.get_markeredgecolor() == 'red'

    def test_markeredgecolor_explicit(self):
        """Explicit mec overrides default."""
        line = Line2D([0, 1], [0, 1], color='red', markeredgecolor='blue')
        assert line.get_markeredgecolor() == 'blue'

    def test_markerfacecolor_inherits(self):
        line = Line2D([0, 1], [0, 1], color='red')
        assert line.get_markerfacecolor() == 'red'

    def test_markerfacecolor_explicit(self):
        line = Line2D([0, 1], [0, 1], color='red', markerfacecolor='green')
        assert line.get_markerfacecolor() == 'green'

    def test_line_fillstyle(self):
        line = Line2D([0, 1], [0, 1])
        assert line.get_fillstyle() == 'full'
        line.set_fillstyle('left')
        assert line.get_fillstyle() == 'left'

    def test_line_update_via_set(self):
        line = Line2D([0, 1], [0, 1])
        line.set(color='red', linewidth=3.0, linestyle='--')
        assert line.get_color() == 'red'
        assert line.get_linewidth() == 3.0
        assert line.get_linestyle() == '--'


class TestUpstreamAxesMethods:
    """Tests for various Axes methods from upstream."""

    def test_cla_resets_lines(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        assert len(ax.get_lines()) == 1
        ax.cla()
        assert len(ax.get_lines()) == 0

    def test_clear_resets_lines(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.clear()
        assert len(ax.get_lines()) == 0

    def test_axes_findobj_lines(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.plot([5, 6], [7, 8])
        # OG findobj traverses all children recursively (includes spine lines etc)
        found = ax.findobj(Line2D)
        assert len(found) >= 2

    def test_axes_findobj_rectangles(self):
        fig, ax = plt.subplots()
        ax.bar([1, 2], [3, 4])
        found = ax.findobj(Rectangle)
        assert len(found) >= 2

    def test_axes_findobj_text(self):
        fig, ax = plt.subplots()
        ax.text(0, 0, 'hello')
        found = ax.findobj(Text)
        assert len(found) >= 1

    def test_get_children(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.bar([1, 2], [3, 4])
        children = ax.get_children()
        assert len(children) >= 3  # at least 1 line + 2 rects

    def test_set_kwargs(self):
        fig, ax = plt.subplots()
        ax.set(xlabel='X', ylabel='Y', title='T', xlim=(0, 10))
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        assert ax.get_title() == 'T'
        assert ax.get_xlim() == (0, 10)


# ===========================================================================
# Patch properties
# ===========================================================================

class TestPatchProperties:
    def test_patch_linestyle(self):
        p = Patch()
        assert p.get_linestyle() == 'solid'
        p.set_linestyle('dashed')
        assert p.get_linestyle() == 'dashed'

    def test_patch_antialiased(self):
        p = Patch()
        assert p.get_antialiased() is True
        p.set_antialiased(False)
        assert p.get_antialiased() is False

    def test_patch_update(self):
        p = Patch()
        p.update({'facecolor': 'red', 'edgecolor': 'blue', 'linewidth': 3.0})
        fc = p.get_facecolor()
        assert fc[0] == 1.0  # red
        ec = p.get_edgecolor()
        assert ec[2] == 1.0  # blue
        assert p.get_linewidth() == 3.0

    def test_rectangle_set(self):
        r = Rectangle((0, 0), 1, 1)
        r.set(facecolor='green', edgecolor='red')
        fc = r.get_facecolor()
        assert fc[1] > 0  # green channel

    def test_circle_update(self):
        c = Circle((0, 0), 1.0)
        c.update({'facecolor': 'blue'})
        fc = c.get_facecolor()
        assert fc[2] == 1.0  # blue


# ===========================================================================
# GridSpec improvements
# ===========================================================================

class TestGridSpec:
    def test_gridspec_figure_attr(self):
        from matplotlib.gridspec import GridSpec
        fig = Figure()
        gs = GridSpec(2, 3, figure=fig)
        assert gs.figure is fig

    def test_gridspec_hspace(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, hspace=0.5)
        assert gs._hspace == 0.5

    def test_gridspec_wspace(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, wspace=0.3)
        assert gs._wspace == 0.3

    def test_gridspec_ratios(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[3, 1])
        # OG uses get_width_ratios/get_height_ratios (not private attrs)
        assert gs.get_width_ratios() == [1, 2]
        assert gs.get_height_ratios() == [3, 1]


# ===========================================================================
# Bulk property tests that match upstream patterns
# ===========================================================================

class TestBulkProperties:
    """These test patterns appear frequently in upstream matplotlib."""

    @pytest.mark.parametrize('prop,val', [
        ('visible', False),
        ('alpha', 0.5),
        ('label', 'test'),
        ('zorder', 5),
    ])
    def test_artist_set_get(self, prop, val):
        a = Artist()
        getattr(a, f'set_{prop}')(val)
        assert getattr(a, f'get_{prop}')() == val

    @pytest.mark.parametrize('prop,val', [
        ('color', 'red'),
        ('linewidth', 3.0),
        ('linestyle', '--'),
        ('marker', 'o'),
        ('markersize', 10.0),
        ('fillstyle', 'left'),
        ('markeredgecolor', 'green'),
        ('markerfacecolor', 'blue'),
        ('markeredgewidth', 2.5),
        ('antialiased', False),
    ])
    def test_line2d_set_get(self, prop, val):
        line = Line2D([0, 1], [0, 1])
        getattr(line, f'set_{prop}')(val)
        assert getattr(line, f'get_{prop}')() == val

    @pytest.mark.parametrize('prop,val', [
        ('linewidth', 3.0),
        ('linestyle', 'dashed'),
        ('antialiased', False),
    ])
    def test_patch_set_get(self, prop, val):
        p = Patch()
        getattr(p, f'set_{prop}')(val)
        assert getattr(p, f'get_{prop}')() == val

    @pytest.mark.parametrize('prop,val', [
        ('text', 'hello'),
        ('fontsize', 20.0),
        ('fontweight', 'bold'),
    ])
    def test_text_set_get(self, prop, val):
        t = Text()
        getattr(t, f'set_{prop}')(val)
        assert getattr(t, f'get_{prop}')() == val

    @pytest.mark.parametrize('prop,val', [
        ('xlabel', 'X'),
        ('ylabel', 'Y'),
        ('title', 'T'),
        ('xscale', 'log'),
        ('yscale', 'log'),
        ('aspect', 'equal'),
        ('facecolor', 'red'),
    ])
    def test_axes_set_get(self, prop, val):
        fig, ax = plt.subplots()
        getattr(ax, f'set_{prop}')(val)
        result = getattr(ax, f'get_{prop}')()
        if prop == 'facecolor':
            # facecolor returns RGBA tuple
            assert result[0] == 1.0  # red
        elif prop == 'aspect' and val == 'equal':
            # OG stores 'equal' as float 1.0
            assert result in ('equal', 1.0, 1)
        else:
            assert result == val


# ===========================================================================
# Additional upstream-inspired tests
# ===========================================================================

class TestUpstreamMisc:
    def test_figure_get_axes(self):
        fig = Figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        axes = fig.get_axes()
        assert len(axes) == 2
        assert ax1 in axes
        assert ax2 in axes

    def test_axes_remove(self):
        fig, ax = plt.subplots()
        assert ax in fig.axes
        ax.remove()
        assert ax not in fig.axes

    def test_line2d_in_axes_lines(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4])
        assert lines[0] in ax.lines
        assert lines[0] in ax.get_lines()

    def test_scatter_in_collections(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2], [3, 4])
        assert pc in ax.collections

    def test_bar_in_patches(self):
        fig, ax = plt.subplots()
        bc = ax.bar([1, 2], [3, 4])
        for rect in bc:
            assert rect in ax.patches

    def test_text_in_texts(self):
        fig, ax = plt.subplots()
        t = ax.text(0, 0, 'hello')
        assert t in ax.texts

    def test_annotate_in_texts(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('test', (0, 0))
        assert ann in ax.texts

    def test_axes_set_batch(self):
        fig, ax = plt.subplots()
        ax.set(xlabel='X', ylabel='Y', title='T')
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        assert ax.get_title() == 'T'

    def test_line2d_set_batch(self):
        line = Line2D([0, 1], [0, 1])
        line.set(color='red', linewidth=3.0)
        assert line.get_color() == 'red'
        assert line.get_linewidth() == 3.0

    def test_axes_get_children_types(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.bar([1, 2], [3, 4])
        ax.text(0, 0, 'hello')
        children = ax.get_children()
        has_line = any(isinstance(c, Line2D) for c in children)
        has_rect = any(isinstance(c, Rectangle) for c in children)
        has_text = any(isinstance(c, Text) for c in children)
        assert has_line
        assert has_rect
        assert has_text

    def test_plot_empty(self):
        fig, ax = plt.subplots()
        lines = ax.plot([], [])
        assert len(lines) == 1
        assert _arr_eq(lines[0].get_xdata(), [])

    def test_plot_kwargs_forwarded(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2], [3, 4], color='red', linewidth=3.0,
                        marker='o', markersize=10)
        line = lines[0]
        assert line.get_linewidth() == 3.0
        assert line.get_marker() == 'o'
        assert line.get_markersize() == 10

    def test_axes_tight_layout(self):
        fig, ax = plt.subplots()
        fig.tight_layout()

    def test_axes_add_patch_returns(self):
        fig, ax = plt.subplots()
        rect = Rectangle((0, 0), 1, 1)
        result = ax.add_patch(rect)
        assert result is rect

    def test_axes_add_line_returns(self):
        fig, ax = plt.subplots()
        line = Line2D([0, 1], [0, 1])
        result = ax.add_line(line)
        assert result is line

    def test_axes_add_collection_returns(self):
        fig, ax = plt.subplots()
        lc = LineCollection()
        result = ax.add_collection(lc)
        assert result is lc
