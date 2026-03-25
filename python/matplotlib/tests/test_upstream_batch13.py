"""
Upstream matplotlib test_axes.py tests — batch 13.
Focus: errorbar, hist, scatter, bar_label, plot format, legend, collections.
"""
import math
import pytest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection


# ------------------------------------------------------------------
# Errorbar tests
# ------------------------------------------------------------------

class TestErrorbar:
    def test_basic(self):
        fig, ax = plt.subplots()
        container = ax.errorbar([0, 1, 2], [1, 2, 3], yerr=0.5)
        assert container is not None

    def test_with_xerr(self):
        fig, ax = plt.subplots()
        container = ax.errorbar([0, 1, 2], [1, 2, 3], xerr=0.1)
        assert container is not None

    def test_with_both_err(self):
        fig, ax = plt.subplots()
        container = ax.errorbar([0, 1, 2], [1, 2, 3], xerr=0.1, yerr=0.2)
        assert container is not None

    def test_returns_container(self):
        fig, ax = plt.subplots()
        container = ax.errorbar([0, 1], [0, 1], yerr=0.1)
        # Container is (plotline, caplines, barlinecols)
        assert len(container) == 3

    def test_nonefmt(self):
        """Passing fmt='none' should still produce error bars."""
        fig, ax = plt.subplots()
        plotline, caplines, barlinecols = ax.errorbar(
            [0, 1, 2], [0, 1, 2], xerr=1, yerr=1, fmt='none')
        assert plotline is None

    def test_empty_errorbar_legend(self):
        fig, ax = plt.subplots()
        ax.errorbar([], [], xerr=[], label='empty y')
        ax.errorbar([], [], yerr=[], label='empty x')
        ax.legend()

    def test_errorbar_kwargs(self):
        """Various kwargs should not cause errors."""
        fig, ax = plt.subplots()
        ax.errorbar([0, 1], [0, 1], yerr=0.1, capsize=5,
                    ecolor='red', elinewidth=2)

    def test_errorbar_single_point(self):
        fig, ax = plt.subplots()
        ax.errorbar([0], [0], yerr=[1])

    def test_errorbar_list_yerr(self):
        fig, ax = plt.subplots()
        ax.errorbar([0, 1, 2], [0, 1, 2], yerr=[0.1, 0.2, 0.3])


# ------------------------------------------------------------------
# Histogram tests
# ------------------------------------------------------------------

class TestHist:
    def test_basic(self):
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist([1, 2, 3, 4, 5])
        assert len(n) > 0
        assert len(bins) > 0

    def test_bins_int(self):
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist([1, 2, 3, 4, 5], bins=3)
        assert len(bins) == 4  # bins+1 edges

    def test_density(self):
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist([1, 1, 2, 3, 3, 3], density=True)
        # Density should integrate to ~1
        total = sum(n[i] * (bins[i+1] - bins[i]) for i in range(len(n)))
        assert abs(total - 1.0) < 0.1

    def test_empty_input(self):
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist([])
        assert len(n) > 0  # bins exist even for empty data

    def test_histtype_bar(self):
        fig, ax = plt.subplots()
        ax.hist([1, 2, 3, 4, 5], histtype='bar')

    def test_histtype_step(self):
        fig, ax = plt.subplots()
        ax.hist([1, 2, 3, 4, 5], histtype='step')

    def test_histtype_stepfilled(self):
        fig, ax = plt.subplots()
        ax.hist([1, 2, 3, 4, 5], histtype='stepfilled')

    def test_range(self):
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist([1, 2, 3, 4, 5], range=(2, 4))
        assert bins[0] >= 2
        assert bins[-1] <= 4

    def test_weights(self):
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist([1, 2, 3], weights=[2, 1, 3])

    def test_multiple_datasets(self):
        fig, ax = plt.subplots()
        ax.hist([[1, 2, 3], [2, 3, 4]])

    def test_hist_color(self):
        fig, ax = plt.subplots()
        ax.hist([1, 2, 3, 4, 5], color='red')

    def test_hist_label(self):
        fig, ax = plt.subplots()
        ax.hist([1, 2, 3, 4, 5], label='data')


# ------------------------------------------------------------------
# Scatter tests
# ------------------------------------------------------------------

class TestScatter:
    def test_basic(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2, 3], [4, 5, 6])
        assert isinstance(pc, PathCollection)

    def test_with_sizes(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [4, 5, 6], s=[10, 20, 30])

    def test_with_colors(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [4, 5, 6], c=['red', 'green', 'blue'])

    def test_with_numeric_c(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [4, 5, 6], c=[1, 2, 3])

    def test_with_alpha(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [4, 5, 6], alpha=0.5)

    def test_empty_data(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([], [])
        assert isinstance(pc, PathCollection)

    def test_marker(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2], [3, 4], marker='s')

    def test_edgecolors(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2], [3, 4], edgecolors='black')

    def test_label(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2], [3, 4], label='points')
        handles, labels = ax.get_legend_handles_labels()
        assert 'points' in labels


# ------------------------------------------------------------------
# Bar tests
# ------------------------------------------------------------------

class TestBar:
    def test_basic(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7])
        assert len(bars) == 3

    def test_width(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7], width=0.5)

    def test_color(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7], color='red')

    def test_bottom(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7], bottom=1)

    def test_label(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7], label='data')
        assert bars.get_label() == 'data'

    def test_edgecolor(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7], edgecolor='black')

    def test_barh(self):
        fig, ax = plt.subplots()
        bars = ax.barh([0, 1, 2], [3, 5, 7])
        assert len(bars) == 3

    def test_bar_broadcast_height(self):
        """Single height broadcasts to all bars."""
        fig, ax = plt.subplots()
        bars = ax.bar(range(4), 1)
        assert len(bars) == 4

    def test_bar_nan_values(self):
        fig, ax = plt.subplots()
        ax.bar([0, 1], [float('nan'), 4])

    def test_bar_empty(self):
        fig, ax = plt.subplots()
        ax.bar([], [])


# ------------------------------------------------------------------
# Bar label tests
# ------------------------------------------------------------------

class TestBarLabel:
    def test_basic(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7])
        labels = ax.bar_label(bars)

    def test_custom_labels(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3, 5, 7])
        labels = ax.bar_label(bars, labels=['a', 'b', 'c'])

    def test_fmt(self):
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1, 2], [3.14, 5.5, 7.1])
        labels = ax.bar_label(bars, fmt='%.1f')


# ------------------------------------------------------------------
# Plot format tests
# ------------------------------------------------------------------

class TestPlotFormat:
    def test_basic_line(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3])
        assert len(lines) == 1

    def test_xy(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6])
        assert len(lines) == 1

    def test_with_fmt(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6], 'r-')
        assert len(lines) == 1

    def test_marker_fmt(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6], 'o')
        assert lines[0].get_marker() == 'o'

    def test_color_fmt(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6], 'r')
        # Color should be red

    def test_linestyle_fmt(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6], '--')
        assert lines[0].get_linestyle() == '--'

    def test_combined_fmt(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6], 'ro-')

    def test_empty_plot(self):
        fig, ax = plt.subplots()
        lines = ax.plot([], [])
        assert len(lines) == 1  # creates line with empty data

    def test_plot_kwargs(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], [4, 5, 6], color='red', linewidth=3)
        assert lines[0].get_linewidth() == 3

    def test_plot_label(self):
        fig, ax = plt.subplots()
        lines = ax.plot([1, 2, 3], label='myline')
        assert lines[0].get_label() == 'myline'

    def test_multiline(self):
        """plot(x1, y1, x2, y2) should produce 2 lines."""
        fig, ax = plt.subplots()
        lines = ax.plot([0, 1], [0, 1], [0, 1], [1, 0])
        assert len(lines) == 2


# ------------------------------------------------------------------
# Legend tests
# ------------------------------------------------------------------

class TestLegend:
    def test_basic(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='a')
        ax.plot([1, 0], label='b')
        leg = ax.legend()
        assert leg is not None

    def test_custom_labels(self):
        fig, ax = plt.subplots()
        l1, = ax.plot([0, 1])
        l2, = ax.plot([1, 0])
        leg = ax.legend([l1, l2], ['first', 'second'])
        assert leg is not None

    def test_loc(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='a')
        leg = ax.legend(loc='upper left')

    def test_no_label_skipped(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1])  # no label
        ax.plot([1, 0], label='b')
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ['b']

    def test_underscore_label_skipped(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label='_hidden')
        ax.plot([1, 0], label='visible')
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ['visible']


# ------------------------------------------------------------------
# Collections tests
# ------------------------------------------------------------------

class TestCollections:
    def test_line_collection(self):
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 1)], [(0, 1), (1, 0)]])
        ax.add_collection(lc)
        assert lc in ax.collections

    def test_path_collection_offsets(self):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2, 3], [4, 5, 6])
        offsets = pc.get_offsets()
        assert len(offsets) == 3


# ------------------------------------------------------------------
# Fill between tests
# ------------------------------------------------------------------

class TestFillBetween:
    def test_basic(self):
        fig, ax = plt.subplots()
        ax.fill_between([0, 1, 2], [0, 1, 0])

    def test_y2(self):
        fig, ax = plt.subplots()
        ax.fill_between([0, 1, 2], [0, 1, 0], [1, 2, 1])

    def test_alpha(self):
        fig, ax = plt.subplots()
        ax.fill_between([0, 1, 2], [0, 1, 0], alpha=0.5)

    def test_color(self):
        fig, ax = plt.subplots()
        ax.fill_between([0, 1, 2], [0, 1, 0], color='blue')


class TestFillBetweenX:
    def test_basic(self):
        fig, ax = plt.subplots()
        ax.fill_betweenx([0, 1, 2], [0, 1, 0])

    def test_x2(self):
        fig, ax = plt.subplots()
        ax.fill_betweenx([0, 1, 2], [0, 1, 0], [1, 2, 1])


# ------------------------------------------------------------------
# Stem tests
# ------------------------------------------------------------------

class TestStem:
    def test_basic(self):
        fig, ax = plt.subplots()
        container = ax.stem([1, 2, 3], [4, 5, 6])
        assert container is not None

    def test_y_only(self):
        fig, ax = plt.subplots()
        container = ax.stem([4, 5, 6])
        assert container is not None

    def test_label(self):
        fig, ax = plt.subplots()
        container = ax.stem([1, 2, 3], label='stems')
        assert container.get_label() == 'stems'


# ------------------------------------------------------------------
# Stairs tests
# ------------------------------------------------------------------

class TestStairs:
    def test_basic(self):
        fig, ax = plt.subplots()
        ax.stairs([1, 2, 3])

    def test_with_edges(self):
        fig, ax = plt.subplots()
        ax.stairs([1, 2, 3], [0, 1, 2, 3])

    def test_empty(self):
        fig, ax = plt.subplots()
        ax.stairs([])


# ------------------------------------------------------------------
# Pie tests
# ------------------------------------------------------------------

class TestPie:
    def test_basic(self):
        fig, ax = plt.subplots()
        result = ax.pie([1, 2, 3])
        assert result is not None

    def test_labels(self):
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3], labels=['a', 'b', 'c'])

    def test_colors(self):
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3], colors=['red', 'green', 'blue'])

    def test_explode(self):
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3], explode=[0, 0.1, 0])

    def test_autopct(self):
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3], autopct='%1.1f%%')

    def test_startangle(self):
        fig, ax = plt.subplots()
        ax.pie([1, 2, 3], startangle=90)


# ------------------------------------------------------------------
# Boxplot tests
# ------------------------------------------------------------------

class TestBoxplot:
    def test_basic(self):
        fig, ax = plt.subplots()
        result = ax.boxplot([[1, 2, 3, 4, 5]])
        assert 'boxes' in result
        assert 'medians' in result
        assert 'whiskers' in result

    def test_multiple(self):
        fig, ax = plt.subplots()
        result = ax.boxplot([[1, 2, 3], [4, 5, 6, 7]])

    def test_vert_false(self):
        fig, ax = plt.subplots()
        result = ax.boxplot([[1, 2, 3, 4, 5]], vert=False)

    def test_showfliers_false(self):
        fig, ax = plt.subplots()
        result = ax.boxplot([[1, 2, 3, 4, 5, 100]], showfliers=False)


# ------------------------------------------------------------------
# Violinplot tests
# ------------------------------------------------------------------

class TestViolinplot:
    def test_basic(self):
        fig, ax = plt.subplots()
        data = list(range(1, 21))
        result = ax.violinplot([data])
        assert 'bodies' in result

    def test_showmeans(self):
        fig, ax = plt.subplots()
        data = list(range(1, 21))
        result = ax.violinplot([data], showmeans=True)
        assert 'cmeans' in result

    def test_showmedians(self):
        fig, ax = plt.subplots()
        data = list(range(1, 21))
        result = ax.violinplot([data], showmedians=True)
        assert 'cmedians' in result

    def test_showextrema(self):
        fig, ax = plt.subplots()
        data = list(range(1, 21))
        result = ax.violinplot([data], showextrema=True)
        assert 'cbars' in result

    def test_vert_false(self):
        fig, ax = plt.subplots()
        data = list(range(1, 21))
        result = ax.violinplot([data], vert=False)


# ------------------------------------------------------------------
# Lines tests
# ------------------------------------------------------------------

class TestLine2D:
    def test_get_set_xdata(self):
        line = Line2D([1, 2, 3], [4, 5, 6])
        assert list(line.get_xdata()) == [1, 2, 3]

    def test_set_xdata(self):
        line = Line2D([1, 2, 3], [4, 5, 6])
        line.set_xdata([10, 20, 30])
        assert list(line.get_xdata()) == [10, 20, 30]

    def test_get_set_ydata(self):
        line = Line2D([1, 2, 3], [4, 5, 6])
        assert list(line.get_ydata()) == [4, 5, 6]

    def test_set_ydata(self):
        line = Line2D([1, 2, 3], [4, 5, 6])
        line.set_ydata([40, 50, 60])
        assert list(line.get_ydata()) == [40, 50, 60]

    def test_get_set_data(self):
        line = Line2D([1, 2], [3, 4])
        x, y = line.get_data()
        assert list(x) == [1, 2]
        assert list(y) == [3, 4]

    def test_set_data(self):
        line = Line2D([1, 2], [3, 4])
        line.set_data([10, 20], [30, 40])
        x, y = line.get_data()
        assert list(x) == [10, 20]

    def test_color(self):
        line = Line2D([0, 1], [0, 1], color='red')
        c = line.get_color()
        assert c is not None

    def test_set_color(self):
        line = Line2D([0, 1], [0, 1])
        line.set_color('blue')

    def test_linewidth(self):
        line = Line2D([0, 1], [0, 1], linewidth=3)
        assert line.get_linewidth() == 3

    def test_set_linewidth(self):
        line = Line2D([0, 1], [0, 1])
        line.set_linewidth(5)
        assert line.get_linewidth() == 5

    def test_linestyle(self):
        line = Line2D([0, 1], [0, 1], linestyle='--')
        assert line.get_linestyle() == '--'

    def test_set_linestyle(self):
        line = Line2D([0, 1], [0, 1])
        line.set_linestyle('-.')
        assert line.get_linestyle() == '-.'

    def test_marker(self):
        line = Line2D([0, 1], [0, 1], marker='o')
        assert line.get_marker() == 'o'

    def test_set_marker(self):
        line = Line2D([0, 1], [0, 1])
        line.set_marker('s')
        assert line.get_marker() == 's'

    def test_markersize(self):
        line = Line2D([0, 1], [0, 1], markersize=10)
        assert line.get_markersize() == 10

    def test_label(self):
        line = Line2D([0, 1], [0, 1], label='test')
        assert line.get_label() == 'test'

    def test_set_label(self):
        line = Line2D([0, 1], [0, 1])
        line.set_label('test2')
        assert line.get_label() == 'test2'

    def test_alpha(self):
        line = Line2D([0, 1], [0, 1])
        line.set_alpha(0.5)
        assert line.get_alpha() == 0.5

    def test_visible(self):
        line = Line2D([0, 1], [0, 1])
        assert line.get_visible() is True
        line.set_visible(False)
        assert line.get_visible() is False

    def test_zorder(self):
        line = Line2D([0, 1], [0, 1])
        line.set_zorder(10)
        assert line.get_zorder() == 10

    def test_fillstyle(self):
        line = Line2D([0, 1], [0, 1])
        line.set_fillstyle('bottom')
        assert line.get_fillstyle() == 'bottom'

    def test_antialiased(self):
        line = Line2D([0, 1], [0, 1])
        line.set_antialiased(False)
        assert line.get_antialiased() is False

    def test_solid_capstyle(self):
        line = Line2D([0, 1], [0, 1])
        line.set_solid_capstyle('round')
        assert line.get_solid_capstyle() == 'round'

    def test_solid_joinstyle(self):
        line = Line2D([0, 1], [0, 1])
        line.set_solid_joinstyle('bevel')
        assert line.get_solid_joinstyle() == 'bevel'

    def test_dash_capstyle(self):
        line = Line2D([0, 1], [0, 1])
        line.set_dash_capstyle('round')
        assert line.get_dash_capstyle() == 'round'

    def test_dash_joinstyle(self):
        line = Line2D([0, 1], [0, 1])
        line.set_dash_joinstyle('miter')
        assert line.get_dash_joinstyle() == 'miter'

    def test_markeredgecolor(self):
        line = Line2D([0, 1], [0, 1])
        line.set_markeredgecolor('red')
        c = line.get_markeredgecolor()
        assert c is not None

    def test_markerfacecolor(self):
        line = Line2D([0, 1], [0, 1])
        line.set_markerfacecolor('blue')
        c = line.get_markerfacecolor()
        assert c is not None

    def test_markeredgewidth(self):
        line = Line2D([0, 1], [0, 1])
        line.set_markeredgewidth(2.0)
        assert line.get_markeredgewidth() == 2.0

    def test_markevery(self):
        line = Line2D([0, 1, 2, 3], [0, 1, 0, 1])
        line.set_markevery(2)
        assert line.get_markevery() == 2

    def test_repr(self):
        line = Line2D([0, 1], [0, 1])
        r = repr(line)
        assert 'Line2D' in r


# ------------------------------------------------------------------
# Imshow tests
# ------------------------------------------------------------------

class TestImshow:
    def test_basic(self):
        fig, ax = plt.subplots()
        im = ax.imshow([[0, 1], [2, 3]])
        assert im is not None

    def test_cmap(self):
        fig, ax = plt.subplots()
        ax.imshow([[0, 1], [2, 3]], cmap='viridis')

    def test_aspect(self):
        fig, ax = plt.subplots()
        ax.imshow([[0, 1], [2, 3]], aspect='auto')


# ------------------------------------------------------------------
# Pcolormesh tests
# ------------------------------------------------------------------

class TestPcolormesh:
    def test_basic(self):
        fig, ax = plt.subplots()
        ax.pcolormesh([[0, 1], [2, 3]])

    def test_cmap(self):
        fig, ax = plt.subplots()
        ax.pcolormesh([[0, 1], [2, 3]], cmap='hot')


# ------------------------------------------------------------------
# Text/Annotation tests
# ------------------------------------------------------------------

class TestTextAnnotation:
    def test_text(self):
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'hello')
        assert t.get_text() == 'hello'

    def test_annotate(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('text', xy=(0.5, 0.5))
        assert ann is not None

    def test_annotate_xytext(self):
        fig, ax = plt.subplots()
        ann = ax.annotate('text', xy=(0.5, 0.5), xytext=(0.3, 0.3))

    def test_text_in_children(self):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'hello')
        assert len(ax.texts) >= 1


# ------------------------------------------------------------------
# Contour stubs
# ------------------------------------------------------------------

class TestContour:
    def test_contour_returns(self):
        fig, ax = plt.subplots()
        # contour is a stub, should not error
        result = ax.contour()
        assert result is not None

    def test_contourf_returns(self):
        fig, ax = plt.subplots()
        result = ax.contourf()
        assert result is not None


class TestBatch13Parametric4:
    """More parametric tests."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_n_lines(self, n):
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i+1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("lo,hi", [(0, 1), (-1, 1), (0, 100)])
    def test_xlim(self, lo, hi):
        fig, ax = plt.subplots()
        ax.set_xlim(lo, hi)
        assert ax.get_xlim() == (lo, hi)
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        fig, ax = plt.subplots()
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0])
    def test_linewidth(self, lw):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-10
        plt.close("all")

    @pytest.mark.parametrize("marker", ["o", "s", "^", "D"])
    def test_marker(self, marker):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_hist(self, bins):
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(list(range(100)), bins=bins)
        assert len(n) == bins
        plt.close("all")

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_bar(self, n):
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(n))
        assert len(bars.patches) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["equal", "auto"])
    def test_aspect(self, aspect):
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close("all")

    @pytest.mark.parametrize("title", ["Title", "Test", ""])
    def test_title(self, title):
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close("all")

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_line_alpha(self, alpha):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-10
        plt.close("all")



class TestBatch13Parametric10:
    """Further parametric tests for batch 13."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_n_lines(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot([0, 1], [i, i + 1])
        assert len(ax.lines) == n
        plt.close("all")

    @pytest.mark.parametrize("xlim", [(-1, 1), (0, 10), (-5, 5), (0, 100)])
    def test_xlim(self, xlim):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        result = ax.get_xlim()
        assert abs(result[0] - xlim[0]) < 1e-9
        assert abs(result[1] - xlim[1]) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
    def test_xscale(self, scale):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xscale(scale)
        assert ax.get_xscale() == scale
        plt.close("all")

    @pytest.mark.parametrize("lw", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_linewidth(self, lw):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], linewidth=lw)
        assert abs(line.get_linewidth() - lw) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("marker", ["o", "s", "^", "D", "v"])
    def test_marker(self, marker):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1], marker=marker)
        assert line.get_marker() == marker
        plt.close("all")

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_bar(self, n):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bars = ax.bar(range(n), range(1, n + 1))
        assert len(bars) == n
        plt.close("all")

    @pytest.mark.parametrize("aspect", ["auto", "equal"])
    def test_aspect(self, aspect):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)
        assert ax.get_aspect() == aspect
        plt.close("all")

    @pytest.mark.parametrize("title", ["Test", "My Plot", "Signal", "", "Results"])
    def test_title(self, title):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title(title)
        assert ax.get_title() == title
        plt.close("all")

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_line_alpha(self, alpha):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line.set_alpha(alpha)
        assert abs(line.get_alpha() - alpha) < 1e-9
        plt.close("all")

    @pytest.mark.parametrize("visible", [True, False])
    def test_visible(self, visible):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_visible(visible)
        assert ax.get_visible() == visible
        plt.close("all")

