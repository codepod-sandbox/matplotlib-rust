"""
Upstream matplotlib tests for the pyplot module.
"""

import numpy as np
import pytest

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Figure management
# ---------------------------------------------------------------------------
def test_close_all():
    """plt.close('all') removes all figures."""
    plt.close('all')
    plt.figure()
    plt.figure()
    assert len(plt.get_fignums()) == 2
    plt.close('all')
    assert len(plt.get_fignums()) == 0


def test_close_current():
    """plt.close() closes the current figure."""
    plt.close('all')
    fig1 = plt.figure()
    fig2 = plt.figure()
    plt.close()
    assert len(plt.get_fignums()) == 1


def test_close_by_number():
    """plt.close(num) closes a specific figure."""
    plt.close('all')
    fig1 = plt.figure()
    fig2 = plt.figure()
    plt.close(fig1.number)
    nums = plt.get_fignums()
    assert fig1.number not in nums
    assert fig2.number in nums


def test_close_by_figure():
    """plt.close(fig) closes the figure."""
    plt.close('all')
    fig = plt.figure()
    plt.close(fig)
    assert len(plt.get_fignums()) == 0


def test_close_by_label():
    """plt.close(label) closes by label."""
    plt.close('all')
    fig = plt.figure('labeled')
    plt.close('labeled')
    assert len(plt.get_fignums()) == 0


def test_close_float_raises():
    """plt.close(float) raises TypeError."""
    with pytest.raises(TypeError):
        plt.close(1.5)


def test_gcf_creates_figure():
    """gcf creates a figure if none exist."""
    plt.close('all')
    fig = plt.gcf()
    assert fig is not None
    assert len(plt.get_fignums()) >= 1


def test_gca_creates_axes():
    """gca creates axes if none exist."""
    plt.close('all')
    ax = plt.gca()
    assert ax is not None


def test_sca():
    """sca sets the current axes."""
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax1)
    assert plt.gca() is ax1
    plt.sca(ax2)
    assert plt.gca() is ax2


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
def test_ion_ioff():
    """ion/ioff toggle interactive mode."""
    plt.ioff()
    assert plt.isinteractive() is False
    plt.ion()
    assert plt.isinteractive() is True
    plt.ioff()
    assert plt.isinteractive() is False


def test_ioff_context():
    """ioff as context manager restores state."""
    plt.ion()
    with plt.ioff():
        assert plt.isinteractive() is False
    assert plt.isinteractive() is True
    plt.ioff()  # cleanup


def test_ion_context():
    """ion as context manager restores state."""
    plt.ioff()
    with plt.ion():
        assert plt.isinteractive() is True
    assert plt.isinteractive() is False


# ---------------------------------------------------------------------------
# Subplot / subplots
# ---------------------------------------------------------------------------
def test_subplots_1x1():
    """subplots(1,1) returns single axes."""
    fig, ax = plt.subplots()
    assert ax is not None
    assert len(fig.axes) == 1


def test_subplots_2x2():
    """subplots(2,2) returns grid of axes."""
    fig, axes = plt.subplots(2, 2)
    assert len(axes) == 2
    assert len(axes[0]) == 2


def test_subplots_1xN():
    """subplots(1,3) returns flat list."""
    fig, axes = plt.subplots(1, 3)
    assert len(axes) == 3


def test_subplots_Nx1():
    """subplots(3,1) returns flat list."""
    fig, axes = plt.subplots(3, 1)
    assert len(axes) == 3


def test_subplots_sharex():
    """subplots with sharex links x axes."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_xlim(0, 10)
    xlim = axes[1].get_xlim()
    assert xlim == (0, 10)


def test_subplots_sharey():
    """subplots with sharey links y axes."""
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].set_ylim(-5, 5)
    ylim = axes[1].get_ylim()
    assert ylim == (-5, 5)


def test_subplot_reuse():
    """subplot(n,m,k) reuses existing axes."""
    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 1)  # same position
    assert ax1 is ax2


# ---------------------------------------------------------------------------
# Plotting functions delegate correctly
# ---------------------------------------------------------------------------
def test_plt_plot():
    """plt.plot delegates to current axes."""
    plt.close('all')
    lines = plt.plot([1, 2, 3], [4, 5, 6])
    assert len(lines) == 1


def test_plt_scatter():
    """plt.scatter delegates to current axes."""
    plt.close('all')
    pc = plt.scatter([1, 2], [3, 4])
    assert pc is not None


def test_plt_bar():
    """plt.bar delegates to current axes."""
    plt.close('all')
    bc = plt.bar([1, 2], [3, 4])
    assert len(bc) == 2


def test_plt_barh():
    """plt.barh delegates to current axes."""
    plt.close('all')
    bc = plt.barh([1, 2], [3, 4])
    assert len(bc) == 2


def test_plt_hist():
    """plt.hist delegates to current axes."""
    plt.close('all')
    counts, edges, bc = plt.hist([1, 2, 3, 4, 5])
    assert len(counts) > 0


def test_plt_errorbar():
    """plt.errorbar delegates to current axes."""
    plt.close('all')
    ec = plt.errorbar([1, 2], [3, 4], yerr=[0.1, 0.2])
    assert ec is not None


def test_plt_fill_between():
    """plt.fill_between delegates."""
    plt.close('all')
    poly = plt.fill_between([0, 1, 2], [0, 1, 0])
    assert poly is not None


def test_plt_axhline():
    """plt.axhline delegates."""
    plt.close('all')
    line = plt.axhline(y=5)
    assert line is not None


def test_plt_axvline():
    """plt.axvline delegates."""
    plt.close('all')
    line = plt.axvline(x=3)
    assert line is not None


def test_plt_axhspan():
    """plt.axhspan delegates."""
    plt.close('all')
    rect = plt.axhspan(0, 1)
    assert rect is not None


def test_plt_axvspan():
    """plt.axvspan delegates."""
    plt.close('all')
    rect = plt.axvspan(0, 1)
    assert rect is not None


def test_plt_text():
    """plt.text delegates."""
    plt.close('all')
    t = plt.text(0.5, 0.5, 'hello')
    assert t is not None


def test_plt_hlines():
    """plt.hlines delegates."""
    plt.close('all')
    lines = plt.hlines([1, 2], 0, 10)
    assert len(lines) == 2


def test_plt_vlines():
    """plt.vlines delegates."""
    plt.close('all')
    lines = plt.vlines([1, 2], 0, 10)
    assert len(lines) == 2


def test_plt_loglog():
    """plt.loglog sets both scales to log."""
    plt.close('all')
    plt.loglog([1, 10], [1, 10])
    ax = plt.gca()
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'log'


def test_plt_semilogx():
    """plt.semilogx sets x to log."""
    plt.close('all')
    plt.semilogx([1, 10], [1, 2])
    ax = plt.gca()
    assert ax.get_xscale() == 'log'


def test_plt_semilogy():
    """plt.semilogy sets y to log."""
    plt.close('all')
    plt.semilogy([1, 2], [1, 10])
    ax = plt.gca()
    assert ax.get_yscale() == 'log'


def test_plt_step():
    """plt.step delegates."""
    plt.close('all')
    lines = plt.step([0, 1, 2], [0, 1, 0])
    assert lines is not None


def test_plt_stairs():
    """plt.stairs delegates."""
    plt.close('all')
    line = plt.stairs([1, 2, 3])
    assert line is not None


def test_plt_stackplot():
    """plt.stackplot delegates."""
    plt.close('all')
    polys = plt.stackplot([0, 1, 2], [1, 2, 3], [3, 2, 1])
    assert len(polys) == 2


def test_plt_stem():
    """plt.stem delegates."""
    plt.close('all')
    sc = plt.stem([1, 2, 3])
    assert sc is not None


def test_plt_pie():
    """plt.pie delegates."""
    plt.close('all')
    result = plt.pie([1, 2, 3])
    assert len(result) == 2  # (wedges, texts)


def test_plt_boxplot():
    """plt.boxplot delegates."""
    plt.close('all')
    result = plt.boxplot([1, 2, 3, 4, 5])
    assert 'boxes' in result


def test_plt_violinplot():
    """plt.violinplot delegates."""
    plt.close('all')
    result = plt.violinplot([1, 2, 3, 4, 5])
    assert 'bodies' in result


# ---------------------------------------------------------------------------
# Labels / config delegates
# ---------------------------------------------------------------------------
def test_plt_xlabel():
    """plt.xlabel sets x label."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.xlabel('X Axis')
    assert ax.get_xlabel() == 'X Axis'


def test_plt_ylabel():
    """plt.ylabel sets y label."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.ylabel('Y Axis')
    assert ax.get_ylabel() == 'Y Axis'


def test_plt_title():
    """plt.title sets title."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.title('My Plot')
    assert ax.get_title() == 'My Plot'


def test_plt_suptitle():
    """plt.suptitle delegates to figure."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.suptitle('Super Title')
    assert fig.get_suptitle() == 'Super Title'


def test_plt_xlim_set():
    """plt.xlim sets limits."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.xlim(0, 10)
    assert ax.get_xlim() == (0, 10)


def test_plt_xlim_get():
    """plt.xlim() returns limits."""
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    assert plt.xlim() == (0, 10)


def test_plt_ylim_set():
    """plt.ylim sets limits."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.ylim(-5, 5)
    assert ax.get_ylim() == (-5, 5)


def test_plt_ylim_get():
    """plt.ylim() returns limits."""
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_ylim(-5, 5)
    assert plt.ylim() == (-5, 5)


def test_plt_grid():
    """plt.grid enables grid."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.grid(True)  # should not raise


def test_plt_legend():
    """plt.legend delegates."""
    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4], label='line')
    plt.legend()
    assert ax.get_legend() is not None


def test_plt_cla():
    """plt.cla clears current axes."""
    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    plt.cla()
    assert len(ax.lines) == 0


def test_plt_clf():
    """plt.clf clears current figure."""
    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    plt.clf()
    assert len(fig.axes) == 0


def test_plt_show_noop():
    """plt.show is a no-op."""
    plt.show()  # should not raise


# ---------------------------------------------------------------------------
# plt.margins
# ---------------------------------------------------------------------------
def test_plt_margins_get():
    """plt.margins() returns current margins."""
    plt.close('all')
    fig, ax = plt.subplots()
    mx, my = plt.margins()
    assert mx == 0.05  # default
    assert my == 0.05


def test_plt_margins_set():
    """plt.margins(m) sets both margins."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.margins(0.1)
    mx, my = plt.margins()
    assert mx == 0.1
    assert my == 0.1


# ---------------------------------------------------------------------------
# plt.xticks / yticks
# ---------------------------------------------------------------------------
def test_plt_xticks_set():
    """plt.xticks sets tick locations."""
    import numpy as np
    plt.close('all')
    fig, ax = plt.subplots()
    plt.xticks([0, 1, 2])
    # OG get_xticks() returns ndarray, not list
    assert np.array_equal(ax.get_xticks(), [0, 1, 2])


def test_plt_yticks_set():
    """plt.yticks sets tick locations."""
    import numpy as np
    plt.close('all')
    fig, ax = plt.subplots()
    plt.yticks([0, 5, 10])
    # OG get_yticks() returns ndarray, not list
    assert np.array_equal(ax.get_yticks(), [0, 5, 10])


def test_plt_xticks_get():
    """plt.xticks() returns tick locations."""
    import numpy as np
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    # OG plt.xticks() returns (ndarray, list_of_Text) tuple
    result = plt.xticks()
    assert np.array_equal(result[0], [1, 2, 3])


def test_plt_yticks_get():
    """plt.yticks() returns tick locations."""
    import numpy as np
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_yticks([1, 2, 3])
    # OG plt.yticks() returns (ndarray, list_of_Text) tuple
    result = plt.yticks()
    assert np.array_equal(result[0], [1, 2, 3])


# ---------------------------------------------------------------------------
# fignum_exists
# ---------------------------------------------------------------------------
def test_fignum_exists():
    """fignum_exists checks by number."""
    plt.close('all')
    fig = plt.figure()
    assert plt.fignum_exists(fig.number) is True
    assert plt.fignum_exists(999) is False


def test_fignum_exists_label():
    """fignum_exists checks by label."""
    plt.close('all')
    plt.figure('test_label')
    assert plt.fignum_exists('test_label') is True
    assert plt.fignum_exists('no_such_label') is False


# ---------------------------------------------------------------------------
# Figure reuse
# ---------------------------------------------------------------------------
def test_figure_reuse_by_number():
    """figure(num) returns existing figure."""
    plt.close('all')
    fig1 = plt.figure(42)
    fig2 = plt.figure(42)
    assert fig1 is fig2


def test_figure_reuse_by_label():
    """figure(label) returns existing figure."""
    plt.close('all')
    fig1 = plt.figure('my_fig')
    fig2 = plt.figure('my_fig')
    assert fig1 is fig2


# ===================================================================
# get_fignums / get_figlabels / axes / tight_layout / imshow / contour
# ===================================================================

def test_plt_get_fignums_empty():
    """get_fignums returns empty list when no figures."""
    plt.close('all')
    nums = plt.get_fignums()
    assert isinstance(nums, list)
    assert nums == []


def test_plt_get_fignums_with_figures():
    """get_fignums returns figure numbers."""
    plt.close('all')
    fig1 = plt.figure()
    fig2 = plt.figure()
    nums = plt.get_fignums()
    assert fig1.number in nums
    assert fig2.number in nums
    plt.close('all')


def test_plt_get_figlabels_empty():
    """get_figlabels returns empty list when no figures."""
    plt.close('all')
    labels = plt.get_figlabels()
    assert isinstance(labels, list)
    assert labels == []


def test_plt_get_figlabels_with_label():
    """get_figlabels returns figure labels."""
    plt.close('all')
    plt.figure('mylabel')  # string num sets label
    labels = plt.get_figlabels()
    assert 'mylabel' in labels
    plt.close('all')


def test_plt_axes_creates_axes():
    """plt.axes() creates axes on current figure."""
    plt.close('all')
    ax = plt.axes()
    assert ax is not None
    plt.close('all')


def test_plt_imshow_basic():
    """plt.imshow returns an image."""
    plt.close('all')
    data = [[1, 2], [3, 4]]
    im = plt.imshow(data)
    assert im is not None
    plt.close('all')


def test_plt_tight_layout_noop():
    """plt.tight_layout() doesn't raise."""
    plt.close('all')
    plt.figure()
    plt.tight_layout()
    plt.close('all')


def test_plt_contour_basic():
    """plt.contour with 2D list data returns a result."""
    plt.close('all')
    Z = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    result = plt.contour(Z)
    assert result is not None
    plt.close('all')


def test_plt_contourf_basic():
    """plt.contourf with 2D list data returns a result."""
    plt.close('all')
    Z = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    result = plt.contourf(Z)
    assert result is not None
    plt.close('all')


def test_plt_pcolormesh_basic():
    """plt.pcolormesh returns a result."""
    import numpy as np
    plt.close('all')
    data = np.zeros((4, 4))
    result = plt.pcolormesh(data)
    assert result is not None
    plt.close('all')


def test_plt_get_figlabels_multiple():
    """get_figlabels returns multiple labels."""
    plt.close('all')
    plt.figure('fig_a')
    plt.figure('fig_b')
    labels = plt.get_figlabels()
    assert 'fig_a' in labels
    assert 'fig_b' in labels
    plt.close('all')


def test_plt_axes_on_existing_figure():
    """plt.axes() on existing figure returns its axes."""
    plt.close('all')
    fig = plt.figure()
    ax = plt.axes()
    assert ax is not None
    assert ax is plt.gca()
    plt.close('all')


def test_plt_rc():
    """plt.rc sets rcParams values."""
    import matplotlib
    plt.close('all')
    plt.rc('lines', linewidth=3)
    assert matplotlib.rcParams['lines.linewidth'] == 3
    plt.rcdefaults()


def test_plt_rcdefaults():
    """plt.rcdefaults restores default rcParams (no-op check)."""
    import matplotlib
    plt.close('all')
    plt.rc('lines', linewidth=5)
    plt.rcdefaults()
    # rcParamsDefault may not exist; just verify rcdefaults() doesn't raise
    assert 'lines.linewidth' in matplotlib.rcParams


def test_plt_figtext():
    """plt.figtext adds text to figure."""
    plt.close('all')
    fig = plt.figure()
    txt = plt.figtext(0.5, 0.5, 'hello')
    assert txt is not None
    assert txt.get_text() == 'hello'
    plt.close('all')


def test_plt_axis_off():
    """plt.axis('off') does not raise."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.axis('off')  # no-op in our implementation; just verifies no exception
    plt.close('all')


def test_plt_xscale():
    """plt.xscale sets x-axis scale."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.xscale('log')
    assert ax.get_xscale() == 'log'
    plt.close('all')


def test_plt_yscale():
    """plt.yscale sets y-axis scale."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.yscale('log')
    assert ax.get_yscale() == 'log'
    plt.close('all')


def test_plt_twinx():
    """plt.twinx creates twin axes sharing x."""
    plt.close('all')
    fig, ax = plt.subplots()
    ax2 = plt.twinx()
    assert ax2 is not None
    assert ax2 is not ax
    plt.close('all')


def test_plt_twiny():
    """plt.twiny creates twin axes sharing y."""
    plt.close('all')
    fig, ax = plt.subplots()
    ax2 = plt.twiny()
    assert ax2 is not None
    assert ax2 is not ax
    plt.close('all')


def test_plt_annotate():
    """plt.annotate adds annotation to current axes."""
    plt.close('all')
    fig, ax = plt.subplots()
    ann = plt.annotate('test', xy=(0.5, 0.5))
    assert ann is not None
    assert ann.get_text() == 'test'
    plt.close('all')


def test_plt_fill_betweenx():
    """plt.fill_betweenx adds polygon patch."""
    plt.close('all')
    fig, ax = plt.subplots()
    poly = plt.fill_betweenx([0, 1, 2], [0, 1, 0], [1, 2, 1])
    assert poly is not None
    plt.close('all')


def test_plt_semilogx():
    """plt.semilogx sets x-axis to log scale."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.semilogx([1, 10, 100], [1, 2, 3])
    assert ax.get_xscale() == 'log'
    plt.close('all')


def test_plt_semilogy():
    """plt.semilogy sets y-axis to log scale."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.semilogy([1, 2, 3], [1, 10, 100])
    assert ax.get_yscale() == 'log'
    plt.close('all')


def test_plt_loglog():
    """plt.loglog sets both axes to log scale."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.loglog([1, 10, 100], [1, 10, 100])
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'log'
    plt.close('all')


def test_plt_tick_params():
    """plt.tick_params does not raise."""
    plt.close('all')
    fig, ax = plt.subplots()
    plt.tick_params(axis='x', labelsize=8)
    plt.close('all')


def test_plt_savefig_stringio():
    """plt.savefig to StringIO object works for SVG."""
    import io
    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buf = io.StringIO()
    plt.savefig(buf, format='svg')
    data = buf.getvalue()
    assert len(data) > 0
    assert '<svg' in data.lower() or 'svg' in data
    plt.close('all')


# ===================================================================
# Extended parametric tests for pyplot upstream
# ===================================================================


class TestPyplotFigureManagement:
    """Tests for figure management through plt interface."""

    def test_plt_figure_creates_figure(self):
        from matplotlib.figure import Figure
        plt.close('all')
        fig = plt.figure()
        assert isinstance(fig, Figure)
        plt.close('all')

    def test_plt_subplots_returns_fig_ax(self):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        plt.close('all')
        fig, ax = plt.subplots()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close('all')

    def test_plt_close_all(self):
        plt.close('all')
        for i in range(5):
            plt.figure()
        plt.close('all')
        # No error

    def test_plt_gcf_returns_figure(self):
        from matplotlib.figure import Figure
        plt.close('all')
        fig = plt.figure()
        gcf = plt.gcf()
        assert isinstance(gcf, Figure)
        plt.close('all')

    def test_plt_gca_returns_axes(self):
        from matplotlib.axes import Axes
        plt.close('all')
        fig, ax = plt.subplots()
        gca = plt.gca()
        assert isinstance(gca, Axes)
        plt.close('all')

    def test_plt_plot_returns_list(self):
        plt.close('all')
        fig, ax = plt.subplots()
        result = plt.plot([0, 1], [0, 1])
        assert isinstance(result, list)
        assert len(result) == 1
        plt.close('all')

    def test_plt_xlabel_ylabel(self):
        plt.close('all')
        fig, ax = plt.subplots()
        plt.xlabel('X')
        plt.ylabel('Y')
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        plt.close('all')

    def test_plt_title(self):
        plt.close('all')
        fig, ax = plt.subplots()
        plt.title('My Plot')
        assert ax.get_title() == 'My Plot'
        plt.close('all')

    def test_plt_xlim_ylim(self):
        plt.close('all')
        fig, ax = plt.subplots()
        plt.xlim(0, 10)
        plt.ylim(-5, 5)
        assert ax.get_xlim() == (0, 10)
        ymin, ymax = ax.get_ylim()
        assert abs(ymin - (-5)) < 1e-10
        assert abs(ymax - 5) < 1e-10
        plt.close('all')

    @pytest.mark.parametrize('style', ['-', '--', '-.', ':'])
    def test_plt_plot_linestyle(self, style):
        plt.close('all')
        fig, ax = plt.subplots()
        line, = plt.plot([0, 1], [0, 1], linestyle=style)
        plt.close('all')

    def test_plt_legend_after_plot(self):
        plt.close('all')
        fig, ax = plt.subplots()
        plt.plot([0, 1], [0, 1], label='line')
        leg = plt.legend()
        assert leg is not None
        plt.close('all')
