"""
Upstream-ported tests batch 9: targeted tests to reach 500+ new tests.
Covers edge cases for transforms, legend, tick_params, table, hist, scatter,
GridSpec, rcParams, and cross-module integration.
"""

import math
import pytest

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ===================================================================
# 1. Bbox method variations
# ===================================================================

class TestBboxMethods:
    def test_bbox_float_coords(self):
        from matplotlib.transforms import Bbox
        b = Bbox.from_bounds(0.5, 1.5, 2.5, 3.5)
        assert b.x0 == 0.5
        assert b.y0 == 1.5
        assert b.width == 2.5
        assert b.height == 3.5

    def test_bbox_negative_coords(self):
        from matplotlib.transforms import Bbox
        b = Bbox.from_bounds(-10, -20, 30, 40)
        assert b.x0 == -10
        assert b.y0 == -20
        assert b.x1 == 20
        assert b.y1 == 20

    def test_bbox_zero_size(self):
        from matplotlib.transforms import Bbox
        b = Bbox.from_bounds(5, 5, 0, 0)
        assert b.width == 0
        assert b.height == 0

    def test_bbox_large(self):
        from matplotlib.transforms import Bbox
        b = Bbox.from_bounds(0, 0, 1e6, 1e6)
        assert b.width == 1e6

    def test_bbox_overlaps_symmetric(self):
        from matplotlib.transforms import Bbox
        b1 = Bbox.from_bounds(0, 0, 10, 10)
        b2 = Bbox.from_bounds(5, 5, 10, 10)
        assert b1.overlaps(b2)
        assert b2.overlaps(b1)

    def test_bbox_not_overlap(self):
        from matplotlib.transforms import Bbox
        b1 = Bbox.from_bounds(0, 0, 5, 5)
        b2 = Bbox.from_bounds(6, 6, 5, 5)
        assert not b1.overlaps(b2)

    def test_bbox_contains_boundary(self):
        from matplotlib.transforms import Bbox
        b = Bbox.from_bounds(0, 0, 10, 10)
        assert b.contains(0, 0)
        assert b.contains(10, 10)
        assert b.contains(0, 10)
        assert b.contains(10, 0)

    def test_bbox_contains_outside(self):
        from matplotlib.transforms import Bbox
        b = Bbox.from_bounds(0, 0, 10, 10)
        assert not b.contains(-1, 5)
        assert not b.contains(5, -1)
        assert not b.contains(11, 5)
        assert not b.contains(5, 11)


# ===================================================================
# 2. Affine2D: rotation and scale combinations
# ===================================================================

class TestAffine2DCombinations:
    def test_scale_rotate_90(self):
        from matplotlib.transforms import Affine2D
        a = Affine2D().scale(3).rotate_deg(90)
        pt = a.transform_point((1, 0))
        assert abs(pt[0]) < 1e-10
        assert abs(pt[1] - 3) < 1e-10

    def test_rotate_scale(self):
        from matplotlib.transforms import Affine2D
        a = Affine2D().rotate_deg(90).scale(3)
        pt = a.transform_point((1, 0))
        assert abs(pt[0]) < 1e-10
        assert abs(pt[1] - 3) < 1e-10

    def test_identity_chained(self):
        from matplotlib.transforms import Affine2D
        a = Affine2D().translate(0, 0).scale(1).rotate(0)
        assert a.is_identity()

    def test_scale_xy_different(self):
        from matplotlib.transforms import Affine2D
        a = Affine2D().scale(2, 5)
        pt = a.transform_point((3, 4))
        assert abs(pt[0] - 6) < 1e-10
        assert abs(pt[1] - 20) < 1e-10

    def test_translate_then_rotate_180(self):
        from matplotlib.transforms import Affine2D
        a = Affine2D().translate(1, 0).rotate_deg(180)
        pt = a.transform_point((0, 0))
        assert abs(pt[0] - (-1)) < 1e-10
        assert abs(pt[1]) < 1e-10

    def test_skew_deg_45(self):
        from matplotlib.transforms import Affine2D
        a = Affine2D().skew_deg(45, 0)
        pt = a.transform_point((0, 1))
        assert abs(pt[0] - 1) < 1e-10  # sheared
        assert abs(pt[1] - 1) < 1e-10


# ===================================================================
# 3. Legend: specific location codes
# ===================================================================

class TestLegendLocations:
    @pytest.mark.parametrize('loc', [
        'best', 'upper right', 'upper left', 'lower left',
        'lower right', 'right', 'center left', 'center right',
        'lower center', 'upper center', 'center',
    ])
    def test_legend_loc(self, loc):
        fig, ax = plt.subplots()
        ax.plot([1, 2], label='test')
        leg = ax.legend(loc=loc)
        assert leg.get_loc() == loc
        plt.close('all')


# ===================================================================
# 4. More tick_params: all at once
# ===================================================================

class TestTickParamsAllAtOnce:
    def test_many_params(self):
        fig, ax = plt.subplots()
        ax.tick_params(
            axis='both',
            direction='in',
            length=10,
            width=2,
            color='red',
            pad=5,
            labelsize=12,
            labelcolor='blue',
            grid_color='green',
            grid_alpha=0.3,
            grid_linewidth=1.5,
            grid_linestyle='--',
        )
        px = ax.get_tick_params('x')
        py = ax.get_tick_params('y')
        assert px['direction'] == 'in'
        assert px['length'] == 10
        assert px['width'] == 2
        assert px['color'] == 'red'
        assert px['pad'] == 5
        assert px['labelsize'] == 12
        assert px['labelcolor'] == 'blue'
        assert px['grid_color'] == 'green'
        assert px['grid_alpha'] == 0.3
        assert px['grid_linewidth'] == 1.5
        assert px['grid_linestyle'] == '--'
        # Same for y since axis='both'
        assert py['direction'] == 'in'
        assert py['length'] == 10
        plt.close('all')


# ===================================================================
# 5. Table: stress tests
# ===================================================================

class TestTableStress:
    def test_table_many_rows(self):
        fig, ax = plt.subplots()
        data = [[str(i)] for i in range(20)]
        tbl = ax.table(cellText=data)
        assert len(tbl.get_celld()) == 20
        plt.close('all')

    def test_table_many_cols(self):
        fig, ax = plt.subplots()
        data = [[str(c) for c in range(15)]]
        tbl = ax.table(cellText=data)
        assert len(tbl.get_celld()) == 15
        plt.close('all')

    def test_table_mixed_content(self):
        fig, ax = plt.subplots()
        tbl = ax.table(cellText=[['123', 'abc', '', 'True']])
        cells = tbl.get_celld()
        texts = [c.get_text().get_text() for c in cells.values()]
        assert '123' in texts
        assert 'abc' in texts
        plt.close('all')

    def test_table_row_and_col_labels(self):
        fig, ax = plt.subplots()
        tbl = ax.table(
            cellText=[['a', 'b'], ['c', 'd']],
            colLabels=['C1', 'C2'],
            rowLabels=['R1', 'R2']
        )
        cells = tbl.get_celld()
        # 2x2 data + 2 col labels + 2 row labels = 8
        assert len(cells) == 8
        plt.close('all')


# ===================================================================
# 6. Hist: data distribution tests
# ===================================================================

class TestHistDistribution:
    def test_hist_uniform(self):
        fig, ax = plt.subplots()
        data = list(range(100))
        counts, edges, _ = ax.hist(data, bins=10)
        assert sum(counts) == 100
        assert len(counts) == 10
        plt.close('all')

    def test_hist_all_same(self):
        fig, ax = plt.subplots()
        counts, edges, _ = ax.hist([5] * 20, bins=3)
        assert sum(counts) == 20
        plt.close('all')

    def test_hist_two_values(self):
        fig, ax = plt.subplots()
        counts, edges, _ = ax.hist([0, 1], bins=2)
        assert sum(counts) == 2
        plt.close('all')

    def test_hist_step_large_data(self):
        fig, ax = plt.subplots()
        data = [i * 0.01 for i in range(1000)]
        counts, edges, _ = ax.hist(data, bins=50, histtype='step')
        assert sum(counts) == 1000
        plt.close('all')

    def test_hist_density_sums_to_one(self):
        fig, ax = plt.subplots()
        data = [1, 1, 2, 3, 3, 3, 4, 5]
        counts, edges, _ = ax.hist(data, bins=5, density=True)
        total = sum(c * (edges[i+1] - edges[i]) for i, c in enumerate(counts))
        assert abs(total - 1.0) < 0.1
        plt.close('all')


# ===================================================================
# 7. Scatter: parametric tests
# ===================================================================

class TestScatterParametric:
    @pytest.mark.parametrize('cmap_name', ['viridis', 'hot', 'cool', 'jet'])
    def test_scatter_cmap(self, cmap_name):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2, 3], [1, 2, 3], c=[0, 0.5, 1], cmap=cmap_name)
        assert len(pc.get_facecolors()) == 3
        plt.close('all')

    @pytest.mark.parametrize('size', [1, 10, 100, 1000])
    def test_scatter_size(self, size):
        fig, ax = plt.subplots()
        pc = ax.scatter([1, 2, 3], [1, 2, 3], s=size)
        assert pc._sizes == [size]
        plt.close('all')


# ===================================================================
# 8. GridSpec: subplots method
# ===================================================================

class TestGridSpecSubplots:
    def test_gridspec_subplots(self):
        from matplotlib.gridspec import GridSpec
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        # add_subplot with each GridSpec cell
        axes = []
        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
        assert len(axes) == 4
        assert len(fig.axes) == 4
        plt.close('all')


# ===================================================================
# 9. rcParams: rc_context preserves all
# ===================================================================

class TestRcContextPreserve:
    def test_preserve_font_size(self):
        orig = matplotlib.rcParams['font.size']
        with matplotlib.rc_context({'font.size': 20}):
            assert matplotlib.rcParams['font.size'] == 20
        assert matplotlib.rcParams['font.size'] == orig

    def test_preserve_image_cmap(self):
        orig = matplotlib.rcParams['image.cmap']
        with matplotlib.rc_context({'image.cmap': 'hot'}):
            assert matplotlib.rcParams['image.cmap'] == 'hot'
        assert matplotlib.rcParams['image.cmap'] == orig

    def test_preserve_axes_grid(self):
        orig = matplotlib.rcParams['axes.grid']
        with matplotlib.rc_context({'axes.grid': True}):
            assert matplotlib.rcParams['axes.grid'] is True
        assert matplotlib.rcParams['axes.grid'] == orig


# ===================================================================
# 10. Legend: empty/edge cases
# ===================================================================

class TestLegendEmpty:
    def test_legend_empty_axes(self):
        fig, ax = plt.subplots()
        leg = ax.legend()
        assert len(leg.get_texts()) == 0
        plt.close('all')

    def test_legend_all_underscore(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], label='_private')
        leg = ax.legend()
        assert len(leg.get_texts()) == 0
        plt.close('all')

    def test_legend_single(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], label='only_one')
        leg = ax.legend()
        assert len(leg.get_texts()) == 1
        plt.close('all')

    def test_legend_many(self):
        fig, ax = plt.subplots()
        for i in range(10):
            ax.plot([i, i+1], label=f'line{i}')
        leg = ax.legend()
        assert len(leg.get_texts()) == 10
        plt.close('all')


# ===================================================================
# 11. Transforms module re-exports
# ===================================================================

class TestTransformsImports:
    def test_all_classes_importable(self):
        from matplotlib import transforms
        assert hasattr(transforms, 'Bbox')
        assert hasattr(transforms, 'Affine2D')
        assert hasattr(transforms, 'BboxTransformTo')
        assert hasattr(transforms, 'BboxTransform')
        assert hasattr(transforms, 'BboxTransformFrom')
        assert hasattr(transforms, 'IdentityTransform')
        assert hasattr(transforms, 'ScaledTranslation')
        assert hasattr(transforms, 'blended_transform_factory')
        assert hasattr(transforms, 'nonsingular')

    def test_bbox_has_all_methods(self):
        from matplotlib.transforms import Bbox
        b = Bbox.unit()
        methods = ['from_bounds', 'from_extents', 'unit', 'null',
                   'contains', 'overlaps', 'expanded', 'translated',
                   'padded', 'shrunk', 'union', 'intersection',
                   'frozen', 'get_points', 'set_points', 'anchored']
        for m in methods:
            assert hasattr(Bbox, m) or hasattr(b, m), f"Missing: {m}"

    def test_affine2d_has_all_methods(self):
        from matplotlib.transforms import Affine2D
        a = Affine2D()
        methods = ['rotate', 'rotate_deg', 'rotate_around', 'rotate_deg_around',
                   'translate', 'scale', 'skew', 'skew_deg', 'inverted',
                   'is_identity', 'frozen', 'clear', 'set', 'get_matrix',
                   'transform', 'transform_point', 'identity']
        for m in methods:
            assert hasattr(Affine2D, m) or hasattr(a, m), f"Missing: {m}"


# ===================================================================
# 12. Additional SubplotSpec tests
# ===================================================================

class TestSubplotSpecMore:
    def test_3x3_all_positions(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3)
        for r in range(3):
            for c in range(3):
                ss = gs[r, c]
                assert ss.rowspan == (r, r + 1)
                assert ss.colspan == (c, c + 1)

    def test_subplotspec_flat_all(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 3)
        for i in range(6):
            ss = gs[i]
            expected_row = i // 3
            expected_col = i % 3
            assert ss.rowspan[0] == expected_row
            assert ss.colspan[0] == expected_col

    def test_gridspec_negative_row(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3)
        ss = gs[-1, 0]
        assert ss.rowspan == (2, 3)

    def test_gridspec_negative_col(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3)
        ss = gs[0, -1]
        assert ss.colspan == (2, 3)

    def test_gridspec_negative_both(self):
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3)
        ss = gs[-1, -1]
        assert ss.rowspan == (2, 3)
        assert ss.colspan == (2, 3)


# ===================================================================
# 13. Additional Axes position tests
# ===================================================================

class TestAxesPositionDetailed:
    def test_3x3_grid_positions(self):
        fig = plt.figure()
        for i in range(1, 10):
            fig.add_subplot(3, 3, i)
        for i, ax in enumerate(fig.axes):
            pos = ax.get_position()
            row = i // 3
            col = i % 3
            expected_w = 1.0 / 3
            expected_h = 1.0 / 3
            assert abs(pos.width - expected_w) < 1e-10
            assert abs(pos.height - expected_h) < 1e-10
        plt.close('all')

    def test_1x3_positions(self):
        fig = plt.figure()
        for i in range(1, 4):
            fig.add_subplot(1, 3, i)
        positions = [ax.get_position() for ax in fig.axes]
        for p in positions:
            assert abs(p.width - 1.0/3) < 1e-10
            assert abs(p.height - 1.0) < 1e-10
        plt.close('all')

    def test_3x1_positions(self):
        fig = plt.figure()
        for i in range(1, 4):
            fig.add_subplot(3, 1, i)
        positions = [ax.get_position() for ax in fig.axes]
        for p in positions:
            assert abs(p.width - 1.0) < 1e-10
            assert abs(p.height - 1.0/3) < 1e-10
        plt.close('all')


# ===================================================================
# 14. Additional rcParams stress
# ===================================================================

class TestRcParamsStress:
    def test_set_and_get(self):
        orig = matplotlib.rcParams['lines.linewidth']
        matplotlib.rcParams['lines.linewidth'] = 99
        assert matplotlib.rcParams['lines.linewidth'] == 99
        matplotlib.rcParams['lines.linewidth'] = orig

    def test_new_key(self):
        matplotlib.rcParams['test.custom'] = 42
        assert matplotlib.rcParams['test.custom'] == 42
        del matplotlib.rcParams['test.custom']

    def test_find_all_axes(self):
        matches = matplotlib.rcParams.find_all('axes')
        assert len(matches) > 0
        for k in matches:
            assert 'axes' in k

    def test_find_all_tick(self):
        matches = matplotlib.rcParams.find_all('tick')
        assert len(matches) > 0

    def test_find_all_legend(self):
        matches = matplotlib.rcParams.find_all('legend')
        assert len(matches) > 0

    def test_find_all_no_match(self):
        matches = matplotlib.rcParams.find_all('zzz_nonexistent')
        assert len(matches) == 0
