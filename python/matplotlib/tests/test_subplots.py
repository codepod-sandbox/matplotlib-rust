"""Tests for subplot layouts --- sharex, sharey, GridSpec, twinx, label_outer."""

import pytest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class TestSharexSharey:
    def test_sharex_true(self):
        """sharex=True links x-limits across all subplots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_xlim(0, 10)
        assert ax2.get_xlim() == (0, 10)
        plt.close('all')

    def test_sharey_true(self):
        """sharey=True links y-limits across all subplots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.set_ylim(-5, 5)
        assert ax2.get_ylim() == (-5, 5)
        plt.close('all')

    def test_sharex_bidirectional(self):
        """Setting limits on either shared axes updates the other."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax2.set_xlim(100, 200)
        assert ax1.get_xlim() == (100, 200)
        plt.close('all')

    def test_sharey_bidirectional(self):
        """Setting limits on either shared axes updates the other."""
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax2.set_ylim(10, 20)
        assert ax1.get_ylim() == (10, 20)
        plt.close('all')

    def test_sharex_grid(self):
        """sharex=True on 2x2 grid links all x-limits."""
        fig, axes = plt.subplots(2, 2, sharex=True)
        axes[0][0].set_xlim(0, 100)
        for row in axes:
            for ax in row:
                assert ax.get_xlim() == (0, 100)
        plt.close('all')

    def test_no_share_independent(self):
        """Without share, axes have independent limits."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_xlim(0, 10)
        ax2.set_xlim(100, 200)
        assert ax1.get_xlim() == (0, 10)
        assert ax2.get_xlim() == (100, 200)
        plt.close('all')


class TestTwinAxes:
    def test_twinx_creates_axes(self):
        """twinx() creates a new Axes on the same Figure."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        assert isinstance(ax2, Axes)
        assert ax2.figure is fig
        plt.close('all')

    def test_twinx_shares_x(self):
        """twinx() shares x-axis with parent."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlim(0, 10)
        assert ax2.get_xlim() == (0, 10)
        plt.close('all')

    def test_twinx_independent_y(self):
        """twinx() has independent y-axis."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_ylim(0, 10)
        ax2.set_ylim(100, 200)
        assert ax1.get_ylim() == (0, 10)
        assert ax2.get_ylim() == (100, 200)
        plt.close('all')

    def test_twiny_creates_axes(self):
        """twiny() creates a new Axes on the same Figure."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        assert isinstance(ax2, Axes)
        assert ax2.figure is fig
        plt.close('all')

    def test_twiny_shares_y(self):
        """twiny() shares y-axis with parent."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.set_ylim(0, 10)
        assert ax2.get_ylim() == (0, 10)
        plt.close('all')

    def test_twiny_independent_x(self):
        """twiny() has independent x-axis."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.set_xlim(0, 10)
        ax2.set_xlim(100, 200)
        assert ax1.get_xlim() == (0, 10)
        assert ax2.get_xlim() == (100, 200)
        plt.close('all')
