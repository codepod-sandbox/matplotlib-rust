# Copyright (c) 2024 CodePod Contributors — BSD 3-Clause License
# Ported from upstream matplotlib tests/test_cm.py and tests/test_colors.py
"""Upstream-ported tests for colormaps, registry, ScalarMappable, and norm classes."""
import numpy as np
import numpy.testing as npt
import pytest


def test_rcparam_image_cmap():
    import matplotlib
    assert matplotlib.rcParams['image.cmap'] == 'viridis'
    assert matplotlib.rcParams['image.lut'] == 256
