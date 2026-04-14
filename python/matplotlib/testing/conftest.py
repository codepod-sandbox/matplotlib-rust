"""Pytest fixtures for matplotlib tests."""
import pytest


@pytest.fixture(autouse=True)
def mpl_test_settings():
    """Reset matplotlib state before/after each test."""
    import matplotlib
    import matplotlib.pyplot as plt
    # Save rcParams before test
    original_rcparams = dict(matplotlib.rcParams)
    yield
    plt.close('all')
    # Restore rcParams after test
    matplotlib.rcParams.clear()
    matplotlib.rcParams.update(original_rcparams)


@pytest.fixture
def pd():
    """Fixture that provides pandas, skipping if not installed."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        pytest.skip("pandas not installed")
