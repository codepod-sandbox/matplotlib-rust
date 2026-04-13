"""Pytest fixtures for matplotlib tests."""
import pytest


@pytest.fixture(autouse=True)
def mpl_test_settings():
    """Reset matplotlib state before/after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


@pytest.fixture
def pd():
    """Fixture that provides pandas, skipping if not installed."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        pytest.skip("pandas not installed")
