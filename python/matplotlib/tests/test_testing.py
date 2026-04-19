import warnings
from types import SimpleNamespace

import pytest

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal


@pytest.mark.filterwarnings("error:This should fail the test")
def test_warn_to_fail():
    with pytest.raises(UserWarning, match="This should fail the test"):
        warnings.warn("This should fail the test")


@pytest.mark.parametrize("a", [1])
@check_figures_equal(extensions=["png"])
@pytest.mark.parametrize("b", [1])
def test_parametrize_with_check_figure_equal(a, fig_ref, b, fig_test):
    assert a == b


def test_wrap_failure():
    with pytest.raises(ValueError, match="^The decorated function"):
        @check_figures_equal()
        def should_fail(test, ref):
            pass


def test_check_figures_equal_extra_fig():
    @check_figures_equal(extensions=["png"])
    def should_fail(fig_test, fig_ref):
        plt.figure()

    request = SimpleNamespace(
        node=SimpleNamespace(name="test_check_figures_equal_extra_fig")
    )
    with pytest.raises(RuntimeError, match="Number of open figures changed"):
        should_fail(ext="png", request=request)


@check_figures_equal()
def test_check_figures_equal_closed_fig(fig_test, fig_ref):
    fig = plt.figure()
    plt.close(fig)
