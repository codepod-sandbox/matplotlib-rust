import sys

import pytest

import matplotlib
from matplotlib import _api


def pytest_configure(config):
    for key, value in [
        ("markers", "flaky: (Provided by pytest-rerunfailures.)"),
        ("markers", "timeout: (Provided by pytest-timeout.)"),
        ("markers", "backend: Set alternate Matplotlib backend temporarily."),
        ("markers", "baseline_images: Compare output against references."),
        ("markers", "pytz: Tests that require pytz to be installed."),
        ("filterwarnings", "error"),
        ("filterwarnings",
         "ignore:.*The py23 module has been deprecated:DeprecationWarning"),
        ("filterwarnings",
         r"ignore:DynamicImporter.find_spec\(\) not found; "
         r"falling back to find_module\(\):ImportWarning"),
    ]:
        config.addinivalue_line(key, value)

    matplotlib.use('agg', force=True)
    matplotlib._called_from_pytest = True
    matplotlib._init_tests()


def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False


@pytest.fixture(autouse=True)
def mpl_test_settings(request):
    from matplotlib.testing.decorators import _cleanup_cm

    with _cleanup_cm():
        backend = None
        backend_marker = request.node.get_closest_marker('backend')
        prev_backend = matplotlib.get_backend()
        if backend_marker is not None:
            assert len(backend_marker.args) == 1, \
                "Marker 'backend' must specify 1 backend."
            backend, = backend_marker.args
            skip_on_importerror = backend_marker.kwargs.get(
                'skip_on_importerror', False)

            if backend.lower().startswith('qt5'):
                if any(sys.modules.get(k) for k in ('PyQt4', 'PySide')):
                    pytest.skip('Qt4 binding already imported')

        matplotlib.testing.setup()
        with _api.suppress_matplotlib_deprecation_warning():
            if backend is not None:
                import matplotlib.pyplot as plt
                try:
                    plt.switch_backend(backend)
                except ImportError as exc:
                    if 'cairo' in backend.lower() or skip_on_importerror:
                        pytest.skip("Failed to switch to backend "
                                    f"{backend} ({exc}).")
                    raise
            matplotlib.style.use(["classic", "_classic_test_patch"])
        try:
            yield
        finally:
            if backend is not None:
                plt.close("all")
                matplotlib.use(prev_backend)


@pytest.fixture
def pd():
    pd = pytest.importorskip('pandas')
    try:
        from pandas.plotting import (
            deregister_matplotlib_converters as deregister)
        deregister()
    except ImportError:
        pass
    return pd


@pytest.fixture
def xr():
    return pytest.importorskip('xarray')


@pytest.fixture
def text_placeholders():
    # Compatibility fixture used by upstream image tests that only need the
    # fixture to exist in the shared test harness.
    yield


@pytest.fixture
def high_memory():
    # Compatibility fixture for upstream tests that gate large-memory
    # regression scenarios through the shared test harness.
    yield
