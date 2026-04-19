import importlib.util
import sys
from pathlib import Path

from .registry import BackendFilter, backend_registry  # noqa: F401


def _load_backend_agg_fallback():
    name = "matplotlib.backends._backend_agg"
    if name in sys.modules:
        return

    ext_path = Path(__file__).with_name("_backend_agg_ext.cpython-312-darwin.so")
    if not ext_path.exists():
        return

    spec = importlib.util.spec_from_file_location(name, ext_path)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_load_backend_agg_fallback()

# NOTE: plt.switch_backend() (called at import time) will add a "backend"
# attribute here for backcompat.
_QT_FORCE_QT5_BINDING = False
