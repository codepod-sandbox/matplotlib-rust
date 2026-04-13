"""
Stub for matplotlib._c_internal_utils (C extension) for RustPython/WASM.

All functions return safe defaults for a headless/sandbox environment.
"""


def display_is_valid():
    """Return False — no display in WASM sandbox."""
    return False


def xdisplay_is_valid():
    """Return False — no X display in WASM sandbox."""
    return False


# Windows-only stubs
def Win32_GetCurrentProcessExplicitAppUserModelID():
    return None


def Win32_SetCurrentProcessExplicitAppUserModelID(appid):
    pass


def Win32_GetForegroundWindow():
    return None


def Win32_SetForegroundWindow(hwnd):
    pass
