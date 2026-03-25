"""Patch pytest's pytester.py for RustPython compatibility.

pytester.py line 177 uses set comprehension syntax that RustPython's compiler
cannot handle in the pytest_runtest_protocol scope. Replace with equivalent
set(list-comprehension) form which RustPython handles correctly.

Usage: python3 scripts/patch-pytest-rustpython.py <pytest-install-dir>
"""
import sys
import os

target = sys.argv[1] if len(sys.argv) > 1 else "python"
path = os.path.join(target, "_pytest", "pytester.py")

if not os.path.exists(path):
    print(f"Nothing to patch: {path} not found")
    sys.exit(0)

with open(path) as f:
    content = f.read()

patched = content.replace(
    "{t[0] for t in lines2}",
    "set([t[0] for t in lines2])",
).replace(
    "{t[0] for t in lines1}",
    "set([t[0] for t in lines1])",
)

if patched == content:
    print(f"No changes needed in {path}")
else:
    with open(path, "w") as f:
        f.write(patched)
    print(f"Patched {path}")
