"""Patch pytest for RustPython compatibility.

RustPython's compiler fails on set comprehensions ({x for x in ...}) in certain
scopes with "no symbol table available". Replace ALL set comprehensions in
pytester.py with equivalent set(list-comprehension) form.

Usage: python3 scripts/patch-pytest-rustpython.py <pytest-install-dir>
"""
import re
import sys
import os

target = sys.argv[1] if len(sys.argv) > 1 else "python"
path = os.path.join(target, "_pytest", "pytester.py")

if not os.path.exists(path):
    print(f"Nothing to patch: {path} not found")
    sys.exit(0)

with open(path) as f:
    content = f.read()

# Replace {expr for var in iterable} with set([expr for var in iterable])
# This handles simple single-clause set comprehensions.
original = content
content = re.sub(
    r'\{([^{}]+\bfor\b[^{}]+)\}',
    lambda m: f'set([{m.group(1)}])',
    content,
)

if content == original:
    print(f"No set comprehensions found in {path}")
else:
    with open(path, "w") as f:
        f.write(content)
    count = len(re.findall(r'set\(\[', content)) - len(re.findall(r'set\(\[', original))
    print(f"Patched {path}: replaced set comprehensions with set([...]) form")
