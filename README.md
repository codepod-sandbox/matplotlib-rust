# matplotlib-py

Pure-Python matplotlib subset for the codepod sandbox.

Provides a minimal `matplotlib.pyplot` API that renders charts to SVG or PIL images, suitable for running inside a WebAssembly sandbox.

## Structure

```
python/matplotlib/
├── __init__.py
├── pyplot.py
├── figure.py
├── axes.py
├── colors.py
├── _svg_backend.py
└── _pil_backend.py
```
