.PHONY: setup build build-ext test test-cpython test-rustpython clean

setup:
	git config core.hooksPath .githooks

# Build the RustPython interpreter binary
build:
	cargo build -p matplotlib-python

# Build _path.so for CPython (maturin or cargo fallback)
build-ext:
	cargo build -p matplotlib-path
	cp target/debug/lib_path.dylib python/matplotlib/_path.so

# CPython test run (primary dev loop)
test: build-ext
	python3 -m pytest python/matplotlib/tests/

# RustPython test run (final compatibility check)
test-rustpython: build
	target/debug/matplotlib-python -m pytest python/matplotlib/tests/

clean:
	cargo clean
