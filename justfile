set shell := ["bash", "-cu"]

PYTHON := ".venv/bin/python"

setup:
    uv venv .venv
    uv pip install --python {{PYTHON}} maturin twine numpy pyyaml viser yourdfpy

fmt:
    cargo fmt --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test
    python3 -m py_compile scripts/*.py examples/*.py

build:
    rm -rf dist
    find crates/dexi-py/python -type d -name __pycache__ -prune -exec rm -rf {} +
    find crates/dexi-py/python -type f -name '*.pyc' -delete
    uvx maturin build --release -m crates/dexi-py/Cargo.toml --out dist
    uvx maturin sdist -m crates/dexi-py/Cargo.toml --out dist

check-dist:
    uvx twine check dist/*
    python3 scripts/check_release.py --dist

test-wheel:
    python3 scripts/test_wheel.py

examples:
    uv pip install --python {{PYTHON}} --force-reinstall dist/*.whl numpy pyyaml viser yourdfpy
    {{PYTHON}} examples/list_configs.py
    {{PYTHON}} examples/quickstart.py
    {{PYTHON}} examples/python_config.py
    {{PYTHON}} examples/batch_retarget.py
    {{PYTHON}} examples/joint_order.py
    {{PYTHON}} examples/visualize_viser.py --smoke-test

preflight: fmt lint test build check-dist test-wheel examples

build-matrix:
    python3 scripts/build_matrix.py

publish:
    python3 scripts/check_release.py --publish --dist
    python3 scripts/publish_pypi.py

# check-dist already ran in preflight, and just runs each dependency once per
# invocation, so re-check the full matrix in a fresh just process.
release: preflight build-matrix
    {{just_executable()}} check-dist
    {{just_executable()}} publish
