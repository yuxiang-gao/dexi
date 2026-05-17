# Installation

## Python users

Install the published Python package:

```bash
uv pip install dexi-py
```

Import name and distribution name differ intentionally:

```python
import dexi_py  # distribution: dexi-py
```

## Local development

```bash
just setup
just build
just test-wheel
```

`just setup` creates `.venv/` and installs development tools. `just build`
builds release wheels into `dist/`. `just test-wheel` installs the wheel into a
fresh uv environment and verifies package resources can build retargeters.

## Rust users

The Rust crate is in `crates/dexi` and can be used from the workspace:

```bash
cargo test -p dexi
```

The public Rust API centers on `RetargetingConfig` and `SeqRetargeting`.
