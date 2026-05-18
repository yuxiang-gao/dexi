# Installation

## Python users

Install the published Python package:

```bash
uv pip install dexi-rs
```

Import with the matching module name:

```python
import dexi_rs  # distribution: dexi-rs
```

Robot assets ship with the package. YAML configs are regular files; in a source
checkout, examples use paths such as `configs/teleop/allegro_hand_right.yml`.

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
