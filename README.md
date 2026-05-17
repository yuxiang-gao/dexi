# dexi

Fast hand retargeting in Rust, with Python bindings published as `dexi-py`.

`dexi` loads robot URDFs and YAML retargeting configs, then maps human-hand
position/vector targets onto robot-hand joint positions. It supports position,
vector, and DexPilot-style retargeting for the bundled hands.

## Highlights

- Pure Rust retargeting core with a Python API (`import dexi_py`).
- Bundled YAML configs and URDFs for common robot hands.
- Python-vs-Rust parity tooling for maintainer validation.
- UV-runnable examples, including a `viser` 3D point-cloud viewer.

## Install

```bash
uv pip install dexi-py
```

For local development from a checkout:

```bash
just setup
just build
just test-wheel
```

## Quickstart

```python
import numpy as np
import dexi_py

config = dexi_py.load_config("teleop/allegro_hand_right.yml")
retargeting = config.build()

# Allegro vector configs expect four wrist-to-fingertip vectors, flattened.
target_vectors = np.array(
    [
        [0.03, -0.02, 0.08],
        [0.04, 0.00, 0.09],
        [0.03, 0.02, 0.085],
        [0.02, 0.04, 0.07],
    ],
    dtype=float,
)

qpos = retargeting.retarget(target_vectors.reshape(-1).tolist())
print(dict(zip(retargeting.joint_names, qpos)))
```

Run the same example through uv:

```bash
uv run examples/quickstart.py
```

## Supported hands and configs

Bundled configs live under `configs/` in this repository and inside the Python
wheel under `dexi_py.resources.configs`.

| Family | Configs |
| --- | --- |
| Allegro | left/right, vector, position, DexPilot |
| Shadow | left/right, vector, position, DexPilot |
| Schunk SVH | left/right, vector, position, DexPilot |
| LEAP | left/right, vector, position, DexPilot |
| Ability | left/right, vector, position, DexPilot |
| Inspire | left/right, vector, position, DexPilot |
| Panda gripper | vector, position, DexPilot |

List available package configs:

```python
import dexi_py
print(dexi_py.available_configs())
```

## Examples

All examples are in `examples/` and are designed to run with uv.

```bash
uv run examples/list_configs.py
uv run examples/quickstart.py
uv run examples/batch_retarget.py
uv run examples/joint_order.py
uv run examples/visualize_viser.py --smoke-test
uv run examples/visualize_viser.py
```

The non-smoke `visualize_viser.py` command starts a local browser-based 3D
point-cloud viewer. It does not render videos or SVG files.

## Documentation

- [Installation](docs/installation.md)
- [Usage guide](docs/usage.md)
- [Configuration guide](docs/configuration.md)
- [API reference](docs/api.md)
- [Examples guide](docs/examples.md)
- [FAQ](docs/faq.md)
- [Release and publishing](docs/release.md)

## Development

```bash
just setup
just fmt
just test
just build
just check-dist
just test-wheel
just examples
just preflight
```

## License and notices

Code is distributed under the MIT License. Bundled robot configs and URDF assets
include third-party materials; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
