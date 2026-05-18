# dexi-rs

`dexi-rs` is a fast hand-retargeting package backed by a Rust engine. It loads
bundled robot-hand YAML configs and URDFs, then maps 3D human-hand targets to
robot-hand joint positions.

Install distribution `dexi-rs`; import module `dexi_rs`.

```bash
uv pip install dexi-rs
```

## Quickstart

```python
import dexi_rs

cfg = dexi_rs.load_config("teleop/allegro_hand_right.yml")
retargeting = cfg.build()

# Four wrist-to-fingertip vectors for this Allegro vector config.
target = [
    0.03, -0.02, 0.08,
    0.04, 0.00, 0.09,
    0.03, 0.02, 0.085,
    0.02, 0.04, 0.07,
]
qpos = retargeting.retarget(target)
print(dict(zip(retargeting.joint_names, qpos)))
```

## Resource helpers

```python
import dexi_rs

print(dexi_rs.available_configs())
config_path = dexi_rs.config_path("teleop/allegro_hand_right.yml")
asset_root = dexi_rs.asset_path("robots/hands")
```

## Retargeting API

- `RetargetingConfig.from_file(path)` loads an external YAML config.
- `dexi_rs.load_config(name)` loads a bundled config.
- `config.build()` creates a stateful `SeqRetargeting` instance.
- `retargeting.retarget(ref_value, fixed_qpos=None)` returns full robot qpos in
  `retargeting.joint_names` order.
- `retargeting.reset()`, `set_qpos()`, and `get_qpos()` control sequence state.
- `retargeting.link_positions(qpos, link_names)` returns world-frame 3D points
  for visualization.

Bundled config families cover Allegro, Shadow, Schunk SVH, LEAP, Ability,
Inspire, and Panda gripper hands for position, vector, and DexPilot-style
retargeting where available.
