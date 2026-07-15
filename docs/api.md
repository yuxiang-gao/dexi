# API reference

## Python package

Install distribution `dexi-rs`; import module `dexi_rs`.

### Resource helpers

```python
dexi_rs.available_configs(root: str | Path = "configs", kind: str | None = None) -> list[str]
dexi_rs.config_path(name: str | Path) -> Path
dexi_rs.asset_path(name: str | Path = "robots/hands") -> Path
dexi_rs.load_config(source: str | Path | dict | RetargetingConfig) -> RetargetingConfig
```

`load_config()` accepts direct filesystem paths, dicts (as constructor kwargs or nested
under a `"retargeting"` key), or existing config objects. The wheel bundles robot assets, not YAML configs.

### `RetargetingConfig`

```python
cfg = dexi_rs.RetargetingConfig.from_file("path/to/config.yml")
cfg.type_                 # "position", "vector", or "dexpilot"
cfg.urdf_path             # path stored in YAML
cfg.target_joint_names    # optional optimized joint subset
retargeting = cfg.build()
```

Configs can also be defined directly in Python — no YAML file needed.
Relative `urdf_path` values resolve against the robot assets bundled in the
wheel, then the current working directory:

```python
cfg = dexi_rs.RetargetingConfig(
    type="vector",                                      # position | vector | dexpilot
    urdf_path="allegro_hand/allegro_hand_right.urdf",   # bundled asset
    target_origin_link_names=["wrist"] * 4,
    target_task_link_names=["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"],
    target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
    scaling_factor=1.6,
)
retargeting = cfg.build()
```

Required fields depend on `type`; dimensions are validated at construction
and raise `ValueError` with the offending field named. Configs are
immutable: all fields are readable attributes, and variants are created by
constructing new objects.

### `SeqRetargeting`

```python
retargeting.retarget(ref_value: list[float], fixed_qpos: list[float] | None = None) -> list[float]
retargeting.reset() -> None
retargeting.set_qpos(robot_qpos: list[float]) -> None
retargeting.get_qpos(fixed_qpos: list[float] | None = None) -> list[float]
retargeting.link_positions(robot_qpos: list[float], link_names: list[str]) -> list[tuple[float, float, float]]

retargeting.joint_names   # full qpos joint order
retargeting.link_names    # all parsed URDF link names
retargeting.fixed_dof     # fixed-joint count expected by retarget()
retargeting.target_dof    # optimized target-joint count
```

## Rust crate

The Rust crate exposes the same core concepts:

```rust
use dexi::RetargetingConfig;

let config = RetargetingConfig::load_from_file("configs/teleop/allegro_hand_right.yml")?;
let mut retargeting = config.build()?;
let qpos = retargeting.retarget(&target_values, &[]);
```

Important modules:

- `retargeting_config`: YAML loading and builder.
- `seq_retarget`: stateful sequential retargeting wrapper.
- `optimizer`: position, vector, and DexPilot objectives.
- `robot_wrapper`: URDF kinematics, joint/link lookup, link positions.
- `kinematics_adaptor`: mimic-joint handling.
