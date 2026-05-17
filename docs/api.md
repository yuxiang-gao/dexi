# API reference

## Python package

Install distribution `dexi-py`; import module `dexi_py`.

### Resource helpers

```python
dexi_py.available_configs(kind: str | None = None) -> list[str]
dexi_py.config_path(name: str | Path) -> Path
dexi_py.asset_path(name: str | Path = "robots/hands") -> Path
dexi_py.load_config(name: str | Path) -> RetargetingConfig
```

### `RetargetingConfig`

```python
cfg = dexi_py.RetargetingConfig.from_file("path/to/config.yml")
cfg.type_                 # "position", "vector", or "dexpilot"
cfg.urdf_path             # path stored in YAML
cfg.target_joint_names    # optional optimized joint subset
retargeting = cfg.build()
```

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
