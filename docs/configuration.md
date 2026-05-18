# Configuration guide

Dexi configs are YAML files with a top-level `retargeting:` block.

## Common fields

| Field | Meaning |
| --- | --- |
| `type` | `position`, `vector`, or `DexPilot` |
| `urdf_path` | Relative path under `assets/robots/hands` or an absolute path |
| `target_joint_names` | Optional optimized joint subset; `null` means all DOF joints |
| `target_link_human_indices` | Mapping from input hand joints/vectors to robot links |
| `scaling_factor` | Vector/DexPilot target scaling |
| `normal_delta` | Temporal regularization strength |
| `huber_delta` | Smooth-L1/Huber loss transition |
| `low_pass_alpha` | Output smoothing (`1` disables smoothing) |
| `ignore_mimic_joint` | Disable URDF mimic-joint handling |

## Position configs

Position configs retarget absolute 3D link positions and use:

- `target_link_names`
- `add_dummy_free_joint` when a free base should be optimized

## Vector configs

Vector configs retarget 3D vectors between origin/task links and use:

- `target_origin_link_names`
- `target_task_link_names`

## DexPilot configs

DexPilot configs retarget fingertip vectors with projection behavior for grasping
and use:

- `wrist_link_name`
- `finger_tip_link_names`
- `project_dist`
- `escape_dist`

## Resource layout

Repository layout:

```text
configs/{offline,teleop}/*.yml
assets/robots/hands/*/*.urdf
```

Installed package layout is exposed through `dexi_rs.config_path()` and
`dexi_rs.asset_path()`.
