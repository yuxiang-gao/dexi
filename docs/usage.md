# Usage guide

## Load a config file

```python
import dexi_rs

config = dexi_rs.load_config("configs/teleop/allegro_hand_right.yml")
retargeting = config.build()
```

Configs are normal YAML files in your checkout or application. The wheel bundles
robot assets, but it does not bundle YAML configs.

## Retarget one frame

Vector and DexPilot configs consume flattened 3D vectors. Position configs
consume flattened 3D positions.

```python
target = [0.03, -0.02, 0.08, 0.04, 0.0, 0.09, 0.03, 0.02, 0.085, 0.02, 0.04, 0.07]
qpos = retargeting.retarget(target)
```

The returned `qpos` is in `retargeting.joint_names` order:

```python
for name, value in zip(retargeting.joint_names, qpos):
    print(f"{name}: {value:.4f}")
```

## Fixed joints

Some configs optimize a subset of robot joints. Pass fixed joint values when
`retargeting.fixed_dof > 0`:

```python
fixed = [0.0] * retargeting.fixed_dof
qpos = retargeting.retarget(target, fixed_qpos=fixed)
```

## Sequential behavior

`SeqRetargeting` stores the previous target-joint solution and uses it as the
next optimizer warm start. Use `reset()` when starting a new stream:

```python
retargeting.reset()
```

`set_qpos()` can seed the internal state from a full robot qpos, and
`get_qpos()` reconstructs the current full robot qpos.

## Link points for visualization

The Python binding exposes world-frame link positions for a full qpos:

```python
links = ["wrist", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
points = retargeting.link_positions(qpos, links)
```

See `examples/visualize_viser.py` for an interactive viser scene with the
retargeted URDF hand model, wrist axis, fingertip targets, actual fingertips,
and residual lines.

For live perception input, `examples/vector_retarget_viser.py` reads a video or
webcam with OpenCV, tracks a human hand with MediaPipe, retargets the vector
targets, and visualizes the camera frame, tracked landmarks, robot hand, targets,
actual fingertips, and residuals in viser.
