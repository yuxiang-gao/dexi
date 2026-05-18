# Changelog

## 0.2.0

- Breaking: rename the Python import package from `dexi_py` to `dexi_rs`.
- Keep the published distribution name and import name aligned.
- Remove the old import path instead of shipping a compatibility shim.

## 0.1.2

- Upgrade the viser example to load the Allegro URDF mesh model.
- Apply the retargeted joint configuration to the displayed hand.
- Overlay wrist axes, fingertip targets, retargeted fingertips, and residual lines.

## 0.1.1

- Add Rust retargeting core for position, vector, and DexPilot objectives.
- Add Python distribution `dexi-rs` with bundled configs and URDF resources.
- Add uv-runnable examples and a viser 3D point-cloud viewer.
- Add release checks, package build recipes, and Python-vs-Rust parity reports.
