# Changelog

## 0.3.2

- Publish Linux manylinux2014 wheels (x86_64 and aarch64) for CPython 3.9-3.14.
- Update Fourier vector configs to retarget all five fingertips.
- Add Fourier/Allegro visualizer presets and per-robot transform conventions.
- Keep Allegro's four-tip mapping while making Fourier's five-tip convention the default.

## 0.3.1

- Add Fourier 6DOF and 12DOF hand URDF assets and example configs.
- Default the viser visualizers to Fourier 6DOF hand configs.
- Resolve visualizer wrist/fingertip links and URDF meshes from the selected config.

## 0.3.0

- Breaking: stop bundling YAML configs inside the wheel.
- Make `dexi_rs.load_config(path)` load direct filesystem config paths.
- Update examples to pass explicit `configs/...` paths while keeping robot assets bundled.

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
- Add Python distribution `dexi-rs` with config helpers and URDF resources.
- Add uv-runnable examples and a viser 3D point-cloud viewer.
- Add release checks, package build recipes, and Python-vs-Rust parity reports.
