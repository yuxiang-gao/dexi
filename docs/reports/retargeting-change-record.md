# Dexi retargeting change record

## Scope

- Reimplemented the `reference/dex-retargeting` retargeting path in pure Rust.
- Added a Python extension module for driving the Rust implementation from Python.
- Added an end-to-end Python-vs-Rust comparison workflow for every supported hand/config.
- Added a rendered HTML report for qualitative and quantitative comparison review.

## Implementation notes

- Rust workspace: `crates/dexi` and `crates/dexi-py`.
- Core components: URDF parsing, tree forward kinematics, finite-difference Jacobians, mimic-joint handling, dummy free joints, low-pass filtering, retargeting config parsing, sequence retargeting, and position/vector/DexPilot optimizers.
- Python binding exposes `RetargetingConfig`, `SeqRetargeting`, config loading, qpos state methods, and retargeting calls.
- Comparison tooling uses deterministic synthetic reference inputs and reports shape, finite-value, norm, mean absolute, max absolute, and RMS differences.

## Verification record

- `cargo fmt && cargo test` passes for the full workspace.
- Python-vs-Rust comparison completed for `39/39` reference configs.
- No comparison runtime errors were reported.
- Current numerical drift remains significant for most configs: worst mean absolute error `1.66135`, worst max absolute error `5.12438`, and `37` configs exceed the reporting thresholds.

## Report artifacts

- HTML report: `docs/reports/python-rust-comparison.html`
- JSON comparison data: `docs/reports/python-rust-comparison.json`
- CSV comparison data: `docs/reports/python-rust-comparison.csv`
