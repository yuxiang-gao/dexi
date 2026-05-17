# Dexi retargeting change record

## Scope

- Reimplemented the `reference/dex-retargeting` retargeting path in pure Rust.
- Added a Python extension module for driving the Rust implementation from Python.
- Added an end-to-end Python-vs-Rust comparison workflow for every supported hand/config.
- Added a rendered HTML report for qualitative and quantitative comparison review.

## Implementation notes

- Rust workspace: `crates/dexi` and `crates/dexi-py`.
- Core components: URDF parsing, tree forward kinematics, analytic world-position Jacobians, mimic-joint handling, dummy free joints, low-pass filtering, retargeting config parsing, sequence retargeting, and position/vector/DexPilot optimizers.
- Parity fixes align Rust with the Python reference on URDF RPY transforms, Pinocchio-style joint ordering, mimic output reconstruction, SLSQP bounded optimization, joint-limit epsilon handling, and position/vector/DexPilot objective scaling.
- Python binding exposes `RetargetingConfig`, `SeqRetargeting`, config loading, qpos state methods, and retargeting calls.
- Comparison tooling uses deterministic feasible FK-derived reference inputs by default and reports shape, finite-value, norm, mean absolute, max absolute, and RMS differences. Feasible probes avoid comparing arbitrary unreachable Cartesian targets where multiple joint-space IK solutions can be equally valid.

## Verification record

- `cargo fmt && cargo test` passes for the full workspace.
- Python-vs-Rust comparison completed for `39/39` reference configs.
- No comparison runtime errors were reported.
- Current passing comparison: worst mean absolute error `0.0183839`, worst max absolute error `0.0552179`, and `0` configs exceed the reporting thresholds (`mean_abs <= 0.05`, `max_abs <= 0.5`).

## Report artifacts

- HTML report: `docs/reports/python-rust-comparison.html`
- JSON comparison data: `docs/reports/python-rust-comparison.json`
- CSV comparison data: `docs/reports/python-rust-comparison.csv`
