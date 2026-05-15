# Dexi: Pure Rust Hand Retargeting Library

## Overview

Reimplement `dex-retargeting` (Python) as a pure Rust library (`dexi`) with Python bindings via PyO3/maturin. The library retargets human hand poses to robot hand joint angles using constrained nonlinear optimization (SLSQP).

## Technology Stack

| Component | Crate | Purpose |
|---|---|---|
| URDF parsing | `urdf-rs` | Parse robot URDF files with mimic joints |
| FK + Jacobian | `k` | Forward kinematics and 6xN Jacobian via per-finger SerialChains |
| Linear algebra | `nalgebra` | SE(3) transforms, matrix operations |
| Optimizer | `slsqp` | SLSQP with box constraints (pure Rust, transpiled from NLopt) |
| YAML config | `serde_yaml` | Parse retargeting config files |
| Python bindings | `pyo3` + `maturin` | Expose Rust API to Python |
| Error handling | `anyhow` / `thiserror` | Result types |

## Project Structure

```
dexi/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── dexi/                     # core Rust library
│   │   ├── Cargo.toml            # name = "dexi"
│   │   └── src/
│   │       ├── lib.rs            # re-exports
│   │       ├── robot.rs          # RobotWrapper (FK, Jacobian via k crate)
│   │       ├── chain_builder.rs  # Build per-finger SerialChains from URDF
│   │       ├── optimizer.rs      # Optimizer trait + Position/Vector/DexPilot
│   │       ├── seq_retargeting.rs # Sequential retargeting with warm-start
│   │       ├── mimic.rs          # MimicJointAdaptor
│   │       ├── config.rs         # RetargetingConfig from YAML
│   │       ├── constants.rs      # RobotName, RetargetingType, HandType enums
│   │       ├── urdf_ext.rs       # URDF extensions (dummy joints, tree building)
│   │       └── filter.rs         # Low-pass filter
│   └── dexi-py/                  # Python bindings
│       ├── Cargo.toml            # crate-type = ["cdylib"]
│       ├── pyproject.toml        # maturin build system
│       └── src/
│           └── lib.rs            # PyO3 module + class wrappers
├── configs/                      # YAML retargeting configs (copied from reference)
│   ├── teleop/                   # real-time teleoperation configs
│   └── offline/                  # offline data processing configs
├── tests/
│   ├── test_optimizer.rs         # Rust tests matching Python test_optimizer.py
│   ├── test_config.rs            # Config parsing tests
│   └── comparison/
│       ├── compare_results.py    # Run both, compare outputs numerically
│       └── visualize.py          # Side-by-side visual comparison
├── assets/ -> reference/dex-retargeting/assets/  # symlink to URDF files
└── reference/                    # original Python implementation (read-only)
```

## Architecture

### Robot / Kinematics Layer (`robot.rs`, `chain_builder.rs`, `urdf_ext.rs`)

**RobotWrapper** wraps `k::Chain<f64>`:

```rust
pub struct RobotWrapper {
    chain: Chain<f64>,
    joint_limits: Vec<(f64, f64)>,
    dof: usize,
    dof_joint_names: Vec<String>,
    link_name_to_node: HashMap<String, Node<f64>>,
    serial_chains: HashMap<String, SerialChain<f64>>,
}
```

Methods:
- `from_urdf_path(path)` — load URDF via `k::Chain::from_urdf_file`
- `forward_kinematics(&mut self, qpos: &[f64])` — update all transforms
- `link_pose(&self, link_name: &str) -> Isometry3<f64>` — SE(3) transform
- `link_jacobian(&mut self, link_name: &str, qpos: &[f64]) -> DMatrix<f64>` — 6xN Jacobian
- `joint_limits(&self) -> &[(f64, f64)]`
- `joint_names(&self) -> &[String]`

**chain_builder.rs** extracts `SerialChain<f64>` for each target link:
- Walk the tree from root to target link
- Build SerialChain containing only the active joints along that path
- Cache in `serial_chains` HashMap

**urdf_ext.rs** handles dummy joint insertion:
- When `add_dummy_free_joint = true`, prepend 6 joints (3 prismatic XYZ + 3 revolute XYZ)
- Write modified URDF to temp file, pass to `k::Chain::from_urdf_file`

### Mimic Joint Adaptor (`mimic.rs`)

```rust
pub struct MimicJointAdaptor {
    mappings: Vec<MimicMapping>,
}

struct MimicMapping {
    source_idx: usize,
    mimic_idx: usize,
    multiplier: f64,
    offset: f64,
}
```

- `forward_qpos(qpos: &mut [f64])` — apply `q[mimic] = q[source] * mult + offset`
- `backward_jacobian(jacobian: &mut DMatrix<f64>)` — chain rule: `J[.., source] += J[.., mimic] * mult`

### Optimizers (`optimizer.rs`)

**Core trait:**

```rust
pub trait Optimizer {
    fn retarget(
        &mut self,
        ref_value: &[f64],
        fixed_qpos: &[f64],
        last_qpos: &[f64],
    ) -> Vec<f64>;
}
```

**PositionOptimizer** — minimizes 3D position error:
- Objective: `huber_loss(body_pos(q), target_pos) + normal_delta * ||q - q_last||^2`
- Gradient: `J^T @ grad_pos + 2 * normal_delta * (q - q_last)`
- Huber gradient (analytical, no autograd): `sign(diff)` if `|diff| > delta`, else `diff / delta`

**VectorOptimizer** — minimizes bone vector differences:
- Objective: `huber_loss(||v_robot - v_target||, 0)`
- Same gradient structure via Jacobian

**DexPilotOptimizer** — extends Vector with projection:
- Generate finger-pair vectors + wrist-fingertip vectors
- Projection: when distance < `project_dist` (0.03), replace target direction * eta
- Hysteresis: escape at `escape_dist` (0.05)
- Weights: normal=1, projected S1=200, projected S2=400, wrist=(num_pairs+num_fingers)

All use `slsqp::minimize()` with box constraints from joint limits.

### Sequential Retargeting (`seq_retargeting.rs`)

```rust
pub struct SeqRetargeting {
    optimizer: Box<dyn Optimizer>,
    last_qpos: Vec<f64>,
    joint_limits: Vec<(f64, f64)>,
    lp_filter: Option<LpFilter>,
    adaptor: Option<MimicJointAdaptor>,
    idx_target: Vec<usize>,
    idx_fixed: Vec<usize>,
}
```

- `retarget(ref_value, fixed_qpos)` — optimize + reassemble + filter
- `warm_start(wrist_pos, wrist_quat, hand_type)` — analytical 6-DOF init from wrist pose
- `reset()` — back to joint midpoints
- `joint_names() -> &[String]`

### Config (`config.rs`)

```rust
pub struct RetargetingConfig {
    pub retargeting_type: RetargetingType,
    pub urdf_path: PathBuf,
    pub add_dummy_free_joint: bool,
    pub target_joint_names: Option<Vec<String>>,
    pub target_link_names: Option<Vec<String>>,
    pub target_link_human_indices: Option<Vec<usize>>,
    pub target_origin_link_names: Option<Vec<String>>,
    pub target_task_link_names: Option<Vec<String>>,
    pub finger_tip_link_names: Option<Vec<String>>,
    pub wrist_link_name: Option<String>,
    pub scaling_factor: f64,
    pub normal_delta: f64,
    pub huber_delta: f64,
    pub low_pass_alpha: Option<f64>,
    pub has_joint_limits: bool,
    pub ignore_mimic_joint: bool,
    // DexPilot params
    pub project_dist: f64,
    pub escape_dist: f64,
}
```

- `load_from_file(path) -> Result<Self>` — parse YAML
- `build(&self) -> Result<SeqRetargeting>` — factory: load URDF, create robot, optimizer, adaptor

### Constants (`constants.rs`)

```rust
pub enum RobotName { Allegro, Shadow, Svh, Leap, Ability, Inspire, Panda }
pub enum RetargetingType { Vector, Position, DexPilot }
pub enum HandType { Right, Left }

pub const OPERATOR2MANO_RIGHT: [[f64; 3]; 3] = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
pub const OPERATOR2MANO_LEFT: [[f64; 3]; 3] = [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
```

### Python Bindings (`dexi-py/`)

Expose idiomatic Rust API via PyO3:

```python
import dexi

config = dexi.RetargetingConfig.from_file("configs/teleop/allegro_hand_right.yml")
retargeting = config.build()

qpos = retargeting.retarget(ref_value, fixed_qpos)
names = retargeting.joint_names()
```

Classes: `RetargetingConfig`, `SeqRetargeting`, enums `RobotName`, `RetargetingType`, `HandType`.

Built with maturin, published as `dexi` Python package.

## Testing & Validation

### Rust Unit Tests

1. **test_optimizer.rs** — parameterized over all robot/hand combos
   - Generate ground truth: random qpos -> FK -> positions/vectors
   - Run optimizer, assert error < 1e-2
   - Test all 3 optimizer types (Position, Vector, DexPilot)

2. **test_config.rs** — config loading
   - Parse all YAML configs, verify optimizer types
   - Test dict-based config creation
   - Test dummy joint insertion

### Numerical Comparison (`tests/comparison/compare_results.py`)

- Run Python `dex_retargeting` and Rust `dexi` on identical inputs
- Compare joint angle outputs, report max/mean error
- Assert tolerance: joint angle difference < 1e-3

### Visual Comparison (`tests/comparison/visualize.py`)

- Load robot URDF, show Python vs Rust retargeted poses side by side
- Generate comparison images per robot/hand

## Supported Robots

| Robot | DOF | Fingers | Left/Right |
|---|---|---|---|
| Allegro Hand | 16 | 4 | Both |
| Shadow Hand | 24+ | 5 | Both |
| Schunk SVH | 9 | 5 | Both |
| LEAP Hand | 16 | 4 | Both |
| Ability Hand | 6 | 5 | Both |
| Inspire Hand | 6 | 5 | Both |
| Panda Gripper | 1-2 | 2 | Single |

## Key Differences from Python Implementation

1. **No PyTorch** — analytical Huber loss gradient replaces autograd
2. **No Pinocchio** — `k` crate provides FK/Jacobian
3. **No NLopt C library** — `slsqp` crate (pure Rust SLSQP)
4. **No yourdfpy** — `urdf-rs` handles URDF parsing
5. **Single-pass URDF** — parse once, no temp file needed (except for dummy joints)
6. **Idiomatic Rust API** — not a 1:1 mirror of Python API

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `k` crate Jacobian convention differs from Pinocchio | Verify in unit tests with known transforms; adjust rotation if needed |
| `slsqp` crate is young (v0.1.x) | Same algorithm as NLopt SLSQP; numerical comparison tests catch divergence |
| SerialChain extraction for branching hands | Test on all supported robots; validate against Python FK results |
| Joint ordering differs between k and Pinocchio | Expose joint names; create index remapping in comparison tests |
