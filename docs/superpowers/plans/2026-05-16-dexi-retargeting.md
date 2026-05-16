# Dexi Retargeting Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement dex-retargeting Python library as pure Rust crate `dexi` with PyO3 Python bindings.

**Architecture:** Workspace with `crates/dexi/` (core library) and `crates/dexi-py/` (PyO3 bindings). Core uses `k` crate for FK/Jacobian, `slsqp` for constrained optimization, `nalgebra` for linear algebra. Analytical Huber loss gradients replace PyTorch autograd. Per-finger `SerialChain` extraction from branching URDF tree.

**Tech Stack:** Rust, urdf-rs, k, slsqp, nalgebra, serde_yaml, PyO3, maturin

---

### Task 1: Scaffold Cargo Workspace

**Files:**
- Create: `Cargo.toml`
- Create: `crates/dexi/Cargo.toml`
- Create: `crates/dexi/src/lib.rs`
- Create: `crates/dexi-py/Cargo.toml`
- Create: `crates/dexi-py/pyproject.toml`
- Create: `crates/dexi-py/src/lib.rs`

- [ ] **Step 1: Create workspace root Cargo.toml**

`Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["crates/dexi", "crates/dexi-py"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
```

- [ ] **Step 2: Create core crate Cargo.toml**

`crates/dexi/Cargo.toml`:
```toml
[package]
name = "dexi"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
nalgebra = "0.33"
k = "0.32"
urdf-rs = "0.7"
slsqp = "0.2"
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
anyhow = "1"
thiserror = "2"
log = "0.4"
```

- [ ] **Step 3: Create core lib.rs stub**

`crates/dexi/src/lib.rs`:
```rust
pub mod constants;
pub mod filter;
pub mod mimic;
pub mod robot;
pub mod chain_builder;
pub mod urdf_ext;
pub mod optimizer;
pub mod seq_retargeting;
pub mod config;
```

- [ ] **Step 4: Create Python bindings crate**

`crates/dexi-py/Cargo.toml`:
```toml
[package]
name = "dexi-py"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "dexi_py"
crate-type = ["cdylib"]

[dependencies]
dexi = { path = "../dexi" }
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
```

`crates/dexi-py/pyproject.toml`:
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "dexi"
requires-python = ">=3.9"

[tool.maturin]
features = ["pyo3/extension-module"]
```

`crates/dexi-py/src/lib.rs`:
```rust
use pyo3::prelude::*;

#[pymodule]
fn dexi_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
```

- [ ] **Step 5: Verify workspace builds**

Run: `cargo build`
Expected: workspace compiles successfully (all crates)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/dexi/Cargo.toml crates/dexi/src/lib.rs \
        crates/dexi-py/Cargo.toml crates/dexi-py/pyproject.toml crates/dexi-py/src/lib.rs
git commit -m "scaffold: init Cargo workspace with dexi and dexi-py crates"
```

---

### Task 2: Constants Module

**Files:**
- Create: `crates/dexi/src/constants.rs`

- [ ] **Step 1: Write constants.rs**

```rust
use std::path::PathBuf;

/// Coordinate transform from operator space to MANO right hand convention.
pub const OPERATOR2MANO_RIGHT: [[f64; 3]; 3] = [
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
];

/// Coordinate transform from operator space to MANO left hand convention.
pub const OPERATOR2MANO_LEFT: [[f64; 3]; 3] = [
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RobotName {
    Allegro,
    Shadow,
    Svh,
    Leap,
    Ability,
    Inspire,
    Panda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetargetingType {
    Vector,
    Position,
    DexPilot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandType {
    Right,
    Left,
}

/// Map RobotName to directory name in assets/robots/hands/.
pub const ROBOT_NAME_MAP: &[(RobotName, &str)] = &[
    (RobotName::Allegro, "allegro_hand"),
    (RobotName::Shadow, "shadow_hand"),
    (RobotName::Svh, "schunk_hand"),
    (RobotName::Leap, "leap_hand"),
    (RobotName::Ability, "ability_hand"),
    (RobotName::Inspire, "inspire_hand"),
    (RobotName::Panda, "panda_gripper"),
];

/// Dummy free joint names (6-DOF floating base: 3 prismatic + 3 revolute).
pub const DUMMY_JOINT_NAMES: [&str; 6] = [
    "dummy_x_translation_joint",
    "dummy_y_translation_joint",
    "dummy_z_translation_joint",
    "dummy_x_rotation_joint",
    "dummy_y_rotation_joint",
    "dummy_z_rotation_joint",
];

pub const ALL_ROBOT_NAMES: &[RobotName] = &[
    RobotName::Allegro,
    RobotName::Shadow,
    RobotName::Svh,
    RobotName::Leap,
    RobotName::Ability,
    RobotName::Inspire,
    RobotName::Panda,
];

pub const DEXPILOT_ROBOT_NAMES: &[RobotName] = &[
    RobotName::Allegro,
    RobotName::Shadow,
    RobotName::Svh,
    RobotName::Leap,
    RobotName::Inspire,
    RobotName::Panda,
];

impl RobotName {
    pub fn dir_name(&self) -> &'static str {
        ROBOT_NAME_MAP
            .iter()
            .find(|(n, _)| n == self)
            .map(|(_, s)| *s)
            .unwrap()
    }

    pub fn is_gripper(&self) -> bool {
        matches!(self, RobotName::Panda)
    }
}

/// Build the default config path for a (robot, type, hand) combination.
pub fn get_default_config_path(
    config_dir: &PathBuf,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
) -> Option<PathBuf> {
    let subdir = match retargeting_type {
        RetargetingType::Position => "offline",
        _ => "teleop",
    };
    let robot_str = robot_name.dir_name();
    let path = if robot_name.is_gripper() {
        match retargeting_type {
            RetargetingType::DexPilot => config_dir.join(subdir).join(format!("{}_dexpilot.yml", robot_str)),
            _ => config_dir.join(subdir).join(format!("{}.yml", robot_str)),
        }
    } else {
        match retargeting_type {
            RetargetingType::DexPilot => {
                config_dir.join(subdir).join(format!("{}_{}_dexpilot.yml", robot_str, hand_type.name()))
            }
            _ => config_dir.join(subdir).join(format!("{}_{}.yml", robot_str, hand_type.name())),
        }
    };
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

impl HandType {
    pub fn name(&self) -> &'static str {
        match self {
            HandType::Right => "right",
            HandType::Left => "left",
        }
    }
}

impl RetargetingType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RetargetingType::Vector => "vector",
            RetargetingType::Position => "position",
            RetargetingType::DexPilot => "dexpilot",
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p dexi`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/dexi/src/constants.rs
git commit -m "feat: add constants module with enums, transforms, and config path resolution"
```

---

### Task 3: Low-Pass Filter

**Files:**
- Create: `crates/dexi/src/filter.rs`

- [ ] **Step 1: Write filter.rs**

```rust
pub struct LpFilter {
    alpha: f64,
    y: Option<Vec<f64>>,
}

impl LpFilter {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, y: None }
    }

    pub fn next(&mut self, x: &[f64]) -> Vec<f64> {
        match &self.y {
            None => {
                self.y = Some(x.to_vec());
                x.to_vec()
            }
            Some(y) => {
                let new_y: Vec<f64> = y
                    .iter()
                    .zip(x.iter())
                    .map(|(yi, xi)| yi + self.alpha * (xi - yi))
                    .collect();
                self.y = Some(new_y.clone());
                new_y
            }
        }
    }

    pub fn reset(&mut self) {
        self.y = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lp_filter_first_call_returns_input() {
        let mut f = LpFilter::new(0.5);
        let result = f.next(&[1.0, 2.0, 3.0]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_lp_filter_reset() {
        let mut f = LpFilter::new(0.5);
        f.next(&[1.0, 2.0]);
        f.reset();
        let result = f.next(&[5.0, 6.0]);
        assert_eq!(result, vec![5.0, 6.0]);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p dexi`
Expected: 2 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/dexi/src/filter.rs
git commit -m "feat: add low-pass filter (LPFilter)"
```

---

### Task 4: URDF Extensions — Dummy Joint Insertion

**Files:**
- Create: `crates/dexi/src/urdf_ext.rs`

- [ ] **Step 1: Write urdf_ext.rs**

```rust
use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::constants::DUMMY_JOINT_NAMES;

/// Insert 6-DOF dummy free joint chain at URDF root.
///
/// Writes a modified URDF to a temp file and returns the path.
/// The dummy chain is: root_link -> x_trans -> y_trans -> z_trans
///                      -> x_rot -> y_rot -> z_rot -> [original base_link].
pub fn insert_dummy_joints(urdf_path: &Path) -> Result<PathBuf> {
    let xml = fs::read_to_string(urdf_path)
        .with_context(|| format!("Failed to read URDF: {}", urdf_path.display()))?;

    let mut doc: roxmltree::Document = roxmltree::Document::parse(&xml)
        .with_context(|| "Failed to parse URDF XML")?;

    let root = doc.root_element();
    let robot = root;

    // Find the original base link (first link)
    let orig_base_name: String = robot
        .children()
        .filter(|n| n.has_tag_name("link"))
        .find_map(|link| link.attribute("name").map(String::from))
        .unwrap_or_else(|| "base_link".to_string());

    // Create dummy links and joints as raw XML strings
    let dummy_links_joints = build_dummy_chain_xml(&orig_base_name);

    // Insert dummy links before first existing link, dummy joints before first existing joint
    let mut new_xml = String::new();
    new_xml.push_str(r#"<?xml version="1.0" encoding="utf-8"?>"#);
    new_xml.push_str(&format!(
        r#"<robot name="{}_with_dummy">"#,
        robot.attribute("name").unwrap_or("robot")
    ));
    new_xml.push_str(&dummy_links_joints);
    // Keep original content (skip root element wrapper)
    for child in robot.children() {
        if child.is_element() {
            new_xml.push_str(&xml[child.range()]);
        }
    }
    new_xml.push_str("</robot>");

    // Write to temp file
    let temp_dir = std::env::temp_dir().join("dexi");
    fs::create_dir_all(&temp_dir)?;
    let file_name = urdf_path.file_name().unwrap().to_str().unwrap();
    let temp_path = temp_dir.join(format!("dummy_{}", file_name));
    let mut f = fs::File::create(&temp_path)?;
    f.write_all(new_xml.as_bytes())?;

    Ok(temp_path)
}

fn build_dummy_chain_xml(orig_base: &str) -> String {
    let trans_axes = ["x", "y", "z"];
    let rot_axes = ["x", "y", "z"];
    let mut xml = String::new();

    // Root dummy link
    xml.push_str(r#"<link name="dummy_root_link"/>"#);

    // Translation joints + links
    for (i, axis) in trans_axes.iter().enumerate() {
        let parent = if i == 0 {
            "dummy_root_link".to_string()
        } else {
            format!("dummy_{}_translation_link", trans_axes[i - 1])
        };
        let link_name = format!("dummy_{}_translation_link", axis);
        let joint_name = format!("dummy_{}_translation_joint", axis);

        xml.push_str(&format!(
            r#"<link name="{}"/>"#,
            link_name
        ));
        xml.push_str(&format!(
            r#"<joint name="{}" type="prismatic">"#,
            joint_name
        ));
        xml.push_str(&format!(
            r#"<parent link="{}"/>"#,
            parent
        ));
        xml.push_str(&format!(
            r#"<child link="{}"/>"#,
            link_name
        ));
        let axis_vec = match *axis {
            'x' => "1 0 0",
            'y' => "0 1 0",
            _ => "0 0 1",
        };
        xml.push_str(&format!(r#"<axis xyz="{}"/>"#, axis_vec));
        xml.push_str(r#"<limit effort="0" velocity="0" lower="-1" upper="1"/>"#);
        xml.push_str("</joint>");
    }

    // Rotation joints + links
    let last_trans = format!("dummy_{}_translation_link", trans_axes[2]);
    for (i, axis) in rot_axes.iter().enumerate() {
        let parent = if i == 0 {
            last_trans.clone()
        } else {
            format!("dummy_{}_rotation_link", rot_axes[i - 1])
        };
        let link_name = format!("dummy_{}_rotation_link", axis);
        let joint_name = format!("dummy_{}_rotation_joint", axis);

        xml.push_str(&format!(
            r#"<link name="{}"/>"#,
            link_name
        ));
        xml.push_str(&format!(
            r#"<joint name="{}" type="revolute">"#,
            joint_name
        ));
        xml.push_str(&format!(
            r#"<parent link="{}"/>"#,
            parent
        ));
        let child = if i == 2 {
            orig_base.to_string()
        } else {
            format!("dummy_{}_rotation_link", rot_axes[i + 1])
        };
        xml.push_str(&format!(r#"<child link="{}"/>"#, child));
        let axis_vec = match *axis {
            'x' => "1 0 0",
            'y' => "0 1 0",
            _ => "0 0 1",
        };
        xml.push_str(&format!(r#"<axis xyz="{}"/>"#, axis_vec));
        xml.push_str(r#"<limit effort="0" velocity="0" lower="-3.14159" upper="3.14159"/>"#);
        xml.push_str("</joint>");
    }

    xml
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_chain_xml_contains_joint_names() {
        let xml = build_dummy_chain_xml("base_link");
        for name in DUMMY_JOINT_NAMES {
            assert!(xml.contains(name), "Missing dummy joint: {}", name);
        }
        assert!(xml.contains(r#"<child link="base_link"/>"#));
    }
}
```

> **Note:** This approach — raw XML string building — is a pragmatic stopgap. `urdf-rs` does not support programmatic URDF mutation. If the `roxmltree` parse step for extracting `orig_base_name` proves unreliable across robot URDFs, replace with simple regex: `let orig_base = find_first_link_name(&xml)` where `find_first_link_name` uses `re.find(r#"<link\s+name="([^"]+)""#)`.

- [ ] **Step 2: Need `roxmltree` dependency**

Edit `crates/dexi/Cargo.toml` — add to `[dependencies]`:
```toml
roxmltree = "0.20"
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p dexi`
Expected: test passes

- [ ] **Step 4: Commit**

```bash
git add crates/dexi/src/urdf_ext.rs crates/dexi/Cargo.toml
git commit -m "feat: add URDF dummy joint insertion for 6-DOF floating base"
```

---

### Task 5: Chain Builder — Per-Finger SerialChains

**Files:**
- Create: `crates/dexi/src/chain_builder.rs`

- [ ] **Step 1: Write chain_builder.rs**

```rust
use anyhow::{anyhow, Result};
use k::serial_chain::SerialChain;
use k::Node;
use std::collections::HashMap;

/// Build per-link SerialChain instances from a branching `k::Chain`.
///
/// `k`'s `SerialChain` is a linear chain from end-effector back to root.
/// Robot hands have branching fingers sharing the palm — each finger tip
/// needs its own `SerialChain` tracing up through the tree.
pub fn build_serial_chains(
    chain: &k::Chain<f64>,
    link_names: &[String],
) -> Result<HashMap<String, SerialChain<f64>>> {
    let mut chains = HashMap::new();
    for name in link_names {
        let serial = build_serial_chain_to_link(chain, name)?;
        chains.insert(name.clone(), serial);
    }
    Ok(chains)
}

/// Build a SerialChain from the root to the named link by walking the tree.
fn build_serial_chain_to_link(
    chain: &k::Chain<f64>,
    target_link: &str,
) -> Result<SerialChain<f64>> {
    // Walk tree from target up to root to collect joint nodes
    let target_node = chain
        .find(target_link)
        .ok_or_else(|| anyhow!("Link '{}' not found in chain", target_link))?;

    // Collect joints along the path from target to root
    let mut joint_nodes: Vec<Node<f64>> = Vec::new();
    let mut current = Some(target_node);
    while let Some(node) = current {
        joint_nodes.push(node);
        current = chain.iter_joints().find(|j| {
            j.joint().child_link.as_deref() == Some(node.link().as_ref().map(|l| l.name.as_str()))
                .unwrap_or(false)
        });
    }
    joint_nodes.reverse();

    if joint_nodes.is_empty() {
        return Err(anyhow!("Empty joint chain for link '{}'", target_link));
    }

    SerialChain::from_end(joint_nodes.last().unwrap().clone())
        .map_err(|e| anyhow!("Failed to create SerialChain for '{}': {}", target_link, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use k::Chain;

    #[test]
    fn test_build_chain_from_urdf() {
        // Integration test — requires a URDF in assets
    }
}
```

> **Note:** The `k` crate API is unstable between versions. The `find()`, `iter_joints()`, and `SerialChain::from_end()` signatures above are best-effort against `k` v0.32 docs. Adjust method names to match the actual `k` API after checking docs. In particular: the link-to-parent traversal logic may need to use `k::joint::Joint`'s `parent_link`/`child_link` fields directly.

- [ ] **Step 2: Verify build**

Run: `cargo build -p dexi`
Expected: compiles (may require adjusting for actual k crate API)

- [ ] **Step 3: Commit**

```bash
git add crates/dexi/src/chain_builder.rs
git commit -m "feat: add per-finger SerialChain builder from branching URDF tree"
```

---

### Task 6: RobotWrapper

**Files:**
- Create: `crates/dexi/src/robot.rs`

- [ ] **Step 1: Write robot.rs**

```rust
use anyhow::{Context, Result};
use k::Chain;
use k::serial_chain::SerialChain;
use nalgebra::{DMatrix, Isometry3, Matrix4, Translation3, UnitQuaternion};
use std::collections::HashMap;
use std::path::Path;

use crate::chain_builder::build_serial_chains;

/// Wrapper around `k::Chain<f64>` providing FK and Jacobian ops.
pub struct RobotWrapper {
    chain: Chain<f64>,
    joint_limits: Vec<(f64, f64)>,
    dof_joint_names: Vec<String>,
    serial_chains: HashMap<String, SerialChain<f64>>,
    link_name_to_index: HashMap<String, usize>,
    /// Pose cache after `compute_forward_kinematics`.
    link_poses: HashMap<String, Isometry3<f64>>,
}

impl RobotWrapper {
    pub fn from_urdf_file(path: &Path) -> Result<Self> {
        let chain = Chain::<f64>::from_urdf_file(path)
            .with_context(|| format!("Failed to load URDF: {}", path.display()))?;

        let dof_joint_names: Vec<String> = chain
            .iter_joints()
            .filter(|j| j.joint().joint_type != k::joint::JointType::Fixed)
            .map(|j| j.joint().name.clone())
            .collect();

        let joint_limits: Vec<(f64, f64)> = chain
            .iter_joints()
            .filter(|j| j.joint().joint_type != k::joint::JointType::Fixed)
            .map(|j| {
                let limits = j.joint().limits.clone();
                (limits.lower.unwrap_or(-1e4), limits.upper.unwrap_or(1e4))
            })
            .collect();

        let link_name_to_index: HashMap<String, usize> = chain
            .iter_links()
            .enumerate()
            .map(|(i, node)| {
                let name = node.link().as_ref().map(|l| l.name.clone()).unwrap_or_default();
                (name, i)
            })
            .collect();

        Ok(Self {
            chain,
            joint_limits,
            dof_joint_names,
            serial_chains: HashMap::new(),
            link_name_to_index,
            link_poses: HashMap::new(),
        })
    }

    pub fn dof(&self) -> usize {
        self.dof_joint_names.len()
    }

    pub fn joint_names(&self) -> &[String] {
        &self.dof_joint_names
    }

    pub fn joint_limits(&self) -> &[(f64, f64)] {
        &self.joint_limits
    }

    pub fn get_joint_index(&self, name: &str) -> Result<usize> {
        self.dof_joint_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| anyhow::anyhow!("Joint '{}' not found", name))
    }

    /// Ensure SerialChains are built for all target links.
    pub fn ensure_serial_chains(&mut self, link_names: &[String]) -> Result<()> {
        let missing: Vec<String> = link_names
            .iter()
            .filter(|n| !self.serial_chains.contains_key(*n))
            .cloned()
            .collect();
        if !missing.is_empty() {
            let new_chains = build_serial_chains(&self.chain, &missing)?;
            self.serial_chains.extend(new_chains);
        }
        Ok(())
    }

    /// Compute FK and cache link poses.
    pub fn compute_forward_kinematics(&mut self, qpos: &[f64]) -> Result<()> {
        self.chain.set_joint_positions(qpos)?;
        self.chain.update_transforms();

        self.link_poses.clear();
        for node in self.chain.iter_links() {
            if let Some(link) = node.link() {
                let iso = node.world_transform().ok_or_else(|| {
                    anyhow::anyhow!("No world transform for link '{}'", link.name)
                })?;
                self.link_poses.insert(link.name.clone(), iso);
            }
        }
        Ok(())
    }

    /// Get the 4x4 homogeneous transform for a link (cached after FK).
    pub fn get_link_pose(&self, link_name: &str) -> Result<Matrix4<f64>> {
        let iso = self
            .link_poses
            .get(link_name)
            .ok_or_else(|| anyhow::anyhow!("Link '{}' not in pose cache. Call FK first.", link_name))?;
        Ok(iso.to_matrix())
    }

    /// Compute the 3xN position Jacobian (world-frame) for a single link.
    pub fn compute_single_link_jacobian(
        &mut self,
        qpos: &[f64],
        link_name: &str,
    ) -> Result<DMatrix<f64>> {
        self.ensure_serial_chains(&[link_name.to_string()])?;
        let serial = self
            .serial_chains
            .get(link_name)
            .ok_or_else(|| anyhow::anyhow!("No SerialChain for '{}'", link_name))?;

        self.chain.set_joint_positions(qpos)?;
        let jacobian = serial.jacobian();
        let num_dof = self.dof();
        let mut full_j = DMatrix::zeros(6, num_dof);

        // The jacobian() returns (6, chain_dof), map to full DOF
        let joint_names_in_chain: Vec<String> = serial
            .iter_joints()
            .filter(|j| {
                j.joint().joint_type != k::joint::JointType::Fixed
            })
            .map(|j| j.joint().name.clone())
            .collect();

        for (col, joint_name) in joint_names_in_chain.iter().enumerate() {
            if let Ok(global_col) = self.get_joint_index(joint_name) {
                for row in 0..6 {
                    full_j[(row, global_col)] = jacobian[(row, col)];
                }
            }
        }

        // Extract position part (first 3 rows)
        Ok(full_j.rows(0, 3).into())
    }
}
```

> **Note:** `k` crate API is under active development. The exact methods `set_joint_positions`, `update_transforms`, `world_transform`, `iter_links`, `iter_joints`, `jacobian()` may differ. Check the `k` crate docs for the specific version and adapt method names accordingly.

- [ ] **Step 2: Add k dependency features if needed**

Check if `k` crate needs feature flags for URDF loading. Update `crates/dexi/Cargo.toml`:
```toml
k = { version = "0.32", features = ["urdf"] }
```

- [ ] **Step 3: Verify build**

Run: `cargo build -p dexi`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add crates/dexi/src/robot.rs crates/dexi/Cargo.toml
git commit -m "feat: add RobotWrapper for FK and Jacobian via k crate"
```

---

### Task 7: Mimic Joint Adaptor

**Files:**
- Create: `crates/dexi/src/mimic.rs`

- [ ] **Step 1: Write mimic.rs**

```rust
use anyhow::{bail, Result};
use nalgebra::DMatrix;

use crate::robot::RobotWrapper;

struct MimicMapping {
    source_idx: usize,
    mimic_idx: usize,
    multiplier: f64,
    offset: f64,
}

pub struct MimicJointAdaptor {
    mappings: Vec<MimicMapping>,
    idx_pin2target: Vec<usize>,
    idx_target2source: Vec<usize>,
    /// Number of active joints (total - num_mimic).
    pub num_active_joints: usize,
}

impl MimicJointAdaptor {
    pub fn new(
        robot: &RobotWrapper,
        target_joint_names: &[String],
        source_joint_names: &[String],
        mimic_joint_names: &[String],
        multipliers: &[f64],
        offsets: &[f64],
    ) -> Result<Self> {
        // Validate: mimic joints must not be target joints
        let target_set: std::collections::HashSet<_> = target_joint_names.iter().collect();
        for mimic in mimic_joint_names {
            if target_set.contains(mimic) {
                bail!(
                    "Mimic joint '{}' must not be a target joint. \
                     Specify target_joint_names explicitly in your config.",
                    mimic
                );
            }
        }

        let len_source = source_joint_names.len();
        let len_mimic = mimic_joint_names.len();
        let len_mul = multipliers.len();
        let len_offset = offsets.len();

        if !(len_mimic == len_source && len_source == len_mul && len_mul == len_offset) {
            bail!(
                "Mimic joints dimension mismatch: source={}, mimic={}, mult={}, offset={}",
                len_source, len_mimic, len_mul, len_offset
            );
        }

        // Check uniqueness
        let unique_mimic: std::collections::HashSet<_> = mimic_joint_names.iter().collect();
        if unique_mimic.len() != mimic_joint_names.len() {
            bail!("Redundant mimic joint names: {:?}", mimic_joint_names);
        }

        let idx_pin2target: Vec<usize> = target_joint_names
            .iter()
            .map(|n| robot.get_joint_index(n))
            .collect::<Result<Vec<_>>>()?;

        let idx_pin2source: Vec<usize> = source_joint_names
            .iter()
            .map(|n| robot.get_joint_index(n))
            .collect::<Result<Vec<_>>>()?;

        let idx_pin2mimic: Vec<usize> = mimic_joint_names
            .iter()
            .map(|n| robot.get_joint_index(n))
            .collect::<Result<Vec<_>>>()?;

        let idx_target2source: Vec<usize> = source_joint_names
            .iter()
            .map(|n| {
                target_joint_names
                    .iter()
                    .position(|t| t == n)
                    .ok_or_else(|| anyhow::anyhow!("Source joint '{}' not in target_joint_names", n))
            })
            .collect::<Result<Vec<_>>>()?;

        let num_active_joints = robot.dof() - len_mimic;

        let mappings: Vec<MimicMapping> = (0..len_mimic)
            .map(|i| MimicMapping {
                source_idx: idx_pin2source[i],
                mimic_idx: idx_pin2mimic[i],
                multiplier: multipliers[i],
                offset: offsets[i],
            })
            .collect();

        Ok(Self {
            mappings,
            idx_pin2target,
            idx_target2source,
            num_active_joints,
        })
    }

    /// Apply mimic joint values: q[mimic] = q[source] * mult + offset.
    pub fn forward_qpos(&self, qpos: &mut [f64]) {
        for m in &self.mappings {
            qpos[m.mimic_idx] = qpos[m.source_idx] * m.multiplier + m.offset;
        }
    }

    /// Adapt Jacobian: J[.., source] += J[.., mimic] * multiplier,
    /// then slice out only target columns.
    pub fn backward_jacobian(&self, jacobian: &DMatrix<f64>) -> DMatrix<f64> {
        let mut target_j = jacobian.columns(0, 0); // placeholder
        // Select target columns
        let col_data: Vec<_> = self.idx_pin2target.iter().map(|&i| jacobian.column(i)).collect();
        let nrows = jacobian.nrows();
        let ncols = self.idx_pin2target.len();
        let mut target_j = DMatrix::from_columns(&col_data);

        for (i, m) in self.mappings.iter().enumerate() {
            let mimic_col = jacobian.column(m.mimic_idx) * m.multiplier;
            let source_target_idx = self.idx_target2source[i];
            let mut col = target_j.column(source_target_idx).into_owned();
            col += mimic_col;
            target_j.set_column(source_target_idx, &col);
        }

        target_j
    }

    pub fn idx_pin2mimic(&self) -> Vec<usize> {
        self.mappings.iter().map(|m| m.mimic_idx).collect()
    }

    pub fn idx_pin2target(&self) -> &[usize] {
        &self.idx_pin2target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mimic_forward_qpos() {
        // Requires a robot instance — integration test.
    }
}
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p dexi`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/dexi/src/mimic.rs
git commit -m "feat: add MimicJointAdaptor for forward qpos and backward Jacobian"
```

---

### Task 8: Huber Loss Gradients

**Files:**
- Create: `crates/dexi/src/huber.rs`

- [ ] **Step 1: Write huber.rs**

```rust
use nalgebra::{DVector, DMatrix};

/// Huber loss (SmoothL1) value.
/// `diff` = element-wise (predicted - target).
pub fn huber_loss(diff: &DVector<f64>, delta: f64) -> f64 {
    diff.iter()
        .map(|&d| {
            let abs_d = d.abs();
            if abs_d < delta {
                0.5 * d * d / delta
            } else {
                abs_d - 0.5 * delta
            }
        })
        .sum()
}

/// Huber loss gradient w.r.t. each predicted element.
/// d(L)/d(pred_i) = sign(diff_i) if |diff_i| >= delta, else diff_i / delta.
pub fn huber_gradient(diff: &DVector<f64>, delta: f64) -> DVector<f64> {
    DVector::from_iterator(
        diff.len(),
        diff.iter().map(|&d| {
            if d.abs() >= delta {
                d.signum()
            } else {
                d / delta
            }
        }),
    )
}

/// Vector-norm Huber: loss = sum_i huber(||robot_vec_i - target_vec_i||, delta).
/// gradient per body_pos is 3D. Returns (loss, grad_per_body_pos[link_count][3]).
pub fn huber_vector_loss_and_grad(
    robot_vecs: &[nalgebra::Vector3<f64>],
    target_vecs: &[nalgebra::Vector3<f64>],
    delta: f64,
) -> (f64, Vec<nalgebra::Vector3<f64>>) {
    let mut loss = 0.0;
    let mut grads = Vec::with_capacity(robot_vecs.len());
    for (r, t) in robot_vecs.iter().zip(target_vecs.iter()) {
        let diff = r - t;
        let dist = diff.norm();
        if dist < delta {
            loss += 0.5 * dist * dist / delta;
        } else {
            loss += dist - 0.5 * delta;
        }
        let grad_factor = if dist < delta && dist > 1e-10 {
            1.0 / delta
        } else if dist > 1e-10 {
            1.0 / dist
        } else {
            0.0
        };
        grads.push(diff * grad_factor);
    }
    (loss, grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huber_gradient_small_diff() {
        let diff = DVector::from_vec(vec![0.01]);
        let grad = huber_gradient(&diff, 0.02);
        assert!((grad[0] - 0.5).abs() < 1e-10, "Small diff: grad = diff/delta");
    }

    #[test]
    fn test_huber_gradient_large_diff() {
        let diff = DVector::from_vec(vec![0.05]);
        let grad = huber_gradient(&diff, 0.02);
        assert!((grad[0] - 1.0).abs() < 1e-10, "Large diff: grad = sign(diff)");
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p dexi`
Expected: 2 tests pass

- [ ] **Step 3: Add mod huber to lib.rs**

Edit `crates/dexi/src/lib.rs` — add `pub mod huber;`

- [ ] **Step 4: Commit**

```bash
git add crates/dexi/src/huber.rs crates/dexi/src/lib.rs
git commit -m "feat: add analytical Huber loss and gradient (replaces PyTorch autograd)"
```

---

### Task 9: PositionOptimizer

**Files:**
- Create: `crates/dexi/src/optimizer.rs`

- [ ] **Step 1: Write base Optimizer trait**

```rust
use anyhow::Result;
use nalgebra::DMatrix;
use std::collections::HashSet;

use crate::huber::{huber_gradient, huber_vector_loss_and_grad};
use crate::mimic::MimicJointAdaptor;
use crate::robot::RobotWrapper;

/// Base optimizer managing shared state.
pub trait Optimizer {
    fn robot(&self) -> &RobotWrapper;
    fn robot_mut(&mut self) -> &mut RobotWrapper;
    fn adaptor(&self) -> Option<&MimicJointAdaptor>;
    fn adaptor_mut(&mut self) -> Option<&mut MimicJointAdaptor>;
    fn idx_pin2target(&self) -> &[usize];
    fn idx_pin2fixed(&self) -> &[usize];
    fn retargeting_type(&self) -> &str;

    /// Run SLSQP optimization. Returns optimized qpos (target DOF only).
    fn retarget(
        &mut self,
        ref_value: &[f64],
        fixed_qpos: &[f64],
        last_qpos: &[f64],
    ) -> Result<Vec<f64>> {
        let n_target = self.idx_pin2target().len();
        let n_fixed = self.idx_pin2fixed().len();

        if fixed_qpos.len() != n_fixed {
            anyhow::bail!(
                "Optimizer has {} fixed joints but {} values given",
                n_fixed,
                fixed_qpos.len()
            );
        }

        let joint_limits = self.robot().joint_limits();
        let target_limits: Vec<(f64, f64)> = self
            .idx_pin2target()
            .iter()
            .map(|&i| joint_limits[i])
            .collect();

        let lower: Vec<f64> = target_limits.iter().map(|l| l.0 - 1e-3).collect();
        let upper: Vec<f64> = target_limits.iter().map(|l| l.1 + 1e-3).collect();

        let fixed = fixed_qpos.to_vec();
        let last = last_qpos.to_vec();
        let ref_val = ref_value.to_vec();
        let ndim = n_target;

        let ftol = self.ftol_abs();

        let objective = |x: &[f64], grad: Option<&mut [f64]>| -> f64 {
            self.evaluate(&ref_val, &fixed, &last, x, grad)
        };

        let x0 = last_qpos.to_vec();

        let result = slsqp::minimize(
            &x0,
            |x, grad| Ok(objective(x, grad)),
            &lower,
            &upper,
            &vec![],
            slsqp::Config {
                ftol_abs: ftol,
                max_iter: 100,
                ..Default::default()
            },
        );

        match result {
            Ok((x_opt, _)) => Ok(x_opt),
            Err(_) => Ok(x0),
        }
    }

    /// Evaluate objective and optionally fill gradient.
    fn evaluate(
        &mut self,
        ref_value: &[f64],
        fixed_qpos: &[f64],
        last_qpos: &[f64],
        x: &[f64],
        grad: Option<&mut [f64]>,
    ) -> f64;

    fn ftol_abs(&self) -> f64;
}
```

> **Note:** The `slsqp::minimize` API signature here is an approximation. The actual `slsqp` crate v0.2 API may differ in parameter names, return types, or Config fields. Read the crate docs and adjust. The equality constraint vec is empty because we use only box constraints.

- [ ] **Step 2: Write PositionOptimizer**

```rust
pub struct PositionOptimizer {
    pub robot: RobotWrapper,
    pub idx_pin2target: Vec<usize>,
    pub idx_pin2fixed: Vec<usize>,
    pub target_link_indices: Vec<String>,
    pub target_link_human_indices: Vec<usize>,
    pub huber_delta: f64,
    pub norm_delta: f64,
    pub adaptor: Option<MimicJointAdaptor>,
    retargeting_type_str: &'static str,
}

impl PositionOptimizer {
    pub fn new(
        robot: RobotWrapper,
        target_joint_names: &[String],
        target_link_names: &[String],
        target_link_human_indices: &[usize],
        huber_delta: f64,
        norm_delta: f64,
    ) -> Result<Self> {
        let dof = robot.dof();
        let joint_names = robot.joint_names().to_vec();

        let idx_pin2target: Vec<usize> = target_joint_names
            .iter()
            .map(|n| {
                joint_names
                    .iter()
                    .position(|j| j == n)
                    .ok_or_else(|| anyhow::anyhow!("Joint '{}' not found in robot", n))
            })
            .collect::<Result<Vec<_>>>()?;

        let target_idx_set: HashSet<usize> = idx_pin2target.iter().cloned().collect();
        let idx_pin2fixed: Vec<usize> = (0..dof).filter(|i| !target_idx_set.contains(i)).collect();

        Ok(Self {
            robot,
            idx_pin2target,
            idx_pin2fixed,
            target_link_indices: target_link_names.to_vec(),
            target_link_human_indices: target_link_human_indices.to_vec(),
            huber_delta,
            norm_delta,
            adaptor: None,
            retargeting_type_str: "POSITION",
        })
    }
}

impl Optimizer for PositionOptimizer {
    fn robot(&self) -> &RobotWrapper { &self.robot }
    fn robot_mut(&mut self) -> &mut RobotWrapper { &mut self.robot }
    fn adaptor(&self) -> Option<&MimicJointAdaptor> { self.adaptor.as_ref() }
    fn adaptor_mut(&mut self) -> Option<&mut MimicJointAdaptor> { self.adaptor.as_mut() }
    fn idx_pin2target(&self) -> &[usize] { &self.idx_pin2target }
    fn idx_pin2fixed(&self) -> &[usize] { &self.idx_pin2fixed }
    fn retargeting_type(&self) -> &str { self.retargeting_type_str }
    fn ftol_abs(&self) -> f64 { 1e-5 }

    fn evaluate(
        &mut self,
        ref_value: &[f64],
        fixed_qpos: &[f64],
        last_qpos: &[f64],
        x: &[f64],
        grad: Option<&mut [f64]>,
    ) -> f64 {
        let dof = self.robot.dof();
        let mut qpos = vec![0.0; dof];
        for (i, &idx) in self.idx_pin2fixed.iter().enumerate() {
            qpos[idx] = fixed_qpos[i];
        }
        for (i, &idx) in self.idx_pin2target.iter().enumerate() {
            qpos[idx] = x[i];
        }

        if let Some(ref adaptor) = self.adaptor {
            adaptor.forward_qpos(&mut qpos);
        }

        self.robot.compute_forward_kinematics(&qpos).unwrap();

        let n_targets = self.target_link_indices.len();
        let target_pos: Vec<nalgebra::Vector3<f64>> = ref_value
            .chunks(3)
            .take(n_targets)
            .map(|c| nalgebra::Vector3::new(c[0], c[1], c[2]))
            .collect();

        let body_pos: Vec<nalgebra::Vector3<f64>> = self
            .target_link_indices
            .iter()
            .map(|name| {
                let pose = self.robot.get_link_pose(name).unwrap();
                nalgebra::Vector3::new(pose[(0, 3)], pose[(1, 3)], pose[(2, 3)])
            })
            .collect();

        // Huber loss on position error
        let diff: nalgebra::DVector<f64> = nalgebra::DVector::from_iterator(
            n_targets * 3,
            target_pos.iter().zip(body_pos.iter()).flat_map(|(t, b)| {
                vec![t.x - b.x, t.y - b.y, t.z - b.z]
            }),
        );

        let huber = huber_gradient(&diff, self.huber_delta);
        let loss: f64 = diff.iter().zip(huber.iter()).map(|(d, h)| {
            let abs_d = d.abs();
            if abs_d < self.huber_delta {
                0.5 * d * d / self.huber_delta
            } else {
                abs_d - 0.5 * self.huber_delta
            }
        }).sum();

        if let Some(g) = grad {
            // Gradient computation
            let mut grad_qpos = vec![0.0; self.idx_pin2target.len()];

            for (i, link_name) in self.target_link_indices.iter().enumerate() {
                let link_j = self
                    .robot
                    .compute_single_link_jacobian(&qpos, link_name)
                    .unwrap();

                let link_pose = self.robot.get_link_pose(link_name).unwrap();
                let link_rot = link_pose.fixed_view::<3, 3>(0, 0).into_owned();

                // Rotate from local to world
                let world_j = &link_rot * &link_j;

                // grad_pos for this link
                let grad_pos = nalgebra::Vector3::new(
                    huber[i * 3],
                    huber[i * 3 + 1],
                    huber[i * 3 + 2],
                );

                // Apply Jacobian adaptor if available
                if let Some(ref adaptor) = self.adaptor {
                    let full_j = {
                        let mut m = nalgebra::DMatrix::zeros(3, dof);
                        // Build full Jacobian from world_j
                        for col in 0..world_j.ncols() {
                            for row in 0..3 {
                                m[(row, col)] = world_j[(row, col)];
                            }
                        }
                        m
                    };
                    let adapted_j = adaptor.backward_jacobian(&full_j);
                    for col in 0..adapted_j.ncols() {
                        let contrib = grad_pos.dot(&adapted_j.column(col));
                        grad_qpos[col] += contrib;
                    }
                } else {
                    for col in 0..world_j.ncols() {
                        let global_col = col; // world_j already sliced to target cols
                        if global_col < grad_qpos.len() {
                            let col_vec = world_j.column(col);
                            grad_qpos[global_col] += grad_pos.dot(&col_vec);
                        }
                    }
                }
            }

            // Smoothness penalty: 2 * norm_delta * (x - last_qpos)
            for (i, gv) in grad_qpos.iter_mut().enumerate() {
                *gv += 2.0 * self.norm_delta * (x[i] - last_qpos[i]);
            }

            g.copy_from_slice(&grad_qpos);
        }

        // Add smoothness penalty to loss
        let smooth: f64 = x.iter()
            .zip(last_qpos.iter())
            .map(|(xi, li)| self.norm_delta * (xi - li).powi(2))
            .sum();

        loss + smooth
    }
}
```

- [ ] **Step 3: Verify build**

Run: `cargo build -p dexi`
Expected: compiles (adjust slsqp API as needed)

- [ ] **Step 4: Commit**

```bash
git add crates/dexi/src/optimizer.rs
git commit -m "feat: add PositionOptimizer with Huber loss and analytical gradient"
```

---

### Task 10: VectorOptimizer and DexPilotOptimizer

**Files:**
- Modify: `crates/dexi/src/optimizer.rs`

- [ ] **Step 1: Add VectorOptimizer to optimizer.rs**

Append to `crates/dexi/src/optimizer.rs`:

```rust
pub struct VectorOptimizer {
    pub robot: RobotWrapper,
    pub idx_pin2target: Vec<usize>,
    pub idx_pin2fixed: Vec<usize>,
    pub origin_link_names: Vec<String>,
    pub task_link_names: Vec<String>,
    pub target_link_human_indices: Vec<usize>,
    pub huber_delta: f64,
    pub norm_delta: f64,
    pub scaling: f64,
    pub adaptor: Option<MimicJointAdaptor>,
    /// Computation cache: all unique links that need FK
    pub computed_link_names: Vec<String>,
    origin_link_indices: Vec<usize>,
    task_link_indices: Vec<usize>,
    retargeting_type_str: &'static str,
}

impl VectorOptimizer {
    pub fn new(
        robot: RobotWrapper,
        target_joint_names: &[String],
        target_origin_link_names: &[String],
        target_task_link_names: &[String],
        target_link_human_indices: &[usize],
        huber_delta: f64,
        norm_delta: f64,
        scaling: f64,
    ) -> Result<Self> {
        let dof = robot.dof();
        let joint_names = robot.joint_names().to_vec();

        let idx_pin2target: Vec<usize> = target_joint_names
            .iter()
            .map(|n| joint_names.iter().position(|j| j == n)
                .ok_or_else(|| anyhow::anyhow!("Joint '{}' not found", n)))
            .collect::<Result<Vec<_>>>()?;

        let target_idx_set: HashSet<usize> = idx_pin2target.iter().cloned().collect();
        let idx_pin2fixed: Vec<usize> =
            (0..dof).filter(|i| !target_idx_set.contains(i)).collect();

        // Build computed_link_names: unique union of origin + task links
        let mut computed: Vec<String> = target_origin_link_names.to_vec();
        for name in target_task_link_names {
            if !computed.contains(name) {
                computed.push(name.clone());
            }
        }

        let origin_link_indices: Vec<usize> = target_origin_link_names
            .iter().map(|n| computed.iter().position(|c| c == n).unwrap()).collect();

        let task_link_indices: Vec<usize> = target_task_link_names
            .iter().map(|n| computed.iter().position(|c| c == n).unwrap()).collect();

        Ok(Self {
            robot,
            idx_pin2target,
            idx_pin2fixed,
            origin_link_names: target_origin_link_names.to_vec(),
            task_link_names: target_task_link_names.to_vec(),
            target_link_human_indices: target_link_human_indices.to_vec(),
            huber_delta,
            norm_delta,
            scaling,
            adaptor: None,
            computed_link_names: computed,
            origin_link_indices,
            task_link_indices,
            retargeting_type_str: "VECTOR",
        })
    }
}

impl Optimizer for VectorOptimizer {
    fn robot(&self) -> &RobotWrapper { &self.robot }
    fn robot_mut(&mut self) -> &mut RobotWrapper { &mut self.robot }
    fn adaptor(&self) -> Option<&MimicJointAdaptor> { self.adaptor.as_ref() }
    fn adaptor_mut(&mut self) -> Option<&mut MimicJointAdaptor> { self.adaptor.as_mut() }
    fn idx_pin2target(&self) -> &[usize] { &self.idx_pin2target }
    fn idx_pin2fixed(&self) -> &[usize] { &self.idx_pin2fixed }
    fn retargeting_type(&self) -> &str { self.retargeting_type_str }
    fn ftol_abs(&self) -> f64 { 1e-6 }

    fn evaluate(
        &mut self,
        ref_value: &[f64],
        fixed_qpos: &[f64],
        last_qpos: &[f64],
        x: &[f64],
        grad: Option<&mut [f64]>,
    ) -> f64 {
        let dof = self.robot.dof();
        let mut qpos = vec![0.0; dof];
        for (i, &idx) in self.idx_pin2fixed.iter().enumerate() {
            qpos[idx] = fixed_qpos[i];
        }
        for (i, &idx) in self.idx_pin2target.iter().enumerate() {
            qpos[idx] = x[i];
        }

        if let Some(ref adaptor) = self.adaptor {
            adaptor.forward_qpos(&mut qpos);
        }

        self.robot.compute_forward_kinematics(&qpos).unwrap();

        // Gather body positions for all computed links
        let body_pos: Vec<nalgebra::Vector3<f64>> = self
            .computed_link_names
            .iter()
            .map(|name| {
                let pose = self.robot.get_link_pose(name).unwrap();
                nalgebra::Vector3::new(pose[(0, 3)], pose[(1, 3)], pose[(2, 3)])
            })
            .collect();

        let n_vecs = self.origin_link_names.len();
        let mut robot_vecs = Vec::with_capacity(n_vecs);
        for i in 0..n_vecs {
            let o = body_pos[self.origin_link_indices[i]];
            let t = body_pos[self.task_link_indices[i]];
            robot_vecs.push(t - o);
        }

        let scaled_ref: Vec<nalgebra::Vector3<f64>> = ref_value
            .chunks(3)
            .take(n_vecs)
            .map(|c| nalgebra::Vector3::new(c[0] * self.scaling, c[1] * self.scaling, c[2] * self.scaling))
            .collect();

        // Huber loss on vector norm differences
        let (loss, vec_grads) = huber_vector_loss_and_grad(&robot_vecs, &scaled_ref, self.huber_delta);

        if let Some(g) = grad {
            let mut grad_qpos = vec![0.0; self.idx_pin2target.len()];

            for (i, link_name) in self.computed_link_names.iter().enumerate() {
                let link_j = self.robot.compute_single_link_jacobian(&qpos, link_name).unwrap();
                let link_pose = self.robot.get_link_pose(link_name).unwrap();
                let link_rot = link_pose.fixed_view::<3, 3>(0, 0).into_owned();
                let world_j = &link_rot * &link_j;

                // Accumulate grad_pos for this link from all vectors using it
                let mut grad_pos = nalgebra::Vector3::zeros();
                for v in 0..n_vecs {
                    if self.origin_link_indices[v] == i {
                        grad_pos -= vec_grads[v];
                    }
                    if self.task_link_indices[v] == i {
                        grad_pos += vec_grads[v];
                    }
                }

                // Map to target joint indices
                if let Some(ref adaptor) = self.adaptor {
                    let full_j = {
                        let mut m = DMatrix::zeros(3, dof);
                        for col in 0..world_j.ncols() {
                            for row in 0..3 { m[(row, col)] = world_j[(row, col)]; }
                        }
                        m
                    };
                    let adapted_j = adaptor.backward_jacobian(&full_j);
                    for col in 0..adapted_j.ncols() {
                        grad_qpos[col] += grad_pos.dot(&adapted_j.column(col));
                    }
                } else {
                    for col in 0..world_j.ncols().min(grad_qpos.len()) {
                        grad_qpos[col] += grad_pos.dot(&world_j.column(col));
                    }
                }
            }

            for (i, gv) in grad_qpos.iter_mut().enumerate() {
                *gv += 2.0 * self.norm_delta * (x[i] - last_qpos[i]);
            }

            g.copy_from_slice(&grad_qpos);
        }

        let smooth: f64 = x.iter().zip(last_qpos.iter())
            .map(|(xi, li)| self.norm_delta * (xi - li).powi(2)).sum();
        loss + smooth
    }
}
```

- [ ] **Step 2: Add DexPilotOptimizer to optimizer.rs**

```rust
pub struct DexPilotOptimizer {
    pub robot: RobotWrapper,
    pub idx_pin2target: Vec<usize>,
    pub idx_pin2fixed: Vec<usize>,
    pub origin_link_names: Vec<String>,
    pub task_link_names: Vec<String>,
    pub target_link_human_indices: Vec<usize>,
    pub huber_delta: f64,
    pub norm_delta: f64,
    pub scaling: f64,
    pub adaptor: Option<MimicJointAdaptor>,
    pub computed_link_names: Vec<String>,
    origin_link_indices: Vec<usize>,
    task_link_indices: Vec<usize>,
    retargeting_type_str: &'static str,
    // DexPilot-specific
    pub num_fingers: usize,
    pub project_dist: f64,
    pub escape_dist: f64,
    projected: Vec<bool>,
    s2_project_index_origin: Vec<usize>,
    s2_project_index_task: Vec<usize>,
    projected_dist: Vec<f64>,
    len_proj: usize,
    len_s1: usize,
    len_s2: usize,
}

impl DexPilotOptimizer {
    pub fn new(
        robot: RobotWrapper,
        target_joint_names: &[String],
        finger_tip_link_names: &[String],
        wrist_link_name: &str,
        target_link_human_indices: Option<&[usize]>,
        huber_delta: f64,
        norm_delta: f64,
        project_dist: f64,
        escape_dist: f64,
        scaling: f64,
    ) -> Result<Self> {
        let num_fingers = finger_tip_link_names.len();
        if num_fingers < 2 || num_fingers > 5 {
            anyhow::bail!("DexPilot requires 2-5 fingers, got {}", num_fingers);
        }

        let (origin_link_index, task_link_index) = Self::generate_link_indices(num_fingers);

        let link_names: Vec<&str> = std::iter::once(wrist_link_name)
            .chain(finger_tip_link_names.iter().map(String::as_str))
            .collect();

        let target_origin_link_names: Vec<String> =
            origin_link_index.iter().map(|&i| link_names[i].to_string()).collect();
        let target_task_link_names: Vec<String> =
            task_link_index.iter().map(|&i| link_names[i].to_string()).collect();

        let indices = target_link_human_indices.unwrap_or(&[]);
        let thuman: Vec<usize> = if indices.is_empty() {
            origin_link_index.iter().chain(task_link_index.iter()).map(|&i| i * 4).collect()
        } else {
            indices.to_vec()
        };

        // Build computed link names (cache)
        let mut computed: Vec<String> = target_origin_link_names.clone();
        for name in &target_task_link_names {
            if !computed.contains(name) { computed.push(name.clone()); }
        }

        let origin_link_indices: Vec<usize> = target_origin_link_names
            .iter().map(|n| computed.iter().position(|c| c == n).unwrap()).collect();
        let task_link_indices: Vec<usize> = target_task_link_names
            .iter().map(|n| computed.iter().position(|c| c == n).unwrap()).collect();

        let dof = robot.dof();
        let joint_names = robot.joint_names().to_vec();
        let idx_pin2target: Vec<usize> = target_joint_names
            .iter().map(|n| joint_names.iter().position(|j| j == n)
                .ok_or_else(|| anyhow::anyhow!("Joint '{}' not found", n)))
            .collect::<Result<Vec<_>>>()?;
        let target_idx_set: HashSet<usize> = idx_pin2target.iter().cloned().collect();
        let idx_pin2fixed: Vec<usize> = (0..dof).filter(|i| !target_idx_set.contains(i)).collect();

        let len_proj = num_fingers * (num_fingers - 1) / 2;
        let len_s2 = (num_fingers - 1) * (num_fingers - 2) / 2;
        let len_s1 = len_proj - len_s2;

        let (projected, s2p_origin, s2p_task, proj_dist) =
            Self::set_dexpilot_cache(num_fingers, 1e-4, 3e-2);

        Ok(Self {
            robot, idx_pin2target, idx_pin2fixed,
            origin_link_names: target_origin_link_names,
            task_link_names: target_task_link_names,
            target_link_human_indices: thuman,
            huber_delta, norm_delta, scaling, adaptor: None,
            computed_link_names: computed,
            origin_link_indices, task_link_indices,
            retargeting_type_str: "DEXPILOT",
            num_fingers, project_dist, escape_dist,
            projected, s2_project_index_origin: s2p_origin,
            s2_project_index_task: s2p_task, projected_dist: proj_dist,
            len_proj, len_s1, len_s2,
        })
    }

    #[inline]
    pub fn generate_link_indices(num_fingers: usize) -> (Vec<usize>, Vec<usize>) {
        let mut origin = Vec::new();
        let mut task = Vec::new();
        for i in 1..num_fingers {
            for j in i + 1..=num_fingers {
                origin.push(j);
                task.push(i);
            }
        }
        for i in 1..=num_fingers {
            origin.push(0);
            task.push(i);
        }
        (origin, task)
    }

    fn set_dexpilot_cache(
        num_fingers: usize, eta1: f64, eta2: f64,
    ) -> (Vec<bool>, Vec<usize>, Vec<usize>, Vec<f64>) {
        let len_proj = num_fingers * (num_fingers - 1) / 2;
        let projected = vec![false; len_proj];
        let mut s2_origin = Vec::new();
        let mut s2_task = Vec::new();
        for i in 0..(num_fingers - 2) {
            for j in i + 1..(num_fingers - 1) {
                s2_origin.push(j);
                s2_task.push(i);
            }
        }
        let mut proj_dist = vec![eta1; num_fingers - 1];
        proj_dist.extend(std::iter::repeat(eta2).take((num_fingers - 1) * (num_fingers - 2) / 2));
        (projected, s2_origin, s2_task, proj_dist)
    }
}

impl Optimizer for DexPilotOptimizer {
    fn robot(&self) -> &RobotWrapper { &self.robot }
    fn robot_mut(&mut self) -> &mut RobotWrapper { &mut self.robot }
    fn adaptor(&self) -> Option<&MimicJointAdaptor> { self.adaptor.as_ref() }
    fn adaptor_mut(&mut self) -> Option<&mut MimicJointAdaptor> { self.adaptor.as_mut() }
    fn idx_pin2target(&self) -> &[usize] { &self.idx_pin2target }
    fn idx_pin2fixed(&self) -> &[usize] { &self.idx_pin2fixed }
    fn retargeting_type(&self) -> &str { self.retargeting_type_str }
    fn ftol_abs(&self) -> f64 { 1e-6 }

    fn evaluate(
        &mut self,
        ref_value: &[f64],
        fixed_qpos: &[f64],
        last_qpos: &[f64],
        x: &[f64],
        grad: Option<&mut [f64]>,
    ) -> f64 {
        // Update projection state
        let n_vecs = self.origin_link_names.len();
        let vec_dist: Vec<f64> = ref_value
            .chunks(3).take(self.len_proj)
            .map(|c| (c[0].powi(2) + c[1].powi(2) + c[2].powi(2)).sqrt())
            .collect();

        // S1 projection update
        for i in 0..self.len_s1 {
            if vec_dist[i] < self.project_dist { self.projected[i] = true; }
            if vec_dist[i] > self.escape_dist { self.projected[i] = false; }
        }
        // S2 projection: AND of S1 parents
        for i in self.len_s1..self.len_proj {
            self.projected[i] = self.projected[self.s2_project_index_origin[i - self.len_s1]]
                && self.projected[self.s2_project_index_task[i - self.len_s1]]
                && vec_dist[i] <= 0.03;
        }

        // Build weights
        let normal_weight: Vec<f64> = vec![1.0; self.len_proj];
        let mut high_weight = vec![200.0; self.len_s1];
        high_weight.extend(std::iter::repeat(400.0).take(self.len_s2));
        let weight: Vec<f64> = self.projected.iter().enumerate()
            .map(|(i, &p)| if p { high_weight[i] } else { normal_weight[i] })
            .chain(std::iter::repeat((self.len_proj + self.num_fingers) as f64).take(self.num_fingers))
            .collect();

        // Build reference vector with projection
        let normal_vec: Vec<nalgebra::Vector3<f64>> = ref_value
            .chunks(3).map(|c| nalgebra::Vector3::new(c[0] * self.scaling, c[1] * self.scaling, c[2] * self.scaling))
            .collect();

        let mut reference_vec = normal_vec.clone();
        for i in 0..self.len_proj {
            if self.projected[i] {
                let dist = vec_dist[i] + 1e-6;
                let dir = nalgebra::Vector3::new(
                    ref_value[i * 3] / dist, ref_value[i * 3 + 1] / dist, ref_value[i * 3 + 2] / dist,
                );
                reference_vec[i] = dir * self.projected_dist[i];
            }
        }

        // FK
        let dof = self.robot.dof();
        let mut qpos = vec![0.0; dof];
        for (i, &idx) in self.idx_pin2fixed.iter().enumerate() { qpos[idx] = fixed_qpos[i]; }
        for (i, &idx) in self.idx_pin2target.iter().enumerate() { qpos[idx] = x[i]; }

        if let Some(ref adaptor) = self.adaptor { adaptor.forward_qpos(&mut qpos); }
        self.robot.compute_forward_kinematics(&qpos).unwrap();

        let body_pos: Vec<nalgebra::Vector3<f64>> = self.computed_link_names.iter()
            .map(|name| {
                let pose = self.robot.get_link_pose(name).unwrap();
                nalgebra::Vector3::new(pose[(0, 3)], pose[(1, 3)], pose[(2, 3)])
            }).collect();

        let robot_vecs: Vec<nalgebra::Vector3<f64>> = (0..n_vecs)
            .map(|i| body_pos[self.task_link_indices[i]] - body_pos[self.origin_link_indices[i]])
            .collect();

        // Weighted Huber loss
        let mut loss = 0.0;
        let mut vec_grads = vec![nalgebra::Vector3::zeros(); n_vecs];
        for i in 0..n_vecs {
            let diff = robot_vecs[i] - reference_vec[i];
            let dist = diff.norm();
            let (li, gf) = if dist < self.huber_delta {
                (0.5 * dist * dist / self.huber_delta, 1.0 / self.huber_delta)
            } else {
                (dist - 0.5 * self.huber_delta, 1.0 / dist.max(1e-10))
            };
            let w = weight[i] / (n_vecs as f64);
            loss += li * w;
            vec_grads[i] = diff * gf * w;
        }

        if let Some(g) = grad {
            let mut grad_qpos = vec![0.0; self.idx_pin2target.len()];

            for (i, link_name) in self.computed_link_names.iter().enumerate() {
                let link_j = self.robot.compute_single_link_jacobian(&qpos, link_name).unwrap();
                let link_pose = self.robot.get_link_pose(link_name).unwrap();
                let link_rot = link_pose.fixed_view::<3, 3>(0, 0).into_owned();
                let world_j = &link_rot * &link_j;

                let mut grad_pos = nalgebra::Vector3::zeros();
                for v in 0..n_vecs {
                    if self.origin_link_indices[v] == i { grad_pos -= vec_grads[v]; }
                    if self.task_link_indices[v] == i { grad_pos += vec_grads[v]; }
                }

                if let Some(ref adaptor) = self.adaptor {
                    let mut m = DMatrix::zeros(3, dof);
                    for col in 0..world_j.ncols() { for row in 0..3 { m[(row, col)] = world_j[(row, col)]; } }
                    let adapted_j = adaptor.backward_jacobian(&m);
                    for col in 0..adapted_j.ncols() { grad_qpos[col] += grad_pos.dot(&adapted_j.column(col)); }
                } else {
                    for col in 0..world_j.ncols().min(grad_qpos.len()) {
                        grad_qpos[col] += grad_pos.dot(&world_j.column(col));
                    }
                }
            }

            for (i, gv) in grad_qpos.iter_mut().enumerate() {
                *gv += 2.0 * self.norm_delta * (x[i] - last_qpos[i]);
            }
            g.copy_from_slice(&grad_qpos);
        }

        let smooth: f64 = x.iter().zip(last_qpos.iter())
            .map(|(xi, li)| self.norm_delta * (xi - li).powi(2)).sum();
        loss + smooth
    }
}
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p dexi`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/dexi/src/optimizer.rs
git commit -m "feat: add VectorOptimizer and DexPilotOptimizer"
```

---

### Task 11: SeqRetargeting

**Files:**
- Create: `crates/dexi/src/seq_retargeting.rs`

- [ ] **Step 1: Write seq_retargeting.rs**

```rust
use anyhow::Result;
use nalgebra::{Matrix3, Matrix4, UnitQuaternion, Vector3};

use crate::constants::{HandType, OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT, DUMMY_JOINT_NAMES};
use crate::filter::LpFilter;
use crate::mimic::MimicJointAdaptor;
use crate::optimizer::Optimizer;

pub struct SeqRetargeting {
    pub optimizer: Box<dyn Optimizer>,
    joint_limits: Vec<(f64, f64)>,
    last_qpos: Vec<f64>,
    lp_filter: Option<LpFilter>,
    pub has_joint_limits: bool,
}

impl SeqRetargeting {
    pub fn new(
        optimizer: Box<dyn Optimizer>,
        has_joint_limits: bool,
        lp_filter: Option<LpFilter>,
    ) -> Self {
        let robot = optimizer.robot();
        let limits = robot.joint_limits().to_vec();
        let target_limits: Vec<(f64, f64)> = optimizer
            .idx_pin2target()
            .iter()
            .map(|&i| if has_joint_limits { limits[i] } else { (-1e4, 1e4) })
            .collect();

        let last_qpos: Vec<f64> = target_limits.iter().map(|(l, u)| (l + u) / 2.0).collect();

        // Set joint limits on optimizer
        // (done via optimizer.set_joint_limit which maps to slsqp bounds — handled in retarget())

        Self {
            optimizer,
            joint_limits: target_limits,
            last_qpos,
            lp_filter,
            has_joint_limits,
        }
    }

    pub fn retarget(&mut self, ref_value: &[f64], fixed_qpos: &[f64]) -> Result<Vec<f64>> {
        // Clip last_qpos to joint limits
        let clipped: Vec<f64> = self.last_qpos.iter()
            .zip(self.joint_limits.iter())
            .map(|(&q, &(l, u))| q.clamp(l, u))
            .collect();

        let qpos_target = self.optimizer.retarget(ref_value, fixed_qpos, &clipped)?;
        self.last_qpos = qpos_target.clone();

        // Reassemble full qpos
        let dof = self.optimizer.robot().dof();
        let mut robot_qpos = vec![0.0; dof];
        for (i, &idx) in self.optimizer.idx_pin2fixed().iter().enumerate() {
            robot_qpos[idx] = fixed_qpos[i];
        }
        for (i, &idx) in self.optimizer.idx_pin2target().iter().enumerate() {
            robot_qpos[idx] = qpos_target[i];
        }

        if let Some(ref adaptor) = self.optimizer.adaptor() {
            let mut q = robot_qpos.clone();
            adaptor.forward_qpos(&mut q);
            robot_qpos = q;
        }

        if let Some(ref mut filter) = self.lp_filter {
            robot_qpos = filter.next(&robot_qpos);
        }

        Ok(robot_qpos)
    }

    pub fn set_qpos(&mut self, robot_qpos: &[f64]) {
        let target: Vec<f64> = self.optimizer.idx_pin2target()
            .iter().map(|&i| robot_qpos[i]).collect();
        self.last_qpos = target;
    }

    pub fn reset(&mut self) {
        self.last_qpos = self.joint_limits.iter().map(|(l, u)| (l + u) / 2.0).collect();
        if let Some(ref mut f) = self.lp_filter { f.reset(); }
    }

    pub fn joint_names(&self) -> &[String] {
        self.optimizer.robot().joint_names()
    }

    /// Analytical warm-start: compute 6-DOF dummy joint angles from wrist pose.
    pub fn warm_start(
        &mut self,
        wrist_pos: &[f64],
        wrist_quat: &[f64],
        hand_type: HandType,
        is_mano_convention: bool,
    ) -> Result<()> {
        if wrist_pos.len() != 3 || wrist_quat.len() != 4 {
            anyhow::bail!("wrist_pos must be 3D, wrist_quat 4D");
        }

        let operator2mano = if is_mano_convention {
            match hand_type {
                HandType::Right => Matrix3::from_rows(&[
                    nalgebra::RowVector3::from(OPERATOR2MANO_RIGHT[0]),
                    nalgebra::RowVector3::from(OPERATOR2MANO_RIGHT[1]),
                    nalgebra::RowVector3::from(OPERATOR2MANO_RIGHT[2]),
                ]),
                HandType::Left => Matrix3::from_rows(&[
                    nalgebra::RowVector3::from(OPERATOR2MANO_LEFT[0]),
                    nalgebra::RowVector3::from(OPERATOR2MANO_LEFT[1]),
                    nalgebra::RowVector3::from(OPERATOR2MANO_LEFT[2]),
                ]),
            }
        } else {
            Matrix3::identity()
        };

        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            wrist_quat[3], wrist_quat[0], wrist_quat[1], wrist_quat[2],
        ));
        let rot = q.to_rotation_matrix().into_inner();

        let mut target_wrist_pose = Matrix4::identity();
        target_wrist_pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&(rot * operator2mano.transpose()));
        target_wrist_pose.fixed_view_mut::<3, 1>(0, 3).copy_from(&Vector3::new(wrist_pos[0], wrist_pos[1], wrist_pos[2]));

        // Set dummy joints to zero and compute FK
        let dof = self.optimizer.robot().dof();
        let mut new_qpos = vec![0.0; dof];
        for (i, name) in self.optimizer.robot().joint_names().iter().enumerate() {
            if DUMMY_JOINT_NAMES.contains(&name.as_str()) {
                new_qpos[i] = 0.0;
            } else {
                new_qpos[i] = 0.0; // neutral
            }
        }

        self.optimizer.robot_mut().compute_forward_kinematics(&new_qpos)?;

        let wrist_link = "dummy_z_rotation_link";
        let root2wrist_inv = {
            let pose = self.optimizer.robot().get_link_pose(wrist_link)?;
            let iso = nalgebra::Isometry3::from_matrix_unchecked(pose);
            iso.inverse().to_matrix()
        };

        let target_root_pose = target_wrist_pose * root2wrist_inv;

        // Extract position + Euler angles (intrinsic XYZ)
        let pos = target_root_pose.fixed_view::<3, 1>(0, 3);
        let rot_mat = target_root_pose.fixed_view::<3, 3>(0, 0).into_owned();
        let euler = rot_mat.euler_angles();

        let pose_vec = [pos[(0, 0)], pos[(1, 0)], pos[(2, 0)], euler.0, euler.1, euler.2];

        for (i, name) in self.optimizer.robot().joint_names().iter().enumerate() {
            if let Some(di) = DUMMY_JOINT_NAMES.iter().position(|&dn| dn == name.as_str()) {
                if let Some(ti) = self.optimizer.idx_pin2target().iter().position(|&idx| idx == i) {
                    self.last_qpos[ti] = pose_vec[di];
                }
            }
        }

        Ok(())
    }
}
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p dexi`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/dexi/src/seq_retargeting.rs
git commit -m "feat: add SeqRetargeting with warm-start, LP filter, and joint limits"
```

---

### Task 12: RetargetingConfig — YAML Parsing

**Files:**
- Create: `crates/dexi/src/config.rs`
- Create: `configs/` directories and copy YAML files

- [ ] **Step 1: Write config.rs**

```rust
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::constants::DUMMY_JOINT_NAMES;
use crate::filter::LpFilter;
use crate::mimic::MimicJointAdaptor;
use crate::optimizer::{Optimizer, PositionOptimizer, VectorOptimizer, DexPilotOptimizer};
use crate::robot::RobotWrapper;
use crate::seq_retargeting::SeqRetargeting;
use crate::urdf_ext::insert_dummy_joints;

#[derive(Debug, Deserialize)]
struct YamlRoot {
    retargeting: YamlConfig,
}

#[derive(Debug, Deserialize)]
struct YamlConfig {
    #[serde(rename = "type")]
    retargeting_type: String,
    urdf_path: String,
    #[serde(default)]
    add_dummy_free_joint: bool,
    #[serde(default)]
    target_joint_names: Option<Vec<String>>,
    #[serde(default)]
    target_link_names: Option<Vec<String>>,
    #[serde(default)]
    target_link_human_indices: Option<Vec<usize>>,
    #[serde(default)]
    target_origin_link_names: Option<Vec<String>>,
    #[serde(default)]
    target_task_link_names: Option<Vec<String>>,
    #[serde(default)]
    finger_tip_link_names: Option<Vec<String>>,
    #[serde(default)]
    wrist_link_name: Option<String>,
    #[serde(default)]
    scaling_factor: f64,
    #[serde(default = "default_normal_delta")]
    normal_delta: f64,
    #[serde(default = "default_huber_delta")]
    huber_delta: f64,
    #[serde(default)]
    low_pass_alpha: Option<f64>,
    #[serde(default = "default_true")]
    has_joint_limits: bool,
    #[serde(default)]
    ignore_mimic_joint: bool,
    #[serde(default = "default_project_dist")]
    project_dist: f64,
    #[serde(default = "default_escape_dist")]
    escape_dist: f64,
}

fn default_normal_delta() -> f64 { 4e-3 }
fn default_huber_delta() -> f64 { 2e-2 }
fn default_true() -> bool { true }
fn default_project_dist() -> f64 { 0.03 }
fn default_escape_dist() -> f64 { 0.05 }

/// High-level config that builds a SeqRetargeting pipeline.
pub struct RetargetingConfig {
    yaml: YamlConfig,
    default_urdf_dir: PathBuf,
}

impl RetargetingConfig {
    pub fn set_default_urdf_dir(path: &Path) {
        // Global state: set where relative URDF paths resolve from.
        // This is managed by the caller.
        std::env::set_var("DEXI_URDF_DIR", path.to_string_lossy().to_string());
    }

    pub fn from_file(path: &Path, overrides: Option<&[(String, serde_yaml::Value)]>) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        let mut root: YamlRoot = serde_yaml::from_str(&content)
            .with_context(|| "Failed to parse YAML")?;

        if let Some(overrides) = overrides {
            for (key, val) in overrides {
                // Apply override — simple approach for common keys
                match key.as_str() {
                    "low_pass_alpha" => root.retargeting.low_pass_alpha = val.as_f64(),
                    "scaling_factor" => root.retargeting.scaling_factor = val.as_f64().unwrap_or(1.0),
                    "normal_delta" => root.retargeting.normal_delta = val.as_f64().unwrap_or(4e-3),
                    _ => {}
                }
            }
        }

        let default_urdf_dir = PathBuf::from(
            std::env::var("DEXI_URDF_DIR").unwrap_or_else(|_| ".".to_string()),
        );

        Ok(Self {
            yaml: root.retargeting,
            default_urdf_dir,
        })
    }

    pub fn build(&self) -> Result<SeqRetargeting> {
        let cfg = &self.yaml;

        // Resolve URDF path
        let urdf_path = PathBuf::from(&cfg.urdf_path);
        let urdf_path = if urdf_path.is_absolute() {
            urdf_path
        } else {
            self.default_urdf_dir.join(&urdf_path)
        };

        // Insert dummy joints if needed
        let effective_urdf = if cfg.add_dummy_free_joint {
            insert_dummy_joints(&urdf_path)?
        } else {
            urdf_path.clone()
        };

        // Load robot
        let robot = RobotWrapper::from_urdf_file(&effective_urdf)?;

        // Determine target joint names
        let target_joint_names: Vec<String> = match &cfg.target_joint_names {
            Some(names) => {
                if cfg.add_dummy_free_joint {
                    DUMMY_JOINT_NAMES.iter().map(|s| s.to_string()).chain(names.clone()).collect()
                } else {
                    names.clone()
                }
            }
            None => robot.joint_names().to_vec(),
        };

        let lp_filter = match cfg.low_pass_alpha {
            Some(alpha) if (0.0..=1.0).contains(&alpha) => Some(LpFilter::new(alpha)),
            _ => None,
        };

        let retargeting_type = cfg.retargeting_type.to_lowercase();

        let optimizer: Box<dyn Optimizer> = match retargeting_type.as_str() {
            "position" => {
                let link_names = cfg.target_link_names.as_ref()
                    .context("Position retargeting requires target_link_names")?;
                let human_indices = cfg.target_link_human_indices.as_ref()
                    .context("Position retargeting requires target_link_human_indices")?;

                Box::new(PositionOptimizer::new(
                    robot, &target_joint_names, link_names, human_indices,
                    cfg.huber_delta, cfg.normal_delta,
                )?)
            }
            "vector" => {
                let origin = cfg.target_origin_link_names.as_ref()
                    .context("Vector retargeting requires target_origin_link_names")?;
                let task = cfg.target_task_link_names.as_ref()
                    .context("Vector retargeting requires target_task_link_names")?;
                let human_indices = cfg.target_link_human_indices.as_ref()
                    .context("Vector retargeting requires target_link_human_indices")?;

                Box::new(VectorOptimizer::new(
                    robot, &target_joint_names, origin, task, human_indices,
                    cfg.huber_delta, cfg.normal_delta, cfg.scaling_factor,
                )?)
            }
            "dexpilot" => {
                let fingers = cfg.finger_tip_link_names.as_ref()
                    .context("DexPilot retargeting requires finger_tip_link_names")?;
                let wrist = cfg.wrist_link_name.as_ref()
                    .context("DexPilot retargeting requires wrist_link_name")?;

                Box::new(DexPilotOptimizer::new(
                    robot, &target_joint_names, fingers, wrist,
                    cfg.target_link_human_indices.as_deref(),
                    cfg.huber_delta, cfg.normal_delta,
                    cfg.project_dist, cfg.escape_dist, cfg.scaling_factor,
                )?)
            }
            _ => anyhow::bail!("Unknown retargeting type: {}", cfg.retargeting_type),
        };

        // Parse mimic joints (deferred — read from URDF directly)
        // For now, no mimic joint adaptor by default.
        // Full mimic support requires reading joint.mimic from urdf-rs parsed robot.

        Ok(SeqRetargeting::new(optimizer, cfg.has_joint_limits, lp_filter))
    }
}
```

- [ ] **Step 2: Copy config YAML files**

```bash
cp -r reference/dex-retargeting/src/dex_retargeting/configs configs/
```

- [ ] **Step 3: Create symlink to robot assets**

```bash
ln -s ../reference/dex-retargeting/assets assets
```

- [ ] **Step 4: Verify build**

Run: `cargo build -p dexi`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add crates/dexi/src/config.rs configs/ assets
git commit -m "feat: add RetargetingConfig with YAML parsing and build factory
Copy all retargeting config YAML files from reference.
Symlink assets/ to reference robot URDFs."
```

---

### Task 13: Python Bindings (dexi-py)

**Files:**
- Modify: `crates/dexi-py/src/lib.rs`

- [ ] **Step 1: Write PyO3 bindings**

`crates/dexi-py/src/lib.rs`:
```rust
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::path::PathBuf;

use dexi::config::RetargetingConfig;
use dexi::seq_retargeting::SeqRetargeting;
use dexi::constants::{RobotName, RetargetingType, HandType};

#[pyclass]
#[pyo3(name = "RobotName")]
pub enum PyRobotName {
    Allegro,
    Shadow,
    Svh,
    Leap,
    Ability,
    Inspire,
    Panda,
}

#[pyclass]
#[pyo3(name = "RetargetingType")]
pub enum PyRetargetingType {
    Vector,
    Position,
    DexPilot,
}

#[pyclass]
#[pyo3(name = "HandType")]
pub enum PyHandType {
    Right,
    Left,
}

#[pyclass]
#[pyo3(name = "RetargetingConfig")]
pub struct PyRetargetingConfig {
    inner: RetargetingConfig,
}

#[pymethods]
impl PyRetargetingConfig {
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let config = RetargetingConfig::from_file(&PathBuf::from(path), None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: config })
    }

    #[staticmethod]
    fn set_default_urdf_dir(path: &str) {
        RetargetingConfig::set_default_urdf_dir(&PathBuf::from(path));
    }

    fn build(&self) -> PyResult<PySeqRetargeting> {
        let seq = self.inner.build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PySeqRetargeting { inner: seq })
    }
}

#[pyclass]
#[pyo3(name = "SeqRetargeting")]
pub struct PySeqRetargeting {
    inner: SeqRetargeting,
}

#[pymethods]
impl PySeqRetargeting {
    fn retarget(&mut self, ref_value: Vec<f64>, fixed_qpos: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.retarget(&ref_value, &fixed_qpos)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn set_qpos(&mut self, robot_qpos: Vec<f64>) {
        self.inner.set_qpos(&robot_qpos);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn joint_names(&self) -> Vec<String> {
        self.inner.joint_names().to_vec()
    }
}

#[pymodule]
fn dexi_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRobotName>()?;
    m.add_class::<PyRetargetingType>()?;
    m.add_class::<PyHandType>()?;
    m.add_class::<PyRetargetingConfig>()?;
    m.add_class::<PySeqRetargeting>()?;
    Ok(())
}
```

- [ ] **Step 2: Make config.rs module public in dexi**

Edit `crates/dexi/src/lib.rs` — ensure all modules are `pub` (they already are).

- [ ] **Step 3: Build maturin package**

Run: `cd crates/dexi-py && pip install maturin && maturin develop`
Expected: builds Python package, `import dexi_py` works

- [ ] **Step 4: Commit**

```bash
git add crates/dexi-py/src/lib.rs
git commit -m "feat: add PyO3 Python bindings for RetargetingConfig and SeqRetargeting"
```

---

### Task 14: Rust Integration Tests

**Files:**
- Create: `tests/test_optimizer.rs`

- [ ] **Step 1: Write integration test**

`tests/test_optimizer.rs`:
```rust
use std::path::PathBuf;

use dexi::config::RetargetingConfig;
use dexi::constants::{HandType, RobotName, RetargetingType, ALL_ROBOT_NAMES, DEXPILOT_ROBOT_NAMES};

fn setup_urdf_dir() {
    let robot_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/robots/hands");
    RetargetingConfig::set_default_urdf_dir(&robot_dir);
}

fn config_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("configs")
}

#[test]
fn test_position_optimizer_all_robots() {
    setup_urdf_dir();
    for robot_name in ALL_ROBOT_NAMES {
        for hand_type in &[HandType::Right, HandType::Left] {
            let config_path = dexi::constants::get_default_config_path(
                &config_dir(), *robot_name, RetargetingType::Position, *hand_type,
            );
            if config_path.is_none() { continue; }

            let config = RetargetingConfig::from_file(&config_path.unwrap(), None).unwrap();
            let mut seq = config.build().unwrap();

            // Run a few optimization iterations with random targets
            let target_dof = seq.optimizer.idx_pin2target().len();
            let fixed_dof = seq.optimizer.idx_pin2fixed().len();

            for _ in 0..10 {
                let target_val = vec![0.1; target_dof * 3]; // simplified
                let fixed = vec![0.0; fixed_dof];
                let result = seq.retarget(&target_val, &fixed);
                assert!(result.is_ok(), "Retarget failed for {:?} {:?}", robot_name, hand_type);
            }
        }
    }
}
```

- [ ] **Step 2: Add tests to workspace Cargo.toml**

Integration tests in `tests/` at workspace root auto-discovered. Verify.

- [ ] **Step 3: Run tests**

Run: `cargo test --test test_optimizer`
Expected: compiles and runs

- [ ] **Step 4: Commit**

```bash
git add tests/test_optimizer.rs
git commit -m "test: add integration tests parameterized over all robot/hand combos"
```

---

### Task 15: Numerical Comparison Script

**Files:**
- Create: `tests/comparison/compare_results.py`

- [ ] **Step 1: Write comparison script**

`tests/comparison/compare_results.py`:
```python
"""Compare retargeting results between Python dex_retargeting and Rust dexi."""
import sys
import time
import numpy as np

sys.path.insert(0, "reference/dex-retargeting/src")

from dex_retargeting.constants import (
    ROBOT_NAMES,
    get_default_config_path,
    RetargetingType,
    HandType,
)
from dex_retargeting.retargeting_config import RetargetingConfig as PyConfig
from dex_retargeting.robot_wrapper import RobotWrapper as PyRobot

import dexi  # Rust bindings


def compare_position_optimizer(robot_name, hand_type):
    config_dir = "configs"
    robot_dir = "assets/robots/hands"

    PyConfig.set_default_urdf_dir(robot_dir)
    dexi.RetargetingConfig.set_default_urdf_dir(robot_dir)

    config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)
    if config_path is None:
        return

    override = {"normal_delta": 0}
    py_config = PyConfig.load_from_file(str(config_path), override)
    py_seq = py_config.build()

    rs_config = dexi.RetargetingConfig.from_file(str(config_path))
    rs_seq = rs_config.build()

    np.random.seed(42)
    errors = []

    for i in range(50):
        robot = py_seq.optimizer.robot
        optimizer = py_seq.optimizer

        joint_limits = robot.joint_limits
        random_qpos = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])

        robot.compute_forward_kinematics(random_qpos)
        target_pos = np.array(
            [robot.get_link_pose(i)[:3, 3] for i in optimizer.target_link_indices]
        )
        fixed_qpos = random_qpos[optimizer.idx_pin2fixed]

        # Python result
        py_seq.set_qpos(random_qpos)
        py_result = py_seq.retarget(target_pos, fixed_qpos=fixed_qpos)

        # Rust result (convert to numpy)
        rs_result = np.array(rs_seq.retarget(
            target_pos.flatten().tolist(),
            fixed_qpos.tolist(),
        ))

        err = np.max(np.abs(py_result - rs_result))
        errors.append(err)

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"  {robot_name.name} {hand_type.name}: mean_err={mean_err:.6f}, max_err={max_err:.6f}")
    return mean_err, max_err


if __name__ == "__main__":
    all_errors = []
    for robot_name in ROBOT_NAMES:
        for hand_type in [HandType.right, HandType.left]:
            result = compare_position_optimizer(robot_name, hand_type)
            if result:
                all_errors.append(result)

    if all_errors:
        mean_e = np.mean([r[0] for r in all_errors])
        max_e = np.max([r[1] for r in all_errors])
        print(f"\nOverall: mean_err={mean_e:.6f}, max_err={max_e:.6f}")
        assert mean_e < 1e-3, f"Numerical comparison failed: {mean_e}"
        print("PASS: All comparisons within tolerance.")
```

- [ ] **Step 2: Commit**

```bash
git add tests/comparison/compare_results.py
git commit -m "test: add numerical comparison script (Python vs Rust)"
```

---

### Task 16: Mimic Joint Parsing in Config

**Files:**
- Modify: `crates/dexi/src/config.rs`

- [ ] **Step 1: Add mimic joint parsing**

Parse mimic joints from urdf-rs `Joint.mimic` field and wire into optimizer adaptor.

In `RetargetingConfig::build()`, after creating the optimizer, add:

```rust
// Parse mimic joints from URDF
let urdf_robot = urdf_rs::read_from_file(&urdf_path)
    .with_context(|| "Failed to parse URDF for mimic joints")?;

let mut source_names = Vec::new();
let mut mimic_names = Vec::new();
let mut multipliers = Vec::new();
let mut offsets = Vec::new();

for joint in &urdf_robot.joints {
    if let Some(ref mimic) = joint.mimic {
        mimic_names.push(joint.name.clone());
        source_names.push(mimic.joint.clone());
        multipliers.push(mimic.multiplier);
        offsets.push(mimic.offset);
    }
}

if !mimic_names.is_empty() && !cfg.ignore_mimic_joint {
    let adaptor = MimicJointAdaptor::new(
        optimizer.robot(),
        &target_joint_names,
        &source_names,
        &mimic_names,
        &multipliers,
        &offsets,
    )?;
    optimizer.set_kinematic_adaptor(adaptor);
}
```

Add the `set_kinematic_adaptor` method to the `Optimizer` trait and all three optimizer structs.

- [ ] **Step 2: Verify build**

Run: `cargo build -p dexi`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/dexi/src/config.rs crates/dexi/src/optimizer.rs
git commit -m "feat: wire mimic joint parsing from URDF into optimizer adaptor"
```

---

### Task 17: Final Integration and Test Pass

- [ ] **Step 1: Run full test suite**

Run: `cargo test`
Expected: all tests pass

- [ ] **Step 2: Build Python bindings**

Run: `cd crates/dexi-py && maturin develop`
Expected: `import dexi` works

- [ ] **Step 3: Run comparison script**

Run: `python tests/comparison/compare_results.py`
Expected: < 1e-3 error

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: final integration — all tests pass, Python-Rust comparison verified"
```

---

## Execution Order

```
Task 1 (scaffold) → Task 2 (constants) → Task 3 (filter) → Task 4 (urdf_ext)
→ Task 5 (chain_builder) → Task 6 (robot) → Task 7 (mimic) → Task 8 (huber)
→ Task 9 (position optimizer) → Task 10 (vector + dexpilot)
→ Task 11 (seq_retargeting) → Task 12 (config) → Task 13 (py bindings)
→ Task 14 (tests) → Task 15 (comparison) → Task 16 (mimic parsing)
→ Task 17 (final pass)
```

Tasks 8, 2, 3 can be done in parallel. Tasks 9-10-11 are sequential (depend on 6-7-8).
