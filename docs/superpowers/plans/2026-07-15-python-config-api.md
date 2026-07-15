# Python Config API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users define `RetargetingConfig` objects directly in Python (kwargs or dict) as an alternative to YAML files, with reference-grade validation shared by both paths.

**Architecture:** The Rust struct `dexi::RetargetingConfig` stays the single schema. A new `validate()` method in the core runs on both YAML parsing and Python construction. The pyo3 class gains a `#[new]` kwargs constructor, getters for every field, and a `__repr__`; the Python wrapper's `load_config()` becomes polymorphic over `str | Path | dict | RetargetingConfig`. Relative URDF paths in Python-defined configs anchor to the wheel's bundled assets, with a new cwd last-resort for all configs.

**Tech Stack:** Rust 2021, pyo3 0.28, maturin, no new dependencies (Rust or Python).

**Spec:** `docs/superpowers/specs/2026-07-15-python-config-api-design.md`

## Global Constraints

- No new dependencies: Rust crates stay as-is; Python wheel keeps zero runtime deps.
- Runtime code must be Python 3.10-compatible (`requires-python` moves to `>=3.10` in separate release-tooling work; stubs may use modern syntax — they are never executed).
- Configs are immutable from Python: getters only, no setters.
- The constructor kwarg is `type` (matches the YAML key); the existing `type_` getter keeps its name.
- Validation errors are `ValueError` naming the offending field with expected vs. actual shape; unknown kwargs are `TypeError`; `build()` failures stay `RuntimeError`.
- NEVER mention `reference/` paths in README, docs/, examples/, or scripts/ — `scripts/check_release.py` fails the release if you do.
- Commit messages use gitmoji (✨ feature, ✅ tests, 📝 docs, 🏷️ types). No Co-Authored-By lines.
- All commands run from the repo root. `.venv` already exists with maturin installed (`just setup` otherwise).
- Rust steps: run `cargo fmt` before every commit (`just fmt` runs `cargo fmt --check`).

---

### Task 1: `RetargetingConfig::validate()` in the Rust core

**Files:**
- Create: `crates/dexi/tests/test_config_validation.rs`
- Modify: `crates/dexi/src/retargeting_config.rs` (add `validate()`; call it at the end of `from_value` and at the top of `build()`)

**Interfaces:**
- Consumes: `dexi::RetargetingConfig` struct (all fields `pub`), `RetargetingType` from `dexi::constants`.
- Produces: `pub fn validate(&self) -> Result<(), String>` on `RetargetingConfig` — Task 3's constructor calls this exact method.

- [ ] **Step 1: Write the failing tests**

Create `crates/dexi/tests/test_config_validation.rs`:

```rust
//! Test: RetargetingConfig::validate() rejects malformed configs with
//! field-specific messages, on both direct construction and YAML loading.

use dexi::{RetargetingConfig, RetargetingType};

fn base(type_: RetargetingType) -> RetargetingConfig {
    RetargetingConfig {
        type_,
        urdf_path: "allegro_hand/allegro_hand_right.urdf".to_string(),
        ..Default::default()
    }
}

fn valid_vector() -> RetargetingConfig {
    RetargetingConfig {
        target_origin_link_names: Some(vec!["wrist".into(), "wrist".into()]),
        target_task_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        target_link_human_indices: Some(vec![0, 0, 4, 8]),
        ..base(RetargetingType::Vector)
    }
}

#[test]
fn vector_valid_config_passes() {
    valid_vector().validate().expect("valid vector config must pass");
}

#[test]
fn empty_urdf_path_fails() {
    let config = RetargetingConfig {
        urdf_path: String::new(),
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("urdf_path"), "unexpected message: {err}");
}

#[test]
fn vector_missing_task_links_fails() {
    let config = RetargetingConfig {
        target_task_link_names: None,
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("target_task_link_names"), "unexpected message: {err}");
}

#[test]
fn vector_origin_task_length_mismatch_fails() {
    let config = RetargetingConfig {
        target_task_link_names: Some(vec!["link_15.0_tip".into()]),
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("target_origin_link_names") && err.contains("2") && err.contains("1"),
        "unexpected message: {err}"
    );
}

#[test]
fn vector_human_indices_dim_mismatch_fails() {
    let config = RetargetingConfig {
        target_link_human_indices: Some(vec![0, 4, 8]),
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("target_link_human_indices"), "unexpected message: {err}");
}

#[test]
fn position_valid_config_passes() {
    let config = RetargetingConfig {
        target_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        target_link_human_indices: Some(vec![4, 8]),
        ..base(RetargetingType::Position)
    };
    config.validate().expect("valid position config must pass");
}

#[test]
fn position_indices_length_mismatch_fails() {
    let config = RetargetingConfig {
        target_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        target_link_human_indices: Some(vec![4]),
        ..base(RetargetingType::Position)
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("target_link_human_indices"), "unexpected message: {err}");
}

#[test]
fn position_missing_links_fails() {
    let config = RetargetingConfig {
        target_link_human_indices: Some(vec![4, 8]),
        ..base(RetargetingType::Position)
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("target_link_names"), "unexpected message: {err}");
}

#[test]
fn dexpilot_valid_config_passes() {
    let config = RetargetingConfig {
        finger_tip_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        wrist_link_name: Some("wrist".into()),
        ..base(RetargetingType::DexPilot)
    };
    config.validate().expect("valid dexpilot config must pass");
}

#[test]
fn dexpilot_missing_wrist_fails() {
    let config = RetargetingConfig {
        finger_tip_link_names: Some(vec!["link_15.0_tip".into()]),
        wrist_link_name: None,
        ..base(RetargetingType::DexPilot)
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("wrist_link_name"), "unexpected message: {err}");
}

#[test]
fn yaml_load_with_dim_mismatch_fails() {
    let yaml = r#"
retargeting:
  type: vector
  urdf_path: allegro_hand/allegro_hand_right.urdf
  target_origin_link_names: [ "wrist", "wrist" ]
  target_task_link_names: [ "link_15.0_tip" ]
  target_link_human_indices: [ [ 0, 0 ], [ 4, 8 ] ]
"#;
    let dir = std::env::temp_dir();
    let path = dir.join("dexi_test_invalid_vector.yml");
    std::fs::write(&path, yaml).expect("write temp yaml");
    let err = RetargetingConfig::load_from_path(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(err.contains("target_origin_link_names"), "unexpected message: {err}");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dexi --test test_config_validation`
Expected: COMPILE ERROR — `no method named 'validate' found for struct RetargetingConfig`.

- [ ] **Step 3: Implement `validate()`**

In `crates/dexi/src/retargeting_config.rs`, add inside `impl RetargetingConfig` (after `load_from_path`):

```rust
    /// Validate required fields and dimensions for the configured type.
    ///
    /// Runs on every entry path (YAML parsing, Python construction) and
    /// again as a backstop at the start of `build()`.
    pub fn validate(&self) -> Result<(), String> {
        if self.urdf_path.is_empty() {
            return Err("retargeting config: urdf_path must not be empty".to_string());
        }
        match self.type_ {
            RetargetingType::Position => {
                let links = self
                    .target_link_names
                    .as_ref()
                    .filter(|v| !v.is_empty())
                    .ok_or("position retargeting requires non-empty target_link_names")?;
                let indices = self
                    .target_link_human_indices
                    .as_ref()
                    .filter(|v| !v.is_empty())
                    .ok_or("position retargeting requires non-empty target_link_human_indices")?;
                if indices.len() != links.len() {
                    return Err(format!(
                        "position retargeting: target_link_human_indices has {} entries but target_link_names has {}",
                        indices.len(),
                        links.len()
                    ));
                }
            }
            RetargetingType::Vector => {
                let origin = self
                    .target_origin_link_names
                    .as_ref()
                    .filter(|v| !v.is_empty())
                    .ok_or("vector retargeting requires non-empty target_origin_link_names")?;
                let task = self
                    .target_task_link_names
                    .as_ref()
                    .filter(|v| !v.is_empty())
                    .ok_or("vector retargeting requires non-empty target_task_link_names")?;
                if origin.len() != task.len() {
                    return Err(format!(
                        "vector retargeting: target_origin_link_names has {} entries but target_task_link_names has {}",
                        origin.len(),
                        task.len()
                    ));
                }
                let indices = self
                    .target_link_human_indices
                    .as_ref()
                    .filter(|v| !v.is_empty())
                    .ok_or("vector retargeting requires non-empty target_link_human_indices")?;
                if indices.len() != 2 * origin.len() {
                    return Err(format!(
                        "vector retargeting: target_link_human_indices must flatten to 2 x {} entries, got {}",
                        origin.len(),
                        indices.len()
                    ));
                }
            }
            RetargetingType::DexPilot => {
                self.finger_tip_link_names
                    .as_ref()
                    .filter(|v| !v.is_empty())
                    .ok_or("dexpilot retargeting requires non-empty finger_tip_link_names")?;
                self.wrist_link_name
                    .as_ref()
                    .filter(|s| !s.is_empty())
                    .ok_or("dexpilot retargeting requires wrist_link_name")?;
            }
        }
        Ok(())
    }
```

Wire it into both entry paths:

1. In `from_value`, replace the final `Ok(Self { ... })` expression with:

```rust
        let config = Self {
            type_,
            urdf_path,
            add_dummy_free_joint,
            target_link_human_indices,
            wrist_link_name,
            target_link_names,
            target_joint_names,
            target_origin_link_names,
            target_task_link_names,
            finger_tip_link_names,
            scaling_factor,
            normal_delta,
            huber_delta,
            project_dist,
            escape_dist,
            has_joint_limits,
            ignore_mimic_joint,
            low_pass_alpha,
            default_urdf_dir: config_dir.to_path_buf(),
        };
        config.validate()?;
        Ok(config)
```

2. In `build()`, add as the first line of the method body (before `let urdf_path = self.resolve_urdf_path()?;`):

```rust
        self.validate()?;
```

Leave the existing `ok_or(...)` extractions inside `build()`'s match arms unchanged — they still unwrap the `Option`s, their error branches are now unreachable backstops.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p dexi --test test_config_validation`
Expected: PASS (11 tests).

Run: `cargo test`
Expected: all existing tests still PASS (existing YAML configs are all valid).

- [ ] **Step 5: Format and commit**

```bash
cargo fmt
git add crates/dexi/src/retargeting_config.rs crates/dexi/tests/test_config_validation.rs
git commit -m "✨ Validate retargeting configs at load and build time"
```

---

### Task 2: cwd last-resort in `resolve_urdf_path`

**Files:**
- Modify: `crates/dexi/src/retargeting_config.rs` (the `resolve_urdf_path` method)
- Test: `crates/dexi/tests/test_config_parsing.rs` (append test)

**Interfaces:**
- Consumes: `RetargetingConfig::resolve_urdf_path()` (existing).
- Produces: same signature; resolution order for relative paths becomes: `default_urdf_dir/<path>` → ancestor `assets/robots/hands/<path>` search → `cwd/<path>`.

- [ ] **Step 1: Write the failing test**

Append to `crates/dexi/tests/test_config_parsing.rs`:

```rust
/// Relative URDF paths fall back to the current working directory as a
/// last resort. cargo test runs with cwd = crates/dexi, so a file relative
/// to the crate root is only reachable through the cwd fallback
/// (resolution is existence-based, so any file works as a probe).
#[test]
fn test_resolve_urdf_path_cwd_fallback() {
    let config = RetargetingConfig {
        urdf_path: "src/constants.rs".to_string(),
        default_urdf_dir: std::env::temp_dir(),
        ..Default::default()
    };
    let resolved = config
        .resolve_urdf_path()
        .expect("cwd fallback should resolve");
    assert!(resolved.ends_with("src/constants.rs"), "got {resolved:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dexi --test test_config_parsing test_resolve_urdf_path_cwd_fallback`
Expected: FAIL with `URDF path src/constants.rs does not exist (tried ...)`.

- [ ] **Step 3: Implement the fallback**

In `resolve_urdf_path`, after the `for ancestor in ...` loop and before the `let attempted = attempted ...` error construction, insert:

```rust
            if let Ok(cwd) = std::env::current_dir() {
                let candidate = cwd.join(path);
                attempted.push(candidate.clone());
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p dexi --test test_config_parsing`
Expected: PASS (all tests, including the new one).

- [ ] **Step 5: Format and commit**

```bash
cargo fmt
git add crates/dexi/src/retargeting_config.rs crates/dexi/tests/test_config_parsing.rs
git commit -m "✨ Add cwd fallback for relative URDF paths"
```

---

### Task 3: pyo3 kwargs constructor, full getters, `__repr__`

**Files:**
- Modify: `crates/dexi-py/src/lib.rs`

**Interfaces:**
- Consumes: `RetargetingConfig::validate()` from Task 1; `dexi::RetargetingType`.
- Produces: `dexi_rs.RetargetingConfig(type, urdf_path, *, add_dummy_free_joint=False, target_link_human_indices=None, wrist_link_name=None, target_link_names=None, target_joint_names=None, target_origin_link_names=None, target_task_link_names=None, finger_tip_link_names=None, scaling_factor=1.0, normal_delta=4e-3, huber_delta=2e-2, project_dist=0.03, escape_dist=0.05, has_joint_limits=True, ignore_mimic_joint=False, low_pass_alpha=0.1, urdf_dir=None)` — Tasks 4, 6, 7 call this exact signature. `target_link_human_indices` accepts a flat `[int]` or nested `[[int]]` (row-major flattened). Getters for every field; nested indices are returned flattened.

- [ ] **Step 1: Write the failing test**

Build the current code into the dev venv, then run the assertion script (it must fail):

```bash
VIRTUAL_ENV="$PWD/.venv" PATH="$PWD/.venv/bin:$PATH" maturin develop -m crates/dexi-py/Cargo.toml
```

Then run:

```bash
.venv/bin/python - <<'EOF'
import dexi_rs

cfg = dexi_rs.RetargetingConfig(
    type="vector",
    urdf_path="allegro_hand/allegro_hand_right.urdf",
    target_origin_link_names=["wrist"] * 4,
    target_task_link_names=["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"],
    target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
    scaling_factor=1.6,
    low_pass_alpha=0.2,
)
assert cfg.type_ == "vector"
assert cfg.urdf_path == "allegro_hand/allegro_hand_right.urdf"
assert cfg.target_origin_link_names == ["wrist"] * 4
assert cfg.target_link_human_indices == [0, 0, 0, 0, 4, 8, 12, 16]
assert cfg.scaling_factor == 1.6
assert cfg.low_pass_alpha == 0.2
assert cfg.has_joint_limits is True
assert "vector" in repr(cfg)

# bundled-assets default anchor: build() must find the URDF
retargeting = cfg.build()
qpos = retargeting.retarget([0.03, -0.02, 0.08, 0.04, 0.0, 0.09, 0.03, 0.02, 0.085, 0.02, 0.04, 0.07])
assert len(qpos) == len(retargeting.joint_names)

# flat indices also accepted
flat = dexi_rs.RetargetingConfig(
    type="vector",
    urdf_path="allegro_hand/allegro_hand_right.urdf",
    target_origin_link_names=["wrist"] * 4,
    target_task_link_names=["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"],
    target_link_human_indices=[0, 0, 0, 0, 4, 8, 12, 16],
)
assert flat.target_link_human_indices == [0, 0, 0, 0, 4, 8, 12, 16]

# validation runs at construction
try:
    dexi_rs.RetargetingConfig(
        type="vector",
        urdf_path="x.urdf",
        target_origin_link_names=["wrist", "wrist"],
        target_task_link_names=["link_15.0_tip"],
        target_link_human_indices=[0, 4],
    )
    raise AssertionError("expected ValueError for origin/task mismatch")
except ValueError as exc:
    assert "target_origin_link_names" in str(exc)

# unknown type string
try:
    dexi_rs.RetargetingConfig(type="nope", urdf_path="x.urdf")
    raise AssertionError("expected ValueError for unknown type")
except ValueError:
    pass

# unknown kwarg is a TypeError
try:
    dexi_rs.RetargetingConfig(type="dexpilot", urdf_path="x.urdf", scaling=1.6)
    raise AssertionError("expected TypeError for unknown kwarg")
except TypeError:
    pass

print("task 3 asserts passed")
EOF
```

- [ ] **Step 2: Verify it fails**

Expected: `TypeError` — pyo3 classes without `#[new]` cannot be instantiated ("No constructor defined").

- [ ] **Step 3: Implement constructor, getters, `__repr__`**

In `crates/dexi-py/src/lib.rs`:

1. Extend imports:

```rust
use std::path::{Path, PathBuf};
```

2. Add above `PyRetargetingConfig`:

```rust
/// Human-indices input: flat list, or nested rows flattened row-major
/// (matching the YAML parser's behavior).
#[derive(FromPyObject)]
enum IndicesInput {
    Flat(Vec<usize>),
    Nested(Vec<Vec<usize>>),
}

impl IndicesInput {
    fn flatten(self) -> Vec<usize> {
        match self {
            IndicesInput::Flat(v) => v,
            IndicesInput::Nested(rows) => rows.into_iter().flatten().collect(),
        }
    }
}

/// The bundled robots directory inside the installed dexi_rs package,
/// discovered from the native module's own location.
fn packaged_hands_dir(py: Python<'_>) -> Option<PathBuf> {
    let module = py.import("dexi_rs._native").ok()?;
    let file: String = module.getattr("__file__").ok()?.extract().ok()?;
    let dir = Path::new(&file).parent()?.join("resources/assets/robots/hands");
    if dir.is_dir() {
        Some(dir)
    } else {
        None
    }
}
```

3. Add the constructor to `#[pymethods] impl PyRetargetingConfig` (keep `from_file` and `build` as they are):

```rust
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        r#type,
        urdf_path,
        *,
        add_dummy_free_joint = false,
        target_link_human_indices = None,
        wrist_link_name = None,
        target_link_names = None,
        target_joint_names = None,
        target_origin_link_names = None,
        target_task_link_names = None,
        finger_tip_link_names = None,
        scaling_factor = 1.0,
        normal_delta = 4e-3,
        huber_delta = 2e-2,
        project_dist = 0.03,
        escape_dist = 0.05,
        has_joint_limits = true,
        ignore_mimic_joint = false,
        low_pass_alpha = 0.1,
        urdf_dir = None,
    ))]
    fn new(
        py: Python<'_>,
        r#type: &str,
        urdf_path: &str,
        add_dummy_free_joint: bool,
        target_link_human_indices: Option<IndicesInput>,
        wrist_link_name: Option<String>,
        target_link_names: Option<Vec<String>>,
        target_joint_names: Option<Vec<String>>,
        target_origin_link_names: Option<Vec<String>>,
        target_task_link_names: Option<Vec<String>>,
        finger_tip_link_names: Option<Vec<String>>,
        scaling_factor: f64,
        normal_delta: f64,
        huber_delta: f64,
        project_dist: f64,
        escape_dist: f64,
        has_joint_limits: bool,
        ignore_mimic_joint: bool,
        low_pass_alpha: f64,
        urdf_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let type_ = RetargetingType::from_str(r#type).map_err(value_error)?;
        let default_urdf_dir = urdf_dir
            .or_else(|| packaged_hands_dir(py))
            .unwrap_or_else(|| PathBuf::from("."));
        let inner = RetargetingConfig {
            type_,
            urdf_path: urdf_path.to_string(),
            add_dummy_free_joint,
            target_link_human_indices: target_link_human_indices.map(IndicesInput::flatten),
            wrist_link_name,
            target_link_names,
            target_joint_names,
            target_origin_link_names,
            target_task_link_names,
            finger_tip_link_names,
            scaling_factor,
            normal_delta,
            huber_delta,
            project_dist,
            escape_dist,
            has_joint_limits,
            ignore_mimic_joint,
            low_pass_alpha,
            default_urdf_dir,
        };
        inner.validate().map_err(value_error)?;
        Ok(Self { inner })
    }
```

4. Add the missing getters and `__repr__` to the same `#[pymethods]` block (the existing `type_`, `urdf_path`, `target_joint_names` getters stay):

```rust
    #[getter]
    fn add_dummy_free_joint(&self) -> bool {
        self.inner.add_dummy_free_joint
    }

    #[getter]
    fn target_link_human_indices(&self) -> Option<Vec<usize>> {
        self.inner.target_link_human_indices.clone()
    }

    #[getter]
    fn wrist_link_name(&self) -> Option<String> {
        self.inner.wrist_link_name.clone()
    }

    #[getter]
    fn target_link_names(&self) -> Option<Vec<String>> {
        self.inner.target_link_names.clone()
    }

    #[getter]
    fn target_origin_link_names(&self) -> Option<Vec<String>> {
        self.inner.target_origin_link_names.clone()
    }

    #[getter]
    fn target_task_link_names(&self) -> Option<Vec<String>> {
        self.inner.target_task_link_names.clone()
    }

    #[getter]
    fn finger_tip_link_names(&self) -> Option<Vec<String>> {
        self.inner.finger_tip_link_names.clone()
    }

    #[getter]
    fn scaling_factor(&self) -> f64 {
        self.inner.scaling_factor
    }

    #[getter]
    fn normal_delta(&self) -> f64 {
        self.inner.normal_delta
    }

    #[getter]
    fn huber_delta(&self) -> f64 {
        self.inner.huber_delta
    }

    #[getter]
    fn project_dist(&self) -> f64 {
        self.inner.project_dist
    }

    #[getter]
    fn escape_dist(&self) -> f64 {
        self.inner.escape_dist
    }

    #[getter]
    fn has_joint_limits(&self) -> bool {
        self.inner.has_joint_limits
    }

    #[getter]
    fn ignore_mimic_joint(&self) -> bool {
        self.inner.ignore_mimic_joint
    }

    #[getter]
    fn low_pass_alpha(&self) -> f64 {
        self.inner.low_pass_alpha
    }

    #[getter]
    fn urdf_dir(&self) -> String {
        self.inner.default_urdf_dir.display().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "RetargetingConfig(type='{}', urdf_path='{}', scaling_factor={})",
            self.type_(),
            self.inner.urdf_path,
            self.inner.scaling_factor
        )
    }
```

- [ ] **Step 4: Rebuild and run the test to verify it passes**

```bash
VIRTUAL_ENV="$PWD/.venv" PATH="$PWD/.venv/bin:$PATH" maturin develop -m crates/dexi-py/Cargo.toml
```

Re-run the Step 1 assertion script. Expected: `task 3 asserts passed`.

Run: `cargo test`
Expected: all tests still PASS.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt
git add crates/dexi-py/src/lib.rs
git commit -m "✨ Add Python constructor and full getters for RetargetingConfig"
```

---

### Task 4: polymorphic `load_config`

**Files:**
- Modify: `crates/dexi-py/python/dexi_rs/__init__.py` (the `load_config` function and its docstring)

**Interfaces:**
- Consumes: `RetargetingConfig` constructor from Task 3.
- Produces: `load_config(source: str | Path | dict | RetargetingConfig) -> RetargetingConfig` — Task 6's wheel test and Task 7's docs rely on all four source kinds.

- [ ] **Step 1: Write the failing test**

```bash
.venv/bin/python - <<'EOF'
import dexi_rs

kwargs = dict(
    type="vector",
    urdf_path="allegro_hand/allegro_hand_right.urdf",
    target_origin_link_names=["wrist"] * 4,
    target_task_link_names=["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"],
    target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
    scaling_factor=1.6,
)

# dict (flat)
cfg = dexi_rs.load_config(kwargs)
assert cfg.type_ == "vector" and cfg.scaling_factor == 1.6

# dict (YAML-shaped nesting)
cfg2 = dexi_rs.load_config({"retargeting": kwargs})
assert cfg2.type_ == "vector"

# config object passes through unchanged
assert dexi_rs.load_config(cfg) is cfg

# YAML path still works
cfg3 = dexi_rs.load_config("configs/teleop/allegro_hand_right.yml")
assert cfg3.type_ == "vector"

print("task 4 asserts passed")
EOF
```

Expected: FAIL — `TypeError` or `FileNotFoundError` from `config_path` when given a dict.

- [ ] **Step 2: Implement**

Replace the `load_config` function in `crates/dexi-py/python/dexi_rs/__init__.py`:

```python
def load_config(source: str | Path | dict | RetargetingConfig) -> RetargetingConfig:
    """Return a :class:`RetargetingConfig` from any supported source.

    Accepts a YAML file path (``str`` / ``Path``), an already-built
    :class:`RetargetingConfig` (returned as-is), or a ``dict`` of constructor
    keyword arguments — optionally nested under a ``"retargeting"`` key so a
    YAML file parsed into a dict round-trips.

    Example
    -------
    >>> cfg = load_config("configs/teleop/allegro_hand_right.yml")
    >>> cfg = load_config({"type": "vector", "urdf_path": "...", ...})
    >>> retargeting = cfg.build()
    """

    if isinstance(source, RetargetingConfig):
        return source
    if isinstance(source, dict):
        fields = source.get("retargeting", source)
        if not isinstance(fields, dict):
            raise ValueError("'retargeting' key must map to a dict of config fields")
        return RetargetingConfig(**fields)
    return RetargetingConfig.from_file(str(config_path(source)))
```

- [ ] **Step 3: Reinstall and run the test to verify it passes**

```bash
VIRTUAL_ENV="$PWD/.venv" PATH="$PWD/.venv/bin:$PATH" maturin develop -m crates/dexi-py/Cargo.toml
```

Re-run the Step 1 script. Expected: `task 4 asserts passed`.

- [ ] **Step 4: Commit**

```bash
git add crates/dexi-py/python/dexi_rs/__init__.py
git commit -m "✨ Accept dicts and config objects in load_config"
```

---

### Task 5: type stubs and packaging

**Files:**
- Create: `crates/dexi-py/python/dexi_rs/_native.pyi`
- Create: `crates/dexi-py/python/dexi_rs/py.typed` (empty file)
- Modify: `crates/dexi-py/pyproject.toml` (include entries)

**Interfaces:**
- Consumes: constructor signature from Task 3, `load_config` union from Task 4.
- Produces: `.pyi` stub shipped in the wheel; no runtime behavior change.

- [ ] **Step 1: Write the stub**

Create `crates/dexi-py/python/dexi_rs/_native.pyi`:

```python
from pathlib import Path
from typing import Optional, Sequence, Union

__version__: str

def load_from_file(path: str) -> RetargetingConfig: ...

class RetargetingConfig:
    def __init__(
        self,
        type: str,
        urdf_path: str,
        *,
        add_dummy_free_joint: bool = False,
        target_link_human_indices: Union[Sequence[int], Sequence[Sequence[int]], None] = None,
        wrist_link_name: Optional[str] = None,
        target_link_names: Optional[Sequence[str]] = None,
        target_joint_names: Optional[Sequence[str]] = None,
        target_origin_link_names: Optional[Sequence[str]] = None,
        target_task_link_names: Optional[Sequence[str]] = None,
        finger_tip_link_names: Optional[Sequence[str]] = None,
        scaling_factor: float = 1.0,
        normal_delta: float = 4e-3,
        huber_delta: float = 2e-2,
        project_dist: float = 0.03,
        escape_dist: float = 0.05,
        has_joint_limits: bool = True,
        ignore_mimic_joint: bool = False,
        low_pass_alpha: float = 0.1,
        urdf_dir: Union[str, Path, None] = None,
    ) -> None: ...
    @staticmethod
    def from_file(path: str) -> RetargetingConfig: ...
    def build(self) -> SeqRetargeting: ...
    @property
    def type_(self) -> str: ...
    @property
    def urdf_path(self) -> str: ...
    @property
    def urdf_dir(self) -> str: ...
    @property
    def add_dummy_free_joint(self) -> bool: ...
    @property
    def target_link_human_indices(self) -> Optional[list[int]]: ...
    @property
    def wrist_link_name(self) -> Optional[str]: ...
    @property
    def target_link_names(self) -> Optional[list[str]]: ...
    @property
    def target_joint_names(self) -> Optional[list[str]]: ...
    @property
    def target_origin_link_names(self) -> Optional[list[str]]: ...
    @property
    def target_task_link_names(self) -> Optional[list[str]]: ...
    @property
    def finger_tip_link_names(self) -> Optional[list[str]]: ...
    @property
    def scaling_factor(self) -> float: ...
    @property
    def normal_delta(self) -> float: ...
    @property
    def huber_delta(self) -> float: ...
    @property
    def project_dist(self) -> float: ...
    @property
    def escape_dist(self) -> float: ...
    @property
    def has_joint_limits(self) -> bool: ...
    @property
    def ignore_mimic_joint(self) -> bool: ...
    @property
    def low_pass_alpha(self) -> float: ...

class SeqRetargeting:
    def retarget(
        self, ref_value: Sequence[float], fixed_qpos: Optional[Sequence[float]] = None
    ) -> list[float]: ...
    def reset(self) -> None: ...
    def set_qpos(self, robot_qpos: Sequence[float]) -> None: ...
    def get_qpos(self, fixed_qpos: Optional[Sequence[float]] = None) -> list[float]: ...
    def link_positions(
        self, robot_qpos: Sequence[float], link_names: Sequence[str]
    ) -> list[tuple[float, float, float]]: ...
    @property
    def joint_names(self) -> list[str]: ...
    @property
    def link_names(self) -> list[str]: ...
    @property
    def fixed_dof(self) -> int: ...
    @property
    def target_dof(self) -> int: ...
```

Create the empty marker file:

```bash
touch crates/dexi-py/python/dexi_rs/py.typed
```

- [ ] **Step 2: Add include entries**

In `crates/dexi-py/pyproject.toml`, extend the `[tool.maturin]` `include` list (after the existing entries):

```toml
  { path = "python/dexi_rs/_native.pyi", format = "wheel" },
  { path = "python/dexi_rs/py.typed", format = "wheel" },
  { path = "python/dexi_rs/_native.pyi", format = "sdist" },
  { path = "python/dexi_rs/py.typed", format = "sdist" },
```

- [ ] **Step 3: Verify the stub ships and the wheel still works**

```bash
just build
unzip -l dist/dexi_rs-*.whl | grep -E "_native.pyi|py.typed"
```

Expected: both files listed in the wheel.

```bash
just check-dist
```

Expected: `release checks passed`.

- [ ] **Step 4: Commit**

```bash
git add crates/dexi-py/python/dexi_rs/_native.pyi crates/dexi-py/python/dexi_rs/py.typed crates/dexi-py/pyproject.toml
git commit -m "🏷️ Ship type stubs for the native module"
```

---

### Task 6: permanent wheel smoke tests

**Files:**
- Modify: `scripts/test_wheel.py` (extend the embedded `code` string)

**Interfaces:**
- Consumes: constructor (Task 3), polymorphic `load_config` (Task 4), built wheel in `dist/` (Task 5's `just build`).
- Produces: regression coverage that runs in `just test-wheel` and `just preflight`.

- [ ] **Step 1: Extend the smoke-test code**

In `scripts/test_wheel.py`, inside the `code = """..."""` block, append after the existing `fourier_qpos` assertions (before `print('wheel smoke test passed')`):

```python
import yaml
ref = [0.03,-0.02,0.08,0.04,0.0,0.09,0.03,0.02,0.085,0.02,0.04,0.07]
kwargs = dict(
    type='vector',
    urdf_path='allegro_hand/allegro_hand_right.urdf',
    target_origin_link_names=['wrist'] * 4,
    target_task_link_names=['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip'],
    target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
    scaling_factor=1.6,
    low_pass_alpha=0.2,
)
qpos_yaml = dexi_rs.load_config(config).build().retarget(ref)
qpos_kwargs = dexi_rs.RetargetingConfig(**kwargs).build().retarget(ref)
assert qpos_kwargs == qpos_yaml, 'python-defined config must match YAML config'
data = yaml.safe_load(config.read_text())
qpos_nested = dexi_rs.load_config(data).build().retarget(ref)
assert qpos_nested == qpos_yaml, 'nested dict config must match YAML config'
qpos_flat = dexi_rs.load_config(data['retargeting']).build().retarget(ref)
assert qpos_flat == qpos_yaml, 'flat dict config must match YAML config'
try:
    dexi_rs.RetargetingConfig(type='vector', urdf_path='x.urdf',
        target_origin_link_names=['wrist', 'wrist'],
        target_task_link_names=['link_15.0_tip'],
        target_link_human_indices=[0, 4])
    raise AssertionError('expected ValueError')
except ValueError:
    pass
try:
    dexi_rs.RetargetingConfig(type='dexpilot', urdf_path='x.urdf', scaling=1.6)
    raise AssertionError('expected TypeError')
except TypeError:
    pass
```

Note: `config` and `dexi_rs` are already defined earlier in the embedded block; `yaml` is available because the venv installs `pyyaml`.

- [ ] **Step 2: Run to verify it passes against the Task 5 wheel**

Run: `just test-wheel`
Expected: `wheel smoke test passed` (plus the existing example runs) with exit 0.

If `dist/` is stale (older than Task 5), run `just build` first.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_wheel.py
git commit -m "✅ Smoke-test Python-defined configs in wheel tests"
```

---

### Task 7: example, docs, changelog

**Files:**
- Create: `examples/python_config.py`
- Modify: `justfile` (examples target), `docs/api.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: constructor (Task 3), `load_config` (Task 4).
- Produces: user-facing docs; example wired into `just examples` and preflight.

- [ ] **Step 1: Write the example**

Create `examples/python_config.py`:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["dexi-rs>=0.4.0", "numpy"]
# ///
"""Define a retargeting config directly in Python — no YAML file needed.

Relative ``urdf_path`` values resolve against the robot assets bundled in
the wheel, so this runs anywhere the package is installed.
"""

from __future__ import annotations

import numpy as np

import dexi_rs


def main() -> None:
    config = dexi_rs.RetargetingConfig(
        type="vector",
        urdf_path="allegro_hand/allegro_hand_right.urdf",
        target_origin_link_names=["wrist"] * 4,
        target_task_link_names=[
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
        ],
        target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
        scaling_factor=1.6,
    )
    print(config)

    retargeting = config.build()
    target_vectors = np.array(
        [
            [0.03, -0.02, 0.08],
            [0.04, 0.00, 0.09],
            [0.03, 0.02, 0.085],
            [0.02, 0.04, 0.07],
        ]
    )
    qpos = retargeting.retarget(target_vectors.flatten().tolist())
    for name, value in zip(retargeting.joint_names, qpos):
        print(f"{name:>28s}: {value: .6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Wire into `just examples`**

In `justfile`, add to the `examples` target after the `quickstart.py` line:

```
    {{PYTHON}} examples/python_config.py
```

- [ ] **Step 3: Run the example**

Run: `just examples`
Expected: exit 0; `python_config.py` prints the config repr and 16 joint values.

- [ ] **Step 4: Update `docs/api.md`**

In the `### RetargetingConfig` section (currently showing only `from_file`), add after the existing `from_file` snippet:

```markdown
Configs can also be defined directly in Python — no YAML file needed.
Relative `urdf_path` values resolve against the robot assets bundled in the
wheel, then the current working directory:

​```python
cfg = dexi_rs.RetargetingConfig(
    type="vector",                                      # position | vector | dexpilot
    urdf_path="allegro_hand/allegro_hand_right.urdf",   # bundled asset
    target_origin_link_names=["wrist"] * 4,
    target_task_link_names=["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"],
    target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
    scaling_factor=1.6,
)
retargeting = cfg.build()
​```

Required fields depend on `type`; dimensions are validated at construction
and raise `ValueError` with the offending field named. Configs are
immutable: all fields are readable attributes, and variants are created by
constructing new objects.
```

(Remove the zero-width characters around the inner code fence when writing the file — they only mark nesting here.)

Update the `load_config` signature line near the top of the file to:

```markdown
dexi_rs.load_config(source: str | Path | dict | RetargetingConfig) -> RetargetingConfig
```

with a sentence noting dicts may be flat constructor kwargs or nested under
a `"retargeting"` key.

- [ ] **Step 5: Update `CHANGELOG.md`**

Add at the top, above `## 0.3.2`:

```markdown
## Unreleased

- Define `RetargetingConfig` directly from Python keyword arguments or dicts.
- Accept YAML paths, dicts, and config objects in `dexi_rs.load_config`.
- Validate config required fields and dimensions on every load path.
- Resolve relative URDF paths against bundled assets, then the working directory.
- Ship type stubs (`_native.pyi`, `py.typed`) for IDE and type-checker support.
```

- [ ] **Step 6: Full verification and commit**

Run: `just fmt && just test && just build && just check-dist && just test-wheel && just examples`
Expected: everything passes.

```bash
git add examples/python_config.py justfile docs/api.md CHANGELOG.md
git commit -m "📝 Document and demo the Python config API"
```

---

## Self-Review Notes

- Spec coverage: validate() (Task 1), cwd fallback + bundled-assets anchor (Tasks 2–3), kwargs constructor + getters + repr + immutability (Task 3), polymorphic load_config with nested dicts (Task 4), .pyi stub (Task 5), wheel smoke incl. bit-for-bit parity + error cases (Task 6), example + api.md + changelog (Task 7). Out-of-scope items (setters, YAML strings, `retargeting_type` alias) appear in no task. ✓
- The YAML-equality assertions rely on `check_release.py`'s resource-drift check keeping repo assets and packaged assets byte-identical; the solver is deterministic, so `==` on qpos lists is sound.
- `type` naming: constructor kwarg `type` (via Rust `r#type`), getter stays `type_` — consistent across Tasks 3, 5, 6, 7.
