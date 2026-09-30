//! Shared helpers for integration tests.
#![allow(dead_code)] // each test crate uses a different subset

use dexi::{RetargetingConfig, SeqRetargeting};
use std::path::{Path, PathBuf};

pub fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

/// Load a bundled config (e.g. "teleop/allegro_hand_right.yml") with URDFs
/// resolved against the repository's assets.
pub fn load(config_name: &str) -> RetargetingConfig {
    let path = workspace_root().join("configs").join(config_name);
    let mut config =
        RetargetingConfig::load_from_path(&path).unwrap_or_else(|e| panic!("{config_name}: {e}"));
    config.set_default_urdf_dir(&workspace_root().join("assets/robots/hands"));
    config
}

pub fn build(config_name: &str) -> SeqRetargeting {
    load(config_name)
        .build()
        .unwrap_or_else(|e| panic!("{config_name}: {e}"))
}

/// All bundled YAML configs as "offline/..." / "teleop/..." names, sorted.
pub fn bundled_configs() -> Vec<String> {
    let mut names = Vec::new();
    for kind in ["offline", "teleop"] {
        let dir: PathBuf = workspace_root().join("configs").join(kind);
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "yml") {
                let file = path.file_name().unwrap().to_str().unwrap();
                names.push(format!("{kind}/{file}"));
            }
        }
    }
    names.sort();
    names
}

pub fn fixed_dof(retargeting: &SeqRetargeting) -> usize {
    retargeting.optimizer.idx_pin2fixed().len()
}

pub fn total_dof(retargeting: &SeqRetargeting) -> usize {
    retargeting.optimizer.dof_joint_names().len()
}

/// Deterministic xorshift64 so inputs are reproducible without a rand dep.
pub struct XorShift(pub u64);

impl XorShift {
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `len` values uniform in [-span/2, span/2).
    pub fn vec(&mut self, len: usize, span: f64) -> Vec<f64> {
        (0..len).map(|_| (self.next_f64() - 0.5) * span).collect()
    }
}
