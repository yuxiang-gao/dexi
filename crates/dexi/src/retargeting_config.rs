//! Retargeting configuration: YAML parsing and config types.

use crate::constants::{RetargetingType, DUMMY_JOINT_NAMES, DUMMY_LINK_NAMES};
use crate::filter::LPFilter;
use crate::kinematics_adaptor::MimicJointKinematicAdaptor;
use crate::optimizer::{DexPilotOptimizer, Optimizer, PositionOptimizer, VectorOptimizer};
use crate::robot_wrapper::RobotWrapper;
use crate::seq_retarget::SeqRetargeting;
use crate::urdf::{JointSpec, LimitSpec, LinkSpec, UrdfRobot};
use nalgebra::{Matrix4, Vector3};
use std::path::{Path, PathBuf};

/// Retargeting configuration
#[derive(Debug, Clone)]
pub struct RetargetingConfig {
    pub type_: RetargetingType,
    pub urdf_path: String,
    pub add_dummy_free_joint: bool,
    pub target_link_human_indices: Option<Vec<usize>>,
    pub wrist_link_name: Option<String>,
    pub target_link_names: Option<Vec<String>>,
    pub target_joint_names: Option<Vec<String>>,
    pub target_origin_link_names: Option<Vec<String>>,
    pub target_task_link_names: Option<Vec<String>>,
    pub finger_tip_link_names: Option<Vec<String>>,
    pub scaling_factor: f64,
    pub normal_delta: f64,
    pub huber_delta: f64,
    pub project_dist: f64,
    pub escape_dist: f64,
    pub has_joint_limits: bool,
    pub ignore_mimic_joint: bool,
    pub low_pass_alpha: f64,
    pub default_urdf_dir: PathBuf,
}

impl Default for RetargetingConfig {
    fn default() -> Self {
        Self {
            type_: RetargetingType::Position,
            urdf_path: String::new(),
            add_dummy_free_joint: false,
            target_link_human_indices: None,
            wrist_link_name: None,
            target_link_names: None,
            target_joint_names: None,
            target_origin_link_names: None,
            target_task_link_names: None,
            finger_tip_link_names: None,
            scaling_factor: 1.0,
            normal_delta: 4e-3,
            huber_delta: 2e-2,
            project_dist: 0.03,
            escape_dist: 0.05,
            has_joint_limits: true,
            ignore_mimic_joint: false,
            low_pass_alpha: 0.1,
            default_urdf_dir: PathBuf::from("./"),
        }
    }
}

impl RetargetingConfig {
    /// Load from a YAML file path
    pub fn load_from_file(config_path: &str) -> Result<Self, String> {
        let path = Path::new(config_path);
        Self::load_from_path(path)
    }

    /// Load from a Path
    pub fn load_from_path(config_path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        let yaml_config: serde_yaml::Value =
            serde_yaml::from_str(&content).map_err(|e| format!("YAML parse error: {}", e))?;

        let cfg = yaml_config
            .get("retargeting")
            .ok_or("Missing 'retargeting' key in config")?;

        let config_dir = config_path.parent().unwrap_or(Path::new("."));
        Self::from_value(cfg, config_dir)
    }

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

    /// Parse from a serde_yaml::Value
    fn from_value(cfg: &serde_yaml::Value, config_dir: &Path) -> Result<Self, String> {
        let type_str = cfg
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'type' in retargeting config")?;
        let type_ = RetargetingType::from_str(type_str)?;

        let urdf_path = cfg
            .get("urdf_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let add_dummy_free_joint = cfg
            .get("add_dummy_free_joint")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let target_link_human_indices = cfg
            .get("target_link_human_indices")
            .and_then(|v| parse_indices(v));

        let wrist_link_name = cfg
            .get("wrist_link_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let target_link_names = cfg
            .get("target_link_names")
            .and_then(|v| parse_string_list(v));

        // target_joint_names can be null, which means "all dof joints"
        let target_joint_names = cfg.get("target_joint_names").and_then(|v| {
            if v.is_null() {
                None
            } else {
                parse_string_list(v)
            }
        });

        let target_origin_link_names = cfg
            .get("target_origin_link_names")
            .and_then(|v| parse_string_list(v));

        let target_task_link_names = cfg
            .get("target_task_link_names")
            .and_then(|v| parse_string_list(v));

        let finger_tip_link_names = cfg
            .get("finger_tip_link_names")
            .and_then(|v| parse_string_list(v));

        let scaling_factor = cfg
            .get("scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let normal_delta = cfg
            .get("normal_delta")
            .and_then(|v| v.as_f64())
            .unwrap_or(4e-3);

        let huber_delta = cfg
            .get("huber_delta")
            .and_then(|v| v.as_f64())
            .unwrap_or(2e-2);

        let project_dist = cfg
            .get("project_dist")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.03);

        let escape_dist = cfg
            .get("escape_dist")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.05);

        let has_joint_limits = cfg
            .get("has_joint_limits")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let ignore_mimic_joint = cfg
            .get("ignore_mimic_joint")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let low_pass_alpha = cfg
            .get("low_pass_alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

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
    }

    /// Set the default URDF directory
    pub fn set_default_urdf_dir(&mut self, dir: &Path) {
        self.default_urdf_dir = dir.to_path_buf();
    }

    /// Resolve the URDF path.
    ///
    /// Relative paths are first resolved relative to the config file's
    /// directory. If that fails, each ancestor directory is checked for the
    /// project/package layout `assets/robots/hands/<urdf_path>`.
    pub fn resolve_urdf_path(&self) -> Result<PathBuf, String> {
        let path = Path::new(&self.urdf_path);
        if path.is_absolute() {
            if path.exists() {
                Ok(path.to_path_buf())
            } else {
                Err(format!("URDF path {} does not exist", path.display()))
            }
        } else {
            let mut attempted = Vec::new();

            let config_relative = self.default_urdf_dir.join(path);
            attempted.push(config_relative.clone());
            if config_relative.exists() {
                return Ok(config_relative);
            }

            for ancestor in self.default_urdf_dir.ancestors() {
                let candidate = ancestor.join("assets/robots/hands").join(path);
                attempted.push(candidate.clone());
                if candidate.exists() {
                    return Ok(candidate);
                }
            }

            if let Ok(cwd) = std::env::current_dir() {
                let candidate = cwd.join(path);
                attempted.push(candidate.clone());
                if candidate.exists() {
                    return Ok(candidate);
                }
            }

            let attempted = attempted
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Err(format!(
                "URDF path {} does not exist (tried {})",
                path.display(),
                attempted
            ))
        }
    }

    /// Build a SeqRetargeting from this config.
    /// The config is consumed since target_joint_names may be modified.
    pub fn build(mut self) -> Result<SeqRetargeting, String> {
        self.validate()?;
        let urdf_path = self.resolve_urdf_path()?;

        // Load URDF
        let mut urdf = UrdfRobot::from_file(&urdf_path)?;

        // Add dummy free joints if needed
        if self.add_dummy_free_joint {
            add_dummy_free_joints(&mut urdf);
        }

        // Determine target joint names
        if self.add_dummy_free_joint {
            if let Some(ref mut names) = self.target_joint_names {
                let dummy_names: Vec<String> =
                    DUMMY_JOINT_NAMES.iter().map(|s| s.to_string()).collect();
                let mut new_names = dummy_names;
                new_names.append(names);
                *names = new_names;
            }
        }

        // Create robot wrapper
        let robot = RobotWrapper::new(urdf)?;

        let joint_names = if let Some(ref names) = self.target_joint_names {
            names.clone()
        } else {
            robot.dof_joint_names()
        };

        // Parse mimic joints from the URDF
        let (has_mimic, source_names, mimic_names, multipliers, offsets) =
            robot.urdf.parse_mimic_joints();

        // Get joint limits before moving robot
        let robot_joint_limits = robot.joint_limits_array();

        // Build optimizer - move robot into it
        let mut optimizer: Box<dyn Optimizer + Send + Sync> = match self.type_ {
            RetargetingType::Position => {
                let link_names = self
                    .target_link_names
                    .as_ref()
                    .ok_or("Position retargeting requires target_link_names")?;
                let human_indices = self
                    .target_link_human_indices
                    .as_ref()
                    .ok_or("Position retargeting requires target_link_human_indices")?;

                Box::new(PositionOptimizer::new(
                    robot,
                    &joint_names,
                    link_names,
                    human_indices,
                    self.huber_delta,
                    self.normal_delta,
                ))
            }
            RetargetingType::Vector => {
                let origin_names = self
                    .target_origin_link_names
                    .as_ref()
                    .ok_or("Vector retargeting requires target_origin_link_names")?;
                let task_names = self
                    .target_task_link_names
                    .as_ref()
                    .ok_or("Vector retargeting requires target_task_link_names")?;
                let human_indices = self
                    .target_link_human_indices
                    .as_ref()
                    .ok_or("Vector retargeting requires target_link_human_indices")?;

                Box::new(VectorOptimizer::new(
                    robot,
                    &joint_names,
                    origin_names,
                    task_names,
                    human_indices,
                    self.huber_delta,
                    self.normal_delta,
                    self.scaling_factor,
                ))
            }
            RetargetingType::DexPilot => {
                let finger_tips = self
                    .finger_tip_link_names
                    .as_ref()
                    .ok_or("DexPilot requires finger_tip_link_names")?;
                let wrist_name = self
                    .wrist_link_name
                    .as_ref()
                    .ok_or("DexPilot requires wrist_link_name")?;

                Box::new(DexPilotOptimizer::new(
                    robot,
                    &joint_names,
                    finger_tips,
                    wrist_name,
                    self.target_link_human_indices.as_deref(),
                    0.03, // huber_delta for dexpilot
                    self.normal_delta,
                    self.project_dist,
                    self.escape_dist,
                    1e-4, // eta1
                    3e-2, // eta2
                    self.scaling_factor,
                ))
            }
        };

        // Set mimic adaptor
        if has_mimic && !self.ignore_mimic_joint {
            // We need a temporary robot to create the adaptor. Since the robot was moved into
            // the optimizer, we need a different approach. Let's create the adaptor using
            // joint name mappings directly.
            // The MimicJointKinematicAdaptor only needs index mappings, which we can compute
            // from the joint_names, source_names, mimic_names.
            // We'll compute the indices manually.
            let dof_joint_names = optimizer.dof_joint_names();

            let idx_pin2target: Vec<usize> = joint_names
                .iter()
                .map(|n| {
                    dof_joint_names
                        .iter()
                        .position(|d| d == n)
                        .ok_or_else(|| format!("Joint {} not found in robot DOF joints", n))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let idx_pin2source: Vec<usize> = source_names
                .iter()
                .map(|n| {
                    dof_joint_names
                        .iter()
                        .position(|d| d == n)
                        .ok_or_else(|| format!("Source joint {} not found", n))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let idx_pin2mimic: Vec<usize> = mimic_names
                .iter()
                .map(|n| {
                    dof_joint_names
                        .iter()
                        .position(|d| d == n)
                        .ok_or_else(|| format!("Mimic joint {} not found", n))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let idx_target2source: Vec<usize> = source_names
                .iter()
                .map(|n| {
                    joint_names
                        .iter()
                        .position(|t| t == n)
                        .ok_or_else(|| format!("Source joint {} not in target joints", n))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let adaptor = MimicJointKinematicAdaptor::from_indices(
                idx_pin2target,
                idx_pin2source,
                idx_pin2mimic,
                idx_target2source,
                &multipliers,
                &offsets,
            );
            optimizer.set_adaptor(adaptor);
        }

        // Set joint limits
        let target_limits: Vec<(f64, f64)> = if self.has_joint_limits {
            optimizer
                .idx_pin2target()
                .iter()
                .map(|&i| robot_joint_limits[i])
                .collect()
        } else {
            vec![(-1e4, 1e4); optimizer.opt_dof()]
        };
        optimizer.set_joint_limit(&target_limits);

        // LP filter
        let lp_filter = if (0.0..=1.0).contains(&self.low_pass_alpha) {
            Some(LPFilter::new(self.low_pass_alpha))
        } else {
            None
        };

        Ok(SeqRetargeting::new(
            optimizer,
            self.has_joint_limits,
            lp_filter,
            target_limits,
        ))
    }
}

/// Helper to parse indices from YAML value
fn parse_indices(v: &serde_yaml::Value) -> Option<Vec<usize>> {
    if v.is_sequence() {
        let mut result = Vec::new();
        // Check if it's a 2D array
        if let Some(first) = v.as_sequence().and_then(|s| s.first()) {
            if first.is_sequence() {
                // 2D array - flatten with row-major order
                for row in v.as_sequence()? {
                    for item in row.as_sequence()? {
                        result.push(item.as_i64()? as usize);
                    }
                }
                return Some(result);
            }
        }
        // 1D array
        for item in v.as_sequence()? {
            result.push(item.as_i64()? as usize);
        }
        Some(result)
    } else {
        None
    }
}

/// Helper to parse string list from YAML value
fn parse_string_list(v: &serde_yaml::Value) -> Option<Vec<String>> {
    v.as_sequence().map(|seq| {
        seq.iter()
            .filter_map(|item| item.as_str().map(|s| s.to_string()))
            .collect()
    })
}

/// Add dummy free joints to the URDF
pub fn add_dummy_free_joints(urdf: &mut UrdfRobot) {
    let root_link = match urdf.root_link() {
        Some(r) => r.to_string(),
        None => return,
    };

    let translation_range = (-5.0, 5.0);
    let rotation_range = (-2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI);
    let joint_types = [
        "prismatic",
        "prismatic",
        "prismatic",
        "revolute",
        "revolute",
        "revolute",
    ];
    let joint_limits_vals = [
        translation_range,
        translation_range,
        translation_range,
        rotation_range,
        rotation_range,
        rotation_range,
    ];

    let link_names: Vec<String> = DUMMY_LINK_NAMES.iter().map(|s| s.to_string()).collect();
    let joint_names: Vec<String> = DUMMY_JOINT_NAMES.iter().map(|s| s.to_string()).collect();

    let mut new_links = Vec::new();
    let mut new_joints = Vec::new();

    for i in 0..6 {
        new_links.push(LinkSpec {
            name: link_names[i].clone(),
        });

        let mut axis = Vector3::zeros();
        axis[i % 3] = 1.0;

        let child_name = if i < 5 {
            link_names[i + 1].clone()
        } else {
            root_link.clone()
        };

        new_joints.push(JointSpec {
            name: joint_names[i].clone(),
            joint_type: joint_types[i].to_string(),
            parent_link: link_names[i].clone(),
            child_link: child_name,
            origin: Matrix4::identity(),
            axis,
            limit: Some(LimitSpec {
                lower: joint_limits_vals[i].0,
                upper: joint_limits_vals[i].1,
                effort: Some(10.0),
                velocity: Some(3.14),
            }),
            mimic: None,
        });
    }

    urdf.links = new_links.into_iter().chain(urdf.links.drain(..)).collect();
    urdf.joints = new_joints
        .into_iter()
        .chain(urdf.joints.drain(..))
        .collect();

    // Rebuild maps
    urdf.link_map.clear();
    for (i, link) in urdf.links.iter().enumerate() {
        urdf.link_map.insert(link.name.clone(), i);
    }
    urdf.joint_map.clear();
    for (i, joint) in urdf.joints.iter().enumerate() {
        urdf.joint_map.insert(joint.name.clone(), i);
    }
}
