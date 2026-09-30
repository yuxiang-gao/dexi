//! URDF parsing with roxmltree.

use nalgebra::{Matrix3, Matrix4, Vector3};
use std::path::Path;

/// Mimic joint specification
#[derive(Debug, Clone)]
pub struct MimicSpec {
    pub joint: String,
    pub multiplier: f64,
    pub offset: f64,
}

/// Joint limit specification
#[derive(Debug, Clone)]
pub struct LimitSpec {
    pub lower: f64,
    pub upper: f64,
    pub effort: Option<f64>,
    pub velocity: Option<f64>,
}

/// Joint specification
#[derive(Debug, Clone)]
pub struct JointSpec {
    pub name: String,
    pub joint_type: String, // "revolute", "prismatic", "continuous", "fixed"
    pub parent_link: String,
    pub child_link: String,
    pub origin: Matrix4<f64>, // 4x4 transform
    pub axis: Vector3<f64>,   // joint axis
    pub limit: Option<LimitSpec>,
    pub mimic: Option<MimicSpec>,
}

/// Link specification
#[derive(Debug, Clone)]
pub struct LinkSpec {
    pub name: String,
}

/// Parsed URDF robot
#[derive(Debug, Clone)]
pub struct UrdfRobot {
    pub name: String,
    pub links: Vec<LinkSpec>,
    pub joints: Vec<JointSpec>,
}

fn parse_xyz(s: &str) -> Vector3<f64> {
    let parts: Vec<f64> = s
        .split_whitespace()
        .map(|x| x.parse().unwrap_or(0.0))
        .collect();
    if parts.len() >= 3 {
        Vector3::new(parts[0], parts[1], parts[2])
    } else {
        Vector3::zeros()
    }
}

fn parse_rpy(s: &str) -> (f64, f64, f64) {
    let parts: Vec<f64> = s
        .split_whitespace()
        .map(|x| x.parse().unwrap_or(0.0))
        .collect();
    if parts.len() >= 3 {
        (parts[0], parts[1], parts[2])
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Build a 4x4 transform from xyz and URDF roll-pitch-yaw angles.
///
/// URDF defines fixed-axis RPY as `R = Rz(yaw) * Ry(pitch) * Rx(roll)`.
/// This matches Pinocchio / urdfdom parsing and is the convention used by the
/// Python reference implementation.
pub fn transform_from_xyz_rpy(xyz: Vector3<f64>, rpy: (f64, f64, f64)) -> Matrix4<f64> {
    let (r, p, y) = rpy;
    let cr = r.cos();
    let sr = r.sin();
    let cp = p.cos();
    let sp = p.sin();
    let cy = y.cos();
    let sy = y.sin();

    let mut m = Matrix4::identity();
    m[(0, 0)] = cy * cp;
    m[(0, 1)] = cy * sp * sr - sy * cr;
    m[(0, 2)] = cy * sp * cr + sy * sr;
    m[(1, 0)] = sy * cp;
    m[(1, 1)] = sy * sp * sr + cy * cr;
    m[(1, 2)] = sy * sp * cr - cy * sr;
    m[(2, 0)] = -sp;
    m[(2, 1)] = cp * sr;
    m[(2, 2)] = cr * cp;
    m[(0, 3)] = xyz[0];
    m[(1, 3)] = xyz[1];
    m[(2, 3)] = xyz[2];
    m
}

/// Build rotation matrix from axis-angle: axis * angle
pub fn rotation_from_axis_angle(axis: &Vector3<f64>, angle: f64) -> Matrix3<f64> {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let x = axis[0];
    let y = axis[1];
    let z = axis[2];

    Matrix3::new(
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
    )
}

/// Build translation transform
pub fn transform_from_translation(v: &Vector3<f64>) -> Matrix4<f64> {
    let mut m = Matrix4::identity();
    m[(0, 3)] = v[0];
    m[(1, 3)] = v[1];
    m[(2, 3)] = v[2];
    m
}

impl UrdfRobot {
    /// Parse a URDF file from disk
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read URDF: {}", e))?;
        Self::from_xml(&content)
    }

    /// Parse a URDF from string
    pub fn from_xml(xml: &str) -> Result<Self, String> {
        let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {}", e))?;
        let root = doc.root();
        let robot_node = root
            .children()
            .find(|n| n.has_tag_name("robot"))
            .ok_or("No <robot> element found")?;

        let robot_name = robot_node
            .attribute("name")
            .unwrap_or("unnamed")
            .to_string();

        let mut links = Vec::new();
        let mut joints = Vec::new();

        for child in robot_node.children() {
            if child.has_tag_name("link") {
                let name = child
                    .attribute("name")
                    .ok_or_else(|| "Link missing name".to_string())?
                    .to_string();
                links.push(LinkSpec { name });
            } else if child.has_tag_name("joint") {
                let name = child
                    .attribute("name")
                    .ok_or_else(|| "Joint missing name".to_string())?
                    .to_string();
                let joint_type = child.attribute("type").unwrap_or("fixed").to_string();

                let parent_link = child
                    .children()
                    .find(|n| n.has_tag_name("parent"))
                    .and_then(|n| n.attribute("link"))
                    .ok_or_else(|| format!("Joint {} missing parent link", name))?
                    .to_string();

                let child_link = child
                    .children()
                    .find(|n| n.has_tag_name("child"))
                    .and_then(|n| n.attribute("link"))
                    .ok_or_else(|| format!("Joint {} missing child link", name))?
                    .to_string();

                // Parse origin
                let origin = if let Some(origin_node) =
                    child.children().find(|n| n.has_tag_name("origin"))
                {
                    let xyz = parse_xyz(origin_node.attribute("xyz").unwrap_or("0 0 0"));
                    let rpy = parse_rpy(origin_node.attribute("rpy").unwrap_or("0 0 0"));
                    transform_from_xyz_rpy(xyz, rpy)
                } else {
                    Matrix4::identity()
                };

                // Parse axis
                let axis =
                    if let Some(axis_node) = child.children().find(|n| n.has_tag_name("axis")) {
                        parse_xyz(axis_node.attribute("xyz").unwrap_or("1 0 0"))
                    } else {
                        Vector3::new(1.0, 0.0, 0.0)
                    };

                // Parse limit
                let limit = child
                    .children()
                    .find(|n| n.has_tag_name("limit"))
                    .map(|lim| LimitSpec {
                        lower: lim
                            .attribute("lower")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0),
                        upper: lim
                            .attribute("upper")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0),
                        effort: lim.attribute("effort").and_then(|s| s.parse().ok()),
                        velocity: lim.attribute("velocity").and_then(|s| s.parse().ok()),
                    });

                // Parse mimic
                let mimic = child
                    .children()
                    .find(|n| n.has_tag_name("mimic"))
                    .map(|mim| MimicSpec {
                        joint: mim.attribute("joint").unwrap_or("").to_string(),
                        multiplier: mim
                            .attribute("multiplier")
                            .map(|s| s.parse().unwrap_or(1.0))
                            .unwrap_or(1.0),
                        offset: mim
                            .attribute("offset")
                            .map(|s| s.parse().unwrap_or(0.0))
                            .unwrap_or(0.0),
                    });

                joints.push(JointSpec {
                    name,
                    joint_type,
                    parent_link,
                    child_link,
                    origin,
                    axis,
                    limit,
                    mimic,
                });
            }
        }

        Ok(Self {
            name: robot_name,
            links,
            joints,
        })
    }

    /// Get the root link name (link that is not a child of any joint)
    pub fn root_link(&self) -> Option<&str> {
        let child_links: std::collections::HashSet<&str> =
            self.joints.iter().map(|j| j.child_link.as_str()).collect();

        self.links
            .iter()
            .find(|l| !child_links.contains(l.name.as_str()))
            .map(|l| l.name.as_str())
    }

    /// Get DOF joint names (non-fixed, non-mimic joints)
    pub fn dof_joint_names(&self) -> Vec<String> {
        self.joints
            .iter()
            .filter(|j| j.joint_type != "fixed" && j.mimic.is_none())
            .map(|j| j.name.clone())
            .collect()
    }

    /// Parse all mimic joints, returning (source_names, mimic_names, multipliers, offsets)
    pub fn parse_mimic_joints(&self) -> (bool, Vec<String>, Vec<String>, Vec<f64>, Vec<f64>) {
        let mut source_names = Vec::new();
        let mut mimic_names = Vec::new();
        let mut multipliers = Vec::new();
        let mut offsets = Vec::new();

        for joint in &self.joints {
            if let Some(mimic) = &joint.mimic {
                mimic_names.push(joint.name.clone());
                source_names.push(mimic.joint.clone());
                multipliers.push(mimic.multiplier);
                offsets.push(mimic.offset);
            }
        }

        let has_mimic = !mimic_names.is_empty();
        (has_mimic, source_names, mimic_names, multipliers, offsets)
    }
}
