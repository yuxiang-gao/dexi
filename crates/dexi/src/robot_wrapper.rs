//! Robot wrapper: FK and Jacobian computation over the URDF tree.

use crate::urdf::{rotation_from_axis_angle, transform_from_translation, JointSpec, UrdfRobot};
use nalgebra::{DMatrix, Matrix4};
use std::collections::HashMap;

/// Robot wrapper providing forward kinematics and Jacobian computation.
pub struct RobotWrapper {
    pub urdf: UrdfRobot,
    /// All non-fixed joints (including mimic) - these are the DOF joints
    pub dof_joints: Vec<JointSpec>,
    /// Map from joint name to index in dof_joints
    pub dof_name_to_idx: HashMap<String, usize>,
    /// Map from link name to index in links list
    pub link_name_to_idx: HashMap<String, usize>,
    /// Map from link name to its parent joint name
    pub joint_parent_map: HashMap<String, String>,
    /// Joint limits (lower, upper) per DOF joint (all non-fixed)
    pub joint_limits: Vec<(f64, f64)>,
    /// Cached link poses after FK
    pub link_poses: Vec<Matrix4<f64>>,
    /// Root link name
    pub root_link: String,
    /// Children map: parent_link -> Vec<joint_index in urdf.joints>
    pub children_map: HashMap<String, Vec<usize>>,
}

impl RobotWrapper {
    /// Create a new RobotWrapper from a URDF file path.
    pub fn from_urdf_path(urdf_path: &str) -> Result<Self, String> {
        let path = std::path::Path::new(urdf_path);
        let urdf = UrdfRobot::from_file(path)?;
        Self::new(urdf)
    }

    /// Create from a parsed UrdfRobot.
    pub fn new(urdf: UrdfRobot) -> Result<Self, String> {
        let root_link = urdf.root_link().ok_or("No root link found")?.to_string();

        // ALL non-fixed joints (including mimic) are DOF joints
        let dof_joints: Vec<JointSpec> = urdf
            .joints
            .iter()
            .filter(|j| j.joint_type != "fixed")
            .cloned()
            .collect();

        let mut dof_name_to_idx = HashMap::new();
        for (i, j) in dof_joints.iter().enumerate() {
            dof_name_to_idx.insert(j.name.clone(), i);
        }

        let mut link_name_to_idx = HashMap::new();
        for (i, l) in urdf.links.iter().enumerate() {
            link_name_to_idx.insert(l.name.clone(), i);
        }

        // Build parent map
        let mut joint_parent_map = HashMap::new();
        for joint in &urdf.joints {
            joint_parent_map.insert(joint.child_link.clone(), joint.name.clone());
        }

        // Joint limits for ALL DOF joints
        let joint_limits: Vec<(f64, f64)> = dof_joints
            .iter()
            .map(|j| {
                if j.mimic.is_some() {
                    // Mimic joints get their limits from the source joint indirectly
                    // Use wide limits for mimic joints
                    (-1e4, 1e4)
                } else if let Some(ref limit) = j.limit {
                    (limit.lower, limit.upper)
                } else {
                    (-1e4, 1e4)
                }
            })
            .collect();

        // Children map: parent_link -> [joint indices in urdf.joints]
        let mut children_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, joint) in urdf.joints.iter().enumerate() {
            children_map
                .entry(joint.parent_link.clone())
                .or_default()
                .push(i);
        }

        let num_links = urdf.links.len();
        let link_poses = vec![Matrix4::identity(); num_links];

        Ok(Self {
            urdf,
            dof_joints,
            dof_name_to_idx,
            link_name_to_idx,
            joint_parent_map,
            joint_limits,
            link_poses,
            root_link,
            children_map,
        })
    }

    /// Number of DOF (all non-fixed joints including mimic)
    pub fn dof(&self) -> usize {
        self.dof_joints.len()
    }

    /// DOF joint names (all non-fixed, including mimic)
    pub fn dof_joint_names(&self) -> Vec<String> {
        self.dof_joints.iter().map(|j| j.name.clone()).collect()
    }

    /// Non-mimic DOF joint names (active joints only)
    pub fn active_joint_names(&self) -> Vec<String> {
        self.dof_joints
            .iter()
            .filter(|j| j.mimic.is_none())
            .map(|j| j.name.clone())
            .collect()
    }

    /// All link names
    pub fn link_names(&self) -> Vec<String> {
        self.urdf.links.iter().map(|l| l.name.clone()).collect()
    }

    /// Get joint index in DOF joints (all non-fixed including mimic)
    pub fn get_joint_index(&self, name: &str) -> Option<usize> {
        self.dof_name_to_idx.get(name).copied()
    }

    /// Get link index by name
    pub fn get_link_index(&self, name: &str) -> Option<usize> {
        self.link_name_to_idx.get(name).copied()
    }

    /// Joint limits as (lower, upper) pairs for all DOF joints
    pub fn joint_limits_array(&self) -> Vec<(f64, f64)> {
        self.joint_limits.clone()
    }

    /// Get the joint's parent and child link names
    pub fn get_joint_parent_child(&self, joint_name: &str) -> Option<(String, String)> {
        self.urdf
            .joints
            .iter()
            .find(|j| j.name == joint_name)
            .map(|j| (j.parent_link.clone(), j.child_link.clone()))
    }

    /// Compute forward kinematics for all links given qpos (all DOF joints).
    /// qpos has length = self.dof(), one value per DOF joint (including mimic).
    /// Mimic joint values in qpos should already be set by the adaptor.
    pub fn compute_forward_kinematics(&mut self, qpos: &[f64]) {
        // Start with root link at identity
        if let Some(&root_idx) = self.link_name_to_idx.get(&self.root_link) {
            self.link_poses[root_idx] = Matrix4::identity();
        }

        // BFS traversal
        let mut queue = vec![self.root_link.clone()];
        while !queue.is_empty() {
            let current_link = queue.remove(0);
            let current_pose = if let Some(&idx) = self.link_name_to_idx.get(&current_link) {
                self.link_poses[idx]
            } else {
                continue;
            };

            if let Some(joint_indices) = self.children_map.get(&current_link) {
                for &ji in joint_indices {
                    let joint = &self.urdf.joints[ji];
                    let child_link = &joint.child_link;

                    // Get the joint value
                    let q_val = if joint.joint_type == "fixed" {
                        0.0
                    } else if let Some(&idx) = self.dof_name_to_idx.get(&joint.name) {
                        if idx < qpos.len() {
                            qpos[idx]
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    // Compute joint transform
                    let joint_transform = self.joint_transform(joint, q_val);
                    let child_pose = current_pose * joint.origin * joint_transform;

                    if let Some(&child_idx) = self.link_name_to_idx.get(child_link) {
                        self.link_poses[child_idx] = child_pose;
                    }

                    queue.push(child_link.clone());
                }
            }
        }
    }

    /// Compute the transform for a single joint given its value
    fn joint_transform(&self, joint: &JointSpec, q: f64) -> Matrix4<f64> {
        match joint.joint_type.as_str() {
            "revolute" | "continuous" => {
                let rot = rotation_from_axis_angle(&joint.axis, q);
                let mut m = Matrix4::identity();
                m.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
                m
            }
            "prismatic" => transform_from_translation(&(&joint.axis * q)),
            _ => Matrix4::identity(), // fixed
        }
    }

    /// Get the cached link pose (4x4 homogeneous transform)
    pub fn get_link_pose(&self, link_id: usize) -> Matrix4<f64> {
        self.link_poses[link_id]
    }

    /// Compute the full (6 x ndof) body Jacobian for a single link via finite differences.
    pub fn compute_single_link_local_jacobian(
        &mut self,
        qpos: &[f64],
        link_id: usize,
    ) -> DMatrix<f64> {
        let eps = 1e-7;
        let ndof = self.dof();
        let mut jac = DMatrix::zeros(6, ndof);

        self.compute_forward_kinematics(qpos);
        let ref_pose = self.link_poses[link_id];

        for i in 0..ndof {
            let mut qpos_pert = qpos.to_vec();
            qpos_pert[i] += eps;
            self.compute_forward_kinematics(&qpos_pert);
            let pert_pose = self.link_poses[link_id];

            // Numerical derivative of position part
            for row in 0..3 {
                jac[(row, i)] = (pert_pose[(row, 3)] - ref_pose[(row, 3)]) / eps;
            }

            // Numerical derivative of orientation part
            let d_r =
                pert_pose.fixed_view::<3, 3>(0, 0) * ref_pose.fixed_view::<3, 3>(0, 0).transpose();
            jac[(3, i)] = (d_r[(2, 1)] - d_r[(1, 2)]) / (2.0 * eps);
            jac[(4, i)] = (d_r[(0, 2)] - d_r[(2, 0)]) / (2.0 * eps);
            jac[(5, i)] = (d_r[(1, 0)] - d_r[(0, 1)]) / (2.0 * eps);
        }

        // Restore original FK
        self.compute_forward_kinematics(qpos);
        jac
    }

    /// Neutral configuration (all zeros)
    pub fn q0(&self) -> Vec<f64> {
        vec![0.0; self.dof()]
    }
}
