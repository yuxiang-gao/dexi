//! Optimizers: Position, Vector, and DexPilot retargeting.

use crate::kinematics_adaptor::MimicJointKinematicAdaptor;
use crate::robot_wrapper::RobotWrapper;
use nalgebra::DMatrix;

/// Base optimizer trait
pub trait Optimizer: Send {
    /// Retarget: compute optimal joint positions.
    fn retarget(&mut self, ref_value: &[f64], fixed_qpos: &[f64], last_qpos: &[f64]) -> Vec<f64>;

    /// Get the number of optimization DOF (target joints)
    fn opt_dof(&self) -> usize;

    /// Get target joint indices in the robot DOF
    fn idx_pin2target(&self) -> &[usize];

    /// Get fixed joint indices in the robot DOF
    fn idx_pin2fixed(&self) -> &[usize];

    /// Set joint limits
    fn set_joint_limit(&mut self, limits: &[(f64, f64)]);

    /// Set kinematic adaptor
    fn set_adaptor(&mut self, adaptor: MimicJointKinematicAdaptor);

    /// Apply the configured kinematic adaptor to a full robot qpos, if any.
    fn apply_adaptor_forward(&self, qpos: &mut [f64]);

    /// Get DOF joint names
    fn dof_joint_names(&self) -> Vec<String>;
}

/// Huber loss (SmoothL1)
#[allow(dead_code)]
fn huber_loss(x: f64, beta: f64) -> f64 {
    if x.abs() <= beta {
        0.5 * x * x / beta
    } else {
        x.abs() - 0.5 * beta
    }
}

/// Huber loss derivative
fn huber_loss_grad(x: f64, beta: f64) -> f64 {
    if x.abs() <= beta {
        x / beta
    } else {
        x.signum()
    }
}

/// Common optimizer data
pub struct OptimizerData {
    pub robot: RobotWrapper,
    pub idx_pin2target: Vec<usize>,
    pub idx_pin2fixed: Vec<usize>,
    pub opt_dof: usize,
    pub target_joint_names: Vec<String>,
    pub joint_lower: Vec<f64>,
    pub joint_upper: Vec<f64>,
    pub adaptor: Option<MimicJointKinematicAdaptor>,
}

impl OptimizerData {
    pub fn new(robot: RobotWrapper, target_joint_names: &[String]) -> Self {
        let joint_names = robot.dof_joint_names();
        let mut idx_pin2target = Vec::new();
        for name in target_joint_names {
            let idx = joint_names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("Joint {} not found in robot", name));
            idx_pin2target.push(idx);
        }

        let target_set: std::collections::HashSet<usize> = idx_pin2target.iter().copied().collect();
        let idx_pin2fixed: Vec<usize> = (0..robot.dof())
            .filter(|i| !target_set.contains(i))
            .collect();

        let opt_dof = idx_pin2target.len();
        let joint_lower = vec![-1e4; opt_dof];
        let joint_upper = vec![1e4; opt_dof];

        Self {
            robot,
            idx_pin2target,
            idx_pin2fixed,
            opt_dof,
            target_joint_names: target_joint_names.to_vec(),
            joint_lower,
            joint_upper,
            adaptor: None,
        }
    }
}

/// Position-based retargeting optimizer
pub struct PositionOptimizer {
    pub data: OptimizerData,
    pub target_link_indices: Vec<usize>,
    pub huber_delta: f64,
    pub norm_delta: f64,
}

impl PositionOptimizer {
    pub fn new(
        robot: RobotWrapper,
        target_joint_names: &[String],
        target_link_names: &[String],
        _target_link_human_indices: &[usize],
        huber_delta: f64,
        norm_delta: f64,
    ) -> Self {
        let target_link_indices: Vec<usize> = target_link_names
            .iter()
            .map(|name| {
                robot
                    .get_link_index(name)
                    .unwrap_or_else(|| panic!("Link {} not found", name))
            })
            .collect();

        let data = OptimizerData::new(robot, target_joint_names);
        Self {
            data,
            target_link_indices,
            huber_delta,
            norm_delta,
        }
    }
}

impl Optimizer for PositionOptimizer {
    fn retarget(&mut self, ref_value: &[f64], fixed_qpos: &[f64], last_qpos: &[f64]) -> Vec<f64> {
        let ndof = self.data.robot.dof();
        let opt_dof = self.data.opt_dof;
        let n_links = self.target_link_indices.len();

        // Target positions: flat array of 3D positions
        let target_pos: Vec<[f64; 3]> = (0..n_links)
            .map(|i| [ref_value[i * 3], ref_value[i * 3 + 1], ref_value[i * 3 + 2]])
            .collect();

        let mut x = if last_qpos.len() == opt_dof {
            last_qpos.to_vec()
        } else {
            vec![0.0; opt_dof]
        };

        // Clip to limits
        for i in 0..opt_dof {
            x[i] = x[i]
                .max(self.data.joint_lower[i])
                .min(self.data.joint_upper[i]);
        }

        let lr = 0.1;
        let max_iter = 300;

        for _iter in 0..max_iter {
            let mut qpos = vec![0.0; ndof];
            for (i, &fi) in self.data.idx_pin2fixed.iter().enumerate() {
                if i < fixed_qpos.len() {
                    qpos[fi] = fixed_qpos[i];
                }
            }
            for (i, &ti) in self.data.idx_pin2target.iter().enumerate() {
                qpos[ti] = x[i];
            }

            if let Some(ref adaptor) = self.data.adaptor {
                adaptor.forward_qpos(&mut qpos);
            }

            self.data.robot.compute_forward_kinematics(&qpos);

            let mut grad = vec![0.0; opt_dof];

            for (li, &link_idx) in self.target_link_indices.iter().enumerate() {
                let pose = self.data.robot.get_link_pose(link_idx);
                let body_pos = [pose[(0, 3)], pose[(1, 3)], pose[(2, 3)]];

                let jac_full = self
                    .data
                    .robot
                    .compute_single_link_local_jacobian(&qpos, link_idx);
                let link_rot = pose.fixed_view::<3, 3>(0, 0);

                // World jacobian: R * body_jacobian_position_rows
                let world_jac_3 = &jac_full.rows(0, 3);
                let link_rot_dm = DMatrix::from_iterator(3, 3, link_rot.iter().cloned());
                let kin_jac = &link_rot_dm * world_jac_3;

                let jac_target = if let Some(ref adaptor) = self.data.adaptor {
                    adaptor.backward_jacobian(&kin_jac)
                } else {
                    let mut sel = DMatrix::zeros(3, opt_dof);
                    for (col, &idx) in self.data.idx_pin2target.iter().enumerate() {
                        for row in 0..3 {
                            sel[(row, col)] = kin_jac[(row, idx)];
                        }
                    }
                    sel
                };

                // Gradient of huber w.r.t. body position
                for d in 0..3 {
                    let err = body_pos[d] - target_pos[li][d];
                    let dloss_dpos = huber_loss_grad(err, self.huber_delta);
                    for j in 0..opt_dof {
                        grad[j] += dloss_dpos * jac_target[(d, j)] / (n_links as f64);
                    }
                }
            }

            // Regularization
            for j in 0..opt_dof {
                grad[j] +=
                    2.0 * self.norm_delta * (x[j] - last_qpos.get(j).copied().unwrap_or(0.0));
            }

            let old_x = x.clone();
            for j in 0..opt_dof {
                x[j] -= lr * grad[j];
                x[j] = x[j]
                    .max(self.data.joint_lower[j])
                    .min(self.data.joint_upper[j]);
            }

            let change: f64 = x
                .iter()
                .zip(old_x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if change < 1e-14 {
                break;
            }
        }

        x
    }

    fn opt_dof(&self) -> usize {
        self.data.opt_dof
    }
    fn idx_pin2target(&self) -> &[usize] {
        &self.data.idx_pin2target
    }
    fn idx_pin2fixed(&self) -> &[usize] {
        &self.data.idx_pin2fixed
    }

    fn set_joint_limit(&mut self, limits: &[(f64, f64)]) {
        for (i, &(lo, hi)) in limits.iter().enumerate() {
            if i < self.data.opt_dof {
                self.data.joint_lower[i] = lo;
                self.data.joint_upper[i] = hi;
            }
        }
    }

    fn set_adaptor(&mut self, adaptor: MimicJointKinematicAdaptor) {
        let mimic_set: std::collections::HashSet<usize> =
            adaptor.idx_pin2mimic.iter().copied().collect();
        self.data.idx_pin2fixed = self
            .data
            .idx_pin2fixed
            .iter()
            .filter(|i| !mimic_set.contains(i))
            .copied()
            .collect();
        self.data.adaptor = Some(adaptor);
    }

    fn apply_adaptor_forward(&self, qpos: &mut [f64]) {
        if let Some(ref adaptor) = self.data.adaptor {
            adaptor.forward_qpos(qpos);
        }
    }

    fn dof_joint_names(&self) -> Vec<String> {
        self.data.robot.dof_joint_names()
    }
}

/// Vector-based retargeting optimizer
pub struct VectorOptimizer {
    pub data: OptimizerData,
    pub origin_link_indices: Vec<usize>,
    pub task_link_indices: Vec<usize>,
    pub computed_link_indices: Vec<usize>,
    pub huber_delta: f64,
    pub norm_delta: f64,
    pub scaling: f64,
}

impl VectorOptimizer {
    pub fn new(
        robot: RobotWrapper,
        target_joint_names: &[String],
        target_origin_link_names: &[String],
        target_task_link_names: &[String],
        _target_link_human_indices: &[usize],
        huber_delta: f64,
        norm_delta: f64,
        scaling: f64,
    ) -> Self {
        let mut computed_names: Vec<String> = Vec::new();
        for name in target_origin_link_names
            .iter()
            .chain(target_task_link_names.iter())
        {
            if !computed_names.contains(name) {
                computed_names.push(name.clone());
            }
        }

        let origin_link_indices: Vec<usize> = target_origin_link_names
            .iter()
            .map(|n| computed_names.iter().position(|c| c == n).unwrap())
            .collect();

        let task_link_indices: Vec<usize> = target_task_link_names
            .iter()
            .map(|n| computed_names.iter().position(|c| c == n).unwrap())
            .collect();

        let computed_link_indices: Vec<usize> = computed_names
            .iter()
            .map(|name| {
                robot
                    .get_link_index(name)
                    .unwrap_or_else(|| panic!("Link {} not found", name))
            })
            .collect();

        let data = OptimizerData::new(robot, target_joint_names);

        Self {
            data,
            origin_link_indices,
            task_link_indices,
            computed_link_indices,
            huber_delta,
            norm_delta,
            scaling,
        }
    }
}

impl Optimizer for VectorOptimizer {
    fn retarget(&mut self, ref_value: &[f64], fixed_qpos: &[f64], last_qpos: &[f64]) -> Vec<f64> {
        let ndof = self.data.robot.dof();
        let opt_dof = self.data.opt_dof;
        let n_vecs = self.origin_link_indices.len();

        let target_vecs: Vec<[f64; 3]> = (0..n_vecs)
            .map(|i| {
                [
                    ref_value[i * 3] * self.scaling,
                    ref_value[i * 3 + 1] * self.scaling,
                    ref_value[i * 3 + 2] * self.scaling,
                ]
            })
            .collect();

        let mut x = if last_qpos.len() == opt_dof {
            last_qpos.to_vec()
        } else {
            vec![0.0; opt_dof]
        };
        for i in 0..opt_dof {
            x[i] = x[i]
                .max(self.data.joint_lower[i])
                .min(self.data.joint_upper[i]);
        }

        let lr = 0.1;
        let max_iter = 300;

        for _iter in 0..max_iter {
            let mut qpos = vec![0.0; ndof];
            for (i, &fi) in self.data.idx_pin2fixed.iter().enumerate() {
                if i < fixed_qpos.len() {
                    qpos[fi] = fixed_qpos[i];
                }
            }
            for (i, &ti) in self.data.idx_pin2target.iter().enumerate() {
                qpos[ti] = x[i];
            }

            if let Some(ref adaptor) = self.data.adaptor {
                adaptor.forward_qpos(&mut qpos);
            }

            self.data.robot.compute_forward_kinematics(&qpos);

            let body_pos: Vec<[f64; 3]> = self
                .computed_link_indices
                .iter()
                .map(|&li| {
                    let p = self.data.robot.get_link_pose(li);
                    [p[(0, 3)], p[(1, 3)], p[(2, 3)]]
                })
                .collect();

            let robot_vecs: Vec<[f64; 3]> = (0..n_vecs)
                .map(|i| {
                    let oi = self.origin_link_indices[i];
                    let ti = self.task_link_indices[i];
                    [
                        body_pos[ti][0] - body_pos[oi][0],
                        body_pos[ti][1] - body_pos[oi][1],
                        body_pos[ti][2] - body_pos[oi][2],
                    ]
                })
                .collect();

            let mut vec_grads: Vec<[f64; 3]> = vec![[0.0; 3]; n_vecs];
            for i in 0..n_vecs {
                let dist = ((robot_vecs[i][0] - target_vecs[i][0]).powi(2)
                    + (robot_vecs[i][1] - target_vecs[i][1]).powi(2)
                    + (robot_vecs[i][2] - target_vecs[i][2]).powi(2))
                .sqrt();
                let d = huber_loss_grad(dist, self.huber_delta);
                if dist > 1e-12 {
                    for dd in 0..3 {
                        vec_grads[i][dd] = d * (robot_vecs[i][dd] - target_vecs[i][dd]) / dist;
                    }
                }
            }

            let mut grad = vec![0.0; opt_dof];

            for (ci, &link_idx) in self.computed_link_indices.iter().enumerate() {
                let pose = self.data.robot.get_link_pose(link_idx);
                let link_rot = pose.fixed_view::<3, 3>(0, 0);
                let jac_full = self
                    .data
                    .robot
                    .compute_single_link_local_jacobian(&qpos, link_idx);
                let world_jac_3 = &jac_full.rows(0, 3);
                let link_rot_dm = DMatrix::from_iterator(3, 3, link_rot.iter().cloned());
                let kin_jac = &link_rot_dm * world_jac_3;

                let jac_target = if let Some(ref adaptor) = self.data.adaptor {
                    adaptor.backward_jacobian(&kin_jac)
                } else {
                    let mut sel = DMatrix::zeros(3, opt_dof);
                    for (col, &idx) in self.data.idx_pin2target.iter().enumerate() {
                        for row in 0..3 {
                            sel[(row, col)] = kin_jac[(row, idx)];
                        }
                    }
                    sel
                };

                let mut link_grad = [0.0_f64; 3];
                for i in 0..n_vecs {
                    let mut coeff = 0.0;
                    if self.origin_link_indices[i] == ci {
                        coeff -= 1.0;
                    }
                    if self.task_link_indices[i] == ci {
                        coeff += 1.0;
                    }
                    if coeff != 0.0 {
                        for d in 0..3 {
                            link_grad[d] += coeff * vec_grads[i][d] / (n_vecs as f64);
                        }
                    }
                }

                for j in 0..opt_dof {
                    for d in 0..3 {
                        grad[j] += link_grad[d] * jac_target[(d, j)];
                    }
                }
            }

            for j in 0..opt_dof {
                grad[j] +=
                    2.0 * self.norm_delta * (x[j] - last_qpos.get(j).copied().unwrap_or(0.0));
            }

            let old_x = x.clone();
            for j in 0..opt_dof {
                x[j] -= lr * grad[j];
                x[j] = x[j]
                    .max(self.data.joint_lower[j])
                    .min(self.data.joint_upper[j]);
            }

            let change: f64 = x
                .iter()
                .zip(old_x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if change < 1e-14 {
                break;
            }
        }

        x
    }

    fn opt_dof(&self) -> usize {
        self.data.opt_dof
    }
    fn idx_pin2target(&self) -> &[usize] {
        &self.data.idx_pin2target
    }
    fn idx_pin2fixed(&self) -> &[usize] {
        &self.data.idx_pin2fixed
    }

    fn set_joint_limit(&mut self, limits: &[(f64, f64)]) {
        for (i, &(lo, hi)) in limits.iter().enumerate() {
            if i < self.data.opt_dof {
                self.data.joint_lower[i] = lo;
                self.data.joint_upper[i] = hi;
            }
        }
    }

    fn set_adaptor(&mut self, adaptor: MimicJointKinematicAdaptor) {
        let mimic_set: std::collections::HashSet<usize> =
            adaptor.idx_pin2mimic.iter().copied().collect();
        self.data.idx_pin2fixed = self
            .data
            .idx_pin2fixed
            .iter()
            .filter(|i| !mimic_set.contains(i))
            .copied()
            .collect();
        self.data.adaptor = Some(adaptor);
    }

    fn apply_adaptor_forward(&self, qpos: &mut [f64]) {
        if let Some(ref adaptor) = self.data.adaptor {
            adaptor.forward_qpos(qpos);
        }
    }

    fn dof_joint_names(&self) -> Vec<String> {
        self.data.robot.dof_joint_names()
    }
}

/// DexPilot retargeting optimizer
pub struct DexPilotOptimizer {
    pub data: OptimizerData,
    pub origin_link_indices: Vec<usize>,
    pub task_link_indices: Vec<usize>,
    pub computed_link_indices: Vec<usize>,
    pub huber_delta: f64,
    pub norm_delta: f64,
    pub scaling: f64,
    pub num_fingers: usize,
    pub project_dist: f64,
    pub escape_dist: f64,
    pub eta1: f64,
    pub eta2: f64,
    pub projected: Vec<bool>,
    pub s2_project_index_origin: Vec<usize>,
    pub s2_project_index_task: Vec<usize>,
    pub projected_dist: Vec<f64>,
}

impl DexPilotOptimizer {
    pub fn new(
        robot: RobotWrapper,
        target_joint_names: &[String],
        finger_tip_link_names: &[String],
        wrist_link_name: &str,
        _target_link_human_indices: Option<&[usize]>,
        huber_delta: f64,
        norm_delta: f64,
        project_dist: f64,
        escape_dist: f64,
        eta1: f64,
        eta2: f64,
        scaling: f64,
    ) -> Self {
        let num_fingers = finger_tip_link_names.len();
        assert!(
            num_fingers >= 2 && num_fingers <= 5,
            "DexPilot requires 2-5 fingers"
        );

        let (origin_link_index, task_link_index) = Self::generate_link_indices(num_fingers);
        let link_names: Vec<String> = std::iter::once(wrist_link_name.to_string())
            .chain(finger_tip_link_names.iter().cloned())
            .collect();

        let target_origin_link_names: Vec<String> = origin_link_index
            .iter()
            .map(|&i| link_names[i].clone())
            .collect();
        let target_task_link_names: Vec<String> = task_link_index
            .iter()
            .map(|&i| link_names[i].clone())
            .collect();

        let mut computed_names: Vec<String> = Vec::new();
        for name in target_origin_link_names
            .iter()
            .chain(target_task_link_names.iter())
        {
            if !computed_names.contains(name) {
                computed_names.push(name.clone());
            }
        }

        let origin_link_indices: Vec<usize> = target_origin_link_names
            .iter()
            .map(|n| computed_names.iter().position(|c| c == n).unwrap())
            .collect();
        let task_link_indices: Vec<usize> = target_task_link_names
            .iter()
            .map(|n| computed_names.iter().position(|c| c == n).unwrap())
            .collect();
        let computed_link_indices: Vec<usize> = computed_names
            .iter()
            .map(|name| {
                robot
                    .get_link_index(name)
                    .unwrap_or_else(|| panic!("Link {} not found", name))
            })
            .collect();

        let data = OptimizerData::new(robot, target_joint_names);
        let (projected, s2_project_index_origin, s2_project_index_task, projected_dist) =
            Self::set_dexpilot_cache(num_fingers, eta1, eta2);

        Self {
            data,
            origin_link_indices,
            task_link_indices,
            computed_link_indices,
            huber_delta,
            norm_delta,
            scaling,
            num_fingers,
            project_dist,
            escape_dist,
            eta1,
            eta2,
            projected,
            s2_project_index_origin,
            s2_project_index_task,
            projected_dist,
        }
    }

    fn generate_link_indices(num_fingers: usize) -> (Vec<usize>, Vec<usize>) {
        let mut origin = Vec::new();
        let mut task = Vec::new();
        for i in 1..num_fingers {
            for j in (i + 1)..=num_fingers {
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
        num_fingers: usize,
        eta1: f64,
        eta2: f64,
    ) -> (Vec<bool>, Vec<usize>, Vec<usize>, Vec<f64>) {
        let num_pairs = num_fingers * (num_fingers - 1) / 2;
        let projected = vec![false; num_pairs];
        let mut s2_origin = Vec::new();
        let mut s2_task = Vec::new();
        for i in 0..num_fingers - 2 {
            for j in i + 1..num_fingers - 1 {
                s2_origin.push(j);
                s2_task.push(i);
            }
        }
        let mut dist = vec![eta1; num_fingers - 1];
        dist.extend(vec![eta2; (num_fingers - 1) * (num_fingers - 2) / 2]);
        (projected, s2_origin, s2_task, dist)
    }
}

impl Optimizer for DexPilotOptimizer {
    fn retarget(&mut self, ref_value: &[f64], fixed_qpos: &[f64], last_qpos: &[f64]) -> Vec<f64> {
        let ndof = self.data.robot.dof();
        let opt_dof = self.data.opt_dof;
        let n_vecs = self.origin_link_indices.len();
        let len_proj = self.projected.len();
        let len_s2 = self.s2_project_index_task.len();
        let len_s1 = len_proj - len_s2;

        let target_vecs: Vec<[f64; 3]> = (0..n_vecs)
            .map(|i| [ref_value[i * 3], ref_value[i * 3 + 1], ref_value[i * 3 + 2]])
            .collect();

        // Update projection
        let target_vec_dist: Vec<f64> = target_vecs
            .iter()
            .map(|v| (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt())
            .collect();
        for i in 0..len_s1 {
            if target_vec_dist[i] < self.project_dist {
                self.projected[i] = true;
            }
            if target_vec_dist[i] > self.escape_dist {
                self.projected[i] = false;
            }
        }
        for i in 0..len_s2 {
            let oi = self.s2_project_index_origin[i];
            let ti = self.s2_project_index_task[i];
            self.projected[len_s1 + i] =
                self.projected[oi] && self.projected[ti] && target_vec_dist[len_s1 + i] <= 0.03;
        }

        let weight_proj: Vec<f64> = (0..len_proj)
            .map(|i| {
                if self.projected[i] {
                    if i < len_s1 {
                        200.0
                    } else {
                        400.0
                    }
                } else {
                    1.0
                }
            })
            .collect();
        let mut weight = weight_proj;
        for _ in 0..self.num_fingers {
            weight.push(len_proj as f64 + self.num_fingers as f64);
        }

        let normal_vec: Vec<[f64; 3]> = target_vecs
            .iter()
            .map(|v| {
                [
                    v[0] * self.scaling,
                    v[1] * self.scaling,
                    v[2] * self.scaling,
                ]
            })
            .collect();
        let dir_vec: Vec<[f64; 3]> = (0..len_proj)
            .map(|i| {
                let d = target_vec_dist[i] + 1e-6;
                [
                    target_vecs[i][0] / d,
                    target_vecs[i][1] / d,
                    target_vecs[i][2] / d,
                ]
            })
            .collect();
        let projected_vec: Vec<[f64; 3]> = (0..len_proj)
            .map(|i| {
                [
                    dir_vec[i][0] * self.projected_dist[i],
                    dir_vec[i][1] * self.projected_dist[i],
                    dir_vec[i][2] * self.projected_dist[i],
                ]
            })
            .collect();
        let reference_vec: Vec<[f64; 3]> = (0..n_vecs)
            .map(|i| {
                if i < len_proj {
                    if self.projected[i] {
                        projected_vec[i]
                    } else {
                        normal_vec[i]
                    }
                } else {
                    normal_vec[i]
                }
            })
            .collect();

        let mut x = if last_qpos.len() == opt_dof {
            last_qpos.to_vec()
        } else {
            vec![0.0; opt_dof]
        };
        for i in 0..opt_dof {
            x[i] = x[i]
                .max(self.data.joint_lower[i])
                .min(self.data.joint_upper[i]);
        }

        let lr = 0.05;
        let max_iter = 300;

        for _iter in 0..max_iter {
            let mut qpos = vec![0.0; ndof];
            for (i, &fi) in self.data.idx_pin2fixed.iter().enumerate() {
                if i < fixed_qpos.len() {
                    qpos[fi] = fixed_qpos[i];
                }
            }
            for (i, &ti) in self.data.idx_pin2target.iter().enumerate() {
                qpos[ti] = x[i];
            }
            if let Some(ref adaptor) = self.data.adaptor {
                adaptor.forward_qpos(&mut qpos);
            }
            self.data.robot.compute_forward_kinematics(&qpos);

            let body_pos: Vec<[f64; 3]> = self
                .computed_link_indices
                .iter()
                .map(|&li| {
                    let p = self.data.robot.get_link_pose(li);
                    [p[(0, 3)], p[(1, 3)], p[(2, 3)]]
                })
                .collect();

            let robot_vecs: Vec<[f64; 3]> = (0..n_vecs)
                .map(|i| {
                    let oi = self.origin_link_indices[i];
                    let ti = self.task_link_indices[i];
                    [
                        body_pos[ti][0] - body_pos[oi][0],
                        body_pos[ti][1] - body_pos[oi][1],
                        body_pos[ti][2] - body_pos[oi][2],
                    ]
                })
                .collect();

            let mut vec_grads: Vec<[f64; 3]> = vec![[0.0; 3]; n_vecs];
            for i in 0..n_vecs {
                let dist = ((robot_vecs[i][0] - reference_vec[i][0]).powi(2)
                    + (robot_vecs[i][1] - reference_vec[i][1]).powi(2)
                    + (robot_vecs[i][2] - reference_vec[i][2]).powi(2))
                .sqrt();
                let d = weight[i] * huber_loss_grad(dist, self.huber_delta) / (n_vecs as f64);
                if dist > 1e-12 {
                    for dd in 0..3 {
                        vec_grads[i][dd] = d * (robot_vecs[i][dd] - reference_vec[i][dd]) / dist;
                    }
                }
            }

            let mut grad = vec![0.0; opt_dof];
            for (ci, &link_idx) in self.computed_link_indices.iter().enumerate() {
                let pose = self.data.robot.get_link_pose(link_idx);
                let link_rot = pose.fixed_view::<3, 3>(0, 0);
                let jac_full = self
                    .data
                    .robot
                    .compute_single_link_local_jacobian(&qpos, link_idx);
                let world_jac_3 = &jac_full.rows(0, 3);
                let link_rot_dm = DMatrix::from_iterator(3, 3, link_rot.iter().cloned());
                let kin_jac = &link_rot_dm * world_jac_3;

                let jac_target = if let Some(ref adaptor) = self.data.adaptor {
                    adaptor.backward_jacobian(&kin_jac)
                } else {
                    let mut sel = DMatrix::zeros(3, opt_dof);
                    for (col, &idx) in self.data.idx_pin2target.iter().enumerate() {
                        for row in 0..3 {
                            sel[(row, col)] = kin_jac[(row, idx)];
                        }
                    }
                    sel
                };

                let mut link_grad = [0.0_f64; 3];
                for i in 0..n_vecs {
                    let mut coeff = 0.0;
                    if self.origin_link_indices[i] == ci {
                        coeff -= 1.0;
                    }
                    if self.task_link_indices[i] == ci {
                        coeff += 1.0;
                    }
                    if coeff != 0.0 {
                        for d in 0..3 {
                            link_grad[d] += coeff * vec_grads[i][d];
                        }
                    }
                }
                for j in 0..opt_dof {
                    for d in 0..3 {
                        grad[j] += link_grad[d] * jac_target[(d, j)];
                    }
                }
            }
            for j in 0..opt_dof {
                grad[j] +=
                    2.0 * self.norm_delta * (x[j] - last_qpos.get(j).copied().unwrap_or(0.0));
            }

            let old_x = x.clone();
            for j in 0..opt_dof {
                x[j] -= lr * grad[j];
                x[j] = x[j]
                    .max(self.data.joint_lower[j])
                    .min(self.data.joint_upper[j]);
            }
            let change: f64 = x
                .iter()
                .zip(old_x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if change < 1e-14 {
                break;
            }
        }
        x
    }

    fn opt_dof(&self) -> usize {
        self.data.opt_dof
    }
    fn idx_pin2target(&self) -> &[usize] {
        &self.data.idx_pin2target
    }
    fn idx_pin2fixed(&self) -> &[usize] {
        &self.data.idx_pin2fixed
    }

    fn set_joint_limit(&mut self, limits: &[(f64, f64)]) {
        for (i, &(lo, hi)) in limits.iter().enumerate() {
            if i < self.data.opt_dof {
                self.data.joint_lower[i] = lo;
                self.data.joint_upper[i] = hi;
            }
        }
    }

    fn set_adaptor(&mut self, adaptor: MimicJointKinematicAdaptor) {
        let mimic_set: std::collections::HashSet<usize> =
            adaptor.idx_pin2mimic.iter().copied().collect();
        self.data.idx_pin2fixed = self
            .data
            .idx_pin2fixed
            .iter()
            .filter(|i| !mimic_set.contains(i))
            .copied()
            .collect();
        self.data.adaptor = Some(adaptor);
    }

    fn apply_adaptor_forward(&self, qpos: &mut [f64]) {
        if let Some(ref adaptor) = self.data.adaptor {
            adaptor.forward_qpos(qpos);
        }
    }

    fn dof_joint_names(&self) -> Vec<String> {
        self.data.robot.dof_joint_names()
    }
}
