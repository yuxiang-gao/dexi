//! Optimizers: Position, Vector, and DexPilot retargeting.

use crate::kinematics_adaptor::MimicJointKinematicAdaptor;
use crate::robot_wrapper::RobotWrapper;
use nalgebra::DMatrix;
use slsqp::{minimize, StopTols};
use std::cell::RefCell;

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

fn make_qpos(
    ndof: usize,
    idx_fixed: &[usize],
    fixed_qpos: &[f64],
    idx_target: &[usize],
    x: &[f64],
    adaptor: Option<&MimicJointKinematicAdaptor>,
) -> Vec<f64> {
    let mut qpos = vec![0.0; ndof];
    for (i, &fi) in idx_fixed.iter().enumerate() {
        if i < fixed_qpos.len() {
            qpos[fi] = fixed_qpos[i];
        }
    }
    for (i, &ti) in idx_target.iter().enumerate() {
        qpos[ti] = x[i];
    }
    if let Some(adaptor) = adaptor {
        adaptor.forward_qpos(&mut qpos);
    }
    qpos
}

fn target_jacobian(
    jac_full: &DMatrix<f64>,
    opt_dof: usize,
    idx_target: &[usize],
    adaptor: Option<&MimicJointKinematicAdaptor>,
) -> DMatrix<f64> {
    let kin_jac = if jac_full.nrows() == 3 {
        jac_full.clone()
    } else {
        jac_full.rows(0, 3).into_owned()
    };
    if let Some(adaptor) = adaptor {
        adaptor.backward_jacobian(&kin_jac)
    } else {
        let mut sel = DMatrix::zeros(3, opt_dof);
        for (col, &idx) in idx_target.iter().enumerate() {
            for row in 0..3 {
                sel[(row, col)] = kin_jac[(row, idx)];
            }
        }
        sel
    }
}

fn add_regularization(
    value: &mut f64,
    grad: Option<&mut [f64]>,
    x: &[f64],
    last_qpos: &[f64],
    norm_delta: f64,
) {
    if norm_delta == 0.0 {
        return;
    }
    match grad {
        Some(g) => {
            for (j, gj) in g.iter_mut().enumerate() {
                let delta = x[j] - last_qpos.get(j).copied().unwrap_or(0.0);
                *value += norm_delta * delta * delta;
                *gj += 2.0 * norm_delta * delta;
            }
        }
        None => {
            for (j, &xj) in x.iter().enumerate() {
                let delta = xj - last_qpos.get(j).copied().unwrap_or(0.0);
                *value += norm_delta * delta * delta;
            }
        }
    }
}

fn slsqp_solve<F>(
    xinit: &[f64],
    bounds: &[(f64, f64)],
    ftol_abs: f64,
    maxeval: usize,
    func: F,
) -> Vec<f64>
where
    F: slsqp::Func<()>,
{
    let cons: Vec<&dyn slsqp::Func<()>> = Vec::new();
    let stop_tol = StopTols {
        ftol_abs,
        ..StopTols::default()
    };
    match minimize(func, xinit, bounds, &cons, (), maxeval, Some(stop_tol)) {
        Ok((_status, x, _value)) => x,
        Err((_status, x, _value)) => x,
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

        let idx_fixed = self.data.idx_pin2fixed.clone();
        let idx_target = self.data.idx_pin2target.clone();
        let joint_lower = self.data.joint_lower.clone();
        let joint_upper = self.data.joint_upper.clone();
        let bounds: Vec<(f64, f64)> = joint_lower.into_iter().zip(joint_upper).collect();
        let target_link_indices = self.target_link_indices.clone();
        let adaptor = self.data.adaptor.clone();
        let fixed = fixed_qpos.to_vec();
        let last = last_qpos.to_vec();
        let huber_delta = self.huber_delta;
        let norm_delta = self.norm_delta;
        let robot_cell = RefCell::new(&mut self.data.robot);

        slsqp_solve(&x, &bounds, 1e-5, 200, |x, gradient, _| {
            let mut robot = robot_cell.borrow_mut();
            let qpos = make_qpos(ndof, &idx_fixed, &fixed, &idx_target, x, adaptor.as_ref());
            robot.compute_forward_kinematics(&qpos);
            let mut value = 0.0;
            let mut maybe_grad = gradient;
            if let Some(g) = maybe_grad.as_deref_mut() {
                g.fill(0.0);
            }
            let denom = (n_links * 3) as f64;
            for (li, &link_idx) in target_link_indices.iter().enumerate() {
                let pose = robot.get_link_pose(link_idx);
                let body_pos = [pose[(0, 3)], pose[(1, 3)], pose[(2, 3)]];
                let jac_target = if maybe_grad.is_some() {
                    let jac_full = robot.compute_single_link_position_jacobian_cached(link_idx);
                    Some(target_jacobian(
                        &jac_full,
                        opt_dof,
                        &idx_target,
                        adaptor.as_ref(),
                    ))
                } else {
                    None
                };
                for d in 0..3 {
                    let err = body_pos[d] - target_pos[li][d];
                    value += huber_loss(err, huber_delta) / denom;
                    if let (Some(g), Some(jac)) = (maybe_grad.as_deref_mut(), jac_target.as_ref()) {
                        let dloss_dpos = huber_loss_grad(err, huber_delta) / denom;
                        for j in 0..opt_dof {
                            g[j] += dloss_dpos * jac[(d, j)];
                        }
                    }
                }
            }
            add_regularization(&mut value, maybe_grad, x, &last, norm_delta);
            value
        })
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
                self.data.joint_lower[i] = lo - 1e-3;
                self.data.joint_upper[i] = hi + 1e-3;
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

        let idx_fixed = self.data.idx_pin2fixed.clone();
        let idx_target = self.data.idx_pin2target.clone();
        let bounds: Vec<(f64, f64)> = self
            .data
            .joint_lower
            .iter()
            .copied()
            .zip(self.data.joint_upper.iter().copied())
            .collect();
        let origin_link_indices = self.origin_link_indices.clone();
        let task_link_indices = self.task_link_indices.clone();
        let computed_link_indices = self.computed_link_indices.clone();
        let adaptor = self.data.adaptor.clone();
        let fixed = fixed_qpos.to_vec();
        let last = last_qpos.to_vec();
        let huber_delta = self.huber_delta;
        let norm_delta = self.norm_delta;
        let robot_cell = RefCell::new(&mut self.data.robot);

        slsqp_solve(&x, &bounds, 1e-6, 500, |x, gradient, _| {
            let mut robot = robot_cell.borrow_mut();
            let qpos = make_qpos(ndof, &idx_fixed, &fixed, &idx_target, x, adaptor.as_ref());
            robot.compute_forward_kinematics(&qpos);
            let body_pos: Vec<[f64; 3]> = computed_link_indices
                .iter()
                .map(|&li| {
                    let p = robot.get_link_pose(li);
                    [p[(0, 3)], p[(1, 3)], p[(2, 3)]]
                })
                .collect();
            let robot_vecs: Vec<[f64; 3]> = (0..n_vecs)
                .map(|i| {
                    let oi = origin_link_indices[i];
                    let ti = task_link_indices[i];
                    [
                        body_pos[ti][0] - body_pos[oi][0],
                        body_pos[ti][1] - body_pos[oi][1],
                        body_pos[ti][2] - body_pos[oi][2],
                    ]
                })
                .collect();
            let mut value = 0.0;
            let mut maybe_grad = gradient;
            if let Some(g) = maybe_grad.as_deref_mut() {
                g.fill(0.0);
            }
            let mut vec_grads = vec![[0.0_f64; 3]; n_vecs];
            for i in 0..n_vecs {
                let diff = [
                    robot_vecs[i][0] - target_vecs[i][0],
                    robot_vecs[i][1] - target_vecs[i][1],
                    robot_vecs[i][2] - target_vecs[i][2],
                ];
                let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
                value += huber_loss(dist, huber_delta) / (n_vecs as f64);
                if maybe_grad.is_some() && dist > 1e-12 {
                    let dloss = huber_loss_grad(dist, huber_delta) / (n_vecs as f64);
                    for dd in 0..3 {
                        vec_grads[i][dd] = dloss * diff[dd] / dist;
                    }
                }
            }
            if let Some(g) = maybe_grad.as_deref_mut() {
                for (ci, &link_idx) in computed_link_indices.iter().enumerate() {
                    let jac_full = robot.compute_single_link_position_jacobian_cached(link_idx);
                    let jac_target =
                        target_jacobian(&jac_full, opt_dof, &idx_target, adaptor.as_ref());
                    let mut link_grad = [0.0_f64; 3];
                    for i in 0..n_vecs {
                        let mut coeff = 0.0;
                        if origin_link_indices[i] == ci {
                            coeff -= 1.0;
                        }
                        if task_link_indices[i] == ci {
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
                            g[j] += link_grad[d] * jac_target[(d, j)];
                        }
                    }
                }
            }
            add_regularization(&mut value, maybe_grad, x, &last, norm_delta);
            value
        })
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
                self.data.joint_lower[i] = lo - 1e-3;
                self.data.joint_upper[i] = hi + 1e-3;
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

        let idx_fixed = self.data.idx_pin2fixed.clone();
        let idx_target = self.data.idx_pin2target.clone();
        let bounds: Vec<(f64, f64)> = self
            .data
            .joint_lower
            .iter()
            .copied()
            .zip(self.data.joint_upper.iter().copied())
            .collect();
        let origin_link_indices = self.origin_link_indices.clone();
        let task_link_indices = self.task_link_indices.clone();
        let computed_link_indices = self.computed_link_indices.clone();
        let adaptor = self.data.adaptor.clone();
        let fixed = fixed_qpos.to_vec();
        let last = last_qpos.to_vec();
        let huber_delta = self.huber_delta;
        let norm_delta = self.norm_delta;
        let robot_cell = RefCell::new(&mut self.data.robot);

        slsqp_solve(&x, &bounds, 1e-6, 500, |x, gradient, _| {
            let mut robot = robot_cell.borrow_mut();
            let qpos = make_qpos(ndof, &idx_fixed, &fixed, &idx_target, x, adaptor.as_ref());
            robot.compute_forward_kinematics(&qpos);
            let body_pos: Vec<[f64; 3]> = computed_link_indices
                .iter()
                .map(|&li| {
                    let p = robot.get_link_pose(li);
                    [p[(0, 3)], p[(1, 3)], p[(2, 3)]]
                })
                .collect();
            let robot_vecs: Vec<[f64; 3]> = (0..n_vecs)
                .map(|i| {
                    let oi = origin_link_indices[i];
                    let ti = task_link_indices[i];
                    [
                        body_pos[ti][0] - body_pos[oi][0],
                        body_pos[ti][1] - body_pos[oi][1],
                        body_pos[ti][2] - body_pos[oi][2],
                    ]
                })
                .collect();
            let mut value = 0.0;
            let mut maybe_grad = gradient;
            if let Some(g) = maybe_grad.as_deref_mut() {
                g.fill(0.0);
            }
            let mut vec_grads = vec![[0.0_f64; 3]; n_vecs];
            for i in 0..n_vecs {
                let diff = [
                    robot_vecs[i][0] - reference_vec[i][0],
                    robot_vecs[i][1] - reference_vec[i][1],
                    robot_vecs[i][2] - reference_vec[i][2],
                ];
                let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
                value += weight[i] * huber_loss(dist, huber_delta) / (n_vecs as f64);
                if maybe_grad.is_some() && dist > 1e-12 {
                    let dloss = weight[i] * huber_loss_grad(dist, huber_delta) / (n_vecs as f64);
                    for dd in 0..3 {
                        vec_grads[i][dd] = dloss * diff[dd] / dist;
                    }
                }
            }
            if let Some(g) = maybe_grad.as_deref_mut() {
                for (ci, &link_idx) in computed_link_indices.iter().enumerate() {
                    let jac_full = robot.compute_single_link_position_jacobian_cached(link_idx);
                    let jac_target =
                        target_jacobian(&jac_full, opt_dof, &idx_target, adaptor.as_ref());
                    let mut link_grad = [0.0_f64; 3];
                    for i in 0..n_vecs {
                        let mut coeff = 0.0;
                        if origin_link_indices[i] == ci {
                            coeff -= 1.0;
                        }
                        if task_link_indices[i] == ci {
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
                            g[j] += link_grad[d] * jac_target[(d, j)];
                        }
                    }
                }
            }
            add_regularization(&mut value, maybe_grad, x, &last, norm_delta);
            value
        })
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
                self.data.joint_lower[i] = lo - 1e-3;
                self.data.joint_upper[i] = hi + 1e-3;
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
