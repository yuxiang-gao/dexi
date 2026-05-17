//! Sequential retargeting: wraps an optimizer with state tracking and filtering.

use crate::filter::LPFilter;
use crate::optimizer::Optimizer;

/// Sequential retargeting: manages state across retargeting calls.
pub struct SeqRetargeting {
    pub optimizer: Box<dyn Optimizer + Send + Sync>,
    pub has_joint_limits: bool,
    pub joint_limits: Vec<(f64, f64)>,
    pub last_qpos: Vec<f64>,
    pub filter: Option<LPFilter>,
    pub num_retargeting: usize,
}

impl SeqRetargeting {
    /// Create a new SeqRetargeting.
    pub fn new(
        optimizer: Box<dyn Optimizer + Send + Sync>,
        has_joint_limits: bool,
        lp_filter: Option<LPFilter>,
        joint_limits: Vec<(f64, f64)>,
    ) -> Self {
        // Initialize last_qpos to midpoint of joint limits
        let last_qpos: Vec<f64> = joint_limits
            .iter()
            .map(|(lo, hi)| (lo + hi) / 2.0)
            .collect();

        Self {
            optimizer,
            has_joint_limits,
            joint_limits,
            last_qpos,
            filter: lp_filter,
            num_retargeting: 0,
        }
    }

    /// Perform retargeting with a reference value.
    pub fn retarget(&mut self, ref_value: &[f64], fixed_qpos: &[f64]) -> Vec<f64> {
        // Clip last_qpos to limits
        let clipped: Vec<f64> = self
            .last_qpos
            .iter()
            .zip(self.joint_limits.iter())
            .map(|(&q, &(lo, hi))| q.max(lo).min(hi))
            .collect();

        let qpos = self.optimizer.retarget(ref_value, fixed_qpos, &clipped);
        self.num_retargeting += 1;
        self.last_qpos = qpos.clone();

        // Reconstruct full robot qpos
        let total_dof = self.optimizer.dof_joint_names().len();
        let mut robot_qpos = vec![0.0; total_dof];

        for (i, &fi) in self.optimizer.idx_pin2fixed().iter().enumerate() {
            if i < fixed_qpos.len() {
                robot_qpos[fi] = fixed_qpos[i];
            }
        }
        for (i, &ti) in self.optimizer.idx_pin2target().iter().enumerate() {
            robot_qpos[ti] = qpos[i];
        }

        self.optimizer.apply_adaptor_forward(&mut robot_qpos);

        // Apply filter
        if let Some(ref mut filter) = self.filter {
            filter.next(&robot_qpos)
        } else {
            robot_qpos
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.last_qpos = self
            .joint_limits
            .iter()
            .map(|(lo, hi)| (lo + hi) / 2.0)
            .collect();
        self.num_retargeting = 0;
    }

    /// Set qpos directly
    pub fn set_qpos(&mut self, robot_qpos: &[f64]) {
        for (i, &ti) in self.optimizer.idx_pin2target().iter().enumerate() {
            if ti < robot_qpos.len() {
                self.last_qpos[i] = robot_qpos[ti];
            }
        }
    }

    /// Get current qpos
    pub fn get_qpos(&self, fixed_qpos: Option<&[f64]>) -> Vec<f64> {
        let total_dof = self.optimizer.dof_joint_names().len();
        let mut robot_qpos = vec![0.0; total_dof];

        for (i, &ti) in self.optimizer.idx_pin2target().iter().enumerate() {
            robot_qpos[ti] = self.last_qpos[i];
        }
        if let Some(fp) = fixed_qpos {
            for (i, &fi) in self.optimizer.idx_pin2fixed().iter().enumerate() {
                if i < fp.len() {
                    robot_qpos[fi] = fp[i];
                }
            }
        }
        self.optimizer.apply_adaptor_forward(&mut robot_qpos);
        robot_qpos
    }
}
