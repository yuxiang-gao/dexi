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
    ///
    /// `ref_value` is the flat reference (3 values per target point/vector).
    /// `fixed_qpos` holds the non-optimized joints; pass an empty slice to
    /// hold them at zero.
    pub fn retarget(&mut self, ref_value: &[f64], fixed_qpos: &[f64]) -> Result<Vec<f64>, String> {
        check_len("ref_value", ref_value.len(), self.optimizer.ref_value_len())?;
        check_finite("ref_value", ref_value)?;
        self.check_fixed_qpos(fixed_qpos)?;

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
        let mut robot_qpos = vec![0.0; self.total_dof()];

        for (&fi, &q) in self.optimizer.idx_pin2fixed().iter().zip(fixed_qpos) {
            robot_qpos[fi] = q;
        }
        for (i, &ti) in self.optimizer.idx_pin2target().iter().enumerate() {
            robot_qpos[ti] = qpos[i];
        }

        self.optimizer.apply_adaptor_forward(&mut robot_qpos);

        // Apply filter
        Ok(match self.filter {
            Some(ref mut filter) => filter.next(&robot_qpos),
            None => robot_qpos,
        })
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

    /// Set qpos directly from a full robot qpos (one value per DOF joint).
    pub fn set_qpos(&mut self, robot_qpos: &[f64]) -> Result<(), String> {
        check_len("robot_qpos", robot_qpos.len(), self.total_dof())?;
        for (i, &ti) in self.optimizer.idx_pin2target().iter().enumerate() {
            self.last_qpos[i] = robot_qpos[ti];
        }
        Ok(())
    }

    /// Get current qpos
    pub fn get_qpos(&self, fixed_qpos: Option<&[f64]>) -> Result<Vec<f64>, String> {
        let mut robot_qpos = vec![0.0; self.total_dof()];

        for (i, &ti) in self.optimizer.idx_pin2target().iter().enumerate() {
            robot_qpos[ti] = self.last_qpos[i];
        }
        if let Some(fp) = fixed_qpos {
            self.check_fixed_qpos(fp)?;
            for (&fi, &q) in self.optimizer.idx_pin2fixed().iter().zip(fp) {
                robot_qpos[fi] = q;
            }
        }
        self.optimizer.apply_adaptor_forward(&mut robot_qpos);
        Ok(robot_qpos)
    }

    /// An empty `fixed_qpos` means "hold fixed joints at zero"; otherwise it
    /// must supply one finite value per fixed joint.
    fn check_fixed_qpos(&self, fixed_qpos: &[f64]) -> Result<(), String> {
        if fixed_qpos.is_empty() {
            return Ok(());
        }
        check_len("fixed_qpos", fixed_qpos.len(), self.fixed_dof())?;
        check_finite("fixed_qpos", fixed_qpos)
    }

    /// Number of joints in a full robot qpos.
    fn total_dof(&self) -> usize {
        self.optimizer.dof_joint_names().len()
    }

    /// Number of non-optimized, non-mimic joints supplied via `fixed_qpos`.
    fn fixed_dof(&self) -> usize {
        self.optimizer.idx_pin2fixed().len()
    }
}

fn check_finite(name: &str, values: &[f64]) -> Result<(), String> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(format!("{name} must contain only finite values"))
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<(), String> {
    if got == expected {
        Ok(())
    } else {
        Err(format!("{name} must have {expected} values, got {got}"))
    }
}
