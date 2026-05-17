//! Kinematics adaptor: mimic joint handling.

/// Mimic joint kinematic adaptor.
pub struct MimicJointKinematicAdaptor {
    /// Joint indices of source joints in the full DOF list
    pub idx_pin2source: Vec<usize>,
    /// Joint indices of mimic joints in the full DOF list
    pub idx_pin2mimic: Vec<usize>,
    /// Joint indices of source joints in the target joint list
    pub idx_target2source: Vec<usize>,
    /// Joint indices of target joints in the DOF list
    pub idx_pin2target: Vec<usize>,
    /// Multipliers for each mimic joint
    pub multipliers: Vec<f64>,
    /// Offsets for each mimic joint
    pub offsets: Vec<f64>,
}

impl MimicJointKinematicAdaptor {
    /// Create from pre-computed index arrays.
    pub fn from_indices(
        idx_pin2target: Vec<usize>,
        idx_pin2source: Vec<usize>,
        idx_pin2mimic: Vec<usize>,
        idx_target2source: Vec<usize>,
        multipliers: &[f64],
        offsets: &[f64],
    ) -> Self {
        Self {
            idx_pin2source,
            idx_pin2mimic,
            idx_target2source,
            idx_pin2target,
            multipliers: multipliers.to_vec(),
            offsets: offsets.to_vec(),
        }
    }

    /// Adapt qpos for mimic joints. Modifies in place.
    pub fn forward_qpos(&self, qpos: &mut [f64]) {
        for i in 0..self.idx_pin2mimic.len() {
            let src_idx = self.idx_pin2source[i];
            let mimic_idx = self.idx_pin2mimic[i];
            qpos[mimic_idx] = qpos[src_idx] * self.multipliers[i] + self.offsets[i];
        }
    }

    /// Adapt jacobian for mimic joints.
    pub fn backward_jacobian(&self, jacobian: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
        let ncols = self.idx_pin2target.len();
        let nrows = jacobian.nrows();
        let mut target_jac = nalgebra::DMatrix::zeros(nrows, ncols);

        // Select columns for target joints
        for (col, &idx) in self.idx_pin2target.iter().enumerate() {
            for row in 0..nrows {
                target_jac[(row, col)] = jacobian[(row, idx)];
            }
        }

        // Add mimic joint contributions to source columns
        for i in 0..self.idx_pin2mimic.len() {
            let mimic_col = self.idx_pin2mimic[i];
            let target_col = self.idx_target2source[i];
            let mult = self.multipliers[i];
            for row in 0..nrows {
                target_jac[(row, target_col)] += jacobian[(row, mimic_col)] * mult;
            }
        }

        target_jac
    }
}
