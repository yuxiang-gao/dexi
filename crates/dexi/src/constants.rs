//! Constants for dexi retargeting.

use nalgebra::Matrix3;

/// Dummy joint names used when add_dummy_free_joint is enabled
pub const DUMMY_JOINT_NAMES: [&str; 6] = [
    "dummy_x_translation_joint",
    "dummy_y_translation_joint",
    "dummy_z_translation_joint",
    "dummy_x_rotation_joint",
    "dummy_y_rotation_joint",
    "dummy_z_rotation_joint",
];

/// Dummy link names used when add_dummy_free_joint is enabled
pub const DUMMY_LINK_NAMES: [&str; 6] = [
    "dummy_x_translation_link",
    "dummy_y_translation_link",
    "dummy_z_translation_link",
    "dummy_x_rotation_link",
    "dummy_y_rotation_link",
    "dummy_z_rotation_link",
];

/// Operator-to-MANO rotation matrix for the right hand
pub fn operator2mano_right() -> Matrix3<f64> {
    Matrix3::new(0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
}

/// Operator-to-MANO rotation matrix for the left hand
pub fn operator2mano_left() -> Matrix3<f64> {
    Matrix3::new(0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
}

/// Retargeting types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetargetingType {
    Vector,
    Position,
    DexPilot,
}

impl std::str::FromStr for RetargetingType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "vector" => Ok(Self::Vector),
            "position" => Ok(Self::Position),
            "dexpilot" => Ok(Self::DexPilot),
            other => Err(format!("Unknown retargeting type: {}", other)),
        }
    }
}
