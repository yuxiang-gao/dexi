//! # dexi - Pure Rust hand retargeting library
//!
//! This library reimplements the dex-retargeting Python package in pure Rust.
//! It provides URDF parsing, forward kinematics, Jacobian computation, mimic joints,
//! optimizers (Position, Vector, DexPilot), sequential retargeting, and low-pass filtering.

pub mod constants;
pub mod filter;
pub mod kinematics_adaptor;
pub mod optimizer;
pub mod retargeting_config;
pub mod robot_wrapper;
pub mod seq_retarget;
pub mod urdf;

pub use constants::*;
pub use filter::LPFilter;
pub use kinematics_adaptor::MimicJointKinematicAdaptor;
pub use optimizer::{DexPilotOptimizer, Optimizer, PositionOptimizer, VectorOptimizer};
pub use retargeting_config::RetargetingConfig;
pub use robot_wrapper::RobotWrapper;
pub use seq_retarget::SeqRetargeting;
