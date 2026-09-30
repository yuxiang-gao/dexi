//! Test: SeqRetargeting rejects malformed runtime inputs with an Err instead
//! of panicking or silently padding.

use dexi::{RetargetingConfig, SeqRetargeting};
use std::path::{Path, PathBuf};

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

fn load(config_name: &str) -> RetargetingConfig {
    let config_path: PathBuf = workspace_root().join("configs").join(config_name);
    let mut config = RetargetingConfig::load_from_path(&config_path).unwrap();
    config.set_default_urdf_dir(&workspace_root().join("assets/robots/hands"));
    config
}

fn build(config_name: &str) -> SeqRetargeting {
    load(config_name).build().unwrap()
}

/// Allegro vector retargeting that only optimizes joints 4-15, leaving the
/// four index-finger joints fixed. No bundled config has fixed joints.
fn build_with_fixed_joints() -> SeqRetargeting {
    let config = RetargetingConfig {
        target_joint_names: Some((4..16).map(|i| format!("joint_{i}.0")).collect()),
        ..load("teleop/allegro_hand_right.yml")
    };
    let retargeting = config.build().unwrap();
    assert_eq!(fixed_dof(&retargeting), 4);
    retargeting
}

fn fixed_dof(retargeting: &SeqRetargeting) -> usize {
    retargeting.optimizer.idx_pin2fixed().len()
}

fn total_dof(retargeting: &SeqRetargeting) -> usize {
    retargeting.optimizer.dof_joint_names().len()
}

#[test]
fn position_ref_value_wrong_length_is_err() {
    // Allegro offline position config: eight target links -> 24 values.
    let mut retargeting = build("offline/allegro_hand_right.yml");
    let fixed = vec![0.0; fixed_dof(&retargeting)];
    let err = retargeting.retarget(&[0.0; 23], &fixed).unwrap_err();
    assert!(
        err.contains("24") && err.contains("23"),
        "unexpected: {err}"
    );
}

#[test]
fn vector_ref_value_wrong_length_is_err() {
    // Allegro teleop vector config: four vectors -> 12 values.
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let err = retargeting.retarget(&[0.0; 11], &[]).unwrap_err();
    assert!(
        err.contains("12") && err.contains("11"),
        "unexpected: {err}"
    );
}

#[test]
fn dexpilot_ref_value_wrong_length_is_err() {
    // Allegro DexPilot: C(4,2) + 4 = 10 vectors -> 30 values.
    let mut retargeting = build("teleop/allegro_hand_right_dexpilot.yml");
    let err = retargeting.retarget(&[0.0; 12], &[]).unwrap_err();
    assert!(
        err.contains("30") && err.contains("12"),
        "unexpected: {err}"
    );
}

#[test]
fn ref_value_non_finite_is_err() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let mut ref_value = vec![0.05; 12];
    ref_value[5] = f64::NAN;
    let err = retargeting.retarget(&ref_value, &[]).unwrap_err();
    assert!(err.contains("finite"), "unexpected: {err}");
}

#[test]
fn valid_ref_value_with_empty_fixed_qpos_is_ok() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let qpos = retargeting.retarget(&[0.05; 12], &[]).unwrap();
    assert_eq!(qpos.len(), total_dof(&retargeting));
}

#[test]
fn fixed_qpos_wrong_length_is_err() {
    let mut retargeting = build_with_fixed_joints();
    let wrong = vec![0.0; fixed_dof(&retargeting) + 1];
    let err = retargeting.retarget(&[0.05; 12], &wrong).unwrap_err();
    assert!(err.contains("fixed_qpos"), "unexpected: {err}");
}

#[test]
fn set_qpos_wrong_length_is_err() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let n = total_dof(&retargeting);
    let err = retargeting.set_qpos(&vec![0.0; n - 1]).unwrap_err();
    assert!(err.contains("robot_qpos"), "unexpected: {err}");
    retargeting.set_qpos(&vec![0.1; n]).unwrap();
    assert_eq!(retargeting.get_qpos(None).unwrap(), vec![0.1; n]);
}

#[test]
fn get_qpos_fixed_qpos_wrong_length_is_err() {
    let retargeting = build_with_fixed_joints();
    let wrong = vec![0.0; fixed_dof(&retargeting) + 1];
    let err = retargeting.get_qpos(Some(&wrong)).unwrap_err();
    assert!(err.contains("fixed_qpos"), "unexpected: {err}");
}

#[test]
fn fixed_qpos_non_finite_is_err() {
    let mut retargeting = build_with_fixed_joints();
    let n = fixed_dof(&retargeting);
    let mut fixed = vec![0.0; n];
    fixed[0] = f64::INFINITY;
    let err = retargeting.retarget(&[0.05; 12], &fixed).unwrap_err();
    assert!(
        err.contains("fixed_qpos") && err.contains("finite"),
        "unexpected: {err}"
    );
}

#[test]
fn get_qpos_empty_fixed_qpos_matches_retarget_semantics() {
    let retargeting = build_with_fixed_joints();
    assert_eq!(
        retargeting.get_qpos(Some(&[])).unwrap(),
        retargeting.get_qpos(None).unwrap()
    );
}
