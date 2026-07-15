//! Test: RetargetingConfig::validate() rejects malformed configs with
//! field-specific messages, on both direct construction and YAML loading.

use dexi::{RetargetingConfig, RetargetingType};

fn base(type_: RetargetingType) -> RetargetingConfig {
    RetargetingConfig {
        type_,
        urdf_path: "allegro_hand/allegro_hand_right.urdf".to_string(),
        ..Default::default()
    }
}

fn valid_vector() -> RetargetingConfig {
    RetargetingConfig {
        target_origin_link_names: Some(vec!["wrist".into(), "wrist".into()]),
        target_task_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        target_link_human_indices: Some(vec![0, 0, 4, 8]),
        ..base(RetargetingType::Vector)
    }
}

#[test]
fn vector_valid_config_passes() {
    valid_vector()
        .validate()
        .expect("valid vector config must pass");
}

#[test]
fn empty_urdf_path_fails() {
    let config = RetargetingConfig {
        urdf_path: String::new(),
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("urdf_path"), "unexpected message: {err}");
}

#[test]
fn vector_missing_task_links_fails() {
    let config = RetargetingConfig {
        target_task_link_names: None,
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("target_task_link_names"),
        "unexpected message: {err}"
    );
}

#[test]
fn vector_origin_task_length_mismatch_fails() {
    let config = RetargetingConfig {
        target_task_link_names: Some(vec!["link_15.0_tip".into()]),
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("target_origin_link_names") && err.contains("2") && err.contains("1"),
        "unexpected message: {err}"
    );
}

#[test]
fn vector_human_indices_dim_mismatch_fails() {
    let config = RetargetingConfig {
        target_link_human_indices: Some(vec![0, 4, 8]),
        ..valid_vector()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("target_link_human_indices"),
        "unexpected message: {err}"
    );
}

#[test]
fn position_valid_config_passes() {
    let config = RetargetingConfig {
        target_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        target_link_human_indices: Some(vec![4, 8]),
        ..base(RetargetingType::Position)
    };
    config.validate().expect("valid position config must pass");
}

#[test]
fn position_indices_length_mismatch_fails() {
    let config = RetargetingConfig {
        target_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        target_link_human_indices: Some(vec![4]),
        ..base(RetargetingType::Position)
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("target_link_human_indices"),
        "unexpected message: {err}"
    );
}

#[test]
fn position_missing_links_fails() {
    let config = RetargetingConfig {
        target_link_human_indices: Some(vec![4, 8]),
        ..base(RetargetingType::Position)
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("target_link_names"),
        "unexpected message: {err}"
    );
}

#[test]
fn dexpilot_valid_config_passes() {
    let config = RetargetingConfig {
        finger_tip_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        wrist_link_name: Some("wrist".into()),
        ..base(RetargetingType::DexPilot)
    };
    config.validate().expect("valid dexpilot config must pass");
}

#[test]
fn dexpilot_missing_wrist_fails() {
    let config = RetargetingConfig {
        finger_tip_link_names: Some(vec!["link_15.0_tip".into()]),
        wrist_link_name: None,
        ..base(RetargetingType::DexPilot)
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("wrist_link_name"), "unexpected message: {err}");
}

#[test]
fn yaml_load_with_dim_mismatch_fails() {
    let yaml = r#"
retargeting:
  type: vector
  urdf_path: allegro_hand/allegro_hand_right.urdf
  target_origin_link_names: [ "wrist", "wrist" ]
  target_task_link_names: [ "link_15.0_tip" ]
  target_link_human_indices: [ [ 0, 0 ], [ 4, 8 ] ]
"#;
    let dir = std::env::temp_dir();
    let path = dir.join("dexi_test_invalid_vector.yml");
    std::fs::write(&path, yaml).expect("write temp yaml");
    let err = RetargetingConfig::load_from_path(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(
        err.contains("target_origin_link_names"),
        "unexpected message: {err}"
    );
}
