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
        finger_tip_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
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

fn build_err(config: RetargetingConfig) -> String {
    match config.build() {
        Ok(_) => panic!("expected build to fail"),
        Err(err) => err,
    }
}

fn allegro_urdf_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/robots/hands/allegro_hand/allegro_hand_right.urdf")
        .display()
        .to_string()
}

fn valid_dexpilot() -> RetargetingConfig {
    RetargetingConfig {
        finger_tip_link_names: Some(vec![
            "link_15.0_tip".into(),
            "link_3.0_tip".into(),
            "link_7.0_tip".into(),
            "link_11.0_tip".into(),
        ]),
        wrist_link_name: Some("wrist".into()),
        ..base(RetargetingType::DexPilot)
    }
}

#[test]
fn dexpilot_single_fingertip_fails_validation() {
    let config = RetargetingConfig {
        finger_tip_link_names: Some(vec!["link_15.0_tip".into()]),
        ..valid_dexpilot()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("finger_tip_link_names") && err.contains('2') && err.contains('5'),
        "unexpected message: {err}"
    );
}

#[test]
fn dexpilot_six_fingertips_fails_validation() {
    let config = RetargetingConfig {
        finger_tip_link_names: Some(vec![
            "a".into(),
            "b".into(),
            "c".into(),
            "d".into(),
            "e".into(),
            "f".into(),
        ]),
        ..valid_dexpilot()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("finger_tip_link_names"),
        "unexpected message: {err}"
    );
}

#[test]
fn dexpilot_two_and_five_fingertips_pass_validation() {
    for count in [2usize, 5] {
        let config = RetargetingConfig {
            finger_tip_link_names: Some((0..count).map(|i| format!("tip_{i}")).collect()),
            ..valid_dexpilot()
        };
        config
            .validate()
            .unwrap_or_else(|e| panic!("{count} fingertips must pass validation: {e}"));
    }
}

#[test]
fn build_with_unknown_vector_link_errs_instead_of_panicking() {
    let config = RetargetingConfig {
        urdf_path: allegro_urdf_path(),
        target_task_link_names: Some(vec!["link_15.0_tip".into(), "typo_link".into()]),
        target_origin_link_names: Some(vec!["wrist".into(), "wrist".into()]),
        target_link_human_indices: Some(vec![0, 0, 4, 8]),
        ..base(RetargetingType::Vector)
    };
    let err = build_err(config);
    assert!(err.contains("typo_link"), "unexpected message: {err}");
}

#[test]
fn build_with_unknown_position_link_errs_instead_of_panicking() {
    let config = RetargetingConfig {
        urdf_path: allegro_urdf_path(),
        target_link_names: Some(vec!["typo_link".into()]),
        target_link_human_indices: Some(vec![4]),
        ..base(RetargetingType::Position)
    };
    let err = build_err(config);
    assert!(err.contains("typo_link"), "unexpected message: {err}");
}

#[test]
fn build_with_unknown_dexpilot_wrist_errs_instead_of_panicking() {
    let config = RetargetingConfig {
        urdf_path: allegro_urdf_path(),
        wrist_link_name: Some("typo_wrist".into()),
        ..valid_dexpilot()
    };
    let err = build_err(config);
    assert!(err.contains("typo_wrist"), "unexpected message: {err}");
}

#[test]
fn build_with_unknown_joint_name_errs_instead_of_panicking() {
    let config = RetargetingConfig {
        urdf_path: allegro_urdf_path(),
        target_joint_names: Some(vec!["typo_joint".into()]),
        ..valid_vector_with_urdf()
    };
    let err = build_err(config);
    assert!(err.contains("typo_joint"), "unexpected message: {err}");
}

fn valid_vector_with_urdf() -> RetargetingConfig {
    RetargetingConfig {
        urdf_path: allegro_urdf_path(),
        target_origin_link_names: Some(vec!["wrist".into(), "wrist".into()]),
        target_task_link_names: Some(vec!["link_15.0_tip".into(), "link_3.0_tip".into()]),
        target_link_human_indices: Some(vec![0, 0, 4, 8]),
        ..base(RetargetingType::Vector)
    }
}

#[test]
fn build_with_valid_links_still_succeeds() {
    let retargeting = valid_vector_with_urdf()
        .build()
        .expect("valid vector config must build");
    assert!(!retargeting.optimizer.dof_joint_names().is_empty());
}

#[test]
fn build_with_unknown_dexpilot_fingertip_errs_instead_of_panicking() {
    let config = RetargetingConfig {
        urdf_path: allegro_urdf_path(),
        finger_tip_link_names: Some(vec!["link_15.0_tip".into(), "typo_tip".into()]),
        ..valid_dexpilot()
    };
    let err = build_err(config);
    assert!(err.contains("typo_tip"), "unexpected message: {err}");
}

#[test]
fn build_with_dummy_joints_and_unknown_joint_name_errs_instead_of_panicking() {
    let config = RetargetingConfig {
        add_dummy_free_joint: true,
        target_joint_names: Some(vec!["typo_joint".into()]),
        ..valid_vector_with_urdf()
    };
    let err = build_err(config);
    assert!(err.contains("typo_joint"), "unexpected message: {err}");
}
