//! Test: all YAML configs can be parsed and loaded successfully.

use dexi::RetargetingConfig;
use std::path::Path;

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

fn robots_dir() -> std::path::PathBuf {
    workspace_root().join("assets/robots/hands")
}

fn configs_dir() -> std::path::PathBuf {
    workspace_root().join("configs")
}

/// Helper to load a config from the root configs directory.
/// The URDF path in the config is relative, so we need to resolve it
/// through the root assets/robots/hands directory.
fn load_config(config_relative: &str) -> RetargetingConfig {
    let config_path = configs_dir().join(config_relative);
    let config = RetargetingConfig::load_from_path(&config_path)
        .unwrap_or_else(|e| panic!("Failed to load config {}: {}", config_relative, e));
    config
}

/// Test that all offline configs can be parsed (YAML structure is valid)
#[test]
fn test_parse_all_offline_configs() {
    let offline_dir = configs_dir().join("offline");
    let entries = std::fs::read_dir(&offline_dir).expect("Failed to read offline configs dir");

    let mut count = 0;
    for entry in entries {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();
        if path
            .extension()
            .map(|e| e == "yml" || e == "yaml")
            .unwrap_or(false)
        {
            let config_name = format!("offline/{}", path.file_name().unwrap().to_string_lossy());
            let config = load_config(&config_name);
            assert!(
                !config.urdf_path.is_empty(),
                "URDF path empty for {}",
                config_name
            );
            count += 1;
        }
    }
    assert!(
        count >= 10,
        "Expected at least 10 offline configs, got {}",
        count
    );
}

/// Test that all teleop configs can be parsed
#[test]
fn test_parse_all_teleop_configs() {
    let teleop_dir = configs_dir().join("teleop");
    let entries = std::fs::read_dir(&teleop_dir).expect("Failed to read teleop configs dir");

    let mut count = 0;
    for entry in entries {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();
        if path
            .extension()
            .map(|e| e == "yml" || e == "yaml")
            .unwrap_or(false)
        {
            let config_name = format!("teleop/{}", path.file_name().unwrap().to_string_lossy());
            let config = load_config(&config_name);
            assert!(
                !config.urdf_path.is_empty(),
                "URDF path empty for {}",
                config_name
            );
            count += 1;
        }
    }
    assert!(
        count >= 10,
        "Expected at least 10 teleop configs, got {}",
        count
    );
}

/// Test that URDF paths in configs can be resolved to actual files
#[test]
fn test_config_urdf_path_resolution() {
    let configs = [
        "offline/allegro_hand_left.yml",
        "offline/allegro_hand_right.yml",
        "offline/leap_hand_left.yml",
        "offline/shadow_hand_left.yml",
        "offline/inspire_hand_left.yml",
        "offline/panda_gripper.yml",
        "offline/schunk_svh_hand_left.yml",
        "offline/ability_hand_left.yml",
        "teleop/allegro_hand_left.yml",
        "teleop/allegro_hand_left_dexpilot.yml",
        "teleop/fourier_hand_left_6dof.yml",
        "teleop/fourier_hand_right_12dof.yml",
    ];

    for config_name in &configs {
        let config = load_config(config_name);
        // Try to resolve the URDF path
        // The config's default_urdf_dir is the config file's parent dir (offline/ or teleop/)
        // We need to set it to assets/robots/hands for resolution
        let mut config = config;
        config.set_default_urdf_dir(&robots_dir());
        let urdf_result = config.resolve_urdf_path();
        assert!(
            urdf_result.is_ok(),
            "Failed to resolve URDF for {}: {:?}",
            config_name,
            urdf_result.err()
        );
    }
}

/// Test config type detection
#[test]
fn test_config_type_parsing() {
    let pos_config = load_config("offline/allegro_hand_left.yml");
    assert!(matches!(pos_config.type_, dexi::RetargetingType::Position));

    let vec_config = load_config("teleop/allegro_hand_left.yml");
    assert!(matches!(vec_config.type_, dexi::RetargetingType::Vector));

    let dexpilot_config = load_config("teleop/allegro_hand_left_dexpilot.yml");
    assert!(matches!(
        dexpilot_config.type_,
        dexi::RetargetingType::DexPilot
    ));
}

/// Test config field parsing
#[test]
fn test_config_field_parsing() {
    let config = load_config("offline/allegro_hand_left.yml");
    assert!(config.add_dummy_free_joint);
    assert!(config.target_link_names.is_some());
    assert!(config.target_link_human_indices.is_some());
    assert!(config.low_pass_alpha > 0.0);
}

/// Test vector config field parsing
#[test]
fn test_vector_config_fields() {
    let config = load_config("teleop/allegro_hand_left.yml");
    assert!(config.target_origin_link_names.is_some());
    assert!(config.target_task_link_names.is_some());
    assert!(config.target_link_human_indices.is_some());
    let origin = config.target_origin_link_names.unwrap();
    let task = config.target_task_link_names.unwrap();
    assert_eq!(origin.len(), task.len());
}

/// Test DexPilot config field parsing
#[test]
fn test_dexpilot_config_fields() {
    let config = load_config("teleop/allegro_hand_left_dexpilot.yml");
    assert!(config.finger_tip_link_names.is_some());
    assert!(config.wrist_link_name.is_some());
    let fingers = config.finger_tip_link_names.unwrap();
    assert_eq!(fingers.len(), 4); // Allegro has 4 fingers
}

/// Relative URDF paths fall back to the current working directory as a
/// last resort. cargo test runs with cwd = crates/dexi, so a file relative
/// to the crate root is only reachable through the cwd fallback
/// (resolution is existence-based, so any file works as a probe).
#[test]
fn test_resolve_urdf_path_cwd_fallback() {
    let config = RetargetingConfig {
        urdf_path: "src/constants.rs".to_string(),
        default_urdf_dir: std::env::temp_dir(),
        ..Default::default()
    };
    let resolved = config
        .resolve_urdf_path()
        .expect("cwd fallback should resolve");
    assert!(resolved.ends_with("src/constants.rs"), "got {resolved:?}");
}
