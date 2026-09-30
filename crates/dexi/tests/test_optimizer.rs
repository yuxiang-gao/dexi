//! Test: optimizer smoke tests for all supported hand config families.

use dexi::{RetargetingConfig, RobotWrapper, SeqRetargeting};
use nalgebra::DMatrix;
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

/// Build a SeqRetargeting from a config name (e.g., "offline/allegro_hand_left.yml")
fn build_retargeting(config_name: &str) -> SeqRetargeting {
    let config_path = configs_dir().join(config_name);
    let mut config = RetargetingConfig::load_from_path(&config_path)
        .unwrap_or_else(|e| panic!("Failed to load config {}: {}", config_name, e));
    config.set_default_urdf_dir(&robots_dir());
    config
        .build()
        .unwrap_or_else(|e| panic!("Failed to build retargeting for {}: {}", config_name, e))
}

/// Test basic FK computation for allegro hand
#[test]
fn test_fk_allegro() {
    let urdf_path = robots_dir().join("allegro_hand/allegro_hand_left.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap())
        .expect("Failed to load allegro URDF");

    let ndof = robot.dof();
    assert_eq!(ndof, 16); // Allegro has 16 DOF

    let qpos = vec![0.0; ndof];
    robot.compute_forward_kinematics(&qpos);

    // Check that tip links are not at origin
    let tip_link = robot
        .get_link_index("link_3.0_tip")
        .expect("tip link not found");
    let pose = robot.get_link_pose(tip_link);
    // Position should not be all zeros
    let pos_norm = (pose[(0, 3)].powi(2) + pose[(1, 3)].powi(2) + pose[(2, 3)].powi(2)).sqrt();
    assert!(
        pos_norm > 0.01,
        "Tip link position norm too small: {}",
        pos_norm
    );
}

/// Test FK for inspire hand (has mimic joints)
#[test]
fn test_fk_inspire() {
    let urdf_path = robots_dir().join("inspire_hand/inspire_hand_left.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap())
        .expect("Failed to load inspire URDF");

    let ndof = robot.dof();
    assert!(ndof > 0, "Inspire hand should have DOF > 0");

    let qpos = vec![0.0; ndof];
    robot.compute_forward_kinematics(&qpos);
}

/// Test position retargeting for allegro hand with synthetic data
#[test]
fn test_position_retargeting_allegro() {
    let config_name = "offline/allegro_hand_left.yml";
    let mut retargeting = build_retargeting(config_name);

    // Generate synthetic target positions from FK at a known qpos
    let urdf_path = robots_dir().join("allegro_hand/allegro_hand_left.urdf");
    let robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    // Sample a qpos within joint limits (6 dummy + 16 = 22 DOF)
    let ndof = robot.dof(); // 16 DOF without dummy
    let dummy_ndof = 6;
    let total_ndof = ndof + dummy_ndof;
    let qpos_full = vec![0.3; total_ndof];

    // Create robot with dummy joints
    let urdf_with_dummy = {
        let mut urdf = dexi::urdf::UrdfRobot::from_file(
            &robots_dir().join("allegro_hand/allegro_hand_left.urdf"),
        )
        .unwrap();
        dexi::retargeting_config::add_dummy_free_joints(&mut urdf);
        urdf
    };
    let mut robot_full = RobotWrapper::new(urdf_with_dummy).unwrap();

    robot_full.compute_forward_kinematics(&qpos_full);

    // Get target link positions
    let target_link_names = [
        "link_15.0_tip",
        "link_11.0_tip",
        "link_7.0_tip",
        "link_3.0_tip",
        "link_14.0",
        "link_10.0",
        "link_6.0",
        "link_2.0",
    ];
    let mut ref_value = Vec::new();
    for name in &target_link_names {
        let idx = robot_full
            .get_link_index(name)
            .unwrap_or_else(|| panic!("Link {} not found", name));
        let pose = robot_full.get_link_pose(idx);
        ref_value.push(pose[(0, 3)]);
        ref_value.push(pose[(1, 3)]);
        ref_value.push(pose[(2, 3)]);
    }

    // Run retargeting
    let result = retargeting.retarget(&ref_value, &[]).unwrap();
    assert!(
        !result.is_empty(),
        "Retargeting should return non-empty result"
    );
    // After retargeting, the result should be close to the original qpos for the finger joints
}

/// Test vector retargeting for allegro hand
#[test]
fn test_vector_retargeting_allegro() {
    let config_name = "teleop/allegro_hand_left.yml";
    let mut retargeting = build_retargeting(config_name);

    // Generate synthetic vectors from FK
    let urdf_path = robots_dir().join("allegro_hand/allegro_hand_left.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    let ndof = robot.dof();
    let qpos = vec![0.3; ndof];
    robot.compute_forward_kinematics(&qpos);

    let wrist_idx = robot.get_link_index("wrist").unwrap();
    let wrist_pose = robot.get_link_pose(wrist_idx);
    let wrist_pos = [wrist_pose[(0, 3)], wrist_pose[(1, 3)], wrist_pose[(2, 3)]];

    let tip_names = [
        "link_15.0_tip",
        "link_11.0_tip",
        "link_7.0_tip",
        "link_3.0_tip",
    ];
    let mut ref_value = Vec::new();
    for name in &tip_names {
        let idx = robot.get_link_index(name).unwrap();
        let pose = robot.get_link_pose(idx);
        ref_value.push(pose[(0, 3)] - wrist_pos[0]);
        ref_value.push(pose[(1, 3)] - wrist_pos[1]);
        ref_value.push(pose[(2, 3)] - wrist_pos[2]);
    }

    let result = retargeting.retarget(&ref_value, &[]).unwrap();
    assert!(!result.is_empty());
}

/// Test DexPilot retargeting for allegro hand
#[test]
fn test_dexpilot_retargeting_allegro() {
    let config_name = "teleop/allegro_hand_left_dexpilot.yml";
    let mut retargeting = build_retargeting(config_name);

    // Generate synthetic vectors from FK
    let urdf_path = robots_dir().join("allegro_hand/allegro_hand_left.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    let ndof = robot.dof();
    let qpos = vec![0.3; ndof];
    robot.compute_forward_kinematics(&qpos);

    let wrist_idx = robot.get_link_index("wrist").unwrap();
    let wrist_pose = robot.get_link_pose(wrist_idx);
    let wrist_pos = [wrist_pose[(0, 3)], wrist_pose[(1, 3)], wrist_pose[(2, 3)]];

    let tip_names = [
        "link_15.0_tip",
        "link_11.0_tip",
        "link_7.0_tip",
        "link_3.0_tip",
    ];
    let mut tip_positions = Vec::new();
    for name in &tip_names {
        let idx = robot.get_link_index(name).unwrap();
        let pose = robot.get_link_pose(idx);
        tip_positions.push([pose[(0, 3)], pose[(1, 3)], pose[(2, 3)]]);
    }

    // DexPilot generate_link_indices(4) produces:
    // origin: [2,3,4, 3,4, 4, 0,0,0,0]
    // task:   [1,1,1, 2,2, 3, 1,2,3,4]
    // where link_names = [wrist, tip0, tip1, tip2, tip3]
    // link 0 = wrist, link 1 = tip0, link 2 = tip1, link 3 = tip2, link 4 = tip3
    let all_positions: Vec<[f64; 3]> = std::iter::once(wrist_pos)
        .chain(tip_positions.iter().cloned())
        .collect();

    let origin_indices = [2, 3, 4, 3, 4, 4, 0, 0, 0, 0];
    let task_indices = [1, 1, 1, 2, 2, 3, 1, 2, 3, 4];

    let mut ref_value = Vec::new();
    for i in 0..10 {
        let oi = origin_indices[i];
        let ti = task_indices[i];
        ref_value.push(all_positions[ti][0] - all_positions[oi][0]);
        ref_value.push(all_positions[ti][1] - all_positions[oi][1]);
        ref_value.push(all_positions[ti][2] - all_positions[oi][2]);
    }

    let result = retargeting.retarget(&ref_value, &[]).unwrap();
    assert!(!result.is_empty());
}

/// Test position retargeting for inspire hand (has mimic joints)
#[test]
fn test_position_retargeting_inspire() {
    let config_name = "offline/inspire_hand_left.yml";
    let mut retargeting = build_retargeting(config_name);

    let urdf_path = robots_dir().join("inspire_hand/inspire_hand_left.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    let ndof = robot.dof();
    let qpos = vec![0.2; ndof];
    robot.compute_forward_kinematics(&qpos);

    let tip_names = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ];
    let mut ref_value = Vec::new();
    for name in &tip_names {
        let idx = robot.get_link_index(name).unwrap();
        let pose = robot.get_link_pose(idx);
        ref_value.push(pose[(0, 3)]);
        ref_value.push(pose[(1, 3)]);
        ref_value.push(pose[(2, 3)]);
    }

    let result = retargeting.retarget(&ref_value, &[]).unwrap();
    assert!(!result.is_empty());
}

/// Test LEAP hand
#[test]
fn test_position_retargeting_leap() {
    let config_name = "offline/leap_hand_left.yml";
    let mut retargeting = build_retargeting(config_name);

    let urdf_path = robots_dir().join("leap_hand/leap_hand_left.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    let ndof = robot.dof();
    let qpos = vec![0.2; ndof];
    robot.compute_forward_kinematics(&qpos);

    let tip_names = [
        "thumb_tip_head",
        "index_tip_head",
        "middle_tip_head",
        "ring_tip_head",
        "thumb_dip",
        "dip",
        "dip_2",
        "dip_3",
    ];
    let mut ref_value = Vec::new();
    for name in &tip_names {
        if let Some(idx) = robot.get_link_index(name) {
            let pose = robot.get_link_pose(idx);
            ref_value.push(pose[(0, 3)]);
            ref_value.push(pose[(1, 3)]);
            ref_value.push(pose[(2, 3)]);
        }
    }

    if !ref_value.is_empty() {
        let result = retargeting.retarget(&ref_value, &[]).unwrap();
        assert!(!result.is_empty());
    }
}

/// Test panda gripper (with mimic joint)
#[test]
fn test_position_retargeting_panda() {
    let config_name = "offline/panda_gripper.yml";
    let mut retargeting = build_retargeting(config_name);

    let urdf_path = robots_dir().join("panda_gripper/panda_gripper_glb.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    let ndof = robot.dof();
    let qpos = vec![0.02; ndof]; // small opening
    robot.compute_forward_kinematics(&qpos);

    let tip_names = ["panda_leftfinger", "panda_rightfinger"];
    let mut ref_value = Vec::new();
    for name in &tip_names {
        if let Some(idx) = robot.get_link_index(name) {
            let pose = robot.get_link_pose(idx);
            ref_value.push(pose[(0, 3)]);
            ref_value.push(pose[(1, 3)]);
            ref_value.push(pose[(2, 3)]);
        }
    }

    if !ref_value.is_empty() {
        let result = retargeting.retarget(&ref_value, &[]).unwrap();
        assert!(!result.is_empty());
    }
}

/// Test shadow hand
#[test]
fn test_position_retargeting_shadow() {
    let config_name = "offline/shadow_hand_left.yml";
    let _retargeting = build_retargeting(config_name);

    let urdf_path = robots_dir().join("shadow_hand/shadow_hand_left.urdf");
    let robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    let ndof = robot.dof();
    assert!(ndof > 0);
}

/// Test ability hand
#[test]
fn test_position_retargeting_ability() {
    let config_name = "offline/ability_hand_left.yml";
    let _retargeting = build_retargeting(config_name);
}

/// Test schunk hand
#[test]
fn test_position_retargeting_schunk() {
    let config_name = "offline/schunk_svh_hand_left.yml";
    let _retargeting = build_retargeting(config_name);
}

/// Test LP filter
#[test]
fn test_lp_filter() {
    use dexi::LPFilter;
    let mut filter = LPFilter::new(0.5);

    let x1 = vec![1.0, 2.0, 3.0];
    let y1 = filter.next(&x1);
    assert_eq!(y1, x1); // First call returns input

    let x2 = vec![3.0, 4.0, 5.0];
    let y2 = filter.next(&x2);
    // y = y + alpha * (x - y) = [1,2,3] + 0.5 * ([3,4,5] - [1,2,3]) = [2, 3, 4]
    assert!((y2[0] - 2.0).abs() < 1e-10);
    assert!((y2[1] - 3.0).abs() < 1e-10);
    assert!((y2[2] - 4.0).abs() < 1e-10);

    filter.reset();
    assert!(!filter.is_init);
}

/// Test URDF parsing for allegro hand
#[test]
fn test_urdf_parsing_allegro() {
    let urdf_path = robots_dir().join("allegro_hand/allegro_hand_left.urdf");
    let robot = dexi::urdf::UrdfRobot::from_file(&urdf_path).unwrap();

    assert_eq!(robot.name, "allegro_left");
    assert!(robot.links.len() > 10);
    assert!(robot.joints.len() > 10);

    let root = robot.root_link().unwrap();
    assert_eq!(root, "base_link");

    let dof_names = robot.dof_joint_names();
    assert_eq!(dof_names.len(), 16);

    // Check no mimic joints
    let (has_mimic, _, _, _, _) = robot.parse_mimic_joints();
    assert!(!has_mimic);
}

/// Test URDF parsing for inspire hand (has mimic joints)
#[test]
fn test_urdf_parsing_inspire() {
    let urdf_path = robots_dir().join("inspire_hand/inspire_hand_left.urdf");
    let robot = dexi::urdf::UrdfRobot::from_file(&urdf_path).unwrap();

    assert!(robot.links.len() > 5);

    let (has_mimic, source_names, mimic_names, multipliers, offsets) = robot.parse_mimic_joints();
    assert!(has_mimic);
    assert_eq!(source_names.len(), mimic_names.len());
    assert_eq!(multipliers.len(), offsets.len());

    // Check specific mimic joint
    assert!(mimic_names.contains(&"thumb_intermediate_joint".to_string()));
}

/// Test Jacobian computation
#[test]
fn test_jacobian_computation() {
    let urdf_path = robots_dir().join("allegro_hand/allegro_hand_left.urdf");
    let mut robot = RobotWrapper::from_urdf_path(urdf_path.to_str().unwrap()).unwrap();

    let ndof = robot.dof();
    let qpos = vec![0.0; ndof];
    let tip_idx = robot.get_link_index("link_3.0_tip").unwrap();

    let numeric = finite_difference_position_jacobian(&mut robot, &qpos, tip_idx);

    // Position Jacobian should not be all zeros for a reachable link
    let norm = numeric.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    assert!(norm > 1e-6, "Jacobian should not be all zeros");

    let analytic = robot.compute_single_link_position_jacobian(&qpos, tip_idx);
    assert_eq!(analytic.nrows(), 3);
    assert_eq!(analytic.ncols(), ndof);
    assert_matrix_close(&analytic, &numeric, 1e-5);
}

/// Forward-difference world-frame position Jacobian, used as the reference
/// for the analytic implementation.
fn finite_difference_position_jacobian(
    robot: &mut RobotWrapper,
    qpos: &[f64],
    link_id: usize,
) -> DMatrix<f64> {
    let eps = 1e-7;
    robot.compute_forward_kinematics(qpos);
    let ref_pose = robot.get_link_pose(link_id);
    let mut jac = DMatrix::zeros(3, qpos.len());
    for col in 0..qpos.len() {
        let mut perturbed = qpos.to_vec();
        perturbed[col] += eps;
        robot.compute_forward_kinematics(&perturbed);
        let pose = robot.get_link_pose(link_id);
        for row in 0..3 {
            jac[(row, col)] = (pose[(row, 3)] - ref_pose[(row, 3)]) / eps;
        }
    }
    jac
}

fn assert_matrix_close(left: &DMatrix<f64>, right: &DMatrix<f64>, tolerance: f64) {
    assert_eq!(left.shape(), right.shape());
    let max_abs = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        max_abs <= tolerance,
        "matrix mismatch: max_abs={} tolerance={}",
        max_abs,
        tolerance
    );
}

#[test]
fn test_pinocchio_joint_order_supported_robots() {
    let cases: [(&str, &[&str]); 6] = [
        (
            "allegro_hand/allegro_hand_left.urdf",
            &[
                "joint_0.0",
                "joint_1.0",
                "joint_2.0",
                "joint_3.0",
                "joint_12.0",
                "joint_13.0",
                "joint_14.0",
                "joint_15.0",
                "joint_4.0",
                "joint_5.0",
                "joint_6.0",
                "joint_7.0",
                "joint_8.0",
                "joint_9.0",
                "joint_10.0",
                "joint_11.0",
            ],
        ),
        (
            "ability_hand/ability_hand_left.urdf",
            &[
                "index_q1",
                "index_q2",
                "middle_q1",
                "middle_q2",
                "pinky_q1",
                "pinky_q2",
                "ring_q1",
                "ring_q2",
                "thumb_q1",
                "thumb_q2",
            ],
        ),
        (
            "inspire_hand/inspire_hand_left.urdf",
            &[
                "index_proximal_joint",
                "index_intermediate_joint",
                "middle_proximal_joint",
                "middle_intermediate_joint",
                "pinky_proximal_joint",
                "pinky_intermediate_joint",
                "ring_proximal_joint",
                "ring_intermediate_joint",
                "thumb_proximal_yaw_joint",
                "thumb_proximal_pitch_joint",
                "thumb_intermediate_joint",
                "thumb_distal_joint",
            ],
        ),
        (
            "leap_hand/leap_hand_left.urdf",
            &[
                "1", "0", "2", "3", "12", "13", "14", "15", "5", "4", "6", "7", "9", "8", "10",
                "11",
            ],
        ),
        (
            "shadow_hand/shadow_hand_left.urdf",
            &[
                "WRJ2", "WRJ1", "FFJ4", "FFJ3", "FFJ2", "FFJ1", "LFJ5", "LFJ4", "LFJ3", "LFJ2",
                "LFJ1", "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2", "RFJ1", "THJ5",
                "THJ4", "THJ3", "THJ2", "THJ1",
            ],
        ),
        (
            "schunk_hand/schunk_svh_hand_left.urdf",
            &[
                "left_hand_Thumb_Opposition",
                "left_hand_Thumb_Flexion",
                "left_hand_j3",
                "left_hand_j4",
                "left_hand_index_spread",
                "left_hand_Index_Finger_Proximal",
                "left_hand_Index_Finger_Distal",
                "left_hand_j14",
                "left_hand_j5",
                "left_hand_Finger_Spread",
                "left_hand_Pinky",
                "left_hand_j13",
                "left_hand_j17",
                "left_hand_ring_spread",
                "left_hand_Ring_Finger",
                "left_hand_j12",
                "left_hand_j16",
                "left_hand_Middle_Finger_Proximal",
                "left_hand_Middle_Finger_Distal",
                "left_hand_j15",
            ],
        ),
    ];

    for (urdf, expected) in cases {
        let robot =
            RobotWrapper::from_urdf_path(robots_dir().join(urdf).to_str().unwrap()).unwrap();
        assert_eq!(
            robot.dof_joint_names(),
            expected,
            "joint order mismatch for {urdf}"
        );
    }
}
