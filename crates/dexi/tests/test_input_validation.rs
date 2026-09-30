//! Test: SeqRetargeting rejects malformed runtime inputs with an Err instead
//! of panicking or silently padding, and valid inputs land where expected.

mod common;

use common::{build, bundled_configs, fixed_dof, load, total_dof, workspace_root, XorShift};
use dexi::urdf::UrdfRobot;
use dexi::{RetargetingConfig, SeqRetargeting};

const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

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

/// Everything a failed call must leave untouched.
fn state(r: &SeqRetargeting) -> (Vec<f64>, usize, Option<Vec<f64>>) {
    let filter = r.filter.as_ref().and_then(|f| f.y.clone());
    (r.last_qpos.clone(), r.num_retargeting, filter)
}

// ---------------------------------------------------------------- ref_value

#[test]
fn every_bundled_config_accepts_exact_ref_value_len_only() {
    for name in bundled_configs() {
        let mut retargeting = build(&name);
        let n = retargeting.optimizer.ref_value_len();
        assert!(n > 0 && n.is_multiple_of(3), "{name}: ref_value_len {n}");

        for bad in [0, n - 1, n + 1, 2 * n] {
            let err = retargeting.retarget(&vec![0.05; bad], &[]).unwrap_err();
            assert_eq!(
                err,
                format!("ref_value must have {n} values, got {bad}"),
                "{name}"
            );
        }
        let qpos = retargeting.retarget(&vec![0.05; n], &[]).unwrap();
        assert_eq!(qpos.len(), total_dof(&retargeting), "{name}");
    }
}

#[test]
fn ref_value_len_matches_config_shape() {
    // Position: 3 per target link. Vector: 3 per origin/task pair.
    // DexPilot: 3 per (C(n,2) finger pairs + n wrist vectors).
    for (name, expected) in [
        ("offline/allegro_hand_right.yml", 3 * 8),
        ("teleop/allegro_hand_right.yml", 3 * 4),
        ("teleop/inspire_hand_right.yml", 3 * 5),
        ("teleop/panda_gripper_dexpilot.yml", 3 * (1 + 2)),
        ("teleop/allegro_hand_right_dexpilot.yml", 3 * (6 + 4)),
        ("teleop/shadow_hand_right_dexpilot.yml", 3 * (10 + 5)),
    ] {
        assert_eq!(build(name).optimizer.ref_value_len(), expected, "{name}");
    }
}

#[test]
fn ref_value_non_finite_is_err_at_any_position() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    for bad in NON_FINITE {
        for pos in [0, 5, 11] {
            let mut ref_value = vec![0.05; 12];
            ref_value[pos] = bad;
            let err = retargeting.retarget(&ref_value, &[]).unwrap_err();
            assert_eq!(err, "ref_value must contain only finite values");
        }
    }
}

#[test]
fn length_is_checked_before_finiteness() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let err = retargeting.retarget(&[f64::NAN; 11], &[]).unwrap_err();
    assert!(err.contains("12 values, got 11"), "unexpected: {err}");
}

// ---------------------------------------------------------------- fixed_qpos

#[test]
fn fixed_qpos_wrong_length_is_err() {
    let mut retargeting = build_with_fixed_joints();
    for bad in [1, 3, 5, 8] {
        let err = retargeting
            .retarget(&[0.05; 12], &vec![0.0; bad])
            .unwrap_err();
        assert_eq!(err, format!("fixed_qpos must have 4 values, got {bad}"));
    }
}

#[test]
fn fixed_qpos_given_when_config_has_no_fixed_joints_is_err() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    assert_eq!(fixed_dof(&retargeting), 0);
    let err = retargeting.retarget(&[0.05; 12], &[0.0]).unwrap_err();
    assert_eq!(err, "fixed_qpos must have 0 values, got 1");
}

#[test]
fn fixed_qpos_non_finite_is_err() {
    let mut retargeting = build_with_fixed_joints();
    for bad in NON_FINITE {
        let fixed = [0.0, 0.0, bad, 0.0];
        let err = retargeting.retarget(&[0.05; 12], &fixed).unwrap_err();
        assert_eq!(err, "fixed_qpos must contain only finite values");
        let err = retargeting.get_qpos(Some(&fixed)).unwrap_err();
        assert_eq!(err, "fixed_qpos must contain only finite values");
    }
}

#[test]
fn fixed_qpos_values_land_on_fixed_joints() {
    let mut retargeting = build_with_fixed_joints();
    let fixed = [0.1, -0.2, 0.3, 0.4];
    let idx = retargeting.optimizer.idx_pin2fixed().to_vec();
    let names = retargeting.optimizer.dof_joint_names();
    let fixed_names: Vec<&str> = idx.iter().map(|&i| names[i].as_str()).collect();
    assert_eq!(
        fixed_names,
        ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"]
    );

    // First call: the low-pass filter initializes to its input, so fixed
    // values come through unfiltered.
    let qpos = retargeting.retarget(&[0.05; 12], &fixed).unwrap();
    for (&i, &want) in idx.iter().zip(&fixed) {
        assert_eq!(qpos[i], want, "{}", names[i]);
    }
    let qpos = retargeting.get_qpos(Some(&fixed)).unwrap();
    for (&i, &want) in idx.iter().zip(&fixed) {
        assert_eq!(qpos[i], want, "{}", names[i]);
    }
}

#[test]
fn empty_fixed_qpos_holds_fixed_joints_at_zero() {
    let mut retargeting = build_with_fixed_joints();
    let idx = retargeting.optimizer.idx_pin2fixed().to_vec();
    let qpos = retargeting.retarget(&[0.05; 12], &[]).unwrap();
    assert!(idx.iter().all(|&i| qpos[i] == 0.0));
    assert_eq!(
        retargeting.get_qpos(Some(&[])).unwrap(),
        retargeting.get_qpos(None).unwrap()
    );
}

// ---------------------------------------------------------------- set/get qpos

#[test]
fn set_qpos_wrong_length_is_err() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let n = total_dof(&retargeting);
    for bad in [0, n - 1, n + 1] {
        let err = retargeting.set_qpos(&vec![0.0; bad]).unwrap_err();
        assert_eq!(err, format!("robot_qpos must have {n} values, got {bad}"));
    }
}

#[test]
fn set_qpos_non_finite_is_err() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let n = total_dof(&retargeting);
    for bad in NON_FINITE {
        let mut qpos = vec![0.1; n];
        qpos[n / 2] = bad;
        let err = retargeting.set_qpos(&qpos).unwrap_err();
        assert_eq!(err, "robot_qpos must contain only finite values");
    }
}

#[test]
fn set_qpos_round_trips_through_get_qpos() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let n = total_dof(&retargeting);
    let qpos: Vec<f64> = (0..n).map(|i| 0.01 * i as f64).collect();
    retargeting.set_qpos(&qpos).unwrap();
    assert_eq!(retargeting.get_qpos(None).unwrap(), qpos);
}

#[test]
fn get_qpos_fixed_qpos_wrong_length_is_err() {
    let retargeting = build_with_fixed_joints();
    for bad in [3, 5] {
        let err = retargeting.get_qpos(Some(&vec![0.0; bad])).unwrap_err();
        assert_eq!(err, format!("fixed_qpos must have 4 values, got {bad}"));
    }
}

// ---------------------------------------------------------------- no side effects

#[test]
fn failed_calls_leave_state_untouched() {
    let mut retargeting = build_with_fixed_joints();
    let mut rng = XorShift(42);
    retargeting.retarget(&rng.vec(12, 0.1), &[0.0; 4]).unwrap();
    let before = state(&retargeting);
    assert!(before.2.is_some(), "filter should be initialized");

    let _ = retargeting.retarget(&[0.05; 11], &[0.0; 4]).unwrap_err();
    let _ = retargeting
        .retarget(&[f64::NAN; 12], &[0.0; 4])
        .unwrap_err();
    let _ = retargeting.retarget(&[0.05; 12], &[0.0; 3]).unwrap_err();
    let _ = retargeting
        .retarget(&[0.05; 12], &[0.0, 0.0, 0.0, f64::NAN])
        .unwrap_err();
    let _ = retargeting.set_qpos(&[0.0; 3]).unwrap_err();
    let _ = retargeting.set_qpos(&[f64::NAN; 16]).unwrap_err();
    assert_eq!(state(&retargeting), before);
}

#[test]
fn failed_calls_do_not_change_later_outputs() {
    let mut clean = build("teleop/allegro_hand_right_dexpilot.yml");
    let mut noisy = build("teleop/allegro_hand_right_dexpilot.yml");
    let mut rng = XorShift(99);
    for _ in 0..4 {
        let ref_value = rng.vec(30, 0.1);
        let _ = noisy.retarget(&ref_value[..29], &[]).unwrap_err();
        let _ = noisy.retarget(&[f64::INFINITY; 30], &[]).unwrap_err();
        assert_eq!(
            noisy.retarget(&ref_value, &[]).unwrap(),
            clean.retarget(&ref_value, &[]).unwrap()
        );
    }
}

// ---------------------------------------------------------------- mimic joints

fn mimic_spec(urdf: &str) -> Vec<(String, String, f64, f64)> {
    let path = workspace_root().join("assets/robots/hands").join(urdf);
    let robot = UrdfRobot::from_file(&path).unwrap();
    robot
        .joints
        .iter()
        .filter_map(|j| {
            j.mimic
                .as_ref()
                .map(|m| (j.name.clone(), m.joint.clone(), m.multiplier, m.offset))
        })
        .collect()
}

#[test]
fn mimic_joints_are_not_fixed_and_follow_their_source() {
    let mut retargeting = build("teleop/inspire_hand_right.yml");
    assert_eq!(total_dof(&retargeting), 12);
    assert_eq!(retargeting.optimizer.opt_dof(), 6);
    assert_eq!(fixed_dof(&retargeting), 0, "mimic joints must not be fixed");

    let names = retargeting.optimizer.dof_joint_names();
    let index = |n: &str| names.iter().position(|x| x == n).unwrap();
    let spec = mimic_spec("inspire_hand/inspire_hand_right.urdf");
    assert_eq!(spec.len(), 6);

    let mut rng = XorShift(5);
    for _ in 0..3 {
        let qpos = retargeting.retarget(&rng.vec(15, 0.2), &[]).unwrap();
        // The low-pass filter is affine, so it preserves the mimic relation.
        for (mimic, source, mult, offset) in &spec {
            let want = qpos[index(source)] * mult + offset;
            assert!(
                (qpos[index(mimic)] - want).abs() < 1e-12,
                "{mimic} does not follow {source}"
            );
        }
    }
}

#[test]
fn excluding_a_mimic_source_from_targets_is_build_err() {
    let config = RetargetingConfig {
        target_joint_names: Some(vec![
            "ring_proximal_joint".into(),
            "middle_proximal_joint".into(),
            "index_proximal_joint".into(),
            "thumb_proximal_pitch_joint".into(),
            "thumb_proximal_yaw_joint".into(),
        ]),
        ..load("teleop/inspire_hand_right.yml")
    };
    let err = config.build().err().expect("build must fail");
    assert!(err.contains("pinky_proximal_joint"), "unexpected: {err}");
}

#[test]
fn ignored_mimic_joints_become_fixed_joints() {
    let config = RetargetingConfig {
        ignore_mimic_joint: true,
        ..load("teleop/inspire_hand_right.yml")
    };
    let mut retargeting = config.build().unwrap();
    assert_eq!(fixed_dof(&retargeting), 6);
    let fixed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let qpos = retargeting.retarget(&[0.05; 15], &fixed).unwrap();
    let idx = retargeting.optimizer.idx_pin2fixed().to_vec();
    for (&i, &want) in idx.iter().zip(&fixed) {
        assert_eq!(qpos[i], want);
    }
    let err = retargeting.retarget(&[0.05; 15], &fixed[..5]).unwrap_err();
    assert_eq!(err, "fixed_qpos must have 6 values, got 5");
}

// ---------------------------------------------------------------- reset / filter

#[test]
fn reset_restores_midpoint_warm_start_and_counter() {
    let mut retargeting = build("teleop/allegro_hand_right.yml");
    let midpoint: Vec<f64> = retargeting
        .joint_limits
        .iter()
        .map(|(lo, hi)| (lo + hi) / 2.0)
        .collect();
    assert_eq!(retargeting.last_qpos, midpoint);
    retargeting.retarget(&[0.05; 12], &[]).unwrap();
    assert_eq!(retargeting.num_retargeting, 1);
    assert_ne!(retargeting.last_qpos, midpoint);

    retargeting.reset();
    assert_eq!(retargeting.last_qpos, midpoint);
    assert_eq!(retargeting.num_retargeting, 0);
}

#[test]
fn out_of_range_low_pass_alpha_disables_filter() {
    let config = RetargetingConfig {
        low_pass_alpha: -1.0,
        ..load("teleop/allegro_hand_right.yml")
    };
    let mut retargeting = config.build().unwrap();
    assert!(retargeting.filter.is_none());

    // Unfiltered output at the optimized joints equals the solver result
    // stored as the next warm start.
    let mut rng = XorShift(11);
    for _ in 0..3 {
        let qpos = retargeting.retarget(&rng.vec(12, 0.1), &[]).unwrap();
        let idx = retargeting.optimizer.idx_pin2target().to_vec();
        let at_targets: Vec<f64> = idx.iter().map(|&i| qpos[i]).collect();
        assert_eq!(at_targets, retargeting.last_qpos);
    }
}
