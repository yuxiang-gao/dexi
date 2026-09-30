//! Test: retargeting outputs stay stable across refactors and respect joint
//! limits for every bundled config.

mod common;

use common::{build, bundled_configs, workspace_root, XorShift};
use serde::Deserialize;

#[derive(Deserialize)]
struct GoldenFrame {
    ref_value: Vec<f64>,
    qpos: Vec<f64>,
}

#[derive(Deserialize)]
struct GoldenCase {
    config: String,
    frames: Vec<GoldenFrame>,
}

/// Outputs recorded at a3ee78d, before the clip-to-limits and DexPilot
/// projection loops were refactored. Frames alternate wide and pinched
/// targets so DexPilot projection toggles in both directions, and run
/// sequentially so warm starts and the low-pass filter are exercised.
#[test]
fn outputs_match_pre_refactor_golden() {
    let fixture = workspace_root().join("crates/dexi/tests/fixtures/pre_refactor_golden.yml");
    let cases: Vec<GoldenCase> =
        serde_yaml::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    assert!(cases.len() >= 9, "fixture lost cases");

    for case in &cases {
        let mut retargeting = build(&case.config);
        let fixed = vec![0.0; retargeting.optimizer.idx_pin2fixed().len()];
        for (i, frame) in case.frames.iter().enumerate() {
            let qpos = retargeting.retarget(&frame.ref_value, &fixed).unwrap();
            assert_eq!(qpos.len(), frame.qpos.len(), "{} frame {i}", case.config);
            for (j, (got, want)) in qpos.iter().zip(&frame.qpos).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-9,
                    "{} frame {i} joint {j}: got {got}, want {want}",
                    case.config
                );
            }
        }
    }
}

/// Optimized joints must stay within URDF limits (widened by the solver's
/// 1e-3 bound slack) for every bundled config, including extreme targets.
#[test]
fn every_bundled_config_respects_joint_limits() {
    let configs = bundled_configs();
    assert_eq!(configs.len(), 43, "bundled config count changed");
    let mut rng = XorShift(0xD1B5_4A32_D192_ED03);

    for name in &configs {
        let mut retargeting = build(name);
        let n_ref = retargeting.optimizer.ref_value_len();
        let targets = retargeting.optimizer.idx_pin2target().to_vec();
        let limits = retargeting.joint_limits.clone();
        for span in [0.1, 2.0] {
            let qpos = retargeting.retarget(&rng.vec(n_ref, span), &[]).unwrap();
            assert!(qpos.iter().all(|q| q.is_finite()), "{name}: non-finite");
            for (&ti, &(lo, hi)) in targets.iter().zip(&limits) {
                assert!(
                    qpos[ti] >= lo - 1e-3 - 1e-9 && qpos[ti] <= hi + 1e-3 + 1e-9,
                    "{name}: joint {ti} = {} outside [{lo}, {hi}]",
                    qpos[ti]
                );
            }
        }
    }
}

/// `SeqRetargeting` clips `last_qpos` before calling the optimizer, so the
/// optimizer's own clipping only matters when `Optimizer::retarget` is called
/// directly with an out-of-bounds warm start.
#[test]
fn optimizer_clips_out_of_bounds_warm_start() {
    for name in [
        "offline/allegro_hand_right.yml",
        "teleop/allegro_hand_right.yml",
        "teleop/allegro_hand_right_dexpilot.yml",
    ] {
        let mut retargeting = build(name);
        let n_ref = retargeting.optimizer.ref_value_len();
        let limits = retargeting.joint_limits.clone();
        let far_out = vec![100.0; retargeting.optimizer.opt_dof()];
        let ref_value = XorShift(7).vec(n_ref, 0.1);
        let x = retargeting.optimizer.retarget(&ref_value, &[], &far_out);
        for (i, (&xi, &(lo, hi))) in x.iter().zip(&limits).enumerate() {
            assert!(
                xi >= lo - 1e-3 - 1e-9 && xi <= hi + 1e-3 + 1e-9,
                "{name}: joint {i} = {xi} outside [{lo}, {hi}]"
            );
        }
    }
}
