#!/usr/bin/env python3
"""Compare reference dex-retargeting against the Rust dexi_py binding.

The script exercises every supported reference YAML config by default:

  python scripts/compare_python_rust.py

Expected setup:
  1. Install the reference package, or run from the repository root so this
     script can add reference/dex-retargeting/src to sys.path.
  2. Install the Rust binding into the active environment, for example:
       maturin develop --manifest-path crates/dexi-py/Cargo.toml

The Rust implementation intentionally uses a pure-Rust kinematics/optimizer
stack instead of Pinocchio + NLopt. This script reports numerical drift for
all configs and only fails on drift thresholds when --fail-on-threshold is set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "reference" / "dex-retargeting"
REFERENCE_SRC = REFERENCE_ROOT / "src"
REFERENCE_CONFIGS = REFERENCE_SRC / "dex_retargeting" / "configs"
REFERENCE_URDF_DIR = REFERENCE_ROOT / "assets" / "robots" / "hands"


@dataclass
class ComparisonRow:
    config: str
    hand: str
    retargeting_type: str
    dof: int
    fixed_dof: int
    max_abs_error: float
    mean_abs_error: float
    rms_error: float
    python_norm: float
    rust_norm: float
    status: str
    message: str = ""


def import_packages() -> tuple[Any, Any]:
    if str(REFERENCE_SRC) not in sys.path:
        sys.path.insert(0, str(REFERENCE_SRC))

    from dex_retargeting.retargeting_config import RetargetingConfig as PyConfig

    try:
        import dexi_py
    except ImportError as exc:
        raise SystemExit(
            "Could not import dexi_py. Build/install it first, e.g.\n"
            "  maturin develop --manifest-path crates/dexi-py/Cargo.toml"
        ) from exc

    return PyConfig, dexi_py


def config_paths() -> list[Path]:
    return sorted(REFERENCE_CONFIGS.glob("offline/*.yml")) + sorted(
        REFERENCE_CONFIGS.glob("teleop/*.yml")
    )


def hand_name(path: Path) -> str:
    name = path.stem.replace("_dexpilot", "")
    return name.replace("schunk_svh", "schunk_svh")


def vector_count(cfg: dict[str, Any]) -> int:
    rtype = str(cfg["type"]).lower()
    if rtype == "position":
        return len(cfg["target_link_names"])
    if rtype == "vector":
        return len(cfg["target_origin_link_names"])
    if rtype == "dexpilot":
        indices = cfg.get("target_link_human_indices")
        if isinstance(indices, list) and indices and isinstance(indices[0], list):
            return len(indices[0])
        num_fingers = len(cfg["finger_tip_link_names"])
        return num_fingers * (num_fingers - 1) // 2 + num_fingers
    raise ValueError(f"Unknown retargeting type: {cfg['type']}")


def deterministic_reference(rng: np.random.Generator, cfg: dict[str, Any]) -> np.ndarray:
    count = vector_count(cfg)
    rtype = str(cfg["type"]).lower()
    # Keep inputs in a plausible hand-sized range. DexPilot needs some small
    # inter-finger vectors too so projection branches are exercised.
    ref = rng.normal(loc=0.0, scale=0.035, size=(count, 3)).astype(np.float32)
    if rtype == "position":
        ref += np.array([0.03, 0.0, 0.04], dtype=np.float32)
    elif rtype == "dexpilot":
        num_fingers = len(cfg["finger_tip_link_names"])
        len_proj = num_fingers * (num_fingers - 1) // 2
        if len_proj:
            ref[:len_proj] *= 0.6
        ref[len_proj:] += np.array([0.02, 0.0, 0.04], dtype=np.float32)
    return ref


def deterministic_feasible_reference(
    rng: np.random.Generator, cfg: dict[str, Any], py_retargeting: Any
) -> np.ndarray:
    """Generate a deterministic target by forwarding a reachable robot qpos.

    The optimizer can have multiple joint-space solutions for one cartesian hand
    target. Comparing qpos for arbitrary random cartesian targets therefore mixes
    implementation drift with IK non-uniqueness. This probe samples a valid robot
    posture near the default warm start, forwards it through the reference
    kinematics, then asks both implementations to retarget that reachable task.
    """

    optimizer = py_retargeting.optimizer
    robot = optimizer.robot
    idx_target = np.asarray(optimizer.idx_pin2target, dtype=int)
    idx_fixed = np.asarray(optimizer.idx_pin2fixed, dtype=int)
    full_limits = np.asarray(robot.joint_limits, dtype=np.float64)
    target_limits = full_limits[idx_target]
    base = np.asarray(py_retargeting.last_qpos, dtype=np.float64)
    # Keep the sample close to the default state so the regularized IK optimum is
    # well-conditioned and not an arbitrary alternative solution.
    span = np.minimum(np.maximum(target_limits[:, 1] - target_limits[:, 0], 1e-3), 0.8)
    sample = base + rng.normal(loc=0.0, scale=0.18, size=base.shape) * span
    sample = np.clip(sample, target_limits[:, 0] + 1e-3, target_limits[:, 1] - 1e-3)

    qpos = np.zeros(robot.dof, dtype=np.float64)
    if idx_fixed.size:
        qpos[idx_fixed] = 0.0
    qpos[idx_target] = sample
    if getattr(optimizer, "adaptor", None) is not None:
        qpos = optimizer.adaptor.forward_qpos(qpos)

    robot.compute_forward_kinematics(qpos)
    rtype = str(cfg["type"]).lower()
    if rtype == "position":
        indices = [robot.get_link_index(name) for name in cfg["target_link_names"]]
        return np.asarray([robot.get_link_pose(index)[:3, 3] for index in indices], dtype=np.float32)

    if rtype == "vector":
        origin_names = cfg["target_origin_link_names"]
        task_names = cfg["target_task_link_names"]
    elif rtype == "dexpilot":
        origin_names = optimizer.origin_link_names
        task_names = optimizer.task_link_names
    else:
        raise ValueError(f"Unknown retargeting type: {cfg['type']}")

    origin = np.asarray(
        [robot.get_link_pose(robot.get_link_index(name))[:3, 3] for name in origin_names],
        dtype=np.float64,
    )
    task = np.asarray(
        [robot.get_link_pose(robot.get_link_index(name))[:3, 3] for name in task_names],
        dtype=np.float64,
    )
    scaling = float(cfg.get("scaling_factor", 1.0))
    return ((task - origin) / scaling).astype(np.float32)


def materialize_absolute_config(path: Path, tmp_dir: Path) -> tuple[Path, dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    cfg = data["retargeting"]
    urdf_path = Path(cfg["urdf_path"])
    if not urdf_path.is_absolute():
        cfg["urdf_path"] = str((REFERENCE_URDF_DIR / urdf_path).resolve())

    out_path = tmp_dir / path.parent.name / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return out_path, cfg


def compare_one(
    path: Path,
    tmp_dir: Path,
    seed: int,
    input_mode: str,
    py_config_cls: Any,
    dexi_py: Any,
) -> ComparisonRow:
    rel = path.relative_to(ROOT).as_posix()
    runtime_path, cfg = materialize_absolute_config(path, tmp_dir)
    rtype = str(cfg["type"]).lower()

    try:
        py_config = py_config_cls.load_from_file(runtime_path)
        py_retargeting = py_config.build()
        rust_config = dexi_py.RetargetingConfig.from_file(str(runtime_path))
        rust_retargeting = rust_config.build()

        py_fixed_dof = len(py_retargeting.optimizer.idx_pin2fixed)
        rust_fixed_dof = int(rust_retargeting.fixed_dof)
        if py_fixed_dof != rust_fixed_dof:
            raise ValueError(f"fixed DOF mismatch: python={py_fixed_dof}, rust={rust_fixed_dof}")

        digest = hashlib.sha256(f"{seed}:{rel}".encode()).digest()
        config_seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        rng = np.random.default_rng(config_seed)
        if input_mode == "feasible":
            ref_value = deterministic_feasible_reference(rng, cfg, py_retargeting)
        else:
            ref_value = deterministic_reference(rng, cfg)
        fixed_qpos = np.zeros(py_fixed_dof, dtype=np.float32)

        py_qpos = np.asarray(py_retargeting.retarget(ref_value, fixed_qpos), dtype=np.float64)
        rust_qpos = np.asarray(
            rust_retargeting.retarget(ref_value.reshape(-1).tolist(), fixed_qpos.tolist()),
            dtype=np.float64,
        )

        if py_qpos.shape != rust_qpos.shape:
            raise ValueError(f"qpos shape mismatch: python={py_qpos.shape}, rust={rust_qpos.shape}")
        if not np.all(np.isfinite(py_qpos)) or not np.all(np.isfinite(rust_qpos)):
            raise ValueError("non-finite qpos returned")

        abs_err = np.abs(py_qpos - rust_qpos)
        return ComparisonRow(
            config=rel,
            hand=hand_name(path),
            retargeting_type=rtype,
            dof=int(py_qpos.size),
            fixed_dof=py_fixed_dof,
            max_abs_error=float(abs_err.max(initial=0.0)),
            mean_abs_error=float(abs_err.mean() if abs_err.size else 0.0),
            rms_error=float(math.sqrt(float(np.mean(abs_err * abs_err))) if abs_err.size else 0.0),
            python_norm=float(np.linalg.norm(py_qpos)),
            rust_norm=float(np.linalg.norm(rust_qpos)),
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001 - report every config independently
        return ComparisonRow(
            config=rel,
            hand=hand_name(path),
            retargeting_type=rtype,
            dof=0,
            fixed_dof=0,
            max_abs_error=float("nan"),
            mean_abs_error=float("nan"),
            rms_error=float("nan"),
            python_norm=float("nan"),
            rust_norm=float("nan"),
            status="error",
            message=str(exc),
        )


def write_outputs(rows: list[ComparisonRow], json_path: Path | None, csv_path: Path | None) -> None:
    records = [row.__dict__ for row in rows]
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(records, indent=2, allow_nan=True) + "\n")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(records[0].keys()) if records else [],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-mean-error", type=float, default=5e-2)
    parser.add_argument("--max-max-error", type=float, default=5e-1)
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument(
        "--input-mode",
        choices=["feasible", "random"],
        default="feasible",
        help="Use reachable FK-derived probes by default; random keeps the older arbitrary cartesian probe.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV report path")
    parser.add_argument("configs", nargs="*", type=Path, help="Optional specific config paths")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    py_config_cls, dexi_py = import_packages()
    py_config_cls.set_default_urdf_dir(REFERENCE_URDF_DIR)

    paths = args.configs or config_paths()
    print("config,type,hand,dof,fixed_dof,mean_abs,max_abs,rms,status")
    with tempfile.TemporaryDirectory(prefix="dexi-compare-configs-") as tmp:
        tmp_dir = Path(tmp)
        rows = []
        for path in paths:
            row = compare_one(
                path.resolve(), tmp_dir, args.seed, args.input_mode, py_config_cls, dexi_py
            )
            rows.append(row)
            print(
                f"{row.config},{row.retargeting_type},{row.hand},{row.dof},{row.fixed_dof},"
                f"{row.mean_abs_error:.6g},{row.max_abs_error:.6g},{row.rms_error:.6g},"
                f"{row.status}{(': ' + row.message) if row.message else ''}",
                flush=True,
            )

    ok_rows = [row for row in rows if row.status == "ok"]
    error_rows = [row for row in rows if row.status != "ok"]
    over_threshold = [
        row
        for row in ok_rows
        if row.mean_abs_error > args.max_mean_error or row.max_abs_error > args.max_max_error
    ]

    if ok_rows:
        print(
            "\nSummary: "
            f"{len(ok_rows)}/{len(rows)} configs compared; "
            f"mean_abs max={max(r.mean_abs_error for r in ok_rows):.6g}, "
            f"max_abs max={max(r.max_abs_error for r in ok_rows):.6g}, "
            f"threshold_exceeded={len(over_threshold)}"
        )
    if error_rows:
        print(f"Errors: {len(error_rows)}")
        for row in error_rows:
            print(f"  {row.config}: {row.message}")

    write_outputs(rows, args.json, args.csv)

    if error_rows:
        return 1
    if args.fail_on_threshold and over_threshold:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
