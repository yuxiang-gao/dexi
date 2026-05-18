# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-rs>=0.3.0", "numpy"]
# ///
"""Retarget one synthetic Allegro right-hand vector frame.

Run this from the repository checkout with ``uv run examples/quickstart.py``.
The wheel bundles robot assets; configs are ordinary YAML files passed in by
filesystem path.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import dexi_rs

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/teleop/allegro_hand_right.yml"


def main() -> None:
    retargeting = dexi_rs.load_config(CONFIG_PATH).build()
    target_vectors = np.array(
        [
            [0.03, -0.02, 0.08],
            [0.04, 0.00, 0.09],
            [0.03, 0.02, 0.085],
            [0.02, 0.04, 0.07],
        ],
        dtype=float,
    )
    qpos = retargeting.retarget(target_vectors.reshape(-1).tolist())

    print(f"config: {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print(f"dof: {len(qpos)}")
    for name, value in zip(retargeting.joint_names, qpos):
        print(f"{name:>28s}: {value: .6f}")


if __name__ == "__main__":
    main()
