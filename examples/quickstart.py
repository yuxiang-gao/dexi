# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-rs", "numpy"]
# ///
"""Retarget one synthetic Allegro right-hand vector frame."""

from __future__ import annotations

import numpy as np

import dexi_rs


def main() -> None:
    config = "configs/teleop/allegro_hand_right.yml"
    retargeting = dexi_rs.load_config(config).build()
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

    print(f"config: {config}")
    print(f"dof: {len(qpos)}")
    for name, value in zip(retargeting.joint_names, qpos):
        print(f"{name:>28s}: {value: .6f}")


if __name__ == "__main__":
    main()
