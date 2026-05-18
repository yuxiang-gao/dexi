# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-rs", "numpy"]
# ///
"""Run one deterministic frame through several config families."""

from __future__ import annotations

import numpy as np

import dexi_rs


def target_for(config_name: str) -> list[float]:
    if config_name.startswith("offline/"):
        # Allegro position configs use eight 3D targets: four tips plus four
        # intermediate links.
        return np.array(
            [
                [0.00, -0.04, 0.08],
                [0.03, -0.02, 0.09],
                [0.04, 0.00, 0.095],
                [0.03, 0.02, 0.09],
                [0.02, 0.04, 0.08],
                [0.02, -0.02, 0.055],
                [0.025, 0.00, 0.06],
                [0.02, 0.02, 0.055],
            ],
            dtype=float,
        ).reshape(-1).tolist()
    if config_name.endswith("_dexpilot.yml"):
        # Allegro DexPilot has C(4,2)+4 vectors.
        return np.tile(np.array([[0.025, 0.01, 0.075]], dtype=float), (10, 1)).reshape(-1).tolist()
    return np.array(
        [[0.03, -0.02, 0.08], [0.04, 0.0, 0.09], [0.03, 0.02, 0.085], [0.02, 0.04, 0.07]],
        dtype=float,
    ).reshape(-1).tolist()


def main() -> None:
    configs = [
        "offline/allegro_hand_right.yml",
        "teleop/allegro_hand_right.yml",
        "teleop/allegro_hand_right_dexpilot.yml",
    ]
    for name in configs:
        retargeting = dexi_rs.load_config(name).build()
        fixed = [0.0] * retargeting.fixed_dof
        qpos = retargeting.retarget(target_for(name), fixed_qpos=fixed)
        print(f"{name:42s} dof={len(qpos):2d} target_dof={retargeting.target_dof:2d}")


if __name__ == "__main__":
    main()
