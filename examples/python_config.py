# /// script
# requires-python = ">=3.10"
# dependencies = ["dexi-rs>=0.4.0", "numpy"]
# ///
"""Define a retargeting config directly in Python — no YAML file needed.

Relative ``urdf_path`` values resolve against the robot assets bundled in
the wheel, so this runs anywhere the package is installed.
"""

from __future__ import annotations

import numpy as np

import dexi_rs


def main() -> None:
    config = dexi_rs.RetargetingConfig(
        type="vector",
        urdf_path="allegro_hand/allegro_hand_right.urdf",
        target_origin_link_names=["wrist"] * 4,
        target_task_link_names=[
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
        ],
        target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
        scaling_factor=1.6,
    )
    print(config)

    retargeting = config.build()
    target_vectors = np.array(
        [
            [0.03, -0.02, 0.08],
            [0.04, 0.00, 0.09],
            [0.03, 0.02, 0.085],
            [0.02, 0.04, 0.07],
        ]
    )
    qpos = retargeting.retarget(target_vectors.flatten().tolist())
    for name, value in zip(retargeting.joint_names, qpos):
        print(f"{name:>28s}: {value: .6f}")


if __name__ == "__main__":
    main()
