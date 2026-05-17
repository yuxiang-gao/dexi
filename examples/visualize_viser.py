# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-py", "numpy", "viser"]
# ///
"""Visualize retargeted link points with viser.

Use ``--smoke-test`` in CI/release checks; it builds the retargeted point cloud
and exits without starting a blocking server.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import viser

import dexi_py


LINKS = ["wrist", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]


def build_points() -> np.ndarray:
    retargeting = dexi_py.load_config("teleop/allegro_hand_right.yml").build()
    target_vectors = np.array(
        [[0.03, -0.02, 0.08], [0.04, 0.0, 0.09], [0.03, 0.02, 0.085], [0.02, 0.04, 0.07]],
        dtype=float,
    )
    qpos = retargeting.retarget(target_vectors.reshape(-1).tolist())
    return np.asarray(retargeting.link_positions(qpos, LINKS), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    points = build_points()
    colors = np.array(
        [[80, 160, 255], [80, 240, 160], [80, 240, 160], [80, 240, 160], [80, 240, 160]],
        dtype=np.uint8,
    )

    if args.smoke_test:
        print(f"built {len(points)} retargeted 3D points")
        return

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world")
    server.scene.add_grid("/ground", width=0.4, height=0.4, cell_size=0.02)
    server.scene.add_point_cloud(
        name="/retargeted_links",
        points=points,
        colors=colors,
        point_size=0.015,
        point_shape="circle",
        point_shading="gradient",
    )
    print(f"Open http://{args.host}:{args.port}")
    if hasattr(server, "sleep_forever"):
        server.sleep_forever()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
