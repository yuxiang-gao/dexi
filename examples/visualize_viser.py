#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-rs>=0.3.0", "numpy", "pyyaml", "viser", "yourdfpy"]
# ///
"""Visualize a retargeted URDF hand model with viser.

The scene shows:

- the URDF hand mesh at the retargeted joint configuration,
- a wrist coordinate axis,
- fingertip target points from the input reference vectors,
- actual retargeted fingertip positions and residual lines.

Use ``--smoke-test`` in CI/release checks; it loads the URDF, applies the
retargeted configuration, creates the overlays, and exits without blocking.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import viser
import yaml
from viser.extras import ViserUrdf

import dexi_rs


DEFAULT_CONFIG = "configs/teleop/fourier_hand_right_6dof.yml"
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_VECTORS = np.array(
    [[0.03, -0.02, 0.08], [0.04, 0.0, 0.09], [0.03, 0.02, 0.085], [0.02, 0.04, 0.07]],
    dtype=np.float64,
)
MARKER_RADIUS = 0.006


@dataclass(frozen=True)
class RetargetedScene:
    config: dexi_rs.RetargetingConfig
    retargeting: dexi_rs.SeqRetargeting
    urdf_path: Path
    wrist_link: str
    fingertip_links: list[str]
    qpos: np.ndarray
    actual_points: np.ndarray
    target_points: np.ndarray


def resolve_urdf_path(config: dexi_rs.RetargetingConfig) -> Path:
    """Resolve the URDF asset referenced by a loaded dexi config."""

    urdf_path = Path(config.urdf_path)
    if urdf_path.is_absolute():
        return urdf_path

    # When this example is run from a source checkout via
    # ``uv run examples/visualize_viser.py``, uv installs ``dexi-rs`` into an
    # isolated script environment. Prefer the checkout assets so the example
    # reflects the files next to the script, including freshly added meshes.
    repo_urdf = REPO_ROOT / "assets" / "robots" / "hands" / urdf_path
    if repo_urdf.exists():
        return repo_urdf

    return dexi_rs.asset_path("robots/hands") / urdf_path


def load_vector_yaml(config_name: str) -> dict:
    """Load the vector-retargeting part of a YAML config file."""

    raw = yaml.safe_load(Path(config_name).read_text(encoding="utf-8"))["retargeting"]
    if raw["type"].lower() != "vector":
        raise ValueError(f"{config_name} is {raw['type']!r}; this visualizer expects a vector config")
    return raw


def build_retargeted_scene(config_name: str) -> RetargetedScene:
    """Build deterministic retargeting outputs for visualization."""

    raw = load_vector_yaml(config_name)
    wrist_link = str(raw["target_origin_link_names"][0])
    fingertip_links = list(raw["target_task_link_names"])
    links = [wrist_link, *fingertip_links]
    config = dexi_rs.load_config(config_name)
    retargeting = config.build()
    qpos = np.asarray(retargeting.retarget(TARGET_VECTORS.reshape(-1).tolist()), dtype=np.float64)

    actual_points = np.asarray(retargeting.link_positions(qpos.tolist(), links), dtype=np.float32)
    wrist_point = actual_points[0]
    target_points = (wrist_point + TARGET_VECTORS.astype(np.float32) * float(raw.get("scaling_factor", 1.0))).astype(
        np.float32
    )

    return RetargetedScene(
        config=config,
        retargeting=retargeting,
        urdf_path=resolve_urdf_path(config),
        wrist_link=wrist_link,
        fingertip_links=fingertip_links,
        qpos=qpos,
        actual_points=actual_points,
        target_points=target_points,
    )


def qpos_in_urdf_order(
    qpos: np.ndarray,
    dexi_joint_names: list[str],
    urdf_joint_names: tuple[str, ...],
) -> np.ndarray:
    """Reorder dexi qpos into ViserUrdf's actuated-joint order."""

    if len(dexi_joint_names) != len(qpos):
        raise ValueError(f"joint/qpos length mismatch: {len(dexi_joint_names)} names vs {len(qpos)} values")
    values_by_name = dict(zip(dexi_joint_names, qpos))
    missing = [name for name in urdf_joint_names if name not in values_by_name]
    if missing:
        raise ValueError(f"URDF joints missing from retargeting output: {missing}")
    return np.asarray([values_by_name[name] for name in urdf_joint_names], dtype=np.float64)


def validate_visual_meshes(urdf_path: Path) -> int:
    """Ensure visual meshes referenced by a URDF are present next to it."""

    root = ElementTree.parse(urdf_path).getroot()
    missing: list[Path] = []
    mesh_count = 0
    for mesh in root.findall(".//visual/geometry/mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            # Shipped example assets use plain relative paths. Keep this example
            # explicit rather than guessing ROS package roots.
            missing.append(Path(filename))
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        mesh_count += 1
        if not mesh_path.exists():
            missing.append(mesh_path)
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:8])
        extra = "" if len(missing) <= 8 else f"\n  ... and {len(missing) - 8} more"
        raise FileNotFoundError(f"URDF visual meshes are missing:\n{preview}{extra}")
    return mesh_count


def matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def add_overlays(
    server: viser.ViserServer,
    actual_points: np.ndarray,
    target_points: np.ndarray,
    wrist_wxyz: tuple[float, float, float, float],
) -> None:
    """Add wrist axis, target points, actual fingertip points, and residuals."""

    wrist = actual_points[0]
    fingertips = actual_points[1:]

    server.scene.add_frame(
        "/targets/wrist_axis",
        position=wrist,
        wxyz=wrist_wxyz,
        show_axes=True,
        axes_length=0.04,
        axes_radius=0.002,
        origin_radius=0.006,
        origin_color=(255, 220, 80),
    )
    for idx, point in enumerate(target_points):
        server.scene.add_icosphere(
            name=f"/targets/fingertip_targets/{idx}",
            radius=MARKER_RADIUS,
            color=(255, 80, 80),
            subdivisions=2,
            material="toon3",
            position=point,
        )
    for idx, point in enumerate(fingertips):
        server.scene.add_icosphere(
            name=f"/retargeted/fingertips/{idx}",
            radius=MARKER_RADIUS,
            color=(80, 240, 160),
            subdivisions=2,
            material="toon3",
            position=point,
        )
    residual_segments = np.stack([fingertips, target_points], axis=1)
    server.scene.add_line_segments(
        name="/targets/residuals",
        points=residual_segments,
        colors=np.full((len(fingertips), 2, 3), [255, 170, 80], dtype=np.uint8),
        line_width=2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Vector config file to visualize")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-meshes", action="store_true", help="Load URDF frames only, without visual meshes")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    scene = build_retargeted_scene(args.config)
    urdf_path = scene.urdf_path
    qpos = scene.qpos
    retargeting = scene.retargeting
    actual_points = scene.actual_points
    target_points = scene.target_points

    if args.no_meshes:
        load_meshes = False
        mesh_count = 0
    else:
        mesh_count = validate_visual_meshes(Path(urdf_path))
        load_meshes = True

    server = viser.ViserServer(host=args.host, port=(0 if args.smoke_test else args.port), verbose=not args.smoke_test)
    server.scene.add_frame("/world", show_axes=True, axes_length=0.05)
    server.scene.add_grid("/ground", width=0.4, height=0.4, cell_size=0.02, plane="xy")
    hand_root = server.scene.add_frame("/retargeted_hand", show_axes=False)
    hand_root.position = (0.0, 0.0, 0.0)

    urdf_vis = ViserUrdf(
        server,
        Path(urdf_path),
        root_node_name="/retargeted_hand",
        mesh_color_override=None,
        load_meshes=load_meshes,
        load_collision_meshes=False,
    )
    urdf_qpos = qpos_in_urdf_order(np.asarray(qpos), list(retargeting.joint_names), urdf_vis.get_actuated_joint_names())
    urdf_vis.update_cfg(urdf_qpos)

    if load_meshes:
        wrist_transform = urdf_vis._urdf.get_transform(scene.wrist_link, urdf_vis._urdf.base_link)  # noqa: SLF001
        wrist_wxyz = matrix_to_wxyz(wrist_transform[:3, :3])
    else:
        wrist_wxyz = (1.0, 0.0, 0.0, 0.0)
    add_overlays(server, np.asarray(actual_points), np.asarray(target_points), wrist_wxyz)

    residual = np.linalg.norm(np.asarray(actual_points)[1:] - np.asarray(target_points), axis=1)
    if args.smoke_test:
        assert len(urdf_vis.get_actuated_joint_names()) == len(urdf_qpos)
        assert len(target_points) == len(scene.fingertip_links)
        if load_meshes:
            assert mesh_count > 0
            assert len(urdf_vis._meshes) > 0  # noqa: SLF001
        print(
            f"loaded URDF hand model ({'meshes' if load_meshes else 'frames-only'}) "
            f"with {len(urdf_qpos)} joints and {mesh_count} visual meshes; "
            f"max target residual {residual.max():.4f} m"
        )
        server.stop()
        return

    print(f"Open http://{args.host}:{args.port}")
    print(f"URDF: {urdf_path}")
    print(f"Mesh rendering: {'on' if load_meshes else 'off (--no-meshes)'}; visual meshes: {mesh_count}")
    try:
        if hasattr(server, "sleep_forever"):
            server.sleep_forever()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viewer.")
        server.stop()


if __name__ == "__main__":
    main()
