#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["dexi-rs>=0.3.0", "mediapipe", "numpy", "opencv-python", "pyyaml", "viser", "yourdfpy"]
# ///
"""Retarget a human hand video/webcam stream to a robot hand in viser.

This is a compact tutorial version of the classic vector-retargeting demo:

- read frames from a video or webcam,
- track one MediaPipe hand,
- retarget wrist-to-fingertip vectors with dexi,
- display the live video, tracked human hand, robot URDF, target fingertips,
  actual fingertips, and residuals in one viser scene.

The important tutorial knobs are intentionally exposed:

- ``--hand`` chooses a matching robot URDF/config and MediaPipe hand label.
- ``--tip-order`` maps robot fingertip links to MediaPipe fingertips.
- ``--scale`` overrides the vector scale used before optimization.
- ``camera_landmarks_to_robot_frame()`` is the coordinate transform seam.

Use ``--smoke-test`` for CI/release checks. It skips MediaPipe/camera I/O and
plays a short synthetic hand trajectory through the same retargeting and viewer
update path.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve
from xml.etree import ElementTree

import cv2
import numpy as np
import viser
import yaml
from viser.extras import ViserUrdf

import dexi_rs


CONFIG_PRESETS = {
    "fourier": {
        "left": {
            "6dof": "configs/teleop/fourier_hand_left_6dof.yml",
            "12dof": "configs/teleop/fourier_hand_left_12dof.yml",
        },
        "right": {
            "6dof": "configs/teleop/fourier_hand_right_6dof.yml",
            "12dof": "configs/teleop/fourier_hand_right_12dof.yml",
        },
    },
    "allegro": {
        "left": {"default": "configs/teleop/allegro_hand_left.yml"},
        "right": {"default": "configs/teleop/allegro_hand_right.yml"},
    },
}
REPO_ROOT = Path(__file__).resolve().parents[1]
WRIST_LINK = "wrist"
FINGERTIP_LANDMARKS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
DEFAULT_TIP_ORDER_BY_ROBOT = {
    "fourier": ["thumb", "index", "middle", "ring", "pinky"],
    "allegro": ["thumb", "index", "middle", "ring"],
}
CONVENTION_TRANSFORMS = {
    # Human frame estimated below: +z out of palm, -y to thumb, +x to fingertips.
    # Fourier hand URDF frame: +x to thumb, -y out of palm, +z to wrist.
    "fourier": np.array(
        [
            [0.0, 0.0, -1.0],  # x_fourier = -y_human, y_fourier = -z_human, z_fourier = -x_human
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    ),
    # Allegro demo frame used by the original visualizer: +x out of palm,
    # -y to thumb, +z to fingertips.
    "allegro": np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    ),
}
PALM_LANDMARKS = [0, 5, 9, 13, 17]
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
MARKER_RADIUS = 0.005
HUMAN_OFFSET = np.array([-0.24, 0.0, 0.05], dtype=np.float64)
# Put the camera image on a YZ-oriented panel next to the tracked-hand skeleton.
# Local image x/y axes become world y/z axes, and the image normal points along x.
VIDEO_PANEL_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
VIDEO_PANEL_POSITION = HUMAN_OFFSET + np.array([-0.035, 0.0, 0.08], dtype=np.float64)
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class VectorConfig:
    name: str
    urdf_path: Path
    wrist_link: str
    origin_links: list[str]
    task_links: list[str]
    origin_indices: np.ndarray
    task_indices: np.ndarray
    scaling: float
    tip_order: list[str]
    convention: str

    @property
    def unique_links(self) -> list[str]:
        return list(dict.fromkeys([*self.origin_links, *self.task_links]))


@dataclass(frozen=True)
class FrameSample:
    rgb: np.ndarray
    landmarks: np.ndarray
    image_points: np.ndarray | None
    wrist_frame: np.ndarray


def resolve_urdf_path(config: dexi_rs.RetargetingConfig) -> Path:
    urdf_path = Path(config.urdf_path)
    if urdf_path.is_absolute():
        return urdf_path
    repo_urdf = REPO_ROOT / "assets" / "robots" / "hands" / urdf_path
    if repo_urdf.exists():
        return repo_urdf
    return dexi_rs.asset_path("robots/hands") / urdf_path


def hand_from_config_name(config_name: str) -> str:
    if "_left" in config_name:
        return "left"
    if "_right" in config_name:
        return "right"
    return "right"


def robot_from_config_name(config_name: str) -> str:
    lowered = config_name.lower()
    if "fourier" in lowered:
        return "fourier"
    if "allegro" in lowered:
        return "allegro"
    return "fourier"


def preset_config(robot: str, hand: str, dof: str) -> str:
    options = CONFIG_PRESETS[robot][hand]
    return options.get(dof) or options["default"]


def parse_tip_order(value: str | None, robot: str) -> list[str]:
    if value is None:
        return DEFAULT_TIP_ORDER_BY_ROBOT[robot]
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in FINGERTIP_LANDMARKS]
    if unknown:
        raise ValueError(f"unknown fingertip names in --tip-order: {unknown}")
    return names


def load_vector_config(
    config_name: str,
    tip_order: list[str] | None = None,
    scale: float | None = None,
    convention: str | None = None,
) -> VectorConfig:
    raw = yaml.safe_load(Path(config_name).read_text(encoding="utf-8"))[
        "retargeting"
    ]
    if raw["type"].lower() != "vector":
        raise ValueError(
            f"{config_name} is {raw['type']!r}; this example expects a vector retargeting config"
        )
    indices = np.asarray(raw["target_link_human_indices"], dtype=np.int64)
    if indices.shape[0] != 2:
        raise ValueError(
            "vector config must provide two target_link_human_indices rows"
        )
    if tip_order is None:
        tip_order = parse_tip_order(None, robot_from_config_name(config_name))
    if len(tip_order) != len(raw["target_task_link_names"]):
        raise ValueError(
            f"tip order {tip_order} has {len(tip_order)} entries but config has "
            f"{len(raw['target_task_link_names'])} task links"
        )

    cfg = dexi_rs.load_config(config_name)
    return VectorConfig(
        name=config_name,
        urdf_path=resolve_urdf_path(cfg),
        wrist_link=str(raw["target_origin_link_names"][0]),
        origin_links=list(raw["target_origin_link_names"]),
        task_links=list(raw["target_task_link_names"]),
        origin_indices=indices[0],
        task_indices=np.asarray(
            [FINGERTIP_LANDMARKS[name] for name in tip_order], dtype=np.int64
        ),
        scaling=float(raw.get("scaling_factor", 1.0) if scale is None else scale),
        tip_order=tip_order,
        convention=convention or robot_from_config_name(config_name),
    )


def qpos_in_urdf_order(
    qpos: np.ndarray, dexi_joint_names: list[str], urdf_joint_names: tuple[str, ...]
) -> np.ndarray:
    by_name = dict(zip(dexi_joint_names, qpos))
    missing = [name for name in urdf_joint_names if name not in by_name]
    if missing:
        raise ValueError(f"URDF joints missing from retargeting output: {missing}")
    return np.asarray([by_name[name] for name in urdf_joint_names], dtype=np.float64)


def validate_visual_meshes(urdf_path: Path) -> int:
    root = ElementTree.parse(urdf_path).getroot()
    missing: list[Path] = []
    count = 0
    for mesh in root.findall(".//visual/geometry/mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if filename.startswith("package://"):
            missing.append(mesh_path)
            continue
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        count += 1
        if not mesh_path.exists():
            missing.append(mesh_path)
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:8])
        extra = "" if len(missing) <= 8 else f"\n  ... and {len(missing) - 8} more"
        raise FileNotFoundError(f"URDF visual meshes are missing:\n{preview}{extra}")
    return count


def estimate_human_wrist_frame(points: np.ndarray) -> np.ndarray:
    """Estimate MediaPipe's wrist-local human hand frame.

    The tracked hand frame we observe in the viewer has +z coming out from the
    palm and -y pointing toward the thumb. The robot URDF convention instead has
    +x out from the palm and the same -y-to-thumb direction, so the conversion is
    a 90-degree axis relabeling after this frame estimate.
    """

    x_axis = points[9] - points[0]
    x_axis /= np.linalg.norm(x_axis) + 1e-8

    thumb_side = points[5] - points[17]
    y_axis = -thumb_side
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis) + 1e-8

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def camera_landmarks_to_robot_frame(points: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    """Map MediaPipe world landmarks into dexi's robot-hand vector frame.

    The retargeting configs expect hand vectors in a wrist-local convention, not
    raw camera coordinates. Estimate the wrist frame from the palm, express all
    landmarks in that frame, then rotate into the robot-hand convention used by
    the local vector configs. This removes the visible 90-degree wrist-frame
    offset between the detected hand and the robot hand.
    """

    centered = points - points[0:1]
    human_frame = estimate_human_wrist_frame(centered)
    human_to_robot_axes = CONVENTION_TRANSFORMS[convention]
    robot_frame = human_frame @ human_to_robot_axes
    return (centered @ robot_frame).astype(np.float64), robot_frame


def matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    quat /= np.linalg.norm(quat) + 1e-8
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def synthetic_landmarks(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    phase = frame_idx * 0.22
    wrist = np.array([0.0, 0.0, 0.0])
    tips = np.array(
        [
            [0.035, -0.030, 0.075],
            [0.045, -0.006, 0.095],
            [0.040, 0.020, 0.090],
            [0.030, 0.045, 0.078],
            [0.015, 0.065, 0.060],
        ],
        dtype=np.float64,
    )
    curl = 0.012 * np.sin(phase + np.arange(5) * 0.7)
    tips[:, 2] -= curl
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = wrist
    chains = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    bases = np.array(
        [
            [0.018, -0.018, 0.018],
            [0.018, -0.006, 0.020],
            [0.014, 0.014, 0.020],
            [0.008, 0.035, 0.018],
            [0.0, 0.052, 0.014],
        ]
    )
    for base, tip, chain in zip(bases, tips, chains):
        for step, idx in enumerate(chain, start=1):
            alpha = step / len(chain)
            points[idx] = (1.0 - alpha) * base + alpha * tip
    return points, np.eye(3)


def draw_landmarks(image: np.ndarray, normalized_xy: np.ndarray | None) -> np.ndarray:
    if normalized_xy is None:
        return image
    out = image.copy()
    h, w = out.shape[:2]
    pixels = np.column_stack([normalized_xy[:, 0] * w, normalized_xy[:, 1] * h]).astype(
        int
    )
    for a, b in HAND_CONNECTIONS:
        cv2.line(
            out, tuple(pixels[a]), tuple(pixels[b]), (80, 220, 255), 2, cv2.LINE_AA
        )
    for point in pixels:
        cv2.circle(out, tuple(point), 4, (255, 80, 80), -1, cv2.LINE_AA)
    return out


def video_panel_image(image: np.ndarray, normalized_xy: np.ndarray | None) -> np.ndarray:
    """Draw image-space landmarks and orient the texture for the YZ panel.

    Viser's image plane texture origin makes the YZ-mounted video appear upside
    down with the panel rotation below, so flip the composed image vertically
    before uploading it to the scene.
    """

    return np.flipud(draw_landmarks(image, normalized_xy))


class HandTracker:
    def __init__(self, hand: str, selfie: bool, convention: str) -> None:
        import mediapipe as mp

        self._mp = mp
        self._hand = hand.lower()
        self._selfie = selfie
        self._convention = convention
        self._timestamp_ms = 0

        # MediaPipe 0.10.x on newer Python versions exposes only the Tasks API;
        # older wheels expose ``mp.solutions.hands``. Support both so the example
        # works across uv-managed environments.
        if hasattr(mp, "solutions"):
            self._mode = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self._landmarker = None
        else:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            model_path = ensure_hand_landmarker_model()
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self._mode = "tasks"
            self._hands = None
            self._landmarker = vision.HandLandmarker.create_from_options(options)

    def _desired_label(self) -> str:
        # Keep the detected human hand and the loaded robot URDF on the same
        # side. Older MediaPipe examples sometimes flipped handedness for selfie
        # streams, but that made this viewer show a right robot hand with a left
        # tracked hand for ordinary video files. If a webcam feed is mirrored,
        # pass the matching ``--hand`` explicitly instead of auto-swapping here.
        _ = self._selfie
        return self._hand

    def detect(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self._mode == "tasks":
            return self._detect_with_tasks(rgb)
        assert self._hands is not None
        return self._detect_with_solutions(rgb)

    def _detect_with_solutions(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        hands = self._hands
        assert hands is not None
        results = hands.process(rgb)
        if not results.multi_hand_world_landmarks:
            return None, None, None
        selected = 0
        if results.multi_handedness:
            desired = self._desired_label()
            for idx, handedness in enumerate(results.multi_handedness):
                label = handedness.classification[0].label.lower()
                if label == desired:
                    selected = idx
                    break
        world = results.multi_hand_world_landmarks[selected].landmark
        image = results.multi_hand_landmarks[selected].landmark
        world_points = np.asarray([[p.x, p.y, p.z] for p in world], dtype=np.float64)
        image_points = np.asarray([[p.x, p.y] for p in image], dtype=np.float64)
        landmarks, wrist_frame = camera_landmarks_to_robot_frame(
            world_points, self._convention
        )
        return landmarks, image_points, wrist_frame

    def _detect_with_tasks(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        assert self._landmarker is not None
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.hand_world_landmarks:
            return None, None, None
        selected = 0
        if result.handedness:
            desired = self._desired_label()
            for idx, handedness in enumerate(result.handedness):
                label = handedness[0].category_name.lower()
                if label == desired:
                    selected = idx
                    break
        world = result.hand_world_landmarks[selected]
        image_points_raw = result.hand_landmarks[selected]
        world_points = np.asarray([[p.x, p.y, p.z] for p in world], dtype=np.float64)
        image_points = np.asarray(
            [[p.x, p.y] for p in image_points_raw], dtype=np.float64
        )
        landmarks, wrist_frame = camera_landmarks_to_robot_frame(
            world_points, self._convention
        )
        return landmarks, image_points, wrist_frame


def ensure_hand_landmarker_model() -> Path:
    cache_dir = Path.home() / ".cache" / "dexi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "hand_landmarker.task"
    if not model_path.exists():
        print(f"Downloading MediaPipe hand landmarker model to {model_path}")
        urlretrieve(HAND_LANDMARKER_URL, model_path)
    return model_path


class ViserRetargetingScene:
    def __init__(self, args: argparse.Namespace, vector_cfg: VectorConfig) -> None:
        self.vector_cfg = vector_cfg
        self.config = dexi_rs.load_config(vector_cfg.name)
        self.retargeting = self.config.build()

        if args.no_meshes:
            self.load_meshes = False
            self.mesh_count = 0
        else:
            self.mesh_count = validate_visual_meshes(vector_cfg.urdf_path)
            self.load_meshes = True

        self.server = viser.ViserServer(
            host=args.host,
            port=(0 if args.smoke_test else args.port),
            verbose=not args.smoke_test,
        )
        self.server.scene.add_frame("/world", show_axes=True, axes_length=0.05)
        self.server.scene.add_grid(
            "/ground", width=0.5, height=0.5, cell_size=0.025, plane="xy"
        )
        self.server.scene.add_frame("/retargeted_hand", show_axes=False)
        self.urdf = ViserUrdf(
            self.server,
            vector_cfg.urdf_path,
            root_node_name="/retargeted_hand",
            mesh_color_override=None,
            load_meshes=self.load_meshes,
            load_collision_meshes=False,
        )
        self.urdf_joint_names = self.urdf.get_actuated_joint_names()
        self.video = self.server.scene.add_image(
            "/input/video",
            np.zeros((240, 320, 3), dtype=np.uint8),
            render_width=0.22,
            render_height=0.165,
            position=VIDEO_PANEL_POSITION,
            wxyz=matrix_to_wxyz(VIDEO_PANEL_ROTATION),
        )
        self.human_markers = [
            self.server.scene.add_icosphere(
                f"/tracked_human/landmarks/{idx}",
                radius=0.0035,
                color=(80, 170, 255),
                subdivisions=1,
            )
            for idx in range(21)
        ]
        self.target_markers = [
            self.server.scene.add_icosphere(
                f"/targets/fingertips/{idx}",
                radius=MARKER_RADIUS,
                color=(255, 80, 80),
                subdivisions=2,
                material="toon3",
            )
            for idx in range(len(vector_cfg.task_links))
        ]
        self.actual_markers = [
            self.server.scene.add_icosphere(
                f"/retargeted/fingertips/{idx}",
                radius=MARKER_RADIUS,
                color=(80, 240, 160),
                subdivisions=2,
                material="toon3",
            )
            for idx in range(len(vector_cfg.task_links))
        ]
        self.wrist_axis = self.server.scene.add_frame(
            "/targets/wrist_axis",
            show_axes=True,
            axes_length=0.04,
            axes_radius=0.002,
            origin_radius=0.006,
            origin_color=(255, 220, 80),
        )
        self.human_wrist_axis = self.server.scene.add_frame(
            "/tracked_human/wrist_axis",
            show_axes=True,
            axes_length=0.035,
            axes_radius=0.0015,
            origin_radius=0.004,
            origin_color=(80, 170, 255),
        )

    def update(
        self,
        rgb: np.ndarray,
        landmarks: np.ndarray,
        image_points: np.ndarray | None,
        wrist_frame: np.ndarray,
    ) -> float:
        ref_vectors = (
            landmarks[self.vector_cfg.task_indices]
            - landmarks[self.vector_cfg.origin_indices]
        )
        qpos = np.asarray(
            self.retargeting.retarget(ref_vectors.reshape(-1).tolist()),
            dtype=np.float64,
        )
        self.urdf.update_cfg(
            qpos_in_urdf_order(
                qpos, list(self.retargeting.joint_names), self.urdf_joint_names
            )
        )

        link_positions = np.asarray(
            self.retargeting.link_positions(
                qpos.tolist(), self.vector_cfg.unique_links
            ),
            dtype=np.float32,
        )
        by_link = dict(zip(self.vector_cfg.unique_links, link_positions))
        actual = np.asarray(
            [by_link[name] for name in self.vector_cfg.task_links], dtype=np.float32
        )
        origins = np.asarray(
            [by_link[name] for name in self.vector_cfg.origin_links], dtype=np.float32
        )
        targets = origins + ref_vectors.astype(np.float32) * self.vector_cfg.scaling

        for marker, point in zip(self.human_markers, landmarks):
            marker.position = tuple((point + HUMAN_OFFSET).tolist())
        for marker, point in zip(self.target_markers, targets):
            marker.position = tuple(point.tolist())
        for marker, point in zip(self.actual_markers, actual):
            marker.position = tuple(point.tolist())
        wrist = by_link.get(self.vector_cfg.wrist_link, origins[0])
        self.wrist_axis.position = tuple(wrist.tolist())
        if self.load_meshes:
            wrist_transform = self.urdf._urdf.get_transform(
                self.vector_cfg.wrist_link, self.urdf._urdf.base_link
            )  # noqa: SLF001
            wrist_wxyz = matrix_to_wxyz(wrist_transform[:3, :3])
        else:
            wrist_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.wrist_axis.wxyz = wrist_wxyz
        self.human_wrist_axis.position = tuple((landmarks[0] + HUMAN_OFFSET).tolist())
        # ``wrist_frame`` was used to express MediaPipe landmarks in the robot
        # convention. Draw the tracked wrist with the same displayed orientation
        # as the robot wrist so axis mismatches are immediately visible.
        _ = wrist_frame
        self.human_wrist_axis.wxyz = wrist_wxyz

        human_segments = np.asarray(
            [
                [landmarks[a] + HUMAN_OFFSET, landmarks[b] + HUMAN_OFFSET]
                for a, b in HAND_CONNECTIONS
            ],
            dtype=np.float32,
        )
        self.server.scene.add_line_segments(
            "/tracked_human/skeleton",
            points=human_segments,
            colors=np.full((len(human_segments), 2, 3), [80, 170, 255], dtype=np.uint8),
            line_width=1.5,
        )
        residual_segments = np.stack([actual, targets], axis=1)
        self.server.scene.add_line_segments(
            "/targets/residuals",
            points=residual_segments,
            colors=np.full((len(actual), 2, 3), [255, 170, 80], dtype=np.uint8),
            line_width=2.0,
        )
        self.video.image = video_panel_image(rgb, image_points)
        return float(np.linalg.norm(actual - targets, axis=1).max())

    def stop(self) -> None:
        self.server.stop()


def open_capture(video_path: str | None, webcam: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(webcam if video_path is None else video_path)
    if not capture.isOpened():
        source = f"webcam {webcam}" if video_path is None else video_path
        raise RuntimeError(f"could not open {source}")
    return capture


def synthetic_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.putText(
        frame,
        "synthetic smoke-test hand",
        (42, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (60, 60, 60),
        2,
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iter_synthetic_frames(count: int) -> Iterable[FrameSample]:
    for idx in range(count):
        landmarks, wrist_frame = synthetic_landmarks(idx)
        yield FrameSample(synthetic_frame(), landmarks, None, wrist_frame)


def collect_camera_samples(
    capture: cv2.VideoCapture,
    tracker: HandTracker,
    scene: ViserRetargetingScene,
    max_frames: int,
) -> list[FrameSample]:
    samples: list[FrameSample] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        landmarks, image_points, wrist_frame = tracker.detect(rgb)
        if landmarks is None:
            scene.video.image = rgb
            continue
        assert wrist_frame is not None
        samples.append(FrameSample(rgb, landmarks, image_points, wrist_frame))
        if max_frames and len(samples) >= max_frames:
            break
    return samples


def playback_samples(
    scene: ViserRetargetingScene, samples: list[FrameSample], fps: float
) -> None:
    if not samples:
        raise RuntimeError("no frames with detected hands were collected")

    with scene.server.gui.add_folder("Playback"):
        timestep = scene.server.gui.add_slider(
            "Frame", min=0, max=len(samples) - 1, step=1, initial_value=0
        )
        playing = scene.server.gui.add_checkbox("Playing", True)
        fps_slider = scene.server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=int(max(1, min(60, fps)))
        )
        prev_button = scene.server.gui.add_button("Prev")
        next_button = scene.server.gui.add_button("Next")
    scene.server.gui.add_markdown(
        "Use Playback to scrub the video, detected skeleton, targets, and robot hand together."
    )

    def show_frame(index: int) -> None:
        sample = samples[index]
        scene.update(
            sample.rgb, sample.landmarks, sample.image_points, sample.wrist_frame
        )

    show_frame(0)

    @timestep.on_update
    def _(_) -> None:
        show_frame(int(timestep.value))

    @prev_button.on_click
    def _(_) -> None:
        timestep.value = (int(timestep.value) - 1) % len(samples)

    @next_button.on_click
    def _(_) -> None:
        timestep.value = (int(timestep.value) + 1) % len(samples)

    try:
        while True:
            if playing.value:
                timestep.value = (int(timestep.value) + 1) % len(samples)
            time.sleep(1.0 / float(fps_slider.value))
    except KeyboardInterrupt:
        print("Stopping viewer.")
        scene.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Vector config file to retarget. Defaults to the selected --robot/--hand preset.",
    )
    parser.add_argument("--robot", choices=["fourier", "allegro"], default="fourier")
    parser.add_argument("--dof", choices=["6dof", "12dof"], default="6dof", help="Fourier preset DOF variant")
    parser.add_argument(
        "--video", help="Video file to read. If omitted, the webcam is used."
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=0,
        help="OpenCV webcam index when --video is omitted",
    )
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument(
        "--tip-order",
        help="Comma-separated MediaPipe fingertip names matching the robot task-link order; e.g. thumb,index,middle,ring",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Override vector scaling_factor from the YAML config",
    )
    parser.add_argument(
        "--transform-convention",
        choices=sorted(CONVENTION_TRANSFORMS),
        help="Coordinate convention for detected landmarks. Defaults to --robot or the config family.",
    )
    parser.add_argument(
        "--selfie", action="store_true", help="Use mirrored webcam handedness labels"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames; 0 means run until source ends",
    )
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config_name = args.config or preset_config(args.robot, args.hand, args.dof)
    robot_name = robot_from_config_name(config_name) if args.config else args.robot
    convention = args.transform_convention or robot_name
    tip_order = parse_tip_order(args.tip_order, robot_name)
    vector_cfg = load_vector_config(
        config_name, tip_order=tip_order, scale=args.scale, convention=convention
    )
    tracker = None if args.smoke_test else HandTracker(args.hand, args.selfie, vector_cfg.convention)
    scene = ViserRetargetingScene(args, vector_cfg)
    capture: cv2.VideoCapture | None = None
    processed = 0
    max_residual = 0.0
    try:
        if args.smoke_test:
            samples = list(iter_synthetic_frames(args.max_frames or 8))
        elif args.video:
            capture = open_capture(args.video, args.webcam)
            assert tracker is not None
            samples = collect_camera_samples(capture, tracker, scene, args.max_frames)
        else:
            capture = open_capture(None, args.webcam)
            assert capture is not None
            assert tracker is not None
            samples = []
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                landmarks, image_points, wrist_frame = tracker.detect(rgb)
                if landmarks is None:
                    scene.video.image = rgb
                    continue
                assert wrist_frame is not None
                sample = FrameSample(rgb, landmarks, image_points, wrist_frame)
                max_residual = max(
                    max_residual,
                    scene.update(
                        sample.rgb,
                        sample.landmarks,
                        sample.image_points,
                        sample.wrist_frame,
                    ),
                )
                processed += 1
                if args.max_frames and processed >= args.max_frames:
                    break
                time.sleep(1 / 30.0)
    finally:
        if capture is not None:
            capture.release()

    if samples:
        try:
            for sample in samples:
                max_residual = max(
                    max_residual,
                    scene.update(
                        sample.rgb,
                        sample.landmarks,
                        sample.image_points,
                        sample.wrist_frame,
                    ),
                )
                processed += 1
                if args.max_frames and processed >= args.max_frames:
                    break
        except KeyboardInterrupt:
            print("Stopping viewer.")

    if args.smoke_test:
        assert processed > 0
        assert scene.mesh_count > 0 or args.no_meshes
        print(
            f"retargeted {processed} frames with {'meshes' if scene.load_meshes else 'frames-only'} URDF; "
            f"max residual {max_residual:.4f} m"
        )
        scene.stop()
        return

    if args.max_frames:
        print(f"Retargeted {processed} frames; max residual {max_residual:.4f} m")
        scene.stop()
        return

    print(f"Open http://{args.host}:{args.port}")
    print(f"Config: {config_name}")
    print(f"URDF: {vector_cfg.urdf_path}")
    print(f"Robot preset: {robot_name}; transform convention: {vector_cfg.convention}")
    print(
        f"Tip mapping: robot links {vector_cfg.task_links} <- MediaPipe tips {vector_cfg.tip_order}"
    )
    print(f"Vector scale: {vector_cfg.scaling}")
    if args.video:
        print(
            f"Loaded {len(samples)} detected frames. Use the Playback controls to scrub the video and scene."
        )
        playback_samples(scene, samples, fps=30.0)
        return
    print("Press Ctrl+C to stop.")
    try:
        if hasattr(scene.server, "sleep_forever"):
            scene.server.sleep_forever()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viewer.")
        scene.stop()


if __name__ == "__main__":
    main()
