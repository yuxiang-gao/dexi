#!/usr/bin/env python3
"""Install the built wheel in a clean uv venv and smoke-test it."""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def local_wheel() -> Path:
    """Pick the dist/ wheel matching this interpreter and platform.

    dist/ may hold a single native wheel (after `just build`) or the full
    release matrix (after `just build-matrix`); either way only the wheel
    for the running CPython on this machine is installable locally.
    """

    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = {"Darwin": "macosx", "Linux": "manylinux"}.get(platform.system(), "")
    arch = platform.machine().lower()
    wheels = [
        wheel
        for wheel in sorted((ROOT / "dist").glob("dexi_rs-*.whl"))
        if f"-{tag}-" in wheel.name and system in wheel.name and arch in wheel.name
    ]
    if not wheels:
        raise SystemExit(
            f"No dexi-rs wheel for {tag}/{system}/{arch} in dist/; run `just build`"
        )
    return wheels[-1]


def main() -> None:
    wheel = local_wheel()

    with tempfile.TemporaryDirectory(prefix="dexi-wheel-") as tmp:
        venv = Path(tmp) / ".venv"
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run(["uv", "venv", str(venv), "--python", python_version], cwd=ROOT, check=True)
        python = venv / "bin" / "python"
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel), "numpy", "pyyaml", "viser", "yourdfpy"],
            cwd=ROOT,
            check=True,
        )
        code = """
import dexi_rs
import importlib.util
from pathlib import Path
assert importlib.util.find_spec('dexi_py') is None
root = Path.cwd()
config = root / 'configs/teleop/allegro_hand_right.yml'
configs = dexi_rs.available_configs(root / 'configs')
assert len(configs) == 43, len(configs)
path = dexi_rs.config_path(config)
assert path.exists(), path
retargeting = dexi_rs.load_config(config).build()
assert retargeting.joint_names
qpos = retargeting.retarget([0.03,-0.02,0.08,0.04,0.0,0.09,0.03,0.02,0.085,0.02,0.04,0.07])
assert len(qpos) == len(retargeting.joint_names)
fourier = dexi_rs.load_config(root / 'configs/teleop/fourier_hand_right_6dof.yml').build()
assert fourier.joint_names
fourier_qpos = fourier.retarget([0.03,-0.02,0.08,0.04,0.0,0.09,0.03,0.02,0.085,0.02,0.04,0.07,0.01,0.055,0.055])
assert len(fourier_qpos) == len(fourier.joint_names)
import yaml
ref = [0.03,-0.02,0.08,0.04,0.0,0.09,0.03,0.02,0.085,0.02,0.04,0.07]
kwargs = dict(
    type='vector',
    urdf_path='allegro_hand/allegro_hand_right.urdf',
    target_origin_link_names=['wrist'] * 4,
    target_task_link_names=['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip'],
    target_link_human_indices=[[0, 0, 0, 0], [4, 8, 12, 16]],
    scaling_factor=1.6,
    low_pass_alpha=0.2,
)
qpos_yaml = dexi_rs.load_config(config).build().retarget(ref)
qpos_kwargs = dexi_rs.RetargetingConfig(**kwargs).build().retarget(ref)
assert qpos_kwargs == qpos_yaml, 'python-defined config must match YAML config'
data = yaml.safe_load(config.read_text())
qpos_nested = dexi_rs.load_config(data).build().retarget(ref)
assert qpos_nested == qpos_yaml, 'nested dict config must match YAML config'
qpos_flat = dexi_rs.load_config(data['retargeting']).build().retarget(ref)
assert qpos_flat == qpos_yaml, 'flat dict config must match YAML config'
try:
    dexi_rs.RetargetingConfig(type='vector', urdf_path='x.urdf',
        target_origin_link_names=['wrist', 'wrist'],
        target_task_link_names=['link_15.0_tip'],
        target_link_human_indices=[0, 4])
    raise AssertionError('expected ValueError')
except ValueError:
    pass
try:
    dexi_rs.RetargetingConfig(type='dexpilot', urdf_path='x.urdf', scaling=1.6)
    raise AssertionError('expected TypeError')
except TypeError:
    pass
try:
    dexi_rs.RetargetingConfig(type='vector', urdf_path='allegro_hand/allegro_hand_right.urdf',
        target_origin_link_names=['wrist', 'wrist'],
        target_task_link_names=['link_15.0_tip', 'typo_link'],
        target_link_human_indices=[[0, 0], [4, 8]]).build()
    raise AssertionError('expected RuntimeError for unknown link name')
except RuntimeError as exc:
    assert 'typo_link' in str(exc), exc
cfg_obj = dexi_rs.RetargetingConfig(**kwargs)
assert dexi_rs.load_config(cfg_obj) is cfg_obj, 'config objects must pass through load_config'
override = dexi_rs.RetargetingConfig(**kwargs, urdf_dir=str(root / 'assets/robots/hands'))
assert override.build().joint_names, 'urdf_dir override must resolve the URDF'
defaults = dexi_rs.RetargetingConfig(**kwargs)
assert defaults.normal_delta == 4e-3 and defaults.huber_delta == 2e-2
assert defaults.project_dist == 0.03 and defaults.escape_dist == 0.05
assert defaults.has_joint_limits is True and defaults.ignore_mimic_joint is False
for bad_call, needle in [
    (lambda: retargeting.retarget(ref[:-1]), 'ref_value'),
    (lambda: retargeting.retarget(ref, fixed_qpos=[0.0] * (retargeting.fixed_dof + 1)), 'fixed_qpos'),
    (lambda: retargeting.retarget([float('nan')] * len(ref)), 'finite'),
    (lambda: retargeting.set_qpos([0.0]), 'robot_qpos'),
]:
    try:
        bad_call()
        raise AssertionError(f'expected ValueError mentioning {needle}')
    except ValueError as exc:
        assert needle in str(exc), exc
print('wheel smoke test passed')
"""
        subprocess.run([str(python), "-c", code], cwd=ROOT, check=True)

        subprocess.run([str(python), "examples/quickstart.py"], cwd=ROOT, check=True)
        subprocess.run([str(python), "examples/visualize_viser.py", "--smoke-test"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
