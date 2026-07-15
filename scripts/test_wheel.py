#!/usr/bin/env python3
"""Install the built wheel in a clean uv venv and smoke-test it."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    wheels = sorted((ROOT / "dist").glob("dexi_rs-*.whl"))
    if not wheels:
        raise SystemExit("No dexi-rs wheel found in dist/")
    wheel = wheels[-1]

    with tempfile.TemporaryDirectory(prefix="dexi-wheel-") as tmp:
        venv = Path(tmp) / ".venv"
        subprocess.run(["uv", "venv", str(venv)], cwd=ROOT, check=True)
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
print('wheel smoke test passed')
"""
        subprocess.run([str(python), "-c", code], cwd=ROOT, check=True)

        subprocess.run([str(python), "examples/quickstart.py"], cwd=ROOT, check=True)
        subprocess.run([str(python), "examples/visualize_viser.py", "--smoke-test"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
