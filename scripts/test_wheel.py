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
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel), "numpy", "viser"], cwd=ROOT, check=True)
        code = """
import dexi_py
configs = dexi_py.available_configs()
assert len(configs) == 39, len(configs)
path = dexi_py.config_path('teleop/allegro_hand_right.yml')
assert path.exists(), path
retargeting = dexi_py.load_config('teleop/allegro_hand_right.yml').build()
assert retargeting.joint_names
qpos = retargeting.retarget([0.03,-0.02,0.08,0.04,0.0,0.09,0.03,0.02,0.085,0.02,0.04,0.07])
assert len(qpos) == len(retargeting.joint_names)
print('wheel smoke test passed')
"""
        subprocess.run([str(python), "-c", code], cwd=ROOT, check=True)

        subprocess.run([str(python), "examples/quickstart.py"], cwd=ROOT, check=True)
        subprocess.run([str(python), "examples/visualize_viser.py", "--smoke-test"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
