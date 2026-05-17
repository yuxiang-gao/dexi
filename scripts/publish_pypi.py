#!/usr/bin/env python3
"""Publish dexi-rs artifacts without printing the PyPI token."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main() -> None:
    load_dotenv()
    token = os.environ.get("PYPI_TOKEN") or os.environ.get("MATURIN_PYPI_TOKEN")
    if not token:
        raise SystemExit("PYPI_TOKEN is not set in .env or the environment")

    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    artifacts = sorted(str(path) for path in (ROOT / "dist").glob("*"))
    if not artifacts:
        raise SystemExit("dist/ is empty; run just build first")
    subprocess.run(
        ["uvx", "twine", "upload", *artifacts],
        cwd=ROOT,
        env=env,
        shell=False,
        check=True,
    )


if __name__ == "__main__":
    main()
