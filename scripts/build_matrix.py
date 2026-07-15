#!/usr/bin/env python3
"""Build the full release artifact matrix for dexi-rs.

Produces, in ``dist/``: one sdist plus wheels for CPython 3.10-3.14 on
macOS (arm64 and x86_64) and Linux (x86_64 and aarch64, manylinux2014).
macOS arm64 wheels build against uv-managed native interpreters; the
other targets cross-compile (Linux links against glibc 2.17 via
maturin's zig support).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "crates/dexi-py/Cargo.toml"
PYTHON_SOURCE = ROOT / "crates/dexi-py/python"
DIST = ROOT / "dist"

PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
CROSS_TARGETS = (
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
)
EXPECTED_WHEELS = len(PYTHON_VERSIONS) * (1 + len(CROSS_TARGETS))
MATURIN = ("uvx", "--from", "maturin[zig]", "maturin")


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def check_rustup_targets() -> None:
    installed = subprocess.check_output(
        ["rustup", "target", "list", "--installed"], text=True
    ).split()
    missing = [target for target in CROSS_TARGETS if target not in installed]
    if missing:
        raise SystemExit(
            "missing rustup targets: "
            + " ".join(missing)
            + "\ninstall with: RUSTUP_DIST_SERVER=https://static.rust-lang.org"
            + " rustup target add "
            + " ".join(missing)
        )


def native_interpreter(version: str) -> str:
    try:
        return subprocess.check_output(
            ["uv", "python", "find", version], text=True
        ).strip()
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"CPython {version} not found; install with: uv python install {version}"
        )


def clean() -> None:
    shutil.rmtree(DIST, ignore_errors=True)
    for pycache in PYTHON_SOURCE.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    for pyc in PYTHON_SOURCE.rglob("*.pyc"):
        pyc.unlink()


def main() -> None:
    check_rustup_targets()
    interpreters = {version: native_interpreter(version) for version in PYTHON_VERSIONS}
    clean()

    run([*MATURIN, "sdist", "-m", str(MANIFEST), "--out", str(DIST)])
    for version in PYTHON_VERSIONS:
        run(
            [
                *MATURIN,
                "build",
                "--release",
                "-m",
                str(MANIFEST),
                "--out",
                str(DIST),
                "-i",
                interpreters[version],
            ]
        )
    for target in CROSS_TARGETS:
        zig = ["--zig"] if "linux" in target else []
        for version in PYTHON_VERSIONS:
            run(
                [
                    *MATURIN,
                    "build",
                    "--release",
                    *zig,
                    "--target",
                    target,
                    "-m",
                    str(MANIFEST),
                    "--out",
                    str(DIST),
                    "-i",
                    f"python{version}",
                ]
            )

    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))
    for artifact in wheels + sdists:
        print(artifact.name)
    if len(wheels) != EXPECTED_WHEELS or len(sdists) != 1:
        raise SystemExit(
            f"expected {EXPECTED_WHEELS} wheels + 1 sdist, "
            f"got {len(wheels)} wheels + {len(sdists)} sdists"
        )
    print(f"built {len(wheels)} wheels + 1 sdist", flush=True)


if __name__ == "__main__":
    sys.exit(main())
