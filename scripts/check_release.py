#!/usr/bin/env python3
"""Release safety checks for dexi-rs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARGO_PACKAGE = "dexi-py"
PYPI_PACKAGE = "dexi-rs"


def run(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def package_version() -> str:
    metadata = json.loads(run(["cargo", "metadata", "--format-version", "1", "--no-deps"]))
    for package in metadata["packages"]:
        if package["name"] == CARGO_PACKAGE:
            return package["version"]
    raise RuntimeError(f"Cargo package {CARGO_PACKAGE!r} not found")


def check_clean_tree() -> None:
    status = run(["git", "status", "--porcelain"])
    if status:
        raise SystemExit("working tree must be clean before publish/release")


def check_tag_matches(version: str) -> None:
    tags = run(["git", "tag", "--points-at", "HEAD"]).splitlines()
    expected = f"v{version}"
    if expected not in tags:
        raise SystemExit(f"current commit must have exact tag {expected}")


def check_pypi_available(version: str) -> None:
    url = f"https://pypi.org/pypi/{PYPI_PACKAGE}/{version}/json"
    try:
        urllib.request.urlopen(url, timeout=10).read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return
        raise
    except urllib.error.URLError:
        print("warning: could not reach PyPI to check version availability", file=sys.stderr)
        return
    raise SystemExit(f"{PYPI_PACKAGE} {version} already exists on PyPI")


def check_forbidden_references() -> None:
    allowed = {Path("scripts/compare_python_rust.py"), Path("scripts/check_release.py")}
    roots = [Path("README.md"), Path("docs"), Path("examples"), Path("scripts")]
    offenders: list[Path] = []
    for root in roots:
        base = ROOT / root
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if rel in allowed or rel.parts[:2] == ("docs", "superpowers"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "reference/" in text or "reference/dex-retargeting" in text:
                offenders.append(rel)
    if offenders:
        raise SystemExit("forbidden reference path mentions: " + ", ".join(map(str, offenders)))


def _dist_members(path: Path) -> list[str]:
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path) as tf:
            return tf.getnames()
    return []


def check_dist_contents() -> None:
    dist = ROOT / "dist"
    artifacts = sorted(dist.glob("*"))
    if not artifacts:
        raise SystemExit("dist/ is empty; run build first")
    banned = ["reference/", ".env", "target/", ".DS_Store", "__pycache__", ".pyc"]
    for artifact in artifacts:
        members = _dist_members(artifact)
        for member in members:
            if any(token in member for token in banned):
                raise SystemExit(f"unexpected file in {artifact.name}: {member}")
        if artifact.stat().st_size > 25 * 1024 * 1024:
            raise SystemExit(f"artifact too large: {artifact.name} ({artifact.stat().st_size} bytes)")


def _tracked_files(root: Path) -> dict[Path, str]:
    files: dict[Path, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if path.name == "__init__.py":
            continue
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def check_resource_drift() -> None:
    pairs = [
        (
            ROOT / "configs",
            ROOT / "crates/dexi-py/python/dexi_py/resources/configs",
            "configs",
        ),
        (
            ROOT / "assets/robots/hands",
            ROOT / "crates/dexi-py/python/dexi_py/resources/assets/robots/hands",
            "robot assets",
        ),
    ]
    for source, packaged, label in pairs:
        if not source.exists() or not packaged.exists():
            raise SystemExit(f"missing {label} resource directory")
        source_files = _tracked_files(source)
        packaged_files = _tracked_files(packaged)
        if source_files != packaged_files:
            source_only = sorted(set(source_files) - set(packaged_files))
            packaged_only = sorted(set(packaged_files) - set(source_files))
            changed = sorted(
                rel
                for rel in set(source_files) & set(packaged_files)
                if source_files[rel] != packaged_files[rel]
            )
            details = []
            if source_only:
                details.append("missing packaged: " + ", ".join(map(str, source_only[:5])))
            if packaged_only:
                details.append("extra packaged: " + ", ".join(map(str, packaged_only[:5])))
            if changed:
                details.append("content drift: " + ", ".join(map(str, changed[:5])))
            raise SystemExit(f"{label} package resources differ from root resources; " + "; ".join(details))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true", help="also require clean tree, exact tag, and free PyPI version")
    parser.add_argument("--dist", action="store_true", help="check built distribution contents")
    args = parser.parse_args()

    version = package_version()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.\-]+)?", version):
        raise SystemExit(f"unexpected Cargo package version: {version}")

    check_forbidden_references()
    check_resource_drift()
    if args.dist:
        check_dist_contents()
    if args.publish:
        check_clean_tree()
        check_tag_matches(version)
        check_pypi_available(version)
    print(f"release checks passed for {PYPI_PACKAGE} {version}")


if __name__ == "__main__":
    main()
