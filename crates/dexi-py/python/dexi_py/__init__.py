"""Python interface for the dexi hand-retargeting engine.

The Python package wraps the Rust implementation and ships the standard robot
URDF/config resources used by the examples.
"""

from __future__ import annotations

from contextlib import ExitStack
from importlib import metadata, resources as il_resources
from pathlib import Path

from ._native import RetargetingConfig, SeqRetargeting, __version__ as _native_version
from ._native import load_from_file

try:
    __version__ = metadata.version("dexi-py")
except metadata.PackageNotFoundError:  # pragma: no cover - editable/local fallback
    __version__ = _native_version

_RESOURCE_STACK = ExitStack()


def _split_resource_path(path: str | Path) -> tuple[str, ...]:
    text = str(path).replace("\\", "/").strip("/")
    if not text:
        raise ValueError("resource path must not be empty")
    return tuple(part for part in text.split("/") if part)


def _resource_file(package: str, path: str | Path) -> Path:
    node = il_resources.files(package).joinpath(*_split_resource_path(path))
    file_path = Path(_RESOURCE_STACK.enter_context(il_resources.as_file(node)))
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    return file_path


def available_configs(kind: str | None = None) -> list[str]:
    """Return packaged config paths such as ``offline/allegro_hand_left.yml``.

    Parameters
    ----------
    kind:
        Optional config group. Use ``"offline"`` for position retargeting or
        ``"teleop"`` for vector/DexPilot retargeting.
    """

    base = il_resources.files("dexi_py.resources.configs")
    kinds = [kind] if kind else [entry.name for entry in base.iterdir() if entry.is_dir()]

    configs: list[str] = []
    for group in sorted(kinds):
        group_dir = base.joinpath(group)
        if not group_dir.is_dir():
            raise ValueError(f"unknown config group: {group}")
        for entry in group_dir.iterdir():
            if entry.name.endswith((".yml", ".yaml")):
                configs.append(f"{group}/{entry.name}")
    return sorted(configs)


def config_path(name: str | Path) -> Path:
    """Return a filesystem path to a packaged YAML config.

    The returned path stays valid for the life of the Python process, including
    zip-import contexts, because extracted resources are held open internally.
    """

    parts = _split_resource_path(name)
    if parts[0] == "configs":
        parts = parts[1:]
    return _resource_file("dexi_py.resources.configs", Path(*parts))


def asset_path(name: str | Path = "robots/hands") -> Path:
    """Return a filesystem path to a packaged asset resource."""

    parts = _split_resource_path(name)
    if parts[0] == "assets":
        parts = parts[1:]
    return _resource_file("dexi_py.resources.assets", Path(*parts))


def load_config(name: str | Path) -> RetargetingConfig:
    """Load a packaged config by name.

    Example
    -------
    >>> cfg = load_config("teleop/allegro_hand_right.yml")
    >>> retargeting = cfg.build()
    """

    return RetargetingConfig.from_file(str(config_path(name)))


__all__ = [
    "RetargetingConfig",
    "SeqRetargeting",
    "__version__",
    "asset_path",
    "available_configs",
    "config_path",
    "load_config",
    "load_from_file",
]
