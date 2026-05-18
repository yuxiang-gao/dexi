"""Python interface for the dexi hand-retargeting engine.

The package wraps the Rust implementation and ships robot assets. YAML configs
are ordinary files: pass their filesystem paths directly to :func:`load_config`.
"""

from __future__ import annotations

from contextlib import ExitStack
from importlib import metadata, resources as il_resources
from pathlib import Path

from ._native import RetargetingConfig, SeqRetargeting, __version__ as _native_version
from ._native import load_from_file

try:
    __version__ = metadata.version("dexi-rs")
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


def available_configs(root: str | Path = "configs", kind: str | None = None) -> list[str]:
    """Return YAML configs found under a filesystem directory.

    ``dexi-rs`` no longer bundles configs in the wheel. This helper is a small
    convenience for source checkouts or projects that keep configs in a local
    ``configs/`` directory.
    """

    base = Path(root).expanduser()
    if kind:
        base = base / kind
    if not base.exists():
        return []

    configs = [path for path in sorted(base.rglob("*")) if path.suffix in {".yml", ".yaml"}]
    if kind:
        return [str(Path(kind) / path.relative_to(base)) for path in configs]
    return [str(path.relative_to(base)) for path in configs]


def config_path(name: str | Path) -> Path:
    """Return an existing filesystem path to a YAML config."""

    path = Path(name).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def asset_path(name: str | Path = "robots/hands") -> Path:
    """Return a filesystem path to a packaged asset resource."""

    parts = _split_resource_path(name)
    if parts[0] == "assets":
        parts = parts[1:]
    return _resource_file("dexi_rs.resources.assets", Path(*parts))


def load_config(name: str | Path) -> RetargetingConfig:
    """Load a YAML config file directly from the filesystem.

    Example
    -------
    >>> cfg = load_config("configs/teleop/allegro_hand_right.yml")
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
