# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-py"]
# ///
"""List packaged dexi configs."""

from __future__ import annotations

import dexi_py


def main() -> None:
    for name in dexi_py.available_configs():
        print(name)


if __name__ == "__main__":
    main()
