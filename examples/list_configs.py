# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-rs"]
# ///
"""List packaged dexi configs."""

from __future__ import annotations

import dexi_rs


def main() -> None:
    for name in dexi_rs.available_configs():
        print(name)


if __name__ == "__main__":
    main()
