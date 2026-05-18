# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-rs>=0.3.0"]
# ///
"""List local dexi config files from a source checkout."""

from __future__ import annotations

import dexi_rs


def main() -> None:
    for name in dexi_rs.available_configs("configs"):
        print(name)


if __name__ == "__main__":
    main()
