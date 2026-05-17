# /// script
# requires-python = ">=3.9"
# dependencies = ["dexi-py"]
# ///
"""Print retargeting output joint order for a config."""

from __future__ import annotations

import argparse

import dexi_py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="teleop/allegro_hand_right.yml")
    args = parser.parse_args()

    retargeting = dexi_py.load_config(args.config).build()
    print(args.config)
    for index, name in enumerate(retargeting.joint_names):
        print(f"{index:02d} {name}")


if __name__ == "__main__":
    main()
