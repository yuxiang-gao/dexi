# FAQ

## What is the Python package name?

Install `dexi-rs`; import `dexi_rs`.

## Why are returned qpos values not in my simulator's order?

Dexi returns qpos in `retargeting.joint_names` order. Build an explicit mapping
from those names to your simulator joints instead of relying on positional order.

## Are mesh assets required?

No for the current core and examples. Dexi parses URDF kinematics, limits, and
mimic tags. The visualization example renders retargeted link points, not robot
meshes.

## Why does `SeqRetargeting` remember state?

Retargeting streams are temporally smooth. The previous target-joint solution is
used as the next optimizer warm start, and configs may apply a low-pass filter.
Call `reset()` when changing streams.

## How do I compare with the Python implementation?

Maintainers can use `scripts/compare_python_rust.py` when the external Python
implementation and its dependencies are available locally. Normal package usage
does not require that comparison environment.
