# Examples

Every example is a single uv-runnable Python file in `examples/`.

## List configs

```bash
uv run examples/list_configs.py
```

## Retarget one frame

```bash
uv run examples/quickstart.py
```

## Batch retarget several config types

```bash
uv run examples/batch_retarget.py
```

## Inspect joint order

```bash
uv run examples/joint_order.py
```

The most common integration mistake is assuming a simulator's joint order is the
same as the retargeting output order. Always consume `retargeting.joint_names`.

## Interactive 3D visualization

Smoke-test mode exits immediately and is used by release automation:

```bash
uv run examples/visualize_viser.py --smoke-test
```

Interactive mode starts a local `viser` server:

```bash
uv run examples/visualize_viser.py
```

The example renders wrist/fingertip link positions as a 3D point cloud. It does
not write videos, meshes, or SVG artifacts.
