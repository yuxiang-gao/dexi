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

The example loads the Fourier 6DOF URDF mesh model by default, applies the retargeted joint
configuration, and overlays the wrist axis, fingertip targets, actual fingertip
positions, and residual lines. It does not write videos or SVG artifacts.

## Human hand video or webcam retargeting

`vector_retarget_viser.py` is the live vector-retargeting demo. It keeps the
pipeline intentionally small: OpenCV reads a video/webcam frame, MediaPipe tracks
one human hand, dexi retargets wrist-to-fingertip vectors, and viser shows the
input video, tracked human skeleton, URDF robot hand, target fingertips, actual
fingertips, and residual lines in one browser scene.

Smoke-test mode uses a synthetic hand trajectory and exits immediately:

```bash
uv run examples/vector_retarget_viser.py --smoke-test
```

Run on a video file:

```bash
uv run examples/vector_retarget_viser.py --video path/to/hand_video.mp4
```

Run from a webcam:

```bash
uv run examples/vector_retarget_viser.py --webcam 0
```

The example avoids rendered videos and SVG output. The visualization is live in
viser, so you can orbit the robot hand while the tracking targets update.
