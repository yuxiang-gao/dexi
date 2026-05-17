# Third-party notices

Dexi bundles robot-hand YAML configs and URDF files so examples and installed
wheels work without extra downloads.

The bundled assets are derived from upstream open-source hand-retargeting robot
descriptions and per-hand license files are preserved next to the copied URDFs
where available:

- `assets/robots/hands/ability_hand/LICENSE.txt`
- `assets/robots/hands/allegro_hand/LICENSE`
- `assets/robots/hands/inspire_hand/LICENSE.txt`
- `assets/robots/hands/leap_hand/LICENSE.txt`
- `assets/robots/hands/schunk_hand/LICENSE`
- `assets/robots/hands/shadow_hand/LICENSE`
- The Panda gripper URDF is included from the same upstream asset set; no
  separate adjacent Panda license file was present in that asset tree.

Only the YAML configs and URDF files needed by the shipped examples/configs are
included. Mesh directories are intentionally not packaged because the Rust core
parses kinematics, limits, and mimic tags from URDFs and the user-facing
visualization examples render 3D point clouds rather than robot meshes.
