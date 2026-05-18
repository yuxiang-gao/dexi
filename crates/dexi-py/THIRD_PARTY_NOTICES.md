# Third-party notices

Dexi bundles robot-hand URDF and mesh assets so examples and installed wheels
can load visible hand models without extra downloads. YAML configs are kept as
normal repository files under `configs/` and are not bundled into the wheel.

The bundled assets are derived from upstream open-source hand-retargeting robot
descriptions and per-hand license files are preserved next to the copied URDFs
where available:

- `assets/robots/hands/ability_hand/LICENSE.txt`
- `assets/robots/hands/allegro_hand/LICENSE`
- `assets/robots/hands/inspire_hand/LICENSE.txt`
- `assets/robots/hands/leap_hand/LICENSE.txt`
- `assets/robots/hands/schunk_hand/LICENSE`
- `assets/robots/hands/shadow_hand/LICENSE`
- `assets/robots/hands/fourier_hand/LICENSE`
- The Panda gripper URDF is included from the same upstream asset set; no
  separate adjacent Panda license file was present in that asset tree.

Only the URDF files needed by the shipped examples/configs are included for most
hands. The Allegro and Fourier visual/collision mesh directories are also
bundled because the viser examples load those URDFs as visible hand models and
update them to retargeted joint configurations.
