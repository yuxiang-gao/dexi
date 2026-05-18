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
included for most hands. The Allegro visual/collision mesh directory is also
bundled because the viser example loads the URDF as a visible hand model and
updates it to the retargeted joint configuration.
