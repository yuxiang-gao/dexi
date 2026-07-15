# Python Config API Design

**Date:** 2026-07-15
**Status:** Approved

## Motivation

`dexi_rs.RetargetingConfig` can only be created from a YAML file
(`from_file`). There is no way to define a config from Python, and only 3 of
its ~18 fields are readable. Users need to construct configs from scratch in
Python (primary use case), and `load_config` should accept either a YAML path
or a Python-defined config object.

Vector retargeting itself is already fully supported end-to-end (Rust
`VectorOptimizer` is a faithful port of the reference implementation, YAML
`type: vector` works, and the parity script exercises it). What dexi lacks,
relative to the reference, is the config-layer validation the reference does
in `__post_init__`; this design folds that in.

## Decisions

- **Approach:** native kwargs constructor on the existing pyo3 class (one
  schema, defined by the Rust struct). No Python dataclass mirror; no
  dict-only API.
- **Primary use case:** define configs from scratch in Python, no YAML file.
- **Immutability:** configs are immutable. All fields get getters; no
  setters. Sweeps construct fresh objects.
- **Relative URDF paths in Python-defined configs:** resolve against the
  wheel's bundled assets first, then cwd.
- **Kwarg naming:** `type` (matches the YAML key used by existing configs).

## API surface

### Rust core (`crates/dexi/src/retargeting_config.rs`)

- New `RetargetingConfig::validate(&self) -> Result<(), String>`:
  - `urdf_path` must be non-empty.
  - `position` requires `target_link_names` + `target_link_human_indices`;
    indices length must equal link count.
  - `vector` requires `target_origin_link_names` + `target_task_link_names`
    (equal length N) + `target_link_human_indices` (flattened `2 x N`).
  - `dexpilot` requires `finger_tip_link_names` + `wrist_link_name`.
- Called from YAML parsing (`from_value`) and at the top of `build()`.
  `build()`'s current ad-hoc `ok_or` required-field checks collapse into it.

### pyo3 layer (`crates/dexi-py/src/lib.rs`)

- `#[new]` constructor with keyword arguments for every config field:
  - `type` (required), `urdf_path` (required).
  - All other fields optional, defaulting exactly to the Rust `Default`
    values.
  - Optional `urdf_dir=` anchoring relative URDF resolution; defaults to the
    packaged hands directory (see below).
  - Runs `validate()` immediately: a bad config fails at construction.
- Getters for all config fields; `__repr__` for debuggability. The existing
  `type_` getter keeps its name (constructor kwarg is `type`).
- `from_file` unchanged.

### Python wrapper (`crates/dexi-py/python/dexi_rs/__init__.py`)

- `load_config(source)` accepts:
  - `str | Path` — YAML file path (unchanged behavior);
  - `RetargetingConfig` — returned as-is;
  - `dict` — `RetargetingConfig(**d)`, also accepting the
    `{"retargeting": {...}}` nesting so a YAML file slurped into a dict
    round-trips.
- `_native.pyi` stub ships in the wheel for IDE autocomplete.

## Data flow

```
YAML file ──parse──▶ RetargetingConfig ──validate()──▶ build() ─▶ SeqRetargeting
Python kwargs/dict ─▶ RetargetingConfig ──validate()──▶ build() ─▶ SeqRetargeting
```

## URDF path resolution

For a **relative** `urdf_path`:

1. `urdf_dir/<path>` — for Python-defined configs `urdf_dir` defaults to the
   packaged hands directory, discovered from the native module's installed
   location (`<dexi_rs package>/resources/assets/robots/hands`); for
   YAML-loaded configs it remains the config file's directory.
2. Existing ancestor search for `assets/robots/hands/<path>`.
3. New last-resort: `cwd/<path>`.

Absolute paths bypass resolution entirely (unchanged). Error messages keep
the "tried: ..." list of every attempted candidate.

## Error handling

- Constructor and YAML parsing raise `ValueError` with field-specific
  messages naming the offending field and expected vs. actual shape.
- Unknown constructor kwargs raise `TypeError` (pyo3 explicit signature).
- `build()` failures remain `RuntimeError`; `validate()` re-runs there as a
  cheap backstop.

## Testing

- **Rust unit tests** (`retargeting_config.rs`): `validate()` accept/reject
  cases per type (missing fields, dim mismatches, empty urdf_path), plus
  cwd-fallback resolution.
- **Python wheel smoke** (`scripts/test_wheel.py`): construct a vector config
  from kwargs and from dicts (flat and nested), `build()`, retarget, and
  assert qpos matches the YAML-loaded equivalent bit-for-bit.
- **New example** `examples/python_config.py`, wired into `just examples` and
  preflight.
- Docs: `docs/api.md` section + CHANGELOG entry.

## Out of scope

- Mutating loaded configs (setters).
- Accepting YAML strings (as opposed to file paths) in `load_config`.
- A `retargeting_type` YAML key alias.
