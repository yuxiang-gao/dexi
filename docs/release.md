# Release and publishing

The Python distribution is `dexi-rs`; the version comes from the Cargo package
metadata for `crates/dexi-py` via maturin dynamic metadata.

## Preflight

```bash
just preflight
```

Preflight runs formatting, tests, package builds, distribution checks, wheel
install checks, examples, and release-safety checks.

## Version/tag rule

Publishing requires a clean tree and an exact tag on the current commit:

```bash
git tag v0.1.2
```

The tag must match the Cargo package version exactly.

## Publish

`PYPI_TOKEN` is read from `.env` without printing it.

```bash
just release
```

`just publish` only uploads existing artifacts. `just release` rebuilds and
revalidates first.
