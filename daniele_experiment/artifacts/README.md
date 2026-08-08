# Experiment artifacts

Generated and frozen outputs are grouped by lifecycle:

- `protocols/` — registered/frozen experiment protocols; validity v5 is tracked.
- `runs/` — run-scoped outputs. Compact validity-v5 reports and provenance are
  tracked, while large generated payloads remain ignored.
- `logs/` — command logs grouped by experiment version.
- `exploratory/` — non-confirmatory analysis outputs.
- `causal/` — causal-evaluation outputs that are not stored inside a run.
- `archive/` — immutable invalidated or superseded runs, including provenance and checksums.

Do not rename archived files or run directories: manifests and checksum files use
their current relative paths. New generated outputs should go in the narrowest
matching directory and use a stable run or protocol identifier.

## Portability of recorded paths

The exploratory JSON files record the absolute checkpoint and run paths used by
the original commands. Portable provenance is provided by the adjacent SHA-256
values and repository-relative artifact names; the absolute paths are not
required inputs.

The tracked validity-v5 audit files retain the paths recorded by the original
commands, including local absolute paths in execution metadata. Those strings
are provenance, not required checkout locations. Portable identity is provided
by repository-relative paths and recorded SHA-256 values.

Parquet datasets, activations, probe weights, row-level outputs, logs,
checkpoints, and archived payloads are excluded from Git. Data and checkpoint
availability are described in `daniele_experiment/README.md`.
