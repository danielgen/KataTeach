# Experiment artifacts

Generated and frozen outputs are grouped by lifecycle:

- `protocols/` — registered/frozen experiment protocols.
- `runs/` — active or completed run-scoped datasets, models, and reports.
- `logs/` — command logs grouped by experiment version.
- `exploratory/` — non-confirmatory analysis outputs.
- `causal/` — causal-evaluation outputs that are not stored inside a run.
- `archive/` — immutable invalidated or superseded runs, including provenance and checksums.

Do not rename archived files or run directories: manifests and checksum files use
their current relative paths. New generated outputs should go in the narrowest
matching directory and use a stable run or protocol identifier.
