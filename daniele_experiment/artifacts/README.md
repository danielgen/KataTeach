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

## Portability of recorded paths

The tracked exploratory JSON files preserve the absolute checkpoint and run
paths reported by the original commands. These strings document where the
commands ran; consumers should use the adjacent SHA-256 values and
repository-relative artifact identities for provenance. The absolute paths are
not required inputs and are not expected to exist on another machine.

Large run directories, logs, checkpoints, and archived payloads are excluded
from Git. Their absence from a fresh clone is intentional, but means this
repository alone is not a complete data deposit. Availability and checkpoint
limitations are stated in `daniele_experiment/README.md`.
