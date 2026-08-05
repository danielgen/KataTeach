# Validity-v5 runbook

This workflow deliberately separates two result namespaces:

- `validity_v5_canonical`: tenuki, forcing proxy, and urgency proxy, all
  recomputed from raw moves/policies under versioned contracts.
- `validity_v5_legacy_exploratory`: the full historical concept set, retrained
  with corrected `idx361` features and nested grouped CV, but clearly labelled
  as depending on legacy-derived snorkel targets.

Neither run reads old probe models, old probe scores, or old causal results.

## Canonical unattended run

From the repository root, run:

```bash
caffeinate -i conda run --no-capture-output -n ml \
  python scripts/run_validity_v5.py canonical
```

The runner performs focused tests, freezes the protocol, generates 150 fresh
games in three CPU shards, rebuilds canonical labels/features, validates the
checkpoint against one saved activation per development game, trains all three
probe representations, runs the held-out tenuki intervention and matched
controls, and writes the causal report plus the original frozen probe report
and its versioned AP-bootstrap correction.

Do not interrupt the fresh-generation stage. A partial prospective cohort is
intentionally not resumed or refilled. If a stage fails, retain the files and
inspect its log under `daniele_experiment/artifacts/logs/validity_v5/`.

The causal replay is pinned to four CPU threads, matching fresh-game
generation; this is required for the strict saved-activation equivalence gate.

Expected local wall time is roughly 2–4 hours, but hardware and probe
convergence can make it longer. It requires no agent/API monitoring.

Canonical outputs are written below:

```text
daniele_experiment/artifacts/runs/validity_v5_canonical/
daniele_experiment/artifacts/runs/validity_v5_canonical/corrections/validated_results_report_apfix_v2/
daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/
```

## Historical exploratory probes

The canonical runner freezes the original 500-game exploratory split before
adding fresh games. After the canonical run—immediately or another day—run:

```bash
caffeinate -i conda run --no-capture-output -n ml \
  python scripts/run_validity_v5.py legacy
```

This may require 8–24 hours of unattended CPU time because it fits every
enabled historical concept in global, local, and combined form across nested
CV. Its output remains separate:

```text
daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/
```

Do not merge its tables with the canonical report or use its concepts in the
headline causal claim.

To run the canonical and legacy-exploratory workflows consecutively in one
unattended command, use:

```bash
caffeinate -i conda run --no-capture-output -n ml \
  python scripts/run_validity_v5.py all
```

This genuinely runs both workflows and may take roughly 10–28 hours in total.

## Results and write-up

Do not copy numerical claims into `project_writeup.md` until the append-only
reports exist and pass their provenance checks. For probe analysis, use
`corrections/validated_results_report_apfix_v2/corrected_results_report.json`.
The prospectively frozen `validated_results_report.json` is retained unchanged
for auditability, but its average-precision bootstrap intervals are superseded:
zero-weight leading score groups were incorrectly propagated as `NaN`. The
versioned correction changes only AP bootstrap records and verifies that every
other result reproduces exactly. Previous pre-`idx361` numerical results remain
archived and are not inputs to either run.
