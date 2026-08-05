# Validated pipeline

This directory provides an organized view of the `validated_*` scripts. The
files are relative links; the root-level paths remain canonical because run
manifests, source hashes, imports, and sibling-path resolution depend on them.

The main flow is:

1. `validated_probe_pipeline.py`
2. `validated_results_report.py`
3. `validated_causal_eval.py`
4. `validated_causal_results_report.py`

`validated_results_report_apfix_v2.py` is the append-only reporting correction.
