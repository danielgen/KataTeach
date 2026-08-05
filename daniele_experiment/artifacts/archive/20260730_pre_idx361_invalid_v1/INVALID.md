# Invalid experiment artifacts — do not use

Status: `invalid_do_not_use`

These files are preserved for auditability only. They must not be used directly
in analysis, figures, prose, or model selection. The sole permitted reuse is an
explicit, hash-checked migration of whitelisted non-central board-analysis
fields by `build_validated_labels.py`; that migration discards every legacy
tenuki, forcing, urgency, feature, probe, and causal result and records the
copied field names in the new run manifest. See `manifest.json` for exact
provenance, checksums, and failure reasons. Raw games and activations were not
moved and remain the inputs to the corrected rebuild.
