# Tests

This directory is the canonical home of the experiment tests.

Run the canonical suite from the repository root:

```bash
pytest daniele_experiment
```

`legacy_snorkel_board_positions_checks.py` and
`legacy_quantile_labeling_check.py` preserve pre-refactor expectations for
historical reference. They are not collected as current regression tests:
their API assumptions no longer match the validated implementation, and some
checks require an ignored local game fixture. The current paper-facing test
suite is the set of files named `test_*.py`.
