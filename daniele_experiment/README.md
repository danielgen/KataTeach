# Daniele experiment

This package contains the KataGo game-generation, concept-labeling, linear-probe,
and causal-validation experiments. Scripts used by the validated pipeline remain
at the package root because imports, manifests, and archived checksums refer to
their current paths.

## Start here

| Goal | Entry point | Guide |
| --- | --- | --- |
| Reproduce the validated v5 experiment | `validated_probe_pipeline.py` | [Validity v5 runbook](docs/VALIDITY_V5_RUNBOOK.md) |
| Read the validated results | `validated_results_report.py` | [Experiments and results](docs/VALIDITY_V5_EXPERIMENTS_AND_RESULTS.md) |
| Run the legacy/exploratory probe workflow | `linear_probe_pipeline.py` | [Linear probes workflow](docs/LINEAR_PROBES_WORKFLOW.md) |
| Generate self-play game data | `generate_games_dataset.py` | [Refactoring summary](docs/REFACTORING_SUMMARY.md) |
| Inspect or change concept labels | `snorkel_board_positions.py` | [Snorkel features](docs/SNORKEL_FEATURES_DOCUMENTATION.md) |
| Run activation interventions | `activation_manipulation.py` | [Activation manipulation](docs/ACTIVATION_MANIPULATION.md) |
| Explore tenuki interventions | `tenuki_gradient_analysis.py` | [Tenuki analysis](docs/TENUKI_INTERVENTION_ANALYSIS.md) |
| Use the interactive Go board | `interactive_play_with_ownership.py` | [Interactive board](docs/README_interactive_board.md) |

## File map

### Validated v5 pipeline

- `operational_definitions.py` — versioned definitions shared by labeling and evaluation.
- `build_validated_labels.py` — builds run-scoped labels with explicit timing.
- `validated_probe_pipeline.py` — leakage-resistant training and nested evaluation.
- `validated_results_report.py` — validates provenance and reports probe results.
- `validated_results_report_apfix_v2.py` — append-only correction to the frozen reporter.
- `checkpoint_activation_fidelity.py` — checks saved activations against a checkpoint.
- `validated_causal_eval.py` — held-out causal evaluation.
- `validated_causal_results_report.py` — validates and reports causal results.
- `causal_controls.py` — control construction and provenance helpers.

### Exploratory analysis

- `activation_manipulation.py` — full-game activation steering.
- `position_causal_eval.py` — matched-position steering experiments.
- `tenuki_gradient_analysis.py` — first-order tenuki gradient analysis.
- `tenuki_single_site_analysis.py` — single-location tenuki interventions.

### Data generation and feature labeling

- `generate_games_dataset.py` — current self-play dataset generator.
- `play_games.py`, `play_and_analyze.py`, `policy.py` — earlier game-generation and policy tools.
- `snorkel_board_positions.py` — spatial/ownership feature extraction.
- `common_utils.py` — shared game and model utilities.
- `concepts.yaml` — legacy concept definitions.
- `concepts_validated_v5.yaml` — frozen validated-v5 definitions.

### Probe output and visualization

- `linear_probe_pipeline.py` — legacy/exploratory probe pipeline.
- `integrate_concepts_html.py` — adds concept scores and commentary to game HTML.
- `visualize_katago_outputs_custom.py`, `visualize_ownership_simple.py` — standalone visualizers.
- `interactive_play_with_ownership.py`, `notebook_example_interactive_board.py` — interactive board tools.
- `katago_python.ipynb` — exploratory model-inspection notebook.

### Supporting directories

- `docs/` — experiment documentation.
- `tests/` — current tests and retained legacy checks.
- `artifacts/` — generated protocols, runs, logs, exploratory output, and invalidated archives.
- `katago/` — small KataGo parsing package.
- Root-level executable source paths are retained for compatibility.

The validated scripts remain at the package root because the pipeline records
their exact source paths and resolves sibling files when checking hashes.
Moving them would invalidate existing runs. Documentation and tests are stored
under `docs/` and `tests/` because they are not part of those provenance hashes.

Generated caches such as `__pycache__/`, `.pytest_cache/`, and
`.ipynb_checkpoints/` are not source code and can be ignored while navigating.

## Path-stability convention

Treat the current Python module names and artifact paths as public interfaces.
When adding work:

1. Extend an existing module when it belongs to an established pipeline.
2. Put generated results under `artifacts/`, not beside source files.
3. Give new experiments a descriptive module plus a matching Markdown guide.
4. Do not move frozen validated-v5 producers or edit archived artifact contents;
   add a versioned successor instead.

## Installation and tests

Create an isolated Python environment and install the experiment dependencies:

```bash
python -m pip install -r daniele_experiment/requirements.txt
```

KataGo's Python model code is imported directly from this repository's
`python/` directory. Run the experiment tests from the repository root:

```bash
pytest -q daniele_experiment/tests
```

GPU support is optional for the tests, but full data generation and replay are
computationally expensive. The original experiment used a dedicated Conda
environment named `ml`; that environment name is not a dependency of the code.

## Data and checkpoint availability

The multi-gigabyte game corpus, run directory, logs, and model checkpoint are
excluded from Git. A clone contains the pipeline and results documentation, but
not the data needed to rerun the experiment or inspect row-level artifacts.

The locally preserved checkpoint has:

- repository-relative expected path: `daniele_experiment/model.ckpt`;
- SHA-256: `9476214872d78c80b53605cf5a654004faa7d59b6a743fd5b68942c36dd4ace3`;
- embedded model configuration: version 15, 512 trunk channels, Mish activation;
- original download filename/source URL: not retained.

The checkpoint reproduced one sampled saved activation from each of the 500
development games within the frozen tolerance. This is empirical compatibility
evidence, not proof that it originally generated every historical activation.
The later 150-game prospective cohort was explicitly bound to this hash.

Exact end-to-end reproduction requires the omitted checkpoint and run data.
Numerical claims, limitations, hashes, and provenance relationships are recorded in
[`docs/VALIDITY_V5_EXPERIMENTS_AND_RESULTS.md`](docs/VALIDITY_V5_EXPERIMENTS_AND_RESULTS.md).

Some exploratory JSON records contain absolute paths from the original machine.
They are execution metadata; the adjacent hashes and repository-relative
artifact names provide the portable identifiers.
