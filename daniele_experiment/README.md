# Daniele experiment

This package contains the KataGo game-generation, concept-labeling, linear-probe,
and causal-validation experiments. Existing files intentionally remain at the
package root: scripts, imports, run manifests, and archived checksums refer to
these exact paths.

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
- `katago_python.ipynb`, `Untitled.ipynb` — exploratory notebooks.

### Supporting directories

- `docs/` — canonical home of all experiment documentation.
- `tests/` — canonical home of the test suite.
- `validated/` — browsable view of the provenance-sensitive `validated_*` scripts.
- `artifacts/` — generated protocols, runs, logs, exploratory output, and invalidated archives.
- `katago/` — small KataGo parsing package.
- Root-level executable source paths remain canonical for compatibility.

The `validated/` view contains relative symbolic links rather than copies. The
validated pipeline records exact source paths and resolves sibling files when
checking hashes, so physically moving those scripts would invalidate existing
runs. Documentation and tests do not participate in those provenance hashes
and therefore live directly under `docs/` and `tests/`.

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
