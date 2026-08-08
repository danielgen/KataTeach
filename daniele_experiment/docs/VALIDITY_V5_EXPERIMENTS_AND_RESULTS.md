# Validity-v5 Experiments and Results

## Artifact availability

The frozen protocol, corrected probe reports, report manifests, label and
training provenance, checkpoint-fidelity record, causal summaries, and F1/F5
exploratory JSON outputs cited below are tracked in this repository at their
listed paths. Some records retain absolute paths from the machine that ran the
experiment; these are execution metadata, while hashes and repository-relative
names provide portable identity.

Large generated inputs and intermediate outputs—including the game corpus,
Parquet datasets, saved activations, trained probes, row-level predictions,
runtime logs, and `model.ckpt`—are not tracked. The small focused-test record is
included. Accordingly, the repository supports
inspection of the final numerical record but not independent end-to-end rerun
without the separately preserved source games and exact hashed checkpoint.

All paths below are relative to the repository root.

The evidence has three distinct statuses:

1. **Canonical predictive evidence:** three operational variables rebuilt from raw game records and evaluated with nested game-grouped probe cross-validation. Here, “canonical” identifies the rebuilt validity-v5 pipeline; it does not imply that the proxy has been semantically validated as the full human concept.
2. **Confirmatory causal evidence:** one prospectively held-out intervention for the tenuki-distance proxy, assessed against a frozen directional hypothesis and two control families.
3. **Historical exploratory evidence:** 24 configured labels derived from 22 hash-checked source fields migrated from the earlier weak-labelling/Snorkel analysis, then retrained with corrected feature indexing and nested grouped CV.

The principal empirical outcome is also two-part: all three canonical operational labels were highly linearly predictable from `trunkfinal` feature summaries, but the sole valid confirmatory intervention did **not** pass its predeclared support criterion.

## 1. Experimental design

### 1.1 Model and activation site

The experiment used `daniele_experiment/model.ckpt`, selecting the raw rather than stochastic-weight-averaged parameters. The checkpoint SHA-256 was `9476214872d78c80b53605cf5a654004faa7d59b6a743fd5b68942c36dd4ace3`. The loaded version-15 model had 512 trunk channels and Mish activation. The recorded `trunkfinal` tensor was the output of the final trunk normalisation and activation, immediately before the policy and value heads, with shape $512 \times 19 \times 19$.

Checkpoint and replay source: `daniele_experiment/artifacts/runs/validity_v5_canonical/checkpoint_activation_fidelity.json`. Hook implementation: `python/model_pytorch.py` and `daniele_experiment/validated_causal_eval.py`.

The analysis concerned the direct neural policy. Monte Carlo tree search was not run during either game generation or intervention evaluation.

Protocol source: `daniele_experiment/artifacts/protocols/validity_v5.json`.

### 1.2 Games and data partitions

The 500 pre-existing games were assigned only to development: nested grouped probe evaluation and fitting the final intervention direction. Their metadata do not declare the source checkpoint, so their exact original checkpoint provenance cannot be recovered from those declarations. The later 500-position replay audit establishes sampled compatibility with the supplied checkpoint, not original-generation identity. After the protocol was frozen, 150 new games were generated with explicit checkpoint and seed provenance and assigned once to two untouched partitions.

| Partition | Games | Use |
|---|---:|---|
| Development | 500 | Nested grouped probe evaluation and final probe fitting |
| Control calibration | 50 | Selection of control dose multipliers by policy disruption |
| Causal test | 100 | Final held-out intervention and control comparison |

Fresh games were played on a 19-by-19 board configured with `GameState.RULES_TT`. The bundled `Board` legality checker explicitly omits superko and blocks only simple-ko recapture, despite `RULES_TT` declaring positional ko; this is a rules-fidelity limitation of the local implementation. At each turn, moves were sampled from at most the ten highest-probability policy actions whose raw probability was at least 0.01. Probabilities were temperature-transformed and renormalised over this candidate set. Temperature declined from 1.2 to 0.8 over the first 60 moves. Games used a 400-move safety limit and a resignation rule requiring three consecutive win-probability estimates below 0.10. Each game had its own frozen NumPy random seed. These were direct-policy games without search.

Frozen design and generation settings: `daniele_experiment/artifacts/protocols/validity_v5.json`. Generation implementation: `daniele_experiment/generate_games_dataset.py` and `daniele_experiment/common_utils.py`. Rules configuration and legality implementation: `python/gamestate.py` and `python/board.py`. Run membership, source-checkpoint declarations, and split provenance: `daniele_experiment/artifacts/runs/validity_v5_canonical/manifest.json`.

For the causal experiment, one eligible position was selected deterministically from each fresh game, stratified by the observed tenuki-distance label. The calibration set contained 25 negative and 25 positive positions; the test set contained 50 negative and 50 positive positions. Consequently, causal means are equal-label-weighted estimands, not corpus-prevalence estimates.

Selection source: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/selected_positions.parquet`. Estimand definition: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report.json`.

### 1.3 Canonical operational variables

The canonical variables were recomputed rather than copied from the earlier Snorkel output. They are proxies for human Go concepts, not interchangeable with the concepts themselves.

| Report name | Versioned contract | Exact operational definition | Representation time |
|---|---|---|---|
| Tenuki-distance proxy | `tenuki_distance6@2` | The recorded move is at least six Manhattan intersections from the most recent non-pass move. | Pre-move |
| Forcing proxy | `reply_peak95@2` | After the recorded move, the opponent's legal-plus-pass reply policy has maximum probability strictly greater than 0.95. | Pre-move activation; post-move label |
| Urgency proxy | `regional_policy_peak@2` | Maximum mass assigned by the current pre-move legal-board policy to any of nine board regions; the highest 15% are positive. | Pre-move |

For the urgency proxy, quantile cut-offs were fitted using only the relevant training games at each level of cross-validation. The nine-region variable is a policy-concentration proxy and is not a direct measurement of human urgency. Likewise, `reply_peak95@2` measures concentration in the opponent's reply policy, not every aspect of the human notion of a forcing move.

The three labels also have different evidence sources. Tenuki-distance is board-geometric. Urgency is computed directly from the current policy produced downstream of the same `trunkfinal` tensor being probed, so strong predictability is partly expected from the architecture. Forcing is derived from the opponent's policy after the selected move, while its probe uses the pre-move activation. Their AUCs therefore measure predictability of different kinds of targets and should not be treated as equivalent evidence for equally rich concepts.

Contract sources: `daniele_experiment/artifacts/runs/validity_v5_canonical/labels_manifest.json`, `daniele_experiment/artifacts/runs/validity_v5_canonical/frozen_config/concepts.yaml`, and `daniele_experiment/operational_definitions.py`. Policy-head architecture: `python/model_pytorch.py`.

### 1.4 Feature extraction and the indexing correction

Three feature summaries were constructed from each relevant `trunkfinal` activation:

- **Global:** the spatial mean of each of the 512 channels.
- **Selected-location:** the 512-channel vector at the recorded move's row-major board coordinate, `idx361`.
- **Combined:** concatenation of global and selected-location features, producing 1,024 features.

KataGo's padded internal `move_loc` is not the same coordinate system as a flat 19-by-19 tensor index. The corrected pipeline converts and validates `idx361`, then indexes `trunkfinal[:, idx361 // 19, idx361 % 19]`. Results made with direct `move_loc` indexing were therefore excluded and archived.

Feature source: `daniele_experiment/artifacts/runs/validity_v5_canonical/build_manifest.json`. Correction implementation: `daniele_experiment/validated_probe_pipeline.py`. Invalidation notice: `daniele_experiment/artifacts/archive/20260730_pre_idx361_invalid_v1/INVALID.md`.

### 1.5 Validation checks

The supplied checkpoint was replayed against one deterministically sampled saved activation from each of the 500 development games. All 500 samples passed the absolute-error tolerance of $10^{-4}$. Across 92,416,000 activation elements, the maximum absolute error was $3.5048 \times 10^{-5}$, the mean absolute error was $2.9000 \times 10^{-7}$, and the RMS error was $4.7509 \times 10^{-7}$. This establishes sampled empirical compatibility between the checkpoint, replay implementation, and stored activations; it does not prove that the checkpoint originally generated every historical activation.

Result source: `daniele_experiment/artifacts/runs/validity_v5_canonical/checkpoint_activation_fidelity.json`.

The frozen validity-v5 run recorded 88 passing tests; its focused-test log is
tracked with the audit artifacts. The current repository test suite contains
additional coverage and passed 118 tests in the `ml` environment on 2026-08-08.

Test source: `daniele_experiment/artifacts/logs/validity_v5/focused_tests.log`.

## 2. Experiment 1: linear prediction of canonical operational labels

### 2.1 Question and procedure

This experiment asked whether each canonical operational label was linearly predictable from global, selected-location, or combined summaries of `trunkfinal`.

Each representation was standardised within its training fold and supplied to a class-balanced logistic regression. Evaluation used five outer folds grouped by game. Within every outer training partition, four game-grouped inner folds selected the regularisation strength from $C \in \{0.001, 0.01, 0.1, 1, 10\}$ by mean average precision. A classification threshold was selected by maximising F1 on inner out-of-fold predictions. There was no probability calibration and no performance-based quality gate.

Uncertainty intervals used 2,000 game-cluster bootstrap draws within outer folds. These intervals are conditional on the fixed nested-CV out-of-fold predictions and fold-specific thresholds: probes, scalers, hyperparameters, and thresholds were not refitted inside each bootstrap draw.

Procedure source: `daniele_experiment/artifacts/runs/validity_v5_canonical/training_manifest.json`. Split assignments: `daniele_experiment/artifacts/runs/validity_v5_canonical/splits.parquet`. Fold results: `daniele_experiment/artifacts/runs/validity_v5_canonical/nested_cv_results.parquet`. Row-level predictions: `daniele_experiment/artifacts/runs/validity_v5_canonical/outer_predictions/`.

### 2.2 Main probe results

All reported scores below are means across the five held-out outer folds. Confidence intervals are percentile game-cluster bootstrap intervals.

| Operational label | Positive / eligible | Prevalence | Global ROC-AUC | Selected-location ROC-AUC | Combined ROC-AUC (95% CI) | Combined AP (95% CI) | Combined balanced accuracy | Outer-fold AUC SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tenuki-distance | 17,450 / 50,003 | 34.90% | 0.8529 | 0.9507 | 0.9577 [0.9559, 0.9594] | 0.9238 [0.9199, 0.9276] | 0.8863 | 0.0017 |
| Forcing proxy | 4,934 / 50,503 | 9.77% | 0.9047 | 0.9577 | 0.9643 [0.9620, 0.9664] | 0.7783 [0.7657, 0.7911] | 0.8466 | 0.0015 |
| Urgency proxy | 7,502 / 50,003 | 15.00% | 0.9700 | 0.9505 | 0.9711 [0.9694, 0.9727] | 0.8601 [0.8517, 0.8690] | 0.8870 | 0.0034 |

Authoritative result source: `daniele_experiment/artifacts/runs/validity_v5_canonical/corrections/validated_results_report_apfix_v2/corrected_results_report.json`.

The result supports the statement that these three operational labels were highly linearly predictable from the tested activation summaries. It does not by itself establish that KataGo represents the full human meanings of tenuki, forcing, or urgency.

### 2.3 Representation ablations

Paired game-cluster bootstraps compared the representations on the same held-out rows and with the same resampled game multiplicities.

| Operational label | Selected-location minus global ROC-AUC (95% CI) | Relevant combined-representation gain (95% CI) |
|---|---:|---:|
| Tenuki-distance | +0.0979 [0.0937, 0.1013] | Combined minus selected-location: +0.0070 [0.0062, 0.0078] |
| Forcing proxy | +0.0529 [0.0483, 0.0579] | Combined minus selected-location: +0.0066 [0.0057, 0.0075] |
| Urgency proxy | -0.0196 [-0.0212, -0.0179] | Combined minus global: +0.0010 [0.0004, 0.0017] |

For urgency, the combined-minus-global average-precision difference was +0.0014 with 95% CI [-0.0020, 0.0048], so the very small combined gain was not robust under that metric. Tenuki-distance and forcing benefited substantially from the selected-move feature; urgency was better captured by the global summary. These are comparisons between engineered summaries and do not anatomically localise information within the network.

Authoritative result source: `daniele_experiment/artifacts/runs/validity_v5_canonical/corrections/validated_results_report_apfix_v2/corrected_results_report.json`.

### 2.4 Reporting correction

The original frozen reporter mishandled average precision in a bootstrap edge case involving a zero-weight leading score group. A post-freeze correction recomputed the AP bootstrap summaries from the unchanged out-of-fold predictions. This correction changes AP bootstrap results only; it does not change the trained probes, predictions, ROC-AUC results, or causal experiment. The corrected report quoted above supersedes the earlier report for AP intervals.

Correction source: `daniele_experiment/artifacts/runs/validity_v5_canonical/corrections/validated_results_report_apfix_v2/corrected_results_report_manifest.json`.

## 3. Experiment 2: held-out tenuki-distance activation intervention

### 3.1 Question and confirmatory hypothesis

This experiment tested whether intervening along the local tenuki-distance probe direction would change the direct policy's mass on moves satisfying the exact distance-at-least-six predicate. It was the sole primary confirmatory causal test.

The frozen prediction was that increasing dose would increase the distance-at-least-six policy mass and decreasing dose would reduce it. Headline support required all three conditions:

1. the trained-direction slope across all doses was strictly positive;
2. the one-sided random-direction empirical $p$-value was at most 0.05; and
3. the one-sided spatial-shuffle empirical $p$-value was at most 0.05.

No individual dose, secondary readout, or minimum exploratory $p$-value was allowed to replace this conjunction.

Frozen hypothesis source: `daniele_experiment/artifacts/protocols/validity_v5.json`.

### 3.2 Direction construction and intervention

The local logistic probe and its scaler were fitted using only the 500 development games. If the standardised probe score is

\[
s(x) = \beta^\top \left(\frac{x-\mu}{\sigma}\right) + b,
\]

then its raw-activation gradient is

\[
g = \frac{\beta}{\sigma},
\]

with elementwise division. The raw-space intervention direction was

\[
\delta = \frac{g}{g^\top g},
\]

so that $g^\top\delta=1$. Thus, before spatial masking, a unit dose corresponds to a one-unit change in the probe's raw linear score.

For each position, the spatial mask was positive on legal board moves whose Manhattan distance from the most recent non-pass move was at least six and negative on all other legal board moves. Positive and negative mask values had equal signed totals. Illegal intersections and pass received zero mask weight, and the mask was divided by its RMS over the active legal support. At nominal dose $d$, the local-only intervention was

\[
h'_{c,y,x}=h_{c,y,x}+d\,m_{y,x}\delta_c.
\]

Because the mask varies spatially, a nominal unit dose is not a uniform one-score-unit shift at every intersection. Equal signed totals also do not mean equal per-site magnitudes: when the far target set contains more legal points than the nearby comparison set, each negative comparison-site coefficient is correspondingly larger. The observed inverse response could therefore reflect strong suppression on relatively few nearby points, weak enhancement over many distant points, or both.

The tested object is a composite of a learned channel direction and a researcher-defined spatial mask, not the channel direction alone. In addition, the local probe was trained on activation vectors at locations of moves actually selected during generated play, whereas the intervention broadcasts its direction over every legal candidate location. The design therefore assumes that the selected-action direction generalises to unselected candidate locations; that extrapolation was not independently tested.

Direction and mask source: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/manifest.json`. Probe metadata: `daniele_experiment/artifacts/runs/validity_v5_canonical/probes/local/probe_tenuki.meta.json`. Implementation: `daniele_experiment/validated_causal_eval.py` and `daniele_experiment/causal_controls.py`.

The frozen doses were $-2,-1,0,+1,+2$. The primary readout was the direct policy probability mass on legal board moves satisfying the same distance-at-least-six predicate, after renormalising over legal board moves. The primary statistic was an ordinary-least-squares slope of paired readout change on nominal dose, with equal weight given to the two selected label strata. Its confidence interval used 2,000 game-cluster bootstrap draws within label strata.

Readout and estimand source: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report.json`.

### 3.3 Causal integrity gates

The causal run passed its implementation and provenance gates.

| Gate | Result |
|---|---|
| Selected saved activations bound to the rebuilt run provenance | All 150 selected calibration/test activations |
| Operational label, mask, and readout alignment | 150 positions checked; 0 failures |
| Full-network versus saved-activation policy-head equivalence | 6 held-out positions |
| Maximum activation error in equivalence sample | 0 |
| Maximum absolute policy error in equivalence sample | $6.56 \times 10^{-7}$, below $10^{-6}$ tolerance |

Activation bindings: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/activation_bindings.parquet`. Operational audit: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/operational_alignment.json`. Backend audit: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/policy_head_equivalence.json`. Validated run status: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/manifest.json`.

These checks support the conclusion that the implemented perturbation was applied at the intended saved activation and that the policy-head shortcut reproduced full-network policy output within tolerance. They do not determine whether the resulting direction is semantically specific; that was the role of the held-out controls.

### 3.4 Dose-response result

The observed slope was negative:

\[
\text{slope}=-0.0008723
\]

policy-mass units per nominal dose, with 95% CI [-0.0010276, -0.0007185]. In percentage-point units, this is -0.0872 percentage points per dose, 95% CI [-0.1028, -0.0719]. This was opposite to the frozen positive-slope prediction.

| Nominal dose | Mean distance-at-least-six mass | Paired change from dose 0, percentage points (95% CI) | Position-level changes in predicted direction |
|---:|---:|---:|---:|
| -2 | 45.192% | +0.180 [+0.148, +0.211] | 0 / 100 |
| -1 | 45.101% | +0.089 [+0.074, +0.105] | 0 / 100 |
| 0 | 45.012% | 0 | Not applicable |
| +1 | 44.926% | -0.086 [-0.101, -0.071] | 0 / 100 |
| +2 | 44.843% | -0.169 [-0.200, -0.140] | 0 / 100 |

At every non-zero dose, all 100 position-level changes had the sign opposite to the prediction. The response was small in absolute policy terms. At dose -2, which had the largest mean Jensen-Shannon disruption among the trained-direction doses, mean policy JS divergence was $7.03 \times 10^{-6}$ and mean policy $L_1$ distance was 0.00422. The top policy move changed in 1 of 100 positions at doses +1 and +2 and in none at doses -1 and -2.

Authoritative result source: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report.json`. Position-level observations: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/causal_test_rows.parquet`.

### 3.5 Control construction and disruption matching

Two control families provided conditional contrasts for the learned channel direction and spatial arrangement.

| Channel direction | Spatial mask | Repetitions | Purpose |
|---|---|---:|---|
| Learned | Contract-aligned | 1 | Full tested intervention |
| Learned | Shuffled within legal support | 50 | Test whether the spatial arrangement matters |
| Random, norm-matched | Contract-aligned | 100 | Test whether the learned channel direction matters |

Control doses were calibrated on the 50 calibration games to approximately match the trained intervention's mean legal-plus-pass policy Jensen-Shannon divergence at each nominal dose. The selected multipliers were then frozen and transferred without refitting to the 100 test games. All 750 control-by-dose combinations passed the frozen **absolute** calibration tolerance. Among the 600 non-zero combinations, however, achieved-to-target JS ratios ranged from 0.624 to 1.481, and 252 differed from the target by more than 10% relatively. “Successful match” therefore denotes the registered absolute-tolerance rule, not tight relative equality.

On the test set, the control-to-trained JS ratio ranged from 0.687 to 2.293 for random directions and from 0.795 to 1.901 for shuffled masks. Disruption was therefore matched approximately on calibration data and transferred only approximately, not exactly, to the test set. A further limitation is that disruption matching used legal-plus-pass policy distributions, whereas the primary readout excluded pass and renormalised over legal board actions. Changes involving pass could consequently affect matching without affecting the primary readout in the same way.

The controls did not form a complete 2-by-2 factorial design: there was no random-direction-plus-shuffled-mask arm. They test learned-versus-random channels conditional on the aligned mask and aligned-versus-shuffled masks conditional on the learned direction, but do not fully decompose an interaction between channel and mask.

Calibration source: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/control_calibration.json`. Calibration and transfer diagnostics: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report.json`.

### 3.6 Control results and confirmatory decision

The random-direction slope distribution had mean 0.0000966, SD 0.0009245, and range [-0.0012135, 0.0012633]. Seventy of 100 random directions were at least as extreme as the trained result in the predeclared positive direction, producing one-sided empirical $p=0.703$. The trained direction ranked 71st from most extreme in that direction.

The spatial-shuffle slope distribution had mean $4.66 \times 10^{-7}$, SD $4.84 \times 10^{-5}$, and range [-0.0001539, 0.0001050]. All 50 shuffles were at least as extreme as the trained result in the predeclared positive direction, producing one-sided empirical $p=1.000$.

Control result source: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report.json`.

All three required conditions failed: the trained slope was not positive, the random-direction $p$-value exceeded 0.05, and the spatial-shuffle $p$-value exceeded 0.05. The registered decision was therefore:

> `does_not_pass_predeclared_headline_support_criterion`

Decision source: `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report.json`.

The valid conclusion is that the intervention pipeline executed as specified, but this held-out experiment did not support concept-specific positive steering of the tenuki-distance readout. The inverse signed response should not be relabelled after inspection as a successful “negative tenuki direction,” because that would reverse the frozen hypothesis and would not resolve the failed direction-specific controls.

## 4. Experiment 3: exploratory probes for historical operational labels

### 4.1 Scope and procedure

The historical run retained the original broader concept inventory while rebuilding all global, selected-location, and combined features with corrected `idx361` indexing and retraining probes with the same nested game-grouped method. Three labels in that run—tenuki-distance, forcing, and urgency—were recomputed canonically and duplicate the canonical results above. The remaining 24 configured labels were constructed from 22 source fields explicitly migrated from the quarantined pre-correction analysis.

These 24 results are exploratory. Their weak-labelling/Snorkel-derived source fields were not independently rebuilt or construct-validated during validity-v5. Corrected probe training shows predictability of the configured labels derived from those fields; it does not validate their intended Go semantics. There is no valid causal experiment for any of them.

Label provenance: `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/labels_manifest.json`. Exact exploratory label configurations: `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/frozen_config/concepts.yaml`. Training provenance: `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/training_manifest.json`.

### 4.2 Exploratory results

The table reports combined-representation means across five held-out outer folds. Twenty-two of the 24 historical configurations use **post-move** activations; only `approaching_corner` and `occupy_corner` use pre-move activations. Consequently, many probes predict labels from a representation that already follows the labelled move, which can materially inflate separability for deterministic move-effect labels.

| Historical configured label | Positive / eligible | Prevalence | Combined ROC-AUC | Combined AP |
|---|---:|---:|---:|---:|
| `aji_reduction` | 5,174 / 50,003 | 10.35% | 0.891582 | 0.483073 |
| `approaching_corner` | 1,310 / 50,503 | 2.59% | 0.997226 | 0.988855 |
| `atari` | 5,517 / 50,003 | 11.03% | 0.997592 | 0.983597 |
| `connect` | 4,359 / 50,003 | 8.72% | 0.999310 | 0.993599 |
| `cut` | 5,407 / 50,003 | 10.81% | 0.998961 | 0.990207 |
| `extend` | 20,270 / 50,003 | 40.54% | 0.999917 | 0.999877 |
| `fight_pressure` | 2,733 / 5,462 | 50.04% | 0.883302 | 0.877423 |
| `fight_wide` | 5,935 / 18,196 | 32.62% | 0.745808 | 0.578739 |
| `group_connectivity_down` | 3,006 / 6,004 | 50.07% | 0.822215 | 0.832932 |
| `group_connectivity_up` | 2,518 / 5,038 | 49.98% | 0.809795 | 0.804187 |
| `group_strength_down` | 2,511 / 5,009 | 50.13% | 0.804654 | 0.804593 |
| `group_strength_up` | 4,466 / 8,935 | 49.98% | 0.762653 | 0.786451 |
| `influence_surge` | 2,680 / 17,074 | 15.70% | 0.721158 | 0.348861 |
| `invasion` | 583 / 50,003 | 1.17% | 0.962912 | 0.297756 |
| `kill_attack` | 5,592 / 50,003 | 11.18% | 0.785961 | 0.298791 |
| `multi_connect` | 758 / 50,003 | 1.52% | 0.999042 | 0.964365 |
| `must_live` | 6,833 / 50,003 | 13.67% | 0.938781 | 0.713841 |
| `occupy_corner` | 1,998 / 50,503 | 3.96% | 0.999993 | 0.999875 |
| `opponent_reduction` | 6,983 / 13,954 | 50.04% | 0.973025 | 0.971575 |
| `sacrifice_commitment` | 324 / 649 | 49.92% | 1.000000 | 1.000000 |
| `sacrifice_direct` | 2,155 / 50,003 | 4.31% | 0.999348 | 0.986428 |
| `sacrifice_indirect` | 1,260 / 50,003 | 2.52% | 0.853990 | 0.157026 |
| `territory_building` | 6,462 / 12,900 | 50.09% | 0.890698 | 0.878284 |
| `territory_securing` | 6,897 / 13,808 | 49.95% | 0.948014 | 0.948721 |

Authoritative result source: `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/corrections/validated_results_report_apfix_v2/corrected_results_report.json`.

Combined ROC-AUC ranged from 0.721158 to 1.000000. Several labels were nearly perfectly predictable, but this can reflect direct board geometry, deterministic label construction, post-move information, or model-output-derived features rather than a rich strategic concept. The perfect `sacrifice_commitment` score is especially uncertain because only 649 positions were eligible. Full global, selected-location, combined, fold-level, ablation, and bootstrap results remain in the quoted report and its upstream artifacts.

Correction provenance: `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/corrections/validated_results_report_apfix_v2/corrected_results_report_manifest.json`. Fold results: `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/nested_cv_results.parquet`. Row-level predictions: `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/outer_predictions/`.

## 5. Evidential boundary

| Target | Probe evidence | Valid causal evaluation | Result that the evidence permits |
|---|---|---|---|
| Tenuki-distance proxy | Canonical; combined ROC-AUC 0.9577 | Yes; failed all predeclared support conditions | The operational label is linearly predictable; the tested positive steering claim is unsupported. |
| Forcing proxy | Canonical; combined ROC-AUC 0.9643 | No | Probe result only. |
| Urgency proxy | Canonical; combined ROC-AUC 0.9711 | No | Probe result only. |
| 24 labels derived from migrated historical fields | Exploratory; combined ROC-AUC 0.721158–1.000000 | No | Hypothesis generation and description of label predictability only. |

There is therefore one valid activation-manipulation experiment in validity-v5, not three. Historical interventions for tenuki, forcing, urgency, and corner variables were produced before the indexing and validity corrections and are excluded. No valid result in this record establishes:

- causal steering for forcing, urgency, or any historical label;
- that the learned tenuki-associated direction is naturally used as a mediator in ordinary play;
- effects after Monte Carlo tree search;
- changes in played-game outcomes; or
- human-like understanding of Go concepts.

The most direct overall statement is:

> Three rebuilt operational labels and 24 exploratory labels derived from 22 migrated historical source fields were linearly probed. The three rebuilt labels were highly predictable from `trunkfinal` feature summaries. One prospective, held-out activation intervention was conducted for the tenuki-distance proxy; it passed implementation-validity checks but failed the predeclared directional and control criteria, so concept-specific policy steering was not supported.

## 6. Excluded and archived results

Pre-`idx361` feature, probe, and causal results are retained only for auditability and must not be used in analysis, figures, or claims. The archive explicitly discards legacy tenuki, forcing, urgency, feature, probe, and causal results from evidential use.

Invalidation notice: `daniele_experiment/artifacts/archive/20260730_pre_idx361_invalid_v1/INVALID.md`. Archive manifest: `daniele_experiment/artifacts/archive/20260730_pre_idx361_invalid_v1/manifest.json`. Checksums: `daniele_experiment/artifacts/archive/20260730_pre_idx361_invalid_v1/checksums.sha256`.

An earlier validity-v5 causal attempt terminated without producing a validated causal result and is also excluded.

Failed-attempt notice: `daniele_experiment/artifacts/archive/20260731_validity_v5_tenuki_failed_09eccc18ab66/INVALID.md`. Failed-attempt manifest: `daniele_experiment/artifacts/archive/20260731_validity_v5_tenuki_failed_09eccc18ab66/manifest.json`.

The uncorrected canonical and exploratory `validated_results_report.json` files remain part of the audit trail but are superseded for average-precision bootstrap reporting by the `apfix_v2` reports quoted above.

## 7. Authoritative artifact index

The compact JSON, YAML, manifest, and focused-test records in this index are
tracked. Entries for Parquet datasets, activation bindings, prediction
directories, and the large feature-build manifests identify preserved local
intermediates and are not part of the Git repository.

| Evidence | Repository-relative path |
|---|---|
| Frozen validity-v5 protocol | `daniele_experiment/artifacts/protocols/validity_v5.json` |
| Canonical run manifest | `daniele_experiment/artifacts/runs/validity_v5_canonical/manifest.json` |
| Canonical label provenance | `daniele_experiment/artifacts/runs/validity_v5_canonical/labels_manifest.json` |
| Canonical label configuration | `daniele_experiment/artifacts/runs/validity_v5_canonical/frozen_config/concepts.yaml` |
| Canonical feature-build provenance | `daniele_experiment/artifacts/runs/validity_v5_canonical/build_manifest.json` |
| Canonical dataset | `daniele_experiment/artifacts/runs/validity_v5_canonical/dataset.parquet` |
| Probe training provenance | `daniele_experiment/artifacts/runs/validity_v5_canonical/training_manifest.json` |
| Canonical fold results | `daniele_experiment/artifacts/runs/validity_v5_canonical/nested_cv_results.parquet` |
| Corrected canonical probe report | `daniele_experiment/artifacts/runs/validity_v5_canonical/corrections/validated_results_report_apfix_v2/corrected_results_report.json` |
| Corrected canonical report manifest | `daniele_experiment/artifacts/runs/validity_v5_canonical/corrections/validated_results_report_apfix_v2/corrected_results_report_manifest.json` |
| Development checkpoint-fidelity audit | `daniele_experiment/artifacts/runs/validity_v5_canonical/checkpoint_activation_fidelity.json` |
| Causal run manifest | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/manifest.json` |
| Selected causal positions | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/selected_positions.parquet` |
| Causal activation bindings | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/activation_bindings.parquet` |
| Causal backend equivalence | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/policy_head_equivalence.json` |
| Causal operational alignment | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/operational_alignment.json` |
| Control calibration | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/control_calibration.json` |
| Position-level causal observations | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/causal_test_rows.parquet` |
| Exhaustive causal summary | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/summary.json` |
| Validated causal report | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report.json` |
| Validated causal report manifest | `daniele_experiment/artifacts/runs/validity_v5_canonical/causal/tenuki_local/validated_causal_results_report_manifest.json` |
| Exploratory label provenance | `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/labels_manifest.json` |
| Exploratory label configuration | `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/frozen_config/concepts.yaml` |
| Exploratory training provenance | `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/training_manifest.json` |
| Corrected exploratory probe report | `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/corrections/validated_results_report_apfix_v2/corrected_results_report.json` |
| Corrected exploratory report manifest | `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/corrections/validated_results_report_apfix_v2/corrected_results_report_manifest.json` |
| Focused validation tests | `daniele_experiment/artifacts/logs/validity_v5/focused_tests.log` |
| Primary invalid archive notice | `daniele_experiment/artifacts/archive/20260730_pre_idx361_invalid_v1/INVALID.md` |
| Primary invalid archive manifest | `daniele_experiment/artifacts/archive/20260730_pre_idx361_invalid_v1/manifest.json` |

## 8. Main implementation files

- Operational contracts: `daniele_experiment/operational_definitions.py`
- Label rebuild: `daniele_experiment/build_validated_labels.py`
- Corrected feature and probe pipeline: `daniele_experiment/validated_probe_pipeline.py`
- Causal evaluator: `daniele_experiment/validated_causal_eval.py`
- Control construction and calibration: `daniele_experiment/causal_controls.py`
- Causal report generator: `daniele_experiment/validated_causal_results_report.py`
- KataGo model implementation and activation site: `python/model_pytorch.py`
- Board legality implementation: `python/board.py`
- Game-state implementation: `python/gamestate.py`
- KataGo feature implementation: `python/features.py`
