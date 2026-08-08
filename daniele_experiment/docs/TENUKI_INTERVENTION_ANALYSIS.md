# Tenuki Intervention: Setup, Verified Inventory, and Design-vs-Conceptual Failure Analysis

This report records the failed confirmatory intervention and the completed F1
and F5 exploratory diagnostics. Sections 6–8 contain the latest interpretation;
the registered validity-v5 verdict remains unchanged. All paths are relative to
the repository root.

## 1. Verified experiment inventory

These counts were re-verified directly from the run artifacts on 2026-08-05.

- **27 concept labels received linear probes** in validity v5. Verified by counting the
  `concepts` entries in the two corrected reports:
  - `daniele_experiment/artifacts/runs/validity_v5_canonical/corrections/validated_results_report_apfix_v2/corrected_results_report.json`
    contains 3: `tenuki`, `forcing`, `urgency_peak`.
  - `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/corrections/validated_results_report_apfix_v2/corrected_results_report.json`
    contains 27: the same 3 (duplicated for comparison) plus 24 historical labels.
- The frozen exploratory `concepts.yaml` actually **configures 30** concepts. Three
  (`seki`, `group_connectivity_shift`, `group_strength_shift`) never trained because their
  source fields are not among the 22 hash-checked migrated legacy fields in
  `daniele_experiment/artifacts/runs/validity_v5_legacy_exploratory/labels_manifest.json`, and they do not
  appear in the training manifest's `trainability_audit` (27 entries). So: 30 configured,
  27 probed, 3 canonical.
- The 3 canonical variables were not "selected from the 27 by probe performance"; they were
  **promoted** by recomputing them from raw game records under versioned contracts
  (`daniele_experiment/operational_definitions.py`), independently of the historical Snorkel output:

  | Human concept | Canonical contract | Legacy Snorkel name |
  |---|---|---|
  | Tenuki | `tenuki_distance6@2` | `tenuki` |
  | Forcing | `reply_peak95@2` | `forcing` |
  | Urgency | `regional_policy_peak@2` | `urgency_peak` |

- **All probes were retrained inside validity v5** with corrected `idx361` feature indexing.
  No reported probe number predates the refactor; pre-correction results are archived under
  `daniele_experiment/artifacts/archive/20260730_pre_idx361_invalid_v1/`.
- **Exactly one valid activation-manipulation experiment exists**: the tenuki-distance proxy,
  on 100 held-out games. Forcing and urgency have probe evidence only. Earlier interventions
  for forcing/urgency/corner concepts are quarantined pre-`idx361` outputs.

## 2. The tenuki activation-manipulation setup

Implementation: `daniele_experiment/validated_causal_eval.py` (`InterventionDirection`, `concept_spatial_mask`,
`tenuki_contrast_mask`); controls in `daniele_experiment/causal_controls.py`; frozen protocol in
`daniele_experiment/artifacts/protocols/validity_v5.json`.

### 2.1 Direction (what gets added, channel-wise)

1. A **local** logistic probe (512 features = the trunk channel vector at the selected
   move's `idx361` location) and its `StandardScaler` were fitted on the 500 development
   games only.
2. The standardized probe score is s(x) = β^T((x − μ)/σ) + b. Its gradient in raw
   activation space is g = β/σ (elementwise).
3. The intervention direction is **δ = g / (gᵀg)**, so gᵀδ = 1: an unmasked unit dose moves
   the raw linear probe score by exactly one unit. This is the dose unit ("one probe-score
   unit"), chosen so doses are interpretable in the probe's own scale rather than raw
   activation norm.

### 2.2 Spatial mask (where it gets added)

For each evaluated position, a 19×19 mask m is built from the same contract that defines the
label:

- **positive** on legal board moves at Manhattan distance ≥ 6 from the most recent non-pass
  move ("far" = the tenuki target set);
- **negative** on all other legal board moves ("near");
- zero on illegal points and pass;
- positive and negative parts scaled to **equal total signed mass**, then the whole active
  support divided by its RMS.

Consequence of equal signed totals: per-site magnitudes are *not* equal. If the far set has
more points than the near set (typical mid-game; mean far policy mass at dose 0 was 45%),
each near point carries a larger negative coefficient than each far point's positive one,
and vice versa.

### 2.3 Application and readout

The hook modifies the saved `trunkfinal` tensor (512×19×19, the layer immediately before the
policy/value heads):

    h'[c,y,x] = h[c,y,x] + d · m[y,x] · δ[c],   d ∈ {−2, −1, 0, +1, +2}

and re-runs only the (unchanged) policy head. A backend equivalence gate confirmed the
saved-trunk shortcut reproduces full-network policy within 6.56×10⁻⁷ (tolerance 10⁻⁶).

**Primary readout:** total policy mass on legal board moves satisfying the same
distance-≥-6 predicate, renormalised over legal board moves (pass excluded).
**Primary statistic:** OLS slope of paired readout change on dose, equal weight to the two
label strata (50 label-positive, 50 label-negative test positions, one per game).

### 2.4 Controls and decision rule (frozen before fresh data existed)

| Arm | Direction | Mask | N |
|---|---|---|---:|
| Tested intervention | learned | contract-aligned | 1 |
| Shuffle control | learned | shuffled within legal support | 50 |
| Random-direction control | random, norm-matched per block | contract-aligned | 100 |

There is **no random-direction + shuffled-mask arm** (incomplete 2×2). Control dose
multipliers were calibrated on 50 calibration games to match the trained intervention's mean
policy JS divergence per dose, then frozen and transferred (only approximately holding on
the test set: ratios 0.687–2.293).

Headline support required all three: strictly positive trained slope; one-sided
random-direction empirical p ≤ 0.05; one-sided shuffle empirical p ≤ 0.05.

### 2.5 Result

- Trained slope: **−0.000872** policy-mass units per dose, 95% CI [−0.001028, −0.000719].
  Opposite sign to prediction; at every non-zero dose, 100/100 positions moved opposite to
  the predicted sign. Response nearly linear and anti-symmetric in dose.
- Effect size tiny: at dose −2, mean policy JS ≈ 7.03×10⁻⁶, mean L1 ≈ 0.0042; top move
  changed in ≤ 1/100 positions per dose.
- Random-direction slopes: mean +0.0000966, SD 0.000924, range [−0.001214, +0.001263];
  70/100 more positive than trained → p = 0.703.
- Shuffled-mask slopes: mean ≈ 0, SD 0.0000484, range [−0.000154, +0.000105]; p = 1.000.
- Registered decision: `does_not_pass_predeclared_headline_support_criterion`.

## 3. The single most diagnostic comparison

Put the three slope scales side by side:

| Quantity | Magnitude |
|---|---:|
| Trained direction, aligned mask | −8.7 × 10⁻⁴ |
| Random directions, aligned mask (SD) | 9.2 × 10⁻⁴ |
| Learned direction, shuffled masks (SD) | 0.48 × 10⁻⁴ |

Two facts follow:

1. **The spatial mask, not the learned direction, carries essentially all of the effect
   magnitude.** Random directions pushed through the aligned mask produce slopes of the
   same order as the trained direction (the trained slope sits at roughly −1 SD of the
   random distribution, rank 71/101, and inside its range). Destroying the spatial
   arrangement while keeping the learned direction collapses the effect by ~20×.
2. **The learned direction is not distinguishable from a random direction within this
   composite intervention.** Whatever concept-specific signal the direction carries is
   below the noise floor set by mask-generic effects.

## 4. Design failure or conceptual failure?

The honest answer is that the experiment establishes "the tested composite intervention does
not steer tenuki as predicted" and cannot, by itself, fully separate the two. But the
evidence constrains the candidates unevenly.

### 4.1 Candidates the evidence disfavors

- **Noise / underpower for the observed effect.** The inverse slope is precisely estimated,
  monotone, anti-symmetric, and unanimous across 100 positions. Whatever produced it is a
  real, systematic first-order coupling — not sampling noise.
- **Implementation error.** Three integrity gates passed: activation-provenance binding on
  all 150 positions, label/mask/readout operational alignment on all 150, and policy-head
  backend equivalence (max activation error 0, max policy error 6.56×10⁻⁷). The perturbation
  was applied where intended and read out as intended.

### 4.2 Design-level candidates (still open)

These would produce failure even if KataGo internally represents something tenuki-like:

- **D1 — Composite confound.** The tested object is (learned direction) × (researcher-
  designed far/near contrast mask). Section 3 shows the mask dominates. The design can only
  detect a direction-specific effect *larger* than mask-generic effects; the direction's
  contribution may be real but smaller than that floor. The missing random-direction ×
  shuffled-mask arm means the channel–mask interaction is not decomposed.
- **D2 — Extrapolation to unselected locations.** The local probe was trained exclusively
  on channel vectors at *selected-move* locations, but the intervention broadcasts δ to
  every legal intersection. Nothing tested whether δ means the same thing at locations the
  policy did not choose. (Flagged prospectively in the protocol as an untested assumption.)
- **D3 — Layer choice.** `trunkfinal` is immediately before the policy head. A direction
  that separates labels there may be "read" by the policy head with arbitrary orientation;
  a sign mismatch between the probe's label orientation and the policy head's effective
  readout of δ at masked locations would produce exactly the observed clean inverse linear
  response. This is testable analytically (Section 5, F1).
- **D4 — Mask asymmetry mechanics.** Equal signed totals with unequal set sizes mean
  per-site coefficients differ between far and near regions; the inverse response could
  reflect strong suppression at few near points or weak enhancement over many far points,
  or interactions with the policy head's spatial convolution. Not resolved by the readout.
- **D5 — Dose range.** Doses ±1, ±2 probe-score units produced JS ≈ 10⁻⁶–10⁻⁵ — far below
  the scale of natural position-to-position variation. A genuine mediator might need larger
  or differently normalised doses to express, though the clean linear inverse signal at
  small doses argues the first-order coupling simply has the wrong sign.

### 4.3 Conceptual-level candidates (still open)

These would mean the probe direction genuinely is not a control axis for the behaviour:

- **C1 — Decodable correlate, not mediator.** The probe may separate labels by reading
  information that is *correlated* with far-move selection (local stone density, settledness
  near the previous move, game phase, whose-turn parity effects) without that direction
  feeding the policy computation. High AUC (0.951 local) is fully compatible with this.
- **C2 — Proxy–concept gap.** `tenuki_distance6` is a geometric predicate. Even a true
  internal "leave the local area" feature need not align with Manhattan distance ≥ 6; the
  probe may have learned the geometry, not the strategic judgement.
- **C3 — No single linear axis.** Tenuki-relevant computation may be distributed /
  non-linear / spread across layers such that no single trunkfinal direction steers it, even
  though it is linearly *readable* there (superposition-style dissociation).

### 4.4 Current best reading

The control decomposition (Section 3) is the strongest clue. The learned direction, deprived
of its aligned mask, does essentially nothing; with the mask, it behaves like a random
direction. That pattern is most parsimoniously explained by **C1/C3 (the probe direction is
not a functional control axis) combined with D1 (the composite design's sensitivity floor is
set by mask-generic effects)**. The unanimous inverse sign is most likely a structural
property of how the policy head reads *this particular mask-shaped perturbation* (D3/D4),
not evidence of an "anti-tenuki" direction — the frozen protocol explicitly forbids that
relabelling, and 30/100 random directions produced even more negative slopes.

What the experiment does *not* license: concluding that KataGo has no tenuki-like internal
feature. It tested one direction, one layer, one mask family, one dose range, without search.

## 5. Follow-up experiments that would separate the hypotheses

Ordered roughly by information-per-effort. All would be new exploratory work; none can
retroactively change the registered v5 verdict.

- **F1 — Analytic first-order check (cheap, decisive for D3).** Compute the directional
  derivative of the far-mass readout with respect to dose at d = 0 by autograd through the
  policy head: ∂(readout)/∂d = ∇_h(readout) · (m ⊗ δ). If this is negative for most
  positions, the inverse slope is a structural property of the policy head's linear response
  to this mask–direction composite, confirming the sign was never a coin flip. Repeating
  with δ replaced by each random direction quantifies the direction-specific component
  exactly, without any calibration machinery.
  **Implemented and executed on 2026-08-05: `daniele_experiment/tenuki_gradient_analysis.py`; results in
  Section 6. The verdict in Sections 3–4 is revised there.**
- **F2 — Complete the 2×2 (D1).** Run random-direction × shuffled-mask controls. Comparing
  the four cells decomposes mask, direction, and interaction contributions.
- **F3 — Activation patching instead of direction addition (C1 vs D2/D3).** Swap
  `trunkfinal` activations (whole tensor, or far-region columns) between matched pairs of
  positions with opposite labels and check whether far-mass moves in the predicted
  direction. Patching uses *naturally occurring* activation differences, removing both the
  linear-direction assumption and the broadcast extrapolation. A positive patching result
  with a failed direction result would localise the failure to the probe direction (design);
  a negative one strengthens the conceptual reading.
- **F4 — Layer sweep (D3, C3).** Fit probes and repeat the (cheap F1 version of the)
  intervention at earlier trunk blocks. Mediating features are often steerable earlier and
  merely readable late.
- **F5 — Single-site intervention (D2).** Apply δ only at one candidate far location (the
  kind of location the probe was actually trained on) and read out that move's policy
  probability. This tests the direction in-distribution instead of broadcast.
  **Implemented and executed on 2026-08-05: `daniele_experiment/tenuki_single_site_analysis.py`; results in
  Section 7.**
- **F6 — Larger / re-normalised doses (D5).** Extend to ±5, ±10
  probe-score units and check for non-linearity or sign change. This remains a
  possible exploratory follow-up; no result from it enters the confirmatory
  record.
- **F7 — Counterfactual label flip probe (C2).** Construct positions where distance ≥ 6 and
  "strategically a tenuki" dissociate (e.g. forced ladders reaching far away) and check
  which one the probe tracks. Distinguishes geometric from strategic content of the
  direction.

## 6. F1 analytic gradient results (2026-08-05)

Implementation: `daniele_experiment/tenuki_gradient_analysis.py` (unit tests in
`daniele_experiment/tests/test_tenuki_gradient_analysis.py`, 7 passing). Output artifact:
`daniele_experiment/artifacts/exploratory/tenuki_gradient_analysis.json`. Exploratory diagnostic only; the
registered validity-v5 verdict is unchanged.

For each of the same 100 hash-bound causal-test positions (selection verified identical to
the confirmatory run), the gradient of the far-mass readout with respect to the saved
`trunkfinal` tensor was computed by autograd through the frozen policy head at dose 0. The
first-order dose response of any direction–mask composite is then a single dot product
∇_h(readout) · (m ⊗ δ), evaluated for the trained direction, the same 100 seeded random
directions, and the same 50 seeded mask shuffles as the confirmatory run — without any
disruption-calibration machinery.

### 6.1 Verification

- Analytic dose-0 derivative (equal-label-strata mean): **−0.0008721** per unit dose.
  Confirmatory OLS slope over doses ±2: −0.0008723. Absolute difference **2.4 × 10⁻⁷**.
- Central finite differences through the validated backend (dose step 0.25, 8 positions):
  max |analytic − central| = 4.5 × 10⁻⁷.
- Differentiable readout vs backend baseline readout: max discrepancy 1.5 × 10⁻⁶ (float32).
- Policy-head equivalence re-passed in the current environment (max policy error
  7.7 × 10⁻⁷ < 10⁻⁶). Full-network trunk replay drifted marginally above the frozen
  activation tolerance (1.0014 × 10⁻⁵ vs 10⁻⁵), consistent with a torch-environment change
  since the confirmatory run; the analysed tensors are the hash-bound originals, so this
  affects nothing downstream.

### 6.2 Findings

| Quantity | Value |
|---|---:|
| Trained direction, aligned mask (stratified mean ∂readout/∂dose) | −0.000872 |
| … on label-positive positions / label-negative positions | −0.001133 / −0.000611 |
| Positions with negative derivative | 100 / 100 |
| Random directions (uncalibrated): mean, SD | +0.0000358, 0.000513 |
| Random directions more positive than trained | 95 / 100 |
| Random directions with \|coupling\| ≥ \|trained\| | 10 / 100 |
| Shuffled masks: mean, SD | +0.0000007, 0.0000378 |

1. **The confirmatory result is entirely first-order (D3 confirmed).** The registered slope
   is reproduced to 3 decimal digits of relative precision by a Jacobian dot product at
   dose 0. There is no dose-range or nonlinearity story (D5 eliminated at these doses): the
   experiment measured a structural property of the frozen policy head's linear response,
   and the inverse sign was never a coin flip — it holds at every position.

2. **The earlier "indistinguishable from random" reading (Section 3) must be revised.** In
   per-unit-dose terms, without the confirmatory run's JS-calibration multipliers, the
   trained direction sits at −1.77 SD of the random-direction distribution (95/100 more
   positive) and its coupling *magnitude* is in the top decile (only 10/100 random
   directions couple as strongly). The empirical p = 0.703 was diluted by calibration:
   multipliers rescaled weakly coupling random directions up to the trained direction's
   disruption level. The learned direction is therefore not behaviourally inert noise — it
   has systematically above-typical coupling to the far-mass readout, but with orientation
   **opposite** to its label orientation, roughly twice as strong on label-positive
   positions.

3. **The mask remains the gate, the direction sets sign and strength within it.** Shuffling
   the mask still collapses everything (SD 23× smaller than the trained effect), so the
   spatial contrast is a necessary component; but within the aligned-mask family the trained
   direction is a distinctly anti-aligned outlier rather than a typical draw.

### 6.3 Revised design-vs-conceptual verdict

The failure is now best described as a **structural sign inversion, not an absence of
coupling and not an execution or sensitivity artifact**. The direction that increases the
local probe's tenuki score, when broadcast over all legal points through the contract mask,
*decreases* far-region policy mass at first order, consistently and with above-random
strength. Candidates C1-strong (pure correlate, no coupling) and D1/D5 (sensitivity floor,
dose range) lose weight; the live explanations are now:

- **D2/D3 (design):** the selected-move-trained direction means something different when
  broadcast to unselected locations, and the policy head reads that broadcast pattern with
  inverted orientation. Testable next by F5 (single-site, in-distribution application) and
  F3 (activation patching with natural differences).
- **C2/C3 (conceptual):** the probe direction encodes a correlate whose behavioural role is
  genuinely anti-aligned with far-move probability (for example a local-urgency or
  local-attention feature that co-varies with the distance label in natural play but
  suppresses distant moves when amplified).

The frozen protocol rightly forbids renaming this a "negative tenuki direction" as a
*confirmatory* claim; but as an exploratory fact, the anti-aligned coupling is real,
precise, and reproducible, and F5/F3 are the experiments that would tell whether it is an
artifact of broadcasting or a property of the encoded feature. **F5 was executed the same
day; see Section 7 — the sign does not flip back in-distribution, which moves the verdict
further toward the conceptual side and identifies a concrete mechanism.**

## 7. F5 single-site results and mechanism (2026-08-05)

Implementation: `daniele_experiment/tenuki_single_site_analysis.py` (unit tests in
`daniele_experiment/tests/test_tenuki_single_site_analysis.py`; 13 passing across both diagnostic tools). Output:
`daniele_experiment/artifacts/exploratory/tenuki_single_site_analysis.json`. Same 100 hash-bound causal-test
positions, same seeded random directions; all quantities are analytic dose-0 derivatives.
Dose unit: one raw probe-score unit at the intervened site (a single-point mask has unit
RMS, so this is the same dose convention as the confirmatory run before spatial shaping).

### 7.1 The in-distribution test: the sign does not flip back

For each position, δ was applied at one site at a time — the actually selected move
(exactly the kind of location the local probe was trained on), the top-5 far candidates,
and the top-5 near candidates by baseline policy — reading out that site's own legal-board
policy probability.

| Site type | Mean self-effect per dose unit | Fraction positive |
|---|---:|---:|
| Actually selected move (100 sites) | −3.40 × 10⁻⁴ | 2% |
| … when the selected move was far (50) | −4.15 × 10⁻⁴ | 0% |
| … when the selected move was near (50) | −2.65 × 10⁻⁴ | 4% |
| Top far candidates (500 sites) | −1.96 × 10⁻⁴ | 1% |
| Top near candidates (500 sites) | −1.29 × 10⁻⁴ | 4% |

Pushing the "tenuki" direction at a location makes the network *less* likely to play that
location — essentially everywhere, and most strongly at far moves the network actually
chose. This is direction-specific, not generic disruption: against the same 100 random
directions (random mean +7.9 × 10⁻⁶, SD 1.1 × 10⁻⁴), 97/100 are more positive than the
trained direction and only 5/100 have larger absolute effect. The learned direction is a
top-5% *local move suppressor*.

### 7.2 Broadcast decomposition: one mechanism explains the confirmatory failure

Decomposing the confirmatory far-mass derivative over board sites (identity error
< 10⁻¹⁸; total reproduces the F1 value −0.000872):

| Component | Stratified mean contribution |
|---|---:|
| Far sites (positive mask weight) | −0.000162 |
| Near sites (negative mask weight) | −0.000710 |

Unmasked per-site couplings close the loop: one probe-score unit of δ at a *near* site
**increases** far-region mass (+7.4 × 10⁻⁶ per site, positive at 99% of sites), while at a
*far* site it decreases it (−1.6 × 10⁻⁶, positive at only 8%). Combined with Section 7.1,
the mechanism is renormalisation of a local suppression: injecting δ at a site suppresses
that site's own probability, and the freed probability mass redistributes across the board.
Suppress a near site and far mass rises; suppress a far site and far mass falls.

The confirmatory broadcast did the worst of both: positive dose added δ at far sites
(suppressing far moves directly) and *subtracted* it at near sites (a negative dose of a
suppressor, i.e. boosting near moves). Both halves push far-region mass down — which is why
the inverse response was unanimous across positions, and why ~80% of the registered
negative slope comes from the near half of the mask.

### 7.3 Final reading: primarily conceptual, with a precise design lesson

The broadcast-extrapolation explanation (D2/D3 as *artifacts*) is now largely closed: the
direction behaves the same way in-distribution, at single sites, including at the exact
selected-move locations the probe was trained on. The dissociation is conceptual in the
sense that matters: **the probe direction's label orientation ("this move is a tenuki") is
opposite to its causal content at a site ("do not play here")**. A plausible unifying
account is that the direction encodes something like local settledness or unattractiveness
— a "this area can be left" signal. That signal is genuinely informative about tenuki
(strongly present at chosen far moves, hence ROC-AUC 0.951), but amplifying it at a
location tells the policy head to avoid that location, not to prefer it.

The design lesson is equally concrete: the frozen mask assumed the direction was a
"play-far" feature and applied it positively on far sites. Under the suppressor reading,
steering *toward* tenuki would instead apply the direction positively on **near** sites
(make the local area look leavable). The decomposition already supports this prediction
exploratorily — near-site injection raises far mass at 99% of sites — but any such claim
would need a fresh frozen protocol and untouched games to be more than hypothesis
generation.

What remains genuinely open: whether the suppressive content is best described as
settledness, low local urgency, low policy confidence, or another correlate (F7-style
dissociation tests would discriminate); and whether any of this survives at earlier layers
(F4) or under natural activation differences (F3).

## 8. Summary

The validity-v5 tenuki intervention failed its frozen criteria, and the two analytic
follow-ups explain why. F1 showed the registered inverse slope (−0.000872 per dose) is a
pure first-order property of the policy head, reproduced to 2.4 × 10⁻⁷ by a Jacobian dot
product, with the trained direction coupling more strongly than 90% of random directions
but with sign opposite to its label orientation. F5 localised the mechanism: the learned
direction is a direction-specific local move suppressor (self-effect negative at ~98% of
tested sites, top-5% magnitude among random directions), so the far-positive/near-negative
broadcast mask suppressed far moves and boosted near moves simultaneously — both reducing
the far-mass readout, with ~80% of the effect coming from the near half of the mask. The
confirmatory failure is therefore neither noise nor an execution artifact, and only
partially a broadcast-design artifact: the probe found a direction that is informative
about tenuki but whose causal content is anti-aligned ("leavable / don't play here" rather
than "play far"). Decodability and steerability dissociate here not because the direction
is causally inert, but because its causal orientation contradicts the label semantics —
a sharper demonstration that probe performance alone cannot license conclusions
about causal use.

## 9. Evidential boundaries

The F1 and F5 analyses are exploratory; the registered validity-v5
verdict (`does_not_pass_predeclared_headline_support_criterion`) is unchanged.

### Findings

1. The registered inverse slope is a first-order structural property of the frozen policy
   head, not noise, dose-range effects, or an execution artifact (F1; analytic–empirical
   agreement 2.4 × 10⁻⁷; finite-difference check 4.5 × 10⁻⁷).
2. The learned direction is not causally inert: its coupling magnitude is in the top decile
   of the 100 seeded random directions under the aligned mask (F1) and in the top 5% for
   single-site self-effects (F5).
3. Causally, at the tested layer, the direction acts as a direction-specific **local move
   suppressor**: self-effects negative at ~98% of 1,100 tested sites, including the exact
   selected-move locations the probe was trained on (F5).
4. The mechanism of the confirmatory failure is local suppression plus renormalisation
   through the far-positive/near-negative mask, with ~80% of the negative slope from the
   near half (F5 decomposition, exact to < 10⁻¹⁸).

### Limitations

- *"The probe actually found a settled-local-position concept."* The
  settledness/leavability reading is one interpretation consistent with both the high AUC
  and the negative self-effects, but F5 does not discriminate it from other anti-aligned
  correlates: low local urgency, low local policy confidence, local "already decided"
  status, or game-phase/density proxies. Naming the feature requires an F7-style
  dissociation battery. The direction predicts the tenuki-distance label but
  causally suppresses play at the intervened site; a possible interpretation
  is that the local area can be left, although the data do not uniquely
  identify that semantics.
- Any *confirmatory* causal claim about the suppressor (including reversing the sign into
  a "negative tenuki direction" or a steering recipe). The frozen protocol forbids post-hoc
  sign reversal, and the diagnostics were computed on the same 100 test positions the
  confirmatory run used, so they cannot serve as fresh held-out evidence.

The first-order version of a reverse test is already contained in F5 (self-effect at the actually selected move:
−3.40 × 10⁻⁴ per dose unit, negative at 98/100 positions, strongest for far-selected
moves). A new experiment would only add value as (a) a finite-dose test of practical
top-move demotion, or (b) a *positive* confirmatory claim about the suppressor direction —
which would require a newly frozen protocol and untouched games.
