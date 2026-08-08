# Activation manipulation and causal concept checks

> **Historical exploratory workflow.** These in-sample interventions predate
> the validated causal experiment and are not included in its results. The
> held-out validity-v5 procedure is implemented by `validated_causal_eval.py`.

The examples below require a locally supplied `daniele_experiment/model.ckpt`
and locally generated probes under `daniele_experiment/linear_probes/`. Those
files are intentionally excluded from Git. The checkpoint cannot currently be
downloaded from a recorded source; its expected SHA-256 is documented in the
experiment README. The probes can be regenerated only when that exact
checkpoint and the required game corpus are available.

## Historical matched-position evaluation

Build the position bank once from the existing games, labels, and probe scores:

```bash
python daniele_experiment/position_causal_eval.py build
```

Then run baseline-versus-hooked evaluation on positions stratified by label and
probe score:

```bash
python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt tenuki \
  --doses -2 -1 0 1 2 \
  --max-positions 200 \
  --output causal_positions_tenuki.json
```

For `tenuki`, the primary metrics are expected distance from the previous move
and policy mass at least six intersections away. Registered metrics also cover
forcing and corner concepts. Every concept reports probability assigned to the
recorded labeled move; when no bespoke metric exists the report clearly marks
this as a fallback.

Interpret steering from a signed change in the named behavior metric that
outperforms the spatial controls. Observed-positive and observed-negative
strata represent reinforcement and induction opportunities respectively; a
policy flip or probe score by itself is not enough.

### Spatial interventions

The spatial modes use the probe's move-location coefficient block as a 1x1
channel direction across the board. `local-far` applies it only at points at
least `--far-distance` from the previous move. `local-contrast` enhances those
far points while suppressing the neighborhood within `--local-radius`, using a
zero-mean, RMS-normalized mask.

```bash
python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt tenuki \
  --intervention-mode local-contrast \
  --local-radius 4 \
  --far-distance 6 \
  --doses -5 -2 0 2 5 \
  --max-positions 500 \
  --output causal_tenuki_local_contrast.json
```

Run the same command with `--mask-control shuffled` as a spatial-alignment
control. `--mask-control inverted` reverses the aligned mask. A useful spatial
result should be stronger for the aligned mask than for the shuffled control.

To test whether the tenuki channel vector matters—not merely the far/near mask—
run multiple seeded random channel directions with exactly the same L2 norm:

```bash
python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt tenuki \
  --intervention-mode local-contrast \
  --direction-control random \
  --random-directions 10 \
  --doses -5 -2 -1 0 1 2 5 \
  --max-positions 500 \
  --output causal_tenuki_random_directions.json
```

An aligned other-concept direction is another useful control:

```bash
python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt tenuki \
  --intervention-mode local-contrast \
  --direction-control other-concept \
  --control-concept atari \
  --doses -5 -2 -1 0 1 2 5 \
  --max-positions 500 \
  --output causal_tenuki_atari_direction.json
```

### Other concept-specific evaluations

Corner concepts were trained with their move-location block excluded, so they
use their global direction. Positions are filtered to those where the behavior
is actually possible. `occupy_corner` requires a completely empty 6x6 corner;
`approaching_corner` requires a 6x6 corner containing exactly one opponent
stone. Metrics include policy mass and top-move incidence in only those eligible
corners.

```bash
python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt occupy_corner \
  --intervention-mode global --doses -20 -10 -5 -2 0 2 5 10 20 \
  --max-positions 500 --output causal_occupy_corner.json

python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt approaching_corner \
  --intervention-mode global --doses -20 -10 -5 -2 0 2 5 10 20 \
  --max-positions 500 --output causal_approaching_corner.json
```

`forcing` uses a zero-mean local contrast between the baseline top board move
and all other intersections. It reports top probability, top-two margin,
entropy, and crossings of the 95% forcing threshold:

```bash
python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt forcing \
  --intervention-mode concept-local --doses -5 -2 -1 0 1 2 5 \
  --max-positions 500 --output causal_forcing.json
```

`urgency_peak` contrasts the baseline highest-policy-mass 6x6/side/center
region against the rest of the board. It reports peak regional mass, the top-two
regional margin, and regional entropy:

```bash
python daniele_experiment/position_causal_eval.py evaluate \
  daniele_experiment/model.ckpt urgency_peak \
  --intervention-mode concept-local --doses -5 -2 -1 0 1 2 5 \
  --max-positions 500 --output causal_urgency_peak.json
```

For either local concept, repeat with `--direction-control random` and
`--random-directions 10`. The tenuki metric and mask use the same Manhattan
distance convention as the Snorkel label.

## Supplementary: full-game evaluation

`activation_manipulation.py` intervenes on KataGo's `trunkfinal` activation in
the direction learned by a linear concept probe. It then plays 1-visit games at
several doses and records:

- Jensen-Shannon and L1 policy changes on the exact same positions
- how often the top move changes
- predicted win-rate and score changes
- final Tromp-Taylor area score and winner
- a signed dose-response slope

## Pre-only rule

**Only `feature_mode: pre` probes are allowed.**

The hook runs on the forward pass that **chooses the next move** (pre-move
trunk). That matches intent / situation probes trained on `h`:

| Mode | Trained on | Use for this script? |
|------|------------|----------------------|
| `pre` | trunk before the move | Yes |
| `post` | trunk after the move | No — observational effect readout |
| `delta` | `h_next - h` | No — not a single forward pass |

Post/delta concepts such as `cut` or `atari` can be highly decodable and still
be the wrong target for this causal loop. Rejecting them is intentional: a null
result on a post probe is hard to interpret.

List eligible concepts (pre + probe artifacts present):

```bash
python daniele_experiment/activation_manipulation.py --list-concepts \
  --probes-dir daniele_experiment/linear_probes
```

Current pre concepts in `concepts.yaml` (when trained): `forcing`, `tenuki`,
`occupy_corner`, `approaching_corner`, `urgency_peak`.

## Dose units

The dose is measured in probe decision-score units. For example, `+2` requests
a two-unit increase in the logistic probe's log-odds score on the **global mean
pool** block. When probes also include move-location features, only the mean-pool
block is steered (the prospective move is unknown until after the policy head).

## Recommended protocol

```bash
# Strong, deterministic check on a pre-mode concept
python daniele_experiment/activation_manipulation.py daniele_experiment/model.ckpt forcing \
  --probes-dir daniele_experiment/linear_probes \
  --doses -8 -4 0 4 8 \
  --games-per-dose 1 \
  --temperature 0 \
  --intervene-player black \
  --output causal_forcing_strong.json
```

Then repeat with `--intervene-player white`.

Notes:

- Prefer `--temperature 0` so `top_move_changed` and game paths are unambiguous.
- With `temperature 0` from an empty board, extra seeds replay the same game;
  `games-per-dose > 1` does not buy opening diversity.
- Each dose reuses the same seeds when temperature sampling is used
  (common-random-number pairing).
- By default probes load from `daniele_experiment/linear_probes`. A relative
  `--probes-dir` is checked from both the cwd and `daniele_experiment`.

## How to interpret results

Primary (same-position) metrics:

1. `policy_js` / `policy_l1` — did the hook move the policy at all?
2. `top_move_changed` — did it change the argmax decision?
3. `winrate_delta` / `scoremean_delta` — value-head shift on that position

Secondary (full-game) metrics:

4. `area_score_black_minus_white` / `winner` — only meaningful if (2) is non-zero
5. `dose_response.*_slope` — quick signed trend across doses, not a significance test

The strongest evidence from this workflow would combine a same-position policy
effect, such as top-move changes, with a repeatable signed dose response across
colours and openings. Probe AUC by itself measures decodability.

A small policy-mass change with no top-move changes and identical games across
doses is a null result for this global mean-pool intervention. It does not
establish that every representation is causally irrelevant.

## Method limits

- Spatially broadcast channel shift: blunt for local concepts.
- Mean-pool direction only: ignores move-location half of 1024-D probes.
- Empty-board self-play: most positions may be concept-irrelevant; effects can
  be diluted when averaged over every move.
- 1-visit policy: isolates the net, not full KataGo search.

Post / localized / counterfactual interventions are out of scope for this script.
