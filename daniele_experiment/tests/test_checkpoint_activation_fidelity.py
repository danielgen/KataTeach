import json
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daniele_experiment.checkpoint_activation_fidelity import (
    CLAIM_SCOPE,
    OUTPUT_NAME,
    FidelityToleranceError,
    FidelityValidationError,
    LoadedCheckpoint,
    PositionCandidate,
    compare_activations,
    deterministic_one_per_game_sample,
    deterministic_stratified_sample,
    load_run_context,
    sha256_file,
    validate_checkpoint_activation_fidelity,
)


def _candidate(root: Path, role: str, phase: str, index: int) -> PositionCandidate:
    return PositionCandidate(
        game_id=f"{role}-{phase}-{index:02d}",
        move_number=index + 1,
        split_role=role,
        game_phase=phase,
        board_size=19,
        activation_path=root / f"{role}-{phase}-{index:02d}.npy",
    )


def test_deterministic_stratified_selection_is_order_independent_and_balanced(tmp_path):
    roles = ("development", "control_calibration", "causal_test")
    phases = ("opening", "middle", "endgame")
    candidates = [
        _candidate(tmp_path, role, phase, index)
        for role in roles
        for phase in phases
        for index in range(4)
    ]

    selected = deterministic_stratified_sample(candidates, 9, seed=712)
    reversed_selected = deterministic_stratified_sample(
        list(reversed(candidates)), 9, seed=712
    )

    assert selected == reversed_selected
    assert len({item.stratum for item in selected}) == 9
    assert deterministic_stratified_sample(candidates, 9, seed=713) != selected
    with pytest.raises(ValueError, match="only 36 candidates"):
        deterministic_stratified_sample(candidates, 37, seed=1)


def test_one_per_game_selection_covers_every_game_in_requested_role(tmp_path):
    candidates = []
    for game_index in range(5):
        for move_index in range(3):
            candidates.append(PositionCandidate(
                game_id=f"dev-{game_index}",
                move_number=move_index + 1,
                split_role="development",
                game_phase="opening",
                board_size=19,
                activation_path=tmp_path / f"dev-{game_index}-{move_index}.npy",
            ))
    candidates.append(_candidate(tmp_path, "causal_test", "opening", 0))
    selected = deterministic_one_per_game_sample(
        candidates, split_role="development", seed=17
    )
    assert len(selected) == 5
    assert len({item.game_id for item in selected}) == 5
    assert {item.split_role for item in selected} == {"development"}
    assert selected == deterministic_one_per_game_sample(
        list(reversed(candidates)), split_role="development", seed=17
    )


def test_compare_activations_reports_exact_max_mean_and_rms():
    saved = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    replayed = np.array([[1.0, 1.0], [5.0, 4.0]], dtype=np.float64)

    comparison = compare_activations(saved, replayed)

    assert comparison.element_count == 4
    assert comparison.max_abs_error == pytest.approx(2.0)
    assert comparison.mean_abs_error == pytest.approx(0.75)
    assert comparison.rms_error == pytest.approx(np.sqrt(5.0 / 4.0))
    assert comparison.sum_abs_error == pytest.approx(3.0)
    assert comparison.sum_squared_error == pytest.approx(5.0)
    with pytest.raises(FidelityValidationError, match="shape mismatch"):
        compare_activations(np.zeros((2, 2)), np.zeros((4,)))
    with pytest.raises(FidelityValidationError, match="NaN or infinity"):
        compare_activations(np.array([np.nan]), np.array([0.0]))


class _FakeBoard:
    def __init__(self):
        self.pla = 1


class _FakeState:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = _FakeBoard()
        self.moves = []

    def play(self, player, move_loc):
        assert player == self.board.pla
        self.moves.append((player, move_loc))
        self.board.pla = 2 if player == 1 else 1


def _fake_activation(state, _model):
    value = len(state.moves) + sum(move for _player, move in state.moves) / 100.0
    return np.full((2, state.board_size, state.board_size), value, dtype=np.float32)


def _fake_model_loader(_checkpoint, device, board_size):
    return LoadedCheckpoint(
        model={"device": device, "board_size": board_size},
        config={"model_kind": "fake", "board_size": board_size},
        use_swa=False,
        selected_weights="raw_model",
    )


def _make_prepared_run(tmp_path: Path, *, mismatch: float = 0.0):
    games = tmp_path / "games"
    games.mkdir()
    game_specs = [
        ("game-dev", "development"),
        ("game-cal", "control_calibration"),
        ("game-test", "causal_test"),
    ]
    split_rows = []
    for game_id, role in game_specs:
        game_dir = games / game_id
        trunk_dir = game_dir / "trunkfinal"
        trunk_dir.mkdir(parents=True)
        (game_dir / "meta.json").write_text(
            json.dumps({"uuid": game_id, "board_size": 3})
        )
        state = _FakeState(3)
        rows = []
        for move_number, move_loc in enumerate((11, 12, 13), start=1):
            player = "b" if state.board.pla == 1 else "w"
            saved = _fake_activation(state, None) + mismatch
            np.save(trunk_dir / f"move_{move_number:03d}.npy", saved)
            rows.append(
                {
                    "move_number": move_number,
                    "player": player,
                    "move_loc": move_loc,
                }
            )
            state.play(state.board.pla, move_loc)
        (game_dir / "moves.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        split_rows.append({"game_id": game_id, "split_role": role})

    run = tmp_path / "run"
    run.mkdir()
    splits_path = run / "splits.parquet"
    pd.DataFrame(split_rows).to_parquet(splits_path, index=False)
    manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "source_games_dir": str(games.resolve()),
        "artifacts": {
            "splits": "splits.parquet",
            "splits_sha256": sha256_file(splits_path),
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fake checkpoint bytes")
    return run, checkpoint


def test_fake_replay_writes_immutable_hash_bound_compatibility_report(tmp_path):
    run, checkpoint = _make_prepared_run(tmp_path)

    report = validate_checkpoint_activation_fidelity(
        run,
        checkpoint,
        sample_count=6,
        seed=2026,
        device="fake:0",
        absolute_tolerance=0.0,
        model_loader=_fake_model_loader,
        state_factory=_FakeState,
        activation_evaluator=_fake_activation,
        player_values={"b": 1, "w": 2},
        replay_rules={"name": "fake TT rules"},
    )

    output = run / OUTPUT_NAME
    on_disk = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert on_disk["claim_scope"] == CLAIM_SCOPE
    assert "not proof" in on_disk["claim_scope"]
    assert on_disk["run"]["stage_at_validation"] == "prepared"
    assert on_disk["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert on_disk["checkpoint"]["use_swa"] is False
    assert on_disk["checkpoint"]["selected_weights"] == "raw_model"
    assert on_disk["checkpoint"]["model_config"]["model_kind"] == "fake"
    assert on_disk["sampling"]["requested_sample_count"] == 6
    assert len(on_disk["samples"]) == 6
    assert all(item["saved_activation_file_sha256"] for item in on_disk["samples"])
    assert on_disk["aggregate_errors"]["max_abs_error"] == 0.0
    assert stat.S_IMODE(output.stat().st_mode) == 0o444

    with pytest.raises(FileExistsError, match="overwrite fidelity report"):
        validate_checkpoint_activation_fidelity(
            run,
            checkpoint,
            sample_count=1,
            seed=2026,
            device="fake:0",
            absolute_tolerance=0.0,
            model_loader=_fake_model_loader,
            state_factory=_FakeState,
            activation_evaluator=_fake_activation,
            player_values={"b": 1, "w": 2},
            replay_rules={"name": "fake TT rules"},
        )


def test_tolerance_failure_is_recorded_then_raises(tmp_path):
    run, checkpoint = _make_prepared_run(tmp_path, mismatch=0.25)

    with pytest.raises(FidelityToleranceError) as caught:
        validate_checkpoint_activation_fidelity(
            run,
            checkpoint,
            sample_count=3,
            seed=7,
            device="cpu",
            absolute_tolerance=0.01,
            model_loader=_fake_model_loader,
            state_factory=_FakeState,
            activation_evaluator=_fake_activation,
            player_values={"b": 1, "w": 2},
            replay_rules={"name": "fake TT rules"},
        )

    output = run / OUTPUT_NAME
    report = json.loads(output.read_text())
    assert caught.value.output_path == output.resolve()
    assert report["status"] == "failed_tolerance"
    assert report["aggregate_errors"]["max_abs_error"] == pytest.approx(0.25)
    assert report["tolerance"]["absolute_tolerance"] == 0.01


def test_archive_namespace_is_rejected_before_it_can_be_a_run_input(tmp_path):
    archived_run = tmp_path / "archive" / "old-run"
    archived_run.mkdir(parents=True)
    with pytest.raises(FidelityValidationError, match="archive namespace"):
        load_run_context(archived_run)
