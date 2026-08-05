import json
import hashlib
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from daniele_experiment.validated_probe_pipeline import (
    ConceptSpec,
    _require_converged,
    _verify_fresh_development_fidelity_gate,
    _verify_fresh_probe_protocol,
    activation_views,
    build_run,
    feature_views,
    flat_index_from_internal_loc,
    nested_group_evaluation,
    load_concept_specs,
    prepare_run,
    train_run,
    validate_contract_specs,
)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_game_tree(root: Path, game_ids, *, moves_per_game=4, channels=2, board_size=3):
    for game_number, game_id in enumerate(game_ids):
        game_dir = root / game_id
        trunk_dir = game_dir / "trunkfinal"
        trunk_dir.mkdir(parents=True)
        moves = []
        labels = []
        for move_number in range(1, moves_per_game + 1):
            label = move_number % 2
            idx = (move_number + game_number) % (board_size * board_size)
            x, y = idx % board_size, idx // board_size
            internal_loc = (x + 1) + (board_size + 1) * (y + 1)
            moves.append({
                "move_number": move_number,
                "player": "b" if move_number % 2 else "w",
                "move_loc": internal_loc,
                "idx361": idx,
            })
            labels.append({
                "move_number": move_number,
                "analysis": {"signal": bool(label)},
            })
            activation = np.zeros((channels, board_size, board_size), dtype=np.float32)
            activation[0] = float(label) + game_number * 0.001
            activation[1, y, x] = 2.0 * label + 0.1
            np.save(trunk_dir / f"move_{move_number:03d}.npy", activation)
        _write_jsonl(game_dir / "moves.jsonl", moves)
        _write_jsonl(game_dir / "snorkel.jsonl", labels)


def _concepts_yaml(path: Path):
    path.write_text(yaml.safe_dump({
        "concepts": {
            "signal": {
                "type": "binary",
                "source": "signal",
                "feature_mode": "pre",
                "enabled": True,
            }
        }
    }))


def _stage_labels(games: Path, run: Path, game_ids):
    records = {}
    for game_id in game_ids:
        destination = run / "labels" / "games" / game_id
        destination.mkdir(parents=True)
        output = destination / "snorkel.jsonl"
        output.write_bytes(
            (games / game_id / "snorkel.jsonl").read_bytes()
        )
        records[game_id] = {
            "moves_sha256": hashlib.sha256(
                (games / game_id / "moves.jsonl").read_bytes()
            ).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_dir = Path(__file__).resolve().parent
    (run / "labels_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "pipeline": "validated_label_builder",
        "status": "complete",
        "run_manifest_sha256": sha(run / "manifest.json"),
        "split_manifest_sha256": sha(run / "splits.parquet"),
        "concepts_yaml_sha256": sha(run / "frozen_config" / "concepts.yaml"),
        "builder_source_sha256": sha(source_dir / "build_validated_labels.py"),
        "operational_definitions_source_sha256": sha(
            source_dir / "operational_definitions.py"
        ),
        "contracts": {},
        "games": records,
    }))


def test_activation_views_uses_flat_idx361_not_internal_move_loc():
    activation = np.stack([
        np.arange(9, dtype=np.float32).reshape(3, 3),
        np.arange(100, 109, dtype=np.float32).reshape(3, 3),
    ])
    global_mean, local = activation_views(activation, 5)
    np.testing.assert_allclose(global_mean, [4.0, 104.0])
    # idx361=5 means tensor coordinate (y=1, x=2).
    np.testing.assert_array_equal(local, activation[:, 1, 2])
    # KataGo's padded internal location for the same point would be 11 on 3x3;
    # it must not be accepted as a tensor-flat index.
    with pytest.raises(ValueError, match="outside flat board range"):
        activation_views(activation, 11)
    _, pass_local = activation_views(activation, 9)
    assert pass_local is None
    assert flat_index_from_internal_loc(11, 3) == 5


def test_prepare_is_deterministic_and_refuses_stale_run(tmp_path):
    games = tmp_path / "games"
    game_ids = [f"game-{index:02d}" for index in range(10)]
    _make_game_tree(games, game_ids, moves_per_game=1)
    for game_id in game_ids:
        (games / game_id / "snorkel.jsonl").unlink()
    concepts = tmp_path / "concepts.yaml"
    _concepts_yaml(concepts)

    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    kwargs = dict(
        seed=123,
        development_games=6,
        control_calibration_games=2,
        causal_test_games=2,
        outer_folds=3,
        inner_folds=2,
    )
    prepare_run(first, games, concepts, **kwargs)
    prepare_run(second, games, concepts, **kwargs)
    provenance = json.loads((first / "manifest.json").read_text())["activation_provenance"]
    assert provenance["checkpoint_attribution"]["status"] == (
        "not_recorded_in_generator_metadata"
    )
    assert provenance["sampled_checkpoint_activation_validation"]["status"] == (
        "not_performed"
    )
    first_splits = pd.read_parquet(first / "splits.parquet").sort_values("game_id")
    second_splits = pd.read_parquet(second / "splits.parquet").sort_values("game_id")
    pd.testing.assert_frame_equal(first_splits.reset_index(drop=True), second_splits.reset_index(drop=True))
    assert first_splits["split_role"].value_counts().to_dict() == {
        "development": 6,
        "control_calibration": 2,
        "causal_test": 2,
    }
    assert first_splits.loc[
        first_splits["split_role"] == "development", "outer_fold"
    ].nunique() == 3
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        prepare_run(first, games, concepts, **kwargs)


def test_prepare_allows_development_only_for_noncausal_exploration(tmp_path):
    games = tmp_path / "games"
    game_ids = [f"game-{index:02d}" for index in range(6)]
    _make_game_tree(games, game_ids, moves_per_game=1)
    for game_id in game_ids:
        (games / game_id / "snorkel.jsonl").unlink()
    concepts = tmp_path / "concepts.yaml"
    _concepts_yaml(concepts)
    run = tmp_path / "run"
    prepare_run(
        run,
        games,
        concepts,
        seed=123,
        development_games=6,
        control_calibration_games=0,
        causal_test_games=0,
        outer_folds=3,
        inner_folds=2,
    )
    splits = pd.read_parquet(run / "splits.parquet")
    assert set(splits["split_role"]) == {"development"}


def test_prepare_reserves_declared_fresh_cohort_for_holdouts(tmp_path):
    games = tmp_path / "games"
    development = [f"old-{index:02d}" for index in range(6)]
    fresh = [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"katateach:post-freeze-v1:{1000 + index}",
            )
        )
        for index in range(4)
    ]
    game_ids = development + fresh
    _make_game_tree(games, game_ids, moves_per_game=1)
    for game_id in game_ids:
        (games / game_id / "snorkel.jsonl").unlink()
    source_dir = Path(__file__).resolve().parent
    concepts = source_dir / "concepts.yaml"
    concepts_document = yaml.safe_load(concepts.read_text())
    enabled_concepts = [
        name
        for name, raw in concepts_document["concepts"].items()
        if raw.get("enabled", True)
    ]
    source_hashes = {
        "daniele_experiment/generate_games_dataset.py": hashlib.sha256(
            (source_dir / "generate_games_dataset.py").read_bytes()
        ).hexdigest(),
        "daniele_experiment/common_utils.py": hashlib.sha256(
            (source_dir / "common_utils.py").read_bytes()
        ).hexdigest(),
        "daniele_experiment/concepts.yaml": hashlib.sha256(
            concepts.read_bytes()
        ).hexdigest(),
    }
    protocol = tmp_path / "protocol.json"
    seed_digest = hashlib.sha256(
        ",".join(map(str, range(1000, 1004))).encode("ascii")
    ).hexdigest()
    protocol.write_text(json.dumps({
        "status": "frozen_before_fresh_data_generation",
        "frozen_at_utc": "2026-07-30T11:59:00+00:00",
        "historical_data_scope": {
            "development_games": 6,
            "development_game_ids_sha256": hashlib.sha256(
                ",".join(sorted(development)).encode("utf-8")
            ).hexdigest(),
        },
        "probes": {
            "concepts_config": "daniele_experiment/concepts.yaml",
            "development_games_only": 6,
            "representations": ["global", "local", "combined"],
            "concepts": enabled_concepts,
            "all_enabled_concepts_required": True,
            "outer_group_folds": 3,
            "inner_group_folds": 2,
            "C_values": [0.001, 0.01, 0.1, 1.0, 10.0],
            "selection_metric": "mean inner-fold average precision",
            "f1_threshold": "inner out-of-fold maximum F1",
            "probability_calibration": False,
            "quality_gate": None,
            "max_iter": 2000,
        },
        "fresh_holdout": {
            "cohort": "post-freeze-v1",
            "games": 4,
            "control_calibration_games": 2,
            "causal_test_games": 2,
            "game_seed_set_sha256": seed_digest,
            "split_seed": 123,
        },
        "checkpoint": {"sha256": "2" * 64},
        "source_sha256": source_hashes,
        "game_generation": {
            "board_size": 3,
            "device": "cpu",
            "torch_threads": 4,
            "initial_temperature": 1.2,
            "final_temperature": 0.8,
            "transition_moves": 60,
            "minimum_raw_policy_probability": 0.01,
            "top_k": 10,
            "resign_threshold": 0.1,
            "resign_consecutive_moves": 3,
            "maximum_moves": 400,
            "save_html": 0,
        },
    }))
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    for index, game_id in enumerate(fresh):
        (games / game_id / "game.sgf").write_text("(;GM[1]SZ[3])\n")
        (games / game_id / "meta.json").write_text(json.dumps({
            "uuid": game_id,
            "cohort": "post-freeze-v1",
            "created_at_utc": f"2026-07-30T12:00:{index:02d}+00:00",
            "protocol_manifest": {
                "path": str(protocol.resolve()),
                "sha256": protocol_hash,
                "verification": {
                    "status": "passed_before_model_load",
                    "protocol_sha256": protocol_hash,
                    "verified_source_sha256": source_hashes,
                    "shard_seed_first": 1000,
                    "shard_seed_last": 1003,
                    "shard_game_count": 4,
                },
            },
            "checkpoint": {
                "sha256": "2" * 64,
                "use_swa": False,
                "selected_weights": "raw_model",
            },
            "generator": {
                "source_sha256": source_hashes[
                    "daniele_experiment/generate_games_dataset.py"
                ],
                "common_utils_source_sha256": source_hashes[
                    "daniele_experiment/common_utils.py"
                ],
            },
            "rng": {
                "algorithm": "numpy.default_rng/PCG64",
                "game_seed": 1000 + index,
            },
            "board_size": 3,
            "device": "cpu",
            "torch_threads": 4,
            "initial_temperature": 1.2,
            "final_temperature": 0.8,
            "transition_moves": 60,
            "min_prob": 0.01,
            "top_k": 10,
            "resign_threshold": 0.1,
            "resign_consec": 3,
            "maximum_moves": 400,
            "save_html": 0,
            "policy_source": "direct_neural_policy_without_mcts",
            "immutable_outputs": True,
        }))
        for path in (games / game_id).rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        (games / game_id / "trunkfinal").chmod(0o555)
        (games / game_id).chmod(0o555)
    run = tmp_path / "run"
    manifest = prepare_run(
        run,
        games,
        concepts,
        seed=123,
        development_games=6,
        control_calibration_games=2,
        causal_test_games=2,
        outer_folds=3,
        inner_folds=2,
        fresh_holdout_cohort="post-freeze-v1",
    )
    splits = pd.read_parquet(run / "splits.parquet")
    role_by_game = splits.set_index("game_id")["split_role"].to_dict()
    assert {role_by_game[game_id] for game_id in development} == {"development"}
    assert {role_by_game[game_id] for game_id in fresh} == {
        "control_calibration", "causal_test"
    }
    assert manifest["split_assignment"] == (
        "legacy_development_fresh_cohort_holdouts"
    )
    assert manifest["fresh_holdout"]["protocol_manifest_sha256"] == protocol_hash
    frozen_specs = load_concept_specs(run / "frozen_config" / "concepts.yaml")
    with pytest.raises(ValueError, match="max_iter"):
        _verify_fresh_probe_protocol(
            manifest,
            specs=frozen_specs,
            representations=("global", "local", "combined"),
            C_values=(0.001, 0.01, 0.1, 1.0, 10.0),
            max_iter=1000,
        )


def test_post_and_delta_views_exclude_missing_next():
    frame = pd.DataFrame({
        "h_pre_global": [[1.0, 2.0], [3.0, 4.0]],
        "h_pre_local": [[5.0, 6.0], [7.0, 8.0]],
        "h_post_global": [[2.0, 4.0], None],
        "h_post_local": [[8.0, 10.0], None],
        "has_next": [True, False],
    })
    post = feature_views(
        frame, ConceptSpec("x", "binary", "x", feature_mode="post")
    )
    delta = feature_views(
        frame, ConceptSpec("x", "binary", "x", feature_mode="delta")
    )
    assert post["global"][1].tolist() == [True, False]
    assert post["local"][1].tolist() == [True, False]
    assert delta["combined"][1].tolist() == [True, False]
    np.testing.assert_allclose(delta["global"][0][0], [1.0, 2.0])
    np.testing.assert_allclose(delta["local"][0][0], [3.0, 4.0])


def test_fresh_training_requires_every_development_game_in_fidelity_gate(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    development = ["dev-a", "dev-b"]
    pd.DataFrame([
        *({"game_id": game, "split_role": "development"} for game in development),
        {"game_id": "cal", "split_role": "control_calibration"},
        {"game_id": "test", "split_role": "causal_test"},
    ]).to_parquet(run / "splits.parquet", index=False)
    validator = Path(__file__).resolve().parent / "checkpoint_activation_fidelity.py"
    validator_hash = hashlib.sha256(validator.read_bytes()).hexdigest()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "development_activation_fidelity_gate": {
            "required_before_training": True,
            "expected_games": 2,
            "absolute_max_error_tolerance": 1e-4,
        },
        "source_sha256": {
            "daniele_experiment/checkpoint_activation_fidelity.py": validator_hash,
        },
    }))
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    manifest = {
        "fresh_holdout": {
            "protocol_path": str(protocol.resolve()),
            "protocol_manifest_sha256": protocol_hash,
            "checkpoint_sha256": "a" * 64,
        }
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    build_manifest = {"games": 4}
    (run / "build_manifest.json").write_text(json.dumps(build_manifest))
    report = {
        "validator": "checkpoint_activation_fidelity",
        "validator_source_sha256": validator_hash,
        "status": "passed",
        "claim_scope": "compatibility, not original provenance",
        "run": {
            "manifest_sha256": hashlib.sha256(
                (run / "manifest.json").read_bytes()
            ).hexdigest(),
            "build_manifest_sha256": hashlib.sha256(
                (run / "build_manifest.json").read_bytes()
            ).hexdigest(),
        },
        "checkpoint": {"sha256": "a" * 64},
        "sampling": {
            "algorithm": "one_deterministic_position_per_game_v1",
            "split_role_filter": "development",
            "requested_sample_count": 2,
        },
        "aggregate_errors": {"sample_count": 2, "max_abs_error": 1e-5},
        "tolerance": {"absolute_tolerance": 1e-4},
        "samples": [{"game_id": game} for game in development],
    }
    (run / "checkpoint_activation_fidelity.json").write_text(json.dumps(report))
    verified = _verify_fresh_development_fidelity_gate(
        run, manifest, build_manifest
    )
    assert verified["sample_count"] == 2
    report["samples"] = [{"game_id": "dev-a"}, {"game_id": "dev-a"}]
    (run / "checkpoint_activation_fidelity.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="every development game"):
        _verify_fresh_development_fidelity_gate(run, manifest, build_manifest)


def test_contract_specs_fail_closed_on_semantic_drift():
    valid = ConceptSpec(
        "urgency_peak",
        "quantile",
        "regional_policy_peak",
        contract_id="regional_policy_peak@2",
        feature_mode="pre",
        q=0.15,
        direction="high",
        no_drop=True,
    )
    validate_contract_specs([valid])

    with pytest.raises(ValueError, match="source .* disagrees"):
        validate_contract_specs([
            ConceptSpec(
                "urgency_peak", "quantile", "wrong_source",
                contract_id="regional_policy_peak@2", feature_mode="pre",
                q=0.15, direction="high", no_drop=True,
            )
        ])
    with pytest.raises(ValueError, match="feature_mode"):
        validate_contract_specs([
            ConceptSpec(
                "tenuki", "binary", "tenuki_distance6",
                contract_id="tenuki_distance6@2", feature_mode="post",
            )
        ])
    with pytest.raises(ValueError, match="positive_quantile"):
        validate_contract_specs([
            ConceptSpec(
                "urgency_peak", "quantile", "regional_policy_peak",
                contract_id="regional_policy_peak@2", feature_mode="pre",
                q=0.10, direction="high", no_drop=True,
            )
        ])
    with pytest.raises(ValueError, match="no_drop=true"):
        validate_contract_specs([
            ConceptSpec(
                "urgency_peak", "quantile", "regional_policy_peak",
                contract_id="regional_policy_peak@2", feature_mode="pre",
                q=0.15, direction="high", no_drop=False,
            )
        ])


def test_iteration_cap_is_rejected_as_nonconverged():
    class CappedModel:
        n_iter_ = np.asarray([7])

    with pytest.raises(RuntimeError, match="did not converge"):
        _require_converged(CappedModel(), max_iter=7, context="test probe")


def test_build_rejects_labels_after_raw_moves_change(tmp_path):
    games = tmp_path / "games"
    game_ids = [f"game-{index:02d}" for index in range(10)]
    _make_game_tree(games, game_ids, moves_per_game=1)
    concepts = tmp_path / "concepts.yaml"
    _concepts_yaml(concepts)
    run = tmp_path / "run"
    prepare_run(
        run,
        games,
        concepts,
        development_games=6,
        control_calibration_games=2,
        causal_test_games=2,
        outer_folds=2,
        inner_folds=2,
    )
    _stage_labels(games, run, game_ids)
    changed = games / game_ids[0] / "moves.jsonl"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Raw moves hash mismatch"):
        build_run(run, concepts=["signal"])


def test_training_rejects_activation_changed_after_build(tmp_path):
    games = tmp_path / "games"
    game_ids = [f"game-{index:02d}" for index in range(10)]
    _make_game_tree(games, game_ids, moves_per_game=1)
    concepts = tmp_path / "concepts.yaml"
    _concepts_yaml(concepts)
    run = tmp_path / "run"
    prepare_run(
        run,
        games,
        concepts,
        development_games=6,
        control_calibration_games=2,
        causal_test_games=2,
        outer_folds=2,
        inner_folds=2,
    )
    _stage_labels(games, run, game_ids)
    build_run(run, concepts=["signal"])
    changed = games / game_ids[0] / "trunkfinal" / "move_001.npy"
    changed.write_bytes(changed.read_bytes() + b"changed-after-build")
    with pytest.raises(ValueError, match="changed after build"):
        train_run(run, concepts=["signal"], C_values=[0.1], max_iter=50)


def test_nested_cv_has_disjoint_game_folds_and_honest_metrics():
    rng = np.random.default_rng(7)
    rows = []
    features = []
    for game_number in range(12):
        for sample in range(8):
            label = sample % 2
            rows.append({
                "row_id": f"g{game_number}:{sample}",
                "game_id": f"g{game_number}",
                "move_number": sample + 1,
                "split_role": "development",
                "outer_fold": game_number % 3,
                "label_signal": label,
                "rawval_signal": float(label),
            })
            features.append([2.0 * label + rng.normal(scale=0.2), rng.normal()])
    frame = pd.DataFrame(rows)
    X = np.asarray(features, dtype=np.float32)
    folds, predictions = nested_group_evaluation(
        frame,
        ConceptSpec("signal", "binary", "signal"),
        X,
        np.ones(len(frame), dtype=bool),
        outer_folds=3,
        inner_folds=2,
        C_values=[0.1, 1.0],
        seed=99,
        max_iter=200,
    )
    assert len(folds) == 3
    assert len(predictions) == len(frame)
    assert {row["row_id"] for row in predictions} == set(frame["row_id"])
    for fold in folds:
        assert set(fold["train_games"]).isdisjoint(fold["test_games"])
        assert 0.0 <= fold["roc_auc"] <= 1.0
        assert 0.0 <= fold["average_precision"] <= 1.0
        assert 0.0 <= fold["f1"] <= 1.0
        assert 0.0 <= fold["balanced_accuracy"] <= 1.0
        for inner in fold["inner_selection"]["folds"]:
            assert set(inner["train_games"]).isdisjoint(inner["validation_games"])


def test_prepare_build_train_integration_and_development_only_final_fit(tmp_path):
    games = tmp_path / "games"
    game_ids = [f"game-{index:02d}" for index in range(12)]
    _make_game_tree(games, game_ids)
    concepts = tmp_path / "concepts.yaml"
    _concepts_yaml(concepts)
    run = tmp_path / "validated-run"
    prepare_run(
        run,
        games,
        concepts,
        seed=44,
        development_games=8,
        control_calibration_games=2,
        causal_test_games=2,
        outer_folds=2,
        inner_folds=2,
    )
    _stage_labels(games, run, game_ids)
    # The validated builder must use only mirrored, run-scoped labels.
    for game_id in game_ids:
        (games / game_id / "snorkel.jsonl").unlink()
    build_run(run, concepts=["signal"])
    build_manifest = json.loads((run / "build_manifest.json").read_text())
    input_provenance = build_manifest["input_provenance"]
    assert input_provenance["trunk_file_count"] == len(game_ids) * 4
    assert input_provenance["trunk_total_bytes"] > 0
    assert len(input_provenance["trunk_identity_bytes_sha256"]) == 64
    assert set(input_provenance["games"]) == set(game_ids)
    assert all(
        record["file_count"] == 4
        and len(record["identity_bytes_sha256"]) == 64
        and len(record["files"]) == 4
        and all(
            len(leaf["sha256"]) == 64
            for leaf in record["files"].values()
        )
        and Path(record["source_game_dir"]).is_absolute()
        for record in input_provenance["games"].values()
    )
    dataset = pd.read_parquet(run / "dataset.parquet")
    first_move = json.loads((games / dataset.iloc[0]["game_id"] / "moves.jsonl").read_text().splitlines()[0])
    source = np.load(
        games / dataset.iloc[0]["game_id"] / "trunkfinal" / "move_001.npy"
    )
    idx = first_move["idx361"]
    y, x = divmod(idx, source.shape[2])
    np.testing.assert_array_equal(dataset.iloc[0]["h_pre_local"], source[:, y, x])

    train_run(
        run,
        concepts=["signal"],
        representations=["global", "local", "combined"],
        C_values=[0.1],
        max_iter=200,
    )
    splits = pd.read_parquet(run / "splits.parquet")
    development_games = set(
        splits.loc[splits["split_role"] == "development", "game_id"]
    )
    held_out_games = set(
        splits.loc[splits["split_role"] != "development", "game_id"]
    )
    for representation, expected_features in (("global", 2), ("local", 2), ("combined", 4)):
        metadata_path = run / "probes" / representation / "probe_signal.meta.json"
        metadata = json.loads(metadata_path.read_text())
        trained_games = set(metadata["final_fit"]["training_game_ids"])
        assert trained_games == development_games
        assert trained_games.isdisjoint(held_out_games)
        assert metadata["training_role"] == "development"
        assert metadata["n_features"] == expected_features
        probe = joblib.load(run / "probes" / representation / "probe_signal.joblib")
        assert probe.coef_.shape[1] == expected_features
    results = pd.read_parquet(run / "nested_cv_results.parquet")
    assert set(results["representation"]) == {"global", "local", "combined"}
    assert {"roc_auc", "average_precision", "f1", "balanced_accuracy"} <= set(results.columns)
    assert (run / "training_manifest.json").is_file()
    with pytest.raises(FileExistsError, match="Refusing to train"):
        train_run(run, concepts=["signal"], C_values=[0.1], max_iter=100)
