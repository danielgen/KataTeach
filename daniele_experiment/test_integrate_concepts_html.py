import json

from daniele_experiment.integrate_concepts_html import (
    commentary_document,
    load_commentary_map,
    process_game,
)


def test_load_commentary_supports_canonical_json(tmp_path):
    game_id = "game-1"
    source = tmp_path / "probes" / "html_data" / game_id
    source.mkdir(parents=True)
    (source / "commentary.json").write_text(json.dumps({
        "game_id": game_id,
        "moves": [{
            "move_number": 3,
            "comment": "A forcing move.",
            "concepts_used": ["forcing"],
        }],
    }))

    result = load_commentary_map(str(tmp_path / "probes"), game_id, None)

    assert result == {3: {"comment": "A forcing move.", "concepts_used": ["forcing"]}}


def test_commentary_document_is_sorted():
    result = commentary_document("game-1", {
        2: {"comment": "Second", "concepts_used": []},
        1: {"comment": "First", "concepts_used": ["tenuki"]},
    })

    assert result["game_id"] == "game-1"
    assert [move["move_number"] for move in result["moves"]] == [1, 2]


def test_process_game_exports_self_contained_artifacts(tmp_path):
    game_id = "game-1"
    game_dir = tmp_path / "games" / game_id
    source = tmp_path / "probes" / "html_data" / game_id
    game_dir.mkdir(parents=True)
    source.mkdir(parents=True)
    (game_dir / "viz.html").write_text("<html><body></body></html>")
    concepts = {
        "game_id": game_id,
        "concept_names": ["atari"],
        "moves": [{
            "move_number": 1,
            "player": "b",
            "scores": {"atari": 1.2},
            "deltas": {"atari": 0.3},
            "probs": {"atari": 0.8},
            "score_percentiles": {"atari": 1.0},
            "top_concepts": ["atari"],
            "top_delta_concepts": [],
            "commentary": {"comment": "Atari.", "concepts_used": ["atari"]},
        }],
    }
    (source / "concepts_with_commentary.json").write_text(json.dumps(concepts))
    meta = {"atari": {"auc": 0.91}}

    assert process_game(game_dir, str(tmp_path / "probes"), meta)
    source_commentary = json.loads((source / "commentary.json").read_text())
    assert source_commentary["moves"][0]["comment"] == "Atari."
    assert json.loads((game_dir / "commentary.json").read_text())["moves"][0]["comment"] == "Atari."
    assert json.loads((game_dir / "concepts.json").read_text()) == concepts
    assert json.loads((game_dir / "concepts_meta.json").read_text()) == meta
    html = (game_dir / "viz.html").read_text()
    assert "Top concepts:" in html
    assert "Probability" in html
    assert "Move Commentary" in html
