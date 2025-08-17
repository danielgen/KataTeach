"""Generate a simple HTML page to label Go board positions from an SGF file.

This script parses an SGF game record and emits a self-contained web page
that lets a human annotator step through each position and assign the
experiment's concept tags.  Tags are defined in ``configs/ontology.yaml``.

The resulting page uses the WGo.js library (loaded from a CDN) to render the
board.  Global tags are presented as checkboxes, while spatial tags can be
annotated by selecting the tag and clicking points on the board.  Labels can
be downloaded as a JSON file for further processing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Mapping

import yaml


def load_tags(ontology_path: Path) -> tuple[Sequence[str], Sequence[str]]:
    """Return lists of global and spatial tag names from the ontology."""
    with ontology_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    global_tags = [t["name"] for t in data.get("tags", []) if t.get("type") == "global"]
    spatial_tags = [t["name"] for t in data.get("tags", []) if t.get("type") == "spatial"]
    return global_tags, spatial_tags


def build_label_page(
    sgf_path: Path,
    html_path: Path,
    ontology_path: Path,
    policy_path: Path | None = None,
) -> None:
    """Create an interactive labeling web page for ``sgf_path``.

    Parameters
    ----------
    sgf_path:
        Input SGF game record.
    html_path:
        Destination HTML file.
    ontology_path:
        Path to the ontology YAML used for tag definitions.
    policy_path:
        Optional JSON file containing pre-computed policy suggestions for
        each move.  The file should map move indices to lists of objects
        with ``move`` and ``winrate`` fields.
    """

    sgf_text = sgf_path.read_text(encoding="utf-8")
    global_tags, spatial_tags = load_tags(ontology_path)

    policy: Mapping[str, object]
    if policy_path is not None and policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    else:
        policy = {}

    sgf_js = json.dumps(sgf_text)
    globals_js = json.dumps(list(global_tags))
    spatial_js = json.dumps(list(spatial_tags))
    policy_js = json.dumps(policy)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8' />
<title>Go Position Labeler</title>
<script src='wgo.min.js'></script>
<style>
  #board {{ width: 400px; margin-bottom: 10px; }}
  .tag-block {{ margin: 5px 0; }}
</style>
</head>
<body>
<div id='board'></div>
<div>
  <button id='prev'>Prev</button>
  <span id='move_idx'>0</span>
  <button id='next'>Next</button>
</div>
<form id='tag_form'></form>
<div id='policy_suggestions'></div>
<button id='export'>Export Labels</button>
<script>
const SGF = {sgf_js};
const GLOBAL_TAGS = {globals_js};
const SPATIAL_TAGS = {spatial_js};
const POLICY = {policy_js};
let player = new WGo.SimplePlayer(document.getElementById('board'), {{ sgf: SGF }});
let currentMove = 0;
let labels = {{}};  // move index -> tag mapping
let selectedSpatial = null;

function renderForm() {{
  const form = document.getElementById('tag_form');
  form.innerHTML = '';
  form.innerHTML += '<h3>Global tags</h3>';
  GLOBAL_TAGS.forEach(tag => {{
    const checked = labels[currentMove] && labels[currentMove][tag] ? 'checked' : '';
    form.innerHTML += `<div class="tag-block"><label><input type="checkbox" data-tag="${{tag}}" ${{checked}}/> ${{tag}}</label></div>`;
  }});
  form.innerHTML += '<h3>Spatial tags</h3>';
  SPATIAL_TAGS.forEach(tag => {{
    const pts = (labels[currentMove] && labels[currentMove][tag]) || [];
    const checked = selectedSpatial === tag ? 'checked' : '';
    form.innerHTML += `<div class="tag-block"><label><input type="radio" name="spatial" value="${{tag}}" ${{checked}}/> ${{tag}}</label> <span id="${{tag}}_pts">${{pts.map(p => '(' + p[0] + ',' + p[1] + ')').join(' ')}}</span></div>`;
  }});
}}

function renderMarkers() {{
  if(player.board && player.board.removeAllObjects) {{
    player.board.removeAllObjects();
    const tags = labels[currentMove] || {{}};
    Object.keys(tags).forEach(tag => {{
      const pts = tags[tag];
      if(Array.isArray(pts)) {{
        pts.forEach(pt => player.board.addObject({{ x: pt[0], y: pt[1], type: 'MA' }}));
      }}
    }});
  }}
}}

function renderPolicy() {{
  const div = document.getElementById('policy_suggestions');
  const opts = POLICY[currentMove] || [];
  if(opts.length === 0) {{
    div.textContent = '';
    return;
  }}
  const lines = opts.map(o => `${{o.move}} (${{(o.winrate * 100).toFixed(1)}}%)`);
  div.innerHTML = '<h3>Top policy moves</h3>' + lines.join('<br/>');
}}

document.getElementById('tag_form').addEventListener('change', e => {{
  const tag = e.target.getAttribute('data-tag');
  if(tag) {{
    labels[currentMove] = labels[currentMove] || {{}};
    labels[currentMove][tag] = e.target.checked;
  }} else if(e.target.name === 'spatial') {{
    selectedSpatial = e.target.value;
  }}
}});

if(player.board && player.board.addEventListener) {{
  player.board.addEventListener('click', (x, y) => {{
    if(selectedSpatial === null) return;
    labels[currentMove] = labels[currentMove] || {{}};
    labels[currentMove][selectedSpatial] = labels[currentMove][selectedSpatial] || [];
    labels[currentMove][selectedSpatial].push([x, y]);
    renderForm();
    renderMarkers();
  }});
}}

function updateMoveDisplay() {{
  document.getElementById('move_idx').textContent = currentMove;
}}

function gotoMove(idx) {{
  if(idx < 0 || idx >= player.kifu.nodes.length) return;
  while(currentMove < idx) {{ player.next(); currentMove++; }}
  while(currentMove > idx) {{ player.previous(); currentMove--; }}
  updateMoveDisplay();
  renderForm();
  renderMarkers();
  renderPolicy();
}}

document.getElementById('next').onclick = () => gotoMove(currentMove + 1);
document.getElementById('prev').onclick = () => gotoMove(currentMove - 1);

document.getElementById('export').onclick = () => {{
  const data = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(labels));
  const a = document.createElement('a');
  a.setAttribute('href', data);
  a.setAttribute('download', 'labels.json');
  a.click();
}};

renderForm();
updateMoveDisplay();
renderPolicy();
</script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SGF labeling web page")
    parser.add_argument("sgf", type=Path, help="Input SGF file")
    parser.add_argument("html", type=Path, help="Output HTML file")
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "ontology.yaml",
        help="Ontology YAML path",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Optional JSON file with policy suggestions",
    )
    args = parser.parse_args()
    build_label_page(args.sgf, args.html, args.ontology, args.policy)


if __name__ == "__main__":
    main()
