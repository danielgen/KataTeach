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
from typing import Sequence

import yaml


def load_tags(ontology_path: Path) -> tuple[Sequence[str], Sequence[str]]:
    """Return lists of global and spatial tag names from the ontology."""
    with ontology_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    global_tags = [t["name"] for t in data.get("tags", []) if t.get("type") == "global"]
    spatial_tags = [t["name"] for t in data.get("tags", []) if t.get("type") == "spatial"]
    return global_tags, spatial_tags


def build_label_page(sgf_path: Path, html_path: Path, ontology_path: Path) -> None:
    """Create an interactive labeling web page for ``sgf_path``."""
    sgf_text = sgf_path.read_text(encoding="utf-8")
    global_tags, spatial_tags = load_tags(ontology_path)

    sgf_js = json.dumps(sgf_text)
    globals_js = json.dumps(list(global_tags))
    spatial_js = json.dumps(list(spatial_tags))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8' />
<title>Go Position Labeler</title>
<script src='https://cdnjs.cloudflare.com/ajax/libs/wgo/2.3.0/wgo.min.js'></script>
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
<button id='export'>Export Labels</button>
<script>
const SGF = {sgf_js};
const GLOBAL_TAGS = {globals_js};
const SPATIAL_TAGS = {spatial_js};
let player = new WGo.BasicPlayer(document.getElementById('board'), {{ sgf: SGF }});
let currentMove = 0;
let labels = {{}};  // move index -> tag mapping
let selectedSpatial = null;

function renderForm() {{
  const form = document.getElementById('tag_form');
  form.innerHTML = '';
  form.innerHTML += '<h3>Global tags</h3>';
  GLOBAL_TAGS.forEach(tag => {{
    const checked = labels[currentMove] && labels[currentMove][tag] ? 'checked' : '';
    form.innerHTML += `<div class="tag-block"><label><input type="checkbox" data-tag="${{tag}}" ${checked}/> ${{tag}}</label></div>`;
  }});
  form.innerHTML += '<h3>Spatial tags</h3>';
  SPATIAL_TAGS.forEach(tag => {{
    const pts = (labels[currentMove] && labels[currentMove][tag]) || [];
    const checked = selectedSpatial === tag ? 'checked' : '';
    form.innerHTML += `<div class="tag-block"><label><input type="radio" name="spatial" value="${{tag}}" ${checked}/> ${{tag}}</label> <span id="${{tag}}_pts">${{pts.map(p => '(' + p[0] + ',' + p[1] + ')').join(' ')}}</span></div>`;
  }});
}}

function renderMarkers() {{
  player.board.removeAllObjects();
  const tags = labels[currentMove] || {{}};
  Object.keys(tags).forEach(tag => {{
    const pts = tags[tag];
    if(Array.isArray(pts)) {{
      pts.forEach(pt => player.board.addObject({{ x: pt[0], y: pt[1], type: 'MA' }}));
    }}
  }});
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

player.board.addEventListener('click', (x, y) => {{
  if(selectedSpatial === null) return;
  labels[currentMove] = labels[currentMove] || {{}};
  labels[currentMove][selectedSpatial] = labels[currentMove][selectedSpatial] || [];
  labels[currentMove][selectedSpatial].push([x, y]);
  renderForm();
  renderMarkers();
}});

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
</script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SGF labeling web page")
    parser.add_argument("sgf", type=Path, help="Input SGF file")
    parser.add_argument("html", type=Path, help="Output HTML file")
    parser.add_argument("--ontology", type=Path, default=Path("daniele_experiment/configs/ontology.yaml"), help="Ontology YAML path")
    args = parser.parse_args()
    build_label_page(args.sgf, args.html, args.ontology)


if __name__ == "__main__":
    main()
