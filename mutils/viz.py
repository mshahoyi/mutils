"""Lightweight visualisation helpers."""

from __future__ import annotations

import html
import json
import uuid
from typing import Any, Mapping, Sequence

__all__ = ["token_heatmap_html", "token_heatmap_explorer"]

POS_RGB, NEG_RGB = (214, 39, 40), (31, 119, 180)  # red for > 0, blue for < 0


def token_heatmap_html(tokens: Sequence[str], values: Sequence[float], vmax: float | None = None, label: str | None = None) -> str:
    """HTML snippet showing `tokens` inline, each shaded by its value (red > 0, blue < 0, white = 0).

    Shade opacity is |value| / vmax, clipped to 1 — pass a shared `vmax` to compare snippets.
    Hover a token to see its value. Snippets are pure ASCII, so they can be joined and written
    straight to a .html file, or shown in a notebook with `IPython.display.HTML`.
    """
    assert len(tokens) == len(values), f"{len(tokens)} tokens but {len(values)} values"
    values = [float(v) for v in values]
    vmax = vmax or max(map(abs, values), default=0.0) or 1.0
    spans = []
    for tok, v in zip(tokens, values):
        r, g, b = POS_RGB if v > 0 else NEG_RGB
        text = html.escape(tok).replace("\n", "↵\n").encode("ascii", "xmlcharrefreplace").decode()
        spans.append(f'<span title="{v:.3f}" style="background:rgba({r},{g},{b},{min(abs(v) / vmax, 1):.2f})">{text}</span>')
    header = f'<div style="font-weight:bold">{html.escape(label)}</div>' if label else ""
    return f'<div style="font-family:monospace;white-space:pre-wrap;line-height:1.6;margin:12px 0">{header}{"".join(spans)}</div>'


_EXPLORER_TEMPLATE = r"""<div id="__UID__">
<style>
#__UID__ {font-family: system-ui, sans-serif}
#__UID__ .controls {display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: 8px 0}
#__UID__ .legend {color: #666; font-size: 12px; margin-bottom: 12px}
#__UID__ .tokens {font-family: monospace; white-space: pre-wrap; line-height: 1.7}
#__UID__ .tok {position: relative}
#__UID__ .tok:hover {outline: 1px solid #333}
#__UID__ .tok:hover::after {content: attr(data-v); position: absolute; left: 0; bottom: 100%; z-index: 1; pointer-events: none;
  background: #222; color: #fff; font: 11px monospace; padding: 1px 4px; border-radius: 3px; white-space: nowrap}
#__UID__ .val {display: none}
#__UID__ .show-values .tok {display: inline-flex; flex-direction: column; vertical-align: top; margin: 0 5px 6px 0}
#__UID__ .show-values .val {display: block; font-size: 9px; color: #444; text-align: center}
</style>
__TITLE__
<div class="controls"><label><input type="checkbox"> show values</label></div>
<div class="legend">red &gt; 0 &gt; blue, full colour at |value| &ge; __VMAX__; hover a token to see its value</div>
<div class="tokens"></div>
<script>
(() => {
  const D = __DATA__, root = document.getElementById("__UID__");
  const out = root.querySelector(".tokens"), box = root.querySelector("input[type=checkbox]");
  const selects = D.dims.map(dim => {
    const sel = document.createElement("select"), label = document.createElement("label");
    for (const v of new Set(D.rows.map(r => r[dim]))) sel.add(new Option(v, v));
    if (dim in D.initial) sel.value = D.initial[dim];
    sel.onchange = render;
    label.append(dim + " ", sel);
    return sel;
  });
  root.querySelector(".controls").prepend(...selects.map(s => s.parentElement));
  box.onchange = () => out.classList.toggle("show-values", box.checked);

  function token(t, v) {  // newlines are drawn as a visible symbol inside the token, then broken after it
    const [r, g, b] = v > 0 ? D.pos : D.neg, [span, text, val] = ["tok", "txt", "val"].map(cls => {
      const el = document.createElement("span");
      el.className = cls;
      return el;
    });
    span.dataset.v = v.toFixed(3);
    text.style.background = `rgba(${r},${g},${b},${Math.min(Math.abs(v) / D.vmax, 1).toFixed(2)})`;
    text.textContent = t.replaceAll("\n", "\u21b5");
    val.textContent = v.toFixed(2);  // shown under the token by the "show values" checkbox
    span.append(text, val);
    return [span, "\n".repeat((t.match(/\n/g) || []).length)];
  }
  function render() {
    const row = D.rows.find(r => D.dims.every((d, i) => r[d] === selects[i].value));
    if (!row) return out.replaceChildren("no data for this selection");
    out.replaceChildren(...row.tokens.flatMap((t, i) => token(t, row.values[i])));
  }
  render();
})();
</script>
</div>"""


def token_heatmap_explorer(
    rows: Sequence[Mapping[str, Any]],
    dims: Sequence[str],
    vmax: float | None = None,
    initial: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """Interactive HTML for browsing many token heatmaps, one at a time.

    Each row has a value for every key in `dims` (e.g. layer, system prompt, user prompt) plus `tokens` and `values`.
    The page gets one dropdown per dim (options in first-seen order, preselected from `initial`) and shows the matching
    row shaded like `token_heatmap_html`. Hovering a token shows its value; a checkbox prints values under the tokens.
    `vmax` (default: max |value| over all rows) is shared by every row so selections compare. Returns a pure-ASCII
    fragment: write it to a .html file or show it in a notebook with `IPython.display.HTML`.
    """
    assert all(len(r["tokens"]) == len(r["values"]) for r in rows), "every row needs one value per token"
    data = {
        "dims": list(dims),
        "rows": [
            {**{d: str(r[d]) for d in dims}, "tokens": list(r["tokens"]), "values": [round(float(v), 4) for v in r["values"]]} for r in rows
        ],
        "initial": {d: str(v) for d, v in (initial or {}).items()},
        "pos": POS_RGB,
        "neg": NEG_RGB,
    }
    data["vmax"] = vmax or max((abs(v) for r in data["rows"] for v in r["values"]), default=0.0) or 1.0
    title_html = f"<h3>{html.escape(title)}</h3>".encode("ascii", "xmlcharrefreplace").decode() if title else ""
    return (
        _EXPLORER_TEMPLATE.replace("__UID__", f"tx-{uuid.uuid4().hex[:8]}")
        .replace("__TITLE__", title_html)
        .replace("__VMAX__", f"{data['vmax']:.2f}")
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))  # keep token text like "</s>" from closing the script
    )
