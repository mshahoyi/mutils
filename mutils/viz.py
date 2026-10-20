"""Lightweight visualisation helpers."""

from __future__ import annotations

import html
import json
import uuid
from typing import Any, Mapping, Sequence

__all__ = ["token_heatmap_html", "token_heatmap_explorer", "in_iframe"]

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
#__UID__ .hit .txt {outline: 2px solid #000; outline-offset: 1px}
#__UID__ .val {display: none}
#__UID__ .show-values .tok {display: inline-flex; flex-direction: column; vertical-align: top; margin: 0 5px 6px 0}
#__UID__ .show-values .val {display: block; font-size: 9px; color: #444; text-align: center}
#__UID__ .topk {display: flex; flex-wrap: wrap; gap: 32px; margin-top: 24px}
#__UID__ table {border-collapse: collapse; font-size: 12px}
#__UID__ caption {font-weight: bold; text-align: left; padding-bottom: 4px}
#__UID__ th, #__UID__ td {padding: 2px 8px; border-bottom: 1px solid #eee; text-align: left}
#__UID__ td.ctx, #__UID__ td.mono {font-family: monospace; white-space: pre}
#__UID__ td svg {display: block}
#__UID__ .nav {display: inline-flex; align-items: center; gap: 4px; margin-right: 10px; font: 11px system-ui; color: #666}
#__UID__ .nav button {border: 1px solid #ccc; background: #fff; border-radius: 3px; padding: 0 5px; cursor: pointer; line-height: 16px}
#__UID__ .count {display: inline-block; min-width: 9.5em; text-align: center}
#__UID__ tbody tr {cursor: pointer}
#__UID__ tbody tr:hover {background: #f4f4f4}
</style>
__TITLE__
<div class="controls">
  <label><input type="checkbox"> show values</label>
  <label>top k <input type="number" min="1" value="__TOPK__" style="width: 4em"></label>
</div>
<div class="legend">red &gt; 0 &gt; blue, full colour at |value| &ge; __VMAX__;
  hover a token to see its value. Tables group repeated tokens; step through a token's occurrences with &lsaquo; &rsaquo;
  (or &larr; &rarr; while hovering its row) and click the row to find the one shown</div>
<div class="tokens"></div>
<div class="topk"></div>
<script>
(() => {
  const D = __DATA__, root = document.getElementById("__UID__");
  const [out, tables] = [".tokens", ".topk"].map(q => root.querySelector(q));
  const [box, kInput] = root.querySelectorAll(".controls input");
  const el = (tag, cls = "", text = "") => Object.assign(document.createElement(tag), {className: cls, textContent: text});
  const show = t => t.replaceAll("\n", "\u21b5");  // newlines are drawn as a visible symbol
  const shade = v => {
    const [r, g, b] = v > 0 ? D.pos : D.neg;
    return `rgba(${r},${g},${b},${Math.min(Math.abs(v) / D.vmax, 1).toFixed(2)})`;
  };
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
  kInput.oninput = render;
  let spans = [], stepHovered = null;  // stepHovered: the occurrence stepper of the table row under the mouse
  document.addEventListener("keydown", e => {
    if (!stepHovered || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    e.preventDefault();
    stepHovered(e.key === "ArrowRight" ? 1 : -1);
  });

  function token(t, v) {
    const span = el("span", "tok"), text = el("span", "txt", show(t));
    span.dataset.v = v.toFixed(3);
    text.style.background = shade(v);
    span.append(text, el("span", "val", v.toFixed(2)));  // shown under the token by the "show values" checkbox
    return span;
  }
  function groups(row) {  // each distinct token with its occurrence positions, sorted values and median
    const by = new Map();
    for (const [i, t] of row.tokens.entries()) {
      if (!by.has(t)) by.set(t, []);
      by.get(t).push(i);
    }
    return [...by].map(([tok, idx]) => {
      const values = idx.map(i => row.values[i]).sort((a, b) => a - b), m = values.length >> 1;
      return {tok, idx, values, median: values.length % 2 ? values[m] : (values[m - 1] + values[m]) / 2};
    });
  }
  function strip(g) {  // every occurrence as a tick on the shared -vmax..vmax scale, the median as a black bar
    const W = 120, H = 16, x = v => (((Math.max(-1, Math.min(1, v / D.vmax)) + 1) * W) / 2).toFixed(1);
    const tick = (v, colour, y0, y1, width, opacity = 1) =>
      `<line x1="${x(v)}" x2="${x(v)}" y1="${y0}" y2="${y1}" stroke="${colour}" stroke-width="${width}" stroke-opacity="${opacity}"/>`;
    const [lo, hi] = [g.values[0], g.values[g.values.length - 1]].map(v => v.toFixed(2));
    const title = `${g.values.length} occurrences: min ${lo}, median ${g.median.toFixed(2)}, max ${hi}`;
    return `<svg width="${W}" height="${H}"><title>${title}</title>`
      + `<line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="#ddd"/>` + tick(0, "#bbb", 2, H - 2, 1)
      + g.values.map(v => tick(v, `rgb(${(v > 0 ? D.pos : D.neg).join(",")})`, 4, H - 4, 1.5, 0.6)).join("")
      + tick(g.median, "#000", 0, H, 2) + "</svg>";
  }
  function examples(g, order, row) {  // one occurrence in context at a time, strongest first; returns [cell, step, current position]
    const idx = [...g.idx].sort((a, b) => order * (row.values[a] - row.values[b])), cell = el("td", "ctx"), nav = el("span", "nav");
    const [prev, count, next, text] = [el("button", "", "\u2039"), el("span", "count"), el("button", "", "\u203a"), el("span")];
    let j = 0;
    const step = d => {
      j = (j + d + idx.length) % idx.length;
      const i = idx[j];
      const [before, after] = [row.tokens.slice(Math.max(0, i - 8), i), row.tokens.slice(i + 1, i + 4)].map(ts => show(ts.join("")));
      count.textContent = `${j + 1}/${idx.length}: ${row.values[i].toFixed(2)} at ${i}`;
      text.replaceChildren(before, el("b", "", show(row.tokens[i])), after);
    };
    prev.onclick = e => (e.stopPropagation(), step(-1));
    next.onclick = e => (e.stopPropagation(), step(1));
    nav.append(prev, count, next);
    cell.append(nav, text);
    step(0);
    return [cell, step, () => idx[j]];
  }
  function topTable(title, gs, order, row) {  // one table row per distinct token
    const table = el("table");
    table.createCaption().textContent = title;
    const head = table.createTHead().insertRow(), body = table.createTBody();
    for (const h of ["#", "token", "n", "median", "distribution", "in context"]) head.append(el("th", "", h));
    gs.forEach((g, rank) => {
      const tr = body.insertRow(), median = el("td", "", g.median.toFixed(3)), dist = el("td");
      const [ctx, step, current] = examples(g, order, row);
      median.style.background = shade(g.median);
      dist.innerHTML = strip(g);
      tr.append(el("td", "", rank + 1), el("td", "mono", JSON.stringify(g.tok)), el("td", "", g.idx.length), median, dist, ctx);
      tr.onmouseenter = () => (stepHovered = step);
      tr.onmouseleave = () => (stepHovered = null);
      tr.onclick = () => {
        spans.forEach(s => s.classList.remove("hit"));
        spans[current()].classList.add("hit");
        spans[current()].scrollIntoView({block: "center"});
      };
    });
    return table;
  }
  function render() {
    const row = D.rows.find(r => D.dims.every((d, i) => r[d] === selects[i].value));
    if (!row) {
      tables.replaceChildren();
      return out.replaceChildren("no data for this selection");
    }
    spans = row.tokens.map((t, i) => token(t, row.values[i]));
    out.replaceChildren(...spans.flatMap((s, i) => [s, "\n".repeat((row.tokens[i].match(/\n/g) || []).length)]));
    const k = Math.max(1, parseInt(kInput.value) || D.topK), gs = groups(row).sort((a, b) => b.median - a.median);
    stepHovered = null;
    tables.replaceChildren(
      topTable(`top ${k} positive tokens (by median)`, gs.slice(0, k), -1, row),
      topTable(`top ${k} negative tokens (by median)`, gs.slice(-k).reverse(), 1, row),
    );
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
    top_k: int = 10,
) -> str:
    """Interactive HTML for browsing many token heatmaps, one at a time.

    Each row has a value for every key in `dims` (e.g. layer, system prompt, user prompt) plus `tokens` and `values`.
    The page gets one dropdown per dim (options in first-seen order, preselected from `initial`) and shows the matching
    row shaded like `token_heatmap_html`. Hovering a token shows its value; a checkbox prints values under the tokens.
    Below the text, two tables list the `top_k` distinct tokens with the highest and lowest median value (k is also
    editable on the page): occurrence count, median, a strip plot of all occurrences, and each occurrence in context,
    stepped through with buttons or the arrow keys; clicking a row scrolls to the occurrence shown.
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
        "topK": top_k,
    }
    data["vmax"] = vmax or max((abs(v) for r in data["rows"] for v in r["values"]), default=0.0) or 1.0
    title_html = f"<h3>{html.escape(title)}</h3>".encode("ascii", "xmlcharrefreplace").decode() if title else ""
    return (
        _EXPLORER_TEMPLATE.replace("__UID__", f"tx-{uuid.uuid4().hex[:8]}")
        .replace("__TITLE__", title_html)
        .replace("__VMAX__", f"{data['vmax']:.2f}")
        .replace("__TOPK__", str(top_k))
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))  # keep token text like "</s>" from closing the script
    )


def in_iframe(page: str, height: int = 900) -> str:
    """`page` inside an <iframe srcdoc>, so its scripts run wherever it is shown. Notebook hosts such as VS Code's
    Interactive Window can insert HTML output with innerHTML, which never executes <script> tags; the iframe is its own
    document, so the page stays interactive. Show it with `IPython.display.HTML(in_iframe(page))`."""
    return f'<iframe srcdoc="{html.escape(page)}" style="width: 100%; height: {height}px; border: 0"></iframe>'
