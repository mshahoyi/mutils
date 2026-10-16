"""Lightweight visualisation helpers."""

from __future__ import annotations

import html
from typing import Sequence

__all__ = ["token_heatmap_html"]

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
