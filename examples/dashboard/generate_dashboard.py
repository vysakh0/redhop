#!/usr/bin/env python3
"""Deliverable B — context-economics dashboard.

Most RAG tooling has near-zero context observability: you stuff the top-k
into the prompt and hope. This generates a self-contained HTML dashboard
(no JS deps) that makes the invisible visible — original vs final tokens,
chunks removed, reasoning rescues, distractor estimates, evidence density,
a context-efficiency score, and a strategy comparison.

    python examples/dashboard/generate_dashboard.py
    open examples/dashboard/dashboard.html

Runs fully offline on the shared sample corpus.
"""

import sys
from pathlib import Path

_PY = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(_PY))
import neorag  # noqa: E402
from sample_corpus import (  # noqa: E402
    QUERY,
    RETRIEVED,
    GOLD_ANSWER,
    DISTRACTOR_MIN_GROUNDING,
    LINK_MIN_JACCARD,
)

OUT = Path(__file__).resolve().parent / "dashboard.html"
STRATEGIES = ["raw_topk", "distractor_filtered", "max_density", "reasoning_preserving"]


def efficiency_score(report) -> float:
    """A simple 0-100 context-efficiency score: reward evidence density and
    second-hop retention, penalize distractor tokens. Heuristic, for display."""
    d = report.data
    density = d["economics"]["evidence_density"]
    distr = d["economics"]["distractor_ratio"]
    return round(100 * max(0.0, density * (1.0 - 0.5 * distr)), 1)


def bar(value: float, max_value: float, color: str, width: int = 220) -> str:
    pct = 0 if max_value <= 0 else min(1.0, value / max_value)
    px = int(pct * width)
    return (
        f'<div style="background:#eee;width:{width}px;height:14px;border-radius:7px;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{px}px;height:14px;border-radius:7px"></div></div>'
    )


def main() -> None:
    runs = {}
    for strat in STRATEGIES:
        runs[strat] = neorag.build_context(
            query=QUERY,
            retrieved_chunks=RETRIEVED,
            token_budget=12000,
            strategy=strat,
            distractor_min_grounding=DISTRACTOR_MIN_GROUNDING,
            link_min_jaccard=LINK_MIN_JACCARD,
        )

    rp = runs["reasoning_preserving"]
    raw = runs["raw_topk"]
    input_tokens = raw.report.total_tokens  # raw keeps everything = the input
    max_tokens = max(r.report.total_tokens for r in runs.values()) or 1

    cards = []
    for strat in STRATEGIES:
        r = runs[strat].report
        kept_hop = "British" in runs[strat].text
        cards.append(f"""
        <tr>
          <td><code>{strat}</code></td>
          <td>{r.n_input_chunks} → {r.n_selected}</td>
          <td>{r.total_tokens} {bar(r.total_tokens, max_tokens, '#4c78a8')}</td>
          <td>{r.distractors_pruned}</td>
          <td>{r.second_hop_rescue_count}</td>
          <td>{r.data['economics']['distractor_ratio']:.2f}</td>
          <td>{r.evidence_density:.2f} {bar(r.evidence_density, 0.5, '#54a24b')}</td>
          <td>{efficiency_score(r)}</td>
          <td>{'✓' if kept_hop else '✗'}</td>
        </tr>""")

    saved = input_tokens - rp.report.total_tokens
    saved_pct = 0 if input_tokens == 0 else round(100 * saved / input_tokens)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NeoRAG — Context Economics</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 40px; color:#222; max-width: 1000px }}
 h1 {{ font-size: 22px; margin-bottom: 2px }}
 .sub {{ color:#666; margin-top:0 }}
 .kpis {{ display:flex; gap:16px; margin:24px 0 }}
 .kpi {{ background:#f6f8fa; border:1px solid #e1e4e8; border-radius:10px; padding:14px 18px; flex:1 }}
 .kpi .v {{ font-size:26px; font-weight:700 }}
 .kpi .l {{ color:#666; font-size:12px; text-transform:uppercase; letter-spacing:.04em }}
 table {{ border-collapse: collapse; width:100%; margin-top:12px; font-size:14px }}
 th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #eee }}
 th {{ color:#666; font-weight:600; font-size:12px; text-transform:uppercase }}
 tr:has(td code:only-child) {{}}
 .note {{ color:#666; font-size:13px; margin-top:18px }}
 code {{ background:#f0f2f4; padding:1px 5px; border-radius:4px }}
 .tag {{ display:inline-block; background:#eef3ff; color:#2b50aa; border-radius:6px; padding:2px 8px; font-size:12px }}
</style></head><body>
<h1>NeoRAG — Context Optimization Report</h1>
<p class="sub">a reasoning-preserving context optimization layer &middot; sits between retrieval and generation</p>
<p><span class="tag">query</span> &nbsp;{QUERY}</p>

<div class="kpis">
  <div class="kpi"><div class="v">{saved_pct}%</div><div class="l">tokens saved (default strategy)</div></div>
  <div class="kpi"><div class="v">{rp.report.distractors_pruned}</div><div class="l">distractors pruned</div></div>
  <div class="kpi"><div class="v">{rp.report.second_hop_rescue_count}</div><div class="l">reasoning rescues</div></div>
  <div class="kpi"><div class="v">{raw.report.evidence_density:.2f} → {rp.report.evidence_density:.2f}</div><div class="l">evidence density</div></div>
</div>

<table>
 <tr><th>strategy</th><th>chunks</th><th>tokens</th><th>distractors pruned</th><th>rescues</th><th>distr ratio</th><th>evidence density</th><th>efficiency</th><th>2nd hop kept</th></tr>
 {''.join(cards)}
</table>
<p class="note">
 reasoning_preserving's distractor ratio counts the deliberately-rescued
 second hop (low-relevance-to-query by nature, but reasoning-critical).
 "2nd hop kept" = does the context still contain the "{GOLD_ANSWER}" nationality fact
 the multi-hop answer depends on. Efficiency is a display heuristic
 (density &times; low-distractor). Generated offline from the sample corpus.
</p>
</body></html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}")
    print(f"  default strategy saved {saved_pct}% of tokens, pruned "
          f"{rp.report.distractors_pruned} distractors, rescued "
          f"{rp.report.second_hop_rescue_count} reasoning chunk(s)")
    print(f"  open {OUT}")


if __name__ == "__main__":
    main()
