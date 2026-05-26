#!/usr/bin/env python3
"""dashboard — a self-contained HTML context-economics report.

Most RAG tooling has near-zero context observability. This renders a
shareable HTML dashboard (no JS deps) from real `redhop` runs: original vs
final tokens, distractors pruned, reasoning rescues, evidence density, a
context-efficiency score, and a strategy comparison.

    python examples/dashboard.py
    open examples/dashboard.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sample import (  # noqa: E402
    DISTRACTOR_MIN_GROUNDING,
    GOLD_ANSWER,
    LINK_MIN_JACCARD,
    QUERY,
    RETRIEVED,
)

import redhop  # noqa: E402

OUT = Path(__file__).resolve().parent / "dashboard.html"
STRATEGIES = ["raw_topk", "distractor_filtered", "max_density", "reasoning_preserving"]


def efficiency_score(r) -> float:
    """0-100 display heuristic: reward density, penalize distractor tokens."""
    return round(100 * max(0.0, r.evidence_density * (1.0 - 0.5 * r.distractor_ratio)), 1)


def bar(value: float, max_value: float, color: str, width: int = 220) -> str:
    px = int((0 if max_value <= 0 else min(1.0, value / max_value)) * width)
    return (
        f'<div style="background:#eee;width:{width}px;height:14px;border-radius:7px;'
        f'display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{px}px;height:14px;border-radius:7px"></div></div>'
    )


def main() -> None:
    kw = dict(distractor_min_grounding=DISTRACTOR_MIN_GROUNDING, link_min_jaccard=LINK_MIN_JACCARD)
    runs = {
        s: redhop.build_context(QUERY, RETRIEVED, strategy=s, token_budget=12000, **kw)
        for s in STRATEGIES
    }

    rp, raw = runs["reasoning_preserving"], runs["raw_topk"]
    input_tokens = raw.report.total_tokens
    max_tokens = max(c.report.total_tokens for c in runs.values()) or 1

    cards = ""
    for s in STRATEGIES:
        r = runs[s].report
        kept = "✓" if GOLD_ANSWER in runs[s].text() else "✗"
        cards += (
            f"<tr><td><code>{s}</code></td>"
            f"<td>{r.n_input_chunks} → {r.n_selected}</td>"
            f"<td>{r.total_tokens} {bar(r.total_tokens, max_tokens, '#4c78a8')}</td>"
            f"<td>{r.distractors_pruned}</td><td>{r.second_hop_rescue_count}</td>"
            f"<td>{r.distractor_ratio:.2f}</td>"
            f"<td>{r.evidence_density:.2f} {bar(r.evidence_density, 0.5, '#54a24b')}</td>"
            f"<td>{efficiency_score(r)}</td><td>{kept}</td></tr>"
        )

    saved_pct = (
        0
        if input_tokens == 0
        else round(100 * (input_tokens - rp.report.total_tokens) / input_tokens)
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RedHop — Context Economics</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 40px; color:#222; max-width: 1000px }}
 h1 {{ font-size: 22px; margin-bottom: 2px }} .sub {{ color:#666; margin-top:0 }}
 .kpis {{ display:flex; gap:16px; margin:24px 0 }}
 .kpi {{ background:#f6f8fa; border:1px solid #e1e4e8; border-radius:10px; padding:14px 18px; flex:1 }}
 .kpi .v {{ font-size:26px; font-weight:700 }}
 .kpi .l {{ color:#666; font-size:12px; text-transform:uppercase; letter-spacing:.04em }}
 table {{ border-collapse: collapse; width:100%; margin-top:12px; font-size:14px }}
 th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #eee }}
 th {{ color:#666; font-weight:600; font-size:12px; text-transform:uppercase }}
 code {{ background:#f0f2f4; padding:1px 5px; border-radius:4px }}
 .tag {{ display:inline-block; background:#eef3ff; color:#2b50aa; border-radius:6px; padding:2px 8px; font-size:12px }}
 .note {{ color:#666; font-size:13px; margin-top:18px }}
</style></head><body>
<h1>RedHop — Context Optimization Report</h1>
<p class="sub">reasoning-preserving context optimization &middot; sits between retrieval and generation</p>
<p><span class="tag">query</span> &nbsp;{QUERY}</p>
<div class="kpis">
  <div class="kpi"><div class="v">{saved_pct}%</div><div class="l">tokens saved (default)</div></div>
  <div class="kpi"><div class="v">{rp.report.distractors_pruned}</div><div class="l">distractors pruned</div></div>
  <div class="kpi"><div class="v">{rp.report.second_hop_rescue_count}</div><div class="l">reasoning rescues</div></div>
  <div class="kpi"><div class="v">{raw.report.evidence_density:.2f} → {rp.report.evidence_density:.2f}</div><div class="l">evidence density</div></div>
</div>
<table>
 <tr><th>strategy</th><th>chunks</th><th>tokens</th><th>distractors pruned</th><th>rescues</th><th>distr ratio</th><th>evidence density</th><th>efficiency</th><th>2nd hop kept</th></tr>
 {cards}
</table>
<p class="note">distr ratio is the TRUE distractor ratio — deliberately-rescued second hops are
 reasoning evidence and are excluded (so reasoning_preserving shows low distr with rescues &ge; 1).
 "2nd hop kept" = does the context still contain the "{GOLD_ANSWER}" fact the multi-hop answer needs.
 Efficiency is a display heuristic. Generated offline by RedHop from the sample corpus.</p>
</body></html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}")
    print(
        f"  default strategy saved {saved_pct}% of tokens, pruned "
        f"{rp.report.distractors_pruned} distractors, rescued "
        f"{rp.report.second_hop_rescue_count} reasoning chunk(s)"
    )


if __name__ == "__main__":
    main()
