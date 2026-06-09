#!/usr/bin/env python3
"""Tie-breaker: who's closer to a third-party LLM judge on faithfulness?

We have RedHop's decomposed faithfulness, Ragas's faithfulness, and
the on-disk per-case scores from `bench/eval_correlation_hotpot.py`.
This script asks **Claude Haiku** to score the same (context, answer)
pairs independently, then computes:

  • MAE(RedHop_decomposed, Claude)  vs  MAE(Ragas, Claude)
  • how often each library agrees with Claude
  • how often each library disagrees and Claude is between them

The library with lower MAE-to-Claude is the one whose absolute
calibration is closer to a third, independent judge. Whoever Claude
agrees with more often on the contested cases is the one a third LLM
considers more accurate.

Claude is invoked via the `claude -p --model haiku` CLI (no API key
needed — uses the user's local Claude Code auth). Each call returns
a single number; we parse robustly.

Run:
  bench/.venv/bin/python bench/eval_third_judge.py \\
      --in reports/eval_correlation_hotpot_n25.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


SYSTEM = (
    "You are a strict, careful judge of faithfulness. Given a CONTEXT and "
    "an ANSWER, decide whether every claim in the ANSWER is supported by "
    "the CONTEXT. Reply with a single number in [0, 1] — 1 if every claim "
    "is supported, 0 if any claim is fabricated or contradicted, partial "
    "values for partial support. Reply with the number only, no commentary."
)


def claude_faithfulness(context: str, answer: str, model: str = "haiku") -> float:
    """Call `claude -p --model haiku` with a faithfulness prompt. Returns
    a score in [0, 1]; falls back to NaN if the response can't be
    parsed."""
    user_prompt = (
        f"SYSTEM: {SYSTEM}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Reply with the single faithfulness number only."
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", model],
            input=user_prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return float("nan")
    if proc.returncode != 0:
        print(f"  claude error: {proc.stderr[:200]}", file=sys.stderr)
        return float("nan")
    text = proc.stdout.strip()
    # The model occasionally adds a leading word; try to parse the first
    # number on its first line.
    first_line = text.split("\n", 1)[0].strip()
    try:
        return float(first_line)
    except ValueError:
        # Fall back: scan for the first float-shaped substring.
        import re
        m = re.search(r"[-+]?\d*\.?\d+", text)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass
        print(f"  unparseable claude output: {text[:120]!r}", file=sys.stderr)
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True, help="hotpot correlation .json")
    p.add_argument("--model", default="haiku", help="claude model (haiku|sonnet|opus)")
    args = p.parse_args()

    data = json.loads(Path(args.input_path).read_text())
    cases = data["cases"]
    print(f"Loaded {len(cases)} cases from {args.input_path}")
    print(f"Asking Claude {args.model} for an independent faithfulness score on each…")

    # Load the original HotpotQA so we can rebuild (context, answer) per qid.
    hotpot = json.loads((REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())
    by_qid = {ex["_id"]: ex for ex in hotpot}

    claude_scores = []
    t0 = time.perf_counter()
    for i, c in enumerate(cases):
        qid = c["qid"]
        ex = by_qid.get(qid)
        if ex is None:
            print(f"  [{i+1}/{len(cases)}] {qid}: MISSING source HotpotQA case → NaN")
            claude_scores.append(float("nan"))
            continue
        # Rebuild the context the same way the correlation bench did
        # (mode 'all' = supporting + distractor).
        paras = {title: sents for title, sents in ex["context"]}
        ctx_text = "\n\n".join(" ".join(paras[t]) for t in paras)
        answer = c["answer"]
        s = claude_faithfulness(ctx_text, answer, model=args.model)
        print(f"  [{i+1}/{len(cases)}] {qid}: Claude→{s:.3f}")
        claude_scores.append(s)

    print(f"\nTotal Claude time: {time.perf_counter() - t0:.1f}s")

    # ── Reporting ───────────────────────────────────────────────────────────
    print()
    print("=" * 88)
    print(f"  Faithfulness — three-way: RedHop / Ragas / Claude {args.model}")
    print("=" * 88)
    print(
        f"  {'qid':<24} {'RedHop S':>10} {'RedHop D':>10} {'Ragas':>8} {'Claude':>8}"
    )
    print("  " + "-" * 64)
    for c, claude_s in zip(cases, claude_scores):
        rs = c.get("redhop_single")
        rd = c.get("redhop_decomposed")
        rg = c.get("ragas_faithfulness")
        rs_s = f"{rs:10.3f}" if rs is not None else "      null"
        rd_s = f"{rd:10.3f}" if rd is not None else "      null"
        rg_s = f"{rg:8.3f}" if rg is not None else "    null"
        cl_s = f"{claude_s:8.3f}" if claude_s == claude_s else "    null"
        print(f"  {c['qid']:<24} {rs_s} {rd_s} {rg_s} {cl_s}")

    def absent(x):
        return x is None or (isinstance(x, float) and x != x)  # None or NaN

    # Per-library MAE to Claude.
    def mae_to_claude(getter):
        diffs = []
        for c, cs in zip(cases, claude_scores):
            v = getter(c)
            if absent(v) or absent(cs):
                continue
            diffs.append(abs(v - cs))
        if not diffs:
            return float("nan"), 0
        return sum(diffs) / len(diffs), len(diffs)

    rh_single_mae, n_s = mae_to_claude(lambda c: c.get("redhop_single"))
    rh_decomp_mae, n_d = mae_to_claude(lambda c: c.get("redhop_decomposed"))
    rg_mae, n_g = mae_to_claude(lambda c: c.get("ragas_faithfulness"))

    print()
    print("  Mean absolute error vs Claude (third-judge tie-breaker):")
    print(f"  RedHop single-prompt → Claude:  MAE={rh_single_mae:.3f}  (n={n_s})")
    print(f"  RedHop decomposed    → Claude:  MAE={rh_decomp_mae:.3f}  (n={n_d})")
    print(f"  Ragas                → Claude:  MAE={rg_mae:.3f}  (n={n_g})")
    print()
    if rh_decomp_mae < rg_mae:
        winner = "RedHop decomposed"
        delta = rg_mae - rh_decomp_mae
    elif rg_mae < rh_decomp_mae:
        winner = "Ragas"
        delta = rh_decomp_mae - rg_mae
    else:
        winner = "TIE"
        delta = 0.0
    print(f"  Closer to Claude: {winner} (by {delta:+.3f} MAE)")

    # On the contested cases (delta ≥ 0.5 between RedHop_decomp and Ragas),
    # which side does Claude pick?
    contested = []
    for c, cs in zip(cases, claude_scores):
        rd = c.get("redhop_decomposed")
        rg = c.get("ragas_faithfulness")
        if absent(rd) or absent(rg) or absent(cs):
            continue
        if abs(rd - rg) >= 0.5:
            contested.append((c["qid"], rd, rg, cs))
    print()
    print(f"  Contested cases (|RedHop_d - Ragas| ≥ 0.5): n={len(contested)}")
    if contested:
        rh_wins = sum(1 for _, rd, rg, cs in contested if abs(rd - cs) < abs(rg - cs))
        rg_wins = sum(1 for _, rd, rg, cs in contested if abs(rg - cs) < abs(rd - cs))
        ties = len(contested) - rh_wins - rg_wins
        for qid, rd, rg, cs in contested:
            who = "RH" if abs(rd - cs) < abs(rg - cs) else ("Rg" if abs(rg - cs) < abs(rd - cs) else "==")
            print(f"    {qid}  RedHop_d={rd:.2f}  Ragas={rg:.2f}  Claude={cs:.2f}  → {who} closer")
        print(f"  Claude is closer to RedHop_decomposed:  {rh_wins}/{len(contested)}")
        print(f"  Claude is closer to Ragas:              {rg_wins}/{len(contested)}")
        if ties:
            print(f"  Ties:                                   {ties}")

    # Persist.
    out_dir = REPO / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval_third_judge_{Path(args.input_path).stem}.json"
    out.write_text(
        json.dumps(
            {
                "input": args.input_path,
                "claude_model": args.model,
                "claude_scores": claude_scores,
                "mae_to_claude": {
                    "redhop_single": rh_single_mae,
                    "redhop_decomposed": rh_decomp_mae,
                    "ragas": rg_mae,
                },
            },
            indent=2,
            default=lambda o: None if isinstance(o, float) and o != o else o,
        )
    )
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
