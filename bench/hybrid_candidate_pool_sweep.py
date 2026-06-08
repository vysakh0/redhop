#!/usr/bin/env python3
"""Does the hybrid `candidate_pool` default (50) actually fit the workload,
or is it over-/under-allocated?

The 0.3.1 audit's `multihop_helpers_probe` measured "larger candidate_k"
on the LEXICAL pool (20 → 60) and found it flat — bridge passages
weren't in the larger BM25 pool either, so a bigger lexical pool didn't
help. But that's a different parameter from the hybrid `candidate_pool`,
which is the BM25 pool that gets *dense-reranked*. Dense rerank can
rescue lexically-distant passages that BM25 missed; the question is
whether 50 candidates is enough rope.

The intuition both ways:
- Too small: BM25 doesn't surface the bridge passage at all → dense
  rerank never sees it → rescue impossible.
- Too big: dense embeds N more chunks per query → latency grows linearly
  with no gain once the bridge is reliably in the pool.

We don't actually know where the curve plateaus. RedHop has shipped
`candidate_pool=50` as a default since 0.3.0 without measuring it on
the workloads where it matters (HotpotQA, MuSiQue — bridge-passage
rescue is the lever per MULTIHOP_HYBRID).

This probe: hybrid mode, sweep candidate_pool ∈ {10, 25, 50, 100, 200,
500} on HotpotQA + MuSiQue at n=100. The result decides whether the
default needs a flip (like the raw-analyzer flip) or stays.

Run:  bench/.venv/bin/python bench/hybrid_candidate_pool_sweep.py

Note: downloads bge-small ONNX on first run (cached).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]

BUDGET = 400
POOLS = [10, 25, 50, 100, 200, 500]

# Chunks-per-doc empirically: HotpotQA ~11, MuSiQue ~16, CUAD ~96 (range 3-343).
# Only CUAD genuinely constrains pool=50 — the others stay in pool ≤ corpus
# regime. Including all three so the comparison spans the regimes.


# ── Data loaders (same shape as multihop_helpers_probe.py) ─────────────────


def hotpot_items(limit: int):
    data = json.loads((REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())
    for ex in data[:limit]:
        paras = {title: sents for title, sents in ex["context"]}
        doc = "\n\n".join(" ".join(s) for s in paras.values())
        gold_sents = []
        for title, idx in ex["supporting_facts"]:
            if title in paras and idx < len(paras[title]):
                gold_sents.append(paras[title][idx])
        gold = " ".join(gold_sents)
        if gold.strip():
            yield doc, ex["question"], gold


def cuad_items(limit_q: int):
    """CUAD contracts (real docs, 3-343 chunks per doc, mean ~96). This is
    the workload where `candidate_pool` actually constrains retrieval —
    pool=50 is below corpus size for the larger contracts."""
    data = json.loads((REPO / "data/cuad/cuad_sample.json").read_text())["data"]
    n = 0
    for c in data:
        for p in c["paragraphs"]:
            for qa in p["qas"]:
                if n >= limit_q:
                    return
                gold = qa["answers"][0]["text"] if qa["answers"] else ""
                if gold:
                    yield p["context"], qa["question"], gold
                    n += 1


def musique_items(limit: int):
    n = 0
    with (REPO / "data/musique/dev.jsonl").open() as f:
        for line in f:
            if n >= limit:
                return
            ex = json.loads(line)
            doc = "\n\n".join(p["paragraph_text"] for p in ex["paragraphs"])
            gold = " ".join(
                p["paragraph_text"] for p in ex["paragraphs"] if p.get("is_supporting")
            )
            if gold.strip() and ex.get("answerable", True):
                yield doc, ex["question"], gold
                n += 1


def words(s: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 1}


def span_recall(gold: str, ctx: str) -> float:
    g = words(gold)
    if not g:
        return 1.0
    cw = words(ctx)
    return len(g & cw) / len(g)


# ── Sweep ──────────────────────────────────────────────────────────────────


def eval_pool(items_list, candidate_pool: int):
    rec_sum = 0.0
    r50 = 0
    r80 = 0
    latencies = []
    for doc_text, query, gold in items_list:
        t0 = time.perf_counter()
        try:
            doc = redhop.Document.from_text(
                doc_text,
                strategy="raw_topk",
                token_budget=BUDGET,
                retrieval="hybrid",
                model="bge-small",
                candidate_pool=candidate_pool,
            )
            ctx = doc.context(query).text()
        except Exception as e:  # noqa: BLE001
            print(f"  [pool={candidate_pool}] error: {e}", file=sys.stderr)
            ctx = ""
        latencies.append((time.perf_counter() - t0) * 1000)
        r = span_recall(gold, ctx)
        rec_sum += r
        r50 += int(r >= 0.5)
        r80 += int(r >= 0.8)
    n = max(len(items_list), 1)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    return rec_sum / n, r50 * 100 / n, r80 * 100 / n, p50


def report(label: str, items_iter):
    items_list = list(items_iter)
    n = len(items_list)
    print()
    print("=" * 78)
    print(f"  {label}  (n={n}, budget={BUDGET}, retrieval=hybrid, model=bge-small)")
    print("=" * 78)
    print(f"  {'candidate_pool':>14} {'mean recall':>12} {'≥0.5':>6} {'≥0.8':>6} {'p50 ms':>9}")
    print("  " + "-" * 56)
    rows = []
    for pool in POOLS:
        r_mean, r50, r80, p50 = eval_pool(items_list, pool)
        rows.append((pool, r_mean, r50, r80, p50))
        marker = "  ← default" if pool == 50 else ""
        print(f"  {pool:>14} {r_mean:>12.2f} {r50:>5.0f}% {r80:>5.0f}% {p50:>8.1f}{marker}")

    # Δ vs default
    default = next(r for r in rows if r[0] == 50)
    print()
    print(f"  {'Δ vs default (pool=50)':>22}")
    print(f"  {'pool':>6} {'Δ recall':>9} {'Δ ≥0.5':>8} {'Δ ≥0.8':>8} {'Δ p50 ms':>10}")
    for pool, r_mean, r50, r80, p50 in rows:
        if pool == 50:
            continue
        print(
            f"  {pool:>6} {r_mean - default[1]:>+9.2f} "
            f"{r50 - default[2]:>+7.0f}  {r80 - default[3]:>+7.0f}  "
            f"{p50 - default[4]:>+9.1f}"
        )


def main() -> None:
    print()
    print("=" * 78)
    print("  Hybrid candidate_pool sweep")
    print("  Is the shipped default (50) optimal, or over-/under-allocated?")
    print("=" * 78)

    report(
        "CUAD (contracts; chunks-per-doc 3-343, mean 96 — pool actually constrains)",
        cuad_items(100),
    )
    report("HotpotQA (2-hop; ~11 chunks/doc — pool ≥ corpus for default)", hotpot_items(100))
    report("MuSiQue (2-4 hop; ~16 chunks/doc — pool ≥ corpus for default)", musique_items(100))

    print()
    print("Reading the result:")
    print("  • Plateau before 50  → default is over-allocated; flip to plateau point")
    print("  • Still rising at 200 → default is under-allocated; consider raising")
    print("  • Flat everywhere    → candidate_pool isn't the lever; default fine")
    print()


if __name__ == "__main__":
    main()
