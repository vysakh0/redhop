#!/usr/bin/env python3
"""Chunk-size sweep: does finer chunking close RedHop's tight-budget gap?

The framework comparison showed RedHop under-filling at a tight budget because
its default ~256-token chunks are too coarse (~1 fits a 400 budget). This sweeps
RedHop's chunk_size at the same budgets and datasets, against the LangChain /
LlamaIndex baselines, to find the budget→granularity relationship and whether a
finer default suffices.

Run (from the bench venv):  bench/.venv/bin/python bench/chunk_sweep.py
"""

from __future__ import annotations

import redhop

from compare import (
    CANDIDATE_K,
    ctx_langchain,
    ctx_llamaindex,
    cuad_items,
    hotpot_items,
    span_recall,
    toks,
)

CHUNK_SIZES = [64, 128, 192, 256]
STRATEGIES = ["reasoning_preserving", "raw_topk"]


def redhop_ctx(doc: str, query: str, budget: int, chunk_size: int, strategy: str) -> str:
    d = redhop.Document.from_text(
        doc, strategy=strategy, chunk_size=chunk_size, candidate_k=CANDIDATE_K
    )
    return d.context(query, budget=budget).text()


def row(label: str, items, build):
    tok = rec = r80 = 0.0
    n = 0
    for doc, q, gold in items:
        try:
            ctx = build(doc, q)
        except Exception:  # noqa: BLE001
            ctx = ""
        r = span_recall(gold, ctx)
        tok += toks(ctx)
        rec += r
        r80 += int(r >= 0.8)
        n += 1
    n = max(n, 1)
    print(f"  {label:<26} tok {tok / n:>5.0f}   recall {rec / n:.2f}   ≥0.8 {100 * r80 / n:>3.0f}%")


def sweep(items, budget: int, label: str):
    items = list(items)
    print(f"\n==== {label}  (budget {budget}, n={len(items)}) ====")
    for strat in STRATEGIES:
        for cs in CHUNK_SIZES:
            row(
                f"redhop[{strat[:6]},cs={cs}]",
                items,
                lambda d, q, cs=cs, strat=strat: redhop_ctx(d, q, budget, cs, strat),
            )
    row("langchain (baseline)", items, lambda d, q: ctx_langchain(d, q, budget))
    row("llamaindex (baseline)", items, lambda d, q: ctx_llamaindex(d, q, budget))


def main() -> None:
    sweep(hotpot_items(200), budget=400, label="HotpotQA multi-hop (tight budget — the gap)")
    sweep(cuad_items(200), budget=2000, label="CUAD contracts (does finer hurt here?)")


if __name__ == "__main__":
    main()
