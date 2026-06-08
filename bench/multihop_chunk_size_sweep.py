#!/usr/bin/env python3
"""Does RedHop's MuSiQue gap close when we use bigger chunks?

The hybrid-competitors probe showed RedHop hybrid loses to LangChain
hybrid on MuSiQue (26% vs 39% ≥0.8) despite winning on HotpotQA. The
leading theory: RedHop's 128-token default chunks fragment the
multi-paragraph bridge passage that links compositional 2-4 hop
reasoning. LangChain's larger ~256-token chunks keep those bridges
whole.

Direct test: run RedHop hybrid on MuSiQue at chunk_size ∈ {128 (default),
256, 384, 512}. If retention lifts at larger chunks, the gap was
chunking; if it doesn't, something else is going on.

Run:  bench/.venv/bin/python bench/multihop_chunk_size_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]
BUDGET = 400
CANDIDATE_K = 20


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


def words(s: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 1}


def span_recall(gold: str, ctx: str) -> float:
    g = words(gold)
    if not g:
        return 1.0
    cw = words(ctx)
    return len(g & cw) / len(g)


def evaluate_at_chunk_size(items_list, chunk_size: int, retrieval: str, label: str):
    rec_sum = r50 = r80 = 0
    latencies = []
    chunk_counts = []
    for doc_text, query, gold in items_list:
        t0 = time.perf_counter()
        try:
            kwargs = dict(
                chunk_size=chunk_size,
                strategy="raw_topk",
                token_budget=BUDGET,
                candidate_k=CANDIDATE_K,
            )
            if retrieval == "hybrid":
                kwargs["retrieval"] = "hybrid"
                kwargs["model"] = "bge-small"
            doc = redhop.Document.from_text(doc_text, **kwargs)
            chunk_counts.append(doc.n_chunks)
            ctx = doc.context(query)
            text = ctx.text()
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] error: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            text = ""
        latencies.append((time.perf_counter() - t0) * 1000)
        r = span_recall(gold, text)
        rec_sum += r
        r50 += int(r >= 0.5)
        r80 += int(r >= 0.8)
    n = len(items_list)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    avg_chunks = sum(chunk_counts) / max(len(chunk_counts), 1)
    print(
        f"  {label:<46} chunk_size={chunk_size:>4}  "
        f"mean={rec_sum/n:.2f} ≥0.5={100*r50/n:>3.0f}% ≥0.8={100*r80/n:>3.0f}% "
        f"chunks/doc≈{avg_chunks:.1f}  p50={p50:.0f}ms"
    )


def sweep(items_iter, dataset_label: str):
    items_list = list(items_iter)
    n = len(items_list)
    print()
    print("=" * 110)
    print(f"  {dataset_label}  (n={n}, budget={BUDGET}, candidate_k={CANDIDATE_K})")
    print("=" * 110)

    # BM25 only sweep — quick (no model)
    print()
    print("  BM25-only baseline at varying chunk sizes:")
    for cs in [128, 256, 384, 512]:
        evaluate_at_chunk_size(items_list, cs, retrieval="lexical", label="redhop BM25")

    # Hybrid sweep — slower (dense rerank)
    print()
    print("  Hybrid (BM25 + bge-small rerank) at varying chunk sizes:")
    for cs in [128, 256, 384, 512]:
        evaluate_at_chunk_size(items_list, cs, retrieval="hybrid", label="redhop hybrid")


def main() -> None:
    n = 100
    # MuSiQue first — that's where RedHop's hybrid loses, the question
    # is whether bigger chunks close it.
    sweep(musique_items(n), "MuSiQue (compositional multi-hop)")
    # HotpotQA second — sanity check that bigger chunks don't TANK the
    # workload where smaller chunks were the win.
    sweep(hotpot_items(n), "HotpotQA (multi-hop)")

    print()
    print("=" * 110)
    print("  INTERPRETATION GUIDE")
    print("=" * 110)
    print("""
  Reference numbers from MULTIHOP_HYBRID_COMPETITORS.md (n=100, same budget,
  same candidate_k, identical bge-small):

    MuSiQue ≥0.8:
      RedHop hybrid (chunk_size=128)  26%      ← current default
      LangChain    hybrid              39%      ← winner
      LlamaIndex   hybrid              31%

    HotpotQA ≥0.8:
      RedHop hybrid (chunk_size=128)  83%      ← current default, winner
      LangChain    hybrid              77%
      LlamaIndex   hybrid              67%

  What we expect to see in this sweep:

  - If MuSiQue chunk_size=256 lifts RedHop to ~35-40% ≥0.8, the gap
    was chunking — switching default for compositional multi-hop is a
    one-flag fix.
  - If MuSiQue stays ~26% across chunk sizes, the gap is in retrieval
    or rerank ordering, not chunking. Different problem.
  - If HotpotQA at chunk_size=256 stays ~83%, the bigger chunks don't
    hurt the original win — bigger could become the new default
    safely.
  - If HotpotQA at chunk_size=256 drops, there's a real tradeoff to
    expose: 128 wins HotpotQA but loses MuSiQue.
""")


if __name__ == "__main__":
    main()
