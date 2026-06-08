#!/usr/bin/env python3
"""Is the +12 multi-hop lift from `retrieval="hybrid"` a property of dense
rerank, or of RedHop specifically?

The previous probe (MULTIHOP_HYBRID.md) showed RedHop's `retrieval="hybrid"`
lifts HotpotQA ≥0.8 retention 71% → 83% (+12) and MuSiQue ≥0.5 66% → 74%
(+8). The natural follow-up the audit reviewer flagged: **does the same
dense rerank lift LangChain and LlamaIndex by the same amount?** If yes,
the +12 is a property of dense rerank, not RedHop. If no, RedHop has a
structural advantage worth understanding.

This probe answers it directly. Six arms total:

  RedHop      BM25 baseline         (already measured; reproduced here)
  RedHop      retrieval="hybrid"    (ships in 0.3.0; reproduced here)
  LangChain   BM25 baseline         (already measured)
  LangChain   BM25 + bge-small rerank   (NEW: same dense model, same shape)
  LlamaIndex  BM25 baseline         (already measured)
  LlamaIndex  BM25 + bge-small rerank   (NEW)

The "same dense model, same shape" arms are constructed deliberately to
hold the dense step constant: each system's BM25 produces its top-K
candidates, we then rerank those K with bge-small cosine and fill the
budget. Differences across systems come from chunking + BM25 ranking,
NOT from a different dense embedder or rerank strategy.

This is the apples-to-apples comparison. We're testing: "given identical
dense rerank applied on top, does any system's BM25 stage produce a
different upper bound?"

Run:  bench/.venv/bin/python bench/multihop_hybrid_competitors_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import redhop
from langchain_community.retrievers import BM25Retriever as LCBm25
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document as LIDoc
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever as LIBm25
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[1]
CHUNK_TOKENS = 256
CANDIDATE_K = 20
BUDGET = 400


# ── Data loaders ───────────────────────────────────────────────────────────


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


# ── Metric ─────────────────────────────────────────────────────────────────


def toks(s: str) -> int:
    return len(s.split())


def words(s: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 1}


def span_recall(gold: str, ctx: str) -> float:
    g = words(gold)
    if not g:
        return 1.0
    cw = words(ctx)
    return len(g & cw) / len(g)


def fill(ordered: list[str], budget: int) -> str:
    out, used = [], 0
    for t in ordered:
        n = toks(t)
        if used + n > budget and out:
            break
        out.append(t)
        used += n
    return "\n\n".join(out)


# ── The shared dense model (bge-small via sentence-transformers) ──────────
# Loaded once at startup, used to rerank BM25 candidates from both
# LangChain and LlamaIndex. RedHop's `retrieval="hybrid"` uses bge-small
# via its own ONNX path; the model is the same so the rerank step is
# directly comparable.

print("Loading bge-small (BAAI/bge-small-en-v1.5) for the dense rerank step...")
DENSE = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Model loaded.\n")


def dense_rerank(query: str, candidates: list[str], k: int = CANDIDATE_K) -> list[str]:
    """Cosine-sim rerank of `candidates` against `query`, returning a new
    ordering. We take all candidates (up to k) and re-sort by cosine
    similarity — same shape as RedHop's hybrid tier."""
    if not candidates:
        return []
    cands = candidates[:k]
    q_emb = DENSE.encode([query], normalize_embeddings=True)
    c_emb = DENSE.encode(cands, normalize_embeddings=True)
    sims = (q_emb @ c_emb.T)[0]
    order = np.argsort(-sims)
    return [cands[i] for i in order]


# ── System arms ────────────────────────────────────────────────────────────


def ctx_redhop_bm25(doc_text: str, query: str) -> str:
    doc = redhop.Document.from_text(
        doc_text, strategy="raw_topk", token_budget=BUDGET, candidate_k=CANDIDATE_K
    )
    return doc.context(query).text()


def ctx_redhop_hybrid(doc_text: str, query: str) -> str:
    doc = redhop.Document.from_text(
        doc_text,
        strategy="raw_topk",
        token_budget=BUDGET,
        candidate_k=CANDIDATE_K,
        retrieval="hybrid",
        model="bge-small",
    )
    return doc.context(query).text()


def ctx_langchain_bm25(doc_text: str, query: str) -> str:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS * 4, chunk_overlap=40
    ).split_text(doc_text)
    if not chunks:
        return ""
    retr = LCBm25.from_texts(chunks)
    retr.k = CANDIDATE_K
    hits = retr.invoke(query)
    return fill([d.page_content for d in hits], BUDGET)


def ctx_langchain_hybrid(doc_text: str, query: str) -> str:
    """LangChain BM25 candidates → bge-small dense rerank → budget-fill.
    Same dense shape as RedHop's hybrid tier; only the BM25 stage
    (chunker + ranking) differs."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS * 4, chunk_overlap=40
    ).split_text(doc_text)
    if not chunks:
        return ""
    retr = LCBm25.from_texts(chunks)
    retr.k = CANDIDATE_K
    hits = retr.invoke(query)
    reranked = dense_rerank(query, [d.page_content for d in hits])
    return fill(reranked, BUDGET)


def ctx_llamaindex_bm25(doc_text: str, query: str) -> str:
    nodes = SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=20).get_nodes_from_documents(
        [LIDoc(text=doc_text)]
    )
    if not nodes:
        return ""
    retr = LIBm25.from_defaults(nodes=nodes, similarity_top_k=min(CANDIDATE_K, len(nodes)))
    hits = retr.retrieve(query)
    return fill([h.node.get_content() for h in hits], BUDGET)


def ctx_llamaindex_hybrid(doc_text: str, query: str) -> str:
    """LlamaIndex BM25 candidates → bge-small dense rerank → budget-fill."""
    nodes = SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=20).get_nodes_from_documents(
        [LIDoc(text=doc_text)]
    )
    if not nodes:
        return ""
    retr = LIBm25.from_defaults(nodes=nodes, similarity_top_k=min(CANDIDATE_K, len(nodes)))
    hits = retr.retrieve(query)
    reranked = dense_rerank(query, [h.node.get_content() for h in hits])
    return fill(reranked, BUDGET)


SYSTEMS = [
    ("redhop[topk]   BM25 baseline", ctx_redhop_bm25),
    ("redhop[topk]   hybrid (bge-small)", ctx_redhop_hybrid),
    ("langchain      BM25 baseline", ctx_langchain_bm25),
    ("langchain      + bge-small rerank", ctx_langchain_hybrid),
    ("llamaindex     BM25 baseline", ctx_llamaindex_bm25),
    ("llamaindex     + bge-small rerank", ctx_llamaindex_hybrid),
]


# ── Runner ─────────────────────────────────────────────────────────────────


def evaluate(items, label: str):
    items_list = list(items)
    print()
    print("=" * 96)
    print(f"  {label}")
    print(f"  n={len(items_list)} queries, budget={BUDGET} tok, candidate_k={CANDIDATE_K}")
    print("=" * 96)
    print(f"  {'arm':<42} {'mean recall':>12} {'≥0.5':>6} {'≥0.8':>6} {'p50 ms':>8}")
    print("  " + "-" * 80)

    for name, fn in SYSTEMS:
        latencies = []
        rec_sum = r50 = r80 = n = 0
        for doc_text, query, gold in items_list:
            t0 = time.perf_counter()
            try:
                ctx = fn(doc_text, query)
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] error: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                ctx = ""
            latencies.append((time.perf_counter() - t0) * 1000)
            r = span_recall(gold, ctx)
            rec_sum += r
            r50 += int(r >= 0.5)
            r80 += int(r >= 0.8)
            n += 1
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0
        n = max(n, 1)
        print(
            f"  {name:<42} {rec_sum / n:>12.2f} "
            f"{100 * r50 / n:>5.0f}% {100 * r80 / n:>5.0f}% "
            f"{p50:>7.1f}"
        )


def main() -> None:
    n = 100  # Match the previous multi-hop helpers probe; dense rerank
             # makes the hybrid arms slower than baseline.

    evaluate(hotpot_items(n), label="HotpotQA (multi-hop) — RedHop vs LangChain vs LlamaIndex, BM25 vs +dense-rerank")
    evaluate(musique_items(n), label="MuSiQue (compositional multi-hop) — RedHop vs LangChain vs LlamaIndex, BM25 vs +dense-rerank")

    print()
    print("=" * 96)
    print("  INTERPRETATION GUIDE")
    print("=" * 96)
    print("""
  The question this probe answers: "Is the +12 lift on HotpotQA from
  RedHop's `retrieval='hybrid'` a property of dense rerank generally, or
  specific to RedHop?"

  - If LangChain and LlamaIndex's BM25+rerank arms also lift to ~83% ≥0.8
    on HotpotQA, the +12 is a property of dense rerank applied to the
    multi-hop bridge-passage problem — not a RedHop architectural
    advantage. The honest framing then: "dense rerank is the multi-hop
    lever; RedHop ships it as one of its retrieval tiers."
  - If LangChain/LlamaIndex's BM25+rerank arms lift to a noticeably lower
    point (say ~76% ≥0.8), RedHop's BM25 candidate selection is producing
    a better pool for the dense reranker — that's a chunking + BM25 win
    that compounds with dense.
  - If LangChain/LlamaIndex's BM25+rerank arms OVERSHOOT RedHop's hybrid
    (e.g., to 85%+), then RedHop's hybrid implementation has room to
    improve.

  Same bge-small model on the dense step across all three. Differences
  come from chunking + BM25 ranking only.
""")


if __name__ == "__main__":
    main()
