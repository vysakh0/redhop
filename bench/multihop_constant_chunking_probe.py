#!/usr/bin/env python3
"""Hold chunking constant across all three systems — which knob wins?

The hybrid-competitors probe varied chunker + BM25 + reranker
simultaneously. To isolate which knob is the differentiator, we now
chunk a document ONCE with each system's chunker and feed those
identical chunks to all three retrievers + the same dense reranker.

3 chunker sources × 3 BM25 retrievers × 2 datasets = 18 arms.
Each arm rerank with the identical bge-small model.

Run:  bench/.venv/bin/python bench/multihop_constant_chunking_probe.py
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
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever as LIBm25
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[1]
BUDGET = 400
CANDIDATE_K = 20

print("Loading bge-small for shared rerank step...")
DENSE = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Model loaded.\n")


# ── Data ────────────────────────────────────────────────────────────────────


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


def words(s: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 1}


def span_recall(gold: str, ctx: str) -> float:
    g = words(gold)
    if not g:
        return 1.0
    cw = words(ctx)
    return len(g & cw) / len(g)


def toks(s: str) -> int:
    return len(s.split())


def fill(ordered: list[str], budget: int = BUDGET) -> str:
    out, used = [], 0
    for t in ordered:
        n = toks(t)
        if used + n > budget and out:
            break
        out.append(t)
        used += n
    return "\n\n".join(out)


def dense_rerank(query: str, candidates: list[str], k: int = CANDIDATE_K) -> list[str]:
    if not candidates:
        return []
    cands = candidates[:k]
    q_emb = DENSE.encode([query], normalize_embeddings=True)
    c_emb = DENSE.encode(cands, normalize_embeddings=True)
    sims = (q_emb @ c_emb.T)[0]
    order = np.argsort(-sims)
    return [cands[i] for i in order]


# ── Chunkers (produce a list[str] from doc_text) ──────────────────────────


def chunk_with_redhop(doc_text: str) -> list[str]:
    """Use redhop.Document.from_text to chunk, then re-emit via context().
    No public chunks() iterator on Document, so we build a coverage
    query from the doc's top high-frequency content words and use a
    huge budget so the assembly returns every chunk.

    This is awkward but consistent: every retrieved chunk text matches
    what redhop's SentenceChunker produced, byte-for-byte."""
    doc = redhop.Document.from_text(doc_text)
    # Coverage query: every distinct alphanumeric token of length ≥ 3 in
    # the doc, capped to 500 to keep the query string bounded. BM25 then
    # matches roughly every chunk that has any reasonable content word.
    tokens = sorted({
        t.lower() for t in "".join(c if c.isalnum() else " " for c in doc_text).split()
        if len(t) >= 3
    })
    if not tokens:
        return [doc_text]
    coverage_q = " ".join(tokens[:500])
    ctx = doc.context(coverage_q, budget=10_000_000)
    chunks = list(ctx.chunks)
    # If the coverage query still missed some (rare), fall back to the
    # raw doc as a single chunk so downstream retrievers have something
    # to chew on rather than empty input.
    return chunks if chunks else [doc_text]


def chunk_with_langchain(doc_text: str) -> list[str]:
    """LangChain's RecursiveCharacterTextSplitter at ~256-token chunks (default
    LC config used in bench/compare.py)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=256 * 4, chunk_overlap=40
    ).split_text(doc_text)


def chunk_with_llamaindex(doc_text: str) -> list[str]:
    """LlamaIndex SentenceSplitter at 256-token chunks (default LI config)."""
    nodes = SentenceSplitter(chunk_size=256, chunk_overlap=20).get_nodes_from_documents(
        [LIDoc(text=doc_text)]
    )
    return [n.get_content() for n in nodes]


# ── Retrievers (BM25 over an arbitrary list[str]) ─────────────────────────


def retrieve_redhop(chunks: list[str], query: str) -> list[str]:
    """Run RedHop's BM25 over `chunks` and return top-K texts."""
    if not chunks:
        return []
    # Convert to redhop.Chunk so we can use Document.from_chunks.
    rh_chunks = [redhop.Chunk(c) for c in chunks]
    doc = redhop.Document.from_chunks(
        rh_chunks, strategy="raw_topk", token_budget=10_000_000, candidate_k=CANDIDATE_K
    )
    ctx = doc.context(query)
    return list(ctx.chunks)


def retrieve_langchain(chunks: list[str], query: str) -> list[str]:
    """Run LangChain BM25Retriever over `chunks`."""
    if not chunks:
        return []
    retr = LCBm25.from_texts(chunks)
    retr.k = min(CANDIDATE_K, len(chunks))
    hits = retr.invoke(query)
    return [d.page_content for d in hits]


def retrieve_llamaindex(chunks: list[str], query: str) -> list[str]:
    """Run LlamaIndex BM25Retriever over `chunks`."""
    if not chunks:
        return []
    nodes = [TextNode(text=c) for c in chunks]
    retr = LIBm25.from_defaults(nodes=nodes, similarity_top_k=min(CANDIDATE_K, len(nodes)))
    hits = retr.retrieve(query)
    return [h.node.get_content() for h in hits]


# ── Matrix ─────────────────────────────────────────────────────────────────

CHUNKERS = [
    ("redhop[128t]", chunk_with_redhop),
    ("langchain[1024c]", chunk_with_langchain),
    ("llamaindex[256t]", chunk_with_llamaindex),
]

RETRIEVERS = [
    ("redhop", retrieve_redhop),
    ("langchain", retrieve_langchain),
    ("llamaindex", retrieve_llamaindex),
]


def evaluate_matrix(items, label: str):
    items_list = list(items)
    n = len(items_list)
    print()
    print("=" * 110)
    print(f"  {label}  (n={n}, budget={BUDGET}, candidate_k={CANDIDATE_K})")
    print("=" * 110)
    print(
        f"  {'chunker':<22} {'retriever':<14} {'mean recall':>12} {'≥0.5':>6} {'≥0.8':>6} {'p50 ms':>8}"
    )
    print("  " + "-" * 80)

    for ck_name, ck_fn in CHUNKERS:
        for rt_name, rt_fn in RETRIEVERS:
            rec_sum = r50 = r80 = 0
            latencies = []
            for doc_text, query, gold in items_list:
                t0 = time.perf_counter()
                try:
                    chunks = ck_fn(doc_text)
                    candidates = rt_fn(chunks, query)
                    reranked = dense_rerank(query, candidates)
                    text = fill(reranked, BUDGET)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"  [{ck_name} × {rt_name}] error: {type(e).__name__}: {str(e)[:80]}",
                        file=sys.stderr,
                    )
                    text = ""
                latencies.append((time.perf_counter() - t0) * 1000)
                r = span_recall(gold, text)
                rec_sum += r
                r50 += int(r >= 0.5)
                r80 += int(r >= 0.8)
            latencies.sort()
            p50 = latencies[len(latencies) // 2] if latencies else 0
            print(
                f"  {ck_name:<22} {rt_name:<14} {rec_sum / n:>12.2f} "
                f"{100 * r50 / n:>5.0f}% {100 * r80 / n:>5.0f}% "
                f"{p50:>7.1f}"
            )


def main() -> None:
    n = 50  # 9 arms × 2 datasets × n=50 with dense rerank → manageable
    evaluate_matrix(hotpot_items(n), "HotpotQA — chunker × retriever × shared bge-small rerank")
    evaluate_matrix(musique_items(n), "MuSiQue — chunker × retriever × shared bge-small rerank")

    print()
    print("=" * 110)
    print("  INTERPRETATION GUIDE")
    print("=" * 110)
    print("""
  Reads the matrix row-by-row (vary retriever, hold chunker constant) to
  see which BM25 implementation wins on the SAME chunks. And column-by-
  column (vary chunker, hold retriever constant) to see how much the
  chunker choice matters.

  Hypothesis going in (from MULTIHOP_HYBRID_COMPETITORS):
  - On HotpotQA, RedHop's chunking won and BM25 winning was a bonus.
  - On MuSiQue, LangChain's chunking won and the BM25 differences were
    smaller. If MuSiQue rows are roughly flat across retrievers but
    vary big across chunkers, that confirms chunking is the lever.

  Three possible patterns:

  a) Chunker dominates: rows (fixed chunker, varying retriever) are flat,
     columns (varying chunker, fixed retriever) differ a lot. Chunk size
     is the real lever.
  b) Retriever dominates: rows differ a lot, columns are flat. The BM25
     implementation matters.
  c) Interaction: specific (chunker, retriever) pairs do well together.
     Coupling matters more than either alone.
""")


if __name__ == "__main__":
    main()
