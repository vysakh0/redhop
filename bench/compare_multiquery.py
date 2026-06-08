#!/usr/bin/env python3
"""Multi-query-per-document benchmark — the chatbot / knowledge-base pattern.

`bench/compare.py` measures one-query-per-doc: build an index from scratch,
ask one question, throw away the index. That's the right benchmark for
stateless "answer this single question from this single PDF" jobs, but
it MISSES the most common production pattern:

  - chat apps with the same document in context across a conversation
  - knowledge-base lookups (build the index once, then answer everything)
  - support agents iterating on the same ticket
  - long-running document review sessions

All of those build the index ONCE and ask many queries against it.
What matters there:

  - **Cold cost**: index-build time (chunking + BM25 indexing +
    embeddings if dense). Paid once.
  - **Warm cost**: per-query time after the index exists. Paid N times.
  - **Total wall-clock** for the realistic pattern.

This bench reports all three.

Setup: CUAD has many questions per contract. We take N=10 contracts and
ask M=10 questions per contract = 100 queries against 10 indices. Each
system gets the same workload.

Run (from the bench venv):  bench/.venv/bin/python bench/compare_multiquery.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import redhop
from langchain_community.retrievers import BM25Retriever as LCBm25
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document as LIDoc
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever as LIBm25

REPO = Path(__file__).resolve().parents[1]
CHUNK_TOKENS = 256
CANDIDATE_K = 40
BUDGET = 2000


# ── Metric (kept identical to compare.py) ──────────────────────────────────


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


# ── System adapters — split into "index" (cold) and "query" (warm) ────────


def redhop_index(doc_text: str):
    return redhop.Document.from_text(
        doc_text,
        strategy="raw_topk",
        token_budget=BUDGET,
        candidate_k=CANDIDATE_K,
    )


def redhop_query(doc, query: str) -> str:
    return doc.context(query).text()


def redhop_raw_index(doc_text: str):
    """Same as redhop_index but with language='raw' — minimal Tantivy
    pipeline (no stemming, no stopword filter, no CamelCase). Opt-in path
    for users who want LangChain-style warm-query latency."""
    return redhop.Document.from_text(
        doc_text,
        strategy="raw_topk",
        token_budget=BUDGET,
        candidate_k=CANDIDATE_K,
        language="raw",
    )


def redhop_raw_query(doc, query: str) -> str:
    return doc.context(query).text()


def langchain_index(doc_text: str):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS * 4, chunk_overlap=40
    ).split_text(doc_text)
    if not chunks:
        return None
    retr = LCBm25.from_texts(chunks)
    retr.k = CANDIDATE_K
    return retr


def langchain_query(retr, query: str) -> str:
    if retr is None:
        return ""
    hits = retr.invoke(query)
    return fill([d.page_content for d in hits], BUDGET)


def llamaindex_index(doc_text: str):
    nodes = SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=20).get_nodes_from_documents(
        [LIDoc(text=doc_text)]
    )
    if not nodes:
        return None
    return LIBm25.from_defaults(nodes=nodes, similarity_top_k=min(CANDIDATE_K, len(nodes)))


def llamaindex_query(retr, query: str) -> str:
    if retr is None:
        return ""
    hits = retr.retrieve(query)
    return fill([h.node.get_content() for h in hits], BUDGET)


SYSTEMS = [
    ("redhop[topk]", redhop_index, redhop_query),
    ("redhop[raw]", redhop_raw_index, redhop_raw_query),
    ("langchain", langchain_index, langchain_query),
    ("llamaindex", llamaindex_index, llamaindex_query),
]


# ── CUAD multi-query workload ──────────────────────────────────────────────


def cuad_multi_query_items(n_docs: int, m_queries_per_doc: int):
    """Yield (doc_text, [(query, gold), ...]) tuples — n_docs contracts,
    each paired with m_queries_per_doc questions about that contract."""
    data = json.loads((REPO / "data/cuad/cuad_sample.json").read_text())["data"]
    yielded = 0
    for c in data:
        for p in c["paragraphs"]:
            queries: list[tuple[str, str]] = []
            for qa in p["qas"]:
                gold = qa["answers"][0]["text"] if qa["answers"] else ""
                if gold:
                    queries.append((qa["question"], gold))
                if len(queries) >= m_queries_per_doc:
                    break
            if len(queries) >= m_queries_per_doc:
                yield p["context"], queries
                yielded += 1
                if yielded >= n_docs:
                    return


# ── Runner ─────────────────────────────────────────────────────────────────


def evaluate_multiquery(n_docs: int, m_queries_per_doc: int):
    items = list(cuad_multi_query_items(n_docs, m_queries_per_doc))
    actual_docs = len(items)
    total_queries = sum(len(q) for _, q in items)
    print()
    print("=" * 96)
    print(
        f"  CUAD multi-query — {actual_docs} contracts × {m_queries_per_doc} questions each = {total_queries} queries"
    )
    print(f"  budget={BUDGET} tok, candidate_k={CANDIDATE_K}")
    print("=" * 96)
    print(
        f"  {'system':<14} {'cold p50':>10} {'warm p50':>10} {'warm p99':>10} {'total':>10} {'mean recall':>13} {'≥0.8':>6}"
    )
    print("  " + "-" * 80)

    for name, index_fn, query_fn in SYSTEMS:
        cold_times: list[float] = []
        warm_times: list[float] = []
        rec_sum = 0.0
        r80 = 0
        total_start = time.perf_counter()
        for doc_text, qa_pairs in items:
            t0 = time.perf_counter()
            try:
                doc_obj = index_fn(doc_text)
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] index error: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                continue
            cold_times.append((time.perf_counter() - t0) * 1000)

            for i, (query, gold) in enumerate(qa_pairs):
                t0 = time.perf_counter()
                try:
                    text = query_fn(doc_obj, query)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"  [{name}] query error: {type(e).__name__}: {str(e)[:80]}",
                        file=sys.stderr,
                    )
                    text = ""
                # First query post-index has some cold-cache overhead even
                # within "warm"; we still call it warm here because the
                # *index* exists. The cold/warm split is index-build vs
                # subsequent queries against the standing index.
                warm_times.append((time.perf_counter() - t0) * 1000)
                r = span_recall(gold, text)
                rec_sum += r
                r80 += int(r >= 0.8)

        total_ms = (time.perf_counter() - total_start) * 1000
        cold_times.sort()
        warm_times.sort()
        n_warm = max(len(warm_times), 1)
        cold_p50 = cold_times[len(cold_times) // 2] if cold_times else 0
        warm_p50 = warm_times[len(warm_times) // 2] if warm_times else 0
        warm_p99 = warm_times[max(0, int(len(warm_times) * 0.99) - 1)] if warm_times else 0
        n_queries = max(len(warm_times), 1)
        print(
            f"  {name:<14} "
            f"{cold_p50:>9.0f}ms "
            f"{warm_p50:>9.1f}ms "
            f"{warm_p99:>9.0f}ms "
            f"{total_ms:>9.0f}ms "
            f"{rec_sum / n_queries:>13.2f} "
            f"{100 * r80 / n_queries:>5.0f}%"
        )

    print()
    print("Interpretation:")
    print("  - cold p50: per-doc index-build time. Paid ONCE per doc.")
    print("  - warm p50: per-query time AFTER the index exists. Paid M times per doc.")
    print("  - total:    end-to-end wall-clock for all docs × queries.")
    print("    For the chatbot/KB pattern, this is the number that matters.")
    print("  - mean recall + ≥0.8: same retention metric as bench/compare.py.")
    print("    Verifies the multi-query split didn't drift the rankings.")


def main() -> None:
    evaluate_multiquery(n_docs=10, m_queries_per_doc=10)


if __name__ == "__main__":
    main()
