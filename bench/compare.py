#!/usr/bin/env python3
"""Head-to-head context-assembly benchmark: RedHop vs LangChain vs LlamaIndex.

Tier 1 (free, local, deterministic — no LLM). For each (document, query) we ask
each framework to assemble a context under the SAME token budget using BM25
retrieval, then measure:

  - tokens   : whitespace tokens in the assembled context (counted identically
               for all three, so it's apples-to-apples)
  - retention: gold-evidence word-recall in the assembled context
               (CUAD: the gold answer span; HotpotQA: the gold supporting
               sentences — the multi-hop evidence)

Fairness:
  - All three retrieve with BM25 (LangChain/LlamaIndex BM25Retriever, RedHop's
    internal Tantivy BM25) — isolates assembly from retrieval-engine choice.
  - Same token budget, chosen BELOW the document size so selection is actually
    forced (otherwise the whole doc fits and every system trivially retains 100%).
  - RedHop runs `reasoning_preserving` (its assembly strategy under test) — note
    that the default Auto policy would simply pass small docs through unpruned.
  - Comparable chunk sizes (~256 tokens) across all three.

This is an honest test: it can show "comparable" as readily as "RedHop wins".

Run (from the bench venv):  bench/.venv/bin/python bench/compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import redhop
from langchain_community.retrievers import BM25Retriever as LCBm25
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document as LIDoc
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever as LIBm25

REPO = Path(__file__).resolve().parents[1]
CHUNK_TOKENS = 256
CANDIDATE_K = 40  # generous retrieval pool; the budget does the cutting


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
    """Greedily concatenate retrieved texts (in rank order) until the token
    budget is hit — the natural 'stuff the top hits' context."""
    out, used = [], 0
    for t in ordered:
        n = toks(t)
        if used + n > budget and out:
            break
        out.append(t)
        used += n
    return "\n\n".join(out)


# ---- the three systems: (doc_text, query, budget) -> assembled context ----

def ctx_redhop(doc_text: str, query: str, budget: int, strategy: str = "reasoning_preserving") -> str:
    doc = redhop.Document.from_text(
        doc_text, strategy=strategy, token_budget=budget, candidate_k=CANDIDATE_K
    )
    return doc.context(query).text()


def ctx_langchain(doc_text: str, query: str, budget: int) -> str:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS * 4, chunk_overlap=40
    ).split_text(doc_text)
    if not chunks:
        return ""
    retr = LCBm25.from_texts(chunks)
    retr.k = CANDIDATE_K
    hits = retr.invoke(query)
    return fill([d.page_content for d in hits], budget)


def ctx_llamaindex(doc_text: str, query: str, budget: int) -> str:
    nodes = SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=20).get_nodes_from_documents(
        [LIDoc(text=doc_text)]
    )
    if not nodes:
        return ""
    retr = LIBm25.from_defaults(nodes=nodes, similarity_top_k=min(CANDIDATE_K, len(nodes)))
    hits = retr.retrieve(query)
    return fill([h.node.get_content() for h in hits], budget)


SYSTEMS = {
    # RedHop variants isolate whether the gap is the *strategy* (dropping /
    # under-filling) or the pipeline: reasoning_preserving drops unlinked
    # low-relevance chunks; max_density / raw_topk fill the budget like the
    # other frameworks do.
    "redhop[reason]": lambda d, q, b: ctx_redhop(d, q, b, "reasoning_preserving"),
    "redhop[density]": lambda d, q, b: ctx_redhop(d, q, b, "max_density"),
    "redhop[topk]": lambda d, q, b: ctx_redhop(d, q, b, "raw_topk"),
    "langchain": ctx_langchain,
    "llamaindex": ctx_llamaindex,
}


def evaluate(items, budget: int, label: str):
    """items: iterable of (doc_text, query, gold_text)."""
    agg = {s: {"tok": 0.0, "rec": 0.0, "r50": 0, "r80": 0, "n": 0} for s in SYSTEMS}
    for doc_text, query, gold in items:
        for name, fn in SYSTEMS.items():
            try:
                ctx = fn(doc_text, query, budget)
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] error: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                ctx = ""
            r = span_recall(gold, ctx)
            a = agg[name]
            a["tok"] += toks(ctx)
            a["rec"] += r
            a["r50"] += int(r >= 0.5)
            a["r80"] += int(r >= 0.8)
            a["n"] += 1

    n_items = next(iter(agg.values()))["n"]
    print(f"\n==== {label}  (budget {budget} tok, BM25, n={n_items}) ====")
    print(f"  {'system':<16} {'avg tokens':>10} {'mean recall':>12} {'≥0.5':>6} {'≥0.8':>6}")
    print("  " + "-" * 54)
    for name in SYSTEMS:
        a = agg[name]
        n = max(a["n"], 1)
        print(
            f"  {name:<16} {a['tok'] / n:>10.0f} {a['rec'] / n:>12.2f} "
            f"{100 * a['r50'] / n:>5.0f}% {100 * a['r80'] / n:>5.0f}%"
        )


def cuad_items(limit_q: int):
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
    """MuSiQue (compositional multi-hop): each example bundles 20 distractor
    paragraphs around the ~2-4 supporting ones. Gold-evidence is the union
    of the supporting paragraphs, mirroring HotpotQA's gold sentences. Same
    word-recall metric — so the multi-hop retention story we tell on
    HotpotQA can be checked on a second dataset."""
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


# ── Fair-preprocessing comparison ────────────────────────────────────────
# The default `evaluate` run above compares "RedHop with its assembly
# strategies" vs "LangChain/LlamaIndex with their defaults" — the raw
# CUAD template lands on every system unmodified, and the public 90.7%
# claim ends up apples-to-oranges (RedHop's published recipe applies
# Stripper + Vocabulary to the query, the other two don't).
#
# This second run applies the EXACT same query-side preprocessing that
# RedHop ships in its Stripper to all three systems before retrieval.
# It answers a different and more useful question: "given the same
# preprocessing budget, which system retrieves best?"

# Boilerplate list matches crates/examples/examples/cuad_clause_expansion.rs
# (the source of the published 87.7% / 90.7% numbers).
CUAD_BOILERPLATE = [
    "highlight", "the", "parts", "if", "any", "of", "this", "contract",
    "related", "to", "that", "should", "be", "reviewed", "by", "a",
    "lawyer", "details",
]


def cuad_stripped_items(limit_q: int):
    """Same CUAD items as `cuad_items` but with each query template-stripped
    via redhop.Stripper. All three systems see identical preprocessed queries
    — the comparison is then apples-to-apples on the assembly side."""
    stripper = redhop.Stripper(CUAD_BOILERPLATE)
    for doc_text, query, gold in cuad_items(limit_q):
        stripped = stripper.apply(query)
        yield doc_text, stripped, gold


def main() -> None:
    # CUAD: long contracts, budget well below doc size → forces selection.
    evaluate(cuad_items(300), budget=2000, label="CUAD (contracts) — raw template, all systems default")
    # FAIR-PREPROCESSING ARM: same Stripper applied to the query before all 3 systems.
    # Lets the reader see "what RedHop's preprocessing gains independent of
    # which retrieval engine carries it" vs the misleading
    # "RedHop+preprocessing vs others+default" framing.
    evaluate(
        cuad_stripped_items(300),
        budget=2000,
        label="CUAD (contracts) — Stripper applied to query for ALL systems (fair preprocessing)",
    )
    # HotpotQA: multi-hop; tight budget forces dropping paragraphs → tests whether
    # the gold supporting (incl. low-relevance bridge) sentences survive.
    evaluate(hotpot_items(300), budget=400, label="HotpotQA (multi-hop) — supporting-sentence retention")
    # MuSiQue: compositional multi-hop (harder than HotpotQA — answers
    # require 2-4 reasoning hops, with 20 distractor paragraphs per example).
    # Second dataset for the multi-hop retention claim that previously
    # rested on HotpotQA alone.
    evaluate(musique_items(300), budget=400, label="MuSiQue (compositional multi-hop) — supporting-paragraph retention")


if __name__ == "__main__":
    main()
