"""Rerank the exported BM25 candidate pool with Model2Vec STATIC embeddings
(token lookup + mean-pool, no transformer runtime) and measure recall@3 — to see
how much of the BGE dense local-rerank result (~0.80) survives without the ONNX
model + native runtime.

Reranks the SAME pool the BGE experiment used (exports/rerank_pool.jsonl), so the
numbers are directly comparable. Baseline = BM25 pool order (no rerank).

Run:  bench/.venv/bin/python python/eval/static_rerank.py
"""

import json
import sys
from pathlib import Path

import numpy as np

POOL = Path(__file__).resolve().parents[2] / "exports" / "rerank_pool.jsonl"
TOP_K = 3
# Model2Vec (numpy-only, distilled static token lookup).
MODELS = [
    "minishlab/potion-retrieval-32M",
]
# SentenceTransformer StaticEmbedding (trained-for-retrieval static model; loads via
# torch EmbeddingBag, but conceptually still token-lookup + mean-pool).
ST_MODELS = [
    "sentence-transformers/static-retrieval-mrl-en-v1",
]
# Reference points measured on the identical pool (BGE-small via ONNX):
BGE_REF = {"lexical": 0.808, "semantic": 0.795, "ALL": 0.801}


def recall_at_k(ranked_ids, gold, k=TOP_K):
    hit = sum(1 for i in ranked_ids[:k] if i in gold)
    return hit / max(len(gold), 1)


def load_pool():
    rows = []
    with open(POOL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(per_subset):
    out = {}
    for s in ("lexical", "semantic", "ALL"):
        vals = per_subset[s]
        out[s] = sum(vals) / len(vals) if vals else 0.0
    return out


def main():
    rows = load_pool()
    print(f"loaded {len(rows)} queries from {POOL.name}")

    # BM25 pool-order baseline (candidates are already in BM25 rank order).
    bm = {"lexical": [], "semantic": [], "ALL": []}
    for r in rows:
        gold = set(r["gold"])
        ids = [c["id"] for c in r["candidates"]]
        v = recall_at_k(ids, gold)
        bm[r["subset"]].append(v)
        bm["ALL"].append(v)
    results = {"BM25 (pool order)": aggregate(bm)}

    from model2vec import StaticModel

    for name in MODELS:
        try:
            model = StaticModel.from_pretrained(name)
        except Exception as e:  # noqa: BLE001
            print(f"!! could not load {name}: {e}", file=sys.stderr)
            continue
        per = {"lexical": [], "semantic": [], "ALL": []}
        for r in rows:
            gold = set(r["gold"])
            cands = r["candidates"]
            texts = [r["question"]] + [c["text"] for c in cands]
            emb = model.encode(texts)
            emb = np.asarray(emb, dtype=np.float32)
            q = emb[0]
            d = emb[1:]
            qn = q / (np.linalg.norm(q) + 1e-9)
            dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
            sims = dn @ qn
            order = np.argsort(-sims)
            ranked_ids = [cands[i]["id"] for i in order]
            v = recall_at_k(ranked_ids, gold)
            per[r["subset"]].append(v)
            per["ALL"].append(v)
        results[name.split("/")[-1]] = aggregate(per)

    # SentenceTransformer static models (torch EmbeddingBag).
    try:
        from sentence_transformers import SentenceTransformer

        for name in ST_MODELS:
            try:
                model = SentenceTransformer(name)
            except Exception as e:  # noqa: BLE001
                print(f"!! could not load {name}: {e}", file=sys.stderr)
                continue
            per = {"lexical": [], "semantic": [], "ALL": []}
            for r in rows:
                gold = set(r["gold"])
                cands = r["candidates"]
                texts = [r["question"]] + [c["text"] for c in cands]
                emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                emb = np.asarray(emb, dtype=np.float32)
                sims = emb[1:] @ emb[0]
                order = np.argsort(-sims)
                ranked_ids = [cands[i]["id"] for i in order]
                v = recall_at_k(ranked_ids, gold)
                per[r["subset"]].append(v)
                per["ALL"].append(v)
            results[name.split("/")[-1]] = aggregate(per)
    except ImportError:
        print("(sentence-transformers not installed; skipping ST static models)", file=sys.stderr)

    # Table
    print(f"\nRecall@{TOP_K} by subset (same BM25 pool, static-embedding rerank)")
    print(f"  {'method':<28}{'lexical':>10}{'semantic':>10}{'ALL':>8}")
    print("  " + "-" * 56)
    for name, agg in results.items():
        print(f"  {name:<28}{agg['lexical']:>10.3f}{agg['semantic']:>10.3f}{agg['ALL']:>8.3f}")
    print(f"  {'BGE-small (ONNX) reference':<28}{BGE_REF['lexical']:>10.3f}{BGE_REF['semantic']:>10.3f}{BGE_REF['ALL']:>8.3f}")

    # Headline deltas vs BGE on the semantic slice
    print("\nvs BGE on semantic-heavy slice (the slice that matters):")
    for name, agg in results.items():
        if name.startswith("BM25"):
            continue
        d = agg["semantic"] - BGE_REF["semantic"]
        print(f"  {name:<28} semantic R@3 {agg['semantic']:.3f}  ({d:+.3f} vs BGE)")


if __name__ == "__main__":
    main()
