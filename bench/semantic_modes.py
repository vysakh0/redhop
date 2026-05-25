#!/usr/bin/env python
"""Lexical vs local-rerank vs global-dense on the semantic-mismatch probe.

Each item has a query, a GOLD passage (semantically right, low lexical overlap),
a TRAP (high lexical overlap, wrong meaning — a BM25 attractor), and distractors.
All passages are pooled into one corpus; for each query we check whether GOLD is
retrieved (recall@1 / recall@3) under each retrieval mode.

This isolates exactly where global dense should beat local rerank: when GOLD shares
no terms with the query, BM25's pool never contains it, so local rerank can't
recover it — but global dense scores every chunk.

Run: HF_HUB_OFFLINE=0 bench/.venv/bin/python bench/semantic_modes.py
(needs the [onnx] build + downloads bge-small on first use).
"""
import json
from collections import defaultdict

import redhop

DATA = "data/semantic_mismatch.json"
items = json.load(open(DATA))["items"]

# Pool every passage into one corpus (gold + trap + distractors from all items).
pool = []
for it in items:
    pool.append(it["gold"])
    pool.append(it["trap"])
    pool.extend(it["distractors"])
pool = list(dict.fromkeys(pool))  # dedup, preserve order
print(f"{len(items)} queries · {len(pool)} passages pooled\n")

MODES = [
    ("lexical (BM25)", dict(retrieval="lexical")),
    ("global dense", dict(retrieval="dense", model="bge-small")),
]

# Short passages (~12–20 tok); budgets sized for ~1 and ~3 passages.
def recall_at(doc, query, gold, budget):
    ctx = doc.context(query, budget=budget).text()
    return gold in ctx

results = {}
for name, kw in MODES:
    # Each passage is its own chunk (1 passage = 1 chunk) so retrieval ranks passages.
    doc = redhop.Document.from_chunks(pool, candidate_k=len(pool), **kw)
    r1 = defaultdict(lambda: [0, 0])  # category -> [hits, total]
    r3 = defaultdict(lambda: [0, 0])
    for it in items:
        cat = it["category"]
        r1[cat][1] += 1; r3[cat][1] += 1
        if recall_at(doc, it["query"], it["gold"], 25):
            r1[cat][0] += 1
        if recall_at(doc, it["query"], it["gold"], 70):
            r3[cat][0] += 1
    results[name] = (r1, r3)

cats = ["paraphrase", "legal_synonymy", "reformulation", "low_overlap", "control"]
def overall(r):
    h = sum(v[0] for v in r.values()); t = sum(v[1] for v in r.values())
    return h, t

for metric, idx in [("recall@1", 0), ("recall@3", 1)]:
    print(f"== {metric} (GOLD retrieved) ==")
    hdr = f"{'mode':<16}" + "".join(f"{c[:10]:>12}" for c in cats) + f"{'ALL':>10}"
    print(hdr); print("-" * len(hdr))
    for name, _ in MODES:
        r = results[name][idx]
        row = f"{name:<16}"
        for c in cats:
            h, t = r[c]
            row += f"{(f'{h}/{t}'):>12}"
        h, t = overall(r)
        row += f"{(f'{100*h//t}%'):>10}"
        print(row)
    print()
