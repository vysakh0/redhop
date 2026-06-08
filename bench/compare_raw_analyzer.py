#!/usr/bin/env python3
"""Does `language="raw"` (no stemming) help, hurt, or wash across workloads?

The multi-query bench surprised us: on CUAD `redhop[raw]` is both faster
AND has higher retention than the Snowball-stemming default. Theory: CUAD
queries use the EXACT clause names from the contracts (`"Change of
Control"`, `"Non-Compete"`), so stemming adds noise (`"settles"` and
`"settling"` both stem to `"settl"`) without recovering any missed matches.

The natural follow-up: does the result flip on workloads where users
PARAPHRASE the document vocabulary? HotpotQA and MuSiQue use natural-
language questions that don't echo the document verbatim — stemming
should help there.

This probe runs RedHop's default (English Snowball) vs `language="raw"`
on three workloads at n=100:
  - CUAD (templated, exact-match)
  - HotpotQA (natural-language, 2-hop)
  - MuSiQue (natural-language, compositional)

Single retrieval mode (lexical / raw_topk). Same chunks, same budget,
same candidate_k. The only thing that varies is the analyzer pipeline.

Run:  bench/.venv/bin/python bench/compare_raw_analyzer.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]
BUDGET = 2000
CANDIDATE_K = 40


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


def eval_arm(items_list, language: str | None, label: str):
    rec_sum = 0.0
    r80 = 0
    latencies = []
    for doc_text, query, gold in items_list:
        t0 = time.perf_counter()
        kwargs = dict(strategy="raw_topk", token_budget=BUDGET, candidate_k=CANDIDATE_K)
        if language is not None:
            kwargs["language"] = language
        try:
            doc = redhop.Document.from_text(doc_text, **kwargs)
            text = doc.context(query).text()
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] error: {e}", file=sys.stderr)
            text = ""
        latencies.append((time.perf_counter() - t0) * 1000)
        r = span_recall(gold, text)
        rec_sum += r
        r80 += int(r >= 0.8)
    n = max(len(items_list), 1)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    return rec_sum / n, r80 * 100 / n, p50


def report(name: str, items_iter):
    items_list = list(items_iter)
    n = len(items_list)
    print()
    print(f"  {name}  (n={n}, budget={BUDGET}, candidate_k={CANDIDATE_K})")
    print(f"  {'analyzer':<20} {'mean recall':>12} {'≥0.8':>6} {'p50 ms':>8}")
    print("  " + "-" * 56)
    eng_r, eng_80, eng_ms = eval_arm(items_list, None, "english (default)")
    raw_r, raw_80, raw_ms = eval_arm(items_list, "raw", "raw")
    print(f"  {'english (default)':<20} {eng_r:>12.2f} {eng_80:>5.0f}% {eng_ms:>7.1f}")
    print(f"  {'raw':<20} {raw_r:>12.2f} {raw_80:>5.0f}% {raw_ms:>7.1f}")
    print(
        f"  {'Δ (raw − english)':<20} {raw_r - eng_r:>+12.2f} "
        f"{raw_80 - eng_80:>+5.0f}  {raw_ms - eng_ms:>+7.1f}"
    )


def main() -> None:
    print()
    print("=" * 66)
    print("  RedHop default (English Snowball) vs language='raw' (no stem)")
    print("=" * 66)

    report("CUAD (templated, exact-match queries)", cuad_items(100))
    report("HotpotQA (natural-language, 2-hop)", hotpot_items(100))
    report("MuSiQue (natural-language, compositional)", musique_items(100))

    print()
    print("Interpretation:")
    print("  - 'raw' should win CUAD: queries use exact clause names from")
    print("    contracts; stemming adds noise (collisions like settles/settling)")
    print("    without recovering any missed matches.")
    print("  - 'raw' should LOSE on HotpotQA/MuSiQue: users paraphrase, so")
    print("    'highlighted' vs 'highlight' matches need stemming to fire.")
    print("  - If both predictions hold: 'language' becomes a real workload-")
    print("    aware knob, not just an i18n parameter.")


if __name__ == "__main__":
    main()
