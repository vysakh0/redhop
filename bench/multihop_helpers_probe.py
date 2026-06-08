#!/usr/bin/env python3
"""Do the helpers (Stripper, Vocabulary, retrieval tiers) push multi-hop
retention further than the BM25 default?

The 0.3.1 audit established that RedHop leads multi-hop retention on
HotpotQA (+8 ≥0.8 over LlamaIndex) and MuSiQue (+3 to +5), and that
`raw_topk` matches `reasoning_preserving` on both — so the assembly
strategy isn't the differentiator. The natural next question: can the
helpers we ship (`Stripper`, `Vocabulary.apply`, `retrieval="hybrid"`,
candidate_k tuning) push the numbers further? Or is BM25-default the cap?

Mechanism predictions before the run:

- **Stripper:** `analyze_query_set` reports template_word_share = 0.000
  (HotpotQA) and 0.118 (MuSiQue) — both below the 0.50 threshold.
  Predict: near no-op. Test that it's not silently *harmful*.

- **Vocabulary.apply:** would need a workload-curated synonym dict.
  HotpotQA/MuSiQue are open-domain Wikipedia Q&A — no obvious shared
  vocabulary. Authoring a dict against the gold answers would be
  curator-conflicted (the SPIDER_ENRICH trap). **Not measured** in this
  probe; documented as "out of scope without an independently-sourced
  synonym corpus" in the writeup.

- **Vocabulary.enrich on chunks:** out of regime — multi-hop paragraphs
  are prose, neither short nor opaque. Predict null-or-harm by the
  four-corner observation (workload-pervasive signal manipulation on
  prose).

- **retrieval="hybrid":** the multi-hop failure mode is the *bridge*
  paragraph — semantically related to the question but lexically
  distant ("Who is the spouse of the Green performer?" needs to retrieve
  the paragraph about "Steve Hillage and Miquette Giraudy" even though
  it shares few words with the query). Dense rerank should rescue some
  of these. Real test of whether we can push past BM25-default.

- **larger candidate_k:** cheap knob; gives the retriever more candidates
  to work with before assembly. Should help when the bridge passage was
  in the candidate pool but pruned by the budget. Test diminishing
  returns.

Run:  bench/.venv/bin/python bench/multihop_helpers_probe.py

Note: the hybrid arm downloads a small ONNX model on first run (cached).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]


# ── Data loaders (same shape as bench/compare.py) ──────────────────────────


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


# ── Arms ────────────────────────────────────────────────────────────────────
# Each arm takes (doc_text, query, budget) → assembled context string.
# The doc construction is what varies between arms; the metric is identical.

BUDGET = 400
CANDIDATE_K_DEFAULT = 20
CANDIDATE_K_LARGE = 60  # 3x default, sees more bridge candidates


def arm_bm25_baseline(doc_text: str, query: str, budget: int = BUDGET) -> str:
    """Arm A: BM25 default + raw_topk (matches what bench/compare.py uses)."""
    doc = redhop.Document.from_text(
        doc_text, strategy="raw_topk", token_budget=budget, candidate_k=CANDIDATE_K_DEFAULT
    )
    return doc.context(query).text()


def arm_bm25_stripper(doc_text: str, query: str, budget: int = BUDGET, stripper=None) -> str:
    """Arm B: BM25 + Stripper preprocessing on the query. Stripper is
    deliberately attached AT THE QUERY level via context_with_rewrites so
    the chunks index stays bare BM25."""
    doc = redhop.Document.from_text(
        doc_text, strategy="raw_topk", token_budget=budget, candidate_k=CANDIDATE_K_DEFAULT
    )
    return doc.context_with_rewrites(query, [stripper]).text()


def arm_bm25_large_k(doc_text: str, query: str, budget: int = BUDGET) -> str:
    """Arm D: BM25 with 3x candidate_k. Tests whether the bridge passage
    was in the larger pool but missed at default candidate_k."""
    doc = redhop.Document.from_text(
        doc_text, strategy="raw_topk", token_budget=budget, candidate_k=CANDIDATE_K_LARGE
    )
    return doc.context(query).text()


def arm_hybrid(doc_text: str, query: str, budget: int = BUDGET) -> str:
    """Arm C: hybrid retrieval (BM25 candidate pool → dense rerank).
    The real test: can dense rerank rescue bridge passages that are
    semantically close but lexically distant?"""
    doc = redhop.Document.from_text(
        doc_text,
        strategy="raw_topk",
        token_budget=budget,
        candidate_k=CANDIDATE_K_DEFAULT,
        retrieval="hybrid",
        model="bge-small",
    )
    return doc.context(query).text()


# ── Runner ─────────────────────────────────────────────────────────────────


def evaluate_arms(items, label: str, queries_for_analyzer: list[str]):
    """items: iterator of (doc_text, query, gold). Runs all arms,
    reports retention + timing."""
    # Detect-before-applying check.
    print()
    print("=" * 96)
    print(f"  {label}")
    print("=" * 96)
    report = redhop.analyze_query_set(queries_for_analyzer)
    print(
        f"  analyze_query_set says: is_templated={report.is_templated}, "
        f"template_word_share={report.template_word_share:.3f}, "
        f"boilerplate_count={len(report.boilerplate_terms)}, "
        f"dilution={report.estimated_dilution_cost}"
    )
    print(
        f"  → Stripper recommendation: "
        f"{'YES (templated workload)' if report.is_templated else 'NO (not templated; expect Stripper no-op)'}"
    )
    # Build Stripper from analyzer-extracted terms if it fires, else use
    # a short generic stopword-like list to verify the no-op claim.
    if report.is_templated:
        stripper_terms = report.boilerplate_terms
    else:
        # Deliberately give Stripper terms that ARE in some queries so
        # we can tell apart "Stripper does nothing because empty input"
        # from "Stripper does nothing because the analyzer-stem mismatch".
        stripper_terms = ["what", "who", "where", "when", "the", "a", "an", "is", "of", "in"]
    stripper = redhop.Stripper(stripper_terms)

    arms = [
        ("A. BM25 baseline (raw_topk, k=20)", lambda d, q, b: arm_bm25_baseline(d, q, b)),
        ("B. BM25 + Stripper (query-side)", lambda d, q, b: arm_bm25_stripper(d, q, b, stripper=stripper)),
        ("D. BM25 with candidate_k=60 (3x default)", lambda d, q, b: arm_bm25_large_k(d, q, b)),
        ("C. retrieval='hybrid' (BM25 + dense rerank)", lambda d, q, b: arm_hybrid(d, q, b)),
    ]

    agg = {a[0]: {"rec": 0.0, "r50": 0, "r80": 0, "n": 0, "ms_total": 0.0} for a in arms}
    items_list = list(items)
    print(f"  n = {len(items_list)} queries, budget = {BUDGET} tok")
    print()
    print(f"  {'arm':<48} {'mean recall':>12} {'≥0.5':>6} {'≥0.8':>6} {'p50 ms':>8}")
    print("  " + "-" * 86)

    for arm_name, arm_fn in arms:
        latencies = []
        for doc_text, query, gold in items_list:
            t0 = time.perf_counter()
            try:
                ctx = arm_fn(doc_text, query, BUDGET)
            except Exception as e:  # noqa: BLE001
                print(f"  [{arm_name}] error: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                ctx = ""
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            r = span_recall(gold, ctx)
            a = agg[arm_name]
            a["rec"] += r
            a["r50"] += int(r >= 0.5)
            a["r80"] += int(r >= 0.8)
            a["n"] += 1
            a["ms_total"] += elapsed_ms

        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0
        a = agg[arm_name]
        n = max(a["n"], 1)
        print(
            f"  {arm_name:<48} {a['rec'] / n:>12.2f} "
            f"{100 * a['r50'] / n:>5.0f}% {100 * a['r80'] / n:>5.0f}% "
            f"{p50:>7.1f}"
        )


def main() -> None:
    # Smaller n than bench/compare.py because the hybrid arm downloads
    # a model on first use and is slower per query (dense embedding +
    # rerank). 100 is enough to see the direction; 300 would tighten
    # but takes ~5× longer on the hybrid arm.
    n = 100

    # HotpotQA: where the multi-hop lead was strongest at default BM25.
    hotpot = list(hotpot_items(n))
    evaluate_arms(
        iter(hotpot),
        label=f"HotpotQA (multi-hop) — helper arms",
        queries_for_analyzer=[q for _, q, _ in hotpot],
    )

    # MuSiQue: where the lead shrunk. Most useful test of "can helpers
    # push the harder workload?"
    musique = list(musique_items(n))
    evaluate_arms(
        iter(musique),
        label=f"MuSiQue (compositional multi-hop) — helper arms",
        queries_for_analyzer=[q for _, q, _ in musique],
    )

    print()
    print("=" * 96)
    print("  INTERPRETATION GUIDE")
    print("=" * 96)
    print("""
  - Arm A is the BM25 baseline. Compare every other arm against this.
  - Arm B (Stripper) is the no-op verification. If retention shifts more
    than ~1 point, it means the analyzer-token Stripper is touching
    something it shouldn't on non-templated queries — that would be a
    bug, not a feature.
  - Arm C (hybrid) is the real test. If dense rerank rescues bridge
    passages, retention should climb 2-10 points. If it doesn't, the
    multi-hop ceiling is set by something other than the BM25 ranker.
  - Arm D (larger candidate_k) is the cheap-knob alternative. If it
    matches or beats hybrid, the answer is "more candidates, not denser
    candidates."

  What this probe does NOT test:
  - Vocabulary.apply: would need an independent synonym corpus to avoid
    curator conflict on these open-domain workloads. Out of scope.
  - Vocabulary.enrich on chunks: predicted null-or-harm by the
    four-corner observation (workload-pervasive signal on prose). Not
    measured to save runtime; predicted negative.
  - Cross-encoder rerank on top of hybrid: would slow per-query latency
    ~10x; probably out of scope for production multi-hop where the user
    cares about latency.
""")


if __name__ == "__main__":
    main()
