#!/usr/bin/env python3
"""Calibration probe for `redhop.analyze_query_set`.

`analyze_query_set` is the workload-shape detector that decides whether a
query set looks templated enough to recommend `Stripper`. Until now the
finding (QUERY_SET_ANALYZER.md) documented that the heuristic *didn't*
flag false positives on HotpotQA + MuSiQue (two negative controls) and
*did* fire on CUAD (one positive). That's three workloads, no precision/
recall, no threshold sensitivity — enough to demonstrate the mechanism,
not enough to calibrate the API for users on a new workload.

This script closes that gap. It runs `analyze_query_set` on:

  Positive (known templated):
    - CUAD's quoted-clause template
    - A synthetic stopword-heavy template
    - A short fixed-prefix support-ticket template

  Negative (known diverse):
    - HotpotQA (natural-language multi-hop)
    - MuSiQue (compositional multi-hop)
    - A synthetic free-text question set

For each: report is_templated (the binary decision) plus the supporting
scalars (template_word_share, boilerplate_terms count, estimated_dilution_cost,
suggested_action). Compute precision and recall over the labelled set and
print a summary that QUERY_SET_ANALYZER.md can cite.

Run:  bench/.venv/bin/python bench/query_set_analyzer_calibration.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]
N_QUERIES = 300  # match QUERY_SET_ANALYZER.md's existing per-workload n


# ── Workload loaders ────────────────────────────────────────────────────────


def cuad_queries(n: int) -> list[str]:
    data = json.loads((REPO / "data/cuad/cuad_sample.json").read_text())["data"]
    out = []
    for c in data:
        for p in c["paragraphs"]:
            for qa in p["qas"]:
                if len(out) >= n:
                    return out
                out.append(qa["question"])
    return out


def hotpotqa_queries(n: int) -> list[str]:
    data = json.loads((REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())
    return [ex["question"] for ex in data[:n]]


def musique_queries(n: int) -> list[str]:
    out = []
    with (REPO / "data/musique/dev.jsonl").open() as f:
        for line in f:
            ex = json.loads(line)
            out.append(ex["question"])
            if len(out) >= n:
                break
    return out


def synthetic_template_queries(n: int) -> list[str]:
    """A short fixed-prefix template — the support-ticket / form-filled
    pattern that's the second canonical positive (per the audit follow-up).
    Heavier boilerplate ratio than CUAD's 24-word frame so the analyzer
    should fire even more confidently."""
    rng = random.Random(42)
    topics = [
        "billing", "refund", "shipping", "return", "warranty",
        "password reset", "account access", "order status",
        "invoice", "subscription", "trial extension", "downgrade",
        "cancellation policy", "tax exemption", "address change",
    ]
    return [
        f"Please help me with my {rng.choice(topics)} issue, my account is broken and "
        f"I would like immediate assistance regarding this matter, thank you."
        for _ in range(n)
    ]


def synthetic_diverse_queries(n: int) -> list[str]:
    """A control negative — natural-language questions with no shared frame.
    Cribbed from a public natural-questions style; if the analyzer fires
    here that's a false positive."""
    rng = random.Random(7)
    starters = [
        "What is", "How does", "Why did", "When was", "Where does",
        "Who wrote", "Which", "Can you explain", "Tell me about",
        "Is it true that",
    ]
    subjects = [
        "the speed of light",
        "photosynthesis work in deep sea environments",
        "the French Revolution end",
        "the Hubble telescope launched",
        "the Amazon rainforest's biome carbon cycle",
        "the Magna Carta",
        "country has the most volcanoes",
        "quantum entanglement to a beginner",
        "the history of the Silk Road",
        "spiders only have eight legs",
    ]
    return [f"{rng.choice(starters)} {rng.choice(subjects)}?" for _ in range(n)]


WORKLOADS: list[tuple[str, str, list[str]]] = [
    # (name, label, queries)
    # label: "positive" (should fire) or "negative" (should NOT fire)
]


def collect_workloads() -> None:
    WORKLOADS.extend(
        [
            ("CUAD", "positive", cuad_queries(N_QUERIES)),
            ("Synthetic template (support-ticket)", "positive", synthetic_template_queries(N_QUERIES)),
            ("HotpotQA", "negative", hotpotqa_queries(N_QUERIES)),
            ("MuSiQue", "negative", musique_queries(N_QUERIES)),
            ("Synthetic diverse", "negative", synthetic_diverse_queries(N_QUERIES)),
        ]
    )


# ── Calibration ─────────────────────────────────────────────────────────────


def main() -> None:
    collect_workloads()
    print()
    print("=" * 96)
    print(f"{'workload':<38} {'label':<10} {'fires':<7} {'word_share':>10} {'boil#':>6} {'dilution':<8}")
    print("-" * 96)
    rows: list[dict] = []
    for name, label, queries in WORKLOADS:
        if not queries:
            print(f"{name:<38} {label:<10} [no data]")
            continue
        report = redhop.analyze_query_set(queries)
        fires = "YES" if report.is_templated else "no"
        rows.append(
            {
                "name": name,
                "label": label,
                "fires": report.is_templated,
                "word_share": report.template_word_share,
                "boilerplate_count": len(report.boilerplate_terms),
                "dilution": report.estimated_dilution_cost,
                "suggested_action": report.suggested_action,
                "boilerplate_sample": report.boilerplate_terms[:8],
            }
        )
        print(
            f"{name:<38} {label:<10} {fires:<7} "
            f"{report.template_word_share:>10.3f} {len(report.boilerplate_terms):>6} "
            f"{report.estimated_dilution_cost:<8}"
        )
    print("=" * 96)

    # ── Precision / recall on the binary decision ──
    tp = sum(1 for r in rows if r["label"] == "positive" and r["fires"])
    fp = sum(1 for r in rows if r["label"] == "negative" and r["fires"])
    fn = sum(1 for r in rows if r["label"] == "positive" and not r["fires"])
    tn = sum(1 for r in rows if r["label"] == "negative" and not r["fires"])
    n_pos = tp + fn
    n_neg = fp + tn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print()
    print(f"Confusion:  TP={tp}  FP={fp}  FN={fn}  TN={tn}   (n_pos={n_pos}, n_neg={n_neg})")
    print(f"Precision:  {precision:.2f}   (false-positive cost: 0 / positives fired)")
    print(f"Recall:     {recall:.2f}   (false-negative cost: missed positives)")
    print()
    print("Boilerplate terms (top 8) per workload — sanity check that the analyzer")
    print("identified workload-pervasive terms, not corpus-pervasive ones:")
    for r in rows:
        if r["fires"]:
            print(f"  {r['name']:<38} {r['boilerplate_sample']}")
    print()
    print("Note: this is a 5-workload calibration with synthetic positives + negatives")
    print("alongside the natural ones (CUAD/HotpotQA/MuSiQue). Not a generalization")
    print("claim — it bounds the heuristic on the shapes we have, not on all possible")
    print("query sets.")


if __name__ == "__main__":
    main()
