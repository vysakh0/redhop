#!/usr/bin/env python3
"""Does `code_neighbors_default=1` actually help on code retrieval, or
is the auto-pull just inflating context with redundant lines?

When `Document.context(query)` finds a code chunk
(`metadata["kind"]=="code"`), the default auto-attaches the ±1 adjacent
chunks in the same file. The design intent: a `def function_name():`
line by itself is useless without the body, and code is chunked at a
fixed token granularity (128 by default), so a function often spans
2-3 chunks. Pulling neighbors keeps the implementation together.

That's the intuition. Nobody has measured whether it actually changes
what queries see — `bench/code_retrieval.py` measures retrieval mode
(lexical vs hybrid vs dense), not neighbor expansion. Same shape of
audit-needed default as `prose_heading_default` was before we measured
it (and validated it).

This probe loads RedHop's own Rust source via `Document.from_file()`
(so the `.rs` extension triggers `kind="code"` automatically) and runs
the same queries `code_retrieval.py` uses, but with **body-bearing
markers** — distinctive tokens that appear *only inside* the function
body, not in its signature or docstring. Whether a marker shows up in
the assembled context tells us whether the user gets the implementation
along with the def-line hit.

Arms:
- **A. neighbors=0** — `doc.context(q, neighbors=0, include_heading=True)`
  bypasses the auto path (code_neighbors_default doesn't fire);
  include_heading=True is a no-op because code chunks carry no heading
  metadata, so this is a clean "no expansion" baseline.
- **B. neighbors=1 (current default)** — `doc.context(q)` lets the auto
  path fire.
- **C. neighbors=2** — `doc.context(q, neighbors=2, include_heading=True)`
  manual override; more aggressive.
- **D. neighbors=3** — manual override; tests diminishing returns.

Metric: % of queries where the body marker appears in the assembled
context. Plus avg context size in words and p50 latency.

Run:  bench/.venv/bin/python bench/code_neighbors_default.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]
BUDGET = 400

# (query, target file path relative to REPO, body marker that lives INSIDE
#  the implementation — not in the function signature or docstring)
# Markers picked by reading each function's body and choosing a phrase that:
#   1. Is in the actual code body, not the doc comment or signature.
#   2. Wouldn't match if only the def-line chunk surfaced.
# (query, target file, body-only marker). Markers verified via:
#   awk '/pub fn <fname>/,/^}/' <file> | grep -F "<marker>"
# All markers appear in the function body (not signature/docstring) of the
# function the query is meant to surface.
QUERIES = [
    (
        "how are two ranked result lists combined by reciprocal rank fusion",
        "crates/redhop/src/retrieval/fusion.rs",
        "contribution = 1.0 / (k",
    ),
    (
        "weighted sum fusion of multiple result lists with per list weights",
        "crates/redhop/src/retrieval/fusion.rs",
        "lists.len()",
    ),
    (
        "decide a chunk is code so it is retrieved lexically not embedded",
        "crates/redhop/src/retrieval/local_rerank.rs",
        'metadata.get("kind")',
    ),
    (
        "BM25 retrieves across text source heading fields",
        "crates/redhop/src/retrieval/bm25.rs",
        "QueryParser::for_index",
    ),
    (
        "auto-attach the section heading chunk for a prose hit",
        "crates/redhop/src/document/mod.rs",
        "build_context_expanded",
    ),
    (
        "snowball stemmer english analyzer pipeline tokenize",
        "crates/redhop/src/analyzer.rs",
        "Algorithm::English",
    ),
    (
        "context strategy reasoning preserving second hop rescue jaccard",
        "crates/redhop/src/context/mod.rs",
        "link_strength",
    ),
    (
        "raw analyzer minimal pipeline tokenize lowercase ASCII fold",
        "crates/redhop/src/analyzer.rs",
        "RawAnalyzer",
    ),
    (
        "stripper boilerplate removal token level word boundary",
        "crates/redhop/src/rewrite.rs",
        "Stripper",
    ),
    (
        "vocabulary apply rewrites query side enrich chunk side",
        "crates/redhop/src/rewrite.rs",
        "Vocabulary",
    ),
]


# ── Arms ───────────────────────────────────────────────────────────────────


def arm_neighbors(path: str, query: str, neighbors: int, use_auto: bool) -> tuple[str, float]:
    """Return (ctx_text, latency_ms). use_auto=True invokes the default
    auto path; otherwise bypass via manual path with explicit neighbors."""
    t0 = time.perf_counter()
    try:
        doc = redhop.Document.from_file(path, token_budget=BUDGET, candidate_k=3)
        if use_auto:
            ctx = doc.context(query)
        else:
            # Manual path: neighbors=N explicit, include_heading=True bypasses
            # the auto branch but is a no-op on code (no heading metadata).
            ctx = doc.context(query, neighbors=neighbors, include_heading=True)
    except Exception as e:  # noqa: BLE001
        print(f"  error: {e}", file=sys.stderr)
        return "", 0.0
    ms = (time.perf_counter() - t0) * 1000
    return ctx.text(), ms


def eval_arm(label: str, neighbors: int, use_auto: bool):
    hits = 0
    ctx_words_total = 0
    latencies = []
    for q, path, marker in QUERIES:
        ctx_text, ms = arm_neighbors(str(REPO / path), q, neighbors, use_auto)
        latencies.append(ms)
        ctx_words_total += len(ctx_text.split())
        if marker in ctx_text:
            hits += 1
    n = len(QUERIES)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    return hits, n, ctx_words_total / n, p50


def main() -> None:
    print()
    print("=" * 78)
    print("  code_neighbors_default probe")
    print("  Does ±1 neighbor auto-pull actually surface implementation bodies?")
    print("=" * 78)
    print()
    print(f"  RedHop Rust source (n={len(QUERIES)} queries)")
    print(f"  Each query's marker is a body-only phrase (not in def/docstring).")
    print(f"  Marker present in ctx ⇒ user got the implementation, not just the signature.")

    global BUDGET
    arms = [
        ("A. neighbors=0 (no expansion)", 0, False),
        ("B. neighbors=1 (current default)", 1, True),
        ("C. neighbors=2", 2, False),
        ("D. neighbors=3", 3, False),
    ]

    for budget in (128, 400, 1000, 4000):
        BUDGET = budget
        print()
        print(f"  budget = {budget} tok")
        print(f"  {'arm':<40} {'body marker hits':>16} {'avg ctx words':>14} {'p50 ms':>8}")
        print("  " + "-" * 80)
        results = {}
        for label, n, auto in arms:
            h, total, w, ms = eval_arm(label, n, auto)
            results[label] = (h, total, w, ms)
            print(f"  {label:<40} {h:>5}/{total:<5}            {w:>14.1f} {ms:>7.1f}")

        baseline = results["A. neighbors=0 (no expansion)"]
        default = results["B. neighbors=1 (current default)"]
        print(f"  Δ default − baseline: "
              f"{default[0]-baseline[0]:+d}/{default[1]} marker hits, "
              f"{default[2]-baseline[2]:+.1f} ctx words, "
              f"{default[3]-baseline[3]:+.1f} ms")
    print()
    print("Reading the result:")
    print("  • Default has more hits than baseline → neighbors=1 is doing real work")
    print("  • Hits ~equal, ctx larger → default just inflates; flip to 0")
    print("  • Neighbors=2/3 same as default → 1 is enough; don't increase")
    print("  • Hits keep rising with N → default may be under-allocated")
    print()


if __name__ == "__main__":
    main()
