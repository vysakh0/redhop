#!/usr/bin/env python3
"""Deliverable D — strategy comparison playground.

Runs the same retrieved set through every strategy and prints a side-by-side
table plus the actual assembled context for each — the clearest single view
of what RedHop does and why `reasoning_preserving` is the safe default.

    python examples/python/strategy_playground.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redhop  # noqa: E402
from sample_corpus import (  # noqa: E402
    QUERY,
    RETRIEVED,
    GOLD_ANSWER,
    DISTRACTOR_MIN_GROUNDING,
    LINK_MIN_JACCARD,
)

STRATEGIES = ["raw_topk", "distractor_filtered", "max_density", "reasoning_preserving"]


def main() -> None:
    print(f"Query: {QUERY}")
    print(f"Retrieved: {len(RETRIEVED)} chunks  (reference answer: {GOLD_ANSWER})\n")

    rows = []
    contexts = {}
    for strat in STRATEGIES:
        ctx = redhop.build_context(
            query=QUERY,
            retrieved_chunks=RETRIEVED,
            token_budget=12000,
            strategy=strat,
            distractor_min_grounding=DISTRACTOR_MIN_GROUNDING,
            link_min_jaccard=LINK_MIN_JACCARD,
        )
        r = ctx.report
        rows.append(
            (
                strat,
                r.n_selected,
                r.total_tokens,
                f"{r.data['economics']['distractor_ratio']:.2f}",
                r.second_hop_rescue_count,
                f"{r.evidence_density:.2f}",
            )
        )
        contexts[strat] = ctx

    # Side-by-side table.
    header = ("strategy", "chunks", "tokens", "distr_ratio", "rescued", "density")
    widths = [22, 7, 7, 12, 8, 8]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("─" * len(line))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    print("\n* reasoning_preserving's distr_ratio counts the rescued second hop —")
    print("  it is low-relevance-to-query by nature, but reasoning-critical.")

    # Which kept the reasoning-critical second hop ("British")?
    print("\nDid the context keep the second hop (the 'British' nationality fact)?")
    for strat in STRATEGIES:
        kept = "British" in contexts[strat].text
        mark = "✓ kept" if kept else "✗ DROPPED"
        print(f"  {strat:<22} {mark}")

    # Show the actual contexts.
    print("\n" + "=" * 70)
    print("Assembled contexts (what each strategy would send to the LLM)")
    print("=" * 70)
    for strat in STRATEGIES:
        print(f"\n── {strat} ──")
        print(contexts[strat].text)

    print("\nTakeaway: relevance-only strategies can drop the low-relevance")
    print("second hop the answer depends on; reasoning_preserving keeps it while")
    print("still pruning distractors. RedHop is a reasoning-preserving context")
    print("optimization layer — not a retriever, vector DB, or agent framework.")


if __name__ == "__main__":
    main()
