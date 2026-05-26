#!/usr/bin/env python3
"""compare_strategies — every strategy side-by-side on the same retrieval set.

The clearest view of what RedHop does: aggressive filtering drops the
reasoning-critical second hop; reasoning_preserving keeps it while pruning
distractors.

    python examples/compare_strategies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sample import (  # noqa: E402
    DISTRACTOR_MIN_GROUNDING,
    GOLD_ANSWER,
    LINK_MIN_JACCARD,
    QUERY,
    RETRIEVED,
)

import redhop  # noqa: E402

STRATEGIES = ["raw_topk", "distractor_filtered", "max_density", "reasoning_preserving"]


def main() -> None:
    print(f"Query: {QUERY}")
    print(f"Retrieved: {len(RETRIEVED)} chunks  (reference answer: {GOLD_ANSWER})\n")

    contexts = {}
    header = ("strategy", "chunks", "tokens", "distr", "rescued", "density")
    widths = (22, 8, 7, 6, 8, 8)
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("─" * (sum(widths) + 2 * len(widths)))

    for strat in STRATEGIES:
        ctx = redhop.build_context(
            query=QUERY,
            retrieved_chunks=RETRIEVED,
            strategy=strat,
            token_budget=12000,
            distractor_min_grounding=DISTRACTOR_MIN_GROUNDING,
            link_min_jaccard=LINK_MIN_JACCARD,
        )
        contexts[strat] = ctx
        r = ctx.report
        row = (
            strat,
            f"{r.n_input_chunks}→{r.n_selected}",
            str(r.total_tokens),
            f"{r.distractor_ratio:.2f}",
            str(r.second_hop_rescue_count),
            f"{r.evidence_density:.2f}",
        )
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))

    print("\n* distr = TRUE distractor ratio; rescued second hops are reasoning")
    print("  evidence and are excluded (note reasoning_preserving's low distr + rescued≥1).")
    print("\nDid the context keep the second hop (the '%s' fact)?" % GOLD_ANSWER)
    for strat in STRATEGIES:
        kept = GOLD_ANSWER in contexts[strat].text()
        print(f"  {strat:<22} {'✓ kept' if kept else '✗ DROPPED'}")

    print("\nReasoning-preserving context optimization — not a retriever, vector DB, or agent.")


if __name__ == "__main__":
    main()
