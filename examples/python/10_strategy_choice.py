"""10 · Assembly strategies — `auto`, `raw_topk`, `reasoning_preserving`.

Real-world scenario:
    A research team has multi-hop questions over Wikipedia-style
    content ("Who invented the safety lamp, and what's their
    nationality?"). The retrieval surfaces two relevant chunks: one
    naming the inventor ("Davy invented the lamp"), one carrying the
    second-hop fact ("Davy was British"). A naive relevance-only
    filter would keep the high-scoring inventor chunk and drop the
    second-hop "Davy was British" chunk as a low-grounding distractor
    — and the LLM downstream would never see the bridge fact.

    RedHop's `reasoning_preserving` strategy keeps both: it rescues
    low-grounding chunks that are linked to high-grounding ones via
    term-set Jaccard overlap (the "bridge" between hops). The `auto`
    strategy is a size-gated wrapper that picks `reasoning_preserving`
    only when the input is large enough to warrant pruning; on small
    contexts it passes through (`raw_topk`) because pruning small
    clean contexts is wash-to-harmful (docs/findings/CONTEXT_DILUTION.md).

What this demonstrates:
    - `strategy="auto"` (default) — size-gated, picks
      `raw_topk` under the gate, `reasoning_preserving` over.
    - `strategy="raw_topk"` — pass-through, no filtering.
    - `strategy="reasoning_preserving"` — second-hop rescue. Reads
      `second_hop_rescue_count` off the report to confirm it fired.
    - `strategy="distractor_filtered"` — relevance-only filter; the
      *naive* baseline that drops the second-hop.
    - The `token_budget` and `candidate_k` knobs in their natural
      role: budget caps assembled tokens, candidate_k caps the
      retrieval pool.
    - Full strategy list: also `redundancy_pruned` and `max_density`
      (less common — see docs/findings/REASONING_PRESERVATION.md).

Run:
    python examples/python/10_strategy_choice.py
"""

import redhop

# A small multi-hop test corpus. The bridge entity is "Humphry Davy."
CHUNKS = [
    # Hop 1: the question's discriminator + the bridge entity.
    redhop.Chunk(
        "The miners' safety lamp was invented by Humphry Davy in 1815.",
        id="hop1",
    ),
    # Hop 2: low query-grounding (no "lamp" / "safety"), but linked to
    # hop1 via "Humphry Davy" — the bridge fact.
    redhop.Chunk(
        "Humphry Davy was a British chemist, born in Penzance, Cornwall, England.",
        id="hop2",
    ),
    # A distractor: high content but no overlap with the query or the
    # bridge.
    redhop.Chunk(
        "Photosynthesis converts sunlight into glucose and oxygen in plants.",
        id="d1",
    ),
]

QUERY = "what nationality was the inventor of the miners' safety lamp"


def show_arm(label: str, strategy: str, **opts: object) -> None:
    ctx = redhop.build_context(QUERY, CHUNKS, strategy=strategy, **opts)  # type: ignore[arg-type]
    print(f"─── {label} ──────────────────────────")
    print(f"  strategy           : {ctx.report.strategy}")
    print(f"  auto_decision      : {ctx.report.auto_decision}")
    print(f"  selected / input   : {ctx.report.n_selected} / {ctx.report.n_input_chunks}")
    print(f"  second-hop rescues : {ctx.report.second_hop_rescue_count}")
    bridge_kept = "British" in ctx.text()
    discr_kept = "safety lamp" in ctx.text()
    print(f"  bridge fact kept?  : {'yes ✓' if bridge_kept else 'no ✗'}")
    print(f"  discriminator kept?: {'yes ✓' if discr_kept else 'no ✗'}")
    print()


def main() -> None:
    print(f"Query: {QUERY!r}\n")
    print(
        "(The gold answer is 'British' — the bridge fact in hop2,"
        " which has low query-grounding.)\n"
    )

    # Default (auto) — size-gated. For this tiny corpus, auto picks
    # passthrough; bridge is kept.
    show_arm("Arm A · strategy='auto' (default)", strategy="auto")

    # Explicit pass-through. Same outcome as auto on small contexts.
    show_arm("Arm B · strategy='raw_topk'", strategy="raw_topk")

    # The naive baseline: relevance-only distractor filter. With a
    # tighter grounding threshold, it would drop the bridge fact as
    # too low-grounding — the failure mode reasoning_preserving fixes.
    show_arm(
        "Arm C · strategy='distractor_filtered' (naive — drops bridge)",
        strategy="distractor_filtered",
        distractor_min_grounding=0.30,
    )

    # The reasoning-preserving rescue. Even at the high distractor
    # threshold, this strategy keeps hop2 because it's linked to hop1
    # via the bridge entity ("Humphry Davy"). `second_hop_rescues`
    # on the report confirms the rescue fired.
    show_arm(
        "Arm D · strategy='reasoning_preserving' (the rescue)",
        strategy="reasoning_preserving",
        distractor_min_grounding=0.30,
    )

    print("─── How to read this ─────────────────────────────")
    print("- `auto` is the default; it picks `raw_topk` under the size")
    print("  gate (small clean contexts) and `reasoning_preserving`")
    print("  over (where pruning recovers accuracy via dilution control).")
    print("  The gate threshold is the `auto_passthrough_max_tokens`")
    print("  knob (default 1500).")
    print("- `raw_topk` is what you want when chunks are short and")
    print("  high-density — code, schemas, error codes.")
    print("- `reasoning_preserving` is what you want for multi-hop QA")
    print("  where the *bridge* between hops can sit below the naive")
    print("  grounding threshold (docs/findings/SECOND_HOP_TAX.md).")
    print("- `distractor_filtered` is the relevance-only baseline that")
    print("  reasoning_preserving improves on; mainly useful as a")
    print("  comparison arm.")
    print()
    print("Full strategy decision tree: docs/CHOOSING_A_CONFIG.md.")


if __name__ == "__main__":
    main()
