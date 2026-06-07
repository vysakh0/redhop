"""05 · Deterministic A/B with `redhop.evaluate(...)` — no LLM judge.

Real-world scenario:
    The legal-ops team from `03_templated_workload.py` wants to know
    whether adding clause-name synonyms (the `Vocabulary` step)
    actually lifts retrieval on *their* contracts, not on a published
    benchmark. They have a small gold set: for each query they
    labeled which chunk id(s) should appear in the assembled context.
    They compare two arms — baseline vs strip + vocab — and the
    `evaluate` API returns context_recall, context_precision, and a
    composite `overall` for each, all from the same primitives the
    runtime uses for its Decision Report. No LLM judge, no API key,
    no money spent, deterministic across runs.

What this demonstrates:
    - `redhop.evaluate(query, ctx, gold_chunks=[...])` returns context
      recall, precision, and composite `overall` against an explicit
      gold set.
    - How to plug it into an A/B that compares two retrieval
      configurations on the same query set.
    - "Refraction not independent measurement" — the eval and the
      Decision Report share primitives, so they cannot disagree
      (docs/findings/EVALUATE_API.md).

Run:
    python examples/python/05_evaluate_ab.py
"""

import redhop

# Same contract corpus as 03 — but here we pre-chunk it into
# section-sized chunks so we can label each chunk with a stable id
# and use those ids as the gold set for `evaluate`.
SECTIONS = [
    {
        "id": "sec-7",
        "heading": "Change of Control",
        "text": "SECTION 7. CHANGE OF CONTROL. In the event of a Change of Control of either party, including any merger, consolidation, or sale of substantially all assets, the non-acquired party shall have the right to terminate this Agreement on thirty days' written notice.",
    },
    {
        "id": "sec-8",
        "heading": "Non-Compete",
        "text": "SECTION 8. NON-COMPETE. During the Term and for two years thereafter, the Distributor shall not, directly or indirectly, engage in any business competitive with the Company within the Territory.",
    },
    {
        "id": "sec-9",
        "heading": "Indemnification",
        "text": "SECTION 9. INDEMNIFICATION. Each party shall indemnify and hold harmless the other from any third-party claims arising from the indemnifying party's gross negligence or willful misconduct.",
    },
    {
        "id": "sec-10",
        "heading": "Confidentiality",
        "text": "SECTION 10. CONFIDENTIALITY. Each party shall keep confidential all non-public information disclosed by the other party in connection with this Agreement.",
    },
    {
        "id": "sec-11",
        "heading": "Termination",
        "text": "SECTION 11. TERMINATION. Either party may terminate this Agreement upon thirty days' written notice in the event of a material breach by the other party.",
    },
    {
        "id": "sec-12",
        "heading": "Notices",
        "text": "SECTION 12. NOTICES. Any notice required under this Agreement shall be in writing and delivered to the address set forth on the signature page.",
    },
]

# Gold set: each templated query maps to the chunk id we expect the
# assembled context to contain. This is *your* labeled truth — you
# write it once per workload, and it stays the same across runs.
GOLD_QUERIES = [
    (
        'Highlight the parts (if any) of this contract related to "Change of Control" that should be reviewed by a lawyer.',
        ["sec-7"],
    ),
    (
        'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer.',
        ["sec-8"],
    ),
    (
        'Highlight the parts (if any) of this contract related to "Indemnification" that should be reviewed by a lawyer.',
        ["sec-9"],
    ),
    (
        'Highlight the parts (if any) of this contract related to "Confidentiality" that should be reviewed by a lawyer.',
        ["sec-10"],
    ),
    (
        'Highlight the parts (if any) of this contract related to "Termination" that should be reviewed by a lawyer.',
        ["sec-11"],
    ),
]

CLAUSE_SYNONYMS = {
    "change of control": ["merger", "consolidation", "acquisition"],
    "non-compete": ["restraint", "compete", "competitive"],
    "indemnification": ["indemnify", "hold harmless"],
    "confidentiality": ["confidential", "non-disclosure"],
    "termination": ["terminate", "expire", "end"],
}


def build_doc() -> redhop.Document:
    """Build the same Document for both arms — only the *query* differs
    between A and B. This isolates the retrieval-side effect."""
    return redhop.Document.from_chunks(
        [
            redhop.Chunk(s["text"], source="msa.txt", id=s["id"], metadata={"heading": s["heading"]})
            for s in SECTIONS
        ]
    )


def evaluate_arm(label: str, doc: redhop.Document, use_rewrites: bool) -> float:
    """Run every gold query through `doc` (optionally with the rewrite
    chain), score each with `redhop.evaluate`, return the mean
    `overall`."""
    boilerplate = [
        "highlight", "the", "parts", "if", "any", "of", "this", "contract",
        "related", "to", "that", "should", "be", "reviewed", "by", "a", "lawyer",
    ]
    stripper = redhop.Stripper(boilerplate)
    vocab = redhop.Vocabulary(CLAUSE_SYNONYMS)

    print(f"─── arm {label} ──────────────────────────────────")
    totals = {
        "context_recall": 0.0,
        "context_precision": 0.0,
        "overall": 0.0,
    }
    n = 0
    for query, gold_ids in GOLD_QUERIES:
        ctx = (
            doc.context_with_rewrites(query, [stripper, vocab])
            if use_rewrites
            else doc.context(query)
        )
        r = redhop.evaluate(query, ctx, gold_chunks=gold_ids)
        n += 1
        totals["context_recall"] += r.context_recall or 0.0
        totals["context_precision"] += r.context_precision or 0.0
        totals["overall"] += r.overall
        # Show one query per arm to make the comparison concrete.
        if n == 1:
            print(
                f"  example query  : {query[:60]}…"
            )
            print(
                f"  context_recall : {r.context_recall:.2f}  "
                f"context_precision : {r.context_precision:.2f}  "
                f"overall : {r.overall:.2f}"
            )
    means = {k: v / n for k, v in totals.items()}
    print(
        f"  mean over {n} queries: "
        f"recall={means['context_recall']:.2f}  "
        f"precision={means['context_precision']:.2f}  "
        f"overall={means['overall']:.2f}"
    )
    print()
    return means["overall"]


def main() -> None:
    doc = build_doc()
    print("Comparing two retrieval arms on the same gold set.\n")
    a = evaluate_arm("A · baseline (no rewrites)", doc, use_rewrites=False)
    b = evaluate_arm(
        "B · stripped + clause-name vocabulary", doc, use_rewrites=True
    )
    print("─── Verdict ──────────────────────────────────────")
    print(f"  ΔB−A on `overall`: {b - a:+.2f}")
    if b > a + 0.02:
        print(
            "  ✓ The rewrite chain lifted retrieval on this gold set."
        )
    elif b < a - 0.02:
        print(
            "  ✗ The rewrite chain regressed retrieval. Inspect the audit"
        )
        print("    trail (ctx.report.query_rewrites) — likely the vocab is")
        print("    appending workload-pervasive terms (the CUAD_PRF_NULL")
        print("    failure mode).")
    else:
        print("  ~ Within sample noise — re-run with a larger gold set.")


if __name__ == "__main__":
    main()
