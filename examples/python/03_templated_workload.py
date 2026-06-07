"""03 · Templated workload — detect → strip → vocabulary → audit trail.

Real-world scenario:
    A legal-ops team uses a fixed query template across hundreds of
    contracts: each query is shaped like
        Highlight the parts (if any) of this contract related to "<X>"
        that should be reviewed by a lawyer. Details: <…>
    where only <X> varies. The boilerplate words dilute BM25's signal
    on the discriminating clause name, costing retention on the
    framework comparison (CUAD: 81% raw → 88% stripped → 90.7%
    stripped + clause-synonyms). RedHop's 0.3.0 surface ships three
    things they need:
      - `analyze_query_set(queries)` to *detect* the template.
      - `Stripper(boilerplate)` to drop the wrapper at retrieval time.
      - `Vocabulary({...})` to append clause-name synonyms.
    Both rewrites run inside `Document.context_with_rewrites(query,
    [stripper, vocab])` so the per-stage audit lands on
    `ctx.report.query_rewrites` and the chain stays observable.

What this demonstrates:
    - `analyze_query_set(...)` — flags whether a query set is
      templated and which words are doing the dilution.
    - `Stripper(boilerplate_terms)` — compiled, token-level boilerplate
      removal. The token-level matcher is the substring-safety guard
      (an "of" stripper does *not* erase the "of" inside "office").
    - `Vocabulary({"change of control": ["merger", ...], ...})` —
      compiled equivalence classes for query expansion.
    - `Document.context_with_rewrites(query, [stripper, vocab])` —
      runs the chain through retrieval; each stage's
      `RewriteRecord` (`stage`, `matched`, `added`, `removed`, before
      and after text) lands on `ctx.report.query_rewrites`.
    - Mechanism source: docs/findings/CUAD_RECALL_GAP.md +
      CUAD_CLAUSE_EXPANSION.md.

Run:
    python examples/python/03_templated_workload.py
"""

import redhop

# A tiny contract excerpt covering a few clause types so the demo can
# show retrieval finding the right one. In production this is your
# actual contract PDF/DOCX loaded via `Document.from_file(...)`.
CONTRACT = """
SECTION 7. CHANGE OF CONTROL

In the event of a Change of Control of either party, including any
merger, consolidation, or sale of substantially all assets, the
non-acquired party shall have the right to terminate this Agreement on
thirty days' written notice.

SECTION 8. NON-COMPETE

During the Term and for two years thereafter, the Distributor shall
not, directly or indirectly, engage in any business competitive with
the Company within the Territory.

SECTION 9. INDEMNIFICATION

Each party shall indemnify and hold harmless the other from any third-
party claims arising from the indemnifying party's gross negligence or
willful misconduct.

SECTION 10. CONFIDENTIALITY

Each party shall keep confidential all non-public information disclosed
by the other party in connection with this Agreement.
"""

# A small representative sample of the legal team's query set. These
# look templated by eye; we'll let the analyzer confirm it.
SAMPLE_QUERIES = [
    'Highlight the parts (if any) of this contract related to "Change of Control" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Indemnification" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Confidentiality" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Termination" that should be reviewed by a lawyer.',
]

# Workload-specific clause-name synonyms. The library deliberately does
# NOT ship a CUAD dict (or any other workload's) — `Vocabulary` is the
# mechanism; your dict is your workload knowledge.
CLAUSE_SYNONYMS = {
    "change of control": ["merger", "consolidation", "acquisition", "successor"],
    "non-compete": ["restraint", "compete", "competitive"],
    "indemnification": ["indemnify", "hold harmless", "third-party claims"],
    "confidentiality": ["confidential", "non-disclosure", "non-public"],
    "termination": ["terminate", "expire", "end"],
}


def main() -> None:
    # ── Step 1: Detect ────────────────────────────────────────────────
    print("─── Step 1 · Detect the template ─────────────────")
    report = redhop.analyze_query_set(SAMPLE_QUERIES)
    print(f"  is_templated            : {report.is_templated}")
    print(f"  template_word_share     : {report.template_word_share:.2f}")
    print(f"  estimated_dilution_cost : {report.estimated_dilution_cost}")
    print(f"  boilerplate_terms       : {report.boilerplate_terms}")
    print(f"  suggested_action        : {report.suggested_action}")
    print()
    if not report.is_templated:
        print("(Template not detected — for non-templated workloads skip the")
        print(" Stripper and use Document.context(query) directly.)")
        return

    # ── Step 2: Compile the rewrites ─────────────────────────────────
    # Compile once, reuse for every query. The token-level matcher
    # makes the analyzer pass once at construction time — chatbot
    # hot paths don't pay it per request.
    stripper = redhop.Stripper(report.boilerplate_terms)
    vocab = redhop.Vocabulary(CLAUSE_SYNONYMS)
    print("─── Step 2 · Compile the rewrites ────────────────")
    print(f"  Stripper: {len(stripper)} boilerplate forms")
    print(f"  Vocabulary: {len(vocab)} clause classes")
    print()

    # ── Step 3: Run a query through the chain ────────────────────────
    print("─── Step 3 · Run a query through the chain ───────")
    doc = redhop.Document.from_text(CONTRACT, source="msa.txt")
    query = SAMPLE_QUERIES[0]
    print(f"  raw query: {query!r}\n")

    ctx = doc.context_with_rewrites(query, [stripper, vocab])

    # The per-stage audit trail. Each `RewriteRecord` documents what
    # one rewrite stage did: input → output, what was matched, what
    # was added, what was removed. Decision-Report-grade
    # observability for every query transformation.
    print("  query_rewrites audit trail:")
    for rec in ctx.report.query_rewrites:
        print(f"    [{rec.stage}]")
        print(f"      from   : {rec.from_query!r}")
        print(f"      to     : {rec.to_query!r}")
        print(f"      matched: {rec.matched}")
        print(f"      added  : {rec.added}")
        print(f"      removed: {rec.removed}")
    print()

    print("  Top citation source : ", ctx.citations[0]["source"])
    print(
        "  Top citation text   : ",
        ctx.citations[0]["text"][:80].replace("\n", " "),
        "…",
    )
    print()
    print("  Decision: ", ctx.report.auto_decision, "/", ctx.report.strategy)


if __name__ == "__main__":
    main()
