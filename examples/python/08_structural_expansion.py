"""08 · Structural expansion — `neighbors=N` and `include_heading=True`.

Real-world scenario:
    A SaaS company's internal handbook is heavily structured: each
    policy section has a heading + multiple paragraphs. When an
    employee asks a question, the BM25 top hit lands on the specific
    *paragraph* that answers the question — but for an LLM to write a
    grounded answer, it usually wants the *surrounding context*: the
    section heading (so it knows what topic it's in), plus the
    paragraphs immediately before and after (so it can quote across
    them when needed).

    RedHop's `Document.context(query, neighbors=N, include_heading=True)`
    handles both. Selection is still relevance-driven; the chunks just
    get padded with adjacent structural context, within the token
    budget.

What this demonstrates:
    - `doc.context(query, neighbors=1)` — pads each selected chunk
      with its 1 left/right neighbor in source order. `neighbors=2`
      doubles the context window per hit, etc.
    - `doc.context(query, include_heading=True)` — pulls in the
      opening chunk of each hit's section (the heading chunk).
    - `ctx.report.n_expanded` — how many extra chunks the structural
      expansion added beyond raw retrieval selection.
    - The contrast: with vs without — same query, same top hit, but
      the assembled context is meaningfully bigger and more grounded
      with expansion enabled.

Run:
    python examples/python/08_structural_expansion.py
"""

import redhop

# A short internal handbook excerpt. The `# heading` markdown markers
# get picked up by RedHop's chunker as section headings, which
# `include_heading=True` then reaches for.
HANDBOOK = """
# PTO (Paid Time Off)

Full-time employees accrue 1.5 days of PTO per month, totaling 18 days per year.

PTO carries over up to a maximum of 30 days at the end of the calendar year. Beyond that, unused PTO is forfeited.

To request PTO, submit a request through Workday at least two weeks in advance. Manager approval is required.

# Sick Leave

Sick leave is separate from PTO. Employees may take up to 10 paid sick days per year for personal illness or family caregiving.

Sick days do not carry over and do not count against your PTO balance.

# Parental Leave

New parents are eligible for 16 weeks of paid parental leave following the birth or adoption of a child.

Leave can be taken continuously or split into two blocks of at least four weeks each, within the first 12 months.
"""


def show_arm(label: str, query: str, **opts: object) -> None:
    """Build a fresh Document (so the chunker stamps headings) and run
    one query with the given expansion options.

    `candidate_k=2` constrains retrieval to the 2 top-scoring chunks so
    the *expansion* contrast is visible — otherwise on this small
    corpus the budget swallows everything and there's nothing to
    expand."""
    doc = redhop.Document.from_text(HANDBOOK, options=redhop.DocumentOptions(source="handbook.md", chunk_size=20, candidate_k=2))
    ctx = doc.context(query, **opts)  # type: ignore[arg-type]
    print(f"─── {label} ─────────────────────────")
    print(f"  n_selected     : {ctx.report.n_selected}")
    print(f"  n_expanded     : {ctx.report.n_expanded}")
    print(f"  total_tokens   : {ctx.report.total_tokens}")
    print("  assembled context:")
    for line in ctx.text().split("\n"):
        print(f"    {line}")
    print()


def main() -> None:
    query = "how many PTO days do I get?"
    print(f"Query: {query!r}\n")

    # Arm A: no expansion. Just the top-scoring chunk.
    show_arm("Arm A · plain context (no expansion)", query)

    # Arm B: include section heading. Pulls in the "# PTO" line so the
    # LLM knows what topic it's in.
    show_arm(
        "Arm B · include_heading=True", query, include_heading=True
    )

    # Arm C: include adjacent chunks (1 each side). Doesn't pull the
    # heading, but covers context around the hit.
    show_arm("Arm C · neighbors=1", query, neighbors=1)

    # Arm D: both. The fully-grounded option for structured docs.
    show_arm(
        "Arm D · neighbors=1 + include_heading=True (recommended for handbooks)",
        query,
        neighbors=1,
        include_heading=True,
    )

    print("─── When to use each ─────────────────────────────")
    print("- include_heading=True : structured docs (handbooks,")
    print("  contracts, runbooks) where the topic label matters for")
    print("  grounding. Lightweight — adds one chunk per hit.")
    print("- neighbors=1          : when the answer often spans")
    print("  adjacent chunks (a fact stated, then qualified).")
    print("- both                 : the safe default for structured")
    print("  document QA. Selection is still relevance-driven; expansion")
    print("  only adds chunks within the token budget.")
    print("- neither              : code search, transcripts,")
    print("  high-density technical content where the hit IS the answer.")


if __name__ == "__main__":
    main()
