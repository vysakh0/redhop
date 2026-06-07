"""01 · Quickstart — load a document, ask a question, read the Decision Report.

Real-world scenario:
    A contract analyst has a Master Services Agreement (MSA) and needs to
    answer "what's the governing law?" before handing the snippet to an
    LLM for summarization. They want a citation back to the clause and
    a reason RedHop chose those chunks.

What this demonstrates:
    - The 3-call surface: `Document.from_text(...)`, `doc.context(query)`,
      `ctx.text() / ctx.citations / ctx.report`.
    - The Decision Report explaining what RedHop did and why.
    - That for a small document, the runtime *deliberately* leaves the
      context untouched (the "Auto → passthrough" decision).

Run:
    pip install redhop
    python examples/python/01_quickstart.py
"""

import redhop

# A short Master Services Agreement excerpt. In production this would be
# `redhop.Document.from_file("msa.pdf")` (requires `pip install "redhop[files]"`),
# but for a self-contained demo we paste the text inline.
MSA = """
SECTION 8. CONFIDENTIALITY

Each party shall keep confidential all non-public information disclosed by
the other party in connection with this Agreement. The receiving party
shall not use such information for any purpose other than performance of
this Agreement.

SECTION 9. GOVERNING LAW AND JURISDICTION

This Agreement shall be governed by and construed in accordance with the
laws of the State of New York, without regard to its conflict-of-laws
principles. The parties consent to the exclusive jurisdiction of the
state and federal courts located in New York County, New York.

SECTION 10. ENTIRE AGREEMENT

This Agreement constitutes the entire agreement between the parties and
supersedes all prior negotiations, representations, and agreements,
whether written or oral, with respect to its subject matter.

SECTION 11. NOTICES

Any notice required under this Agreement shall be in writing and
delivered to the address set forth on the signature page.
"""


def main() -> None:
    # 1. Load. `from_text` runs the default sentence chunker and indexes
    #    every chunk with BM25 — no model download, no vector DB.
    doc = redhop.Document.from_text(MSA, source="acme_msa.txt")
    print(f"Indexed {len(doc)} chunks from acme_msa.txt\n")

    # 2. Ask. RedHop retrieves, scores, and budgets the prompt all
    #    in-process. The `BuiltContext` carries the assembled text,
    #    citations back to source chunks, and the Decision Report.
    ctx = doc.context("what's the governing law?")

    # 3. Hand the assembled text to whatever LLM you use — RedHop has no
    #    LLM lock-in. For the demo we just print it.
    print("─── Prompt (ctx.text()) ──────────────────────────")
    print(ctx.text())
    print()

    # Citations: where did each kept chunk come from?
    print("─── Citations ────────────────────────────────────")
    for c in ctx.citations:
        print(f"  source={c['source']!r} text={c['text'][:80]!r}…")
    print()

    # The Decision Report explains the runtime's choice. For a small
    # document like this, the size gate fires and the context is passed
    # through untouched — pruning small contexts is wash-to-harmful per
    # docs/findings/CONTEXT_DILUTION.md.
    print("─── Decision Report ──────────────────────────────")
    print(f"  strategy          : {ctx.report.strategy}")
    print(f"  auto_decision     : {ctx.report.auto_decision}")
    print(f"  input_chunks      : {ctx.report.n_input_chunks}")
    print(f"  selected_chunks   : {ctx.report.n_selected}")
    print(f"  total_tokens      : {ctx.report.total_tokens}")
    print(f"  retained_evidence : {ctx.report.retained_evidence_ratio:.2f}")
    print()
    print("(For a human-readable version, just print(ctx.report))")


if __name__ == "__main__":
    main()
