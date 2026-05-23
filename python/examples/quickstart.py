#!/usr/bin/env python3
"""RedHop quickstart — reason over a document.

The hero path: you have a document and a question; RedHop chunks + indexes it
internally, retrieves the relevant pieces, and assembles a token-budgeted
context — explaining every decision. No retriever wiring, no vector DB, no LLM
needed to run this (bring your own generation at the end).

    python examples/quickstart.py
"""

import redhop

# A document. In practice this is your parsed text (PyMuPDF, Marker, …) — RedHop
# does not parse PDFs; it takes text and owns everything from chunking onward.
DOCUMENT = """
The miners' safety lamp was invented by Humphry Davy in 1815. It allowed miners
to work in the presence of flammable gases without igniting them.

Humphry Davy was born in Penzance, Cornwall, England, in 1778. He was a chemist
and inventor, and later served as President of the Royal Society.

Photosynthesis is the process by which green plants convert sunlight into
chemical energy. It is unrelated to mining safety.

The Davy lamp reduced explosions in coal mines across Britain in the early
nineteenth century, though it did not eliminate them entirely.
"""


def main() -> None:
    # 1. Build a document — chunking + indexing happen lazily, internally.
    doc = redhop.Document.from_text(DOCUMENT)
    print(f"document: {doc.n_chunks} chunks, {doc!r}\n")

    # 2. Ask for the reasoning context for a query. `strategy="auto"` is the
    #    default: pass through when the context is small, prune when it's large
    #    and diluted — and tell you which, and why.
    ctx = doc.context("What nationality was the inventor of the safety lamp?")

    # 3. The assembled prompt — drop straight into any LLM provider.
    #    e.g. openai.responses.create(model="gpt-4o-mini", input=ctx.text())
    print("Assembled context (feed this to your LLM):")
    print("-" * 60)
    print(ctx.text())
    print("-" * 60, "\n")

    # 4. The decision report — what RedHop did and why. Interpretability is a
    #    feature: a no-op is an explained, intentional choice.
    print(ctx.report)

    # 5. The same decision is available programmatically.
    print(f"\nauto_decision = {ctx.report.auto_decision!r}   "
          f"(requested={ctx.report.requested_strategy!r}, ran={ctx.report.strategy!r})")

    # 6. analyze() answers "should I prune?" WITHOUT modifying anything
    #    (the document carries the Auto policy from construction).
    diag = doc.analyze("safety lamp inventor")
    print(f"analyze() would: {diag.auto_decision}")


if __name__ == "__main__":
    main()
