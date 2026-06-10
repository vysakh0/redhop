"""13 · Workload audit — point RedHop's diagnostics at your existing pipeline.

Real-world scenario:
    A team already has a retrieval pipeline (LangChain BM25 over their
    contracts, in this sketch). They are not ready to migrate. They
    want to know, across their last few hundred production queries,
    *why* retrieval sometimes fails, and which single knob the data
    says to reach for first.

What this demonstrates:
    - The bring-your-own-retrieval (BYO) loop: caller-supplied chunks
      via `redhop.analyze_context(query, chunks)`. RedHop never owns
      the retriever; it observes what the retriever returned.
    - Workload-level aggregation via `redhop.summarize_diagnoses`:
      hint histogram, failure rates, top vocabulary gaps, and at most
      one focus recommendation citing the measured finding behind it.
    - Layer-1 facts (BYO, no corpus access) vs Layer-2 (full corpus
      diagnosis via `Document.from_chunks`). Two lines to upgrade.

Run:
    pip install redhop
    python examples/python/13_workload_audit.py

LangChain glue (not executed in CI — install `langchain-community` to
try this against your real retriever):

    # from langchain_community.retrievers import BM25Retriever
    # retriever = BM25Retriever.from_texts(your_corpus_texts)
    # def external_search(query):
    #     return [d.page_content for d in retriever.invoke(query)]
"""

import redhop


# Stand-in for "your existing retriever". In your real code this is
# LangChain / LlamaIndex / pgvector / whatever already runs. RedHop
# never wraps it; we only diagnose what it returned.
CORPUS = [
    "Refund Policy. Refunds are available within thirty days of purchase.",
    "Termination for convenience. Either party may terminate this agreement.",
    "Governing Law. This agreement is governed by the laws of California.",
    "Limitation of Liability. The cap is twelve months of fees.",
    "Confidentiality. Each party shall keep the other party's information confidential.",
]


def external_search(query: str, k: int = 3):
    """Toy keyword retriever standing in for your real one."""
    q_terms = set(query.lower().split())
    scored = []
    for text in CORPUS:
        score = sum(1 for w in text.lower().split() if w in q_terms)
        scored.append((score, text))
    scored.sort(reverse=True)
    return [text for _, text in scored[:k]]


# A real workload: 60% paraphrased questions (vocab mismatch), 40%
# direct-vocabulary queries (healthy). Tweak the mix to see different
# focus recommendations.
QUERIES = (
    [
        "how do I cancel and get my money back",
        "when can I quit this contract",
        "what is the cap on damages",
        "who keeps secrets",
    ]
    * 6
    + [
        "refund policy",
        "termination for convenience",
        "governing law",
        "limitation of liability cap",
    ]
    * 4
)


def main() -> None:
    # ── Layer 1: BYO retrieval, observe what it returned ──────────────
    layer1_reports = []
    for q in QUERIES:
        texts = external_search(q)
        chunks = [
            redhop.Chunk(t, id=str(i), source="external") for i, t in enumerate(texts)
        ]
        # analyze_context observes the candidate pool without
        # modifying it. corpus_stats_available is False on these
        # reports (RedHop has no Document to derive vocab from).
        layer1_reports.append(redhop.analyze_context(q, chunks))

    print("── Layer 1: observe what your retriever returned ──")
    layer1_summary = redhop.summarize_diagnoses(layer1_reports)
    print(layer1_summary)

    # ── Layer 2: also point RedHop at the same corpus, once ──────────
    # Two lines. RedHop indexes a copy in memory; your retrieval is
    # untouched.
    doc = redhop.Document.from_chunks(
        [redhop.Chunk(t, id=str(i), source="corpus") for i, t in enumerate(CORPUS)]
    )

    layer2_reports = [doc.context(q).report for q in QUERIES]
    print()
    print("── Layer 2: same queries against an in-memory corpus index ──")
    layer2_summary = redhop.summarize_diagnoses(layer2_reports)
    print(layer2_summary)

    # ── Step 3: ship it to your telemetry ─────────────────────────────
    # The rest of your trace machinery can hang attributes off any
    # report. Zero new dependencies.
    from redhop.otel import report_to_attributes

    attrs = report_to_attributes(layer2_reports[0])
    print("── OTel-legal attributes for the first report ──")
    for k, v in list(attrs.items())[:8]:
        print(f"  {k} = {v!r}")
    print("  ... and so on.")


if __name__ == "__main__":
    main()
