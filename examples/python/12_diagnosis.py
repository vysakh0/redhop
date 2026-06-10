"""12 · Diagnosis — when retrieval looks weak, the Decision Report tells you why.

Real-world scenario:
    A support team is wiring up Q&A over a policy doc. A user asks
    *"how long do I have to cancel and get my money back?"* and gets
    an empty answer. The doc uses *refund* and *termination*, not
    *cancel* and *money back*, so BM25 has nothing to match.

What this demonstrates:
    - `ctx.report.diagnosis` populated on every `context()` call.
    - Layer-2 facts: `query_terms`, `zero_match_terms`, `term_stats`
      computed against the corpus vocabulary.
    - The closed `hints` registry: one bounded hint per documented
      failure shape, each citing the finding that justifies it.
    - A healthy query produces zero hints. No spam on the happy path.

Run:
    pip install redhop
    python examples/python/12_diagnosis.py
"""

import redhop


def main() -> None:
    doc = redhop.Document.from_chunks(
        [
            redhop.Chunk(
                "Refund Policy. Refunds are available within thirty days of purchase.",
                id="a",
                source="policy.md",
            ),
            redhop.Chunk(
                "Termination for convenience. Either party may terminate this agreement.",
                id="b",
                source="policy.md",
            ),
            redhop.Chunk(
                "Governing Law. This agreement is governed by the laws of California.",
                id="c",
                source="policy.md",
            ),
        ]
    )

    # ── 1. A healthy query: facts populated, no hints ──────────────────
    healthy = doc.context("refund policy thirty days")
    d = healthy.report.diagnosis
    print("Healthy query:")
    print(f"  query_terms              = {d['query_terms']}")
    print(f"  corpus_stats_available   = {d['corpus_stats_available']}")
    print(f"  zero_match_terms         = {d['zero_match_terms']}")
    print(f"  hints                    = {d['hints']}")
    print()

    # ── 2. Vocabulary-mismatch query: H2 fires, evidence cited ────────
    paraphrase = doc.context("How long do I have to cancel and get my money back?")
    d = paraphrase.report.diagnosis
    print("Vocabulary-mismatch query:")
    print(f"  query_terms              = {d['query_terms']}")
    print(f"  zero_match_terms         = {d['zero_match_terms']}")
    print(f"  empty_context            = {d['empty_context']}")
    for hint in d["hints"]:
        print(f"  hint {hint['code']!r}")
        print(f"    evidence  : {hint['evidence']}")
        print(f"    message   : {hint['message']}")
    print()

    # ── 3. The rendered Decision Report shows the same data ──────────
    # (Format conventions match the rest of the report. No "Query
    # diagnosis" section appears for healthy calls; on bad calls it
    # appears below the existing Warnings block.)
    print("Rendered report (excerpt):")
    rendered = str(paraphrase.report)
    if "Query diagnosis" in rendered:
        section = rendered[rendered.index("Query diagnosis"):]
        print(section)


if __name__ == "__main__":
    main()
