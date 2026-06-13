"""07 · Retrieval tiers — lexical / hybrid / semantic on the same query.

Real-world scenario:
    A B2C support team's FAQ uses the company's polite phrasings
    ("refund", "return") but customers ask in colloquial English
    ("send back", "money back"). The same five-line FAQ corpus, hit
    with three different retrieval tiers, shows where each one fails
    and where each one succeeds — the trade-off documented in
    docs/findings/SEMANTIC_MISMATCH.md.

What this demonstrates:
    - The three `retrieval=` tiers: `"lexical"` (BM25, default,
      no model), `"hybrid"` (BM25 candidate pool + dense rerank,
      ~80MB model), `"semantic"` (global exact-cosine dense, ~80MB
      model).
    - That for a *synonym-mismatch* query (query and gold share zero
      surface tokens), lexical and hybrid can both miss the right
      chunk because hybrid only reranks within BM25's pool — if BM25
      didn't surface the right chunk, hybrid can't recover it.
    - Why `"semantic"` exists: bounded synonym-heavy corpora where
      global dense scoring catches what lexical pruning would lose.
    - When to climb the ladder vs stay on the default: see
      docs/CHOOSING_A_CONFIG.md.

First-run note:
    `retrieval="hybrid"` and `retrieval="semantic"` need an embedding
    model. The first call to either downloads `bge-small` (~80MB) to
    your local model cache; subsequent runs are fast. The lexical
    tier needs nothing.

Run:
    python examples/python/07_retrieval_tiers.py
"""

import time

import redhop

# A small B2C support FAQ. The deliberate stylistic choice: the FAQ
# uses formal vocabulary ("refund", "return", "warranty"); customer
# queries tend to be colloquial. That mismatch is what each tier
# handles differently.
SUPPORT_FAQ = """
Q: When will my package arrive?
A: Standard shipping takes 3-5 business days from when your order leaves our warehouse.

Q: How do I get my money back if I'm not satisfied?
A: We offer a full refund within 30 days of delivery. Return the item using the prepaid label.

Q: What's the warranty?
A: Our products have a one-year manufacturer warranty against defects.

Q: Can I cancel a subscription?
A: You can cancel anytime from Settings, no fee.

Q: Do you ship internationally?
A: Yes, we ship to 50 countries. Express international is 5-7 days.
"""

# A query that uses "send back" / "do not want" — neither phrase
# appears in the right-answer FAQ ("refund", "return"). Pure
# synonym-mismatch.
QUERY = "how do I send back something I do not want?"


def try_tier(label: str, **options: object) -> None:
    """Build a Document with the given options, run QUERY, print the top hit."""
    t0 = time.time()
    doc = redhop.Document.from_text(SUPPORT_FAQ, options=redhop.DocumentOptions(chunk_size=30))  # type: ignore[arg-type]
    ctx = doc.context(QUERY)
    elapsed = time.time() - t0
    top = ctx.citations[0]["text"][:80] if ctx.citations else "(none)"
    print(f"  {label:10}  build+query: {elapsed:>5.2f}s")
    print(f"               top hit  : {top!r}")
    print()


def main() -> None:
    print(f"Query: {QUERY!r}")
    print(
        f'Gold (the right answer): "How do I get my money back …" / '
        f'"We offer a full refund within 30 days …"\n'
    )

    print("─── Arm A · retrieval='lexical' (BM25, default, no model) ─")
    # No model needed. Will likely miss because BM25 has no surface
    # overlap between "send back" / "do not want" and "refund" /
    # "return".
    try_tier("lexical", retrieval="lexical")

    print("─── Arm B · retrieval='hybrid' (BM25 pool + dense rerank) ─")
    # `hybrid` runs BM25 first to build a candidate pool, then a small
    # embedding model reranks within that pool. If BM25's pool didn't
    # include the right chunk, hybrid can't recover it. First run
    # downloads bge-small (~80MB).
    try_tier("hybrid", retrieval="hybrid", model="bge-small")

    print("─── Arm C · retrieval='semantic' (global exact-cosine dense) ─")
    # `semantic` scores every chunk with the embedding model — no
    # BM25 pruning step. Catches synonym-mismatch cases that hybrid
    # can't. Costs more per query (every chunk encoded); only
    # practical on bounded corpora.
    try_tier("semantic", retrieval="semantic", model="bge-small")

    # Honest framing:
    print("─── How to read this ─────────────────────────────")
    print("On this tiny 5-chunk corpus, BM25's candidate pool happens")
    print("to fit all 5 chunks, so `hybrid` finds the right answer too.")
    print("On a *real* synonym-heavy corpus (HR FAQs, support tickets")
    print("translated from internal phrasing, multilingual content),")
    print("BM25's top-K will often *exclude* the synonym-mismatch")
    print("answer entirely — and then hybrid can't recover it because")
    print("it only reranks within BM25's pool. That's the regime where")
    print("`semantic` (global, no pruning) earns its keep, at the cost")
    print("of embedding every chunk per query (only practical on small")
    print("to medium corpora).")
    print()
    print("Don't read this as 'always use semantic.' For most document")
    print("QA — code, runbooks, contracts, financial filings — the")
    print("question and answer DO share surface words, and lexical")
    print("wins on latency. Climb the ladder only when measured.")
    print("Decision tree: docs/CHOOSING_A_CONFIG.md.")
    print("Mechanism + measurement: docs/findings/SEMANTIC_MISMATCH.md.")


if __name__ == "__main__":
    main()
