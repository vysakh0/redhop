"""04 · Chunk-side enrich — `Vocabulary.enrich(chunk_text)` at ingest.

Real-world scenario:
    A platform engineering team maintains a runbook keyed by short
    error codes (`ERR_4012`, `EVT_CHRGBCK`, `DB_5001`). When alerts
    fire, on-call engineers search the runbook in natural language
    ("payment declined", "checkout broken", "database timeout") —
    almost never by the code itself. The runbook entries are short and
    coded; the natural-language queries share no surface words with
    them. That's exactly the regime where chunk-side enrich is
    *predicted* to help: append each code's plain-language meaning to
    its chunk at ingest, so the natural-language query has matchable
    surface area to land on.

⚠ Honest framing (read before applying to your corpus):
    Enrich is shipped as a primitive on **mechanism reasoning with
    asymmetric measured evidence**:
      - Measured negative: CUAD prose chunks regressed −2.0pt
        (docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md).
      - Measured positive: none on RedHop's eval rigs yet.
      - This example shows the mechanism on short opaque coded
        units (the regime where it's *predicted* to help) — but it
        is a *synthetic demo with a hand-crafted dictionary*, not a
        benchmark. Whether it lifts retention on your runbook depends
        on your specific corpus and your dictionary.
    **Always A/B with `redhop.evaluate(...)` against your gold set
    before adopting in production.** See `05_evaluate_ab.py` for how.

What this demonstrates:
    - `redhop.Vocabulary({...})` and `vocab.enrich(chunk_text)` — the
      chunk-side mirror of query-side `vocab.apply(query)`.
    - The audit `RewriteRecord` returned per enrichment (so you can
      log what was matched / appended at ingest time).
    - The `value ∝ shortness × opacity × dictionary` regime
      hypothesis (see docs/findings/VOCABULARY_ENRICH.md).

Run:
    python examples/python/04_chunk_enrich.py
"""

import redhop

# A toy runbook — short coded titles + minimal descriptions. The
# titles are what you'd find in your alerting system; the descriptions
# are what an engineer would write down in the runbook.
RUNBOOK_ENTRIES = [
    {
        "code": "ERR_4012",
        "title": "ERR_4012: PAYMENT_GATEWAY_DECLINED",
        "body": "Stripe returned a 4012. Check the customer's card. Common causes: insufficient funds, expired card, blocked transaction. Retry strategy: exponential backoff with a max of 3 attempts.",
    },
    {
        "code": "ERR_5001",
        "title": "ERR_5001: DB_CONNECTION_TIMEOUT",
        "body": "The Postgres pool exhausted. Check `pg_stat_activity` for long-running queries. Restart the worker if connections aren't returning to the pool.",
    },
    {
        "code": "EVT_CHRGBCK",
        "title": "EVT_CHRGBCK: chargeback notification",
        "body": "Stripe sent a chargeback webhook. Flag the order, freeze the customer's account pending review. Respond to Stripe within 7 days with evidence.",
    },
    {
        "code": "ERR_6201",
        "title": "ERR_6201: SHIPPING_LABEL_INVALID",
        "body": "ShipStation rejected the label. Check the customer's address validity. Re-print the label after the address is corrected.",
    },
    {
        "code": "ERR_7301",
        "title": "ERR_7301: EMAIL_DELIVERY_FAILED",
        "body": "SendGrid bounced. Check the recipient's domain status. Most common cause: customer mistyped their email at signup.",
    },
]

# Workload-specific decoder dictionary — *the user supplies this*. In a
# real runbook system this would be auto-generated from your alerting
# tags, your incident database, or a hand-curated glossary. The
# library ships `Vocabulary`; this dict is your knowledge of the
# domain.
#
# Note on dictionary content: each key gets a small set of TERM-
# SPECIFIC synonyms. **Do not** add generic words like "error",
# "system", "alert" — those are workload-pervasive and re-create the
# CUAD_PRF_NULL low-IDF dilution failure mode (which CUAD_ENRICH_
# DEFINITIONS_NULL just measured on the chunk side).
ERROR_CODE_VOCAB = {
    "ERR_4012": ["payment", "card", "charge", "stripe declined"],
    "ERR_5001": ["database", "postgres", "timeout", "connection pool"],
    "EVT_CHRGBCK": ["chargeback", "dispute", "refund request"],
    "ERR_6201": ["shipping", "label", "address", "delivery"],
    "ERR_7301": ["email", "bounce", "deliverability"],
}


def main() -> None:
    vocab = redhop.Vocabulary(ERROR_CODE_VOCAB)
    print(f"Compiled vocabulary with {len(vocab)} classes\n")

    # ── Step 1: Enrich each chunk at ingest time ─────────────────────
    print("─── Step 1 · Enrich chunks at ingest ─────────────")
    chunks: list[redhop.Chunk] = []
    for entry in RUNBOOK_ENTRIES:
        chunk_text = f"{entry['title']}\n{entry['body']}"
        # `vocab.enrich(text)` returns `(enriched_text, record)`. The
        # record is the audit trail — what was matched, what was
        # added — so you can log it at ingest time and see exactly
        # which chunks got which appended synonyms.
        enriched_text, record = vocab.enrich(chunk_text)
        if record.matched:
            print(
                f"  {entry['code']:>14} ← matched={record.matched} "
                f"added={record.added}"
            )
        chunks.append(
            redhop.Chunk(
                enriched_text,
                source=f"runbook/{entry['code']}.md",
                id=entry["code"],
                metadata={"heading": entry["title"]},
            )
        )
    print()

    # ── Step 2: Build the document and run a natural-language query ──
    doc = redhop.Document.from_chunks(chunks)
    query = "customer's card got declined at checkout, what do we do?"
    print(f"─── Step 2 · Query (natural language) ────────────")
    print(f"  {query!r}\n")

    ctx = doc.context(query)
    print(f"─── Top hit ──────────────────────────────────────")
    top = ctx.citations[0]
    print(f"  source : {top['source']}")
    print(f"  heading: {top['heading']}")
    print(f"  excerpt: {top['text'][:100].replace(chr(10), ' ')}…")
    print()

    # Note that the query shares no surface forms with `ERR_4012` — the
    # match landed via the enriched tokens "payment", "card",
    # "charge". Without enrichment, BM25 would have nothing to score
    # the ERR_4012 chunk on. (You can verify that by building a
    # second Document without the enrich step and comparing top hits.)
    print(
        "Mechanism: the query has no overlap with the bare error code"
        " `ERR_4012` —"
    )
    print(
        "the match landed via the appended `payment`/`card`/`charge`"
        " tokens that enrich"
    )
    print("attached at ingest. On your real runbook, A/B with")
    print(
        "`redhop.evaluate(...)` against a gold set (see"
        " 05_evaluate_ab.py) before"
    )
    print("committing to this in production.")


if __name__ == "__main__":
    main()
