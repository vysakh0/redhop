"""02 · Structured corpus — `redhop.Chunk(...)` for content you already
   chunked elsewhere, with metadata that flows through to citations.

Real-world scenario:
    A SaaS company has a customer-support knowledge base: each FAQ pair
    is one row in a database (question, answer, category, last_updated,
    article_url). Support agents query it in natural language while
    chatting with a customer. They need:
      - Citations that point back to a specific article (`article_url`)
      - Metadata visible on the citation (e.g. `category`, `last_updated`)
        so the agent can see provenance at a glance.
    The 0.3.0 typed `redhop.Chunk` constructor + open `metadata` dict
    is what makes this clean — pre-0.3.0 the dict path couldn't carry
    arbitrary metadata onto citations.

What this demonstrates:
    - `redhop.Chunk(text, source=..., id=..., metadata={...})` — the
      typed constructor for hand-built chunks.
    - **source vs id**: `source` is *provenance* (what the citation
      displays); `id` is *identity* (stable handle for dedup / gold).
    - **Citations pick up known metadata keys**: `page` (int),
      `heading` (str), `line` (int) on `metadata={...}` flow through
      to `ctx.citations[i]`. Arbitrary metadata (your own keys like
      `category`, `last_updated`) is preserved on the chunk but is
      not yet surfaced through the citation getter — keep a parallel
      dict keyed by chunk id if you need them at display time.
    - `Document.from_chunks(chunks)` — no chunker re-split; what you
      pass in is what gets indexed, 1-to-1.

Run:
    python examples/python/02_structured_corpus.py
"""

import redhop

# Toy FAQ corpus — eight Q&A pairs across four categories. In production
# you'd pull these from your DB / CMS / CSV; the shape doesn't change.
FAQ_ROWS = [
    {
        "id": "faq-001",
        "category": "billing",
        "question": "When is my credit card charged?",
        "answer": "Your card is charged on the first day of each billing cycle. You can view upcoming charges under Settings → Billing.",
        "url": "https://help.acme.com/billing/charge-date",
        "last_updated": "2026-04-12",
    },
    {
        "id": "faq-002",
        "category": "billing",
        "question": "How do I request a refund?",
        "answer": "Refunds are available within 30 days of charge. Email finance@acme.com with your invoice number and reason. We process refunds within 5 business days.",
        "url": "https://help.acme.com/billing/refunds",
        "last_updated": "2026-05-03",
    },
    {
        "id": "faq-003",
        "category": "account",
        "question": "How do I change my email address?",
        "answer": "Settings → Account → Email. We send a confirmation link to the new address; click it within 24 hours to complete the change.",
        "url": "https://help.acme.com/account/email",
        "last_updated": "2026-03-21",
    },
    {
        "id": "faq-004",
        "category": "account",
        "question": "How do I delete my account?",
        "answer": "Settings → Account → Delete Account. We retain billing records for 7 years for tax compliance but anonymize all profile data immediately.",
        "url": "https://help.acme.com/account/delete",
        "last_updated": "2026-02-18",
    },
    {
        "id": "faq-005",
        "category": "shipping",
        "question": "When will my order arrive?",
        "answer": "Standard shipping is 3-5 business days. Express is 1-2 days. You'll get a tracking link by email once the package leaves our warehouse.",
        "url": "https://help.acme.com/shipping/delivery-time",
        "last_updated": "2026-05-30",
    },
    {
        "id": "faq-006",
        "category": "shipping",
        "question": "Can I change my shipping address after ordering?",
        "answer": "Yes, if the order hasn't shipped yet. Go to Orders → Edit. After shipment we cannot reroute — you'll need to contact the carrier directly.",
        "url": "https://help.acme.com/shipping/change-address",
        "last_updated": "2026-04-05",
    },
    {
        "id": "faq-007",
        "category": "returns",
        "question": "What is your return policy?",
        "answer": "Unworn items in original packaging may be returned within 30 days of delivery for a full refund. Print a prepaid label from Orders → Return.",
        "url": "https://help.acme.com/returns/policy",
        "last_updated": "2026-05-15",
    },
    {
        "id": "faq-008",
        "category": "returns",
        "question": "Do you cover return shipping?",
        "answer": "Yes — return shipping is free in the US for unworn items. International returns are paid by the customer.",
        "url": "https://help.acme.com/returns/shipping-costs",
        "last_updated": "2026-04-22",
    },
]


def build_chunks(rows: list[dict]) -> list[redhop.Chunk]:
    """Project DB rows to typed chunks.

    Decisions worth noting:
      - `text` combines question + answer so retrieval sees both. In
        production you might also include the question separately as
        a second chunk per FAQ — depends on your queries.
      - `source` is the article URL — that's what `ctx.citations[*].source`
        will display. The agent's UI can render it as a clickable link.
      - `id` is the FAQ row id — stable across runs; used for dedup
        and (if you eval) gold-chunk lookup.
      - `metadata` carries the rest: category and last_updated are
        useful for the agent to see; they're preserved through the
        pipeline and accessible from the report JSON.
    """
    return [
        redhop.Chunk(
            f"Q: {r['question']}\nA: {r['answer']}",
            source=r["url"],
            id=r["id"],
            metadata={
                "category": r["category"],
                "last_updated": r["last_updated"],
                # If you want a Markdown-style `heading` to show up on
                # the citation, set it here — citations machinery
                # picks up the conventional keys `page`, `heading`,
                # `line` if present.
                "heading": r["question"],
            },
        )
        for r in rows
    ]


def main() -> None:
    chunks = build_chunks(FAQ_ROWS)
    doc = redhop.Document.from_chunks(chunks)
    print(f"Indexed {len(doc)} FAQ entries.\n")

    # Build a parallel `by_id` dict so we can look up the original
    # row's full metadata when displaying citations. RedHop's citation
    # getter surfaces `source`, `page`, `heading`, `line` directly;
    # everything else stays on your side.
    by_id = {r["id"]: r for r in FAQ_ROWS}

    # A real customer query. BM25 matches "refund" + "deadline" against
    # the refunds FAQ; the result is one citation pointing at the
    # billing/refunds article URL.
    query = "what's the deadline for getting a refund?"
    print(f"Query: {query!r}\n")

    ctx = doc.context(query)

    print("─── Top hit ──────────────────────────────────────")
    cite = ctx.citations[0]
    # `source` and `heading` come from the citation getter directly.
    print(f"  source        : {cite['source']}")
    print(f"  heading       : {cite['heading']}")

    # `category` and `last_updated` aren't first-class citation fields,
    # but we attached them to the chunk's metadata. Look them up by id
    # from your parallel dict. The id RedHop assigned to the top chunk
    # is on the (private) BuiltContext internals; in production you'd
    # carry the chunk id in your own retrieval shim or use a single
    # source-of-truth lookup keyed by `source` (URL).
    matching_row = next((r for r in FAQ_ROWS if r["url"] == cite["source"]), None)
    if matching_row:
        print(f"  category      : {matching_row['category']}")
        print(f"  last_updated  : {matching_row['last_updated']}")
    print(f"  text (excerpt): {cite['text'][:80]}…")
    print()

    print("─── Decision Report ──────────────────────────────")
    print(f"  Final context tokens : {ctx.report.total_tokens}")
    print(
        f"  Decision             : {ctx.report.auto_decision} "
        f"(strategy={ctx.report.strategy})"
    )
    print(f"  Chunks selected      : {ctx.report.n_selected} of {ctx.report.n_input_chunks}")


if __name__ == "__main__":
    main()
