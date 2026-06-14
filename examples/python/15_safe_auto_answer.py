"""15 · Safe auto-answers — when should a chatbot answer vs ask?

Real-world scenario:
    A US store's help bot answers FAQs. The expensive failure is a
    *confident wrong answer*, so the bot should auto-answer only when
    retrieval clearly matched, and otherwise ask a clarifying question
    (or hand off). RedHop does not ship a router or a threshold — it
    gives you the *signals* and a deterministic eval to measure the gate.
    You own the "if confident then answer, else ask" logic. This is the
    pattern from the safe-auto-answers guide.

What this demonstrates:
    - `ctx.report.low_confidence_retrieval` — the primary gate ("nothing
      relevant matched").
    - `evaluate(query, ctx).mean_grounding` — a no-gold confidence
      *strength* in [0,1] (how query-relevant the assembled context is).
      Confidence is a measured signal, not the model's self-report.
    - `evaluate(query, ctx, gold_chunks=[...])` to MEASURE the gate on a
      labeled set: auto-precision (correct among auto-answered) and
      unsafe-auto (auto-answered when we should have asked, target 0).
    - The headline: a good gate "gets cautious, not wrong" — it routes
      weak retrievals to clarify, keeping auto-precision high and
      unsafe-auto at 0.

    tau here is illustrative. In production you DERIVE it: sweep on a
    labeled dev set and pick the smallest tau hitting your precision
    target (e.g. 99%). See the guide.

Run:
    pip install redhop
    python examples/python/15_safe_auto_answer.py
"""

import redhop

FAQ = [
    ("faq-refund", "Refunds. Return any item within 30 days for a full refund, no questions asked."),
    ("faq-shipping", "Shipping. Standard shipping is free on orders over 35 dollars and arrives in 5 to 7 business days."),
    ("faq-hours", "Store hours. Our stores are open 9am to 9pm Monday through Saturday, and 10am to 6pm on Sunday."),
    ("faq-giftcard", "Gift cards. Gift cards never expire and can be used online or in any store."),
    ("faq-track", "Order tracking. Track your order from the Orders page using the tracking number in your confirmation email."),
]

# Labeled eval set: each query maps to the FAQ id that answers it, or None
# when there is no confident answer (the bot SHOULD ask, not guess).
LABELED = [
    ("how do I return something for a refund", "faq-refund"),
    ("when are you open on sunday", "faq-hours"),
    ("how do I track my package", "faq-track"),
    ("do gift cards expire", "faq-giftcard"),
    ("can you help me", None),             # too vague — should ask
    ("do you price match competitors", None),  # not in the KB — should ask
]

# Illustrative threshold. DERIVE this on a dev set in production (see guide):
# sweep tau and pick the smallest value that hits your auto-precision target.
TAU = 0.2


def main() -> None:
    doc = redhop.Document.from_chunks(
        [redhop.Chunk(text, id=fid, source="faq") for (fid, text) in FAQ]
    )
    print("Routing each query AUTO vs CLARIFY on redhop's confidence signals.")
    print(f"(AUTO only when retrieval is confident: not low_confidence AND grounding >= {TAU})\n")
    print(f"  {'query':<38} {'low_conf':>9} {'grounding':>9} {'route':>8}  outcome")

    auto_total = auto_correct = unsafe_auto = clarify_total = 0
    for query, gold in LABELED:
        ctx = doc.context(query)
        # One eval per query; mean_grounding is a self-eval populated with or
        # without gold. Pass gold (when we have it) to also check correctness.
        r = redhop.evaluate(query, ctx, gold_chunks=[gold]) if gold else redhop.evaluate(query, ctx)
        low = ctx.report.low_confidence_retrieval
        grounding = r.mean_grounding
        auto = (not low) and grounding >= TAU
        gold_present = bool(gold) and (r.context_recall or 0.0) >= 1.0

        if auto:
            auto_total += 1
            if gold and gold_present:
                auto_correct += 1
                outcome = "AUTO ✓ correct"
            elif gold:
                outcome = "AUTO ✗ WRONG (auto-answered, missed the gold)"
            else:
                unsafe_auto += 1
                outcome = "AUTO ☠ UNSAFE (should have asked)"
        else:
            clarify_total += 1
            outcome = "clarify (asks the user)"

        print(f"  {query[:38]:<38} {str(low):>9} {grounding:>9.2f} "
              f"{'AUTO' if auto else 'CLARIFY':>8}  {outcome}")

    auto_precision = auto_correct / auto_total if auto_total else 1.0
    n = len(LABELED)
    print("\n─── Scorecard ────────────────────────────────────")
    print(f"  auto-resolve rate   : {auto_total}/{n} answered without asking")
    print(f"  auto-precision ⭐    : {auto_precision:.3f}  (correct among auto-answered; aim >= 0.99)")
    print(f"  unsafe-auto ☠       : {unsafe_auto}      (auto-answered when it should have asked; target 0)")
    print(f"  clarify rate        : {clarify_total}/{n} routed to a question")
    print("\nThe gate degrades weak retrievals to clarify, so the bot")
    print("'gets cautious, not wrong'. DERIVE tau on your own dev set by")
    print("sweeping it to your precision target — see the safe-auto-answers guide.")


if __name__ == "__main__":
    main()
