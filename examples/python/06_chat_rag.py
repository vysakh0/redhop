"""06 · Chat RAG with chronology preserved — `preserve_order=True`.

Real-world scenario:
    A customer-support agent's chat session has been going for an hour
    and has 30+ turns. Rather than summarizing or compacting the
    history (lossy), the team retrieves the few past turns relevant to
    the user's *current* question and pulls those into the LLM prompt.
    But causality breaks if the retrieved turns are presented in
    relevance order — "after the refund came in" reads strangely if
    it's shown before "ordered the laptop." They want the same
    relevance-driven selection but with **chronological emission**.

    That's exactly what `ContextConfig::preserve_order = True` does
    (see docs/findings/ + crates/examples/examples/chat_rag.rs for
    the Rust worked example). Selection is identical between the two
    modes; only the final emission order differs.

What this demonstrates:
    - `Document.from_chunks(chunks, preserve_order=True)` — selection
      stays relevance-driven, emission becomes chronological.
    - The contrast between the two modes on the same chat history —
      the same chunks come back, in different order.
    - That `chunk_index` metadata is stamped automatically on user-
      supplied chunks by `from_chunks` so the chronology key works
      without you having to do anything.

Run:
    python examples/python/06_chat_rag.py
"""

import redhop

# A 12-turn synthetic chat history. Each turn is one chunk. The
# turn-XX prefix is the chronology signal.
CHAT_HISTORY = [
    ("turn-00", "Hi, I have a question about my order."),
    ("turn-01", "I ordered a laptop last Tuesday."),
    ("turn-02", "It was the new MacBook Air, 15-inch."),
    ("turn-03", "Shipping confirmation came in yesterday — said tomorrow."),
    ("turn-04", "Actually I'd like to cancel and get my money back."),
    ("turn-05", "Sure — what is your refund policy on a shipped order?"),
    ("turn-06", "We offer a thirty-day refund window from the delivery date."),
    ("turn-07", "So I just send it back after it arrives?"),
    ("turn-08", "Yes — print the return label from your Orders page and drop it off."),
    ("turn-09", "Does the refund come right away?"),
    ("turn-10", "We refund within five business days of receiving the return."),
    ("turn-11", "Got it, thanks for your help!"),
]


def build_doc(preserve_order: bool) -> redhop.Document:
    chunks = [
        redhop.Chunk(text, source="chat", id=tid, metadata={"heading": tid})
        for tid, text in CHAT_HISTORY
    ]
    return redhop.Document.from_chunks(chunks, preserve_order=preserve_order)


def show_arm(label: str, preserve_order: bool, query: str) -> None:
    doc = build_doc(preserve_order=preserve_order)
    ctx = doc.context(query)
    print(f"─── {label} ──────────────────────────────────────")
    print("  selected turns, in emission order:")
    for c in ctx.citations:
        print(f"    {c['heading']}: {c['text']}")
    print()


def main() -> None:
    # A new user question that needs the refund subplot from the chat
    # history. The relevant past turns are scattered (turn-03 to
    # turn-10) across the conversation.
    query = "remind me — when does the refund actually come back?"
    print(f"Current user question: {query!r}\n")

    # Arm A: default (preserve_order=False). RedHop selects by
    # relevance and emits in the strategy's order — typically
    # relevance-first. Great for one-shot QA, wrong for chat.
    show_arm("Arm A · default (relevance-emitted)", preserve_order=False, query=query)

    # Arm B: preserve_order=True. Same selection, sorted back into
    # source-document order so the LLM reads the turns in the order
    # they happened.
    show_arm("Arm B · preserve_order=True (chronological)", preserve_order=True, query=query)

    print("Both arms select the same turns by relevance; only the")
    print("emission order differs. For the LLM, that ordering controls")
    print("whether causality reads correctly — `refund` after `return`")
    print("after `ordered`. See docs/findings/CHAT_RAG.md (when shipped)")
    print("and crates/examples/examples/chat_rag.rs for the Rust mirror.")


if __name__ == "__main__":
    main()
