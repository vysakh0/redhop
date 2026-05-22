#!/usr/bin/env python3
"""Deliverable A — where RedHop fits in a RAG stack.

    retrieval  →  RedHop.build_context  →  generation

RedHop sits between your retriever and your LLM. You give it the retrieved
chunks and a token budget; it returns the prompt context to generate from,
having removed distractors and preserved reasoning-critical evidence.

Runs fully offline (a simulated retriever + no LLM call). Set OPENAI_API_KEY
to also see a real generation (optional).

    python examples/python/basic_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `redhop` importable
import redhop  # noqa: E402
from sample_corpus import (  # noqa: E402
    QUERY,
    RETRIEVED,
    GOLD_ANSWER,
    DISTRACTOR_MIN_GROUNDING,
    LINK_MIN_JACCARD,
)


def fake_retriever(query: str):
    """Stand-in for your vector DB / BM25 retriever."""
    return RETRIEVED


def main() -> None:
    query = QUERY

    # 1. Retrieve (your existing stack).
    chunks = fake_retriever(query)

    # 2. RedHop: optimize the context.
    ctx = redhop.build_context(
        query=query,
        retrieved_chunks=chunks,
        token_budget=12000,
        strategy="reasoning_preserving",
        distractor_min_grounding=DISTRACTOR_MIN_GROUNDING,
        link_min_jaccard=LINK_MIN_JACCARD,
    )

    # 3. Generate (your existing LLM).
    print("=" * 70)
    print("RedHop sits between retrieval and generation.")
    print("=" * 70)
    print(f"\nQuery: {query}\n")
    print(ctx.report)  # the Context Optimization Report

    print("\n── Assembled context (this is what the LLM sees) ──\n")
    print(ctx.text)

    print(f"\n(reference answer: {GOLD_ANSWER})")
    _maybe_generate(query, ctx.text)


def _maybe_generate(query: str, context: str) -> None:
    """Optional real generation if an OpenAI key is present."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[set OPENAI_API_KEY to also run a real generation]")
        return
    try:
        from openai import OpenAI
    except ImportError:
        print("\n[pip install openai to run a real generation]")
        return
    client = OpenAI()
    prompt = (
        "Answer the question using ONLY the context. Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    print(f"\n── LLM answer ──\n{resp.choices[0].message.content.strip()}")


if __name__ == "__main__":
    main()
