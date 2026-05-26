#!/usr/bin/env python3
"""basic_rag — where RedHop fits in a RAG stack.

    retrieval  →  redhop.build_context  →  generation

Runs fully offline (simulated retriever, no LLM). Set OPENAI_API_KEY to also
see a real generation.

    python examples/basic_rag.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sample import (  # noqa: E402
    DISTRACTOR_MIN_GROUNDING,
    GOLD_ANSWER,
    LINK_MIN_JACCARD,
    QUERY,
    RETRIEVED,
)

import redhop  # noqa: E402


def fake_retriever(query: str):
    return RETRIEVED  # stand-in for your vector DB / BM25


def main() -> None:
    chunks = fake_retriever(QUERY)

    ctx = redhop.build_context(
        query=QUERY,
        retrieved_chunks=chunks,
        strategy="reasoning_preserving",
        token_budget=12000,
        distractor_min_grounding=DISTRACTOR_MIN_GROUNDING,
        link_min_jaccard=LINK_MIN_JACCARD,
    )

    print("=" * 70)
    print("RedHop sits between retrieval and generation.")
    print("=" * 70)
    print(f"\nQuery: {QUERY}\n")
    print(ctx.report)
    print("\n── Assembled context (what the LLM sees) ──\n")
    print(ctx.text())
    print(f"\n(reference answer: {GOLD_ANSWER})")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError:
            print("\n[pip install openai to run a real generation]")
            return
        client = OpenAI()
        prompt = (
            "Answer the question using ONLY the context. Be concise.\n\n"
            f"Context:\n{ctx.text()}\n\nQuestion: {QUERY}\n\nAnswer:"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        print(f"\n── LLM answer ──\n{resp.choices[0].message.content.strip()}")
    else:
        print("\n[set OPENAI_API_KEY to also run a real generation]")


if __name__ == "__main__":
    main()
