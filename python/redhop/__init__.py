"""RedHop — reasoning-preserving context optimization for RAG systems.

RedHop sits **between retrieval and generation**. You hand it the chunks your
retriever returned and a token budget; it assembles the prompt context —
pruning distractors, preserving reasoning-critical "second-hop" evidence, and
reporting exactly what it did. It is *not* a retriever, vector DB, agent
framework, or workflow engine.

    import redhop

    ctx = redhop.build_context(
        query=query,
        retrieved_chunks=chunks,          # list of dicts/strings
        strategy="reasoning_preserving",  # the safe default
        token_budget=12000,
    )
    response = llm.generate(ctx.text())
    print(ctx.report)                     # Context Optimization Report

This package is a thin binding over the Rust `redhop-context` crate (built
with pyo3/maturin). Rust is the source of truth; no logic is duplicated here.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ._redhop import (
    BuiltContext,
    ContextReport,
    analyze_context as _analyze_context,
    build_context as _build_context,
    context_economics as _context_economics,
    filter_context as _filter_context,
    __version__,
)

Chunk = Mapping[str, Any] | str
"""A retrieved chunk: a string, or a dict with at least ``text`` (and optional
``id``, ``source``, ``token_count``, ``embedding``, ``score``)."""


def build_context(
    query: str,
    retrieved_chunks: Sequence[Chunk],
    strategy: str = "reasoning_preserving",
    token_budget: int = 8192,
    *,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
    redundancy_max_cosine: float = 0.92,
) -> BuiltContext:
    """Assemble a finite-attention context. Returns a :class:`BuiltContext`
    with ``.text()`` (the prompt string) and ``.report`` (telemetry)."""
    return _build_context(
        query, list(retrieved_chunks), strategy, token_budget,
        distractor_min_grounding, link_min_jaccard, redundancy_max_cosine,
    )


def filter_context(
    query: str,
    retrieved_chunks: Sequence[Chunk],
    strategy: str = "reasoning_preserving",
    *,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
    redundancy_max_cosine: float = 0.92,
) -> BuiltContext:
    """Filter junk without budget truncation ("clean it up, I'll manage the
    budget"). Returns a :class:`BuiltContext`."""
    return _filter_context(
        query, list(retrieved_chunks), strategy,
        distractor_min_grounding, link_min_jaccard, redundancy_max_cosine,
    )


def analyze_context(
    query: str,
    retrieved_chunks: Sequence[Chunk],
    *,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
) -> ContextReport:
    """Characterize a retrieved set **without** modifying it (pure diagnostics):
    distractor load, evidence density, and rescuable second-hop candidates."""
    return _analyze_context(
        query, list(retrieved_chunks), distractor_min_grounding, link_min_jaccard,
    )


def context_economics(
    query: str,
    retrieved_chunks: Sequence[Chunk],
    *,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
) -> dict[str, Any]:
    """Economics of a chunk set as-is (no filtering, no budget): evidence
    density, distractor ratio, redundancy, estimated wasted tokens."""
    return json.loads(
        _context_economics(query, list(retrieved_chunks), distractor_min_grounding, link_min_jaccard)
    )


def report_to_dict(report: ContextReport) -> dict[str, Any]:
    """The full :class:`ContextReport` as a plain dict."""
    return json.loads(report.json())


__all__ = [
    "build_context",
    "filter_context",
    "analyze_context",
    "context_economics",
    "report_to_dict",
    "BuiltContext",
    "ContextReport",
    "__version__",
]
