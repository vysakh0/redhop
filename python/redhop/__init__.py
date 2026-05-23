"""RedHop — reasoning-preserving context optimization for RAG systems.

A **reasoning-aware context runtime for document reasoning**. You have
documents and need reasoning; you should not have to wire up retrievers, vector
DBs, or query engines. It is *not* a retriever, vector DB, agent framework, or
workflow engine — and it does not parse PDFs (bring your own text).

High-level surface — reason over a document:

    import redhop

    doc = redhop.Document.from_text(text)          # chunked + indexed internally
    ctx = doc.context("Why did the proposed method fail?")
    response = llm.generate(ctx.text())            # any provider; no lock-in
    print(ctx.report)                              # what was retrieved/pruned, and why

Low-level surface — you already have chunks (still first-class):

    ctx = redhop.build_context(
        query=query,
        retrieved_chunks=chunks,   # list of dicts/strings
        strategy="auto",           # size-gated: pass under headroom, prune under dilution
        token_budget=12000,
    )

This package is a thin binding over the Rust `redhop-context` /
`redhop-document` crates (built with pyo3/maturin). Rust is the source of
truth; no logic is duplicated here.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ._redhop import (
    BuiltContext,
    ContextReport,
    Document,
    analyze_context as _analyze_context,
    build_context as _build_context,
    context_economics as _context_economics,
    filter_context as _filter_context,
    grounding_score,
    link_strength,
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
    auto_passthrough_max_tokens: int = 1500,
    redundancy_max_cosine: float = 0.92,
) -> BuiltContext:
    """Assemble a finite-attention context. Returns a :class:`BuiltContext`
    with ``.text()`` (the prompt string) and ``.report`` (telemetry).

    ``strategy="auto"`` is the size-gated policy: pass the context through
    unchanged when it is small (pruning is wash-to-harmful under headroom), and
    prune only when the input exceeds ``auto_passthrough_max_tokens`` (the
    large-context dilution regime, where pruning recovers accuracy). See
    ``docs/findings/CONTEXT_DILUTION.md``."""
    return _build_context(
        query, list(retrieved_chunks), strategy, token_budget,
        distractor_min_grounding, link_min_jaccard,
        auto_passthrough_max_tokens, redundancy_max_cosine,
    )


def filter_context(
    query: str,
    retrieved_chunks: Sequence[Chunk],
    strategy: str = "reasoning_preserving",
    *,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
    auto_passthrough_max_tokens: int = 1500,
    redundancy_max_cosine: float = 0.92,
) -> BuiltContext:
    """Filter junk without budget truncation ("clean it up, I'll manage the
    budget"). Returns a :class:`BuiltContext`."""
    return _filter_context(
        query, list(retrieved_chunks), strategy,
        distractor_min_grounding, link_min_jaccard,
        auto_passthrough_max_tokens, redundancy_max_cosine,
    )


def analyze_context(
    query: str,
    retrieved_chunks: Sequence[Chunk],
    *,
    strategy: str | None = None,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
    auto_passthrough_max_tokens: int = 1500,
) -> ContextReport:
    """Characterize a retrieved set **without** modifying it (pure diagnostics):
    distractor load, evidence density, and rescuable second-hop candidates.

    Pass ``strategy="auto"`` to ask *"would the size gate prune this?"* — the
    returned ``report.strategy`` is the decision (``"raw_topk"`` = pass through,
    ``"reasoning_preserving"`` = prune) without modifying the context."""
    return _analyze_context(
        query, list(retrieved_chunks), strategy,
        distractor_min_grounding, link_min_jaccard, auto_passthrough_max_tokens,
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
    "Document",
    "build_context",
    "filter_context",
    "analyze_context",
    "context_economics",
    "grounding_score",
    "link_strength",
    "report_to_dict",
    "BuiltContext",
    "ContextReport",
    "__version__",
]
