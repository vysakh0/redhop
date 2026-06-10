"""RedHop — reasoning-preserving context optimization for RAG systems.

A **reasoning-preserving context runtime for document reasoning**. You have
documents and need reasoning; you should not have to wire up retrievers, vector
DBs, or query engines. It is *not* a retriever, vector DB, agent framework, or
workflow engine. The core takes text (bring your own parser); the optional
``redhop[files]`` tier reads PDF/DOCX/PPTX/XLSX for you.

High-level surface — reason over a document:

    import redhop

    doc = redhop.Document.from_text(text)          # chunked + indexed internally
    ctx = doc.context("Why did the proposed method fail?")
    response = llm.generate(ctx.text())            # any provider; no lock-in
    print(ctx.report)                              # what was retrieved/pruned, and why

Or load straight from disk (a file, or a whole folder — no vector DB to run):

    doc = redhop.Document.from_file("contract.pdf")    # pip install "redhop[files]"
    doc = redhop.Document.from_folder("./docs")        # every readable file, one index
    ctx = doc.context("what's our deprecation policy?")
    for c in ctx.citations:                            # cite where it came from
        print(c["source"], c["page"], c["heading"], c["line"])

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
from collections.abc import Sequence
from typing import Any

from ._redhop import (
    Aspect,
    BuiltContext,
    Chunk,
    ContextReport,
    CritiqueReport,
    Document,
    EvalReport,
    EvalSummary,
    Judge,
    QuerySetReport,
    RewriteRecord,
    Stripper,
    Vocabulary,
    __version__,
    analyze_query_set,
    critique,
    evaluate,
    grounding_score,
    link_strength,
    summarize,
    summarize_diagnoses,
)
from ._redhop import (
    analyze_context as _analyze_context,
)
from ._redhop import (
    build_context as _build_context,
)
from ._redhop import (
    context_economics as _context_economics,
)
from ._redhop import (
    filter_context as _filter_context,
)



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
        query,
        list(retrieved_chunks),
        strategy,
        token_budget,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        redundancy_max_cosine,
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
        query,
        list(retrieved_chunks),
        strategy,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        redundancy_max_cosine,
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
        query,
        list(retrieved_chunks),
        strategy,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
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
        _context_economics(
            query, list(retrieved_chunks), distractor_min_grounding, link_min_jaccard
        )
    )


def report_to_dict(report: ContextReport) -> dict[str, Any]:
    """The full :class:`ContextReport` as a plain dict."""
    return json.loads(report.json())


def context_with_timeout(
    doc: Document,
    query: str,
    *,
    timeout_ms: int,
    budget: int | None = None,
    neighbors: int = 0,
    include_heading: bool = False,
) -> BuiltContext:
    """Like ``doc.context(query, ...)`` but raise :class:`TimeoutError` if
    the call doesn't return within ``timeout_ms`` milliseconds.

    Intended for **agent integrations** that need to bail on stale queries
    (user typed more, prior context is no longer needed). The retrieval
    itself can't be interrupted mid-flight — Tantivy and ONNX have no
    cooperative cancellation — but the watchdog thread returns control to
    the caller on schedule. The background thread continues until it
    finishes; its result is discarded.

    Example:
        try:
            ctx = redhop.context_with_timeout(doc, q, timeout_ms=5000)
        except TimeoutError:
            # serve a fallback, prompt the user, etc.
            ...

    Raises:
        TimeoutError: if the call doesn't complete within ``timeout_ms``.
        ValueError: if ``timeout_ms <= 0``.
    """
    if timeout_ms <= 0:
        raise ValueError(f"timeout_ms must be positive; got {timeout_ms}")
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FutTimeout

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="redhop-ctx") as ex:
        fut = ex.submit(
            doc.context,
            query,
            budget=budget,
            neighbors=neighbors,
            include_heading=include_heading,
        )
        try:
            return fut.result(timeout=timeout_ms / 1000.0)
        except _FutTimeout:
            # The native call is still running in the worker thread; we just
            # stop waiting. The thread pool's __exit__ will block on it
            # finishing — which we DON'T want for "fire and forget" timeout
            # semantics. Cancel the future (Python won't actually interrupt
            # the running thread, but it prevents new work being scheduled).
            fut.cancel()
            raise TimeoutError(
                f"redhop context() exceeded {timeout_ms}ms; the underlying "
                "retrieval can't be interrupted but the watchdog returned. "
                "The background work will finish and its result be discarded."
            ) from None


__all__ = [
    "Aspect",
    "BuiltContext",
    "Chunk",
    "ContextReport",
    "CritiqueReport",
    "Document",
    "EvalReport",
    "EvalSummary",
    "Judge",
    "QuerySetReport",
    "RewriteRecord",
    "Stripper",
    "Vocabulary",
    "__version__",
    "analyze_context",
    "analyze_query_set",
    "build_context",
    "context_economics",
    "context_with_timeout",
    "evaluate",
    "filter_context",
    "grounding_score",
    "critique",
    "link_strength",
    "report_to_dict",
    "summarize",
    "summarize_diagnoses",
]
