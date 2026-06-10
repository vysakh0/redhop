"""Telemetry conventions for RedHop's Decision Report.

A dependency-free helper that flattens a ``ContextReport`` into a dict
of OpenTelemetry-legal span attributes (or Langfuse metadata, or any
key-value telemetry sink). RedHop imports no SDK; the user attaches
the returned dict to their own span.

Example (OpenTelemetry)::

    from opentelemetry import trace
    from redhop import Document
    from redhop.otel import report_to_attributes

    doc = Document.from_chunks(chunks)
    with trace.get_tracer(__name__).start_as_current_span("rag.query") as span:
        ctx = doc.context(query)
        span.set_attributes(report_to_attributes(ctx.report))

Example (Langfuse)::

    langfuse.trace(name="rag.query", metadata=report_to_attributes(ctx.report))

The full report is available via ``ctx.report.json()`` for span events
or log bodies. Attribute conventions are documented in
``docs/DIAGNOSE_YOUR_PIPELINE.md``.
"""

from __future__ import annotations

from typing import Any, Dict

_ZERO_MATCH_TERMS_CAP = 16


def report_to_attributes(report: Any, prefix: str = "redhop.") -> Dict[str, Any]:
    """Flatten a ``ContextReport`` into OTel-legal span attributes.

    All values are bool / int / float / str / list-of-str so they pass
    OpenTelemetry's attribute-value rules. Optional fields are omitted
    rather than emitted as ``None``: ``redhop.diagnosis.score_spread``
    is absent when the report's was ``None``.

    Hint *messages* are excluded on purpose (size/cardinality); the
    code list is enough to alert and aggregate on, and the evidence
    path is recoverable from the code. Full report bodies belong in
    ``report.json()`` events, not attributes.

    Parameters
    ----------
    report:
        A ``ContextReport`` from ``ctx.report``.
    prefix:
        Attribute-name prefix. Defaults to ``"redhop."``; override if
        the host SDK reserves the namespace.

    Returns
    -------
    dict
        Attribute-name to value, every value OTel-legal.
    """
    p = prefix
    out: Dict[str, Any] = {
        f"{p}strategy": report.strategy,
        f"{p}requested_strategy": report.requested_strategy,
        f"{p}auto_decision": report.auto_decision,
        f"{p}input_tokens": int(report.input_tokens),
        f"{p}total_tokens": int(report.total_tokens),
        f"{p}token_budget": int(report.token_budget),
        f"{p}n_input_chunks": int(report.n_input_chunks),
        f"{p}n_selected": int(report.n_selected),
        f"{p}retained_evidence_ratio": float(report.retained_evidence_ratio),
        f"{p}evidence_density": float(report.evidence_density),
        f"{p}estimated_waste_tokens": int(report.estimated_waste_tokens),
        f"{p}second_hop_rescues": int(report.second_hop_rescue_count),
        f"{p}low_confidence": bool(report.low_confidence_retrieval),
    }

    d = report.diagnosis
    out[f"{p}diagnosis.empty_context"] = bool(d["empty_context"])
    out[f"{p}diagnosis.n_candidates"] = int(d["n_candidates"])
    # Hint codes only (in fire order). Hint messages stay out: too
    # large, too high-cardinality. Evidence is recoverable from the code.
    out[f"{p}diagnosis.hints"] = [str(h["code"]) for h in d["hints"]]
    # Cap zero_match_terms so a pathological query can't blow up the
    # span. The order from the diagnosis dict is already first-occurrence
    # of the analyzed query terms; truncate from the head.
    zero = list(d["zero_match_terms"])[:_ZERO_MATCH_TERMS_CAP]
    out[f"{p}diagnosis.zero_match_terms"] = [str(t) for t in zero]
    # score_spread is Optional; omit the key when absent so dashboards
    # can detect "not present" cleanly.
    if d["score_spread"] is not None:
        out[f"{p}diagnosis.score_spread"] = float(d["score_spread"])
    return out


__all__ = ["report_to_attributes"]
