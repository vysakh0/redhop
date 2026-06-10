"""Workload-audit + observability-export tests.

Design: docs/design/WORKLOAD_AUDIT.md
"""

import redhop
from redhop.otel import report_to_attributes


def _refund_chunks():
    return [
        redhop.Chunk(
            "Refund Policy. Refunds are available within thirty days of purchase.",
            id="a",
            source="policy.md",
        ),
        redhop.Chunk(
            "Termination for convenience. Either party may terminate this agreement.",
            id="b",
            source="policy.md",
        ),
        redhop.Chunk(
            "Governing Law. This agreement is governed by the laws of California.",
            id="c",
            source="policy.md",
        ),
    ]


def _mixed_queries():
    return (
        # Vocab-mismatch failures (no surface overlap with the corpus).
        [
            "How do I cancel and get my money back?",
            "When can I quit this contract?",
            "Way to receive cash refund?",
        ]
        * 6
        # Healthy queries (terms exist in the corpus).
        + [
            "refund policy",
            "termination for convenience",
            "governing law",
        ]
        * 3
    )


def test_byo_workload_resolves_to_vocab_mismatch():
    """End-to-end BYO loop on a vocab-skewed workload."""
    doc = redhop.Document.from_chunks(_refund_chunks())
    queries = _mixed_queries()
    assert len(queries) >= 20, "test setup must clear SUMMARY_MIN_QUERIES"

    reports = [doc.context(q).report for q in queries]
    summary = redhop.summarize_diagnoses(reports)

    assert summary.n == len(queries)
    assert summary.corpus_stats_coverage == 1.0, "Document path = Layer 2 = full coverage"
    assert summary.focus["code"] == "vocab_mismatch", summary.focus
    assert summary.focus["evidence"].endswith("MULTIHOP_HYBRID.md")
    assert "—" not in summary.focus["message"], "no em dashes in focus prose"

    # Rendered string is unstable but must contain the key sections.
    rendered = str(summary)
    assert "RedHop Workload Audit" in rendered
    assert "Focus" in rendered


def test_layer1_loop_via_analyze_context_has_no_corpus_stats():
    """The BYO funnel-door path: caller-supplied chunks via analyze_context.

    Without a Document, corpus stats can't be computed (Layer 1 only),
    so corpus_stats_coverage is 0.0. The summary still resolves; it
    just can't fire `vocab_mismatch` (that hint needs Layer 2).
    """
    chunks = [
        redhop.Chunk(
            "Refund policy. Refunds are available within thirty days.",
            id="a",
            source="external",
        ),
        redhop.Chunk(
            "Termination for convenience. Either party may terminate.",
            id="b",
            source="external",
        ),
        redhop.Chunk(
            "Governing law. California law applies.",
            id="c",
            source="external",
        ),
    ]
    reports = []
    for q in _mixed_queries():
        report = redhop.analyze_context(q, chunks)
        reports.append(report)

    summary = redhop.summarize_diagnoses(reports)
    assert summary.n == len(reports)
    assert summary.corpus_stats_coverage == 0.0
    # Should resolve without error to one of the known FocusCodes.
    assert summary.focus["code"] in {
        "sample_too_small",
        "healthy",
        "vocab_mismatch",
        "templated_queries",
        "underdetermined_queries",
        "weak_retrieval",
    }


def test_summarize_below_min_queries_recommends_nothing():
    doc = redhop.Document.from_chunks(_refund_chunks())
    reports = [doc.context("refund").report for _ in range(5)]
    summary = redhop.summarize_diagnoses(reports)
    assert summary.focus["code"] == "sample_too_small"
    assert summary.focus["evidence"] == ""


def test_otel_attributes_are_all_legal_types():
    """report_to_attributes must emit only OTel-legal scalar/list types."""
    doc = redhop.Document.from_chunks(_refund_chunks())
    ctx = doc.context("how do I cancel")
    attrs = report_to_attributes(ctx.report)

    # Every key starts with the namespace.
    for key in attrs:
        assert key.startswith("redhop.")

    # Every value is bool / int / float / str / list[str].
    for key, value in attrs.items():
        if isinstance(value, list):
            assert all(isinstance(x, str) for x in value), f"{key} has non-str element"
        else:
            assert isinstance(value, (bool, int, float, str)), (
                f"{key} = {value!r} ({type(value).__name__}) is not OTel-legal"
            )


def test_otel_skips_score_spread_when_none():
    """Optional fields are omitted (not emitted as None / null)."""
    doc = redhop.Document.from_chunks(_refund_chunks())
    # A query that produces zero candidates has no score_spread.
    ctx = doc.context("quokka platypus axolotl")
    assert ctx.report.diagnosis["score_spread"] is None
    attrs = report_to_attributes(ctx.report)
    assert "redhop.diagnosis.score_spread" not in attrs


def test_otel_caps_zero_match_terms_list():
    """Pathological queries can't blow up the span attributes."""
    doc = redhop.Document.from_chunks(_refund_chunks())
    # A very long query whose terms all miss the corpus.
    long_q = " ".join(f"missingterm{i}" for i in range(30))
    attrs = report_to_attributes(doc.context(long_q).report)
    zero = attrs["redhop.diagnosis.zero_match_terms"]
    assert len(zero) <= 16, f"zero_match_terms not capped, len={len(zero)}"
