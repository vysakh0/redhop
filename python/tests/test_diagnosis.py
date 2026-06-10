"""Smoke tests for ctx.report.diagnosis (the self-service retrieval diagnostic).

Design: docs/design/REPORT_DIAGNOSIS.md
"""

import redhop


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


def test_vocab_mismatch_query_lands_h2_with_evidence():
    doc = redhop.Document.from_chunks(_refund_chunks())
    ctx = doc.context("How long do I have to cancel and get my money back?")
    d = ctx.report.diagnosis

    assert d["corpus_stats_available"] is True
    assert "cancel" in d["zero_match_terms"]
    assert "money" in d["zero_match_terms"]

    hint_codes = [h["code"] for h in d["hints"]]
    assert "vocab_mismatch" in hint_codes, f"unexpected hints: {hint_codes}"

    h2 = next(h for h in d["hints"] if h["code"] == "vocab_mismatch")
    assert h2["evidence"].endswith("MULTIHOP_HYBRID.md")
    assert "—" not in h2["message"], "em dash leaked into hint"


def test_healthy_query_produces_no_hints():
    doc = redhop.Document.from_chunks(_refund_chunks())
    ctx = doc.context("refund policy thirty days")
    d = ctx.report.diagnosis

    assert d["corpus_stats_available"] is True
    assert d["query_terms"], "query_terms should not be empty"
    assert d["hints"] == [], f"healthy query produced hints: {d['hints']}"


def test_diagnosis_dict_shape_is_stable():
    doc = redhop.Document.from_chunks(_refund_chunks())
    ctx = doc.context("refund")
    d = ctx.report.diagnosis

    expected_keys = {
        "query_terms",
        "corpus_stats_available",
        "zero_match_terms",
        "term_stats",
        "terms_unmatched_in_candidates",
        "n_candidates",
        "score_spread",
        "empty_context",
        "hints",
    }
    assert set(d.keys()) == expected_keys, f"missing keys: {expected_keys - set(d.keys())}"
    assert isinstance(d["n_candidates"], int)
    # score_spread is f32 or None depending on the candidate pool.
    assert d["score_spread"] is None or isinstance(d["score_spread"], float)
