"""Binding-surface tests for `redhop.evaluate` + `EvalReport`.

Mirrors the Rust unit tests on `redhop::evaluate` through the pyo3
boundary, so a dropped field on `EvalReport`, a wrong `gold_chunks`
kwarg shape, or a misrouted `EvalGold` variant at the FFI edge
surfaces here, not in user code.

The mechanism + the "refraction not independent measurement" design
choice are documented in docs/findings/EVALUATE_API.md.

Run with: pytest python/tests/test_evaluate.py -q
"""

import pytest

import redhop


def _chunks_for(text: str, chunk_id: str = "a"):
    """Helper: build a single-chunk retrieved list for build_context."""
    return [redhop.Chunk(text, id=chunk_id)]


# ─── evaluate without any gold ──────────────────────────────────────────────


def test_self_eval_no_gold_returns_self_eval_only():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    r = redhop.evaluate("refund window", ctx)
    assert isinstance(r, redhop.EvalReport)
    assert r.context_recall is None
    assert r.context_precision is None
    assert r.answer_token_recall is None
    # Self-eval populated.
    assert r.mean_grounding > 0.0
    assert 0.0 < r.overall <= 1.0
    assert isinstance(r.low_confidence, bool)


# ─── gold_chunks only ───────────────────────────────────────────────────────


def test_gold_chunks_perfect_recall():
    ctx = redhop.build_context(
        "refund window",
        [
            redhop.Chunk("the refund window is thirty days", id="hit1"),
            redhop.Chunk("refund policy details and timing", id="hit2"),
        ],
        strategy="raw_topk",
    )
    r = redhop.evaluate("refund window", ctx, gold_chunks=["hit1", "hit2"])
    assert r.context_recall == pytest.approx(1.0)
    assert r.context_precision == pytest.approx(1.0)
    assert r.answer_token_recall is None


def test_gold_chunks_partial_recall_precision_distinct():
    ctx = redhop.build_context(
        "policy",
        [
            redhop.Chunk("policy section about refunds", id="hit"),
            redhop.Chunk("totally unrelated cooking recipe", id="noise_a"),
            redhop.Chunk("more cooking instructions", id="noise_b"),
        ],
        strategy="raw_topk",
    )
    r = redhop.evaluate("policy", ctx, gold_chunks=["hit", "missing"])
    assert r.context_recall == pytest.approx(0.5)         # 1 of 2 gold present
    assert r.context_precision == pytest.approx(1.0 / 3.0)  # 1 of 3 selected is gold


def test_empty_gold_chunks_perfect_recall_but_no_precision():
    """`gold_chunks=[]` is "no chunks needed" — vacuously perfect recall,
    precision undefined → None (NOT 0.0)."""
    ctx = redhop.build_context(
        "q", _chunks_for("some text"), strategy="raw_topk"
    )
    r = redhop.evaluate("q", ctx, gold_chunks=[])
    assert r.context_recall == pytest.approx(1.0)
    assert r.context_precision is None


# ─── gold_answer only ───────────────────────────────────────────────────────


def test_gold_answer_only_leaves_chunk_metrics_none():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    r = redhop.evaluate("refund window", ctx, gold_answer="thirty days")
    assert r.context_recall is None
    assert r.context_precision is None
    assert r.answer_token_recall is not None
    assert r.answer_token_recall > 0.0


def test_gold_answer_uses_stemming():
    """`refunds` (plural) in gold should still match `refund` (singular) in
    context — the runtime's Snowball stemmer runs on both sides."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days from purchase"),
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund window", ctx, gold_answer="refunds within thirty days"
    )
    # If stemming were dropped, the overlap would be only "thirty days" → ~0.5
    # of the {refund, within, thirty, day} set. With stemming, "refunds" maps
    # to "refund" too → ~0.75.
    assert r.answer_token_recall >= 0.5


# ─── both gold signals at once ──────────────────────────────────────────────


def test_both_gold_signals_populate_all_three_metrics():
    ctx = redhop.build_context(
        "refund window",
        [
            redhop.Chunk("the refund window is thirty days", id="hit"),
            redhop.Chunk("shipping policy details", id="noise"),
        ],
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund window",
        ctx,
        gold_chunks=["hit"],
        gold_answer="thirty days",
    )
    assert r.context_recall == pytest.approx(1.0)
    assert r.context_precision is not None
    assert r.answer_token_recall is not None
    assert r.answer_token_recall > 0.0
    assert 0.0 < r.overall <= 1.0


# ─── low-confidence + composite cap ─────────────────────────────────────────


def test_off_topic_query_flags_low_confidence_and_caps_overall():
    ctx = redhop.build_context(
        "quantum chromodynamics gluon coupling",
        [
            redhop.Chunk("the refund window is thirty days", id="a"),
            redhop.Chunk("shipping policy and delivery times", id="b"),
        ],
        strategy="raw_topk",
    )
    r = redhop.evaluate("quantum chromodynamics gluon coupling", ctx)
    assert r.low_confidence is True
    # Capped to ≤ 0.25 by the runtime (matches the Rust test contract).
    assert r.overall <= 0.25


# ─── detect → strip → evaluate workflow (the user-visible promise) ──────────


def test_detect_strip_evaluate_workflow():
    """The full workflow from the docs: analyze a query set, strip
    boilerplate, evaluate before vs after on the same gold."""
    cuad_shape = [
        'Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer.',
    ]
    report = redhop.analyze_query_set(cuad_shape)
    assert report.is_templated

    stripper = redhop.Stripper(report.boilerplate_terms)

    def strip(q):
        return stripper.apply(q)

    raw = cuad_shape[0]
    stripped = strip(raw)
    assert "Document Name" in stripped
    # The workflow ends in a call to evaluate; we don't assert the lift here
    # (that's a workload-specific measurement, not an FFI contract), only
    # that the call shape works end-to-end.
    chunks = _chunks_for("Document Name: Acme Co. Master Agreement", "hit")
    ctx_a = redhop.build_context(raw, chunks, strategy="raw_topk")
    ctx_b = redhop.build_context(stripped, chunks, strategy="raw_topk")
    eval_a = redhop.evaluate(raw, ctx_a, gold_chunks=["hit"])
    eval_b = redhop.evaluate(stripped, ctx_b, gold_chunks=["hit"])
    assert isinstance(eval_a.overall, float)
    assert isinstance(eval_b.overall, float)


# ─── repr is informative ────────────────────────────────────────────────────


def test_eval_report_repr_includes_key_fields():
    ctx = redhop.build_context(
        "q", _chunks_for("text"), strategy="raw_topk"
    )
    r = redhop.evaluate("q", ctx)
    s = repr(r)
    assert "EvalReport" in s
    assert "overall=" in s
    assert "mean_grounding=" in s


# ─── field types ────────────────────────────────────────────────────────────


def test_eval_report_field_types():
    ctx = redhop.build_context(
        "refund",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund", ctx, gold_chunks=["a"], gold_answer="thirty days"
    )
    assert isinstance(r.context_recall, float)
    assert isinstance(r.context_precision, float)
    assert isinstance(r.answer_token_recall, float)
    assert isinstance(r.mean_grounding, float)
    assert isinstance(r.evidence_density, float)
    assert isinstance(r.retained_evidence_ratio, float)
    assert isinstance(r.second_hop_rescues, int)
    assert isinstance(r.low_confidence, bool)
    assert isinstance(r.estimated_waste_tokens, int)
    assert isinstance(r.overall, float)
