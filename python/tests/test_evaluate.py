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


# ─── Tier-1 lexical answer-quality metrics ──────────────────────────────────
# Added in Phase 1 of the eval-parity work. These are deterministic
# token-overlap proxies for faithfulness/relevancy/correctness — useful for
# CI regression detection but explicitly NOT a substitute for an LLM judge.
# The Rust unit tests in crates/redhop/src/context/eval.rs are authoritative
# on the metric semantics; these guard the Python binding surface and
# kwarg names.


def test_tier1_none_when_answer_omitted():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days from purchase"),
        strategy="raw_topk",
    )
    r = redhop.evaluate("refund window", ctx)
    assert r.faithfulness_lexical is None
    assert r.relevancy_lexical is None
    assert r.correctness_lexical is None


def test_faithfulness_lexical_high_when_answer_grounded():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for(
            "the refund window is thirty days from purchase. customers may return items."
        ),
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="The refund window is thirty days from purchase.",
    )
    assert r.faithfulness_lexical is not None
    assert (
        r.faithfulness_lexical >= 0.9
    ), f"answer paraphrasing the context should score near 1.0; got {r.faithfulness_lexical}"


def test_faithfulness_lexical_low_when_answer_fabricated():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days from purchase"),
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer=(
            "Quantum chromodynamics couples gluons. "
            "Schrödinger equations describe quantum states. "
            "Heisenberg uncertainty bounds measurement."
        ),
    )
    assert r.faithfulness_lexical is not None
    assert r.faithfulness_lexical <= 0.5, (
        "fabricated answer with no context overlap should score low; "
        f"got {r.faithfulness_lexical}"
    )


def test_relevancy_lexical_higher_for_on_topic_answer():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    on_topic = redhop.evaluate(
        "refund window",
        ctx,
        answer="The refund window is thirty days.",
    ).relevancy_lexical
    off_topic = redhop.evaluate(
        "refund window",
        ctx,
        answer="Photosynthesis converts sunlight into glucose.",
    ).relevancy_lexical
    assert on_topic is not None and off_topic is not None
    assert (
        on_topic > off_topic
    ), f"on-topic answer must score higher; on={on_topic}, off={off_topic}"


def test_correctness_lexical_requires_both_answer_and_gold_answer():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    # Answer only — no correctness.
    r1 = redhop.evaluate(
        "refund window", ctx, answer="Thirty days from purchase."
    )
    assert r1.correctness_lexical is None

    # Gold answer only — also no correctness.
    r2 = redhop.evaluate("refund window", ctx, gold_answer="thirty days")
    assert r2.correctness_lexical is None

    # Both — correctness populated and positive on overlap.
    r3 = redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days from purchase.",
        gold_answer="thirty days",
    )
    assert r3.correctness_lexical is not None
    assert r3.correctness_lexical > 0.0


def test_empty_answer_treated_as_no_answer():
    """A whitespace-only `answer=` is the same as not passing it."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("refund window thirty days"),
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="   ",
        gold_answer="thirty days",
    )
    assert r.faithfulness_lexical is None
    assert r.relevancy_lexical is None
    assert r.correctness_lexical is None


def test_eval_report_tier1_field_types():
    """Type pin: Tier-1 metric getters return Optional[float] consistently."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days.",
        gold_answer="thirty days",
    )
    assert isinstance(r.faithfulness_lexical, float)
    assert isinstance(r.relevancy_lexical, float)
    assert isinstance(r.correctness_lexical, float)


# ─── Tier-2 LLM-judged metrics (Phase 3) ────────────────────────────────────
# These use a stub callable judge so we never hit a real LLM in CI. The
# Rust unit tests in crates/redhop/src/context/eval.rs are authoritative
# on the metric semantics; here we guard the Judge.from_callable bridge
# layer + the `judge=` kwarg on `evaluate`.


def _stub_judge_returning(score: float, call_log: list):
    """Build a Judge that always returns `score`, logging each call."""
    def fn(prompt, system):
        call_log.append((prompt, system))
        return score
    return redhop.Judge.from_callable(fn, name="stub")


def test_tier2_metrics_none_when_no_judge():
    """Without a judge, the _judged fields stay None even with all the
    other ingredients present."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days.",
        gold_answer="thirty days",
    )
    assert r.faithfulness_judged is None
    assert r.relevancy_judged is None
    assert r.correctness_judged is None


def test_tier2_metrics_populated_with_judge():
    """Judge supplied → all three _judged metrics populated."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    calls = []
    judge = _stub_judge_returning(0.85, calls)
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days from purchase.",
        gold_answer="thirty days",
        judge=judge,
    )
    assert r.faithfulness_judged is not None
    assert r.relevancy_judged is not None
    assert r.correctness_judged is not None
    # 3 judge calls — one per metric.
    assert len(calls) == 3
    # All three metrics see the same stub score (clamped to [0,1]).
    assert abs(r.faithfulness_judged - 0.85) < 0.01
    assert abs(r.relevancy_judged - 0.85) < 0.01
    assert abs(r.correctness_judged - 0.85) < 0.01


def test_tier2_correctness_skipped_without_gold_answer():
    """correctness_judged requires `gold_answer` AND `answer` AND `judge`.
    Without gold, only faithfulness + relevancy fire (2 judge calls)."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    calls = []
    judge = _stub_judge_returning(0.7, calls)
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days.",
        judge=judge,
    )
    assert r.faithfulness_judged is not None
    assert r.relevancy_judged is not None
    assert r.correctness_judged is None
    assert len(calls) == 2


def test_tier2_judge_cached_avoids_repeat_calls():
    """Wrapping a judge in .cached() should serve identical
    `(prompt, system)` pairs from cache on re-runs."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    calls = []
    judge = _stub_judge_returning(0.9, calls).cached()
    # First run — 3 calls.
    redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days.",
        gold_answer="thirty days",
        judge=judge,
    )
    assert len(calls) == 3
    # Second run with identical inputs — cache should serve everything.
    redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days.",
        gold_answer="thirty days",
        judge=judge,
    )
    assert len(calls) == 3, f"cache should suppress re-calls; got {len(calls)} total"


def test_tier2_judge_error_leaves_metric_none():
    """A judge that raises an exception should leave the _judged fields
    None — eval is best-effort, a transport error shouldn't crash."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )

    def fail(prompt, system):
        raise RuntimeError("transport error")

    judge = redhop.Judge.from_callable(fail, name="err")
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days.",
        gold_answer="thirty days",
        judge=judge,
    )
    assert r.faithfulness_judged is None
    assert r.relevancy_judged is None
    assert r.correctness_judged is None
    # Lexical fields still populated — judge failure is isolated.
    assert r.faithfulness_lexical is not None
    assert r.relevancy_lexical is not None


def test_tier2_judge_accepts_dict_return():
    """The callable may return a dict {score, raw_text?, model?} instead
    of a bare float — useful when the user wants to log raw_text."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )

    def rich(prompt, system):
        return {"score": 0.55, "raw_text": "0.55 because…", "model": "fake-gpt"}

    judge = redhop.Judge.from_callable(rich, name="rich")
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="Thirty days.",
        judge=judge,
    )
    assert r.faithfulness_judged is not None
    assert abs(r.faithfulness_judged - 0.55) < 0.01


def test_judge_repr_includes_name():
    judge = redhop.Judge.from_callable(lambda p, s: 0.5, name="myname")
    rep = repr(judge)
    assert "Judge" in rep
    assert "myname" in rep


# ─── Phase 4: summarize(reports) ────────────────────────────────────────────


def _make_reports(n: int, with_answer: bool = True):
    """Build n EvalReports against a tiny doc, varying the query so the
    overall scores differ across reports."""
    doc = redhop.Document.from_text(
        "the refund window is thirty days. customers may return items in original packaging."
    )
    queries = [f"refund {i}" for i in range(n)]
    reports = []
    for q in queries:
        ctx = doc.context(q)
        if with_answer:
            reports.append(
                redhop.evaluate(q, ctx, answer="Thirty days from purchase.", gold_answer="thirty days")
            )
        else:
            reports.append(redhop.evaluate(q, ctx))
    return reports


def test_summarize_empty_input_returns_zero_shape():
    s = redhop.summarize([])
    assert s.n == 0
    assert s.mean_overall == 0.0
    assert s.median_overall == 0.0
    assert s.mean_faithfulness_lexical is None
    assert s.n_with_faithfulness_lexical == 0
    assert s.mean_faithfulness_judged is None


def test_summarize_aggregates_self_eval_always():
    reports = _make_reports(3, with_answer=False)
    s = redhop.summarize(reports)
    assert s.n == 3
    # Self-eval and overall always populated.
    assert 0.0 <= s.mean_overall <= 1.0
    assert 0.0 <= s.median_overall <= 1.0
    assert 0.0 <= s.low_confidence_rate <= 1.0
    # No answer was supplied → all Tier-1 lexical fields stay None.
    assert s.mean_faithfulness_lexical is None
    assert s.n_with_faithfulness_lexical == 0


def test_summarize_populates_lexical_when_reports_have_them():
    reports = _make_reports(3, with_answer=True)
    s = redhop.summarize(reports)
    assert s.n == 3
    # All 3 reports have answer set → faithfulness_lexical populated 3/3.
    assert s.mean_faithfulness_lexical is not None
    assert s.n_with_faithfulness_lexical == 3
    assert 0.0 <= s.mean_faithfulness_lexical <= 1.0
    # Tier-2 still None — no judge.
    assert s.mean_faithfulness_judged is None
    assert s.n_with_faithfulness_judged == 0


def test_summarize_repr_includes_key_fields():
    reports = _make_reports(2, with_answer=True)
    s = redhop.summarize(reports)
    rep = repr(s)
    assert "EvalSummary" in rep
    assert "n=2" in rep
    assert "mean_overall" in rep


# ─── Phase 6: claim decomposition for faithfulness ──────────────────────────


def _decomposer_judge(claims: list[str], verification_score: float, call_log: list):
    """Stub judge that returns the claim list as raw_text on the extraction
    pass, and a numeric score on verification passes. The Rust core
    routes between the two by the system prompt content."""
    extracted_text = "\n".join(claims)

    def fn(prompt, system):
        call_log.append((prompt[:60], system[:60] if system else None))
        if system and "Decompose answers" in system:
            return {"score": 0.0, "raw_text": extracted_text, "model": "stub"}
        return verification_score

    return redhop.Judge.from_callable(fn, name="decomposer")


def test_decomposition_off_by_default():
    """Without decompose_faithfulness=True, the legacy single-prompt path
    runs and the n_faithfulness_claims_* fields stay None."""
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    calls = []
    judge = _stub_judge_returning(0.9, calls)
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="The refund window is thirty days.",
        judge=judge,
    )
    assert r.faithfulness_judged is not None
    assert r.n_faithfulness_claims_extracted is None
    assert r.n_faithfulness_claims_supported is None
    # 2 calls: faithfulness + relevancy.
    assert len(calls) == 2


def test_decomposition_extracts_and_verifies():
    ctx = redhop.build_context(
        "refund window",
        _chunks_for("the refund window is thirty days"),
        strategy="raw_topk",
    )
    calls = []
    judge = _decomposer_judge(
        ["claim 1", "claim 2", "claim 3"],
        verification_score=0.8,
        call_log=calls,
    )
    r = redhop.evaluate(
        "refund window",
        ctx,
        answer="Three claims here.",
        judge=judge,
        decompose_faithfulness=True,
    )
    # 3 claims at 0.8 each → mean 0.8.
    assert r.faithfulness_judged is not None
    assert abs(r.faithfulness_judged - 0.8) < 0.01
    assert r.n_faithfulness_claims_extracted == 3
    # 0.8 ≥ 0.5 → all supported.
    assert r.n_faithfulness_claims_supported == 3
    # Calls: 1 extraction + 3 verifications + 1 relevancy = 5.
    assert len(calls) == 5


def test_decomposition_zero_claims_returns_none():
    """If extraction yields no claims (e.g. refusal answer), the metric
    stays None rather than falling back to single-prompt scoring."""
    ctx = redhop.build_context(
        "q", _chunks_for("anything"), strategy="raw_topk"
    )
    calls = []
    judge = _decomposer_judge(
        claims=[],
        verification_score=0.5,
        call_log=calls,
    )
    r = redhop.evaluate(
        "q",
        ctx,
        answer="I cannot answer that.",
        judge=judge,
        decompose_faithfulness=True,
    )
    assert r.faithfulness_judged is None
    assert r.n_faithfulness_claims_extracted is None
    assert r.n_faithfulness_claims_supported is None


def test_decomposition_unsupported_threshold():
    """A verification score < 0.5 counts as 'not supported'."""
    ctx = redhop.build_context(
        "q", _chunks_for("anything"), strategy="raw_topk"
    )
    calls = []
    judge = _decomposer_judge(
        claims=["a", "b", "c", "d"],
        verification_score=0.3,
        call_log=calls,
    )
    r = redhop.evaluate(
        "q",
        ctx,
        answer="Four claims.",
        judge=judge,
        decompose_faithfulness=True,
    )
    assert r.n_faithfulness_claims_extracted == 4
    assert r.n_faithfulness_claims_supported == 0  # all below 0.5


# ─── Phase 7: aspect critique ──────────────────────────────────────────────


def test_aspect_constructor_and_getters():
    a = redhop.Aspect("harmfulness", "Does it contain harmful content?", high_is_good=False)
    assert a.name == "harmfulness"
    assert a.definition == "Does it contain harmful content?"
    assert a.high_is_good is False
    rep = repr(a)
    assert "Aspect" in rep
    assert "harmfulness" in rep


def test_critique_each_aspect_one_judge_call():
    calls = []
    judge = _stub_judge_returning(0.7, calls)
    aspects = [
        redhop.Aspect("a", "First aspect"),
        redhop.Aspect("b", "Second aspect"),
        redhop.Aspect("c", "Third aspect"),
    ]
    report = redhop.critique("Some answer.", aspects, judge=judge)
    assert len(report) == 3
    assert report["a"] == pytest.approx(0.7, abs=0.01)
    assert report["b"] == pytest.approx(0.7, abs=0.01)
    assert report["c"] == pytest.approx(0.7, abs=0.01)
    assert len(calls) == 3


def test_critique_high_is_good_false_inverts_score():
    # LLM says raw 0.9 ("very harmful"). high_is_good=False → 1.0 - 0.9 = 0.1.
    calls = []
    judge = _stub_judge_returning(0.9, calls)
    aspects = [
        redhop.Aspect("harmfulness", "Is it harmful?", high_is_good=False),
    ]
    report = redhop.critique("anything", aspects, judge=judge)
    s = report["harmfulness"]
    assert s is not None
    assert s == pytest.approx(0.1, abs=0.01)


def test_critique_empty_aspects_makes_no_judge_calls():
    calls = []
    judge = _stub_judge_returning(0.5, calls)
    report = redhop.critique("x", [], judge=judge)
    assert len(report) == 0
    assert report.scores == []
    assert len(calls) == 0


def test_critique_judge_error_isolates_to_that_aspect():
    """A judge that errors for one aspect leaves only that score None."""
    n = [0]

    def fail_on_second(prompt, system):
        n[0] += 1
        if n[0] == 2:
            raise RuntimeError("transient")
        return 0.6

    judge = redhop.Judge.from_callable(fail_on_second, name="flaky")
    aspects = [
        redhop.Aspect("a", "first"),
        redhop.Aspect("b", "second"),
        redhop.Aspect("c", "third"),
    ]
    report = redhop.critique("x", aspects, judge=judge)
    assert report["a"] is not None
    assert report["b"] is None, "second aspect should be None on transient error"
    assert report["c"] is not None


def test_critique_context_and_query_are_optional():
    """Both context= and query= are optional kwargs; omitting them works."""
    calls = []
    judge = _stub_judge_returning(0.8, calls)
    aspects = [redhop.Aspect("x", "test")]
    r1 = redhop.critique("answer", aspects, judge=judge)
    r2 = redhop.critique("answer", aspects, judge=judge, context="ctx")
    r3 = redhop.critique(
        "answer", aspects, judge=judge, context="ctx", query="q"
    )
    for r in (r1, r2, r3):
        assert r["x"] == pytest.approx(0.8, abs=0.01)


def test_critique_get_returns_none_for_missing_aspect_name():
    judge = _stub_judge_returning(0.5, [])
    report = redhop.critique("x", [redhop.Aspect("present", "x")], judge=judge)
    assert report["present"] == pytest.approx(0.5, abs=0.01)
    assert report["missing"] is None
    assert report.get("missing") is None


def test_critique_report_repr():
    judge = _stub_judge_returning(0.7, [])
    report = redhop.critique(
        "x", [redhop.Aspect("conciseness", "concise?")], judge=judge
    )
    rep = repr(report)
    assert "CritiqueReport" in rep
    assert "conciseness" in rep
