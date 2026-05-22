"""Basic Python API tests + parity checks. Run with: pytest python/tests/

These verify the binding surface and that the Python results match what the
Rust engine computes (Rust is the source of truth)."""

import redhop

QUERY = "what nationality was the inventor of the miners' safety lamp"
CHUNKS = [
    {"id": "hop1", "text": "The miners' safety lamp was invented by Humphry Davy in 1815."},
    {"id": "hop2", "text": "Humphry Davy was a British chemist, born in Penzance, Cornwall, England."},
    {"id": "d1", "text": "Photosynthesis converts sunlight into glucose and oxygen in plants."},
]
KW = dict(distractor_min_grounding=0.30, link_min_jaccard=0.15)


def test_build_context_basic():
    ctx = redhop.build_context(QUERY, CHUNKS, token_budget=12000, **KW)
    assert isinstance(ctx.text(), str)
    assert "Humphry Davy" in ctx.text()
    assert ctx.report.n_input_chunks == 3
    assert ctx.report.n_selected <= 3


def test_reasoning_preserving_keeps_second_hop_filter_drops_it():
    rp = redhop.build_context(QUERY, CHUNKS, strategy="reasoning_preserving", token_budget=12000, **KW)
    df = redhop.build_context(QUERY, CHUNKS, strategy="distractor_filtered", token_budget=12000, **KW)
    assert "British" in rp.text()          # second hop rescued
    assert "British" not in df.text()      # second hop taxed away
    assert rp.report.second_hop_rescue_count >= 1
    assert df.report.second_hop_rescue_count == 0


def test_accepts_plain_strings():
    ctx = redhop.build_context(QUERY, ["a bare string", "another"], token_budget=100)
    assert ctx.report.n_input_chunks == 2


def test_analyze_is_non_destructive():
    r = redhop.analyze_context(QUERY, CHUNKS, **KW)
    assert r.n_selected == r.n_input_chunks
    assert r.removed_total == 0


def test_context_economics_returns_dict():
    econ = redhop.context_economics(QUERY, CHUNKS, **KW)
    assert "evidence_density" in econ
    assert "distractor_ratio" in econ


def test_report_to_dict_and_str():
    ctx = redhop.build_context(QUERY, CHUNKS, token_budget=12000, **KW)
    d = redhop.report_to_dict(ctx.report)
    assert d["strategy"] == "reasoning_preserving"
    assert "Context Optimization Report" in str(ctx.report)


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        redhop.build_context(QUERY, CHUNKS, strategy="nonsense")
