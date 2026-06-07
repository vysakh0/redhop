"""Binding-surface tests for the query-set analyzer.

Mirrors the Rust unit tests on `redhop::analyze_query_set` through the
pyo3 boundary, so a dropped field on `QuerySetReport` or a wrong
list ↔ Vec mapping at the FFI edge surfaces here, not in user code.

The mechanism + thresholds are documented in
docs/findings/QUERY_SET_ANALYZER.md. These tests cover the API
contract, not the heuristic — the heuristic is validated by the
cross-workload probe (`query_set_analyzer_probe` example).

Run with: pytest python/tests/test_query_set_analyzer.py -q
"""

import redhop


# ─── analyze_query_set ───────────────────────────────────────────────────────


def _cuad_shape_queries() -> list[str]:
    return [
        'Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer.',
        'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer.',
    ]


def _diverse_queries() -> list[str]:
    return [
        "Who is the current president of France?",
        "When was the Eiffel Tower built?",
        "What language do they speak in Brazil?",
        "How tall is Mount Everest?",
        "Which planet is closest to the sun?",
        "When did World War II end?",
        "Who wrote Pride and Prejudice?",
        "What is the capital of Japan?",
    ]


def test_analyze_query_set_returns_report_object():
    report = redhop.analyze_query_set(_cuad_shape_queries())
    assert isinstance(report, redhop.QuerySetReport)


def test_analyze_query_set_detects_cuad_shape():
    report = redhop.analyze_query_set(_cuad_shape_queries())
    assert report.is_templated, repr(report)
    assert report.template_word_share > 0.6
    assert report.estimated_dilution_cost == "high"
    for expected in ("highlight", "contract", "lawyer"):
        assert expected in report.boilerplate_terms, repr(report.boilerplate_terms)


def test_analyze_query_set_does_not_fire_on_diverse_queries():
    report = redhop.analyze_query_set(_diverse_queries())
    assert not report.is_templated, repr(report)
    # Diverse queries should land in `low` or `none` cost band.
    assert report.estimated_dilution_cost in ("low", "none")


def test_analyze_query_set_empty_list():
    report = redhop.analyze_query_set([])
    assert report.n_queries == 0
    assert not report.is_templated
    assert report.template_word_share == 0.0
    assert report.boilerplate_terms == []
    assert report.estimated_dilution_cost == "none"
    assert "empty" in report.suggested_action.lower()


def test_query_set_report_field_types():
    report = redhop.analyze_query_set(_cuad_shape_queries())
    assert isinstance(report.n_queries, int)
    assert isinstance(report.is_templated, bool)
    assert isinstance(report.template_word_share, float)
    assert isinstance(report.boilerplate_terms, list)
    assert all(isinstance(t, str) for t in report.boilerplate_terms)
    assert isinstance(report.estimated_dilution_cost, str)
    assert isinstance(report.suggested_action, str)


def test_query_set_report_repr_is_informative():
    report = redhop.analyze_query_set(_cuad_shape_queries())
    r = repr(report)
    assert "QuerySetReport" in r
    assert "n=" in r
    assert "is_templated" in r


def test_stripper_consumes_analyze_output():
    """End-to-end: detect → strip via `Stripper(boilerplate_terms).apply(q)`
    is the documented user workflow."""
    queries = _cuad_shape_queries()
    report = redhop.analyze_query_set(queries)
    assert report.is_templated
    stripper = redhop.Stripper(report.boilerplate_terms)
    stripped = stripper.apply(queries[0])
    assert "Document Name" in stripped
    for noise in ("Highlight", "contract", "lawyer"):
        assert noise.lower() not in stripped.lower(), (
            f"expected {noise!r} to be stripped; got: {stripped!r}"
        )
