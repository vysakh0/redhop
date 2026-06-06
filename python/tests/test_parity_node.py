"""Cross-binding parity tests: Python and Node must produce IDENTICAL
results for identical inputs.

Spawns the Node runner at `nodejs/test/parity_runner.cjs` for each call
and compares the JSON response field-by-field with what Python's
`redhop` returns from the equivalent function.

These tests catch a class of regression that no single-binding test can:
**silent divergence** between the two language surfaces (one side updates,
the other doesn't). They were the missing safety net under the
binding-parity arc; everything else asserted "this binding works", never
"both bindings agree".

Skip rules:
- Node not installed → skip (don't fail the suite).
- The Node addon not built → skip with a clear instruction.

Run with: pytest python/tests/test_parity_node.py -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import redhop

# ── Locate the Node runner + skip cleanly if Node or the addon isn't there ──

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "nodejs" / "test" / "parity_runner.cjs"
NODE_BIN = shutil.which("node")

# napi-rs writes the platform-tagged .node alongside the JS loader after a
# successful `napi build`. If it's missing the addon hasn't been built;
# `npm run build` from `nodejs/` produces it.
ADDON_BUILT = any((REPO_ROOT / "nodejs").glob("redhop.*.node"))

pytestmark = [
    pytest.mark.skipif(
        NODE_BIN is None, reason="node not on PATH (install Node.js to run parity tests)"
    ),
    pytest.mark.skipif(
        not ADDON_BUILT, reason="napi addon not built (run `npm run build` in nodejs/)"
    ),
    pytest.mark.skipif(not RUNNER.exists(), reason="parity runner missing"),
]


def node_call(fn: str, args: list[Any]) -> Any:
    """Invoke the Node binding via the subprocess runner. Returns the
    parsed `result` payload or raises on a bad response."""
    proc = subprocess.run(
        [NODE_BIN, str(RUNNER)],
        input=json.dumps({"fn": fn, "args": args}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"node parity_runner exit {proc.returncode}: {proc.stderr}")
    resp = json.loads(proc.stdout)
    if not resp.get("ok"):
        raise RuntimeError(f"node side error: {resp.get('error')}")
    return resp["result"]


# ── Shared fixtures ─────────────────────────────────────────────────────────

# A small fixed corpus exercising both the on-topic and off-topic paths so
# the parity tests cover real branches (filtering, assembly, scoring).
CORPUS = [
    {"id": "g1", "text": "The refund window is thirty days from the purchase date."},
    {"id": "g2", "text": "Customers may return items within thirty days for a full refund."},
    {"id": "d1", "text": "Photosynthesis converts sunlight into glucose in plants."},
    {"id": "d2", "text": "The capital of France is Paris."},
]


# ── Scalar parity (no float fuzz: same Rust source, deterministic) ─────────


def test_grounding_score_parity():
    """`grounding_score(query, text)` must return bit-for-bit identical
    floats on both bindings — same Rust call underneath, same inputs."""
    query = "refund window"
    text = "the refund window is thirty days"
    py = redhop.grounding_score(query, text)
    node = node_call("groundingScore", [query, text])
    assert py == node, f"grounding_score diverged: python={py!r} node={node!r}"


def test_link_strength_parity():
    """`link_strength(a, b)` must agree across bindings."""
    a = "refund within thirty days"
    b = "thirty-day refund policy"
    py = redhop.link_strength(a, b)
    node = node_call("linkStrength", [a, b])
    assert py == node, f"link_strength diverged: python={py!r} node={node!r}"


# ── Structural parity over BuiltContext ─────────────────────────────────────


def _normalize_built(ctx_or_dict: Any, side: str) -> dict[str, Any]:
    """Normalize a BuiltContext to a JSON-shaped dict for comparison.

    Python: returns the pyclass directly — we project to dict.
    Node: returns the napi object as a dict already (camelCase).
    Returns the same snake_case shape so the assertion is symmetric.
    """
    if side == "python":
        return {
            "text": ctx_or_dict.text(),
            "chunks": list(ctx_or_dict.chunks),
            "n_citations": len(ctx_or_dict.citations),
            "report": {
                # `strategy` + `requested_strategy` are explicitly compared
                # because they were once silently absent from the Node
                # binding (0.2.1 added them). Direct dict-key access here
                # means a future regression that drops either field on
                # either side raises KeyError instead of silently passing.
                "strategy": ctx_or_dict.report.strategy,
                "requested_strategy": ctx_or_dict.report.requested_strategy,
                "auto_decision": ctx_or_dict.report.auto_decision,
                "total_tokens": ctx_or_dict.report.total_tokens,
                "retained_evidence_ratio": ctx_or_dict.report.retained_evidence_ratio,
                "second_hop_rescues": ctx_or_dict.report.second_hop_rescue_count,
                "n_expanded": ctx_or_dict.report.n_expanded,
                "low_confidence_retrieval": ctx_or_dict.report.low_confidence_retrieval,
                "low_confidence_threshold": ctx_or_dict.report.low_confidence_threshold,
            },
        }
    # node side: camelCase dict from napi
    d = ctx_or_dict
    return {
        "text": d["text"],
        "chunks": d["chunks"],
        "n_citations": len(d["citations"]),
        "report": {
            "strategy": d["report"]["strategy"],
            "requested_strategy": d["report"]["requestedStrategy"],
            "auto_decision": d["report"]["autoDecision"],
            "total_tokens": d["report"]["totalTokens"],
            "retained_evidence_ratio": d["report"]["retainedEvidenceRatio"],
            "second_hop_rescues": d["report"]["secondHopRescues"],
            "n_expanded": d["report"]["nExpanded"],
            "low_confidence_retrieval": d["report"]["lowConfidenceRetrieval"],
            "low_confidence_threshold": d["report"]["lowConfidenceThreshold"],
        },
    }


@pytest.mark.parametrize(
    "query,expect_on_topic",
    [
        ("what is the refund window?", "refund"),
        ("when does the return policy expire?", "refund"),
    ],
)
def test_build_context_parity(query, expect_on_topic):
    """`build_context` end-to-end: same query + same chunks should produce
    identical assembled text, identical chunk count, identical totals."""
    py = redhop.build_context(query, CORPUS)
    node = node_call("buildContext", [query, CORPUS])
    py_norm = _normalize_built(py, "python")
    node_norm = _normalize_built(node, "node")
    assert py_norm["text"] == node_norm["text"], (
        f"build_context text diverged:\n  python: {py_norm['text']!r}\n  node:   {node_norm['text']!r}"
    )
    assert py_norm["chunks"] == node_norm["chunks"], (
        f"build_context chunks diverged:\n  python: {py_norm['chunks']!r}\n  node:   {node_norm['chunks']!r}"
    )
    assert py_norm["report"] == node_norm["report"], (
        f"build_context report diverged:\n  python: {py_norm['report']!r}\n  node:   {node_norm['report']!r}"
    )
    # Sanity: on-topic chunk should survive on both
    assert expect_on_topic in py_norm["text"], "python should keep on-topic"
    assert expect_on_topic in node_norm["text"], "node should keep on-topic"


def test_analyze_context_parity():
    """`analyze_context` returns just a report. Verify both sides agree on
    decision + tokens + retention + the resolved strategy."""
    query = "what is the refund window?"
    py = redhop.analyze_context(query, CORPUS)
    node = node_call("analyzeContext", [query, CORPUS])
    # `strategy` + `requested_strategy` were missing from the Node binding
    # before 0.2.2 — pin them here so a regression that drops either field
    # fails with a clear message instead of silently degrading.
    assert py.strategy == node["strategy"], (
        f"strategy diverged: python={py.strategy!r} node={node['strategy']!r}"
    )
    assert py.requested_strategy == node["requestedStrategy"], (
        f"requested_strategy diverged: python={py.requested_strategy!r} "
        f"node={node['requestedStrategy']!r}"
    )
    assert py.auto_decision == node["autoDecision"], (
        f"auto_decision diverged: python={py.auto_decision!r} node={node['autoDecision']!r}"
    )
    assert py.total_tokens == node["totalTokens"]
    assert py.retained_evidence_ratio == node["retainedEvidenceRatio"]
    assert py.second_hop_rescue_count == node["secondHopRescues"]


def test_context_economics_parity():
    """`context_economics` returns the SAME economics struct on both sides
    — Python's `__init__.py` wrapper json-decodes the Rust JSON string into
    a dict; the node runner does the same with `JSON.parse`. So both sides
    arrive at a dict of the same shape and we compare directly."""
    query = "what is the refund window?"
    py = redhop.context_economics(query, CORPUS)
    node = node_call("contextEconomics", [query, CORPUS])
    assert py == node, f"context_economics diverged:\n  python: {py!r}\n  node:   {node!r}"


# ── Determinism within a binding (rerun must equal first run) ──────────────


def test_python_determinism_repeat_run():
    """Same call twice on the Python side must produce byte-identical
    results. The cross-binding parity tests above presume both sides are
    deterministic; this pins that assumption directly."""
    query = "what is the refund window?"
    a = redhop.build_context(query, CORPUS)
    b = redhop.build_context(query, CORPUS)
    assert a.text() == b.text(), "python build_context text not deterministic"
    assert list(a.chunks) == list(b.chunks), "python chunks order not deterministic"
    assert a.report.total_tokens == b.report.total_tokens, "python total_tokens not deterministic"
    assert a.report.auto_decision == b.report.auto_decision


def test_node_determinism_repeat_run():
    """Same buildContext call twice on the Node side must produce
    byte-identical results."""
    query = "what is the refund window?"
    a = node_call("buildContext", [query, CORPUS])
    b = node_call("buildContext", [query, CORPUS])
    assert a["text"] == b["text"], "node buildContext text not deterministic"
    assert a["chunks"] == b["chunks"], "node chunks order not deterministic"
    assert a["report"]["totalTokens"] == b["report"]["totalTokens"]
    assert a["report"]["autoDecision"] == b["report"]["autoDecision"]


# ── Field-set parity (structural, auto-detects future divergences) ─────────
#
# The data-value tests above (`test_build_context_parity` etc.) only check
# fields that the test author explicitly listed in `_normalize_built`. That
# is exactly how `Report.strategy` was silently missing from the Node
# binding before 0.2.2 — the test normalized the field away and nobody
# noticed. These three tests close that gap class:
#
#   For each FFI-crossing return type (Report, BuiltContext, ContextEconomics),
#   compare the SET of fields each binding exposes. A future field added to
#   one side without the other catches a clear failure naming the offending
#   field, without anyone having to remember to extend a manual list.
#
# Allowlists below are tiny — they document the handful of intentional
# shape asymmetries (most importantly the call-shape differences pinned in
# docs/API_STABILITY.md "Known call-shape asymmetries"). New entries here
# should reference a finding or design doc; otherwise close the gap.


import re  # noqa: E402  (intentionally late — locality with the field-set tests)


def _camel_to_snake(name: str) -> str:
    """`requestedStrategy` → `requested_strategy`."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _py_data_fields(obj: Any) -> set[str]:
    """Names of READABLE data fields on a Python pyclass instance.

    Includes `#[getter]` attributes (they look like properties at the
    Python level), excludes callable methods and dunders. The point is
    'what fields can a caller READ' — not the entire dir() surface.
    """
    out: set[str] = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        out.add(name)
    return out


def _node_data_fields(d: dict) -> set[str]:
    """Snake-cased keys of a Node-side result dict."""
    return {_camel_to_snake(k) for k in d.keys()}


# Intentional shape asymmetries — documented in docs/API_STABILITY.md
# ("Known call-shape asymmetries"). Adding a new entry requires a comment
# explaining WHY it's not just a bug to fix on the missing side.
_REPORT_PY_ONLY: set[str] = set()  # if this grows, add Node fields or document why
_REPORT_NODE_ONLY = {
    # Node exposes the rendered Decision Report as a string field;
    # Python exposes it via `str(report)` / `__str__`. Different idiom,
    # same intent. Pinned in docs/API_STABILITY.md.
    "rendered",
    # `secondHopRescues` is the short name Node has shipped since 0.2.0.
    # Python's canonical field is `second_hop_rescue_count`. To keep
    # parity without a breaking rename, 0.2.2 added `secondHopRescueCount`
    # to Node as a permanent alias (== `secondHopRescues`); both names
    # are now exposed in Node, only the long name in Python. This entry
    # documents why the short alias `second_hop_rescues` (snake-cased)
    # appears in Node only.
    "second_hop_rescues",
}
_BUILT_PY_ONLY: set[str] = set()
_BUILT_NODE_ONLY = {
    # Python's `ctx.text()` is a callable method (pyo3 idiom: the
    # underlying Rust value is borrowed). Node's `ctx.text` is a string
    # property (napi-rs `#[napi(object)]` idiom). Documented as a stable
    # 0.x asymmetry — same value, different call shape.
    "text",
}
# ContextEconomics is a dict on BOTH sides (Python wrapper json-decodes
# the Rust JSON string; Node parity_runner JSON.parses the napi result).
# Direct dict comparison is meaningful — no intentional asymmetry expected.
_ECON_PY_ONLY: set[str] = set()
_ECON_NODE_ONLY: set[str] = set()


def _diff_msg(label: str, py_only: set[str], node_only: set[str]) -> str:
    parts = [f"{label} field surface diverged:"]
    if py_only:
        parts.append(f"  present in Python only: {sorted(py_only)}")
    if node_only:
        parts.append(f"  present in Node only:   {sorted(node_only)}")
    parts.append(
        "  If a divergence is intentional, add it to the corresponding "
        "_*_PY_ONLY / _*_NODE_ONLY set with a comment explaining why "
        "(and update docs/API_STABILITY.md if it's a stable asymmetry)."
    )
    return "\n".join(parts)


def test_report_field_surface_parity():
    """The set of `Report` fields must match across bindings (modulo the
    documented `rendered` idiom). Auto-catches the gap class where a new
    `#[getter]` is added to Python's `ContextReport` but not to Node's
    `Report` struct — the failure mode that hid `strategy` /
    `requested_strategy` for months before 0.2.2."""
    query = "what is the refund window?"
    py_report = redhop.analyze_context(query, CORPUS)
    node_report = node_call("analyzeContext", [query, CORPUS])
    py = _py_data_fields(py_report)
    node = _node_data_fields(node_report)
    py_only = py - node - _REPORT_PY_ONLY
    node_only = node - py - _REPORT_NODE_ONLY
    assert not (py_only or node_only), _diff_msg("Report", py_only, node_only)


def test_built_context_field_surface_parity():
    """The set of `BuiltContext` fields must match across bindings
    (modulo the documented `text` callable-vs-property asymmetry)."""
    query = "what is the refund window?"
    py_built = redhop.build_context(query, CORPUS)
    node_built = node_call("buildContext", [query, CORPUS])
    py = _py_data_fields(py_built)
    node = _node_data_fields(node_built)
    py_only = py - node - _BUILT_PY_ONLY
    node_only = node - py - _BUILT_NODE_ONLY
    assert not (py_only or node_only), _diff_msg("BuiltContext", py_only, node_only)


def test_context_economics_field_surface_parity():
    """`context_economics` is a dict on both sides — pin that the key set
    matches. The data-value test above (`test_context_economics_parity`)
    already asserts `py == node`, which would also catch a key
    mismatch, but this test names the offending field on failure rather
    than just reporting a dict inequality."""
    query = "what is the refund window?"
    py = redhop.context_economics(query, CORPUS)
    node = node_call("contextEconomics", [query, CORPUS])
    py_keys = {_camel_to_snake(k) for k in py.keys()}
    node_keys = {_camel_to_snake(k) for k in node.keys()}
    py_only = py_keys - node_keys - _ECON_PY_ONLY
    node_only = node_keys - py_keys - _ECON_NODE_ONLY
    assert not (py_only or node_only), _diff_msg("ContextEconomics", py_only, node_only)
