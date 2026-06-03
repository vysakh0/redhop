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
    pytest.mark.skipif(NODE_BIN is None, reason="node not on PATH (install Node.js to run parity tests)"),
    pytest.mark.skipif(not ADDON_BUILT, reason="napi addon not built (run `npm run build` in nodejs/)"),
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
    decision + tokens + retention."""
    query = "what is the refund window?"
    py = redhop.analyze_context(query, CORPUS)
    node = node_call("analyzeContext", [query, CORPUS])
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
    assert py == node, (
        f"context_economics diverged:\n  python: {py!r}\n  node:   {node!r}"
    )
