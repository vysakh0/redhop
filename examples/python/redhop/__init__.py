"""redhop — a reasoning-preserving context optimization layer.

RedHop is NOT a retriever, a vector DB, an agent framework, or a workflow
engine. It sits *between retrieval and generation*: you hand it the chunks
your retriever returned and a token budget, and it assembles the prompt
context — removing distractors, preserving reasoning-critical "second hop"
evidence, and reporting exactly what it did.

    chunks = retriever.retrieve(query)            # your retriever
    ctx = redhop.build_context(                   # RedHop
        query=query,
        retrieved_chunks=chunks,
        token_budget=12000,
        strategy="reasoning_preserving",
    )
    response = llm.generate(ctx.text)             # your LLM
    print(ctx.report)                             # observability

This module is a thin shim over the Rust engine, invoked through a JSON
bridge (the `redhop_bridge` example binary). It is the "minimal thing that
works today"; native wheels (`pip install redhop`, backed by pyo3) are the
future packaging path — the API here is the one those wheels will expose.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

# Repo root: examples/python/redhop/__init__.py -> repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE_BIN_NAMES = ("redhop_bridge",)


class RedhopError(RuntimeError):
    """Raised when the Rust bridge cannot be located or fails."""


@lru_cache(maxsize=1)
def _bridge_path() -> str:
    """Locate the compiled bridge binary, building it once if necessary."""
    # Allow an explicit override.
    env = os.environ.get("REDHOP_BRIDGE")
    if env and Path(env).exists():
        return env
    for profile in ("release", "debug"):
        for name in _BRIDGE_BIN_NAMES:
            cand = _REPO_ROOT / "target" / profile / "examples" / name
            if cand.exists():
                return str(cand)
    # Not built yet — build it (release) once.
    if shutil.which("cargo") is None:
        raise RedhopError(
            "redhop bridge not found and `cargo` is unavailable. Build it with:\n"
            "  cargo build -p redhop-examples --example redhop_bridge --release"
        )
    subprocess.run(
        ["cargo", "build", "-p", "redhop-examples", "--example", "redhop_bridge", "--release"],
        cwd=_REPO_ROOT,
        check=True,
    )
    cand = _REPO_ROOT / "target" / "release" / "examples" / "redhop_bridge"
    if not cand.exists():
        raise RedhopError("bridge build succeeded but binary not found")
    return str(cand)


@dataclass(frozen=True)
class ContextReport:
    """Observability trace for one context assembly. `str(report)` is the
    pretty, human-readable Context Optimization Report."""

    data: Mapping[str, Any]
    rendered: str

    # Convenience accessors over the underlying telemetry.
    @property
    def strategy(self) -> str:
        return self.data["strategy"]

    @property
    def n_input_chunks(self) -> int:
        return self.data["n_input_chunks"]

    @property
    def n_selected(self) -> int:
        return self.data["n_selected"]

    @property
    def total_tokens(self) -> int:
        return self.data["total_tokens"]

    @property
    def distractors_pruned(self) -> int:
        return self.data["removed"]["distractor"]

    @property
    def second_hop_rescue_count(self) -> int:
        return self.data["second_hop_rescue_count"]

    @property
    def evidence_density(self) -> float:
        return self.data["economics"]["evidence_density"]

    @property
    def estimated_waste_tokens(self) -> int:
        return self.data["economics"]["estimated_waste_tokens"]

    def __str__(self) -> str:
        return self.rendered

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class BuiltContext:
    """The assembled context (`.text`) plus its `.report`."""

    text: str
    report: ContextReport


def _normalize_chunk(c: Any, i: int) -> dict[str, Any]:
    """Accept a dict, a string, or any object with `.text`/`.page_content`."""
    if isinstance(c, str):
        return {"id": f"c{i}", "text": c}
    if isinstance(c, Mapping):
        out = {"id": str(c.get("id", f"c{i}")), "text": c["text"]}
        for k in ("source", "token_count", "embedding"):
            if c.get(k) is not None:
                out[k] = c[k]
        return out
    # LangChain Document and friends.
    text = getattr(c, "page_content", None) or getattr(c, "text", None)
    if text is None:
        raise RedhopError(f"chunk {i} has no text (.page_content/.text/dict['text'])")
    meta = getattr(c, "metadata", {}) or {}
    return {"id": str(meta.get("id", f"c{i}")), "text": text, "source": str(meta.get("source", "input"))}


def _call_bridge(request: Mapping[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [_bridge_path()],
        input=json.dumps(request),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RedhopError(f"bridge failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def build_context(
    query: str,
    retrieved_chunks: Sequence[Any],
    token_budget: int = 8192,
    strategy: str = "reasoning_preserving",
    *,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
    redundancy_max_cosine: float = 0.92,
) -> BuiltContext:
    """Assemble a finite-attention context from retrieved chunks.

    `retrieved_chunks` may be strings, dicts ({"id","text",...}), or objects
    with `.page_content`/`.text` (e.g. LangChain Documents).
    """
    request = {
        "query": query,
        "chunks": [_normalize_chunk(c, i) for i, c in enumerate(retrieved_chunks)],
        "token_budget": token_budget,
        "strategy": strategy,
        "distractor_min_grounding": distractor_min_grounding,
        "link_min_jaccard": link_min_jaccard,
        "redundancy_max_cosine": redundancy_max_cosine,
        "mode": "build",
    }
    out = _call_bridge(request)
    return BuiltContext(text=out["text"], report=ContextReport(out["report"], out["rendered"]))


def analyze_context(
    query: str,
    retrieved_chunks: Sequence[Any],
    *,
    distractor_min_grounding: float = 0.10,
    link_min_jaccard: float = 0.12,
) -> ContextReport:
    """Characterize a retrieved set without modifying it (pure diagnostics)."""
    request = {
        "query": query,
        "chunks": [_normalize_chunk(c, i) for i, c in enumerate(retrieved_chunks)],
        "strategy": "reasoning_preserving",
        "distractor_min_grounding": distractor_min_grounding,
        "link_min_jaccard": link_min_jaccard,
        "mode": "analyze",
    }
    out = _call_bridge(request)
    return ContextReport(out["report"], out["rendered"])


__all__ = ["build_context", "analyze_context", "BuiltContext", "ContextReport", "RedhopError"]
