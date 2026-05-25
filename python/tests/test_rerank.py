"""Dense-rerank tier tests — require an ONNX-enabled build and a local model.

These are skipped cleanly unless the model env vars point at real ONNX files, so
the default (lexical) wheel and CI without models still pass. To run them:

    # build the onnx-enabled extension into your venv
    maturin develop -m python/Cargo.toml --features onnx

    # symmetric model (BGE/MiniLM/GTE) — CLS or mean pooled
    export REDHOP_BGE_MODEL=/path/bge/model.onnx
    export REDHOP_BGE_TOKENIZER=/path/bge/tokenizer.json

    # optional: an asymmetric E5-style model (mean pooled, query:/passage: prefixes)
    export REDHOP_E5_MODEL=/path/e5/model.onnx
    export REDHOP_E5_TOKENIZER=/path/e5/tokenizer.json

    pytest python/tests/test_rerank.py -v
"""

import os

import pytest

import redhop

# A tiny HR corpus where the answer shares no surface words with the query
# ("leave" vs "terminated") — the case BM25 alone misses and dense rerank recovers.
TEXT = (
    "The employee was terminated for cause and a severance review followed. "
    "The annual budget review was approved by the board after a long discussion. "
    "The cafeteria introduced a new vegetarian menu on Fridays. "
    "Quarterly revenue rose twelve percent year over year."
)
QUERY = "why did the employee leave the company?"

BGE_MODEL = os.environ.get("REDHOP_BGE_MODEL")
BGE_TOK = os.environ.get("REDHOP_BGE_TOKENIZER")

# E5 (asymmetric) — fall back to the repo's local bench export if present, so the
# dense path actually runs here without any env setup.
_BENCH_E5 = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "models", "e5-small-onnx")
E5_MODEL = os.environ.get("REDHOP_E5_MODEL") or os.path.join(_BENCH_E5, "model.onnx")
E5_TOK = os.environ.get("REDHOP_E5_TOKENIZER") or os.path.join(_BENCH_E5, "tokenizer.json")

_have_bge = bool(BGE_MODEL and BGE_TOK and os.path.exists(BGE_MODEL))
_have_e5 = bool(E5_MODEL and E5_TOK and os.path.exists(E5_MODEL))


def test_invalid_retrieval_mode_errors_clearly():
    """The retrieval tier was renamed: 'rerank'/'dense' are gone, and a bogus
    mode must error listing the valid options."""
    for bad in ("rerank", "dense", "bogus"):
        with pytest.raises(Exception) as e:
            redhop.Document.from_text(TEXT, retrieval=bad)
        msg = str(e.value).lower()
        assert "lexical" in msg and "hybrid" in msg and "semantic" in msg


def test_lexical_default_still_works():
    doc = redhop.Document.from_text(TEXT, chunk_size=16)
    assert doc.context("who was terminated?").text().strip()


@pytest.mark.skipif(not _have_bge, reason="REDHOP_BGE_MODEL not set")
def test_hybrid_bge_symmetric():
    doc = redhop.Document.from_text(
        TEXT,
        chunk_size=16,
        retrieval="hybrid",
        embedder_model=BGE_MODEL,
        embedder_tokenizer=BGE_TOK,
        embedder_dim=384,
        embedder_pooling="cls",
    )
    text = doc.context(QUERY).text().lower()
    assert text.strip()
    assert "terminated" in text  # the paraphrase-matched chunk surfaced


@pytest.mark.skipif(not _have_e5, reason="no E5 model (set REDHOP_E5_MODEL or build bench/models/e5-small-onnx)")
@pytest.mark.parametrize("mode", ["hybrid", "semantic"])
def test_dense_e5_asymmetric_prefixes(mode):
    doc = redhop.Document.from_text(
        TEXT,
        chunk_size=16,
        retrieval=mode,
        embedder_model=E5_MODEL,
        embedder_tokenizer=E5_TOK,
        embedder_dim=384,
        embedder_pooling="mean",
        embedder_query_prefix="query: ",
        embedder_passage_prefix="passage: ",
    )
    text = doc.context(QUERY).text().lower()
    assert text.strip()
    assert "terminated" in text


@pytest.mark.skipif(not _have_e5, reason="no E5 model available")
def test_unknown_pooling_rejected():
    with pytest.raises(Exception):
        redhop.Document.from_text(
            TEXT,
            retrieval="hybrid",
            embedder_model=E5_MODEL,
            embedder_tokenizer=E5_TOK,
            embedder_pooling="bogus",
        )
