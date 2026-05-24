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
E5_MODEL = os.environ.get("REDHOP_E5_MODEL")
E5_TOK = os.environ.get("REDHOP_E5_TOKENIZER")

_have_bge = bool(BGE_MODEL and BGE_TOK and os.path.exists(BGE_MODEL))
_have_e5 = bool(E5_MODEL and E5_TOK and os.path.exists(E5_MODEL))


def test_rerank_without_embedder_errors_clearly():
    """retrieval='rerank' with no model path errors with a helpful message,
    on any build (lexical wheel: 'onnx build'; onnx wheel: 'embedder_model')."""
    with pytest.raises(Exception) as e:
        redhop.Document.from_text(TEXT, retrieval="rerank")
    msg = str(e.value).lower()
    assert "onnx" in msg or "embedder" in msg


def test_lexical_default_still_works():
    doc = redhop.Document.from_text(TEXT, chunk_size=16)
    assert doc.context("who was terminated?").text().strip()


@pytest.mark.skipif(not _have_bge, reason="REDHOP_BGE_MODEL not set")
def test_dense_rerank_bge_symmetric():
    doc = redhop.Document.from_text(
        TEXT,
        chunk_size=16,
        retrieval="rerank",
        embedder_model=BGE_MODEL,
        embedder_tokenizer=BGE_TOK,
        embedder_dim=384,
        embedder_pooling="cls",
    )
    text = doc.context(QUERY).text().lower()
    assert text.strip()
    assert "terminated" in text  # the paraphrase-matched chunk surfaced


@pytest.mark.skipif(not _have_e5, reason="REDHOP_E5_MODEL not set")
def test_dense_rerank_e5_asymmetric_prefixes():
    doc = redhop.Document.from_text(
        TEXT,
        chunk_size=16,
        retrieval="rerank",
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


def test_unknown_pooling_rejected():
    """Only meaningful on an onnx build; lexical build errors earlier (also fine)."""
    with pytest.raises(Exception):
        redhop.Document.from_text(
            TEXT,
            retrieval="rerank",
            embedder_model=BGE_MODEL or "x.onnx",
            embedder_tokenizer=BGE_TOK or "x.json",
            embedder_pooling="bogus",
        )
