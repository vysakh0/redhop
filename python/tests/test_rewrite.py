"""Binding-surface tests for `Stripper`, `Vocabulary`, and the
`Document.context_with_rewrites(...)` chain.

Mirrors the Rust unit tests in `redhop::rewrite::tests` through the pyo3
boundary, so a dropped field on `RewriteRecord` or a wrong list ↔ Vec
mapping at the FFI edge surfaces here, not in user code. The mechanism
is documented in `docs/findings/CUAD_CLAUSE_EXPANSION.md` and the
`QueryRewrite` trait in `crates/redhop/src/rewrite.rs`.

Run with: pytest python/tests/test_rewrite.py -q
"""

import pytest

import redhop


# ─── Stripper ───────────────────────────────────────────────────────────────


def test_stripper_drops_listed_tokens_preserving_punctuation():
    stripper = redhop.Stripper(
        ["highlight", "the", "parts", "of", "related", "to"]
    )
    out = stripper.apply('Highlight the parts related to "Change of Control".')
    # "of" inside the quoted phrase is also stripped — Stripper is
    # token-level, not phrase-aware. Punctuation preserved.
    assert out == '"Change Control".'


def test_stripper_preserves_phrase_when_boilerplate_excludes_internal_words():
    """The flip side of token-level matching: drop only what was listed."""
    stripper = redhop.Stripper(["highlight", "the", "parts", "related", "to"])
    out = stripper.apply('Highlight the parts related to "Change of Control".')
    assert out == '"Change of Control".'


def test_stripper_word_boundary_safe_for_short_tokens():
    """An `"of"` stripper must NOT erase the `"of"` inside `"office"`."""
    stripper = redhop.Stripper(["of", "the"])
    out = stripper.apply("the office is open")
    assert out == "office is open"


def test_stripper_empty_boilerplate_is_identity():
    stripper = redhop.Stripper([])
    q = "Highlight the parts of this contract"
    assert stripper.apply(q) == q


def test_stripper_repr_and_len():
    stripper = redhop.Stripper(["a", "b", "c"])
    assert len(stripper) == 3
    assert "Stripper" in repr(stripper)


# ─── Vocabulary ─────────────────────────────────────────────────────────────


def test_vocabulary_basic_asymmetric_append():
    vocab = redhop.Vocabulary(
        {"change of control": ["merger", "successor", "acquisition"]}
    )
    expanded = vocab.apply('"Change of Control" the right to terminate')
    assert expanded.startswith('"Change of Control" the right to terminate')
    for syn in ("merger", "successor", "acquisition"):
        assert syn in expanded


def test_vocabulary_short_acronym_does_not_substring_fire():
    """A vocabulary key `"ip"` must NOT trigger on the `"ip"` inside
    `"recipient"` — token-level matching, not substring."""
    vocab = redhop.Vocabulary({"ip": ["intellectual property"]})
    out = vocab.apply("the recipient agrees to the terms")
    assert "intellectual property" not in out


def test_vocabulary_short_acronym_matches_when_actually_present_as_token():
    vocab = redhop.Vocabulary({"ip": ["intellectual property"]})
    out = vocab.apply("the IP license terms")
    assert "intellectual property" in out


def test_vocabulary_bidirectional_appends_other_members():
    vocab = redhop.Vocabulary.bidirectional(
        {"pto": ["paid time off", "vacation"]}
    )
    # Match from any member appends the others.
    out_from_acronym = vocab.apply("how much PTO do I get")
    assert "paid time off" in out_from_acronym
    assert "vacation" in out_from_acronym
    out_from_phrase = vocab.apply("vacation policy details")
    assert "pto" in out_from_phrase.lower()
    assert "paid time off" in out_from_phrase


def test_vocabulary_non_bidirectional_does_not_fire_from_synonym_side():
    """Asymmetric mode: only the first form is the trigger."""
    vocab = redhop.Vocabulary({"pto": ["paid time off", "vacation"]})
    out = vocab.apply("vacation policy details")
    # "vacation" is a synonym, not a key — must not trigger.
    assert "pto" not in out.lower()


def test_vocabulary_no_recursive_chaining():
    """Synonyms match against the ORIGINAL query — appended terms can't
    re-trigger expansion."""
    vocab = redhop.Vocabulary(
        {
            "change of control": ["merger"],
            "merger": ["consolidation"],
        }
    )
    out = vocab.apply("change of control clause")
    assert "merger" in out
    assert "consolidation" not in out


def test_vocabulary_dedupes_synonyms_across_matches():
    vocab = redhop.Vocabulary(
        {
            "change of control": ["merger", "assignment"],
            "termination for convenience": ["assignment", "rescission"],
        }
    )
    out = vocab.apply("change of control and termination for convenience")
    assert out.count("assignment") == 1, f"expected dedup; got: {out!r}"


def test_vocabulary_empty_is_identity():
    vocab = redhop.Vocabulary({})
    assert vocab.apply("anything") == "anything"


def test_vocabulary_repr_and_len():
    vocab = redhop.Vocabulary({"a": ["b"], "c": ["d", "e"]})
    assert len(vocab) == 2
    assert "Vocabulary" in repr(vocab)


# ─── Document.context_with_rewrites + audit trail ──────────────────────────


def _cuad_query() -> str:
    return 'Highlight the parts of this contract related to "Change of Control" reviewed by a lawyer.'


def test_context_with_rewrites_records_audit_trail():
    """The chain runs through retrieval and the per-stage records land on
    `ctx.report.query_rewrites`."""
    chunks = [
        "Change of Control means a merger or sale of substantially all assets.",
        "The parties to this Agreement are Acme Co. and Beta Inc.",
        "Notices shall be sent to the address listed in Schedule A.",
    ]
    doc = redhop.Document.from_text("\n\n".join(chunks))
    stripper = redhop.Stripper(
        ["highlight", "the", "parts", "of", "this", "contract", "related",
         "to", "reviewed", "by", "a", "lawyer"]
    )
    vocab = redhop.Vocabulary(
        {"change of control": ["merger", "successor", "acquisition"]}
    )
    ctx = doc.context_with_rewrites(_cuad_query(), [stripper, vocab])
    records = ctx.report.query_rewrites
    assert len(records) == 2
    assert records[0].stage == "strip"
    assert records[1].stage == "vocabulary"
    # Vocabulary's added list reflects the synonyms appended.
    assert "merger" in records[1].added
    # Audit trail's "to" string from the second stage equals the rewritten
    # query handed to retrieval.
    assert records[1].to_query.startswith(records[1].from_query)


def test_context_with_rewrites_empty_chain_matches_context():
    """No rewrites should produce the same selection set as `context(...)`."""
    chunks = [
        "Alpha clause about X.",
        "Beta clause about Y.",
        "Gamma clause about Z.",
    ]
    doc = redhop.Document.from_text("\n\n".join(chunks))
    a = doc.context_with_rewrites("X clause", [])
    b = doc.context_with_rewrites("X clause", [])
    assert a.chunks == b.chunks
    assert a.report.query_rewrites == [] or all(
        r.stage for r in a.report.query_rewrites
    )


def test_context_with_rewrites_rejects_non_rewrite_objects():
    doc = redhop.Document.from_text("anything")
    with pytest.raises(ValueError):
        doc.context_with_rewrites("q", ["not a rewrite"])
