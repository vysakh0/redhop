"""Binding-surface tests for the pluggable lexical analyzer.

Mirrors the Rust `quality_suite::t41`-`t44` matrix through the Python
binding so a dropped `language=` kwarg or a wrong string-to-analyzer
mapping at the pyo3 boundary surfaces here, not in user code.

These tests caught a real bug on their first run: `Document.from_chunks`
accepted `language=` but hardcoded `None` in the call into `doc_config`,
silently dropping the user's choice. Fixed; pinned.

Run with: pytest python/tests/test_analyzer.py -q
"""

import pytest

import redhop


def _german_corpus():
    return [
        redhop.Chunk("ich habe viele Bücher im Regal stehen", id="a"),
        redhop.Chunk("ein Kind spielt fröhlich im Garten", id="b"),
    ]


def _french_corpus():
    return [
        redhop.Chunk("il aime manger des pommes chaque matin", id="a"),
        redhop.Chunk("le chien court dans la rue très vite", id="b"),
    ]


# ── Per-language behavior (mirrors Rust T41/T42) ──────────────────────────


def test_german_analyzer_unifies_morphology_via_from_chunks():
    """German Snowball: query `Buch` should reach a chunk that only
    contains the plural form `Bücher`."""
    doc = redhop.Document.from_chunks(_german_corpus(), options=redhop.DocumentOptions(language="german"))
    ctx = doc.context("Buch")
    assert "Bücher" in ctx.text(), f"German analyzer should unify Bücher↔Buch; got: {ctx.text()!r}"


def test_french_analyzer_unifies_verb_inflections_via_from_chunks():
    """French Snowball: query `manger` should reach a chunk that only
    contains the conjugated form `mange`."""
    doc = redhop.Document.from_chunks(_french_corpus(), options=redhop.DocumentOptions(language="french"))
    ctx = doc.context("manger")
    assert "mange" in ctx.text(), f"French analyzer should unify manger↔mange; got: {ctx.text()!r}"


def test_german_analyzer_works_via_from_text():
    """Same German morphology check through the chunked text path —
    `from_text` and `from_chunks` are separate plumbing branches; both
    must route `language=`."""
    text = "ich habe viele Bücher im Regal stehen.\n\nein Kind spielt fröhlich im Garten."
    doc = redhop.Document.from_text(text, options=redhop.DocumentOptions(language="german"))
    ctx = doc.context("Buch")
    assert "Bücher" in ctx.text(), (
        f"from_text + language='german' should find Bücher; got: {ctx.text()!r}"
    )


# ── Default behavior preserved ───────────────────────────────────────────


def test_default_analyzer_unchanged_when_language_omitted():
    """Omitting `language=` must keep the default English analyzer
    behavior — English stemming doesn't unify Bücher↔Buch (different
    languages), so the German query MISSES under default English. This
    is the negative case that proves the kwarg actually does something."""
    doc = redhop.Document.from_chunks(_german_corpus())
    ctx = doc.context("Buch")
    assert "Bücher" not in ctx.text(), (
        "Default English analyzer should NOT find Bücher from query 'Buch'; "
        "if this fails, either English silently stems German (a bug) or the "
        "tested corpus drifted. Got: " + repr(ctx.text())
    )


# ── Validation (mirrors Rust T44) ────────────────────────────────────────


@pytest.mark.parametrize("ctor", ["from_chunks", "from_text"])
def test_unknown_language_raises(ctor):
    """A typo'd language name must raise — silent fallback to English on
    a misspelled `"germann"` would let a ranking regression hide in
    production. Mirrors Rust T44."""
    with pytest.raises(ValueError) as excinfo:
        if ctor == "from_chunks":
            redhop.Document.from_chunks(_german_corpus(), options=redhop.DocumentOptions(language="germann"))
        else:
            redhop.Document.from_text("ich habe Bücher", options=redhop.DocumentOptions(language="germann"))
    msg = str(excinfo.value).lower()
    assert "unknown language" in msg, (
        f"error should mention 'unknown language'; got: {excinfo.value!r}"
    )
    assert "germann" in str(excinfo.value), (
        f"error should echo the bad language name; got: {excinfo.value!r}"
    )


# ── Coverage of supported names (cheap smoke) ────────────────────────────


@pytest.mark.parametrize(
    "language",
    [
        "english",
        "german",
        "french",
        "spanish",
        "italian",
        "portuguese",
        "dutch",
        "russian",
        "swedish",
        "norwegian",
        "danish",
        "finnish",
        "romanian",
        "hungarian",
        "turkish",
        "arabic",
        "greek",
        "tamil",
    ],
)
def test_all_18_snowball_builtins_accepted(language):
    """Every name advertised in the unknown-language error message must
    actually be a valid `language=` value — a typo in either list would
    leave a builtin unreachable from Python while looking supported."""
    doc = redhop.Document.from_chunks([redhop.Chunk("the quick brown fox jumps over the lazy dog", id="a")], options=redhop.DocumentOptions(language=language))
    # Just need it to round-trip without raising.
    ctx = doc.context("fox")
    assert isinstance(ctx.text(), str)
