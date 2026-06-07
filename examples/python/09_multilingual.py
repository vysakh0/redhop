"""09 · Multilingual analyzer — `language="german"` / "french" / ...

Real-world scenario:
    A pharmaceutical company has internal policy documents in
    English, German, and French (one corpus per region). They want
    BM25-quality retrieval on each — which means the tokenizer needs
    to understand each language's morphology. German `Bücher` should
    find a chunk that only contains `Buch`; French `manger` should
    find a chunk with `mange`. The default English analyzer would
    miss both because Snowball Porter2 (English) doesn't stem
    foreign words correctly.

    RedHop ships an 18-language analyzer matrix (the Snowball Porter2
    family). Pass `language="german"` to `Document.from_*` and the
    whole pipeline — chunking-side tokenization, BM25 stemming, and
    the grounding scorer's term extraction — uses the German
    analyzer. Each layer agrees on what counts as "the same term."

What this demonstrates:
    - `Document.from_text(text, language="german")` and the same on
      `from_chunks`, `from_file`, `from_folder`.
    - The 18 supported languages: arabic, danish, dutch, english,
      finnish, french, german, greek, hungarian, italian, norwegian,
      portuguese, romanian, russian, spanish, swedish, tamil,
      turkish.
    - Why the morphology unification matters: a German query for
      `Buch` lands on a chunk containing only the plural `Bücher`
      because the analyzer stems both to the same root.
    - That an unknown language string ERRORS (no silent fallback to
      English) — caught by docs/findings/MULTILINGUAL_ANALYZER.md.

Run:
    python examples/python/09_multilingual.py
"""

import redhop

GERMAN_CORPUS = [
    redhop.Chunk("Ich habe viele Bücher im Regal stehen.", id="de-1"),
    redhop.Chunk("Ein Kind spielt fröhlich im Garten.", id="de-2"),
    redhop.Chunk("Der Hund läuft schnell durch den Park.", id="de-3"),
]

FRENCH_CORPUS = [
    redhop.Chunk("Il aime manger des pommes chaque matin.", id="fr-1"),
    redhop.Chunk("Le chien court dans la rue très vite.", id="fr-2"),
    redhop.Chunk("Les enfants jouent au parc le weekend.", id="fr-3"),
]


def demo(label: str, corpus: list[redhop.Chunk], query: str, language: str) -> None:
    doc = redhop.Document.from_chunks(corpus, language=language)
    ctx = doc.context(query)
    print(f"─── {label} ────────────────────────────────")
    print(f"  language={language!r}, query={query!r}")
    if ctx.citations:
        print(f"  top hit: {ctx.citations[0]['text']}")
    else:
        print("  (no hits)")
    print()


def main() -> None:
    # German Snowball: "Buch" (singular) should reach a chunk containing
    # only "Bücher" (plural). Both stem to roughly "buch".
    demo("Arm A · German morphology", GERMAN_CORPUS, query="Buch", language="german")

    # French Snowball: "manger" (infinitive) should reach a chunk with
    # only "mange" (1st-person conjugation). Both stem to roughly "mang".
    demo("Arm B · French morphology", FRENCH_CORPUS, query="manger", language="french")

    # Counter-example: same German corpus, default English analyzer.
    # The English analyzer doesn't know German stems, so "Buch" doesn't
    # reach "Bücher" — they're treated as different tokens.
    demo(
        "Arm C · German corpus + English analyzer (the bug it prevents)",
        GERMAN_CORPUS,
        query="Buch",
        language="english",
    )

    # Unknown language: deliberate error rather than silent English
    # fallback. The error message lists the 18 supported languages so
    # the user knows what's available.
    print("─── Arm D · Unknown language string ──────────────")
    try:
        redhop.Document.from_chunks(GERMAN_CORPUS, language="germann")
        print("  (oops — should have raised)")
    except ValueError as e:
        print(f"  ValueError: {str(e)[:140]}…")
    print()

    print("─── How to read this ─────────────────────────────")
    print("Arm A and B: language=… routes the whole pipeline through")
    print("  the right Snowball stemmer, so morphological variants of")
    print("  the same word unify across queries and chunks.")
    print("Arm C: same German corpus + default English analyzer →")
    print("  miss. Picking the wrong language SILENTLY (e.g.")
    print("  forgetting to pass language=) is the real failure mode.")
    print("Arm D: unknown language strings ERROR with the supported")
    print("  list so a typo'd 'germann' is caught at construction,")
    print("  not after retention has silently dropped 10 points.")
    print()
    print("Validated cross-language by docs/findings/MULTILINGUAL_ANALYZER.md.")


if __name__ == "__main__":
    main()
