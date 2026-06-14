"""14 · Catalog search — short, noisy queries over a near-duplicate catalog.

Real-world scenario:
    A corner-store ordering assistant takes short, messy product
    requests ("liberty root beer", "summit cola", and plenty of typos
    like "1iberty"). The catalog is a near-duplicate lattice: one brand
    has the same product at several sizes and prices that differ by a
    token or two. Three things break here that don't break on prose QA,
    and each has a lever.

What this demonstrates:
    - `language="char_ngram"` — the subword typo tier. A transcription
      typo ("1iberty") still matches via shared character n-grams, with
      no model. Word-token BM25 scores it at zero.
    - `bm25_field_weights=[text, source, heading]` — per-field BM25
      boosts, a domain lever for near-duplicate corpora (the default,
      equal weight, is bit-for-bit unchanged).
    - `evaluate(..., gold_families=[...])` -> `set_coverage` — a catalog
      query maps to a SET (all sizes of a product). recall@k hides a
      half-retrieved family; set_coverage catches it.

    Honest framing (docs/findings/CATALOG_REGIME.md): char-ngram is a
    recall booster, not a drop-in (pair it with word-BM25). Field
    weights help only when the boosted field separates the answer from
    its near-duplicates, so sweep, don't assume.

Run:
    pip install redhop
    python examples/python/14_catalog_search.py
"""

import redhop

# A small American convenience-store catalog. Each SKU is one chunk:
# `text` is the full product line; `heading` (metadata) is the
# brand+product key a field-weight boost can amplify, and the label
# `citations` surface back.
CATALOG = [
    # (sku id, brand+product key, full text)
    ("summit-cola-12", "Summit Cola", "Summit Cola 12 oz 1.49"),
    ("summit-cola-20", "Summit Cola", "Summit Cola 20 oz 1.99"),
    ("summit-cola-2l", "Summit Cola", "Summit Cola 2 liter 2.49"),
    ("summit-diet-12", "Summit Diet Cola", "Summit Diet Cola 12 oz 1.49"),
    ("summit-diet-20", "Summit Diet Cola", "Summit Diet Cola 20 oz 1.99"),
    ("liberty-rb-12", "Liberty Root Beer", "Liberty Root Beer 12 oz 1.49"),
    ("liberty-rb-20", "Liberty Root Beer", "Liberty Root Beer 20 oz 1.99"),
    ("eagle-bbq-2", "Eagle Potato Chips", "Eagle Potato Chips BBQ 2 oz 1.29"),
    ("eagle-bbq-8", "Eagle Potato Chips", "Eagle Potato Chips BBQ 8 oz 3.49"),
    ("eagle-salt-2", "Eagle Potato Chips", "Eagle Potato Chips Salted 2 oz 1.29"),
    ("pioneer-jerky-3", "Pioneer Beef Jerky", "Pioneer Beef Jerky Original 3 oz 5.99"),
    ("coastal-mix-6", "Coastal Trail Mix", "Coastal Trail Mix 6 oz 4.29"),
]


def build(language=None, bm25_field_weights=None):
    chunks = [
        redhop.Chunk(text, id=sku, source="catalog", metadata={"heading": heading})
        for (sku, heading, text) in CATALOG
    ]
    opts = redhop.DocumentOptions(language=language, bm25_field_weights=bm25_field_weights)
    return redhop.Document.from_chunks(chunks, options=opts)


def products(ctx):
    """Distinct brand+product labels in the assembled context, in order
    (citations carry the `heading` we set per chunk)."""
    seen = []
    for c in ctx.citations:
        h = c.get("heading")
        if h and h not in seen:
            seen.append(h)
    return seen


def main() -> None:
    # ── 1. Transcription typo: char-ngram recovers what word-BM25 drops ──
    # A realistic noisy order: the brand is typo'd ("1iberty") AND the
    # product is run together ("rootbeer"), so word-BM25 has no exact token
    # to match. char-ngram bridges both via shared character n-grams.
    print("1) Typo recovery — query: '1iberty rootbeer'\n")
    word = build(language="raw")          # default word-token analyzer
    ngram = build(language="char_ngram")  # subword typo tier
    q = "1iberty rootbeer"
    print(f"   word-BM25  found : {products(word.context(q))}")
    ngram_found = products(ngram.context(q))
    print(f"   char-ngram found : {ngram_found}")
    print(f"   -> char-ngram recovered Liberty Root Beer despite the typo: "
          f"{'Liberty Root Beer' in ngram_found}\n")

    # ── 2. Per-field weighting is a knob (default = equal weight) ─────────
    print("2) Field weights — boost the brand/product 'heading' field 2x\n")
    boosted = build(language="char_ngram", bm25_field_weights=[1.0, 1.0, 2.0])
    print(f"   'summit cola' -> {products(boosted.context('summit cola'))}")
    print("   (a domain lever: sweep on your own gold set, it is not a")
    print("    guaranteed lift; see docs/findings/CATALOG_REGIME.md)\n")

    # ── 3. set_coverage: did we retrieve the WHOLE variant family? ───────
    print("3) Set coverage — 'summit cola' should return ALL its sizes\n")
    family = ["summit-cola-12", "summit-cola-20", "summit-cola-2l"]
    ctx = ngram.context("summit cola")
    r = redhop.evaluate("summit cola", ctx, gold_families=[family])
    print(f"   products offered : {products(ctx)}")
    print(f"   set_coverage     : {r.set_coverage}   (1.0 = whole family offerable)")
    print(f"   context_recall   : {r.context_recall}")
    print("   recall@k can read fine while a family is half-retrieved;")
    print("   set_coverage is the metric a disambiguation UX should gate on.")


if __name__ == "__main__":
    main()
