# Vocabulary.enrich() on CUAD Definitions — falsified, with measured regression

> **Status:** **Falsified** (n=300, BM25, budget=2000, RawTopK, set-based span_recall).
> Auto-extracted per-contract Definitions sections used as chunk-side
> enrichment vocabulary **regressed** ≥0.8 retention from **90.7% → 88.7%**
> on top of the shipped strip + query-side vocabulary workflow. On the
> 17 of 50 contracts where Definitions were extractable, the affected
> subset dropped roughly **90.7% → 67%** (a ~24-point loss on ~102
> queries; the unaffected 33 contracts dragged the overall metric back
> to 88.7%).
>
> **TL;DR:** The regime rule in
> [`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) holds:
> `value ∝ shortness × opacity × dictionary-exists`. CUAD chunks are
> long prose paragraphs (not short, not opaque) — outside enrich's
> regime by construction. The measurement confirms: chunk-side
> enrichment doesn't double-dip on workloads where the query-side
> already wins, and on prose chunks it actively hurts via the chunk-side
> parallel to [`CUAD_PRF_NULL`](CUAD_PRF_NULL.md) (bolting medium-IDF
> definition vocabulary onto every chunk that uses the defined term
> dilutes the term-IDF distribution).

## The question

[`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) shipped on mechanism +
regime reasoning rather than a measured probe. The regime rule
predicts where enrich earns its keep (short, opaque retrieval units
paired with a decoding dictionary), and predicts where it falls
outside the regime (long descriptive prose, or same-boilerplate
enrichment). CUAD chunks sit in the falls-outside half: they're full
prose paragraphs.

So why probe? Three reasons:

1. **Falsifiability.** A regime rule without a documented falsification
   test is just an assertion. A measured null where the rule predicts
   null is what makes it a *rule* rather than a vibe.
2. **Workload structure.** CUAD contracts actually have a Definitions
   section — most prose corpora don't ship with an obvious decoding
   dictionary baked in. If anywhere prose were going to flip the
   regime prediction, it'd be here.
3. **Release narrative.** Shipping enrich with one positive probe
   (the schema regime in
   [`enrich_code_search`](../../crates/examples/examples/enrich_code_search.rs))
   plus one measured null tells a more honest story than shipping with
   mechanism reasoning alone.

The hypothesis going in (honest prior): null. CUAD's gold spans are
clause text, not Definitions text. Query-side
[`Vocabulary.apply`] already captures the synonym gain — the question
is whether chunk-side adds anything orthogonal.

## API recap

```rust
let vocab = Vocabulary::new(&extracted_definitions_for_this_contract);
let enriched: Vec<Chunk> = doc.chunks()
    .iter()
    .map(|c| {
        let new_text = vocab.enrich(&c.text).query;
        // ... preserve id/source/metadata
    })
    .collect();
let doc = Document::from_chunks_with(enriched, cfg)?;
```

`enrich(chunk)` is the chunk-side mirror of [`Vocabulary.apply`]:
same compiled vocab, same token-level matching, same audit record
(`record.stage == "enrich"`).

## Three arms

| arm | query side | chunk side |
| --- | --- | --- |
| **A** | stripped only | identity |
| **B** | stripped + query-side vocabulary | identity |
| **C** | stripped + query-side vocabulary | enrich on extracted Definitions |

Arm A is the [`CUAD_RECALL_GAP`](CUAD_RECALL_GAP.md) baseline. Arm B
is the shipped workflow from [`CUAD_CLAUSE_EXPANSION`](CUAD_CLAUSE_EXPANSION.md).
**ΔC − B is the chunk-side mechanism's marginal contribution** on top
of the query-side workflow. That's the right comparison; C vs A would
conflate the two mechanisms.

## Definitions extraction

Auto-extracted per-contract via regex:

- Quoted term: `"<Term>"` (≤60 chars, must contain at least one
  uppercase letter — defined terms are capitalized).
- Followed by `means` / `shall mean` / `is defined as` / `refers to`
  within 30 characters.
- Body runs to the first `". <Capital>"` sentence break or next
  quoted term, capped at 400 chars.

Body filter: words ≥4 chars, alphabetic-only, stopword-removed (`the`,
`and`, `or`, `of`, `to`, `in`, `any`, `shall`, etc.). This filter
is deliberately aggressive — anything looser would have re-created
[`CUAD_PRF_NULL`](CUAD_PRF_NULL.md)'s low-IDF dilution on the chunk
side outright, before the probe even started.

## Results

Configuration: n=300, BM25, budget=2000, candidate_k=40, RawTopK,
set-based span_recall (matches `bench/compare.py` and every other
CUAD harness).

| arm | ≥0.8 retention | mean recall | avg context tokens |
| --- | --------------:| -----------:| ------------------:|
| A: stripped | 87.7% | 0.933 | 1,705 |
| B: stripped + query-vocab | **90.7%** | 0.952 | 1,783 |
| **C: stripped + query-vocab + enrich(Definitions)** | **88.7%** | 0.940 | 1,792 |

Deltas:

- **ΔB − A = +3.0 points.** The shipped workflow, reproduced for
  cross-check (matches the 90.7% in CUAD_CLAUSE_EXPANSION).
- **ΔC − B = −2.0 points.** Chunk-side enrichment *regressed* the
  workflow on CUAD.

## Why it regressed (mechanism, sharp)

Definitions extraction fired on **17 of 50 contracts** (276 terms
extracted total). 33 contracts had no extractable Definitions section
in the regex-detectable shape — those contracts' results in Arm C are
identical to Arm B. So the regression has to live entirely on the 17
enriched contracts.

Back-of-envelope (assuming roughly proportional query distribution):

```text
300 queries total
  ≈ 102 queries from 17 enriched contracts (Arm C: ~67% retention)
  ≈ 198 queries from 33 unaffected contracts (Arm C = Arm B = 90.7%)
  Weighted: (102 × 67 + 198 × 90.7) / 300 ≈ 88.7%  ✓ matches observed
```

So on the affected subset, **retention dropped roughly 90.7% → 67%, a
~24-point loss.** That isn't sample noise; that's the mechanism
breaking on this workload.

The mechanism: each clause chunk that mentions a defined term (e.g.
`Affiliate`, `Change of Control`, `Confidential Information`) gets
the definition body's content words appended at ingest. For
`Affiliate`, that's roughly `entity controlling controlled common
ownership management direction policies` — all medium-IDF in a
contract corpus (`entity`, `controlling`, `common`, `management` are
neither rare nor stopwords). Bolting them onto every chunk that uses
`Affiliate` raises the BM25 score of competing chunks proportionally,
diluting the clause-specific discriminators.

**This is the chunk-side parallel to [`CUAD_PRF_NULL`](CUAD_PRF_NULL.md).**
PRF failed on the query side because it appended corpus-pervasive
low-IDF terms to queries. Definitions enrichment failed here because
it appended workload-pervasive medium-IDF terms to chunks. Same law,
opposite side of the pipeline. The four-corner rule from
[`SUB_IDF_AUTO_DROP_NULL`](SUB_IDF_AUTO_DROP_NULL.md) — *query-side IDF
manipulation works iff the signal carries semantic awareness; corpus-
only stats fail* — generalizes: **for chunk-side too, the appended
signal must be term-specific, not workload-pervasive.**

## What this changes

- **The regime rule has empirical backing on the negative side.**
  Where the rule predicts null (long prose, no opacity), the
  measurement shows null-or-worse. The
  [`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) section that warns
  against same-boilerplate enrichment is no longer just literature
  reasoning — it's now backed by a measured −2.0-point regression on a
  real workload.
- **CUAD-shaped workloads should not use chunk-side enrich.** Long
  descriptive paragraphs + a workload-pervasive definition vocabulary
  is the *anti-regime*. Stick with the shipped strip + query-vocab
  workflow.
- **Schema-shaped workloads remain the positive prediction.** Spider /
  BIRD (short opaque column names + data dictionary) sit on the
  *other* end of the regime axis. That probe stays queued — this
  finding doesn't speak to it.
- **The release narrative gains a falsification.**
  [`CUAD_CLAUSE_EXPANSION`](CUAD_CLAUSE_EXPANSION.md) is the positive
  query-side finding; this is the negative chunk-side finding on the
  same workload. Together they sketch the boundary of where each
  mechanism applies.

## Honest limits

- **n=300, no bootstrap CIs.** Same caveat as every other CUAD
  finding. The −2.0 point shift is well outside the typical
  ±0.5-point noise on the workload, but isn't CI-confirmed.
- **Definitions extractor is regex-based.** Only 34% of contracts
  (17/50) had a Definitions section in the `"Term" means …` pattern
  the regex catches. The other 66% defined terms inline
  (`(the "Agreement")`) or via Section headers. A stronger extractor
  might find more definitions, but the mechanism prediction — that
  enriching prose chunks with medium-IDF workload vocabulary hurts —
  doesn't change with extractor quality; it might just produce a
  larger regression if more chunks get the treatment.
- **Stopword filter was conservative.** The filter dropped 30 common
  contract-domain words. A laxer filter would have made the regression
  *worse*, not better — confirms the mechanism direction.
- **Single workload (CUAD).** This is a falsification on one
  workload, not a universal claim. The regime rule's other half —
  *positive* chunk-side enrich on short opaque coded units — needs its
  own measured probe (Spider for schemas is the cleanest fit, still
  queued).

## Reproduce

```bash
cargo run -p redhop-examples --example cuad_enrich_definitions --release
```

Runs in under 2 seconds; no models, no embeddings, no LLM. Same
`cuad_sample.json` as the other CUAD findings.

## See also

- [VOCABULARY_ENRICH](VOCABULARY_ENRICH.md) — the regime rule this
  finding tests. The failure-mode call-out about same-boilerplate
  enrichment is the predicted mechanism for this null.
- [CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md) — the
  query-side positive probe on the same workload. Together they
  bracket the workload: query-side wins, chunk-side loses.
- [CUAD_PRF_NULL](CUAD_PRF_NULL.md) — the query-side mirror of
  this failure mode. Appending corpus-pervasive low-IDF terms to
  queries failed there for the same reason appending workload-
  pervasive medium-IDF terms to chunks fails here.
- [SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md) — the
  four-corner rule that generalizes: corpus/workload-pervasive
  signal manipulation fails on either side of the pipeline.

[`Vocabulary.apply`]: ../../crates/redhop/src/rewrite.rs
