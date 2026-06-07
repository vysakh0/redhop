# `Vocabulary.enrich` — chunk-side rewrite primitive (asymmetric evidence: measured negative, mechanism-predicted positive)

> **Status:** **Shipped as a primitive on mechanism reasoning** —
> [`Vocabulary::enrich`](../../crates/redhop/src/rewrite.rs) is the
> symmetric to query-side [`QueryRewrite::apply`] (appends a compiled
> dictionary's synonyms to chunk text at ingest time, audit-trail
> intact). Its **mechanism** is well-established in IR (the doc2query
> family). Its **measured evidence is asymmetric**:
>
> - **Measured negative:** 1 datapoint.
>   [CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md)
>   regressed retention by **−2.0 pts** vs the 90.7% workflow
>   baseline (~24-point loss on the 17/50 contracts where enrichment
>   fired). The chunk-side parallel to CUAD_PRF_NULL.
> - **Measured positive:** 0 datapoints. The "where it earns its
>   keep" sections below describe **mechanism-predicted** use cases
>   — schema retrieval, code/API search, error codes, clinical
>   abbreviations. None of these have been measured by RedHop yet.
>   The IR literature on doc2query is the closest external
>   evidence, but it tests different setups.
> - **Queued:** Spider / BIRD as the positive probe for the schema
>   regime — the cleanest predicted fit. Not yet run.
>
> **What this means in practice.** If your retrieval unit is a long
> prose paragraph, enrich is *predicted to fail* and *was measured to
> fail* on at least one such workload (CUAD). If your retrieval
> unit is a short opaque coded token (schema column, error code, API
> symbol), enrich is *predicted to help* but **not yet measured to
> help on RedHop's eval rigs**. A/B against your own corpus before
> adopting; treat the regime rule below as a *hypothesis to test*,
> not a guarantee.
>
> **TL;DR:** Same compiled vocab as query-side `apply`, applied at
> ingest time to chunks. Use only when your retrieval units are
> short and opaque, you have a decoding dictionary, and you're
> willing to A/B-verify the lift on your own data.

## The regime hypothesis

A compact, testable predictor — **a hypothesis the docs/code haven't
yet measured a positive case for**:

```text
expected value ∝ shortness × opacity × (dictionary exists)
```

- **Shortness.** A long prose paragraph already has plenty of surface
  area for BM25 (or any lexical scorer) to match against; adding more
  signal is *predicted to help marginally at best, hurt at worst*.
  CUAD_ENRICH_DEFINITIONS_NULL measured the "hurt at worst" half of
  that prediction (−2.0pt). A bare token — `emp_compensation`,
  `ERR_4012`, `MI`, `usrSvc` — has almost no matchable surface, so
  enrichment *should* give it meaning. The "should" is unmeasured.
- **Opacity.** The retrieval unit is a coded identifier: abbreviated,
  domain-internal, machine-generated. Its surface form has nothing in
  common with how users would phrase their question. Mechanism
  prediction; unmeasured at write time.
- **Dictionary exists.** The cost of building a vocabulary from scratch
  may exceed the benefit; the mechanism predicts wins when you
  *already* have a decoding dictionary (data dictionary, glossary,
  abbreviation list, API reference) and the missing piece is wiring it
  into retrieval. Unmeasured at write time.

Where the rule predicts failure — long descriptive prose — we have
measured evidence (CUAD_ENRICH_DEFINITIONS_NULL: −2.0pt). Where it
predicts success — short opaque tokens + dictionary — we don't yet.
The rule is testable; pending its positive validation, treat it as
**design rationale, not a performance claim**.

## How it differs from query-side `apply` (mechanism only)

Query-side [`apply`] is surgical: you enumerate the synonyms you
expect users to use. It can't help a query phrasing you didn't
foresee. Chunk-side `enrich` is a different mechanism: it describes
the *content* once and (the mechanism predicts) serves future queries
— including phrasings you never listed — because it raises the
content's surface area rather than patching specific holes.

They're different jobs, not strictly comparable on a single axis:

|  | `apply` (query-side) | `enrich` (chunk-side) |
| --- | --- | --- |
| **Direction** | rewrites the query | rewrites the chunks |
| **Time** | query time | ingest time |
| **Covers** (mechanism) | query reformulations you anticipated | (predicted) unanticipated queries against content you described |
| **Cost** | per-query, small | one-time at ingest, larger |
| **Audit trail** | `report.query_rewrites` | per-chunk `RewriteRecord` from `vocab.enrich(...)` |
| **Measured on RedHop** | yes — `CUAD_CLAUSE_EXPANSION` (+3.0pt) | partially — `CUAD_ENRICH_DEFINITIONS_NULL` (−2.0pt on a workload outside the regime). Positive case unmeasured. |
| **Mechanism prediction strongest when** | you can enumerate user phrasings | you can describe what your content is *and* your retrieval units are short/opaque |

The "Covers" cell on the enrich column says *predicted to* — not
*measured to*. Don't read past that. Use `apply` based on its
measured CUAD positive; use `enrich` based on the mechanism
prediction plus your own A/B on your corpus.

## Use cases where the mechanism prediction is strongest (none measured)

The following are **regime-predicted** to benefit. None of them have
been measured on RedHop eval rigs at the time of writing. A/B
against your own corpus before committing.

**Predicted-strong — short, opaque units + a real dictionary:**

- **Schemas (text-to-SQL).** Column names like `emp_compensation`,
  `ord_dt`. Data dictionaries are standard practice. Maximally short +
  opaque on the regime axis. (Public benchmarks: Spider, BIRD —
  *queued, not yet run by RedHop.* This is the natural positive
  probe.)
- **Code / API search.** Symbols (`usrSvc`, `calcAmt`), endpoints
  (`POST /v1/payment_intents`). Mechanism predicts the strongest
  effect on undocumented / legacy codebases; where docstrings already
  exist, the documentation already supplies the surface area.
  Synthetic demo:
  [`enrich_code_search`](../../crates/examples/examples/enrich_code_search.rs)
  (hand-crafted corpus + dictionary, *not* a measurement).
- **Logs / observability.** Error and event codes (`ERR_4012`,
  `evt_chrgbck`). Mechanism predicts enrichment lets on-call
  engineers search natural-language symptoms (`"payment failed"`) and
  reach the right runbook section.
- **Clinical / medical notes.** Abbreviation soup (`MI`, `SOB`,
  `HTN`, `ICD-10` codes). Medical glossaries → lay/NL queries match
  cryptic chart notes. Mechanism prediction; domain-specialized.
- **Acronym-heavy enterprise wikis.** Codenames, team abbreviations,
  internal tool names. Mechanism predicts enrichment lets full-name
  queries reach codename documentation.

**Predicted-medium — fits the regime but the dictionary gap is often
already filled by the corpus itself:**

- **Product / parts catalogs.** SKUs and part numbers — predicted
  effect higher on industrial/technical catalogs; consumer ones
  usually already have rich descriptions.
- **Financial / tickers.** `AAPL` → "Apple Inc.", accounting
  line-item codes.
- **Scientific / bio.** Gene symbols (`TP53`), chemical identifiers,
  field abbreviations.

## Where we *measured* failure (or the mechanism predicts it)

- **Legal contracts — defined-terms enrichment.** The Definitions
  section is a glossary the document ships with itself, so this
  *seemed* like a fit. **Measured: it isn't.**
  [CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md)
  measured a −2.0pt regression on top of the 90.7% workflow baseline;
  ~24-point loss on the 17/50 contracts where Definitions extraction
  fired. The chunks are long prose paragraphs, so the regime rule
  predicts failure (which it does); and the appended definition
  vocabulary is workload-pervasive enough to dilute the term-IDF
  distribution. **If you have a legal-contracts corpus, don't reach
  for enrich.**
- **Normal prose corpora generally.** Mechanism predicts that if
  your chunks are already descriptive paragraphs, enrichment adds
  nothing helpful and risks dilution. CUAD is the measured datapoint
  on this; the prediction generalizes.
- **Same-boilerplate enrichment.** Bolt the *identical* description
  onto every chunk (e.g. "this is a function" on every code chunk)
  and you re-create the low-IDF dilution falsified twice in
  [CUAD_PRF_NULL](CUAD_PRF_NULL.md) and
  [SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md), now also on
  the chunk side ([CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md)).
  Enrichment must add *term-specific* signal, not repeated filler.
  Under dense retrieval the failure mode is vector-collapse instead
  of IDF dilution, but the law is the same.
- **No dictionary exists.** If you'd have to build the descriptions
  from scratch, the cost may exceed the benefit. Sometimes the right
  fix is upstream — annotate the source — not enrich.

## API

### Rust

```rust
use redhop::{Document, Vocabulary};

let vocab = Vocabulary::new(&[
    ("usrSvc",  &["user service", "signup", "account creation"][..]),
    ("calcAmt", &["calculate amount", "billing total"]),
]);

let raw: Vec<String> = read_my_corpus();
let enriched: Vec<String> = raw
    .into_iter()
    .map(|c| vocab.enrich(&c).query) // .record carries the audit trail
    .collect();

let mut doc = Document::from_chunks(into_chunks(enriched))?;
let ctx = doc.context_with("how do we handle account creation", None, None)?;
```

`enrich(chunk) -> RewriteResult { query, record }` — same shape as
[`QueryRewrite::apply`]. The `record.stage` field is `"enrich"` (vs
`"vocabulary"` for the query-side call), so downstream consumers can
tell at-a-glance which side of the pipeline a record came from. The
matching is *identical* to query-side: token-level through the same
analyzer, with the substring-safety property (an `"ip"` vocabulary key
does not enrich the `"ip"` inside `"recipient"`).

### Python

```python
import redhop

vocab = redhop.Vocabulary({
    "usrSvc":  ["user service", "signup", "account creation"],
    "calcAmt": ["calculate amount", "billing total"],
})

enriched_chunks = []
for chunk in raw_chunks:
    text, record = vocab.enrich(chunk)
    enriched_chunks.append(redhop.Chunk(text, source=chunk_source(chunk)))
    # record.stage == "enrich"; record.matched / .added describe the change.

doc = redhop.Document.from_chunks(enriched_chunks)
```

### Node

```js
const vocab = new redhop.Vocabulary({
  usrSvc:  ["user service", "signup", "account creation"],
  calcAmt: ["calculate amount", "billing total"],
});

const enriched = rawChunks.map((c) => new redhop.Chunk(vocab.enrich(c).text));
const doc = redhop.Document.fromChunks(enriched);
```

## Discipline guardrails the API encodes

- **Token-level matching, not substring.** The same matcher
  query-side `apply` uses — `"ip"` does not enrich `"recipient"`. The
  substring-correctness bug class is closed at the matcher.
- **The library ships the mechanism, not the dictionary.** Same
  workload-specific discipline as [`Stripper`] and
  [`Vocabulary` (query-side)]: the user supplies the dict, the
  library does the matching.
- **Auditability.** Every enrichment returns a `RewriteRecord` the
  caller can collect — what was matched on, what surface forms got
  appended, the before/after text. This is what lets you A/B
  `enrich`'s value on your corpus without flying blind.

## What this changes for RedHop's surface

- **`Vocabulary` becomes a two-sided primitive.** Same compiled
  object; `.apply(query)` for the query side, `.enrich(chunk)` for the
  ingest side. No new trait, no new constructor — the data dictionary
  authored once serves both.
- **Use pattern: chunk pre-processing, not a `Document` mode.**
  Users call `vocab.enrich(...)` on chunk text *before*
  `Document.from_chunks(...)`. No new constructor variants on
  `Document`. If the pattern proves popular enough to want
  ergonomic sugar (e.g. `Document.from_chunks(chunks, enrich=vocab)`),
  it can be added without redesigning the surface.
- **The Decision Report does not currently carry per-chunk enrichment
  records.** Those are returned to the caller from `vocab.enrich(...)`
  directly. If a future need arises to surface them on the assembled
  report, it would be additive — the underlying `RewriteRecord` shape
  is identical to what already lives on `report.query_rewrites`.

## Honest limits

- **Measured falsification on the negative side, no measured
  confirmation on the positive side yet.** The CUAD probe
  ([CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md))
  measured a −2.0-point regression where the regime rule predicted
  null, confirming the rule's negative half — but the positive half
  (Spider/BIRD-style schema retrieval) is still mechanism reasoning
  awaiting a real harness. The asymmetry is honest: it's easier to
  measure "this didn't help on a workload outside the regime" than
  "this helped on a workload inside the regime, controlled against
  the right baseline."
- **Worked example uses a small toy corpus.** The
  [`enrich_code_search`](../../crates/examples/examples/enrich_code_search.rs)
  example demonstrates the audit trail and the use pattern; on the
  tiny 8-chunk corpus the ranking lift is muted because BM25 still
  surfaces partial matches. On real-world legacy code or schema
  corpora the contrast is sharper.
- **Failure modes are real and documented.** Same-boilerplate
  enrichment is the parallel failure to CUAD_PRF_NULL — same law,
  different side of the pipeline. Users who don't read the regime
  rule can hurt their corpus.

## Reproduce

```bash
cargo run -p redhop-examples --example enrich_code_search --release
```

Runs in under a second; no models, no embeddings, no LLM.

## See also

- [CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md) — the query-side
  mirror of this finding (`Vocabulary::apply`), with the +3pt
  measured lift on CUAD.
- [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) — `Stripper`, the
  subtractive query-side primitive paired with `Vocabulary` in the
  detect → strip → vocabulary → A/B workflow.
- [CUAD_PRF_NULL](CUAD_PRF_NULL.md) — query-side failure mode that
  added corpus-pervasive low-IDF terms and regressed by −3.7. The
  same-boilerplate-on-every-chunk failure mode for `enrich` is the
  parallel.
- [SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md) — corpus-only
  IDF manipulation failure; reinforces the "term-specific signal,
  not repeated filler" rule.

[`QueryRewrite::apply`]: ../../crates/redhop/src/rewrite.rs
[`apply`]: ../../crates/redhop/src/rewrite.rs
[`Stripper`]: ../../crates/redhop/src/rewrite.rs
[`Vocabulary` (query-side)]: ../../crates/redhop/src/rewrite.rs
