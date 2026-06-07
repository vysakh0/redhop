# `Vocabulary.enrich` — the chunk-side mirror of query rewriting, for short opaque retrieval units

> **Status:** **Shipped on mechanism + regime reasoning**; measured
> probes (Spider/BIRD for schemas) queued not run. The mechanism is
> well-established in IR (the doc2query family); the regime where it
> earns its keep is sharp enough that users can predict whether it
> applies to their workload without their own probe. Failure modes —
> especially the "same boilerplate on every chunk" trap — are
> documented so the prediction stays honest.
>
> **TL;DR:** [`Vocabulary::enrich`](../../crates/redhop/src/rewrite.rs)
> is the symmetric to query-side [`QueryRewrite::apply`]. Applied at
> ingest time, it appends a compiled dictionary's synonyms to chunk
> text where the chunk contains a vocabulary key. The use case is
> *short, opaque, coded retrieval units paired with a decoding
> dictionary* — schema columns, API symbols, error codes, defined
> contract terms, clinical abbreviations.

## The rule (when enrich earns its keep)

A compact, testable predictor:

```text
value ∝ shortness × opacity × (dictionary exists)
```

- **Shortness.** A long prose paragraph already has plenty of surface
  area for BM25 (or any lexical scorer) to match against; adding more
  signal helps marginally at best. A bare token — `emp_compensation`,
  `ERR_4012`, `MI`, `usrSvc` — has almost no matchable surface, so
  enrichment is what gives it meaning at all.
- **Opacity.** The retrieval unit is a coded identifier: abbreviated,
  domain-internal, machine-generated. Its surface form has nothing in
  common with how users would phrase their question.
- **Dictionary exists.** The cost of building a vocabulary from scratch
  may exceed the benefit; the mechanism predicts wins when you
  *already* have a decoding dictionary (data dictionary, glossary,
  abbreviation list, API reference) and the missing piece is wiring it
  into retrieval.

Outside this regime — long descriptive prose — enrichment is redundant
(matching already works) and risks the failure modes below.

## Why this is more general than query-side `apply` (not less)

Query-side [`apply`] is surgical: you enumerate the synonyms you expect
users to use. It can't help a query phrasing you didn't foresee. Chunk
-side `enrich` describes the *content* once and serves every future
query — including phrasings you never listed — because it raises the
content's semantic floor rather than patching specific holes.

So they're not competing on the same axis. They're different jobs:

|  | `apply` (query-side) | `enrich` (chunk-side) |
| --- | --- | --- |
| **Direction** | rewrites the query | rewrites the chunks |
| **Time** | query time | ingest time |
| **Covers** | the query reformulations you anticipated | unanticipated queries against content you described |
| **Cost** | per-query, small | one-time at ingest, larger |
| **Audit trail** | `report.query_rewrites` | per-chunk `RewriteRecord` from `vocab.enrich(...)` |
| **Best when** | you can enumerate user phrasings | you can describe what your content is |
| **Works on dense?** | partial (rewrites the query embedded) | yes (raises chunk's semantic floor in both lexical + dense) |

Use both when both apply. They compose naturally — `apply` for the
queries you anticipate, `enrich` for content meaning that needs to be
made explicit.

## Use cases, ranked by how well they fit the rule

**Strong — short, opaque units + a real dictionary:**

- **Schemas (text-to-SQL).** Column names like `emp_compensation`,
  `ord_dt`. Data dictionaries are standard practice. Maximally short +
  opaque; the cleanest case. (Public benchmarks: Spider, BIRD.)
- **Code / API search.** Symbols (`usrSvc`, `calcAmt`), endpoints
  (`POST /v1/payment_intents`). Enrichment helps most on undocumented
  / legacy codebases — where docstrings already exist, the
  documentation already enriches.
- **Logs / observability.** Error and event codes (`ERR_4012`,
  `evt_chrgbck`). Enrichment with their meanings lets on-call engineers
  search natural-language symptoms (`"payment failed"`) and reach the
  right runbook section. Underrated and very real for ops correlation.
- **Clinical / medical notes.** Abbreviation soup (`MI`, `SOB`, `HTN`,
  `ICD-10` codes). Medical glossaries let lay or NL queries match
  cryptic chart notes. High value, domain-specialized.
- **Legal contracts — defined terms.** The Definitions section is a
  glossary the document ships with itself; enrich each clause chunk
  with the relevant defined-term meanings. Bonus: the dictionary is
  often auto-extractable from the document.
- **Acronym-heavy enterprise wikis.** Codenames, team abbreviations,
  internal tool names. Enrich so full-name queries reach codename
  documentation.

**Medium — fits the rule but the dictionary gap is often already
filled:**

- **Product / parts catalogs.** SKUs and part numbers — strong for
  industrial/technical catalogs; weak for consumer ones that already
  have rich descriptions.
- **Financial / tickers.** `AAPL` → "Apple Inc.", accounting line-item
  codes.
- **Scientific / bio.** Gene symbols (`TP53`), chemical identifiers,
  field abbreviations.

## Where this is *not* useful (the honest failure cases)

- **Normal prose corpora.** If your chunks are already descriptive
  paragraphs, enrichment adds nothing — you'd be re-indexing meaning
  that's already there.
- **Same-boilerplate enrichment.** Bolt the *identical* description
  onto every chunk (e.g. "this is a function" on every code chunk) and
  you re-create the low-IDF dilution falsified twice in
  [CUAD_PRF_NULL](CUAD_PRF_NULL.md) and
  [SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md). Enrichment must
  add *term-specific* signal, not repeated filler. Under dense
  retrieval the failure mode is vector-collapse instead of IDF
  dilution, but the law is the same.
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
    enriched_chunks.append(text)
    # record.stage == "enrich"; record.matched / .added describe the change.

doc = redhop.Document.from_chunks(enriched_chunks)
```

### Node

```js
const vocab = new redhop.Vocabulary({
  usrSvc:  ["user service", "signup", "account creation"],
  calcAmt: ["calculate amount", "billing total"],
});

const enriched = rawChunks.map((c) => vocab.enrich(c).text);
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

- **No measured probe yet.** Mechanism prediction is sharp and
  IR-literature established, but until Spider/BIRD or a similar
  workload is harnessed, we can't put a number on the lift. The
  measured probes are queued, not run.
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
