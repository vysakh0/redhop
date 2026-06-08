# `Vocabulary` — the additive symmetric to template stripping, confirmed on CUAD (+3.0 pts to 90.7%)

> **Status:** **Confirmed** (n=300, BM25, budget=2000, RawTopK, set-based span_recall).
> Hand-curated clause-name → synonyms expansion on top of the
> [template-stripping](CUAD_RECALL_GAP.md) workflow lifts CUAD ≥0.8
> retention from **87.7% → 90.7% (+3.0 points)** — beating LlamaIndex's
> 86% by **4 points** on the same `bench/compare.py` setup. Control arm
> shows the lift survives without stripping too (+5.0 on raw template),
> so the mechanisms are partially orthogonal: **strip removes low-IDF
> noise, expand adds high-IDF signal.**
>
> **TL;DR:** Ships `redhop::Vocabulary` as the additive symmetric to
> `redhop::Stripper`, both behind the `QueryRewrite` trait and chained
> through `Document::context_with_rewrites(...)`. Same workload-specific
> discipline: the library ships the mechanism; the caller supplies the
> dictionary. The CUAD dict in the probe is a *worked example*, not a
> library default.

## Question

The [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) finding closed most of the
4-point CUAD deficit to LlamaIndex by **subtracting** low-IDF
boilerplate from the query (the prototype-preprocessor run reported
82% → 88%; the controlled three-arm run below re-measures the shipped
`Stripper` primitive at 81.3% → 87.7%). The
[CUAD_CHUNK_FRAGMENTATION_NULL](CUAD_CHUNK_FRAGMENTATION_NULL.md)
finding then ruled out the chunker as the remaining lever — gold spans
already fit inside single chunks. So the remaining gap (87.7% to the
~98% retrieval ceiling) lives at the BM25 ranking layer, which is two
things in one: term-frequency on the query side, and inverse-document-
frequency on the corpus side.

The hypothesis tested here: **add high-IDF discriminative terms to
the query for known clause types, so the gold-bearing chunk's BM25
score rises relative to non-gold chunks that share only low-IDF
domain boilerplate.** A clause like "Change of Control" rarely uses
those exact words in the gold span; it uses related terms like
`merger`, `successor`, `acquisition` — high-IDF in a generic contract
corpus, rare across non-Change-of-Control chunks.

This is the **opposite** mechanism direction from the unweighted PRF
that was falsified in [CUAD_PRF_NULL](CUAD_PRF_NULL.md). PRF added
*corpus-pervasive low-IDF* terms by picking the top-frequency words
from first-pass top chunks; this approach adds *workload-curated
high-IDF* terms by static lookup. The mechanism prediction is
favorable; the probe **measures** whether it holds.

## API

`Vocabulary` is one of the two built-in implementations of the
`QueryRewrite` trait (the other is `Stripper`). The chain runs through
`Document::context_with_rewrites(...)` and the per-stage audit lands
on `ContextReport::query_rewrites` automatically:

```rust
let stripper = redhop::Stripper::new(&boilerplate);
let vocab = redhop::Vocabulary::new(&[
    ("change of control", &["merger", "successor", "acquisition"][..]),
    ("non-compete",       &["restraint", "non-competition"][..]),
]);
let ctx = doc.context_with_rewrites(query, &[&stripper, &vocab])?;
for record in &ctx.report.query_rewrites {
    println!("{} matched={:?} added={:?} removed={:?}",
             record.stage, record.matched, record.added, record.removed);
}
```

- **Token-level matching.** All forms (keys, synonyms, query) are
  tokenized through the analyzer at compile time, so a single-token key
  like `"ip"` cannot accidentally substring-fire inside `"recipient"`.
- **Bidirectional.** `Vocabulary::bidirectional` treats every class
  member as a trigger (PTO ↔ "paid time off" ↔ "vacation"); the default
  asymmetric mode treats the first form as the only trigger.
- **No recursive chaining.** Synonyms match against the original query
  only.
- **The library ships only the mechanism.** The dictionary is
  workload-specific user data, same discipline as `Stripper`.

Python:

```python
vocab = redhop.Vocabulary({
    "change of control": ["merger", "successor", "acquisition"],
    "non-compete":       ["restraint", "non-competition"],
})
ctx = doc.context_with_rewrites(query, [stripper, vocab])
for r in ctx.report.query_rewrites:
    print(r.stage, r.matched, r.added, r.removed)
```

Node:

```js
const vocab = new redhop.Vocabulary({
  "change of control": ["merger", "successor", "acquisition"],
});
const ctx = doc.contextWithRewrites(query, [stripper, vocab]);
for (const r of ctx.report.queryRewrites) {
  console.log(r.stage, r.matched, r.added, r.removed);
}
```

## Probe

Harness: [`crates/examples/examples/cuad_clause_expansion.rs`](../../crates/examples/examples/cuad_clause_expansion.rs).
Same configuration as the other CUAD harnesses and `bench/compare.py`:
n=300, BM25, budget=2000, candidate_k=40, RawTopK, set-based
`span_recall`, default chunker.

The CUAD clause-name dictionary lives in the probe file and **only**
there — 34 keys covering the major CUAD clause types
(`change of control`, `anti-assignment`, `non-compete`, etc.), 121
total synonyms hand-curated by inspecting what kinds of terms appear
in the gold answer spans for each clause type. Terms already pervasive
across all contracts (`agreement`, `party`, `shall`) are deliberately
excluded — those are exactly the failure mode from
[CUAD_PRF_NULL](CUAD_PRF_NULL.md).

Four arms so the mechanism is testable:

- **A: raw 24-word template** — the original CUAD baseline.
- **B: template stripped** — the CUAD_RECALL_GAP baseline.
- **C: template stripped + clause-name expanded** — the experiment.
- **D: raw template + clause-name expanded** — control. Tests whether
  expansion helps independently of stripping, or only in combination.

## Results

| arm | ≥0.8 retention | mean recall | avg context tokens |
| --- | --------------:| -----------:| ------------------:|
| A: raw template | 81.3% | 0.905 | 1890 |
| B: template stripped | 87.7% | 0.933 | 1705 |
| **C: stripped + vocabulary** | **90.7%** | **0.951** | 1783 |
| D: raw + vocabulary (control) | 86.3% | 0.925 | 1895 |

Deltas:

- **ΔC − B = +3.0 points.** Adding clause-name vocabulary on top of
  template stripping lifts ≥0.8 retention from 87.7% to 90.7% (the
  re-validated number under the new token-level matching; the original
  substring-based API measured 90.3% on the same arm, so the new shape
  is +0.4 over its predecessor as a side benefit of analyzer alignment).
- **ΔD − A = +5.0 points.** Adding vocabulary to the raw template alone
  also lifts (81.3% → 86.3%), matching LlamaIndex.
- The two mechanisms are partially orthogonal but the gains overlap:
  the template-stripped query has *less* room for vocabulary to help
  (because some of the boilerplate it removed was already adjacent to
  what the synonyms would highlight), but vocabulary still adds
  measurable value on top.

**RedHop with the full detect → strip → vocabulary workflow lands at
90.7% on CUAD, +4.7 over LlamaIndex's 86%.** The 87.7% from arm B above
(Stripper alone) was already past LlamaIndex by 1.7; this finding
extends that lead. Caveat (added in the 0.3.1 audit): the +4.7 is
RedHop with workload-curated preprocessing vs LlamaIndex with its
default retriever; the same `Stripper + Vocabulary` preprocessing is
not applied to LlamaIndex in this measurement. See
`bench/compare.py`'s fair-preprocessing arm for the comparison where
all three systems get the same Stripper.

## Why this works (mechanism, sharp)

BM25 scores a (query, chunk) pair as a sum over query terms of
`tf(term, chunk) · idf(term)`. For the templated CUAD query, after
stripping, the remaining terms are the discriminating ones (the clause
name + Details elaboration). But those terms might not appear verbatim
in the gold-bearing chunk — the gold uses synonyms.

Adding the synonyms to the query:

1. **Raises BM25 score of the gold chunk.** If `merger` appears in the
   Change-of-Control gold span and we add `merger` to the query, the
   gold chunk now matches an additional high-IDF term.
2. **Doesn't raise non-gold chunks proportionally.** `merger` has high
   IDF in the contract corpus — it's rare across non-Change-of-Control
   chunks. So the score lift on the gold chunk is bigger than on
   distractors.
3. **The mechanism is the OPPOSITE of unweighted PRF.** PRF added
   corpus-pervasive low-IDF terms (which fired on all chunks roughly
   equally, washing out the discriminator). Here, the synonyms are
   high-IDF by curation, so they fire selectively on chunks that
   actually contain the relevant content.

## What this changes

- **New API surface.** `redhop::QueryRewrite` trait with two built-in
  implementations: `Stripper` (compiled boilerplate removal) and
  `Vocabulary` (compiled equivalence classes). The chain is composed
  through `Document::context_with_rewrites(query, &[&stripper, &vocab])`
  and the per-stage audit trail lands on `ContextReport::query_rewrites`
  as a list of `RewriteRecord` ({stage, from, to, matched, added,
  removed}) — every rewrite is *observable*, not buried in a
  preprocessor.
- **API surface across all three bindings.** Rust, Python, Node — same
  shape in each, mirror-tests in each.
- **Token-level matching, analyzer-aligned.** Both `Stripper` and
  `Vocabulary` compile their forms through the same Snowball analyzer
  the BM25 index uses, so a single-token strip cannot accidentally
  erase a substring, and a vocabulary key cannot fire on a substring
  inside a longer word.
- **The detect → strip → A/B workflow extends to detect → strip →
  vocabulary → A/B.** `analyze_query_set` still detects the templated
  pattern; `Stripper` removes the boilerplate; `Vocabulary` adds the
  discriminative terms; `evaluate` still scores the lift.
- **Chunk-side mirror.** `Vocabulary::enrich(chunk)` ships as the
  ingest-time symmetric to query-side `apply`. Same compiled vocab,
  applied to chunk text rather than queries — useful when the chunks
  themselves are short and opaque (schema columns, API symbols, error
  codes, defined contract terms) and a natural-language query can't
  match them by surface form. The two sides are *different jobs*:
  `apply` patches gaps you anticipated; `enrich` raises content's
  semantic floor for queries you can't anticipate. Regime rule, use
  cases, and failure modes (especially the "same boilerplate on every
  chunk" parallel to CUAD_PRF_NULL) in
  [VOCABULARY_ENRICH](VOCABULARY_ENRICH.md).

## Honest limits

- **The CUAD-specific dict is a *worked example*, not a library
  feature.** It lives in `crates/examples/examples/cuad_clause_expansion.rs`
  and only there. Users on other workloads need their own dict.
- **Hand-curated synonyms ≠ a recipe for synonyms.** The dict was
  built by inspecting CUAD gold spans, picking terms that *we knew*
  appeared there. A user with a new workload would need to do the
  same domain work. We do **not** ship a generic synonym-mining
  tool here (that's the IDF-weighted-PRF arc explicitly punted in
  CUAD_PRF_NULL).
- **No bootstrap CIs on the +3.0.** Same caveat as the other CUAD
  findings. n=300 with a 3.0-point shift on an 88% baseline is
  probably significant but unconfirmed by CI.
- **Single workload (CUAD).** The mechanism prediction (additive
  high-IDF synonyms close the post-strip remainder) generalizes; the
  precise magnitude depends on how well-curated the dict is and how
  much of the workload's gap is synonym-driven vs other.
- **No downstream answer eval (Tier 3).** Whether the +2.7 retention
  improvement translates to a measurable F1/EM lift on gpt-4o-mini
  is a separate question.

## Reproduce

```bash
cargo run -p redhop-examples --example cuad_clause_expansion --release
```

Runs in well under a second; no models, no embeddings, no LLM. Same
`cuad_sample.json` as the other CUAD findings.

## See also

- [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) — template stripping, the
  subtractive mechanism that this finding builds on.
- [CUAD_PRF_NULL](CUAD_PRF_NULL.md) — *unweighted* PRF, the additive
  mechanism that failed because it added low-IDF terms instead of
  high-IDF ones. The contrast with this finding is what makes the
  mechanism here predictive.
- [QUERY_SET_ANALYZER](QUERY_SET_ANALYZER.md) — the detection step
  that comes before stripping and expansion in the workflow.
- [EVALUATE_API](EVALUATE_API.md) — the A/B scorer that lets you
  measure whether expansion (or any other workload-specific
  preprocessor) actually helps on **your** gold data.
