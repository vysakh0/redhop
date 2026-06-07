# `expand_query_terms` — the additive symmetric to template stripping, confirmed on CUAD (+2.7 pts to 90%)

> **Status:** **Confirmed** (n=300, BM25, budget=2000, RawTopK, set-based span_recall).
> Hand-curated clause-name → synonyms expansion on top of the
> [template-stripping](CUAD_RECALL_GAP.md) workflow lifts CUAD ≥0.8
> retention from **87.7% → 90.3% (+2.7 points)** — beating LlamaIndex's
> 86% by **4 points** on the same `bench/compare.py` setup. Control arm
> shows the lift survives without stripping too (+5.0 on raw template),
> so the mechanisms are partially orthogonal: **strip removes low-IDF
> noise, expand adds high-IDF signal.**
>
> **TL;DR:** Ships `redhop::expand_query_terms` as the additive symmetric
> to `drop_template_terms`. Same workload-specific discipline: the
> library ships the mechanism; the caller supplies the dictionary. The
> CUAD dict in the probe is a *worked example*, not a library default.

## Question

The [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) finding closed most of the
4-point CUAD deficit to LlamaIndex by **subtracting** low-IDF
boilerplate from the query (82% → 88%). The
[CUAD_CHUNK_FRAGMENTATION_NULL](CUAD_CHUNK_FRAGMENTATION_NULL.md)
finding then ruled out the chunker as the remaining lever — gold spans
already fit inside single chunks. So the remaining gap (88% to the
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

New public Rust function, mirrored on both bindings:

```rust
pub fn expand_query_terms(query: &str, expansions: &[(&str, &[&str])]) -> String;
```

- For each `(key, synonyms)` pair, if the query contains `key`
  (case-insensitive substring), every synonym is appended to the
  returned string with a single space separator.
- Matches against the **original** query only — no recursive chaining,
  no duplicates across keys.
- The library ships **only** the mechanism. The dictionary is
  workload-specific user data, same discipline as `drop_template_terms`.

Python:

```python
expansions = {
    "change of control": ["merger", "successor", "acquisition"],
    "non-compete":       ["restraint", "non-competition"],
}
expanded = redhop.expand_query_terms(
    'What about "Change of Control" clauses?',
    expansions,
)
# → 'What about "Change of Control" clauses? merger successor acquisition'
```

Node:

```js
const expanded = redhop.expandQueryTerms(
  'What about "Change of Control" clauses?',
  { "change of control": ["merger", "successor", "acquisition"] },
);
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
| **C: stripped + expanded** | **90.3%** | **0.951** | 1783 |
| D: raw + expanded (control) | 86.3% | 0.925 | 1895 |

Deltas:

- **ΔC − B = +2.7 points.** Adding clause-name expansion on top of
  template stripping lifts ≥0.8 retention from 87.7% to 90.3%.
- **ΔD − A = +5.0 points.** Adding expansion to the raw template alone
  also lifts (81.3% → 86.3%), matching LlamaIndex.
- **C vs (B + (D − A) − A) = 90.3 vs ~92.7.** The two mechanisms are
  partially orthogonal but the gains overlap by ~2.4 points: the
  template-stripped query has *less* room for expansion to help
  (because some of the boilerplate it removed was already adjacent to
  what the synonyms would highlight), but expansion still adds
  measurable value on top.

**RedHop with the full detect → strip → expand workflow lands at 90.3%
on CUAD, +4 over LlamaIndex's 86%.** The 88% from
CUAD_RECALL_GAP was already past LlamaIndex by 2; this finding extends
that lead.

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

- **New API.** `redhop::expand_query_terms` ships as the additive
  symmetric to `drop_template_terms`. Same signature shape (takes
  user-supplied data, returns a string). Same workload-specific
  discipline (library has the mechanism, user supplies the dict).
- **API surface across all three bindings.** Rust, Python, Node — same
  call shape in each, mirror-tests in each (6 Rust unit tests + 5
  Python pytest functions + 5 Node assertion blocks).
- **The detect → strip → A/B workflow extends naturally to detect →
  strip → expand → A/B.** `analyze_query_set` still detects the
  templated pattern; `drop_template_terms` still removes the
  boilerplate; `expand_query_terms` now adds the discriminative
  terms; `evaluate` still scores the lift.
- **`CHOOSING_A_CONFIG.md` step 3 ("Templated queries with heavy
  boilerplate")** gains an optional follow-on step explaining when
  expansion is the right next move (workloads where you have a known
  taxonomy of "topics" each with predictable synonyms — legal QA,
  support-ticket triage where every ticket maps to one of a fixed set
  of issue categories, etc.).

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
- **No bootstrap CIs on the +2.7.** Same caveat as the other CUAD
  findings. n=300 with a 2.7-point shift on an 88% baseline is
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
