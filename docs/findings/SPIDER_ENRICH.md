# `Vocabulary.enrich` on Spider schemas — suggestive, mechanism observed

> **Status:** **Suggestive** (n=30 hand-authored questions across 5
> hand-selected Spider databases, BM25, candidate_k=10, set-based
> column recall). Two measurements with different strengths:
>
> 1. **Auto-enrichment lifts mean recall 0.772 → 0.900 (+0.128).**
>    This arm is **unconflicted**: the enrichment text is derived
>    deterministically from schema metadata (cleaned column name + type
>    + parent-table name). No human judgment is applied at the
>    enrichment step, so the lift cannot be an artifact of the
>    enrichment author also writing the questions.
> 2. **Hand-curated synonyms add +0.067 on top (0.900 → 0.967).** The
>    synonyms were authored by the same agent that authored the
>    questions; **this is author-curator overlap and the +0.067 is
>    therefore an upper bound, not a generalization estimate.** Read it
>    as "the mechanism *can* deliver another +0.067 when synonyms
>    align" — not as a forecast of what blind-authored synonyms would
>    yield on the same workload.
>
> **TL;DR:** The mechanism is observed and the *free* (schema-derived,
> deterministic) part of the lift is unconflicted. The *paid* (hand-
> curated synonyms) part is suggestive only on this probe because of
> author-curator overlap. The
> [`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) regime guidance still
> rests primarily on its negative side
> ([`CUAD_ENRICH_DEFINITIONS_NULL`](CUAD_ENRICH_DEFINITIONS_NULL.md)).

## Question

[`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) shipped a regime-rule
prediction: `value ∝ shortness × opacity × dictionary-exists`. The
*negative* prediction has been measured directly on CUAD
([CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md):
−2.0 pts on prose chunks). The *positive* prediction — that schema-
style retrieval (short opaque column names + a data-dictionary)
should benefit — was mechanism reasoning awaiting any measurement.

This probe runs that measurement on a Spider-format sample. The
probe is **not** a generalization claim; it is a demonstration that
the mechanism does what it predicts on at least one schema-shape
workload, with explicit caveats about which part of the lift is
measurement-clean and which is suggestive.

## Probe

Harness: [`crates/examples/examples/spider_enrich_probe.rs`](../../crates/examples/examples/spider_enrich_probe.rs).
Sample data: [`data/spider/spider_sample.json`](../../data/spider/spider_sample.json)
— 5 databases (concert_singer, pets_1, world_1,
employee_hire_evaluation, car_1) × 6 questions = **n=30 hand-labeled
examples**. Each column has an optional hand-curated `synonyms` array
representing the high-IDF natural-language terms a workload's data
dictionary would attach.

`REDHOP_SPIDER_PATH` swaps in the full Spider distribution if you have
it locally. Default uses the committed sample.

### Three arms

- **A — bare column-name chunks.** Each column becomes one
  `Chunk(name)`. BM25 sees only the analyzer-tokenized name.
- **B — auto-enriched chunks.** Each column is enriched with its
  cleaned name (snake_case + camelCase → spaces), type, and parent-
  table name. **Deterministic from schema metadata; no human judgment
  applied during enrichment.**
- **C — curated-enriched chunks.** The auto layer plus workload-
  specific synonyms (`Age` → `["old", "young", "years"]`,
  `Population` → `["people", "residents", "inhabitants"]`, etc.).
  **Synonyms hand-authored by the same agent that authored the
  questions.** Mirrors the worked CUAD clause-name dictionary in
  [CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md).

### Configuration

n=30, BM25, RawTopK, **candidate_k=10**, set-based column-level recall.
Each db has 15-30 columns; k=10 forces meaningful ranking pressure
without being so tight that any retrieval is trivial.

### Metric

Per-question column recall: of the gold-labeled columns the SQL answer
references, how many appear in the top-k retrieved set? Reported as
mean recall + the fraction of questions where recall ≥0.5 / ≥0.8.

## Results

| arm | mean recall | ≥0.5 retention | ≥0.8 retention | conflict status |
| --- | -----------:| --------------:| --------------:| --- |
| A: bare | 0.772 | 90% | 63% | baseline |
| B: auto-enriched | **0.900** (+0.128) | 97% | 83% (+20) | **unconflicted** (schema-derived) |
| C: curated-enriched | **0.967** (+0.194) | **100%** | **93% (+30)** | **suggestive only** (author-curator overlap) |

Deltas:

- **ΔB − A = +0.128 mean recall, +20 pts on ≥0.8.** Auto-enrichment
  (cleaned name + type + table) helps. Inspection of which token
  drives the lift: the table-name prepended to each column
  (`Age` → `"Age age int singer"`) lets a query about "singers" hit
  the Age column even with no further synonyms. The cleaned column
  name alone (`StuID` → `"StuID stu id"`) adds little because the
  analyzer-tokenized form is often subsumed by the bare column name.
- **ΔC − B = +0.067 mean recall, +10 pts on ≥0.8.** The marginal
  contribution of curated synonyms *on top of* auto-enrichment. This
  is the curator-conflicted measurement (see Methodology limitations).
- **ΔC − A = +0.194 mean recall, +30 pts on ≥0.8.** Total lift; not
  cited as the headline because the +0.067 portion is upper-bounded by
  the author-curator overlap.

### k-sensitivity (sanity check)

| candidate_k | ΔB − A (auto) | ΔC − A (curated) | ΔC − B |
| -----------:| -------------:| ----------------:| ------:|
| 5  | +0.006 | +0.061 | +0.056 |
| 10 | **+0.161** | **+0.228** | **+0.067** |
| 15 | +0.178 | +0.178 | +0.000 |

At k=5 (tight): auto is flat, curated still helps. At k=15 (loose):
both arms ceiling at 1.0, contrast washes out. k=10 is the
discriminating setting where the mechanism is most visible.

## Methodology limitations

The probe is structurally compromised in ways worth naming up front so
readers don't take it for more than it is:

1. **Author-curator overlap on arm C.** The same agent authored both
   the 30 questions and the column synonyms. There is no separation
   between the person who picked what would be asked and the person
   who picked which synonyms to attach. A synonym like
   `MPG → ["fuel", "efficiency", "miles per gallon"]` was added knowing
   that questions in the sample ask about "fuel efficiency." The
   +0.067 marginal lift from curated synonyms therefore measures
   *aligned authoring*, not blind authoring. Treat as an upper bound.
2. **Sample-schema selection bias.** The 5 databases were chosen
   because their columns are short and abbreviated (`StuID`, `MPG`,
   `Edispl`, `GNP`, `IndepYear`) — exactly the conditions
   [`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) predicts enrich should
   help most. Other Spider databases with self-descriptive columns
   (`order_total`, `customer_name`) would likely show smaller lifts.
   The probe shows the mechanism on a *favorable* workload, not the
   distribution-average.
3. **n=30, no bootstrap CIs.** The shifts are large relative to
   typical retrieval noise (≈0.05) but no confidence intervals are
   computed. A larger run via `REDHOP_SPIDER_PATH` against the full
   Spider distribution would tighten this.
4. **No downstream answer eval.** Whether the column-recall lift
   translates to a measurable SQL-generation lift on an actual
   text-to-SQL model is a separate question (Tier 3, not run).
5. **Arm B is "auto" only at the enrichment step.** The table-name
   prepending — which delivers most of arm B's +0.128 — relies on the
   schema author having named tables meaningfully. That naming is
   itself a form of curation, just one that exists for free as a
   byproduct of building the schema. Arm B is unconflicted *given a
   reasonably-named schema*; it is not a "zero-curation" baseline in
   the absolute sense.

What the probe *does* support, cleanly:

- **Mechanism observed.** Appending high-IDF terms that match query
  vocabulary to short opaque chunks lifts BM25 recall. This was the
  prediction; the probe confirms the direction on a favorable
  workload.
- **Free part of the lift is real.** Arm B's +0.128 does not require
  human authoring at the enrichment step — only that the schema have
  meaningful table names. Users get this without writing a synonym
  dictionary.

What it does *not* support:

- That hand-curated synonyms reliably add +0.067 on Spider-shape
  workloads when authored without knowledge of the queries.
- That the mechanism generalizes to schemas with self-descriptive
  column names.
- A "regime rule" with universal force across schema-shape workloads.
  See [findings README](README.md) on the four-corner *observation*.

## Why this works (mechanism, sharp)

For a query like *"Find singers older than 40 and their country of
origin"*, the BM25-relevant query tokens after stopword removal are
`["find", "singer", "old", "country", "origin"]`. The gold columns
are `singer.Age` and `singer.Country`.

**Arm A** — `singer.Age` tokenizes to `["age"]`. None of the query
tokens match → low BM25 score → may rank outside top-10. Miss.

**Arm B** — `singer.Age` enriched to `"Age age int singer"` tokenizes
to `["age", "int", "singer"]`. `"singer"` matches the query → BM25
score lifts → ranks into top-10. **Hit (via table context).**

**Arm C** — `singer.Age` enriched to
`"Age age int singer old young years"` tokenizes to
`["age", "int", "singer", "old", "young", "year"]`. Now `"singer"`
*and* `"old"` match → BM25 score lifts further → ranks in top
candidates. **Hit (with margin) — but the "old" synonym was added
knowing the questions ask about old singers.**

The mechanism is **additive high-IDF terms appended to the chunk that
match the query's high-IDF terms** — the exact symmetric mirror to
[`CUAD_CLAUSE_EXPANSION`](CUAD_CLAUSE_EXPANSION.md)'s query-side
mechanism.

## What this changes

- **`VOCABULARY_ENRICH`'s positive side has *suggestive* evidence,
  not confirmation.** The unconflicted part of the lift (auto-
  enrichment, +0.128) is measurement-clean and supports the
  mechanism. The curated part is upper-bounded by author-curator
  overlap. Pair this with the cleaner negative side
  ([`CUAD_ENRICH_DEFINITIONS_NULL`](CUAD_ENRICH_DEFINITIONS_NULL.md):
  −2.0pt on prose) and the rule still has more measured evidence on
  its boundary than on its core — which is appropriate for a
  conservative regime guidance.
- **The four-corner *observation*** ([findings README](README.md)) has
  one cleanly-measured corner per axis (CUAD_RECALL_GAP query-side
  curated, CUAD_PRF_NULL query-side auto, CUAD_ENRICH_DEFINITIONS_NULL
  chunk-side auto, plus this probe's unconflicted arm B chunk-side).
  Calling it a "rule" overstates universality from n≤2 datasets per
  corner; the findings README has been updated accordingly.

## Reproduce

```bash
cargo run -p redhop-examples --example spider_enrich_probe --release
```

Or with the full Spider distribution:

```bash
REDHOP_SPIDER_PATH=/path/to/spider/tables.json \
  cargo run -p redhop-examples --example spider_enrich_probe --release
```

The harness sweeps `candidate_k` via `REDHOP_SPIDER_K`:

```bash
REDHOP_SPIDER_K=5  cargo run -p redhop-examples --example spider_enrich_probe --release
REDHOP_SPIDER_K=10 cargo run -p redhop-examples --example spider_enrich_probe --release
REDHOP_SPIDER_K=15 cargo run -p redhop-examples --example spider_enrich_probe --release
```

## Open follow-ups

The natural extensions that would tighten this probe:

- **Blind-authored synonyms.** Generate synonyms from a source that
  did not see the questions: WordNet, ConceptNet, a different agent
  session, or a SQL-comment extraction over the canonical Spider data
  dictionary. Re-run arm C; check whether +0.067 holds, shrinks, or
  flips.
- **Held-out questions.** Use the Spider train/dev split rather than
  hand-authored questions, so the question authoring is independent
  of the synonym authoring.
- **Full Spider distribution.** n=30 → n=full (≈10,000 questions).
  Bootstrap CIs.
- **Cross-schema generalization.** Run on databases the synonym
  authoring did *not* target.

## See also

- [VOCABULARY_ENRICH](VOCABULARY_ENRICH.md) — the regime guidance this
  finding partially supports on its positive side (free arm cleanly;
  curated arm suggestively).
- [CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md) — the
  measured negative side on prose chunks.
- [CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md) — the query-side
  mirror. Same workload-curated dictionary discipline; same mechanism
  (additive high-IDF terms); opposite side of the pipeline. Also
  carries an author-curator overlap caveat for the curated synonyms.
- [CUAD_PRF_NULL](CUAD_PRF_NULL.md) +
  [SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md) — the other two
  corners of the four-corner observation.
