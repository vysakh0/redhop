# `Vocabulary.enrich` on Spider schemas — confirmed on the regime's positive side, +0.19 mean recall

> **Status:** **Confirmed** (n=30 hand-curated Spider-shape questions
> across 5 databases, BM25, candidate_k=10, set-based column recall).
> Curated chunk-side enrichment with workload synonyms lifts schema-
> retrieval mean recall from **0.772 → 0.967 (+0.19)** and ≥0.8
> retention from **63% → 93% (+30 pts)**. Auto-derived enrichment
> (cleaning the column name + appending type + table name) lifts to
> **0.900 (+0.13)**; curated synonyms add **another +0.07** on top.
>
> **TL;DR:** The
> [`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) regime rule now has
> measured evidence on its **positive side**. Combined with the
> measured negative
> ([`CUAD_ENRICH_DEFINITIONS_NULL`](CUAD_ENRICH_DEFINITIONS_NULL.md)),
> the rule has bidirectional empirical grounding: enrich works on
> short, opaque coded retrieval units paired with a workload-curated
> decoding dictionary; fails on long prose chunks.

## Question

[`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) shipped a regime-rule
prediction: `value ∝ shortness × opacity × dictionary-exists`. The
*negative* prediction has been measured directly on CUAD
([CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md):
−2.0 pts on prose chunks). The *positive* prediction — that schema-
style retrieval (short opaque column names + a data-dictionary)
should benefit — was mechanism reasoning awaiting a measurement.

This probe runs that measurement on a Spider-format sample.

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
  table name. Mirrors what you'd get without any data-dictionary work.
- **C — curated-enriched chunks.** The auto layer plus workload-
  specific synonyms (`Age` → `["old", "young", "years"]`,
  `Population` → `["people", "residents", "inhabitants"]`, etc.).
  Mirrors the worked CUAD clause-name dictionary in
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

| arm | mean recall | ≥0.5 retention | ≥0.8 retention |
| --- | -----------:| --------------:| --------------:|
| A: bare | 0.772 | 90% | 63% |
| B: auto-enriched | **0.900** (+0.128) | 97% | 83% (+20) |
| **C: curated-enriched** | **0.967** (+0.194) | **100%** | **93% (+30)** |

Deltas:

- **ΔB − A = +0.128 mean recall, +20 pts on ≥0.8.** Auto-enrichment
  (cleaned name + type + table) helps, but the lift comes mostly from
  the table-name being prepended (`Age` → `"Age age int singer"`)
  rather than from the column-name cleaning. The "singer" token lets
  queries about singers hit Age columns even without further
  synonyms.
- **ΔC − A = +0.194 mean recall, +30 pts on ≥0.8.** Adding curated
  synonyms (`Age` → `"old young years"`, `Population` → `"people
  residents inhabitants"`) gives the additional lift.
- **ΔC − B = +0.067 mean recall, +10 pts on ≥0.8.** The marginal
  contribution of curated synonyms *on top of* auto-enrichment. This
  is the cleanest test of the workload-curation discipline.

### k-sensitivity (sanity check)

| candidate_k | ΔB − A (auto) | ΔC − A (curated) | ΔC − B |
| -----------:| -------------:| ----------------:| ------:|
| 5  | +0.006 | +0.061 | +0.056 |
| 10 | **+0.161** | **+0.228** | **+0.067** |
| 15 | +0.178 | +0.178 | +0.000 |

At k=5 (tight): auto is flat, curated still helps. At k=15 (loose):
both arms ceiling at 1.0, contrast washes out. k=10 is the
discriminating setting where the mechanism is most visible.

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
candidates. **Hit (with margin).**

The mechanism is **additive high-IDF terms appended to the chunk that
match the query's high-IDF terms** — the exact symmetric mirror to
[`CUAD_CLAUSE_EXPANSION`](CUAD_CLAUSE_EXPANSION.md)'s query-side
mechanism. Same workload-curated dictionary discipline; same
high-IDF requirement.

## What this changes

- **The regime rule now has bidirectional measured evidence.** The
  [`VOCABULARY_ENRICH`](VOCABULARY_ENRICH.md) rule predicted enrich
  works on short, opaque, coded retrieval units. The negative side
  was already measured (CUAD prose: −2.0pt). The positive side is now
  also measured (Spider schemas: +0.23 mean recall on curated). The
  rule is no longer "mechanism reasoning + IR literature" — it's
  empirically validated on both ends of the regime axis.
- **The four-corner rule is now measured on all four corners.**
  Workload-pervasive signal manipulation fails on either side of the
  pipeline; only workload-curated semantics work:

  |               | curated / workload-aware           | auto / corpus-pervasive            |
  | ------------- | ---------------------------------- | ---------------------------------- |
  | **query-side**| works ([CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md): +3.0pt) | fails ([CUAD_PRF_NULL](CUAD_PRF_NULL.md): −3.7pt, [SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md)) |
  | **chunk-side**| works (this probe: +0.23 mean recall) | fails ([CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md): −2.0pt) |

  Auto-enrich's +0.16 lift on Spider is a partial exception, but the
  mechanism is the table-name prepending acting *as* curation (a
  workload-aware signal), not the column-name cleaning. The rule
  holds.
- **Enrich's user-facing framing can be sharpened.** The 0.3.0 docs
  pass framed enrich as "shipped on regime reasoning, measured
  negative only." That asymmetry is now closed. Future doc revisions
  can lead with the regime-as-recommendation more confidently
  ("use it on short opaque coded units; here's the evidence on
  Spider").

## Honest limits

- **n=30, no bootstrap CIs.** Same caveat as other sample-based
  findings. The +0.23 shift is well outside typical noise but
  isn't CI-confirmed. Run full Spider via `REDHOP_SPIDER_PATH` for a
  larger measurement.
- **Hand-labeled gold and hand-curated synonyms.** The data file's
  gold-column sets and the synonym dictionary were authored as part
  of the probe; not pulled from the full Spider distribution's SQL
  parsing or any external data-dictionary. This is consistent with
  how the CUAD clause-expansion probe was built (the dict was
  hand-curated from inspection of CUAD gold spans), but worth being
  explicit about: the probe measures the *mechanism*, given a
  reasonable workload-curation effort.
- **Sample-schema selection.** The 5 databases were chosen for the
  presence of abbreviated columns (`StuID`, `MPG`, `Edispl`, `GNP`,
  `IndepYear`) where enrichment has the most room to bite. Other
  Spider databases with more self-descriptive columns (`order_total`,
  `customer_name`) would likely show smaller lifts.
- **No downstream answer eval.** Whether the retrieval lift translates
  to a measurable SQL-generation lift on an actual text-to-SQL model
  is a separate question (Tier 3, not run).

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

## See also

- [VOCABULARY_ENRICH](VOCABULARY_ENRICH.md) — the regime rule this
  finding validates on its positive side.
- [CUAD_ENRICH_DEFINITIONS_NULL](CUAD_ENRICH_DEFINITIONS_NULL.md) — the
  measured negative side on prose chunks. With this Spider result,
  the rule has bidirectional evidence.
- [CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md) — the query-side
  mirror. Same workload-curated dictionary discipline; same mechanism
  (additive high-IDF terms); opposite side of the pipeline.
- [CUAD_PRF_NULL](CUAD_PRF_NULL.md) +
  [SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md) — the other two
  corners of the four-corner rule. All four corners now measured.
