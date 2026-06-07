# Sub-IDF auto-drop — NULL RESULT (the win was query-set overlap, not the corpus IDF profile)

> **Status:** **Null result / falsified.** Automatically dropping
> low-IDF terms from a query using **corpus-side document-frequency
> statistics** doesn't lift CUAD meaningfully (+0.7 at the best
> threshold vs the +6.4 from user-supplied template stripping) and
> **regresses diverse natural-language workloads** by 1–5 points.
>
> **TL;DR:** The clause-expansion and template-stripping wins
> manipulate the IDF profile of the *query*, but they only worked
> because the user (or `analyze_query_set` from the user's query
> sample) supplied the partition between **wrapper** and
> **discriminator**. Corpus DF alone can't make that distinction —
> a token can be high-DF in the corpus AND high-discriminative
> for a given query (e.g., "France" in a HotpotQA distractor pool).
> The mechanism worth remembering: **the win in
> CUAD_RECALL_GAP / CUAD_CLAUSE_EXPANSION was query-set overlap,
> not corpus IDF.**

## Question

[CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md) showed that
manipulating the IDF profile of the query closes the CUAD gap:
adding *workload-curated high-IDF synonyms* lifts +2.7 points on top
of the +6.4 from template stripping (88% → 90.3%). The natural
follow-up: **can the symmetric subtractive operation —
automatically dropping low-IDF terms from the query — be done
without user-supplied data?** If yes, the library could auto-apply
it as a generic improvement; users wouldn't need to call
`drop_template_terms` or build a clause dict for every workload.

The mechanism prediction was favorable: high-DF (= low-IDF) terms
contribute less BM25 score per match, so dropping them should
preserve ranking quality while reducing query-side noise. The
question is empirical: **does it actually help CUAD without
regressing diverse workloads?**

Two failure modes to rule out simultaneously:

- **CUAD true positive.** A templated workload on a boilerplate-
  heavy corpus should benefit. If auto-drop doesn't lift CUAD, the
  mechanism is null.
- **HotpotQA / MuSiQue false-positive regression.** Diverse
  natural-language queries are short (5–15 words); dropping any
  could destroy retrieval. If diverse workloads regress, the
  mechanism can't be auto-default; it has to be opt-in.

## Setup

Harness: [`crates/examples/examples/sub_idf_reweighting_probe.rs`](../../crates/examples/examples/sub_idf_reweighting_probe.rs).

For each (query, document) pair:

1. Compute per-chunk document frequency for every alphanumeric
   token across the document's chunks.
2. For each threshold `cap_share ∈ {1.0 (control), 0.70, 0.50, 0.30}`,
   drop query tokens whose chunk-DF share exceeds `cap_share`.
3. If filtering empties the query, fall back to the original (don't
   tank the workload by handing BM25 an empty string).
4. Run BM25 retrieval at `candidate_k=40`, assembly at the same
   `budget` and `RawTopK` strategy as the other CUAD harnesses.
5. Measure set-based `span_recall` against the gold answer.

Three workloads, same n=300 as the other findings:

- **CUAD** (templated workload, boilerplate-heavy corpus,
  budget=2000) — should benefit.
- **HotpotQA** (diverse natural language, distractor pool concatenated
  per question, budget=400) — should not regress.
- **MuSiQue** (diverse natural language, per-question paragraph pool,
  budget=400) — should not regress.

## Results

### CUAD

| threshold | ≥0.8 retention | mean query length |
| --------- | --------------:| -----------------:|
| none (control) | 81.3% | 36.0 tokens |
| drop df > 70% | 81.0% | 27.6 |
| drop df > 50% | 81.0% | 24.1 |
| **drop df > 30%** | **82.0%** | 21.8 |

**Δ best = +0.7 points** — within sample noise; not a meaningful
lift. For reference, template stripping
([CUAD_RECALL_GAP](CUAD_RECALL_GAP.md)) gets +6.4 on the same baseline.
Auto-drop captures roughly **10% of the lift** that user-side
template stripping delivers.

### HotpotQA

| threshold | ≥0.8 retention | mean query length |
| --------- | --------------:| -----------------:|
| none (control) | 87.1% | 17.3 tokens |
| drop df > 70% | 85.7% | 11.5 |
| drop df > 50% | 85.3% | 9.4 |
| **drop df > 30%** | **82.1%** | 7.6 |

**Δ worst = −5.0 points.** Even moderate IDF filtering
(`df > 70%`) costs 1.4 points; the aggressive threshold drops mean
query length from 17 → 8 tokens and retention falls 5 points. Diverse
queries can't absorb blind subtraction.

### MuSiQue

| threshold | ≥0.8 retention | mean query length |
| --------- | --------------:| -----------------:|
| none (control) | 37.7% | 13.1 tokens |
| drop df > 70% | 37.0% | 8.8 |
| **drop df > 50%** | **35.0%** | 7.4 |
| drop df > 30% | 35.7% | 6.0 |

**Δ worst = −2.7 points.** Same shape as HotpotQA — every IDF
threshold regresses retention, with the size of the regression
roughly proportional to how much of the query gets stripped.

## Mechanism — why it fails (sharp)

Template stripping
([CUAD_RECALL_GAP](CUAD_RECALL_GAP.md)) and clause expansion
([CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md)) both work by
exploiting **query-set DF**: the fact that templated workloads share
a wrapper across all queries means high cross-query DF identifies
the wrapper precisely. Discriminators stay because they vary across
queries.

Corpus-side DF doesn't have access to that signal. A token can be:

- **High corpus DF, low query-set DF, discriminative** — e.g.,
  "France" appears in many distractor paragraphs but only in queries
  about France. The auto-drop strips it; user-side
  `analyze_query_set` keeps it because it's not shared across the
  query set.
- **High corpus DF, high query-set DF, boilerplate** — e.g., CUAD's
  "highlight", "contract", "lawyer". Both signals agree; both
  approaches drop it. *This is the overlap region where auto-drop
  works.* But on CUAD, this region is mostly already covered by the
  Tantivy stop-word filter and the natural BM25 IDF math — so there's
  little room left to add value.
- **Low corpus DF, low query-set DF, discriminative** — e.g., a
  specific case name. Both approaches keep it.

The auto-drop misses the second-row distinction (high corpus DF +
discriminative) and that's exactly where diverse workloads live.
HotpotQA / MuSiQue queries are *all* about specific entities whose
names are common in their respective paragraph pools — dropping them
empties the query.

### Why this contrasts with CUAD_PRF_NULL

[CUAD_PRF_NULL](CUAD_PRF_NULL.md) is the additive symmetric failure:
unweighted PRF *adds* corpus-pervasive low-IDF terms (re-injects the
boilerplate). Sub-IDF auto-drop is the subtractive symmetric attempt:
*removes* corpus-pervasive low-IDF terms. Both fail at the same
mechanism boundary: **corpus-only statistics can't replace
query-set or user-curated semantics.**

Combined with CUAD_CLAUSE_EXPANSION (which succeeds, additively, on
user-curated high-IDF terms) and CUAD_RECALL_GAP (which succeeds,
subtractively, on user-derived query-set boilerplate), the pattern
is now clear:

| direction | source | works? | finding |
| --------- | ------ | ------ | ------- |
| subtract  | user-derived query-set DF (template wrapper) | ✓ | CUAD_RECALL_GAP |
| subtract  | corpus-only DF (auto stop list) | ✗ | this null |
| add       | corpus-only DF (PRF top-k) | ✗ | CUAD_PRF_NULL |
| add       | user-curated workload synonyms | ✓ | CUAD_CLAUSE_EXPANSION |

**The four corners settle the design space.** Query-side IDF
manipulation works iff the source of the IDF signal carries semantic
awareness (query-set overlap for the user is doing it, or curated
synonyms for the workload). Corpus-only statistics, in either
direction, fail.

## What this rules in / out

**Rules out:**

- Shipping an auto-default sub-IDF drop in BM25. The HotpotQA / MuSiQue
  regressions disqualify it as a universal default.
- Shipping an opt-in `auto_drop_low_idf=true` config for BM25
  retrieval **with a "just works" claim**. Conditional on the user
  knowing it's a templated workload, but they can already get a
  bigger lift from `analyze_query_set` + `drop_template_terms`.
  Adding a noisier, narrower API doesn't help anyone.

**Rules in (as direction for future work):**

- **The library doesn't need a corpus-side auto-stop-list pipeline.**
  The Tantivy default analyzer + BM25's IDF math already handle the
  trivial boilerplate; the marginal terms that matter need
  query-set or workload-curated signal, both already covered by the
  existing API surface.
- **The four-corner table above is the rule.** When someone proposes
  a new query-manipulation idea, place it on the table. Corpus-only
  approaches in either direction have been falsified twice
  independently.

## Honest limits

- **Three workloads, n=300 each.** The mechanism is sharp; the precise
  numbers would shift on other datasets, but the direction (auto-drop
  hurts diverse) is mechanism-predicted and robust.
- **One threshold parameter.** Tried `{0.70, 0.50, 0.30}` of corpus
  document frequency. A more elaborate scheme (e.g., IDF-weighted
  drop with continuous reweighting instead of binary drop) was not
  tested. The mechanism prediction is the same — corpus-only signal
  is the limiting factor.
- **No BM25 reweighting (only filtering).** A separate question is
  whether *down-weighting* (rather than dropping) low-IDF terms helps.
  BM25 already partially does this via its IDF math; further
  multiplicative down-weighting would amount to per-term query boost,
  which Tantivy supports but wasn't tested.
- **MuSiQue baseline is low (37.7%).** It's a deliberately hard
  multi-hop dataset; the absolute number is below other findings.
  What matters here is the *direction* of the threshold sweep, which
  is consistent with HotpotQA: every threshold regresses retention.

## Reproduce

```bash
cargo run -p redhop-examples --example sub_idf_reweighting_probe --release
```

Runs in well under a minute; no models, no embeddings, no LLM. Uses
the three dataset files already present from earlier findings:
`data/cuad/cuad_sample.json`, `data/hotpotqa/hotpot_dev_distractor_v1.json`,
`data/musique/dev.jsonl`.

## What this changes

Nothing in the runtime, no public API. This is a research record —
evidence that **automatic sub-IDF drop using corpus statistics is
the wrong primitive** for the CUAD-shape recall problem.

The actionable surface remains: `analyze_query_set` +
`drop_template_terms` for query-set-aware subtraction
([QUERY_SET_ANALYZER](QUERY_SET_ANALYZER.md) +
[CUAD_RECALL_GAP](CUAD_RECALL_GAP.md)), and `expand_query_terms` for
workload-curated addition
([CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md)). Both ship across
all three bindings. Both rely on something the library cannot
auto-derive: the user's knowledge of the query-set shape (for
stripping) or the workload taxonomy (for expansion).

If someone (human or AI agent) lands here looking for "can the
library auto-improve CUAD without a user dict": the answer is no via
sub-IDF — the win is in the query-set overlap, and that needs
either the user's queries or the user's knowledge. Read
CUAD_CLAUSE_EXPANSION for what to do instead.
