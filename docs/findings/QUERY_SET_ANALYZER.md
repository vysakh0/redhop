# `analyze_query_set` — cross-workload validation of a templated-query diagnostic

> **Status:** **Confirmed on the extremes; boundary behavior bounded
> but unmeasured.** Calibration probe: 7 workloads × n=300 queries.
>
> - **Obviously-distinct workloads (5):** precision 1.00, recall 1.00.
>   The two templated workloads (CUAD; a synthetic support-ticket
>   frame) fire; the three diverse natural-language workloads (HotpotQA,
>   MuSiQue, synthetic free-text) stay quiet. Boilerplate-term
>   extraction surfaces the actual template tokens, not corpus-pervasive
>   function words.
> - **Boundary-adjacent workloads (2):** both stay quiet at
>   `template_word_share` 0.291 and 0.334 — comfortably below the 0.50
>   threshold. The shipped threshold is on the **conservative** side:
>   workloads with a short fixed prefix (≤30% shared words) won't trigger
>   a recommendation. Whether that's "the right call" depends on
>   measurement against the user's own corpus.
>
> **What the calibration does NOT tell you:** the exact crossover point.
> All measured workloads are either well above 0.50 (fire) or well below
> 0.50 (quiet). The space 0.334–0.50 is unprobed.
>
> **TL;DR:** [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) showed that template
> stripping at the query boundary lifts CUAD ≥0.8 retention 82% → 88%,
> but offered no way for users to *detect* the same dilution pattern on
> their own workloads. `analyze_query_set` takes a representative
> sample of queries and reports whether they share enough boilerplate
> to be templated, which terms are doing the dilution, and what to do
> about it. Probe confirms the heuristic does not light up on diverse
> queries.

## Question

[CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) discovered that BM25 dilution
from query-side template boilerplate costs ≥0.8 retention measurably on
CUAD (4 points to LlamaIndex; recoverable to a 2-point lead with a
6-line stripper). The mechanism is general — any workload with
templated queries on a corpus that doesn't share the boilerplate will
hit the same dilution.

But mechanisms aren't products. A user with a different workload has
no way to know whether *their* queries are templated enough for
stripping to matter, which terms are the boilerplate, or what
preprocessor to write. The natural follow-up: **can we offer a small
diagnostic that takes a sample of the user's queries and tells them
whether they have a CUAD-shape dilution problem?**

Two failure modes to guard against:

- **False positive on diverse workloads.** If `analyze_query_set` fires
  on natural-language QA queries (HotpotQA, support-style queries), it
  pushes users toward a workaround that won't help. Worse than no
  diagnostic.
- **False negative on CUAD.** If `analyze_query_set` doesn't detect the
  canonical templated workload, the heuristic is too conservative to
  ship.

This finding measures both.

## Heuristic

(`crates/redhop/src/analyzer.rs::analyze_query_set`.)

1. **Tokenize.** For every alphanumeric token of length ≥ 2 in each
   query (lowercase, split on non-alphanumeric — matches the
   tokenization used across the CUAD harnesses, so the analyzer sees
   what BM25 will see), record per-query token sets.
2. **Query-set document frequency.** For each token, count the number
   of queries that contain it.
3. **Boilerplate = high-DF tokens.** Tokens with `df / n_queries ≥
   0.80` (appearing in 80%+ of the query set).
4. **Template word share.** Mean across queries of `(boilerplate-token
   count in query) / (total tokens in query)`. CUAD measures ~0.66 here.
5. **`is_templated`.** True when share ≥ 0.50 **and** at least 2
   boilerplate terms. Both conjuncts matter: a single shared word
   ("the") in diverse queries is not a template, and a high share with
   zero boilerplate is impossible.
6. **Dilution cost bands:** High (≥ 0.70), Medium (0.40–0.70),
   Low (0.20–0.40), None (< 0.20).
7. **`suggested_action`.** A workload-shape recommendation tailored to
   each (templated × cost) combination, pointing at the relevant
   findings doc and the [`drop_template_terms`] helper. Conservative on
   Medium — recommends an A/B rather than asserting a lift, because the
   analyzer can't know whether the user's specific workload behaves like
   CUAD beyond having a similar shape.

Stop-word filtering is **intentionally absent**. If "the" appears in
every query, it's still boilerplate from BM25's perspective and worth
surfacing.

## Probe — three workloads, n = 300 each

Harness:
[`crates/examples/examples/query_set_analyzer_probe.rs`](../../crates/examples/examples/query_set_analyzer_probe.rs).
No models, no embeddings, no retrieval — string-only diagnostic.

| workload  | expected behavior | n   | `template_word_share` | `is_templated` | `cost` band | boilerplate term count |
| --------- | ----------------- | --- | ---------------------:| --------------:|:----------- | ----------------------:|
| **CUAD**     | should fire   | 300 | **0.663**             | **true**       | Medium      | **15**                 |
| HotpotQA  | should not fire   | 300 | 0.000                 | false          | None        | 0                      |
| MuSiQue   | should not fire   | 300 | 0.118                 | false          | None        | 1 (`the`)              |

The CUAD boilerplate terms surfaced by the analyzer match the template
verbatim — `any, be, by, contract, details, highlight, if, lawyer, of,
parts, related, reviewed, should, that, the`. The discriminator (the
quoted clause name and the `Details:` elaboration) is correctly
excluded.

HotpotQA and MuSiQue are independent multi-hop QA workloads with
diverse natural-language queries; neither shares a template. The
analyzer reports zero (HotpotQA) and one (MuSiQue: `the`) boilerplate
terms — neither passes the `≥ 2 terms` guard, so neither fires.

**Both failure modes ruled out at this sample size.** Heuristic ships.

## Why CUAD lands in Medium, not High

CUAD's 0.66 share is below the High threshold (0.70). This is
correct, not a bug: the CUAD query is the 24-word fixed template plus a
variable `Details:` elaboration of typically 5–10 content tokens, so
roughly 65–70% of the total query is boilerplate. The CUAD finding's
"79% boilerplate" number was over the *template-only* portion (19 of
24 words), not the full query including the elaboration. The
analyzer's number is the apples-to-apples share over what BM25 will
actually see.

The Medium band's `suggested_action` recommends an A/B before
committing — that's the right call when the analyzer can't measure the
user's actual retention, only the shape. The user follows the link to
[CUAD_RECALL_GAP](CUAD_RECALL_GAP.md), runs the A/B harness pattern on
their own data, and decides.

## What this also rules in / out

**Rules in:** shipping `analyze_query_set` as public Rust API, plus
`drop_template_terms` as the mechanical helper that consumes the
analyzer's output. Both are documented in the analyzer module
(`crates/redhop/src/analyzer.rs`) with examples and re-exported at the
crate root.

**Rules out** (by design — discipline carried over from
[CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) and
[CUAD_PRF_NULL](CUAD_PRF_NULL.md)):

- A *built-in* CUAD-specific stripper. Templates are workload-specific;
  RedHop ships the analyzer that finds the boilerplate and the helper
  that drops it, but the user still owns the call to which boilerplate
  applies to their workload.
- A diagnostic that claims a *numeric retention lift*. The analyzer
  measures the *shape* of the query set, not the user's actual
  workload retention. Claiming "this will give you +6 points" without
  measuring on the user's documents would be the same overclaiming
  pattern the CUAD finding called out.

## Calibration (added in 0.3.1 audit, 2026-06-08)

Run via `bench/.venv/bin/python bench/query_set_analyzer_calibration.py`.
Seven workloads — five obviously-distinct (two positives + three
negatives) plus two **boundary-adjacent** workloads designed to land
near the threshold so we learn where the heuristic actually flips.
All n=300.

### Obviously-distinct workloads

| workload                              | label    | fires | template_word_share | boilerplate count | dilution |
|---------------------------------------|----------|-------|--------------------:|------------------:|----------|
| CUAD                                  | positive | ✓     | 0.663               | 17                | medium   |
| Synthetic template (support-ticket)   | positive | ✓     | 0.936               | 19                | high     |
| HotpotQA                              | negative | —     | 0.000               | 0                 | none     |
| MuSiQue                               | negative | —     | 0.118               | 1                 | none     |
| Synthetic diverse                     | negative | —     | 0.000               | 0                 | none     |

**On the extremes:** Confusion TP=2 / FP=0 / FN=0 / TN=3 → precision
1.00, recall 1.00. The 0.50 `template_word_share` threshold cleanly
separates these — but P=R=1.00 here only tells you the heuristic
distinguishes obviously-templated from obviously-diverse. It does
**not** tell you where the boundary breaks down.

### Boundary-adjacent workloads (where does it actually flip?)

| workload                                 | fires | template_word_share | boilerplate count |
|------------------------------------------|-------|--------------------:|------------------:|
| Boundary synthetic (~30% prefix share)   | quiet | 0.291               | 2                 |
| HotpotQA / "What is" prefix-filtered     | quiet | 0.334               | 4                 |

Both stay below the 0.50 threshold and don't fire. **The threshold sits
on the conservative side**: a workload where ~30% of words are shared
(a brief fixed prefix on otherwise-varied content) won't trigger a
Stripper recommendation. The closest measured firing point is CUAD at
0.663, the closest measured quiet point is HotpotQA-What-is at 0.334.
The actual crossover sits somewhere in `0.334 < threshold ≤ 0.50` (the
hard cutoff); we haven't probed it with finer-grained workloads.

What this means practically:

- If your workload has a short fixed prefix (≤4 shared opening words
  on an otherwise-diverse body), the analyzer will be quiet. Whether
  that's the right call depends on whether stripping a 4-word prefix
  would actually lift retention on your corpus — measure with
  `evaluate(..., gold_chunks=...)` rather than relying on the analyzer
  alone.
- If your workload has CUAD-style ≥50% boilerplate, the analyzer fires
  reliably.
- The space in between (40%-50% shared) is **measurement-undetermined**
  in this calibration. The shipped threshold won't fire there; whether
  that's a feature (avoids spurious recommendations) or a bug (misses
  light templates) requires a workload near that boundary, which we
  haven't tested.

The synthetic positive (`Please help me with my <topic> issue, my
account is broken…`) is a deliberately heavier-boilerplate support-
ticket frame than CUAD's 24-word legal template, confirming the
heuristic fires on workloads heavier than CUAD too.

## Honest limits

- **Five workloads is still a small sample.** Three positives are
  legal-template (CUAD), support-ticket (synthetic), and synthetic
  template; three negatives are HotpotQA, MuSiQue, and synthetic-diverse.
  Real-world templated-but-not-frame workloads (clinical SOAP-note
  queries, code-review templates, financial 10-K extractive QA) are
  not yet measured. The conservative `≥ 0.50 AND ≥ 2 terms` thresholds
  were chosen with room for those cases; the calibration confirms they
  don't fire on the negatives we have, but doesn't prove they won't on
  unseen domains.
- **Single configuration.** Thresholds (0.80 for boilerplate
  membership; 0.50 + 2-term floor for `is_templated`; 0.70 / 0.40 /
  0.20 for cost bands) were tuned to the three workloads measured here.
  No sweep. The numbers are interpretable as "what worked on this
  measurement", not "the optimal thresholds across all workloads."
- **No bootstrap CIs on the shares.** Sample-size variance at n=300 is
  small for the metric (it's a stable mean across queries), but we
  haven't quantified it.
- **No downstream answer eval.** Whether following the
  `suggested_action` produces a measured retention lift on a *new*
  workload is the next-level question. CUAD itself was the only
  workload where the lift was measured directly
  ([CUAD_RECALL_GAP](CUAD_RECALL_GAP.md)).
- **English tokenization.** Boilerplate detection is over lowercased
  alphanumeric tokens. Non-English workloads will tokenize correctly
  but the action copy mentions English findings. Multilingual probe is
  a future arc.

## Reproduce

```bash
cargo run -p redhop-examples --example query_set_analyzer_probe --release
```

The harness expects `data/cuad/cuad_sample.json`,
`data/hotpotqa/hotpot_dev_distractor_v1.json`, and
`data/musique/dev.jsonl` (CUAD path overridable via
`REDHOP_CUAD_PATH`). The probe runs end-to-end in well under a
second; no models, no embeddings.

## What this changes

- New public API: `redhop::analyze_query_set`,
  `redhop::drop_template_terms`, `redhop::QuerySetReport`,
  `redhop::DilutionCost`. Re-exported from the crate root.
- `docs/CHOOSING_A_CONFIG.md` "Templated queries with heavy
  boilerplate" section gains the analyzer call as the first step
  (detect → strip → A/B), with `drop_template_terms` as the mechanical
  helper for the second step.
- The pattern documented in this finding (and in
  [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md)) becomes
  *self-discoverable*: a user lands on their own templated workload,
  runs `analyze_query_set` on a sample, and learns from the
  `suggested_action` what to do next, without having to read the CUAD
  story first.
