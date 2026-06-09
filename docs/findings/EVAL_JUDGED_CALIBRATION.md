# Eval Tier-2 — calibration probe + Ragas comparison harness

> **Status: bench shipped (Phase 8). Real-LLM numbers pending an API
> key.** A 5-case hand-curated test set demonstrates the full Tier-2
> surface (`evaluate(..., judge=...)`, `decompose_faithfulness=True`,
> `critique(...)`) end-to-end. With a deterministic stub judge,
> **13 of 15 bucket checks pass** (`high`/`mid`/`low` against
> per-case expectations) — and the 2 misses are the cases where a
> token-overlap proxy *should* fail. Validates the wiring; the LLM
> judge is what closes the calibration gap.

## What's in the bench

`bench/eval_judged_calibration.py` runs RedHop's full Tier-2 surface
on a 5-case test set:

| case | what's special | expected behavior |
|---|---|---|
| `CLEAN` | answer paraphrases the context | faithfulness, relevancy, correctness all high |
| `HALLUCINATION` | answer adds unsupported tokens ("notarized affidavit", "$25 fee") | faithfulness low, others mid/high |
| `OFF_TOPIC` | answer is about photosynthesis | all metrics low |
| `WRONG_FACT` | answer says 90 days instead of 30 (gold = 30) | faithfulness + correctness low, relevancy high |
| `REFUSAL` | answer is "I cannot answer that question" | all metrics low |

For each case, the script computes:
- Single-prompt Tier-2 metrics (faithfulness_judged, relevancy_judged, correctness_judged)
- Claim-decomposed faithfulness (Phase 6)
- Two-aspect critique bundle (Phase 7): conciseness + harmfulness

…then checks whether each metric landed in the expected `high` /
`mid` / `low` bucket.

## Two judge modes

The bench supports two judges, controlled by environment:

- **Deterministic stub** (default — no API key, no network). Computes
  scores by token-overlap on the prompt's CONTEXT / ANSWER / QUESTION
  blocks. Honest about what it can and can't do; the misses are
  educational.
- **Real OpenAI judge** (if `OPENAI_API_KEY` is set and `openai` is
  installed). Real LLM calls, real numbers, costs real money. Uses
  `gpt-4o-mini` by default.

The script picks the right judge at startup; the banner names which.

## Results — stub judge

```
case             faith   relev    corr  faith_d  claims  critique
CLEAN            1.000   0.750   1.000   1.000     1/1   conciseness=0.500 harmfulness=0.500
HALLUCINATION    0.357   0.750   0.500   0.357     0/1   conciseness=0.500 harmfulness=0.500
OFF_TOPIC        0.000   0.000   0.000   0.000     0/1   conciseness=0.500 harmfulness=0.500
WRONG_FACT       0.875   0.750   0.833   0.875     1/1   conciseness=0.500 harmfulness=0.500
REFUSAL          0.000   0.000   0.000   0.000     0/1   conciseness=0.500 harmfulness=0.500
```

Bucket check: **13 of 15 metric-cases land in the expected bucket.**

The 2 misses are both on `WRONG_FACT`:

- `faithfulness` expected `low` (the answer says 90 days, ctx says 30) but landed `high` (0.875).
- `correctness` expected `low` (90 ≠ 30) but landed `high` (0.833).

This is **exactly where a token-overlap proxy breaks** — the answer
shares almost all its tokens with the context and the gold; only one
number differs. A real LLM judge would catch the contradiction. The
2 misses are themselves evidence that the upgrade from Tier-1
(`_lexical`) → Tier-2 (`_judged`) matters for fact-correctness
workloads.

The critique scores all sit at 0.5 because the stub returns a neutral
constant for aspect prompts (there's no obvious token-overlap proxy
for "is this concise" or "is this harmful" — those need a real
judge by definition).

## Side-by-side with Ragas

The bench is wired to also run Ragas's `faithfulness`,
`answer_relevancy`, and `answer_similarity` on the same dataset when
both Ragas and an `OPENAI_API_KEY` are present:

```bash
pip install ragas
OPENAI_API_KEY=sk-... bench/.venv/bin/python bench/eval_judged_calibration.py
```

When both run, the script prints:
- The full per-case table for both frameworks side-by-side.
- Per-metric **Pearson correlation** and **mean absolute error**
  between RedHop and Ragas on the matching pairs
  (`faithfulness_judged` vs Ragas faithfulness,
  `relevancy_judged` vs `answer_relevancy`,
  `correctness_judged` vs `answer_similarity`).
- Total wallclock for each framework.

That's the real "Ragas parity validation" run. We didn't include the
numeric output in this commit because (a) it costs money to run and
(b) the bench is run-able by anyone with a key; canning specific
numbers from one run-with-one-model would imply more authority than
the run deserves.

## Honest limits

- **n=5 hand-curated.** Verifies the SHAPE of the output is sensible.
  A larger workload (HotpotQA-50, CUAD-50) would tighten any
  agreement numbers from the Ragas comparison.
- **Single LLM model.** The bench defaults to `gpt-4o-mini` for
  cost. Different judge models will produce somewhat different
  scores; that's true for both frameworks.
- **"Agreement" isn't ground truth.** Both frameworks are LLM-judged
  and both have prompt-specific variance. Big divergence is
  interesting; full agreement is also interesting; neither validates
  "the metric is correct" in an absolute sense.
- **Critique scores are stub-neutral.** The stub returns 0.5 for all
  critique prompts because "is this concise" / "is this harmful"
  don't have a sensible token-overlap proxy. Real LLM runs will
  produce real critique scores; the stub is here for wiring
  validation only.
- **No batching.** Both frameworks pay per-query LLM cost. Ragas's
  `evaluate(dataset, metrics)` may batch internally; RedHop's
  cache makes re-runs free but a fresh run is 4-8 LLM calls per
  case.

## Reproduce

```bash
# Stub mode — fast, free, no network
bench/.venv/bin/python bench/eval_judged_calibration.py

# Real OpenAI mode (costs a few cents per run)
OPENAI_API_KEY=sk-... bench/.venv/bin/python bench/eval_judged_calibration.py

# With Ragas side-by-side
pip install ragas
OPENAI_API_KEY=sk-... bench/.venv/bin/python bench/eval_judged_calibration.py
```

Output goes to stdout. A machine-readable snapshot is also written to
`reports/eval_judged_calibration.json` for downstream tooling.

Raw stub run kept at
[`reports/eval_judged_calibration_stub_2026-06-09.txt`](../../reports/eval_judged_calibration_stub_2026-06-09.txt).

## See also

- [`EVAL_RAGAS_PARITY.md`](EVAL_RAGAS_PARITY.md) — the overall
  Phase-1-through-Phase-7 surface and side-by-side with Ragas as a
  conceptual comparison.
- `bench/eval_judged_calibration.py` — the script. Self-documents its
  test set, its expected buckets, and its stub-judge methodology.
- `crates/redhop/src/critique.rs` and `crates/redhop/src/context/eval.rs`
  — the authoritative metric implementations.
