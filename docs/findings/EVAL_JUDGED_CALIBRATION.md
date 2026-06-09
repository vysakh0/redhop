# Judged-eval calibration probe

> **Status: bench shipped.** A 5-case hand-curated test set
> demonstrates the full judged surface (`evaluate(..., judge=...)`,
> `decompose_faithfulness=True`, `decompose_correctness=True`,
> `critique(...)`) end-to-end. With a deterministic stub judge,
> **13 of 15 bucket checks pass** (`high`/`mid`/`low` against
> per-case expectations) — and the 2 misses are the cases where a
> token-overlap proxy *should* fail. Validates the wiring; a real LLM
> judge is what closes the calibration gap.

## What's in the bench

`bench/eval_judged_calibration.py` runs RedHop's full judged surface
on a 5-case test set:

| case | what's special | expected behavior |
|---|---|---|
| `CLEAN` | answer paraphrases the context | faithfulness, relevancy, correctness all high |
| `HALLUCINATION` | answer adds unsupported tokens ("notarized affidavit", "$25 fee") | faithfulness low, others mid/high |
| `OFF_TOPIC` | answer is about photosynthesis | all metrics low |
| `WRONG_FACT` | answer says 90 days instead of 30 (gold = 30) | faithfulness + correctness low, relevancy high |
| `REFUSAL` | answer is "I cannot answer that question" | all metrics low |

For each case, the script computes:
- Single-prompt judged metrics (`faithfulness_judged`, `relevancy_judged`, `correctness_judged`)
- Claim-decomposed faithfulness
- Two-aspect critique bundle (conciseness + harmfulness)

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
2 misses are themselves evidence that the upgrade from lexical
(`_lexical`) → judged (`_judged`) matters for fact-correctness
workloads.

The critique scores all sit at 0.5 because the stub returns a neutral
constant for aspect prompts — there's no obvious token-overlap proxy
for "is this concise" or "is this harmful." Those need a real LLM
judge by definition.

## Honest limits

- **n=5 hand-curated.** Verifies the SHAPE of the output is sensible.
  A larger workload (HotpotQA-50, CUAD-50) would tighten any absolute
  numbers.
- **Single LLM model.** The bench defaults to `gpt-4o-mini` for
  cost. Different judge models will produce somewhat different
  scores.
- **Critique scores are stub-neutral.** The stub returns 0.5 for all
  critique prompts because "is this concise" / "is this harmful"
  don't have a sensible token-overlap proxy. Real LLM runs will
  produce real critique scores; the stub is here for wiring
  validation only.

## Side-by-side with Ragas (run 2026-06-09, `openai/gpt-4o-mini` via OpenRouter)

When `ragas` is installed and `OPENROUTER_API_KEY` (or
`OPENAI_API_KEY`) is set, the script runs Ragas's faithfulness on
the same dataset with the same LLM and prints a Pearson r + MAE
agreement matrix.

**Result on this 5-case set (single run):**

| case | RedHop single-prompt | RedHop decomposed | Ragas faithfulness |
|---|---:|---:|---:|
| CLEAN | 1.000 | 1.000 | 1.000 |
| HALLUCINATION | 0.000 | 0.333 | 0.333 |
| OFF_TOPIC | 0.000 | 0.000 | 0.000 |
| WRONG_FACT | 0.000 | 0.500 | 0.500 |
| REFUSAL | 1.000 | 0.000 | 0.000 |

**Pairwise agreement:**

| comparison | Pearson r | MAE |
|---|---:|---:|
| RedHop single-prompt ↔ Ragas | +0.293 | 0.367 |
| RedHop decomposed ↔ Ragas | **+1.000** | **0.000** |

The decomposed-faithfulness path (`decompose_faithfulness=True`)
matches Ragas exactly across all 5 cases — same mechanism (extract
claims, verify each), same numerical outputs. The single-prompt path
gives coarse 0/1 verdicts that miss partial truth (HALLUCINATION:
1 of 3 claims supported = 0.333; WRONG_FACT: 1 of 2 = 0.500). The
REFUSAL case is where single-prompt fails its vacuous-truth check
(no claims = "fully supported" = 1.0); decomposed correctly returns
0 because there are no extracted claims to verify.

**What this proves:**
- The few-shot + batched-verification work brings our faithfulness
  numerically equivalent to Ragas's, for one LLM and one test set.
- The single-prompt path is a useful fast/cheap fallback but
  should not be used when accuracy matters; default to
  `decompose_faithfulness=True` for serious eval runs.

**Honest limits:**
- n=5 hand-curated. A real-workload bench (HotpotQA-50,
  CUAD-50) would strengthen the result; the 5 cases are
  picked to span obvious failure modes, not edge cases.
- Single LLM (`openai/gpt-4o-mini`). Other judge models will
  produce somewhat different absolute numbers, though the
  agreement pattern should hold.
- Only faithfulness was compared. Ragas's answer_relevancy /
  answer_similarity / answer_correctness need an embedder
  (OpenRouter doesn't expose embeddings); routing through a
  second vendor for the comparison would muddy the apples-to-apples.

## Reproduce

```bash
# Stub mode — fast, free, no network
bench/.venv/bin/python bench/eval_judged_calibration.py

# Real OpenAI mode (costs a few cents per run)
OPENAI_API_KEY=sk-... bench/.venv/bin/python bench/eval_judged_calibration.py

# Real OpenAI + side-by-side with Ragas
pip install ragas
OPENAI_API_KEY=sk-... bench/.venv/bin/python bench/eval_judged_calibration.py
```

Output goes to stdout. A machine-readable snapshot is also written to
`reports/eval_judged_calibration.json` for downstream tooling.

Raw stub run kept at
[`reports/eval_judged_calibration_stub_2026-06-09.txt`](../../reports/eval_judged_calibration_stub_2026-06-09.txt).

## See also

- [`ANSWER_QUALITY_EVAL.md`](ANSWER_QUALITY_EVAL.md) — the full
  judged surface (faithfulness / relevancy / correctness / critique
  / summarize) it exercises.
- `bench/eval_judged_calibration.py` — the script. Self-documents its
  test set, its expected buckets, and its stub-judge methodology.
- `crates/redhop/src/critique.rs` and `crates/redhop/src/context/eval.rs`
  — the authoritative metric implementations.
