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

Two bench scripts:
1. `bench/eval_judged_calibration.py` — the 5-case hand-curated probe
   above (wiring + edge-case bucket check). Also runs Ragas
   side-by-side when installed.
2. `bench/eval_correlation_hotpot.py` — a real-workload Pearson
   r / MAE measurement on HotpotQA. The honest numbers live here.

### 5-case probe (extreme failure modes)

| case | RedHop single-prompt | RedHop decomposed | Ragas |
|---|---:|---:|---:|
| CLEAN | 1.000 | 1.000 | 1.000 |
| HALLUCINATION | 0.000 | 0.333 | 0.333 |
| OFF_TOPIC | 0.000 | 0.000 | 0.000 |
| WRONG_FACT | 0.000 | 0.500 | 0.500 |
| REFUSAL | 1.000 | 0.000 | 0.000 |

Decomposed-faithfulness matches Ragas exactly here, but these are
all extreme cases — clean signal, no ambiguity. The interesting
question is whether the match holds on noisy real-world inputs. It
does not, exactly. See below.

### Real-workload (HotpotQA n=25, full distractor context)

The LLM generates an answer to each question given the full HotpotQA
distractor context (supporting paragraphs + distractor paragraphs).
RedHop and Ragas both score faithfulness against that context.

| comparison | Pearson r | MAE |
|---|---:|---:|
| RedHop single-prompt ↔ Ragas | **-0.059** | **0.367** |
| RedHop decomposed ↔ Ragas | **+0.559** | **0.187** |

Breakdown of decomposed-vs-Ragas agreement on n=25:
- 13 of 25 cases agree perfectly (delta = 0.000).
- 22 of 25 cases agree within ±0.4 (88%).
- 3 of 25 cases have large divergence (delta ≥ 0.5), all in the same
  direction: RedHop decomposed scores 1.0 while Ragas scores 0.0.

**What this means:**
- **The single-prompt path is NOT a stand-in for Ragas.** r=-0.059
  means essentially no correlation; the anti-trend comes from
  vacuous-truth on refusals (single-prompt gives 1.0 for "I don't
  know" answers because there are no claims to contradict).
- **The decomposed path is *similar* to Ragas, not identical.** r=0.56
  with MAE=0.19 says they trend together but disagree on ~3-4 cases
  per 25 by a meaningful margin. The 5-case probe's perfect agreement
  was misleading because all 5 cases were extreme.
- **The disagreement direction is informative.** When the two diverge,
  RedHop decomposed scores HIGHER than Ragas — suggesting our
  extraction produces fewer claims OR our verifier is more permissive
  than Ragas's. Worth investigating in a follow-up if absolute
  calibration to Ragas matters.

### Third-judge tie-breaker (Claude haiku via `claude -p`)

Two LLM-judge libraries disagreeing doesn't tell us which one is
"correct." So we ran the same 25 (context, answer) pairs through a
third, independent judge: **Claude haiku**, invoked via the local
`claude -p --model haiku` CLI. The bench is
[`bench/eval_third_judge.py`](../../bench/eval_third_judge.py); raw
output at
[`reports/eval_third_judge_n25.txt`](../../reports/eval_third_judge_n25.txt).

For each library we compute MAE to Claude's score — lower means
closer to an independent third opinion.

| library | MAE to Claude haiku | n |
|---|---:|---:|
| RedHop decomposed | **0.212** | 24 |
| Ragas | 0.262 | 24 |
| RedHop single-prompt | 0.175 | 24 |

(One case dropped: Claude's reply was unparseable as a single
number.)

**Read carefully:**

1. **RedHop decomposed is ~0.05 closer to Claude than Ragas is.** Not
   a huge margin, but consistent.
2. **Single-prompt's low MAE is misleading here.** This n=25 has
   mostly non-refusal answers where single-prompt scores 1.0 and
   Claude also tends toward 1.0; the vacuous-truth failure on
   refusals (visible in the n=15 distractor-only run) is what
   actually rules it out.
3. **On the 4 contested cases** (|RedHop_decomp − Ragas| ≥ 0.5),
   Claude is closer to RedHop on 2 and closer to Ragas on 2 — a tie.
   The genuine disagreements split down the middle, so neither
   library's verdict is consistently the "right" one in the third
   judge's view.

**What it doesn't prove.** Claude haiku and RedHop's judge
(`gpt-4o-mini`) are both modern LLMs and may share calibration habits
that Ragas's older verifier prompt doesn't. So the result is
"RedHop is not *worse* than Ragas under an independent LLM's view,"
not "RedHop is correct." A human ground-truth pass on the contested
cases would close that gap.

### Honest limits

- **n=25.** A reasonable sanity check but not a benchmark. n=200 with
  the same setup would be more authoritative.
- **Single LLM (`openai/gpt-4o-mini`).** Different judge models will
  produce different absolute numbers and could shift the agreement
  pattern. The trend (decomposed similar to Ragas, single-prompt
  anti-correlated) is likely robust; the absolute r/MAE numbers
  aren't.
- **Only faithfulness was compared.** Ragas's relevancy / similarity /
  correctness need an embedder (OpenRouter doesn't expose
  embeddings); cross-vendor would muddy the apples-to-apples.
- **The "correct" answer is unknown for both libraries.** We're
  measuring whether RedHop and Ragas agree, not whether either
  agrees with human judgment.

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
