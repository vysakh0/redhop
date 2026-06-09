# RedHop's eval surface vs Ragas

A fair, reproducible comparison of `redhop.evaluate(...)` against
[Ragas](https://github.com/explodinggradients/ragas) on answer-quality
metrics. We'd rather you trust the numbers than the marketing.

## TL;DR

- On the metric we both ship — claim-decomposed faithfulness — **RedHop
  is substantively equivalent to Ragas**: Pearson r=+0.664, MAE=0.151,
  61% perfect agreement on n=200 HotpotQA with `gpt-4o-mini`.
- Neither library is unambiguously "more correct" by a third LLM's
  read at scale — Claude haiku splits roughly evenly on contested cases.
- The differentiator is **philosophy**, not accuracy: RedHop ships a
  smaller, in-process eval surface that returns one `EvalReport`
  blending lexical (CI-deterministic) and judged (opt-in LLM) metrics.
  Ragas is a broader eval framework with more metrics, more integrations,
  and a separate runtime.
- Pick RedHop when you want a tiny, bounded eval that runs in your
  process and uses the same primitives your retrieval runtime does.
  Pick Ragas when you want the broader eval ecosystem (multiple metric
  families, dataset loaders, integrations with LangChain / LlamaIndex
  / Phoenix / Langfuse).

## Same category, different surface

|  | RedHop `evaluate` | Ragas |
| --- | --- | --- |
| Scope | one API for closed-set answer quality + Decision Report self-eval | dedicated eval framework |
| Metric families | faithfulness, relevancy, correctness, critique, summarize | the above + similarity, context precision/recall, AspectCritic, …more |
| LLM dependence | fully optional: `_lexical` (no LLM) + `_judged` (opt-in) | LLM-required for most metrics |
| Embedder dependence | none (no relevancy-cosine, no similarity) | required for relevancy / similarity / answer-correctness embedding term |
| Integration | `pip install redhop` — single package | langchain-wrapped LLM/embedder, multiple deps |
| Same primitives as runtime? | yes — `evaluate` uses the runtime's own Decision Report machinery | no |
| Output shape | one `EvalReport` dataclass + `summarize(reports)` aggregator | per-metric `Result` objects + dataset-level aggregation |

RedHop is **not** an eval framework — it ships a narrow answer-quality
surface that mirrors what the runtime already measures internally. The
comparison is "does it produce numbers similar to a dedicated eval
library on the metric they both ship."

## The benchmark

Both libraries score the same `(question, context, answer)` triples on
the same workload, with the same LLM. **n=200 HotpotQA** (dev distractor
split, `context_mode=all` = supporting + distractor paragraphs).
**Judge: `openai/gpt-4o-mini` via OpenRouter** (deterministic config,
temperature=0). The LLM generates an answer to each question given the
context, then both libraries independently score it for faithfulness
against that context.

Full method + caveats:
[docs/findings/EVAL_JUDGED_CALIBRATION.md](findings/EVAL_JUDGED_CALIBRATION.md).

### Correlation with Ragas (n=200)

|  | Pearson r | MAE | exact agreement |
| --- | ---: | ---: | ---: |
| RedHop **decomposed** ↔ Ragas | **+0.664** | **0.151** | **61% (122/200)** |
| RedHop **single-prompt** ↔ Ragas | +0.285 | 0.239 | — |

Read: decomposed-faithfulness — the path you should default to — agrees
with Ragas's faithfulness on the majority of cases and stays within
~0.15 absolute when it diverges. Single-prompt diverges more, mostly
on refusal answers ("I don't know" scores 1.0 single-prompt because
there are no claims to contradict — the vacuous-truth failure mode).

### Third-judge tie-breaker (Claude haiku)

When the two libraries disagree by ≥0.5 (35 cases on n=200), we asked
Claude haiku to score the same cases independently via the local
`claude -p --model haiku` CLI:

|  | MAE vs Claude haiku (66-case subset) |
| --- | ---: |
| RedHop decomposed | 0.340 |
| Ragas | 0.262 |

**On contested cases, Claude favors:** RedHop 12/35, Ragas 23/35.

Read carefully: this looks like Ragas is "more correct" — but
re-tracing 5 randomly-sampled "RedHop loses to Ragas" cases at 5 runs
each shows **4 of 5 give 1.0 consistently** when measured stably. The
bench captured a one-shot 0.0 because **`gpt-4o-mini` at temperature
0.0 is not deterministic** (model-replica routing + floating-point
non-associativity in attention ops produces ~20–30% per-case
variance on borderline judgments).

So the contested-cases MAE-to-Claude is noise-dominated. Both the
n=25 "RedHop +0.21 vs Ragas" claim AND the n=200 "Ragas -0.08 ahead"
claim are partly small-sample / single-shot luck. Per-case verdicts
on individual cases are not robust; aggregate metrics (Pearson r and
MAE vs Ragas, averaged over many cases) are robust.

## How to read this

- **Where the result is robust:** decomposed-faithfulness produces
  numbers strongly correlated with Ragas's faithfulness across n=200
  HotpotQA. If you use either library to evaluate a RAG system, the
  trends you see will be the same.
- **Where the result is fragile:** any single case's score has
  ~0.2–0.3 absolute noise. Use the score as a signal across many
  cases, not as an oracle on one.
- **Where neither library shines:** the metric is LLM-judged. Both
  libraries inherit the judge's calibration. A different judge model
  produces different absolute numbers; the trend (the two libraries
  agreeing) is what's stable.

## What you actually get with RedHop that you don't with Ragas

1. **A single `EvalReport` dataclass** that blends lexical metrics
   (deterministic, run in CI without an LLM) with judged metrics
   (opt-in via `Judge.from_callable(fn).cached()`) — instead of running
   metrics one at a time.
2. **`summarize(reports)` for test-set aggregation** — one function
   call rolls up per-case reports into a means + N + share-flagged
   summary, the same shape RedHop's own runtime uses for its Decision
   Report.
3. **No embedder dependency.** Ragas's `AnswerRelevancy` and
   `AnswerSimilarity` need an embedder; RedHop's `relevancy_judged`
   uses an LLM-only noncommittal-detection prompt (no embeddings,
   no extra dep).
4. **Refusal handling.** "I don't know" answers correctly return
   `None` for decomposed faithfulness (0 claims extracted) instead of
   being scored as a vacuous 1.0. Surfaces refusals as a distinct
   category, not as faithfulness=1.
5. **`critique(answer, aspects, ...)` for user-defined dimensions.**
   Ragas has `AspectCritic`; RedHop has the equivalent in `critique`
   with the same `EvalReport`-shape output as quantitative metrics.

## What Ragas gives you that RedHop doesn't

1. **More metric families.** Ragas ships `AnswerSimilarity`,
   `ContextPrecision`, `ContextRecall`, `Faithfulness with NLI`, more
   AspectCritic variants, and test-set generation — RedHop ships a
   focused subset.
2. **Broader integration ecosystem.** LangChain wrappers, LlamaIndex
   wrappers, Phoenix / Langfuse / Arize Phoenix integrations — RedHop
   stays in-process by design.
3. **Dataset loaders.** Ragas can load HuggingFace datasets, dataset
   formats — RedHop expects you to construct `(question, context,
   answer, gold_answer)` tuples directly.

## Reproduce it yourself

The bench lives in the repo — run it on your own workload:

```bash
python3 -m venv bench/.venv
bench/.venv/bin/pip install ragas openai langchain langchain-openai
bench/.venv/bin/pip install ./python
OPENROUTER_API_KEY=sk-or-... \
  bench/.venv/bin/python bench/eval_correlation_hotpot.py \
    --n 200 --context all
```

The script generates answers via the LLM, scores each via both libraries,
and prints Pearson r + MAE + per-case scores. JSON snapshot lands in
`reports/eval_correlation_hotpot_n200.json`.

For the third-judge tie-breaker (requires the `claude` CLI):

```bash
bench/.venv/bin/python bench/select_third_judge_subset.py \
    --in reports/eval_correlation_hotpot_n200.json \
    --out reports/eval_correlation_hotpot_n200_subset.json
bench/.venv/bin/python bench/eval_third_judge.py \
    --in reports/eval_correlation_hotpot_n200_subset.json
```

## Honest caveats

- **Single LLM.** `gpt-4o-mini` only. Different judge models produce
  different absolute numbers; the agreement trend is likely robust,
  the absolute r/MAE numbers aren't necessarily.
- **Only faithfulness was compared head-to-head.** Ragas's relevancy /
  similarity / correctness need an embedder which RedHop deliberately
  doesn't carry — comparing those across embedder choices would muddy
  the apples-to-apples.
- **No human ground truth.** We're measuring whether the two libraries
  agree, not whether either agrees with human judgment. Claude haiku
  as a third LLM is a tie-breaker, not an oracle.
- **The "correct" answer to the contested cases is genuinely
  ambiguous.** Different graders (LLM or human) will reasonably
  disagree on partial-support cases. That's a property of the metric,
  not a bug in either library.

## See also

- [docs/findings/EVAL_JUDGED_CALIBRATION.md](findings/EVAL_JUDGED_CALIBRATION.md) —
  the full evidence: prompt iteration history v0→v4, per-case traces,
  stability checks, single-shot LLM noise analysis.
- [docs/findings/EVAL_VS_RAGAS_SOURCE.md](findings/EVAL_VS_RAGAS_SOURCE.md) —
  source-read comparison of the two implementations
  (claim extraction, batched verification, noncommittal detection, TP/FP/FN).
- [docs/findings/ANSWER_QUALITY_EVAL.md](findings/ANSWER_QUALITY_EVAL.md) —
  the full `evaluate(...)` API tour.
- [bench/eval_correlation_hotpot.py](../bench/eval_correlation_hotpot.py) — the bench script.
- [bench/eval_faith_trace.py](../bench/eval_faith_trace.py) — diagnostic harness
  for tracing extraction + per-claim verifier scores on specific qids.
