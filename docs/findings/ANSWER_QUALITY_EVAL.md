# Answer-quality eval — the full surface

> **Status: shipped (Rust + Python + Node).** Lexical and judged
> answer-quality metrics live under `evaluate(...)`. Aspect critique
> lives under `critique(...)`. Test-set aggregation lives under
> `summarize(reports)`.
>
> Comparing to other eval libraries? See
> [**docs/COMPARISON_RAGAS.md**](../COMPARISON_RAGAS.md) for the
> Ragas head-to-head (n=200 HotpotQA, r=+0.664).

RedHop's eval has two complementary surfaces:

- **`evaluate(query, ctx, answer=, gold_answer=, judge=, ...)`** —
  closed-set metrics every RAG pipeline cares about: faithfulness,
  relevancy, correctness, context recall/precision, plus self-eval
  signals from the Decision Report.
- **`critique(answer, aspects, judge=, ...)`** — open-ended
  user-defined dimensions: harmfulness, conciseness, brand voice,
  whatever the caller pins.

Each metric on the report has one of three forms:

- `_lexical` — deterministic, no LLM, runs in CI on every PR.
  Token-overlap proxy. Catches obvious failure modes (fabricated
  tokens, off-topic answers, wrong-token outputs) but won't catch a
  confidently-wrong paraphrase.
- `_judged` — LLM-scored, opt-in via `judge=`. Same conceptual
  metric as the lexical version, but uses an LLM to score; catches
  the paraphrase-aware failures the lexical version can't.
- Diagnostic counters (prefixed `n_`) — surface intermediate classifier
  counts so callers can debug WHY a metric landed where it did
  (e.g. how many claims were extracted, how many were TP vs FP vs FN).

## The Judge

`evaluate(..., judge=...)` and `critique(..., judge=...)` both accept
a `Judge` constructed from the caller's LLM client:

```python
from openai import OpenAI

client = OpenAI()

def score(prompt: str, system: str | None) -> float:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system or ""},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return float(resp.choices[0].message.content.strip())

judge = redhop.Judge.from_callable(score, name="gpt-4o-mini").cached()
```

`.cached()` memoizes identical `(prompt, system)` pairs so re-runs
across CI / debugging are free. The cache key is namespaced by the
judge's `name=` so swapping models doesn't reuse stale scores.

A judge error on any single metric leaves only THAT metric `None` —
other metrics are unaffected. Eval is best-effort; a transport blip
doesn't crash the run.

## Faithfulness — `evaluate(..., decompose_faithfulness=True)`

By default, `faithfulness_judged` is a single LLM call: "is the answer
supported by the context?" One prompt, one number, fast and cheap.

For more accurate scoring on partial-truth answers, opt in to claim
decomposition:

```python
report = redhop.evaluate(
    query, ctx,
    answer=ans,
    judge=judge,
    decompose_faithfulness=True,
)
print(report.faithfulness_judged)              # mean per-claim score
print(report.n_faithfulness_claims_extracted)  # e.g. 4
print(report.n_faithfulness_claims_supported)  # e.g. 3 (claims scoring ≥ 0.5)
```

How it works:
1. Extract atomic claims from the answer (1 LLM call, few-shot prompt).
2. Verify all claims in a single batched LLM call (`N: SCORE` per line).
3. Final score = mean of per-claim verifications.

Cost: 2 LLM calls regardless of claim count (vs 1 for the default).
A claim counts as "supported" when its verification score ≥ 0.5.

## Correctness — `evaluate(..., decompose_correctness=True)`

By default, `correctness_judged` is a single LLM call: "does the
answer match the gold?" One prompt, one number.

For diagnostic detail (which facts were missed vs hallucinated), opt in
to claim classification:

```python
report = redhop.evaluate(
    query, ctx,
    answer=ans,
    gold_answer=gold,
    judge=judge,
    decompose_correctness=True,
)
print(report.correctness_judged)   # F1 over (TP, FP, FN)
print(report.n_correctness_tp)     # claims in answer supported by gold
print(report.n_correctness_fp)     # claims in answer NOT in gold (hallucinated)
print(report.n_correctness_fn)     # claims in gold NOT in answer (missed)
```

How it works:
1. Extract claims from the answer (1 LLM call).
2. Extract claims from the gold (1 LLM call).
3. Classify each answer claim as TP/FP and each gold claim as FN (1 LLM call).
4. Final score = `2·TP / (2·TP + FP + FN)` (F1).

Cost: 3 LLM calls. The diagnostic counters are the value — for
"this answer is wrong, why?" the TP/FP/FN breakdown often makes the
failure mode obvious.

## Relevancy

Single LLM call. The prompt explicitly handles noncommittal answers
("I don't know", "It depends", refusal of any form) — those score 0
regardless of vocabulary overlap with the question. A noncommittal
answer that happens to mention query terms doesn't accidentally score
high.

## Aspect critique — `critique(answer, aspects, judge=, ...)`

For arbitrary qualitative dimensions:

```python
aspects = [
    redhop.Aspect(
        "harmfulness",
        "Does the ANSWER contain harmful, offensive, or unsafe content?",
        high_is_good=False,   # high LLM verdict → low final score
    ),
    redhop.Aspect(
        "conciseness",
        "Is the ANSWER free of unnecessary repetition or padding?",
    ),
]
report = redhop.critique(
    answer,
    aspects,
    judge=judge,
    context=ctx.text(),  # optional
    query=q,             # optional
)
for name, score in report.scores:
    print(f"{name}: {score}")
print(report["harmfulness"])  # also indexable by name
```

`high_is_good=False` inverts the LLM's raw score so high values mean
"good answer" across the report regardless of an aspect's polarity.
One LLM call per aspect (cached on `Judge.cached()`).

## Test-set aggregation — `summarize(reports)`

```python
reports = []
for query, gold_answer in test_set:
    ctx = doc.context(query)
    answer = my_llm(query, ctx.text)
    reports.append(redhop.evaluate(
        query, ctx, answer=answer, gold_answer=gold_answer, judge=judge,
    ))

summary = redhop.summarize(reports)
print(f"n={summary.n}  overall={summary.mean_overall:.3f}")
print(f"faithfulness_judged: {summary.mean_faithfulness_judged:.3f} "
      f"({summary.n_with_faithfulness_judged}/{summary.n})")
```

Each `Option<f32>` field aggregates over the subset where it was
populated; the `n_with_<field>` counters surface how big each subset
was. A mean from 3 of 200 reports should look different from one from
200 of 200 — the counter makes that visible.

## Node availability

Lexical metrics: same surface as Python via sync `evaluate(...)`.

Judged metrics + critique: async on Node — `evaluateWithJudge(query,
ctx, judge, options)` and `await critique(answer, aspects, judge,
options)`. JS is single-threaded; callbacks can't fire during a sync
napi call, so the binding moves the eval onto a tokio worker, calls
back into JS via a `ThreadsafeFunction`, and resumes when the
callback settles. The Judge callback signature is
`(err, prompt, system) => number | string` — `err` is the napi error
channel (null on the normal path).

```javascript
const judge = redhop.Judge.fromCallable(async (err, prompt, system) => {
  if (err) throw err;
  const resp = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system ?? "" },
      { role: "user", content: prompt },
    ],
    temperature: 0,
  });
  return parseFloat(resp.choices[0].message.content.trim());
}, "gpt-4o-mini").cached();

const report = await redhop.evaluateWithJudge(query, ctx, judge, {
  answer,
  goldAnswer,
  decomposeCorrectness: true,
});
```

## What's distinctive

- **`overall` composite** in `[0, 1]` blending whichever metrics are
  populated, with `low_confidence_retrieval` capping it at 0.25 on
  weak retrieval. Single headline number.
- **Self-eval signals** computed from the runtime's Decision Report:
  `mean_grounding`, `evidence_density`, `retained_evidence_ratio`,
  `second_hop_rescues`, `low_confidence`, `estimated_waste_tokens`.
  External eval libraries don't have access to these because they're
  runtime-internal.
- **Lexical metrics that run in CI** — deterministic token-overlap
  proxies that catch obvious regressions without an LLM.
- **Built-in cached Judge** — re-runs are free on identical prompts.
  Determinism in CI without payment surprises.
- **Three-language surface** — Python (full), Node (full via async),
  Rust (core).

## See also

- `crates/redhop/src/context/eval.rs` — authoritative metric semantics.
- `crates/redhop/src/critique.rs` — aspect critique.
- `crates/redhop/src/judge.rs` — Judge trait + Cache + parse_score.
- [`EVALUATE_API.md`](EVALUATE_API.md) — the underlying
  "refraction not independent measurement" design.
- [`EVAL_JUDGED_CALIBRATION.md`](EVAL_JUDGED_CALIBRATION.md) — a
  5-case calibration probe + reproducible bench.
