# Eval parity with Ragas — what `redhop.evaluate` measures, side-by-side

> **Status: in-progress.** The four-phase eval expansion that closes the
> gap to Ragas has landed three of four phases — Tier-1 lexical
> proxies, the Judge trait scaffolding, and Tier-2 LLM-judged metrics
> via Python `Judge.from_callable(...)`. Phase 4 adds `summarize()`
> for test-set aggregation. Node Tier-2 callback surface ships in Phase 5
> as the async `evaluateWithJudge(...)` entry-point.
>
> The goal isn't to BE Ragas — it's to **ship the Ragas metric set
> plus what's distinctive about RedHop**, in-process with no extra
> infra. This doc maps the surface so users can decide whether they
> need both or just one.

## Quick comparison

| metric / capability | Ragas | RedHop |
|---|---|---|
| **Faithfulness** | LLM-judged (claim extraction + entailment) | `faithfulness_lexical` (free, in CI) + `faithfulness_judged` (LLM) |
| **Answer relevancy** | LLM-judged | `relevancy_lexical` + `relevancy_judged` |
| **Context precision** | LLM-judged + classic (recall@k) | `context_precision` (chunk-set; gold required) |
| **Context recall** | LLM-judged + classic | `context_recall` (chunk-set; gold required) |
| **Answer correctness** | LLM-judged + classic | `correctness_lexical` + `correctness_judged` |
| **Aspect critique** | LLM-judged | not yet |
| **Self-eval (no gold, no LLM)** | none | `mean_grounding`, `evidence_density`, `retained_evidence_ratio`, `second_hop_rescues`, `low_confidence`, `estimated_waste_tokens` |
| **Composite score** | none (raw metrics only) | `overall` ∈ [0,1] (weighted blend) |
| **Determinism in CI** | requires LLM (non-deterministic + paid) | Tier 1 deterministic, Tier 2 deterministic via `Judge.cached()` |
| **In-process** | needs OpenAI/HF client setup | yes (Tier 1); yes with BYO client (Tier 2) |
| **Single dependency** | Ragas + HF + pandas + LLM SDK | one package, optional LLM SDK only when using Tier 2 |
| **Multi-language** | Python only | Python (full), Node (full — Tier 1 sync `evaluate`, Tier 2 async `evaluateWithJudge`), Rust |

## The two-tier model

RedHop's eval has **two tiers in the same API**, distinguished by
field suffix:

- `_lexical` — Tier 1, deterministic token-overlap proxy. Free, no
  LLM, runs in CI on every PR. Catches obvious failure modes
  (fabricated tokens, off-topic answers, wrong-token outputs) but
  won't catch a confidently-wrong paraphrase.
- `_judged` — Tier 2, LLM-scored. Same conceptual metric as Ragas's
  faithfulness / answer relevancy / answer correctness, with prompts
  calibrated to be short and parseable by small models.

Same `EvalReport` shape, same `evaluate()` call. Add `judge=judge` to
unlock the `_judged` fields:

```python
import redhop
from openai import OpenAI

client = OpenAI()

def score(prompt, system):
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

# In CI — Tier 1 only, free, deterministic.
report = redhop.evaluate(query, ctx, answer=ans, gold_answer=gold)

# Before promoting a config — Tier 1 + Tier 2.
report = redhop.evaluate(query, ctx, answer=ans, gold_answer=gold, judge=judge)
# .faithfulness_lexical AND .faithfulness_judged both populated; compare them.
```

### Using Tier 2 from Node.js

Same `Judge.fromCallable(fn).cached()` shape; the only API difference
is that Node's Tier-2 entry-point is **async** (returns a Promise)
because JS callbacks can't fire during a sync napi call. Sync
`evaluate(...)` still works for Tier-1-only callers.

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI();

const judge = redhop.Judge.fromCallable(async (prompt, system) => {
  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system ?? "" },
      { role: "user", content: prompt },
    ],
    temperature: 0,
  });
  return parseFloat(resp.choices[0].message.content.trim());
}, "gpt-4o-mini").cached();

// In CI — Tier 1 only, sync.
const report1 = redhop.evaluate(query, ctx, { answer, goldAnswer });

// With a judge — async, populates Tier 1 + Tier 2.
const report2 = await redhop.evaluateWithJudge(query, ctx, judge, {
  answer, goldAnswer,
});
// report2.faithfulnessJudged, .relevancyJudged, .correctnessJudged
```

A JS-side exception in the callable leaves the corresponding `_judged`
metric as `null` (same semantics as Python) — failure is isolated, the
process doesn't crash, lexical fields stay populated.

## Test-set aggregation: `summarize(reports)`

The Ragas equivalent of looping over a dataset and computing means is
`redhop.summarize(reports)`:

```python
reports = []
for query, gold_answer in test_set:
    ctx = doc.context(query)
    answer = my_llm(query, ctx.text)
    reports.append(redhop.evaluate(
        query, ctx, answer=answer, gold_answer=gold_answer, judge=judge,
    ))

summary = redhop.summarize(reports)
print(f"n={summary.n}  overall={summary.mean_overall:.3f} (median {summary.median_overall:.3f})")
print(f"faithfulness_judged: {summary.mean_faithfulness_judged:.3f} "
      f"({summary.n_with_faithfulness_judged}/{summary.n})")
```

`summarize` aggregates each `Option<f32>` field over the **subset of
reports where it was populated**, and emits `n_with_<field>` so callers
can see how big each subset was. A mean computed from 3 out of 200
reports should look different from one computed from 200 out of 200 —
the counter makes that visible.

## What we have over Ragas

1. **`overall` composite.** Single headline number in `[0, 1]` blending
   whichever metrics are populated. Ragas leaves you with 5+ raw
   numbers to interpret; the composite is a deliberate opinion about
   which metrics matter more (gold-relative > judged > lexical >
   self-eval).
2. **Self-eval metrics (no gold, no LLM).** Ragas has nothing like
   `evidence_density`, `retained_evidence_ratio`, `second_hop_rescues`,
   `low_confidence`. These are diagnostic signals computed from the
   runtime's own Decision Report — useful in development before you
   have a gold set.
3. **`low_confidence_retrieval` is a hard cap on `overall`.** When the
   runtime itself says "this retrieval was weak", the composite is
   capped at 0.25 regardless of the answer-side metrics — so a
   confidently-wrong answer on a noise-only context can't accidentally
   score high.
4. **`Judge.cached()` is deterministic across re-runs.** Same prompt
   produces the same score. Ragas's LLM-judged metrics are
   non-deterministic by default; you pay every time you run the eval.
   RedHop's cache lives inside the Judge object, so a second pass over
   the same dataset has zero LLM cost.
5. **Sync API.** Ragas is async-heavy; RedHop's `evaluate` is sync,
   matching how most Python eval scripts and CI runners are
   structured. Async users wrap their LLM client themselves.

## What Ragas has over us

1. **Aspect critique** (harmfulness, conciseness, malicious-intent,
   etc.). Tier-2 in RedHop only ships the three core metrics today.
   The Judge trait is general — adding aspect-critique prompts is
   straightforward — but it's not yet built in.
2. **Larger calibrated metric set.** Ragas has more specialized
   metrics for specific failure modes (e.g. NoiseSensitivity,
   ResponseRelevancy variants). RedHop sticks to the three load-bearing
   ones.
3. **HuggingFace `datasets` integration.** Ragas operates natively on
   HF dataset objects. RedHop's `summarize(reports)` takes a list of
   `EvalReport` — users can build that loop themselves but don't get
   `evaluate.from_dataset(...)` ergonomics for free.
4. **Multiple LLM-judge providers built in.** Ragas ships adapters for
   OpenAI, Anthropic, Azure, etc. RedHop ships none — the user wraps
   their preferred client in `Judge.from_callable(fn)`. Five lines but
   not zero.

## When to reach for which

- **CI on every PR.** RedHop Tier-1 (`_lexical` fields). Free,
  deterministic, no API key. Ragas has no equivalent.
- **Sampled production traffic, before promoting a config.** Either
  RedHop Tier-2 with `Judge.cached()` (cheaper after the first run) or
  Ragas with your own caching layer.
- **Eval suite with HF datasets you already have.** Ragas integrates
  more directly; bridge to RedHop via `redhop.evaluate(...)` in your
  per-row loop.
- **You want a headline composite + the actionable
  `low_confidence_retrieval` signal.** RedHop only. The composite +
  the cap are RedHop-specific.
- **You want aspect critique (harmfulness, etc.).** Ragas only today.

## Honest limits

- **Tier-2 metric calibration is "designed-not-measured".** Our
  prompts were written to be short and parseable, calibrated against
  small models like gpt-4o-mini. We haven't published a side-by-side
  numeric comparison against Ragas's prompts on a standard workload.
  That bench is the right follow-up; it doesn't change the API shape
  either way.
- **Node Tier-2 lives on the async `evaluateWithJudge` path.** Sync
  `evaluate(...)` always leaves `_judged` as `null` because JS callbacks
  can't safely fire during a sync napi call (single-threaded JS).
  `evaluateWithJudge` is async and returns a Promise; same
  `Judge.fromCallable(fn).cached()` shape as Python under the hood. See
  the Node example in the "Node availability" section.
- **Single Judge call per metric.** We don't decompose the answer into
  individual claims before judging (Ragas does, for faithfulness).
  Simpler prompt, lower cost, but potentially lower precision than a
  claim-by-claim verification. We'll revisit if a measured bench shows
  the gap is large enough to matter.

## Reproduce

```bash
# Tier-1 unit tests (Rust + Python + Node)
cargo test -p redhop --features files,semantic --lib context::eval
python/.venv/bin/pytest python/tests/test_evaluate.py
cd nodejs && npm test

# Tier-2 unit tests (Rust + Python; uses stub callable judges, no LLM)
cargo test -p redhop --features files,semantic --lib judge
python/.venv/bin/pytest python/tests/test_evaluate.py -k tier2
```

## See also

- [`EVALUATE_API.md`](EVALUATE_API.md) — the original eval-design
  document. Explains "refraction not independent measurement" and
  the relationship between `evaluate.overall` and the runtime's
  `low_confidence_retrieval`.
- `crates/redhop/src/context/eval.rs` — the implementation
  (authoritative on metric semantics).
- `crates/redhop/src/judge.rs` — the Judge trait, CallableJudge,
  CachedJudge, and `parse_score`.
