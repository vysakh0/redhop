# Retrieval & Context Engineering Tips

Practical rules for getting more out of the context you feed an LLM — and how
RedHop's API encodes each one so you don't have to remember them.

These are not paper claims; they are **operational laws** that RedHop's
experiments converged on (n=1,275 hermetic + n=300/200 end-to-end across four
model families, all CI-backed — see [docs/findings/](findings/)). Every rule
below links to the measurement that produced it and the API that applies it.

> **The one rule, if you read nothing else:**
> **Context optimization is conditional. Optimize only under *dilution*, do it
> conservatively, and measure what you removed.** Stuffing a small, clean context
> through untouched is usually the right move; pruning a large, junk-heavy one
> recovers real accuracy. RedHop's `strategy="auto"` makes this decision for you.

---

## Should I optimize this context at all?

```
                ┌─────────────────────────────────────────┐
                │  How big / diluted is the retrieved set?  │
                └─────────────────────────────────────────┘
                                │
        small & focused ────────┼──────── large or junk-heavy
        (≲1.5k tokens,          │         (≳1.5k tokens, many
         few distractors)       │          off-topic chunks)
                │                          │
                ▼                          ▼
        PASS IT THROUGH              PRUNE IT (to budget)
        pruning is neutral-to-       pruning recovers accuracy
        harmful here; you risk       (+0.10 to +0.25 measured);
        dropping reasoning evidence  removes attention dilution
                │                          │
                └────────────┬─────────────┘
                             ▼
                   redhop.build_context(strategy="auto")
                   decides this per-call from input size
```

`strategy="auto"` is the size-gated policy: below `auto_passthrough_max_tokens`
(default 1,500) it passes the context through; above it, it prunes. The gate is
calibrated from a size sweep where pruning helped at every size from ~1.5k tokens
up, with no harmful regime above it ([CONTEXT_DILUTION.md](findings/CONTEXT_DILUTION.md)).

```python
import redhop

ctx = redhop.build_context(
    query=query,
    retrieved_chunks=chunks,   # list of strings or {"text", "id", ...} dicts
    strategy="auto",           # ← decide-when-to-prune, the recommended default
    token_budget=8000,
)
response = llm.generate(ctx.text())
print(ctx.report)              # what it did and why (see "Measure" below)
```

---

## The tips

### 1. Relevance ≠ reasoning usefulness — don't prune by relevance alone
A chunk can be *low-relevance to the query* yet *essential for reasoning* — the
classic case is the **second hop** in multi-hop QA ("X was invented by Davy" →
"Davy was British"; the answer chunk barely matches the query). Aggressive
relevance filtering/reranking systematically removes exactly these.
→ **What to do:** never hard-filter by query relevance on multi-hop. If you
prune, keep low-relevance chunks that are *linked* to a relevant one.
→ **RedHop:** `ReasoningPreserving` (the pruning strategy `Auto` uses) keeps
query-relevant seeds **and** rescues below-bar chunks linked to a seed; only
unlinked junk is dropped. Evidence: [SECOND_HOP_TAX.md](findings/SECOND_HOP_TAX.md),
[REASONING_PRESERVATION.md](findings/REASONING_PRESERVATION.md).

### 2. Removing the wrong chunk is worse than keeping extra junk
Across four model families, **aggressive filtering was net-harmful** (−0.06 to
−0.15 vs. not filtering): the reasoning evidence it incidentally removes costs
more than the distractors it targets. Transformers tolerate irrelevant context
far better than missing reasoning links.
→ **What to do:** bias toward under-filtering. Set a *low* distractor cutoff —
remove only near-zero-overlap junk, not "moderately relevant" chunks.
→ **RedHop:** `distractor_min_grounding` defaults to a deliberately low `0.10`
(only near-zero-overlap chunks are below it). Don't crank it up to "clean
aggressively" — that's the move the measurements punish.

### 3. Optimize under dilution, not by token count
A large context is not the problem; a *diluted* one is. 20k focused tokens beat
5k noisy tokens. The driver is evidence density / irrelevant-token mass, not raw
length.
→ **What to do:** decide to prune based on how junk-heavy the retrieval is, not
just its size — and remember the benefit grows with dilution.
→ **RedHop:** `analyze_context()` reports `input_distractor_ratio`,
`evidence_density`, and `estimated_waste_tokens` *without modifying anything* —
use them to see dilution before you act. `Auto` uses a size gate as a cheap
proxy; for finer control, gate on the distractor ratio yourself.

### 4. The win is *deciding when* to optimize — not a magic algorithm
Naive density-truncation captured essentially the same downstream gain as the
reasoning-aware pruner in the dilution regime (tie on every model tested). No
universally dominant pruning algorithm emerged.
→ **What to do:** don't shop for a clever compressor. Get the *decision* right
(prune iff diluted) and use any sensible pruning underneath.
→ **RedHop:** `Auto` is the decision; the pruning it delegates to is intentionally
simple. The value is the gate + the diagnostics, not secret sauce.

### 5. Stronger rerankers are not universally safer
A uniform cross-encoder reranker *lowered* multi-hop recall (−0.029): it scores
query↔passage relevance and confidently demotes the orthogonal bridge evidence.
→ **What to do:** don't apply a reranker uniformly on multi-hop. Escalate
selectively, or skip it where the second hop matters.
→ **RedHop:** keeps reranking out of the safe default path. Evidence:
[RERANKING_LIMITS.md](findings/RERANKING_LIMITS.md).

### 6. "Retrieve more" is often the wrong fix
ExpandTopK (more similar neighbors) frequently fails on multi-hop, because the
missing evidence is *dissimilar* to the query in embedding space — more neighbors
never reach it.
→ **What to do:** if multi-hop recall is missing, don't widen top-k blindly;
the gap is structural, not a quantity problem.

### 7. Optimization is model-aware
Frontier models (e.g. GPT-4o-mini, Claude Haiku) are surprisingly robust to
distractors; smaller/open models (Llama-3.3-70B, Qwen3.5-flash) are measurably
hurt. The *dilution* recovery is also largest on dilution-sensitive models and
~neutral on robust ones.
→ **What to do:** the same context policy isn't optimal across models. Prune more
willingly for models that struggle with long noisy contexts; for a very robust
model on a modest context, leaving it alone may be best.
→ **RedHop:** `Auto`'s gate is a safe default — it helps where the model is
sensitive and is ~neutral (not harmful) where it isn't. Tune
`auto_passthrough_max_tokens` per deployment.

### 8. Safe optimization is asymmetric — easy to do nothing, hard to over-modify
The most stable success criterion is **avoid damaging recall**, not **maximize
average lift**. Conservative, zero-harm policies outperform eager optimizers
operationally.
→ **What to do:** make "do nothing" the default and intervention the exception.
→ **RedHop:** the default strategy never aggressively prunes by relevance; `Auto`
passes small contexts straight through.

### 9. Treat context optimization as an economic decision
Optimization has costs: reranker compute, embedding latency, token cost, and
attention dilution. The benefit is conditional. So is the spend.
→ **What to do:** weigh the token/latency savings against the (conditional)
quality change — don't optimize reflexively.
→ **RedHop:** the `ContextReport` quantifies what each decision bought
(`total_tokens`, `estimated_waste_tokens`, `retained_evidence_ratio`).

---

## Measure what you did (observability > cleverness)

The most useful thing RedHop gives you is *seeing* dilution, second-hop risk, and
what got rescued — not magic compression. Every `build_context` returns a report;
`analyze_context` gives the same readout **without touching the context**, so you
can decide before acting.

```python
report = redhop.analyze_context(query, chunks)   # pure diagnostics, non-destructive
print(report.strategy)                 # for Auto: "raw_topk" (pass) or "reasoning_preserving" (prune)
print(report.input_distractor_ratio)   # how junk-heavy the retrieval is
print(report.evidence_density)         # query-relevant token fraction
print(report.second_hop_rescue_count)  # reasoning-critical chunks a relevance filter would drop
print(report.estimated_waste_tokens)   # attention spent on distractors
```

Useful fields on the report: `strategy`, `total_tokens`, `n_input_chunks` →
`n_selected`, `input_distractor_ratio`, `evidence_density`,
`retained_evidence_ratio`, `second_hop_rescue_count`,
`reasoning_preservation_delta`, `distractors_pruned`, `estimated_waste_tokens`.

---

## Knobs (and sane defaults)

| Knob | Default | What it does | When to change |
| ---- | ------- | ------------ | -------------- |
| `strategy` | `reasoning_preserving` | how to assemble; use `"auto"` to gate on size | `"auto"` recommended for mixed context sizes |
| `auto_passthrough_max_tokens` | `1500` | `Auto`: pass through at/below, prune above | raise to prune less (more conservative) |
| `token_budget` | — | hard cap on assembled tokens | set to your prompt budget |
| `distractor_min_grounding` | `0.10` | grounding bar below which a chunk is "junk" | keep low; raising it risks the second-hop tax |
| `link_min_jaccard` | `0.12` | linkage at/above which a low-relevance chunk is rescued as a second hop | lower to rescue more aggressively |

---

## Honest limits

- The dilution recovery is **large on dilution-sensitive (frontier) models and
  ~neutral on dilution-robust ones** — it's a "help where possible, harmless
  elsewhere" default, not a guaranteed universal lift.
- The win is **generic pruning** under dilution; the reasoning-aware strategy
  does not beat naive density-truncation downstream, it just keeps you safe on
  multi-hop where naive relevance filtering would tax the bridge.
- Measured on HotpotQA-style multi-hop with lexical grounding; thresholds may
  shift on your workload. The *directions* are robust; the exact numbers are not
  promises.

Full evidence and falsified hypotheses: [docs/findings/](findings/).
