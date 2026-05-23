# Context Dilution — Does Pruning a Bloated-but-Fitting Context Recover Accuracy? (n=200, CIs)

> **Hypothesis:** when a large context *fits* in the window but is mostly junk, the model still degrades (lost-in-the-middle); pruning it back to the load-bearing evidence recovers accuracy — a value that has nothing to do with hard token budgets.
> **Status:** Confirmed on the dilution-sensitive case, **model-dependent in magnitude**, across 3 model families, n=200, paired bootstrap 95% CIs.
> **Setup:** 200 gap-qualified multi-hop HotpotQA queries; gold scattered through ~1000 off-document distractor chunks (**polluted ≈ 30k tokens**); pruned to a 2k-token budget. Generators: gpt-4o-mini, qwen3.5-flash, llama-3.3-70b; gold-keyword recall.
> **Headline:** stuffing a 30k-token noisy context in **collapses accuracy** (gold→polluted −0.09 to −0.33, sig on all 3); **pruning recovers it where dilution bites hard** (gpt-4o-mini **+0.211**), modestly where it bites less (qwen +0.059), and **not at all on a model that tolerates the bloat** (llama −0.039 ns, because pruning's own gold-loss cancels the small benefit).
> **Reproduce:** `cargo run -p redhop-examples --example emit_dilution --release` then `python python/eval/score_dilution.py --n 200 --model <id>` (raw output in [reports/](../../reports/) `dilution_*.txt`).
> **Justifies API:** `build_context` as a **dilution-mitigation** tool at large context sizes — *not* an accuracy booster at small ones. The value is **generic pruning**, not the `ReasoningPreserving` strategy specifically (see the bridge-aware null below).
> **Caveats:** lexical kw-recall proxy; single dataset; the gain is conditional on (a) a large diluted context and (b) a dilution-sensitive model; `ReasoningPreserving` does **not** beat naive density-truncation downstream in this regime.

---

## Why this experiment exists

The earlier reasoning-QA test ([REASONING_PRESERVATION.md](REASONING_PRESERVATION.md))
ran at a **generous token budget where nothing had to be cut**, and found
pruning ≈ no-op on accuracy. That regime is rigged: when everything fits, any
filter can only *lose* information. It tells you nothing about the regime that
actually matters for a 1M-window world.

The real question is **dilution**: *fitting ≠ using well.* A model with a huge
window still degrades when you actually fill it with mostly-junk
("lost-in-the-middle"). If pruning a bloated-but-fitting context back to the
load-bearing evidence **recovers** accuracy, `build_context` has a real accuracy
home that is independent of any hard budget limit. This is the test the small-
context experiment couldn't see.

## Setup

Per gap-qualified multi-hop HotpotQA query, four contexts from the SAME large
polluted pool (gold scattered, deterministically, through ~1000 off-document
distractor chunks):

| condition | what it is | avg tokens |
| --------- | ---------- | ---------- |
| `ctx_gold_only` | the supporting gold only (clean ceiling) | ~0.25k |
| `ctx_polluted` | gold + ~1000 distractors, **all of it** (stuff-it-all-in) | **~30k** |
| `ctx_pruned` | polluted → `ReasoningPreserving`, pruned to budget | ~1.7k |
| `ctx_topk` | polluted → `MaxDensity`, truncated to budget (naive relevance) | ~2k |

Decisive comparisons (paired bootstrap 95% CI):
- **pruned − polluted** — does pruning the bloat *recover* accuracy? (the library's reason to exist at scale)
- **pruned − topk** — does bridge-aware pruning beat *naive* truncation at the same budget?

## Results

| model | gold | polluted | pruned | topk | **recovery** (pruned−poll) | bridge (pruned−topk) | dilution hit (gold−poll) |
| ----- | ---- | -------- | ------ | ---- | -------------------------- | -------------------- | ------------------------ |
| gpt-4o-mini | 0.727 | **0.402** | 0.614 | 0.614 | **+0.211** [.140,.281] ✓ | −0.000 [−.045,.045] ns | +0.325 [.255,.390] ✓ |
| qwen3.5-flash | 0.746 | 0.541 | 0.600 | 0.579 | **+0.059** [.000,.120] borderline | +0.021 [−.022,.065] ns | +0.205 [.141,.265] ✓ |
| llama-3.3-70b | 0.701 | 0.612 | 0.574 | 0.568 | **−0.039** [−.114,.037] ns | +0.005 [−.045,.055] ns | +0.089 [.014,.159] ✓ |

## Reading the result

**1. Dilution is real on every model.** The clean→polluted gap (gold − polluted)
is significant for all three (+0.089 to +0.325). Filling the window with 30k
tokens of mostly-junk measurably hurts answer quality even though it *fits*. On
gpt-4o-mini the collapse is dramatic — 0.727 → 0.402, with refusals tripling
(14% → 43%): the model simply fails to find the needle in the haystack.

**2. Pruning recovers accuracy in proportion to how hard dilution hits.** This is
the causal pattern, and it's clean:
- **gpt-4o-mini** is crushed by the bloat → pruning recovers a large, significant
  **+0.211** (two-thirds of the lost ground; refusals fall 43% → 24%).
- **qwen3.5-flash** is moderately hit → modest, borderline recovery **+0.059**.
- **llama-3.3-70b** barely cares about the bloat (only −0.089) → there is almost
  nothing to recover, and pruning's *own cost* (it drops ~9% of gold; refusals
  rise 10% → 26%) **cancels the small benefit** and tips slightly negative.

So the honest law is conditional: **pruning pays when the bloat-removal benefit
exceeds the gold-loss cost — i.e. when the model is dilution-sensitive enough.**
It is not a universal win.

**3. The win is generic pruning, NOT the bridge-aware strategy.** `pruned − topk`
is **not significant on any model** (−0.000 to +0.021). `ReasoningPreserving`
does not separate from plain `MaxDensity` density-truncation downstream. Worse:
under the tight dilution budget, its link-rescue admits extra chunks and crowds
out the very second hop it's meant to protect (second-hop retention 80% vs topk's
86% at emit time). **The differentiation that motivates `ReasoningPreserving`
does not earn its complexity in the dilution regime** — any sensible pruning
captures the entire recoverable gain.

## What is established vs what is not

**Established (CI-backed):**
- Large, noisy, *fitting* contexts dilute accuracy on all 3 models (sig).
- Pruning **recovers a large, significant fraction** of that loss on a
  dilution-sensitive frontier model (gpt-4o-mini, +0.211), and a modest amount on
  qwen (borderline).
- The recovery comes from **pruning per se**; the bridge-aware strategy adds no
  measurable downstream value here (pruned−topk ns on all models).

**Not established / honest caveats:**
- **Not universal.** On a model that tolerates long noisy contexts (llama here),
  pruning offers no net benefit and can slightly hurt by dropping borderline gold.
  The gain is conditional on context size *and* model dilution-sensitivity.
- The 2k budget is aggressive; a gentler prune would keep more gold and might turn
  llama's small negative neutral — untested, a tuning question, not a claim.
- Lexical kw-recall proxy; single dataset (HotpotQA); n=200.

## Where this leaves the product

This is the experiment that decides the library, and the verdict is **alive, with
an honest and conditional claim**:

- **Small contexts / everything fits:** don't prune — it's a wash-to-harmful
  (the reasoning-QA finding). 
- **Large, diluted contexts on a dilution-sensitive model:** prune — it recovers
  large, significant accuracy.

The crossover *is* the product thesis: **context optimization is useless-to-harmful
under headroom and strongly positive under dilution.** And because the value is
generic pruning, `build_context` should ship as **a well-instrumented pruner + the
second-hop-tax diagnostic**, not as a claim that `ReasoningPreserving` beats
naive truncation.

## Size sweep — pinning the crossover (gpt-4o-mini, n=120 per size)

To set the `Auto` gate from data, we swept the input size (number of injected
distractors) and measured pruning's recovery at each:

| distractors | polluted tokens | polluted | pruned | recovery (pruned−poll) | sig |
| ----------- | --------------- | -------- | ------ | ---------------------- | --- |
| 50 | 1,545 | 0.597 | 0.699 | **+0.102** [+.040,+.167] | ✓ |
| 150 | 4,500 | 0.562 | 0.673 | **+0.110** [+.040,+.185] | ✓ |
| 300 | 8,932 | 0.507 | 0.672 | **+0.165** [+.091,+.242] | ✓ |
| 600 | 17,763 | 0.487 | 0.652 | **+0.165** [+.087,+.249] | ✓ |
| 1000 | 29,542 | 0.394 | 0.648 | **+0.253** [+.167,+.342] | ✓ |

**There is no harmful regime above ~1.5k tokens.** Pruning recovers accuracy at
*every* size tested — significantly — and the benefit grows monotonically with
dilution (+0.10 at 1.5k → +0.25 at 30k). The crossover sits *below* the smallest
sweep point: between the reasoning-QA regime (~0.5k tokens / 8 distractors, where
pruning was neutral-to-harmful) and ~1.5k / 50 distractors (where it already
helps +0.10). `bridge-aware: pruned − topk` stayed ns at every size — the win is
generic pruning across the whole curve.

**Gate decision:** `auto_passthrough_max_tokens = 1500` — the conservative low
edge of the measured-benefit range. Prune above it (where every measured point
shows a CI-significant gain), pass through below it (where evidence is absent and
the second-hop tax counsels caution). Calibrated on gpt-4o-mini; the gain is
realized on dilution-sensitive (frontier) models and is ~neutral on dilution-
robust ones (llama), so the gate is a "help where possible, harmless elsewhere"
default.

## Next (measurement, not architecture)

1. ~~Size sweep to pin the `Auto` gate~~ **(done — gate set to 1500, above.)**
2. ~~A size-gated `Auto` policy~~ **(done — `ContextStrategy::Auto`, gated on
   input size, shipped with tests.)**
3. **Refine the sub-1.5k crossover** — one more sweep point (~20 distractors,
   ~0.7k tokens) would pin where benefit begins, between the 8- and 50-distractor
   regimes.
4. **Larger dilution** (100k+) — does the recovery keep growing, and does llama
   cross into positive once the bloat actually hurts it?
