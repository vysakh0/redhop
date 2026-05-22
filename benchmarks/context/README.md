# benchmarks/context/

Head-to-head benchmark of the four context strategies, driven through the
public `build_context` API + `ContextReport` telemetry.

```bash
cargo run -p neorag-examples --example bench_context_strategies --release
```

Hermetic — no LLM, no embeddings — so it is deterministic and CI-friendly.
Dense-retrieval and cross-generator axes are covered by the onnx/LLM
harnesses (see [../README.md](../README.md)); this isolates the *assembly
strategy* and the second-hop tax.

## What it measures

For each (population × strategy × token-budget):

- **gold_ret** / **second_hop_ret** — label-based retention, with 95%
  bootstrap CIs (the second hop = lowest-grounding gold chunk).
- **density**, **out_distr** (output distractor ratio), **util** (token
  utilization), **tokens**, **rescue** — straight from `ContextReport`.

Two populations, split within HotpotQA:

- **multihop_gap** — the second hop is genuinely less query-relevant than
  the first (the tax regime).
- **shallow_nogap** — hops similarly relevant (a single-hop-like proxy).

## Outputs (committed artifacts)

- `results.json` — metadata (sample, params, n per population) + every
  cell's metrics and CIs. Machine-readable for regression tracking.
- `SUMMARY.md` — human-readable tables.

## Headline (multihop_gap, see SUMMARY.md for the full sweep + CIs)

- **`distractor_filtered` pays the tax and never recovers it:** second-hop
  retention plateaus at **0.749** no matter how large the budget — it
  *filtered the second hop out*, so budget can't bring it back.
- **`reasoning_preserving` lifts that plateau to 0.833** (gold 0.886→0.922)
  while keeping the output distractor ratio low (~0.12 vs raw's ~0.65),
  with a measured ~0.5 rescues/query.
- **`raw_topk` / `max_density`** reach perfect retention only at large
  budgets — by keeping everything, distractors included (out_distr ~0.65).

So the trade is explicit: reasoning_preserving dominates aggressive
filtering on retention at comparable junk suppression, and buys most of
raw's retention while removing most of raw's distractors. Full evidence:
[../../docs/findings/SECOND_HOP_TAX.md](../../docs/findings/SECOND_HOP_TAX.md),
[../../docs/findings/REASONING_PRESERVATION.md](../../docs/findings/REASONING_PRESERVATION.md).
