# Context strategies

A strategy decides *which* retrieved chunks make it into the final context
under the token budget. RedHop ships five; `reasoning_preserving` is the
default.

| strategy | what it does | recall-safety | when |
| -------- | ------------ | ------------- | ---- |
| `reasoning_preserving` *(default)* | keep query-relevant seeds **and** rescue low-relevance chunks linked to a seed; drop only unlinked junk | multi-hop safe | general / multi-hop |
| `distractor_filtered` | drop everything below a query-grounding bar | safe **only** at a low threshold | single-hop, or low threshold |
| `max_density` | greedily pack the highest evidence-density chunks | recall-risky on multi-hop | single-hop / brutal budgets |
| `redundancy_pruned` | skip chunks too similar to one already kept (needs embeddings) | neutral | duplicated corpora |
| `raw_topk` | keep retrieval order until the budget fills | keeps everything (incl. junk) | baseline / no optimization |

## The threshold-vs-ranking distinction

There are two ways to prune, with different risk:

- **Absolute-threshold** (`distractor_filtered`: "drop below a grounding bar")
  is recall-safe **at a low bar** — it only removes near-zero-overlap junk. Raise
  the bar and it starts taxing the second hop.
- **Relative-ranking** (`max_density`: "keep the top by density") is
  recall-risky on multi-hop — the low-relevance second hop loses the ranking
  competition even though the answer needs it.

Both converge with `raw_topk` at large budgets — context economics only *bites*
under attention scarcity.

## Side-by-side

Run `redhop compare` or `python python/examples/compare_strategies.py` to see
all strategies on one retrieval set. On a multi-hop question, you'll see
`distractor_filtered` and (under budget) `max_density` drop the reasoning-
critical hop, while `reasoning_preserving` keeps it. See
[ReasoningPreserving](./reasoning-preserving.md) for how the rescue works, and
[Findings](./findings.md) for the measurements.
