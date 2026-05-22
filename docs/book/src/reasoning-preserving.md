# ReasoningPreserving

The default strategy, and the one that resists the [second-hop tax](./findings-second-hop-tax.md).

## How it works

Two classes of chunk are kept:

1. **Seeds** — chunks above the query-grounding bar (`distractor_min_grounding`).
   The clearly query-relevant evidence.
2. **Rescued second hops** — chunks *below* the bar that are lexically **linked**
   to a seed (term-set Jaccard ≥ `link_min_jaccard`), i.e. they share the bridge
   entity with relevant evidence.

Everything else — low query relevance **and** unlinked to any seed — is dropped
as true junk.

```text
query ──relevance──▶ [seed]  ──shares "Humphry Davy"──▶ [second hop]   ✅ kept
                     [seed]                                            ✅ kept
                              (low relevance, unlinked) [distractor]   ❌ dropped
```

This is a **single linkage step at assembly time** — not graph traversal, not
iterative retrieval, not query decomposition. No graph is built; it is the
minimal operation that distinguishes a distractor from a reasoning-critical
second hop.

## What it costs

The rescue is not free: it readmits a little junk that happens to be lexically
linked to a seed. The tradeoff is "keep a bit more junk to save a lot more
second hops" — a good trade exactly in the aggressive-filtering regime where the
tax is expensive. The [reasoning-preservation finding](./findings.md) measured
this end-to-end (n=300, CIs): it beat aggressive filtering, and the gain was
**causally localized** to the rescued evidence.

## Tuning

- `distractor_min_grounding` (default 0.10) — the seed bar. Lower = more
  permissive seeds.
- `link_min_jaccard` (default 0.12) — the rescue link threshold. Higher = stricter
  about what counts as "linked to a seed".

On tiny corpora the lexical signal is stopword-sensitive; at dataset scale the
defaults are what the findings use. A semantic-linkage variant (embedding
similarity instead of lexical Jaccard) is on the roadmap as a signal upgrade.

## Limits

The link signal is **lexical** by default: a second hop that shares no surface
tokens with its seed (pure paraphrase) won't be rescued. And retention is
*reachability*, not a guarantee of a correct answer — it ensures the needed
evidence survives; the model still has to use it.
