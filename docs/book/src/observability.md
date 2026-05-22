# Observability

Every assembly emits a `ContextReport` — the record of what the optimizer did.
Most RAG stacks have none of this; they stuff the top-k and hope.

## The report

```text
Context Optimization Report
───────────────────────────
Strategy: ReasoningPreserving

Input chunks:        8
Output chunks:       2
Tokens:              100 → 30  (-70%)
Distractors pruned:  6
Reasoning rescues:   1

Evidence density:    0.10 → 0.20
Retained evidence:   100%
Token utilization:   0%
Estimated waste:     13 tokens on distractors

Warnings:
- rescued 1 low-relevance linked chunk(s) (possible second hops)
```

`str(report)` (Python) / `report.render(before)` (Rust) produces this. Pass the
`analyze_context` report as `before` to get the token/density **deltas**.

## Fields

| field | meaning |
| ----- | ------- |
| `total_tokens` / `token_utilization` | tokens used; fraction of budget |
| `n_input_chunks` / `n_selected` | sizes in and out |
| `input_distractor_ratio` | how distractor-heavy the retrieval was |
| `retained_evidence_ratio` | seeds kept / seeds in (a label-free gold proxy) |
| `second_hop_rescue_count` | low-relevance chunks deliberately rescued |
| `reasoning_preservation_delta` | chunks a plain filter would have dropped |
| `distractors_pruned` / `removed_total` | per-reason removals |
| `evidence_density` / `distractor_ratio` | of the assembled context |
| `estimated_waste_tokens` | tokens spent on distractor chunks |

Get the full telemetry as data: `redhop.report_to_dict(report)` (Python) or
serialize the Rust struct (it is `Serialize`). The CLI's `analyze-context` and
`compare` surface the same numbers; `report` renders artifacts to markdown/HTML.

## Why it matters

The whole point of RedHop is that context tradeoffs should be **visible**. When
a strategy drops a chunk, you can see *why* (distractor / redundant / budget),
how many reasoning hops were rescued, and how much attention budget went to
waste — before it silently costs you an answer.
