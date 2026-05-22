# Python usage

```bash
pip install redhop
```

```python
import redhop

ctx = redhop.build_context(
    query="...",
    retrieved_chunks=chunks,          # list[dict | str]
    strategy="reasoning_preserving",  # default
    token_budget=12000,
    distractor_min_grounding=0.10,    # below this query-overlap = distractor
    link_min_jaccard=0.12,            # rescue link threshold (reasoning_preserving)
)
```

## Inputs

A chunk is a string, or a dict with at least `text` and optional `id`,
`source`, `token_count`, `embedding`, `score`. Unknown fields are ignored.

## Return types

`BuiltContext`:
- `.text()` — the assembled prompt string (drop-in for `llm.generate`).
- `.chunks` — selected chunk texts, in order.
- `.report` — a `ContextReport`.

`ContextReport` (getters): `strategy`, `token_budget`, `total_tokens`,
`token_utilization`, `n_input_chunks`, `n_selected`, `input_distractor_ratio`,
`retained_evidence_ratio`, `second_hop_rescue_count`,
`reasoning_preservation_delta`, `distractors_pruned`, `removed_total`,
`evidence_density`, `distractor_ratio`, `estimated_waste_tokens`.
`str(report)` renders the report; `redhop.report_to_dict(report)` returns the
full telemetry as a dict.

## Other functions

```python
redhop.filter_context(query, chunks, strategy="reasoning_preserving")  # no budget cap
redhop.analyze_context(query, chunks)        # non-destructive → ContextReport
redhop.context_economics(query, chunks)      # dict of density/distractor/waste
```

## Local development

```bash
cd python
pip install maturin
maturin develop --release       # build the Rust engine + install editable
python -m pytest tests/
maturin build --release         # produce a wheel in target/wheels/
```
