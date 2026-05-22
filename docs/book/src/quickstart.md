# Quickstart

## Install (Python)

```bash
pip install redhop          # native wheel; no Rust toolchain needed to use it
```

## Use it

`retrieved_chunks` accepts plain Python — a list of dicts (only `text` is
required) or bare strings:

```python
import redhop

chunks = [
    {"id": "c1", "text": "...", "score": 0.82},
    {"id": "c2", "text": "..."},
    "a bare string also works",
]

ctx = redhop.build_context(
    query="who ...?",
    retrieved_chunks=chunks,
    strategy="reasoning_preserving",
    token_budget=12000,
)

print(ctx.text())             # the assembled prompt context
print(ctx.report)             # the Context Optimization Report
r = ctx.report
print(r.total_tokens, r.distractors_pruned, r.second_hop_rescue_count)
print(redhop.report_to_dict(r))   # full telemetry as a dict
```

A `ContextReport` looks like this:

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
```

## Run the examples (offline)

```bash
python python/examples/basic_rag.py
python python/examples/compare_strategies.py
python python/examples/economics_demo.py
python python/examples/dashboard.py && open python/examples/dashboard.html
```

## Next

- [Python usage](./usage-python.md) · [Rust usage](./usage-rust.md) · [CLI](./usage-cli.md)
- [Context Strategies](./strategies.md) — pick the right one.
