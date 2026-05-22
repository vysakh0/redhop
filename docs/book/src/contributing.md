# Contributing

Full guide: [`CONTRIBUTING.md`](https://github.com/redhop/redhop/blob/main/CONTRIBUTING.md).

## Build & test

```bash
cargo build --workspace && cargo test --workspace
cd python && maturin develop --release && python -m pytest tests/
cargo build -p redhop-cli --release
```

## What we welcome

Bug fixes, docs, examples, and especially **new findings** — measured
experiments, including ones that *falsify* a hypothesis. Add a doc under
`docs/findings/` using the template (hypothesis · setup · metrics · failure
cases · interpretation · caveats · reproduce command · verdict).

## What's out of scope

RedHop is not a framework: no agents, planners, workflow/orchestration DAGs,
graph traversal, query decomposition, embedded LLM/vector DB, or RL controllers.

## Discipline

- Don't over-claim; report effect sizes with caveats and CIs.
- Don't sanitize; an unstable or negative result *is* the finding.
- Rust stays the source of truth; bindings wrap, they don't fork logic.
- Defaults change only with evidence.
