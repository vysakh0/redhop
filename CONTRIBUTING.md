# Contributing to RedHop

Thanks for your interest. RedHop is a small, deliberately bounded library — the
best contributions sharpen what exists rather than expand scope.

## Build & test

```bash
cargo build --workspace        # pure-Rust workspace (no Python needed)
cargo test --workspace

# Python bindings (needs maturin in a virtualenv)
cd python && maturin develop --release && python -m pytest tests/

# CLI
cargo build -p redhop-cli --release   # → target/release/redhop
```

The hermetic benchmark and examples run offline:

```bash
cargo run -p redhop-examples --example bench_context_strategies --release
python python/examples/compare_strategies.py
```

## What we welcome

- Bug fixes, docs improvements, more examples, better error messages.
- New **findings** — measured experiments, including ones that *falsify* a
  hypothesis. These are first-class. Add a doc under `docs/findings/` using the
  template (hypothesis · setup · metrics · failure cases · interpretation ·
  caveats · reproduce command · verdict) and link it from the index.
- Signal improvements that don't change the architecture class (e.g. a
  semantic-linkage rescue variant behind the existing strategy).

## What's out of scope

RedHop is not a framework. We will decline additions that introduce agents,
planners, workflow/orchestration DAGs, graph traversal, query decomposition, an
embedded LLM/vector DB, or RL controllers.

## Discipline (the part that matters)

The project's credibility is the evidence layer. Please:

- **Don't over-claim.** Report effect sizes with caveats and CIs where possible.
- **Don't sanitize.** If a result is unstable or negative, that *is* the finding.
- **Keep Rust the source of truth.** Bindings (Python/CLI) wrap the core API;
  don't fork logic into them.
- **Defaults change only with evidence.** A new default needs a finding doc.

## PRs

Keep PRs focused. Include: what changed, why, and how you verified it
(tests/benchmarks/examples). If you touched a default or a strategy, link the
finding that justifies it. Run `cargo test --workspace` and the Python tests
before opening.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
