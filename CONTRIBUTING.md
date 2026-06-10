# Contributing to RedHop

Thanks for your interest. RedHop is a small, deliberately bounded library: the
best contributions sharpen what exists rather than expand scope.

## Build & test

```bash
cargo build --workspace        # pure-Rust workspace (no Python needed)
cargo test --workspace

# Optional features on the redhop crate: `semantic` (ONNX embeddings +
# cross-encoder reranker) and `files` (Document.from_file / from_folder
# parsing). Mirror this when running examples or the per-crate tests.
cargo test -p redhop --features files,semantic

# Python bindings (needs maturin in a virtualenv). The `files,semantic`
# features mirror the published wheel — without them, test_loader_errors
# and test_rerank can't exercise the right surface.
cd python && maturin develop --release --features files,semantic && python -m pytest tests/

# Node binding
cd nodejs && npm install && npm run build && npm test

# CLI
cargo build -p redhop-cli --release   # → target/release/redhop
```

The hermetic benchmark and examples run offline:

```bash
cargo run -p redhop-examples --example bench_context_strategies --release
python python/examples/compare_strategies.py
```

## Checks (what CI enforces)

Run these before opening a PR (CI runs the same set):

```bash
# Rust
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check                  # licenses + advisories  (cargo install cargo-deny)

# Python (in the venv, with the extension built — see Build & test above
# for the right `maturin develop` invocation)
cd python
ruff check . && ruff format --check .      # pip install ruff
python -m pytest tests/ -q

# Node (build the addon, then run both smoke + analyzer suites)
cd nodejs && npm run build && npm test

# Coverage (optional, local)
cargo llvm-cov --workspace                 # cargo install cargo-llvm-cov
```

## What we welcome

- Bug fixes, docs improvements, more examples, better error messages.
- New **findings**: measured experiments, including ones that *falsify* a
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
- **Keep Rust the source of truth.** Bindings (Python/CLI) wrap the core API.
  Don't fork logic into them.
- **Defaults change only with evidence.** A new default needs a finding doc.

## PRs

Keep PRs focused. Include: what changed, why, and how you verified it
(tests/benchmarks/examples). If you touched a default or a strategy, link the
finding that justifies it. Run `cargo test --workspace` and the Python tests
before opening.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
