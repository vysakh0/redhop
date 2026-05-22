# Usage

RedHop exposes the same small, stable surface in every language. Rust is the
source of truth; the Python bindings and the CLI wrap it (no logic is
duplicated).

| Function | Purpose |
| -------- | ------- |
| `build_context(query, chunks, strategy, token_budget, ...)` | budget-aware assembly → `BuiltContext` |
| `filter_context(query, chunks, strategy, ...)` | filter junk, **no** budget truncation → `BuiltContext` |
| `analyze_context(query, chunks, ...)` | non-destructive diagnostics → `ContextReport` |
| `context_economics(query, chunks, ...)` | economics of a set as-is |

- [Python](./usage-python.md)
- [Rust](./usage-rust.md)
- [CLI](./usage-cli.md)
