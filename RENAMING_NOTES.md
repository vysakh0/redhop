# Renaming Notes (internal record)

How the NeoRAG → RedHop rename was executed, what was deliberately left
alone, and the residual risks. Companion to [MIGRATION.md](MIGRATION.md)
(the consumer-facing version).

## Method

Token-class passes, applied with path guards so absolute/relative paths to
the *separate* Python lab repo were never rewritten:

- `neorag-*` → `redhop-*` (crate/package/dep names, hyphenated mentions)
- `neorag_*` → `redhop_*` (Rust identifiers, lib names, output filenames)
- `NeoRAG` / `Neo-RAG` / `Neo RAG` → `RedHop` (types, prose, branding)
- `NEORAG` → `REDHOP` (env vars / constants)
- standalone lowercase `neorag` → `redhop`, **excluding** `projects/neorag…`
  and `neorag1`, then a follow-up fix restored `../neorag/` lab-repo paths
  that the absolute-path guard had missed.

Crate **directories** were already un-prefixed (`crates/core`, not
`crates/neorag-core`), so no directory renames were needed — only package
`name`/`dependency` keys.

## Deliberately preserved

| Item | Why |
| ---- | --- |
| `neotrace/1` schema string, `.neotrace.jsonl`, the loader | data-format compatibility (per decision); ~5,190 exported records + lab exporters |
| `../neorag/` and `/Users/vysakh/projects/neorag/` paths | the *separate* Python research-lab repo, not renamed |
| `…/neorag1` workspace directory + its hardcoded paths | renaming the dir is out of scope and would break absolute paths |

## Verification performed

- `cargo build --workspace` + `cargo test --workspace` — all green.
- CLI smoke (`redhop compare`), legacy-name deprecation notice fires.
- Python `basic_rag.py` (rebuilds `redhop_bridge`), dashboard regen, hermetic
  `bench_context_strategies` regen — **0** `neorag` strings in generated
  artifacts (dashboard.html, results.json, SUMMARY.md).
- Residual `neorag` in the tree is **only** the preserved lab-repo / `neorag1`
  paths and the `neotrace` data-format name.

## Residual risks / cleanup suggestions

- **Cargo.lock is gitignored**, so the rename isn't reflected there (regenerated
  on build). Fine for a library workspace.
- **Publish metadata** (`keywords`/`categories`/`repository`) was added to the
  workspace and inherited by the flagship crates (`core`, `context`, `cli`,
  `pipeline`). The remaining crates can inherit it at publish time.
- **Deprecated aliases** (`NeoRAG`, `NeoRAGBuilder`) should be removed in the
  next minor release.
- The `redhop1` directory name is cosmetic only; rename the working directory
  separately if desired (not required for build/publish).
- Bindings (`pyo3`/`napi`) and a published `redhop` PyPI/npm package remain
  future work; the Python API in `examples/python/redhop/` is the intended
  surface for them.
