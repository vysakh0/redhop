# Migration: NeoRAG → RedHop

The project was renamed **NeoRAG → RedHop**. Positioning is unchanged in
spirit: *reasoning-preserving context optimization and retrieval
observability for RAG systems*. No architecture, APIs semantics, findings,
or benchmark methodology changed — this was a naming/packaging pass only.

## Crate mapping

| Old | New |
| --- | --- |
| `neorag-core` | `redhop-core` |
| `neorag-chunking` | `redhop-chunking` |
| `neorag-retrieval` | `redhop-retrieval` |
| `neorag-reranking` | `redhop-reranking` |
| `neorag-diagnostics` | `redhop-diagnostics` |
| `neorag-storage` | `redhop-storage` |
| `neorag-embeddings` | `redhop-embeddings` |
| `neorag-context` | `redhop-context` |
| `neorag-observability` | `redhop-observability` |
| `neorag-orchestration` | `redhop-orchestration` |
| `neorag-calibration` | `redhop-calibration` |
| `neorag-pipeline` | `redhop-pipeline` |
| `neorag-benchmarks` | `redhop-benchmarks` |
| `neorag-examples` | `redhop-examples` |
| `neorag-cli` | `redhop-cli` |

Rust module paths follow: `use neorag_core::…` → `use redhop_core::…`
(every `neorag_*` → `redhop_*`).

## API renames

| Old | New |
| --- | --- |
| `NeoRAG` (facade) | `RedHop` |
| `NeoRAGBuilder` | `RedHopBuilder` |
| `NeoRAG::builder()` | `RedHop::builder()` |

This is a **clean rename** — pre-release, no external users, no semver
obligations. No deprecated aliases, compatibility shims, forwarding
binaries, or legacy crate re-exports are kept. Update call sites directly.

All other public types (`ContextConfig`, `ContextStrategy`, `BuiltContext`,
`ContextReport`, `build_context`, `analyze_context`, `context_economics`,
diagnostics, calibration, …) keep their names unchanged — only their crate
moved from `neorag-*` to `redhop-*`.

## CLI changes

- Binary `neorag` → **`redhop`**. Same subcommands: `compare`,
  `analyze-context`, `benchmark`, `report`. No legacy `neorag` binary or
  forwarding is provided.

## Python

- `import neorag` → **`import redhop`**.
- The Python package is now a real **pyo3 + maturin** native extension under
  `python/` (`pip install redhop`). The earlier stop-gap — a subprocess shim
  (`examples/python/`) that shelled out to a `context_bridge`/`redhop_bridge`
  example binary — has been **removed**; it was only needed before native
  bindings existed. The Python API (`build_context`, `analyze_context`, …) is
  unchanged.

## Intentionally preserved (NOT renamed)

- **NeoTrace data format** — the wire format keeps `schema_version: "neotrace/1"`,
  the `.neotrace.jsonl` extension, and the loader. Renaming would break the
  ~5,190 already-exported records and the (separate) Python-lab exporters.
  Prose now refers to it as "RedHop Trace (neotrace/1 wire format)".
- **The Python research-lab repo** at `../neorag/` (a *separate* repository)
  keeps its name; example/doc paths still point at `../neorag/…`.
- **This workspace directory** (`…/neorag1`) is unchanged; hardcoded absolute
  paths to it are preserved.

## Compatibility policy

Pre-release: nothing was published to crates.io/PyPI/npm under the `neorag`
names, so there are **no external consumers and no semver obligations**. The
rename is therefore clean and complete:

- **No** deprecated type aliases.
- **No** compatibility shims or legacy crate re-exports.
- **No** forwarding `neorag` binary.

Update any external references to the new names directly.
