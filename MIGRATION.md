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

| Old | New | Compatibility |
| --- | --- | --- |
| `RedHop` facade (was `NeoRAG`) | `RedHop` | `pub type NeoRAG = RedHop` kept, `#[deprecated]` |
| `NeoRAGBuilder` | `RedHopBuilder` | `pub type NeoRAGBuilder = RedHopBuilder` kept, `#[deprecated]` |
| `NeoRAG::builder()` | `RedHop::builder()` | works via the alias (deprecated) |

All other public types (`ContextConfig`, `ContextStrategy`, `BuiltContext`,
`ContextReport`, `build_context`, `analyze_context`, `context_economics`,
diagnostics, calibration, …) keep their names unchanged — only their crate
moved from `neorag-*` to `redhop-*`.

## CLI changes

- Binary `neorag` → **`redhop`**. Same subcommands: `compare`,
  `analyze-context`, `benchmark`, `report`.
- If the binary is invoked under a legacy `neorag` name (e.g. a symlink), it
  prints `warning: \`neorag\` is deprecated; the binary is now \`redhop\`` to
  stderr and proceeds normally.

## Python

- `import neorag` → **`import redhop`**. The shim package moved
  `examples/python/neorag/` → `examples/python/redhop/`.
- The JSON bridge binary `context_bridge` → **`redhop_bridge`**
  (`cargo build -p redhop-examples --example redhop_bridge`).
- Env override `NEORAG_BRIDGE` → **`REDHOP_BRIDGE`**.

## Intentionally preserved (NOT renamed)

- **NeoTrace data format** — the wire format keeps `schema_version: "neotrace/1"`,
  the `.neotrace.jsonl` extension, and the loader. Renaming would break the
  ~5,190 already-exported records and the (separate) Python-lab exporters.
  Prose now refers to it as "RedHop Trace (neotrace/1 wire format)".
- **The Python research-lab repo** at `../neorag/` (a *separate* repository)
  keeps its name; example/doc paths still point at `../neorag/…`.
- **This workspace directory** (`…/neorag1`) is unchanged; hardcoded absolute
  paths to it are preserved.

## Compatibility guarantees & deprecation policy

- Nothing was published to crates.io/PyPI/npm under the `neorag` names, so
  there are **no external consumers to break**; the rename is clean.
- In-source deprecated aliases (`NeoRAG`, `NeoRAGBuilder`) are provided as a
  courtesy and will be **removed in the next minor release**. New code should
  use `RedHop` / `RedHopBuilder`.
- No shim crates (`neorag-*` re-exporting `redhop-*`) are published — there is
  no consumer that needs them.
