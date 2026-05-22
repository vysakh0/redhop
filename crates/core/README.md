# redhop-core

Shared traits and data types for [RedHop](https://github.com/redhop/redhop) —
the common vocabulary every other crate builds on. No logic, just the types.

- **Types**: `Chunk`, `ChunkId`, `Query`, `Document`, `RetrievalResult`,
  `Score`, `Embedding`, `TokenCount`, …
- **Traits**: `Chunker`, `Retriever`, `Reranker`, `DiagnosticsEngine`,
  `TokenizerBackend`, and the adaptive-control traits.

All data types are `Serialize + Deserialize` to keep FFI and JSON boundaries
cheap. `#![forbid(unsafe_code)]`. Apache-2.0.

Most users want [`redhop-context`](https://crates.io/crates/redhop-context), the
context-optimization API; depend on `redhop-core` directly only when
implementing your own retriever/reranker/chunker against the traits.
