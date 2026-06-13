//! Storage backends used by retrievers.
//!
//! Two abstractions live here:
//!
//! - [`ChunkStore`] — owns the chunks themselves, keyed by [`ChunkId`].
//!   Retrievers index chunks but typically only persist *ids* in their own
//!   internal structures; the `ChunkStore` is the source of truth for the
//!   payload.
//! - [`FlatVectorIndex`] — a brute-force, exact-cosine vector index
//!   implementing [`VectorIndex`]. This is fast enough for tens of thousands
//!   of vectors, has zero external dependencies, and is the right default
//!   while higher-performance ANN backends (HNSW via `usearch`, IVF, …) are
//!   plugged in behind feature flags.
//!
//! [`ChunkId`]: crate::core::ChunkId
//! [`VectorIndex`]: crate::core::VectorIndex

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod chunk_store;
pub mod flat;

pub use chunk_store::ChunkStore;
pub use flat::FlatVectorIndex;
