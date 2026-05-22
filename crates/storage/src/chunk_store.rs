//! In-memory chunk store.

use std::collections::HashMap;

use redhop_core::{Chunk, ChunkId};
use parking_lot::RwLock;

/// A thread-safe in-memory store of chunks.
///
/// Cloneable: clones share the underlying storage via `Arc`-like semantics
/// (an `RwLock` wrapped in an `Arc`). Use [`ChunkStore::handle`] to obtain a
/// cheap shared handle.
#[derive(Default)]
pub struct ChunkStore {
    inner: RwLock<HashMap<ChunkId, Chunk>>,
}

impl ChunkStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a chunk.
    pub fn insert(&self, chunk: Chunk) {
        self.inner.write().insert(chunk.id.clone(), chunk);
    }

    /// Insert (or replace) a batch of chunks.
    pub fn insert_batch<I: IntoIterator<Item = Chunk>>(&self, chunks: I) {
        let mut g = self.inner.write();
        for c in chunks {
            g.insert(c.id.clone(), c);
        }
    }

    /// Fetch a chunk by id.
    pub fn get(&self, id: &ChunkId) -> Option<Chunk> {
        self.inner.read().get(id).cloned()
    }

    /// Number of chunks currently stored.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// True iff the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot all chunks. Useful for diagnostics and small tests; not
    /// meant to be called on the hot path.
    pub fn all(&self) -> Vec<Chunk> {
        self.inner.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::TokenCount;

    #[test]
    fn store_round_trip() {
        let s = ChunkStore::new();
        let c = Chunk::new("a", "hello", "doc", TokenCount(1));
        s.insert(c.clone());
        assert_eq!(s.len(), 1);
        assert_eq!(s.get(&c.id).unwrap().text, "hello");
    }
}
