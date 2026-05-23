//! Flat (brute-force, exact) cosine-similarity vector index.
//!
//! Trades scan cost for simplicity, correctness, and zero dependencies. At a
//! few tens of thousands of vectors this is faster than the round-trip into
//! an external ANN library would be, especially on warm caches. Above that,
//! swap in a `VectorIndex` backed by `usearch`/`hnswlib-rs`/`faiss`; the
//! retriever above does not care.

use parking_lot::RwLock;
use redhop_core::{ChunkId, Embedding, Error, Result, VectorIndex};

/// Flat brute-force vector index.
pub struct FlatVectorIndex {
    dim: usize,
    inner: RwLock<Inner>,
}

#[derive(Default)]
struct Inner {
    ids: Vec<ChunkId>,
    // Unit-normalized vectors stored contiguously row-major so the search
    // loop is one tight `dot_product` per row.
    vectors: Vec<f32>,
}

impl FlatVectorIndex {
    /// Construct a new index expecting `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            inner: RwLock::new(Inner::default()),
        }
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }
}

impl VectorIndex for FlatVectorIndex {
    fn add(&mut self, id: ChunkId, vector: Embedding) -> Result<()> {
        if vector.dim() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.dim(),
            });
        }
        let mut g = self.inner.write();
        let normalized = Self::normalize(vector.as_slice());
        g.ids.push(id);
        g.vectors.extend_from_slice(&normalized);
        Ok(())
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<(ChunkId, f32)>> {
        if query.dim() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: query.dim(),
            });
        }
        let g = self.inner.read();
        if g.ids.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let qn = Self::normalize(query.as_slice());
        let mut scored: Vec<(usize, f32)> = (0..g.ids.len())
            .map(|i| {
                let off = i * self.dim;
                let row = &g.vectors[off..off + self.dim];
                let mut dot = 0.0f32;
                for j in 0..self.dim {
                    dot += row[j] * qn[j];
                }
                (i, dot)
            })
            .collect();
        // Partial sort by score desc.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top = scored
            .into_iter()
            .take(k)
            .map(|(i, s)| (g.ids[i].clone(), s))
            .collect();
        Ok(top)
    }

    fn len(&self) -> usize {
        self.inner.read().ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_self_first() {
        let mut idx = FlatVectorIndex::new(3);
        idx.add(ChunkId::new("a"), Embedding::from(vec![1.0, 0.0, 0.0]))
            .unwrap();
        idx.add(ChunkId::new("b"), Embedding::from(vec![0.0, 1.0, 0.0]))
            .unwrap();
        idx.add(ChunkId::new("c"), Embedding::from(vec![0.0, 0.0, 1.0]))
            .unwrap();
        let res = idx
            .search(&Embedding::from(vec![1.0, 0.1, 0.0]), 2)
            .unwrap();
        assert_eq!(res[0].0, ChunkId::new("a"));
        assert!(res[0].1 > res[1].1);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut idx = FlatVectorIndex::new(3);
        let e = idx
            .add(ChunkId::new("x"), Embedding::from(vec![1.0, 0.0]))
            .unwrap_err();
        assert!(matches!(e, Error::DimensionMismatch { .. }));
    }
}
