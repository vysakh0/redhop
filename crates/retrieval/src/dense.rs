//! Dense vector retrieval.
//!
//! [`DenseRetriever`] is a thin orchestrator over two collaborators:
//!
//! 1. A [`VectorIndex`] (typically `FlatVectorIndex` from `redhop-storage`,
//!    or any user-supplied ANN implementation).
//! 2. A [`ChunkStore`] that owns the chunk payloads, keyed by id.
//!
//! Embeddings must be precomputed on the chunk side. RedHop does *not* embed
//! text itself; that is intentionally pushed to a user-supplied
//! [`EmbeddingProvider`] so the library stays neutral about model choice.
//!
//! [`EmbeddingProvider`]: redhop_core::EmbeddingProvider

use std::sync::Arc;

use async_trait::async_trait;
use redhop_core::{
    Chunk, Error, Query, RetrievalMethod, RetrievalResult, Retriever, Score, ScoreBreakdown,
    VectorIndex,
};
use redhop_storage::ChunkStore;
use parking_lot::RwLock;

/// Dense retriever pairing a [`VectorIndex`] with a [`ChunkStore`].
pub struct DenseRetriever {
    index: Arc<RwLock<dyn VectorIndex>>,
    store: Arc<ChunkStore>,
}

impl DenseRetriever {
    /// Construct a new dense retriever.
    pub fn new(index: Arc<RwLock<dyn VectorIndex>>, store: Arc<ChunkStore>) -> Self {
        Self { index, store }
    }
}

#[async_trait]
impl Retriever for DenseRetriever {
    async fn index(&mut self, chunks: &[Chunk]) -> redhop_core::Result<()> {
        for c in chunks {
            let emb = c.embedding.clone().ok_or_else(|| {
                Error::Embedding(format!(
                    "chunk {} is missing an embedding; the dense retriever requires one",
                    c.id
                ))
            })?;
            self.index.write().add(c.id.clone(), emb)?;
            self.store.insert(c.clone());
        }
        Ok(())
    }

    async fn retrieve(
        &self,
        query: &Query,
        top_k: usize,
    ) -> redhop_core::Result<Vec<RetrievalResult>> {
        let qe = query.embedding.as_ref().ok_or_else(|| {
            Error::Embedding(
                "dense retrieval requires a precomputed query embedding".into(),
            )
        })?;
        let hits = self.index.read().search(qe, top_k.max(1))?;
        let mut out = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            let Some(chunk) = self.store.get(&id) else {
                // Index/store skew — skip silently; the retriever stays
                // correct even if a chunk is evicted from the store.
                continue;
            };
            let breakdown = ScoreBreakdown {
                dense: Some(score),
                ..Default::default()
            };
            out.push(RetrievalResult {
                chunk,
                score: Score {
                    value: score,
                    method: RetrievalMethod::Dense,
                },
                breakdown,
            });
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "dense"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::{Embedding, TokenCount};
    use redhop_storage::FlatVectorIndex;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn dense_round_trip() {
        rt().block_on(async {
            let idx: Arc<RwLock<dyn VectorIndex>> =
                Arc::new(RwLock::new(FlatVectorIndex::new(3)));
            let store = Arc::new(ChunkStore::new());
            let mut r = DenseRetriever::new(idx, store);

            let mk = |id: &str, e: Vec<f32>| {
                Chunk::new(id, id, "doc", TokenCount(1))
                    .with_embedding(Embedding::from(e))
            };
            r.index(&[
                mk("a", vec![1.0, 0.0, 0.0]),
                mk("b", vec![0.0, 1.0, 0.0]),
                mk("c", vec![0.0, 0.0, 1.0]),
            ])
            .await
            .unwrap();

            let q = Query::new("ignored").with_embedding(Embedding::from(vec![0.9, 0.1, 0.0]));
            let res = r.retrieve(&q, 2).await.unwrap();
            assert_eq!(res[0].chunk.id.as_str(), "a");
            assert_eq!(res[0].score.method, RetrievalMethod::Dense);
        });
    }

    #[test]
    fn dense_requires_query_embedding() {
        rt().block_on(async {
            let idx: Arc<RwLock<dyn VectorIndex>> =
                Arc::new(RwLock::new(FlatVectorIndex::new(3)));
            let store = Arc::new(ChunkStore::new());
            let r = DenseRetriever::new(idx, store);
            assert!(r.retrieve(&Query::new("no emb"), 1).await.is_err());
        });
    }
}
