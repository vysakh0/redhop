//! [`CachedEmbedder`] — a bounded LRU cache wrapping any
//! [`EmbeddingProvider`].
//!
//! Query embedding is pure and deterministic for a given model, so a
//! cache is free quality and real latency savings on repeated or
//! templated queries — enterprise FAQ, dashboards, agent loops that
//! re-ask. The cache keys on a hash of the input text; collisions are
//! astronomically unlikely at FNV-64 and a collision would only return
//! a wrong-but-valid vector, never crash.
//!
//! The wrapper is generic over the inner provider, so it composes with
//! the hashing baseline today and the ONNX backend tomorrow with no
//! changes.
//!
//! Concurrency: a single `Mutex<LruCache>` guards the cache. The lock is
//! held only for synchronous get/put, never across the inner provider's
//! `.await`, so it does not serialize inference.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use async_trait::async_trait;
use lru::LruCache;
use redhop_core::{Embedding, EmbeddingProvider, Result};

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// An LRU-cached wrapper around an [`EmbeddingProvider`].
pub struct CachedEmbedder<E> {
    inner: E,
    cache: Mutex<LruCache<u64, Embedding>>,
    hits: Mutex<u64>,
    misses: Mutex<u64>,
}

impl<E: EmbeddingProvider> CachedEmbedder<E> {
    /// Wrap `inner` with a cache of the given capacity (number of
    /// distinct texts retained).
    pub fn new(inner: E, capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            inner,
            cache: Mutex::new(LruCache::new(cap)),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// Cumulative (hits, misses) since construction. Useful for the
    /// observability layer to report cache effectiveness.
    pub fn stats(&self) -> (u64, u64) {
        (*self.hits.lock().unwrap(), *self.misses.lock().unwrap())
    }
}

#[async_trait]
impl<E: EmbeddingProvider> EmbeddingProvider for CachedEmbedder<E> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let keys: Vec<u64> = texts.iter().map(|t| fnv1a64(t.as_bytes())).collect();

        // Phase 1: synchronous cache probe. Collect misses to embed.
        let mut results: Vec<Option<Embedding>> = vec![None; texts.len()];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<String> = Vec::new();
        {
            let mut cache = self.cache.lock().unwrap();
            for (i, key) in keys.iter().enumerate() {
                if let Some(e) = cache.get(key) {
                    results[i] = Some(e.clone());
                } else {
                    miss_indices.push(i);
                    miss_texts.push(texts[i].clone());
                }
            }
        } // lock released before the await below

        *self.hits.lock().unwrap() += (texts.len() - miss_indices.len()) as u64;
        *self.misses.lock().unwrap() += miss_indices.len() as u64;

        // Phase 2: embed the misses (no lock held).
        if !miss_texts.is_empty() {
            let embedded = self.inner.embed(&miss_texts).await?;
            // Phase 3: synchronous insert + fill.
            let mut cache = self.cache.lock().unwrap();
            for (slot, emb) in miss_indices.iter().zip(embedded.into_iter()) {
                cache.put(keys[*slot], emb.clone());
                results[*slot] = Some(emb);
            }
        }

        Ok(results
            .into_iter()
            .map(|o| o.expect("all slots filled"))
            .collect())
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn name(&self) -> &'static str {
        "cached"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// A provider that counts how many texts it actually embedded, so we
    /// can prove the cache prevents re-embedding.
    struct CountingProvider {
        calls: Arc<AtomicU64>,
        dim: usize,
    }
    #[async_trait]
    impl EmbeddingProvider for CountingProvider {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
            self.calls.fetch_add(texts.len() as u64, Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| {
                    // Deterministic toy embedding from text length.
                    let mut v = vec![0f32; self.dim];
                    v[t.len() % self.dim] = 1.0;
                    Embedding(v)
                })
                .collect())
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn name(&self) -> &'static str {
            "counting"
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
    }

    #[test]
    fn cache_prevents_re_embedding() {
        rt().block_on(async {
            let calls = Arc::new(AtomicU64::new(0));
            let inner = CountingProvider {
                calls: calls.clone(),
                dim: 16,
            };
            let cached = CachedEmbedder::new(inner, 100);

            // First call: all misses.
            let _ = cached
                .embed(&["alpha".to_string(), "beta".to_string()])
                .await
                .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 2);

            // Second call with overlap: only the new text is embedded.
            let _ = cached
                .embed(&["alpha".to_string(), "gamma".to_string()])
                .await
                .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 3); // +1 for gamma only

            let (hits, misses) = cached.stats();
            assert_eq!(hits, 1); // alpha on the second call
            assert_eq!(misses, 3); // alpha, beta, gamma
        });
    }

    #[test]
    fn cached_results_match_uncached() {
        rt().block_on(async {
            let calls = Arc::new(AtomicU64::new(0));
            let inner = CountingProvider { calls, dim: 16 };
            let direct = CountingProvider {
                calls: Arc::new(AtomicU64::new(0)),
                dim: 16,
            };
            let cached = CachedEmbedder::new(inner, 100);

            let texts = vec!["one".to_string(), "two".to_string(), "one".to_string()];
            let c = cached.embed(&texts).await.unwrap();
            let d = direct.embed(&texts).await.unwrap();
            for (a, b) in c.iter().zip(d.iter()) {
                assert_eq!(a.as_slice(), b.as_slice());
            }
            // The two "one"s must be identical.
            assert_eq!(c[0].as_slice(), c[2].as_slice());
        });
    }

    #[test]
    fn dim_and_capacity_are_respected() {
        let cached = CachedEmbedder::new(HashingForTest, 0); // capacity floored to 1
        assert_eq!(cached.dim(), 8);
    }

    struct HashingForTest;
    #[async_trait]
    impl EmbeddingProvider for HashingForTest {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
            Ok(texts.iter().map(|_| Embedding(vec![0f32; 8])).collect())
        }
        fn dim(&self) -> usize {
            8
        }
        fn name(&self) -> &'static str {
            "test"
        }
    }
}
