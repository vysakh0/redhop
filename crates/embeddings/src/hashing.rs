//! Hashing-trick TF embedding provider — the always-available, zero-dep
//! baseline.
//!
//! Implements [`EmbeddingProvider`] so it drops into the real retrieval
//! pipeline anywhere a transformer embedder would go. It is deterministic
//! (reproducible across processes) and requires no model files, which
//! makes it the right default for tests, CI, cold-start, and as the
//! control arm in embedder bake-offs.
//!
//! The math is identical to the calibration crate's `HashingEmbedder`:
//! FNV-1a hash each non-stopword Unicode word into a fixed-dim slot,
//! count, L2-normalize. A real model would beat it on paraphrase recall;
//! it exists to be beaten, measurably.

use async_trait::async_trait;
use redhop_core::{Embedding, EmbeddingProvider, Result};
use unicode_segmentation::UnicodeSegmentation;

use crate::pooling::l2_normalize;

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "with", "as", "by", "on", "at", "it",
    "its", "from", "but", "if", "then", "than", "so", "such", "do", "does", "did", "have", "has",
    "had", "will", "would", "could", "should",
];

fn is_stopword(s: &str) -> bool {
    STOPWORDS.contains(&s)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Feature-hashing TF embedding provider.
#[derive(Debug, Clone, Copy)]
pub struct HashingProvider {
    dim: usize,
}

impl Default for HashingProvider {
    fn default() -> Self {
        Self { dim: 256 }
    }
}

impl HashingProvider {
    /// Construct with the given output dimensionality.
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }

    fn embed_one(&self, text: &str) -> Embedding {
        let mut v = vec![0f32; self.dim];
        for w in text.unicode_words() {
            let w = w.to_lowercase();
            if w.chars().count() <= 1 || is_stopword(&w) {
                continue;
            }
            let slot = (fnv1a64(w.as_bytes()) as usize) % self.dim;
            v[slot] += 1.0;
        }
        l2_normalize(&mut v);
        Embedding(v)
    }
}

#[async_trait]
impl EmbeddingProvider for HashingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &'static str {
        "hashing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread().build().unwrap()
    }

    #[test]
    fn embeds_batch_with_correct_dim() {
        rt().block_on(async {
            let p = HashingProvider::with_dim(128);
            let out = p
                .embed(&["rust memory safety".to_string(), "cooking recipes".to_string()])
                .await
                .unwrap();
            assert_eq!(out.len(), 2);
            assert_eq!(out[0].dim(), 128);
        });
    }

    #[test]
    fn related_text_more_similar_than_unrelated() {
        rt().block_on(async {
            let p = HashingProvider::default();
            let v = p
                .embed(&[
                    "rust systems programming memory safety".to_string(),
                    "rust focuses on memory safety in systems".to_string(),
                    "baking sourdough bread with flour and yeast".to_string(),
                ])
                .await
                .unwrap();
            let cos = |a: &Embedding, b: &Embedding| -> f32 {
                a.as_slice().iter().zip(b.as_slice()).map(|(x, y)| x * y).sum()
            };
            assert!(cos(&v[0], &v[1]) > cos(&v[0], &v[2]) + 0.2);
        });
    }
}
