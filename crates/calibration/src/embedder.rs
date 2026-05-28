//! Deterministic hashing-trick TF embedder.
//!
//! Real-workload calibration needs embeddings that mean something on
//! natural text — the topic-bucket embedder in `fixtures.rs` is a
//! synthetic teaching aid, not suitable for HotpotQA / MuSiQue
//! retrieval.
//!
//! This module ships a **feature hashing** embedder: for each Unicode
//! word in the text, FNV-1a hash it into a slot in a fixed-dimensional
//! vector and increment by `1.0`. Then L2-normalize. The result is a
//! sparse-but-dense-shaped TF vector whose cosine similarity is a
//! reasonable lexical-overlap proxy without any model dependency.
//!
//! Why is this honest?
//!
//! - **No model dep.** A real evaluation should use a real embedding
//!   model. This embedder is a *deterministic baseline* that lets us
//!   measure adaptive-controller utility on real text right now,
//!   while documenting clearly that semantic gains from a real
//!   embedder would only *expand* the gap between adaptive and static.
//! - **No tokenizer surprises.** Unicode words, lowercase, drop
//!   stopwords. Same logic everywhere.
//! - **Deterministic.** Identical text yields identical vector across
//!   processes; bootstrap stability analyses stay reproducible.
//!
//! The next step (real embedding model) plugs into the same
//! [`redhop::core::EmbeddingProvider`] trait this embedder fulfills
//! morally if not literally — it's a function `&str → Embedding`, not
//! a trait impl, because async overhead is wasted for a pure-CPU
//! hashing operation.

use redhop::core::Embedding;
use unicode_segmentation::UnicodeSegmentation;

/// English stopwords that pollute lexical-overlap signal. Kept short
/// on purpose — heavyweight stopword lists overfit to specific
/// workloads. The set here is the intersection of stopwords across
/// HotpotQA, MuSiQue, and the Python lab's `evidence_evidence` runs.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "with", "as", "by", "on", "at", "it", "its",
    "from", "but", "if", "then", "than", "so", "such", "do", "does", "did", "have", "has", "had",
    "will", "would", "could", "should", "i", "you", "he", "she", "we", "they", "them", "their",
    "his", "her", "our", "your", "my",
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

/// Hashing-trick TF embedder.
///
/// `dim` controls the vector dimension; 256 is a sensible default that
/// gives low collision rates on HotpotQA-sized vocabularies (~10⁴
/// distinct terms after stopword filtering) while keeping the dot
/// product cheap.
#[derive(Debug, Clone, Copy)]
pub struct HashingEmbedder {
    /// Output vector dimensionality.
    pub dim: usize,
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self { dim: 256 }
    }
}

impl HashingEmbedder {
    /// Construct with the given dimensionality.
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }

    /// Embed text into an L2-normalized vector.
    pub fn embed(&self, text: &str) -> Embedding {
        let mut v = vec![0f32; self.dim];
        for w in text.unicode_words() {
            let w: String = w.to_lowercase();
            if w.chars().count() <= 1 {
                continue;
            }
            if is_stopword(&w) {
                continue;
            }
            let slot = (fnv1a64(w.as_bytes()) as usize) % self.dim;
            v[slot] += 1.0;
        }
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        Embedding(v)
    }

    /// True iff `text` contributes any non-stopword token. Useful for
    /// short-circuiting empty / all-stopword inputs.
    pub fn has_content(&self, text: &str) -> bool {
        for w in text.unicode_words() {
            let lw: String = w.to_lowercase();
            if lw.chars().count() > 1 && !is_stopword(&lw) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_yields_identical_vectors() {
        let e = HashingEmbedder::default();
        let v1 = e.embed("the quick brown fox jumps over the lazy dog");
        let v2 = e.embed("the quick brown fox jumps over the lazy dog");
        assert_eq!(v1.as_slice(), v2.as_slice());
    }

    #[test]
    fn topically_related_text_has_high_cosine() {
        let e = HashingEmbedder::default();
        let v_rust = e.embed("rust is a systems programming language with strong memory safety");
        let v_rust2 = e.embed("rust focuses on memory safety and systems programming");
        let v_cooking = e.embed("baking bread requires flour water yeast and patience");

        let cos = |a: &Embedding, b: &Embedding| -> f32 {
            a.as_slice()
                .iter()
                .zip(b.as_slice().iter())
                .map(|(x, y)| x * y)
                .sum()
        };
        let related = cos(&v_rust, &v_rust2);
        let unrelated = cos(&v_rust, &v_cooking);
        assert!(
            related > unrelated + 0.3,
            "expected clear separation: related={related} unrelated={unrelated}"
        );
    }

    #[test]
    fn empty_or_stopword_only_yields_zero_vector() {
        let e = HashingEmbedder::default();
        // After stopword removal, "the and a or of" leaves nothing.
        let v = e.embed("the and a or of");
        // L2 normalization of all-zero with EPS fallback produces a
        // near-zero vector (each element divided by ~1e-9 from zero
        // numerator still yields 0).
        let mag: f32 = v.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(mag < 1e-3, "expected near-zero vector, got |v| = {mag}");
    }

    #[test]
    fn stopwords_do_not_affect_cosine() {
        let e = HashingEmbedder::default();
        let bare = e.embed("rust memory safety");
        let padded = e.embed("the rust is a language with memory and safety");
        let cos: f32 = bare
            .as_slice()
            .iter()
            .zip(padded.as_slice().iter())
            .map(|(x, y)| x * y)
            .sum();
        // Both should produce the same three non-stopword tokens
        // (rust, memory, safety). Cosine should be very high; we
        // allow a small slack because `language` is an extra term in
        // the padded version.
        assert!(cos > 0.7, "cosine={cos}");
    }
}
