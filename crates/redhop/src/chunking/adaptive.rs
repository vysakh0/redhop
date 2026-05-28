//! Adaptive chunking foundation.
//!
//! [`AdaptiveChunker`] is the long-term home for evidence-aware chunking.
//! Today it implements a *clean foundation*: sentence segmentation plus
//! a lightweight lexical-cohesion heuristic that decides where to close a
//! chunk early instead of packing strictly to the token budget.
//!
//! ## Roadmap
//!
//! Future work — to be added behind feature flags so it does not regress the
//! default behavior:
//!
//! - Topic-purity scoring per sentence using on-line term-frequency drift.
//! - Embedding-based cohesion gating (requires an [`EmbeddingProvider`]).
//! - Cross-sentence redundancy detection.
//! - Entropy-based boundary detection (sentence-level surprisal).
//!
//! What this file deliberately does *not* do today:
//!
//! - Fake "AI magic" boundary detection.
//! - Hard-coded model-specific assumptions.
//!
//! [`EmbeddingProvider`]: crate::core::EmbeddingProvider

use std::collections::HashSet;
use std::sync::Arc;

use crate::core::{Chunk, ChunkId, Chunker, Document, Error, Result, TokenizerBackend};
use unicode_segmentation::UnicodeSegmentation;

/// Adaptive sentence chunker with lexical-cohesion gating.
pub struct AdaptiveChunker {
    tokenizer: Arc<dyn TokenizerBackend>,
    target_tokens: usize,
    max_tokens: usize,
    /// Jaccard similarity threshold below which we close the current chunk
    /// before adding the next sentence. `0.0` disables the heuristic and the
    /// chunker degrades to plain sentence packing.
    cohesion_threshold: f32,
}

impl AdaptiveChunker {
    /// Construct a new adaptive chunker.
    pub fn new(
        tokenizer: Arc<dyn TokenizerBackend>,
        target_tokens: usize,
        max_tokens: usize,
        cohesion_threshold: f32,
    ) -> Result<Self> {
        if target_tokens == 0 {
            return Err(Error::InvalidConfig("target_tokens must be > 0".into()));
        }
        if max_tokens < target_tokens {
            return Err(Error::InvalidConfig(
                "max_tokens must be >= target_tokens".into(),
            ));
        }
        if !(0.0..=1.0).contains(&cohesion_threshold) {
            return Err(Error::InvalidConfig(
                "cohesion_threshold must be in [0,1]".into(),
            ));
        }
        Ok(Self {
            tokenizer,
            target_tokens,
            max_tokens,
            cohesion_threshold,
        })
    }

    /// Convenience constructor with sensible defaults
    /// (`target=256`, `max=384`, `cohesion=0.15`).
    pub fn with_defaults(tokenizer: Arc<dyn TokenizerBackend>) -> Result<Self> {
        Self::new(tokenizer, 256, 384, 0.15)
    }
}

fn term_set(text: &str) -> HashSet<String> {
    text.unicode_words().map(|w| w.to_lowercase()).collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

impl Chunker for AdaptiveChunker {
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        let sentences = self.tokenizer.split_sentences(&doc.text)?;
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        // Precompute token counts and term sets per sentence.
        let mut counts = Vec::with_capacity(sentences.len());
        let mut terms: Vec<HashSet<String>> = Vec::with_capacity(sentences.len());
        for s in &sentences {
            counts.push(self.tokenizer.count_tokens(&s.text)?.value());
            terms.push(term_set(&s.text));
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut buf_idx: Vec<usize> = Vec::new();
        let mut buf_terms: HashSet<String> = HashSet::new();
        let mut buf_tokens: usize = 0;
        let mut idx = 0usize;

        for i in 0..sentences.len() {
            let next_tokens = counts[i];
            let next_terms = &terms[i];

            // Oversized sentence: emit current buffer, then sentence solo.
            if next_tokens > self.max_tokens {
                if !buf_idx.is_empty() {
                    flush_chunk(
                        &doc.source,
                        &doc.metadata,
                        &sentences,
                        &buf_idx,
                        buf_tokens,
                        &mut idx,
                        &mut chunks,
                    );
                    buf_idx.clear();
                    buf_terms.clear();
                    buf_tokens = 0;
                }
                let solo = vec![i];
                flush_chunk(
                    &doc.source,
                    &doc.metadata,
                    &sentences,
                    &solo,
                    next_tokens,
                    &mut idx,
                    &mut chunks,
                );
                continue;
            }

            let would_exceed_target =
                !buf_idx.is_empty() && buf_tokens + next_tokens > self.target_tokens;

            // Cohesion gate: if the next sentence is lexically incoherent
            // with what we have, prefer closing the chunk now even if we
            // are below the target. Only activates once we have *some*
            // material to compare against.
            let incoherent = if buf_terms.is_empty() || self.cohesion_threshold <= 0.0 {
                false
            } else {
                jaccard(&buf_terms, next_terms) < self.cohesion_threshold
                    && buf_tokens >= self.target_tokens / 2
            };

            if would_exceed_target || incoherent {
                flush_chunk(
                    &doc.source,
                    &doc.metadata,
                    &sentences,
                    &buf_idx,
                    buf_tokens,
                    &mut idx,
                    &mut chunks,
                );
                buf_idx.clear();
                buf_terms.clear();
                buf_tokens = 0;
            }

            buf_idx.push(i);
            buf_terms.extend(next_terms.iter().cloned());
            buf_tokens += next_tokens;

            if buf_tokens >= self.max_tokens {
                flush_chunk(
                    &doc.source,
                    &doc.metadata,
                    &sentences,
                    &buf_idx,
                    buf_tokens,
                    &mut idx,
                    &mut chunks,
                );
                buf_idx.clear();
                buf_terms.clear();
                buf_tokens = 0;
            }
        }
        if !buf_idx.is_empty() {
            flush_chunk(
                &doc.source,
                &doc.metadata,
                &sentences,
                &buf_idx,
                buf_tokens,
                &mut idx,
                &mut chunks,
            );
        }
        Ok(chunks)
    }

    fn chunk_batch(&self, docs: &[Document]) -> Result<Vec<Chunk>> {
        use rayon::prelude::*;
        let chunks: std::result::Result<Vec<Vec<Chunk>>, Error> =
            docs.par_iter().map(|d| self.chunk(d)).collect();
        Ok(chunks?.into_iter().flatten().collect())
    }

    fn name(&self) -> &'static str {
        "adaptive"
    }
}

fn flush_chunk(
    source: &str,
    metadata: &crate::core::ChunkMetadata,
    sentences: &[crate::core::Sentence],
    buf_idx: &[usize],
    buf_tokens: usize,
    idx: &mut usize,
    chunks: &mut Vec<Chunk>,
) {
    let first = buf_idx[0];
    let last = *buf_idx.last().unwrap();
    let text = sentences[first..=last]
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let id = ChunkId::new(format!("{}::adapt::{}", source, idx));
    let mut chunk = Chunk::new(id, text, source, crate::core::TokenCount(buf_tokens))
        .with_metadata(metadata.clone());
    chunk.metadata.insert(
        "sentence_range".to_string(),
        serde_json::json!({ "start": first, "end": last + 1 }),
    );
    chunks.push(chunk);
    *idx += 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::tokenizer::WhitespaceTokenizer;

    fn tok() -> Arc<dyn TokenizerBackend> {
        Arc::new(WhitespaceTokenizer::new())
    }

    #[test]
    fn defaults_construct_ok() {
        AdaptiveChunker::with_defaults(tok()).unwrap();
    }

    #[test]
    fn respects_max_tokens() {
        let c = AdaptiveChunker::new(tok(), 8, 12, 0.0).unwrap();
        // Capitalize each sentence start so unicode sentence segmentation
        // actually splits them.
        let doc = Document::new(
            "d",
            "Aa bb cc dd ee ff. Gg hh ii jj kk ll. Mm nn oo pp qq rr. Ss tt uu vv ww xx.",
        );
        for chunk in c.chunk(&doc).unwrap() {
            assert!(
                chunk.token_count.value() <= 12,
                "got {}",
                chunk.token_count.value()
            );
        }
    }

    #[test]
    fn cohesion_gate_can_close_early() {
        // Two topical clusters with no shared vocabulary.
        let c = AdaptiveChunker::new(tok(), 20, 40, 0.2).unwrap();
        let doc = Document::new(
            "d",
            "Cats purr. Cats nap. Cats hunt mice. Cats stalk. \
             Tokio runs futures. Tokio polls tasks. Tokio drives executors.",
        );
        let chunks = c.chunk(&doc).unwrap();
        assert!(chunks.len() >= 2, "expected cohesion gate to split topics");
    }
}
