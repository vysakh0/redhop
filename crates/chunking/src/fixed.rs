//! Fixed-window chunking.
//!
//! [`FixedChunker`] splits documents into windows of approximately
//! `chunk_tokens` tokens with `overlap_tokens` overlap between adjacent
//! chunks. The implementation is deterministic and trivially reproducible;
//! it is the right baseline against which adaptive chunkers should be
//! measured.
//!
//! ## When to use
//!
//! - Reproducibility-sensitive evaluation harnesses.
//! - Workloads where sentence segmentation is unreliable (machine-generated
//!   text, log data).
//! - Diagnostics control runs.

use std::sync::Arc;

use redhop_core::{Chunk, ChunkId, Chunker, Document, Error, Result, TokenCount, TokenizerBackend};
use unicode_segmentation::UnicodeSegmentation;

/// Splits documents into fixed token-windowed chunks.
pub struct FixedChunker {
    tokenizer: Arc<dyn TokenizerBackend>,
    chunk_tokens: usize,
    overlap_tokens: usize,
}

impl FixedChunker {
    /// Construct a new fixed chunker.
    ///
    /// `overlap_tokens` must be strictly less than `chunk_tokens`, otherwise
    /// the chunker would make no forward progress.
    pub fn new(
        tokenizer: Arc<dyn TokenizerBackend>,
        chunk_tokens: usize,
        overlap_tokens: usize,
    ) -> Result<Self> {
        if chunk_tokens == 0 {
            return Err(Error::InvalidConfig("chunk_tokens must be > 0".into()));
        }
        if overlap_tokens >= chunk_tokens {
            return Err(Error::InvalidConfig(
                "overlap_tokens must be < chunk_tokens".into(),
            ));
        }
        Ok(Self {
            tokenizer,
            chunk_tokens,
            overlap_tokens,
        })
    }
}

impl Chunker for FixedChunker {
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        // Tokenize once into a `(byte_offset, word)` stream so we can both
        // count tokens and reconstruct chunk text by slicing the source.
        let words: Vec<(usize, &str)> = doc.text.unicode_word_indices().collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let stride = self.chunk_tokens.saturating_sub(self.overlap_tokens).max(1);
        let mut out = Vec::new();
        let mut start = 0usize;
        let mut idx = 0usize;
        while start < words.len() {
            let end = (start + self.chunk_tokens).min(words.len());
            let byte_start = words[start].0;
            let last_word = words[end - 1];
            let byte_end = last_word.0 + last_word.1.len();
            let slice = &doc.text[byte_start..byte_end];

            let id = ChunkId::new(format!("{}::fixed::{}", doc.source, idx));
            let tokens = self.tokenizer.count_tokens(slice)?;
            let mut chunk = Chunk::new(id, slice, &doc.source, tokens)
                .with_metadata(doc.metadata.clone());
            chunk.metadata.insert(
                "byte_offset".to_string(),
                serde_json::json!({ "start": byte_start, "end": byte_end }),
            );
            out.push(chunk);
            idx += 1;

            if end == words.len() {
                break;
            }
            start += stride;
        }
        Ok(out)
    }

    fn chunk_batch(&self, docs: &[Document]) -> Result<Vec<Chunk>> {
        use rayon::prelude::*;
        let chunks: std::result::Result<Vec<Vec<Chunk>>, Error> =
            docs.par_iter().map(|d| self.chunk(d)).collect();
        Ok(chunks?.into_iter().flatten().collect())
    }

    fn name(&self) -> &'static str {
        "fixed"
    }
}

// Silence unused-import warning for `TokenCount`; it's part of the public
// trait signatures referenced above.
#[allow(dead_code)]
fn _assert_used(t: TokenCount) -> usize {
    t.value()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::WhitespaceTokenizer;

    fn tok() -> Arc<dyn TokenizerBackend> {
        Arc::new(WhitespaceTokenizer::new())
    }

    #[test]
    fn produces_overlapping_windows() {
        let chunker = FixedChunker::new(tok(), 5, 2).unwrap();
        let doc = Document::new(
            "doc1",
            "one two three four five six seven eight nine ten eleven twelve",
        );
        let chunks = chunker.chunk(&doc).unwrap();
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.token_count.value() <= 5);
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(FixedChunker::new(tok(), 0, 0).is_err());
        assert!(FixedChunker::new(tok(), 4, 4).is_err());
        assert!(FixedChunker::new(tok(), 4, 5).is_err());
    }

    #[test]
    fn empty_doc_yields_no_chunks() {
        let chunker = FixedChunker::new(tok(), 5, 1).unwrap();
        let doc = Document::new("doc1", "   ");
        assert!(chunker.chunk(&doc).unwrap().is_empty());
    }
}
