//! Sentence-aware token-budgeted chunking.
//!
//! [`SentenceChunker`] segments documents into sentences first, then packs
//! sentences into chunks while respecting a token budget. This is the right
//! default for prose: sentence boundaries are natural evidence boundaries
//! and avoid the "split mid-sentence" failure mode of pure fixed-window
//! chunking, which is one of the largest contributors to spurious distractors
//! in retrieval results.

use std::sync::Arc;

use crate::core::{Chunk, ChunkId, Chunker, Document, Error, Result, TokenizerBackend};

/// Sentence-budgeted chunker.
pub struct SentenceChunker {
    tokenizer: Arc<dyn TokenizerBackend>,
    target_tokens: usize,
    max_tokens: usize,
    overlap_sentences: usize,
}

impl SentenceChunker {
    /// Construct a new sentence chunker.
    ///
    /// - `target_tokens`: soft target chunk size. The packer flushes the
    ///   current chunk once adding the next sentence would exceed this.
    /// - `max_tokens`: hard cap. A single sentence longer than this is
    ///   emitted as its own chunk (it would be incorrect to drop it, and
    ///   splitting it would require word-level chunking that callers can
    ///   opt into via [`FixedChunker`] if needed).
    /// - `overlap_sentences`: how many trailing sentences of chunk *N* to
    ///   prepend to chunk *N + 1*. Set to `0` for crisp, non-overlapping
    ///   chunks; set to `1`–`2` to soften boundary effects in retrieval.
    ///
    /// [`FixedChunker`]: crate::FixedChunker
    pub fn new(
        tokenizer: Arc<dyn TokenizerBackend>,
        target_tokens: usize,
        max_tokens: usize,
        overlap_sentences: usize,
    ) -> Result<Self> {
        if target_tokens == 0 {
            return Err(Error::InvalidConfig("target_tokens must be > 0".into()));
        }
        if max_tokens < target_tokens {
            return Err(Error::InvalidConfig(
                "max_tokens must be >= target_tokens".into(),
            ));
        }
        Ok(Self {
            tokenizer,
            target_tokens,
            max_tokens,
            overlap_sentences,
        })
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        let sentences = self.tokenizer.split_sentences(&doc.text)?;
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        // Precompute token counts to avoid quadratic retokenization while
        // packing.
        let mut counts = Vec::with_capacity(sentences.len());
        for s in &sentences {
            counts.push(self.tokenizer.count_tokens(&s.text)?.value());
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut buf_idx: Vec<usize> = Vec::new();
        let mut buf_tokens: usize = 0;
        let mut idx = 0usize;

        let flush = |chunks: &mut Vec<Chunk>,
                     buf_idx: &mut Vec<usize>,
                     buf_tokens: &mut usize,
                     idx: &mut usize|
         -> Result<()> {
            if buf_idx.is_empty() {
                return Ok(());
            }
            let first = buf_idx[0];
            let last = *buf_idx.last().unwrap();
            let text = sentences[first..=last]
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let id = ChunkId::new(format!("{}::sent::{}", doc.source, idx));
            let mut chunk = Chunk::new(id, text, &doc.source, crate::core::TokenCount(*buf_tokens))
                .with_metadata(doc.metadata.clone());
            chunk.metadata.insert(
                "sentence_range".to_string(),
                serde_json::json!({ "start": first, "end": last + 1 }),
            );
            chunks.push(chunk);
            *idx += 1;

            // Apply overlap by retaining trailing sentences.
            if self.overlap_sentences > 0 && buf_idx.len() > self.overlap_sentences {
                let keep_from = buf_idx.len() - self.overlap_sentences;
                let kept: Vec<usize> = buf_idx[keep_from..].to_vec();
                buf_idx.clear();
                *buf_tokens = 0;
                for k in kept {
                    buf_idx.push(k);
                    *buf_tokens += counts[k];
                }
            } else {
                buf_idx.clear();
                *buf_tokens = 0;
            }
            Ok(())
        };

        for i in 0..sentences.len() {
            let next_tokens = counts[i];

            // A sentence larger than `max_tokens` is emitted on its own.
            if next_tokens > self.max_tokens {
                flush(&mut chunks, &mut buf_idx, &mut buf_tokens, &mut idx)?;
                buf_idx.push(i);
                buf_tokens = next_tokens;
                flush(&mut chunks, &mut buf_idx, &mut buf_tokens, &mut idx)?;
                continue;
            }

            if !buf_idx.is_empty() && buf_tokens + next_tokens > self.target_tokens {
                flush(&mut chunks, &mut buf_idx, &mut buf_tokens, &mut idx)?;
            }
            buf_idx.push(i);
            buf_tokens += next_tokens;

            if buf_tokens >= self.max_tokens {
                flush(&mut chunks, &mut buf_idx, &mut buf_tokens, &mut idx)?;
            }
        }
        flush(&mut chunks, &mut buf_idx, &mut buf_tokens, &mut idx)?;
        Ok(chunks)
    }

    fn chunk_batch(&self, docs: &[Document]) -> Result<Vec<Chunk>> {
        use rayon::prelude::*;
        let chunks: std::result::Result<Vec<Vec<Chunk>>, Error> =
            docs.par_iter().map(|d| self.chunk(d)).collect();
        Ok(chunks?.into_iter().flatten().collect())
    }

    fn name(&self) -> &'static str {
        "sentence"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::tokenizer::WhitespaceTokenizer;

    fn tok() -> Arc<dyn TokenizerBackend> {
        Arc::new(WhitespaceTokenizer::new())
    }

    #[test]
    fn packs_sentences_to_budget() {
        let chunker = SentenceChunker::new(tok(), 6, 10, 0).unwrap();
        let doc = Document::new(
            "doc1",
            "Alpha bravo charlie. Delta echo. Foxtrot golf hotel india. Juliet.",
        );
        let chunks = chunker.chunk(&doc).unwrap();
        for c in &chunks {
            assert!(c.token_count.value() <= 10);
        }
        // Should not be one giant chunk.
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn oversized_sentence_emitted_solo() {
        let chunker = SentenceChunker::new(tok(), 3, 4, 0).unwrap();
        let big = "one two three four five six seven eight nine ten eleven twelve.";
        let doc = Document::new("doc1", big);
        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].token_count.value() > 4);
    }

    #[test]
    fn overlap_retains_tail_sentences() {
        let chunker = SentenceChunker::new(tok(), 4, 6, 1).unwrap();
        let doc = Document::new("doc1", "Aa bb. Cc dd. Ee ff. Gg hh. Ii jj.");
        let chunks = chunker.chunk(&doc).unwrap();
        // With overlap=1, adjacent chunks should share at least one sentence
        // worth of bytes.
        if chunks.len() >= 2 {
            assert!(chunks[0]
                .text
                .split_whitespace()
                .last()
                .map(|w| chunks[1].text.contains(w))
                .unwrap_or(false));
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(SentenceChunker::new(tok(), 0, 10, 0).is_err());
        assert!(SentenceChunker::new(tok(), 10, 5, 0).is_err());
    }
}
