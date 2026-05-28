//! Tokenizer backends.
//!
//! RedHop itself does not bind to any specific tokenizer family. This module
//! provides one zero-dependency default ([`WhitespaceTokenizer`]) that is good
//! enough for development, tests, and lexical retrieval; downstream crates
//! can implement [`TokenizerBackend`] against HuggingFace `tokenizers` or
//! `tiktoken-rs` behind feature flags without changing the chunker API.
//!
//! Why a whitespace fallback? Lexical retrieval (BM25) gets its real
//! tokenization from Tantivy's analyzer; dense retrieval gets it from the
//! caller's embedding model. The role of the tokenizer here is to count
//! tokens *for chunk budgeting* and split sentences. A simple
//! Unicode-aware word splitter is sufficient for the first, and
//! `unicode-segmentation` handles the second.

use crate::core::{Result, Sentence, TokenCount, TokenizerBackend};
use unicode_segmentation::UnicodeSegmentation;

/// Unicode-aware whitespace tokenizer.
///
/// Token boundaries are unicode word boundaries; sentence boundaries are
/// unicode sentence boundaries. This is *not* a substitute for a real
/// model-specific tokenizer when you care about exact token-budget alignment
/// with an LLM, but it is more than adequate for chunk budgeting and for
/// driving lexical-grounding diagnostics.
#[derive(Debug, Clone, Default)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Construct a new tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl TokenizerBackend for WhitespaceTokenizer {
    fn count_tokens(&self, text: &str) -> Result<TokenCount> {
        let n = text.unicode_words().count();
        Ok(TokenCount(n))
    }

    fn split_sentences(&self, text: &str) -> Result<Vec<Sentence>> {
        let mut out = Vec::new();
        for (start, raw) in text.split_sentence_bound_indices() {
            // Skip whitespace-only spans produced by the splitter.
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Recompute trimmed start/end relative to the source.
            let leading = raw.len() - raw.trim_start().len();
            let s = start + leading;
            let e = s + trimmed.len();
            out.push(Sentence {
                text: trimmed.to_string(),
                start: s,
                end: e,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_unicode_words() {
        let tok = WhitespaceTokenizer::new();
        assert_eq!(tok.count_tokens("hello world").unwrap().value(), 2);
        // unicode_words splits CJK into per-character words; treat each glyph
        // as one token. "一个 测试 example" → 4 CJK glyphs + "example".
        assert_eq!(tok.count_tokens("一个 测试 example").unwrap().value(), 5);
    }

    #[test]
    fn splits_sentences_on_terminators() {
        let tok = WhitespaceTokenizer::new();
        let s = tok
            .split_sentences("First sentence. Second one! And a third?")
            .unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].text, "First sentence.");
    }

    #[test]
    fn offsets_match_source() {
        let tok = WhitespaceTokenizer::new();
        let src = "Alpha. Bravo.";
        let s = tok.split_sentences(src).unwrap();
        for sent in &s {
            assert_eq!(&src[sent.start..sent.end], sent.text);
        }
    }
}
