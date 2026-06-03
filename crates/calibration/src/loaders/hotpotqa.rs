//! HotpotQA loader.
//!
//! HotpotQA examples have this shape (one JSON array of these):
//!
//! ```json
//! {
//!   "_id": "...",
//!   "question": "...",
//!   "answer": "...",
//!   "type": "comparison" | "bridge",
//!   "level": "easy" | "medium" | "hard",
//!   "supporting_facts": [["title1", 0], ["title2", 2]],
//!   "context": [
//!     ["title1", ["sent1.", "sent2.", "sent3."]],
//!     ["title2", ["sent1.", "sent2."]]
//!   ]
//! }
//! ```
//!
//! The loader turns the entire dataset into one `LabeledCorpus`:
//!
//! - Documents = the deduplicated set of `(title, joined sentences)`
//!   tuples seen across all examples. Title becomes the document's
//!   `source`; the sentences are concatenated with spaces to form the
//!   document text.
//! - Queries = one per example. The query text is the question; the
//!   gold chunk ids are determined by finding which chunks (after
//!   chunking) *contain* each supporting sentence's text. A supporting
//!   sentence that doesn't fully fit in any chunk is matched to the
//!   chunk with the longest substring overlap.
//!
//! ## Regime labeling
//!
//! HotpotQA does not have a native "regime" label. The default
//! [`default_regime`] heuristic maps:
//!
//! | level + type            | regime          |
//! |-------------------------|------------------|
//! | easy                    | Easy             |
//! | medium, comparison      | Easy             |
//! | medium, bridge          | Ambiguous        |
//! | hard, comparison        | Ambiguous        |
//! | hard, bridge            | DistractorHeavy  |
//!
//! Bridge-style hard questions tend to require multi-hop reasoning
//! through similar-looking distractors, which matches the
//! `DistractorHeavy` regime well. Callers wanting a different mapping
//! pass their own classifier to [`HotpotQADataset::to_labeled_corpus`].

use std::collections::BTreeMap;

use redhop::core::{Chunk, ChunkId, Chunker, Document, Embedding, Error, Result, RetrievalRegime};
use serde::{Deserialize, Serialize};

use crate::dataset::{LabeledCorpus, LabeledQuery};

/// One HotpotQA example, as it appears in the canonical JSON.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HotpotQAExample {
    /// Stable example id.
    #[serde(rename = "_id")]
    pub id: String,
    /// The natural-language question.
    pub question: String,
    /// The gold answer string (RedHop does not use this directly, but
    /// we keep it so calibration loops can compare retrieval lift to
    /// downstream answer accuracy later).
    #[serde(default)]
    pub answer: String,
    /// `"comparison"` or `"bridge"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// `"easy"`, `"medium"`, or `"hard"`.
    pub level: String,
    /// `(title, sentence_idx)` pairs identifying gold evidence sentences.
    pub supporting_facts: Vec<(String, usize)>,
    /// `(title, sentences)` pairs for the paragraphs available to this
    /// example.
    pub context: Vec<(String, Vec<String>)>,
}

/// A loaded HotpotQA dataset.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HotpotQADataset {
    /// All loaded examples.
    pub examples: Vec<HotpotQAExample>,
}

impl HotpotQADataset {
    /// Parse from a JSON string in the canonical HotpotQA shape.
    pub fn from_json(s: &str) -> Result<Self> {
        let examples: Vec<HotpotQAExample> =
            serde_json::from_str(s).map_err(|e| Error::msg(format!("hotpotqa parse: {e}")))?;
        Ok(Self { examples })
    }

    /// Parse from a file at `path`.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json(&text)
    }

    /// Number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// True iff the dataset has no examples.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Convert this dataset into a [`LabeledCorpus`].
    ///
    /// - `chunker` is invoked to chunk each unique document so we can
    ///   map supporting sentences to gold chunk ids.
    /// - `embed_query` produces query embeddings; pass a noop closure
    ///   if you don't need them (e.g. lexical-only evaluation).
    /// - `label_regime` decides the [`RetrievalRegime`] for each
    ///   example. Use [`default_regime`] for the standard heuristic.
    pub fn to_labeled_corpus(
        &self,
        chunker: &dyn Chunker,
        mut embed_query: impl FnMut(&str) -> Option<Embedding>,
        mut label_regime: impl FnMut(&HotpotQAExample) -> RetrievalRegime,
    ) -> Result<LabeledCorpus> {
        // Deduplicate (title → sentences) across all examples. We trust
        // that the same title always carries the same sentences in
        // HotpotQA; if a future variant breaks this we'll need to key
        // on `(title, sentences_hash)` instead.
        let mut docs_by_title: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for ex in &self.examples {
            for (title, sentences) in &ex.context {
                docs_by_title
                    .entry(title.clone())
                    .or_insert_with(|| sentences.clone());
            }
        }

        // Build Documents and chunk them.
        let mut docs = Vec::new();
        // Keep a record of `(title, sentence_idx) → chunk_id` so we can
        // resolve gold chunk ids per query.
        let mut sentence_to_chunk: BTreeMap<(String, usize), ChunkId> = BTreeMap::new();

        for (title, sentences) in &docs_by_title {
            let joined = sentences.join(" ");
            let doc = Document::new(title.as_str(), joined);
            let chunks = chunker.chunk(&doc)?;
            for (s_idx, sentence) in sentences.iter().enumerate() {
                let s = sentence.trim();
                if s.is_empty() {
                    continue;
                }
                let chunk =
                    find_chunk_containing(&chunks, s).or_else(|| best_overlap_chunk(&chunks, s));
                if let Some(c) = chunk {
                    sentence_to_chunk.insert((title.clone(), s_idx), c.id.clone());
                }
            }
            docs.push(doc);
        }

        // Build labeled queries.
        let mut queries = Vec::with_capacity(self.examples.len());
        for ex in &self.examples {
            let mut q = LabeledQuery::new(ex.id.clone(), ex.question.clone(), label_regime(ex));
            if let Some(e) = embed_query(&ex.question) {
                q = q.with_embedding(e);
            }
            let mut gold = Vec::new();
            for (title, sent_idx) in &ex.supporting_facts {
                if let Some(id) = sentence_to_chunk.get(&(title.clone(), *sent_idx)) {
                    if !gold.contains(id) {
                        gold.push(id.clone());
                    }
                }
            }
            q.gold_chunk_ids = gold;
            queries.push(q);
        }

        Ok(LabeledCorpus { docs, queries })
    }
}

/// The standard heuristic mapping from `(level, type)` to
/// [`RetrievalRegime`]. See module-level documentation for the
/// rationale.
pub fn default_regime(ex: &HotpotQAExample) -> RetrievalRegime {
    match (ex.level.as_str(), ex.kind.as_str()) {
        ("easy", _) => RetrievalRegime::Easy,
        ("medium", "comparison") => RetrievalRegime::Easy,
        ("medium", _) => RetrievalRegime::Ambiguous,
        ("hard", "comparison") => RetrievalRegime::Ambiguous,
        ("hard", _) => RetrievalRegime::DistractorHeavy,
        _ => RetrievalRegime::Easy,
    }
}

fn find_chunk_containing<'a>(chunks: &'a [Chunk], needle: &str) -> Option<&'a Chunk> {
    chunks.iter().find(|c| c.text.contains(needle))
}

/// Fallback for sentences that don't appear verbatim in any chunk
/// (e.g. when whitespace was normalized differently). Picks the chunk
/// with the longest shared whitespace-tokenized substring against the
/// sentence.
fn best_overlap_chunk<'a>(chunks: &'a [Chunk], needle: &str) -> Option<&'a Chunk> {
    if chunks.is_empty() {
        return None;
    }
    let needle_words: Vec<&str> = needle.split_whitespace().collect();
    if needle_words.is_empty() {
        return Some(&chunks[0]);
    }
    let mut best: Option<(&Chunk, usize)> = None;
    for c in chunks {
        let c_words: Vec<&str> = c.text.split_whitespace().collect();
        let overlap = longest_contiguous_overlap(&c_words, &needle_words);
        if best.map(|(_, o)| overlap > o).unwrap_or(true) {
            best = Some((c, overlap));
        }
    }
    best.map(|(c, _)| c)
}

fn longest_contiguous_overlap(haystack: &[&str], needle: &[&str]) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut best = 0usize;
    for start in 0..haystack.len() {
        let mut k = 0;
        while k < needle.len() && start + k < haystack.len() && haystack[start + k] == needle[k] {
            k += 1;
        }
        if k > best {
            best = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
    use redhop::core::TokenizerBackend;
    use std::sync::Arc;

    const MINI_HOTPOTQA: &str = r#"[
        {
            "_id": "ex1",
            "question": "Are penguins flightless birds?",
            "answer": "Yes",
            "type": "comparison",
            "level": "easy",
            "supporting_facts": [["Penguin", 0]],
            "context": [
                ["Penguin", ["Penguins are flightless seabirds.", "They live mostly in the Southern Hemisphere."]],
                ["Albatross", ["Albatrosses are large seabirds that can fly long distances."]]
            ]
        },
        {
            "_id": "ex2",
            "question": "Was the Eiffel Tower built before the Statue of Liberty?",
            "answer": "No",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": [["Eiffel Tower", 1], ["Statue of Liberty", 0]],
            "context": [
                ["Eiffel Tower", ["The Eiffel Tower is in Paris.", "It was completed in 1889."]],
                ["Statue of Liberty", ["The Statue of Liberty was dedicated in 1886."]]
            ]
        }
    ]"#;

    fn chunker() -> SentenceChunker {
        let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
        SentenceChunker::new(tok, 40, 60, 0).unwrap()
    }

    #[test]
    fn parses_canonical_json() {
        let d = HotpotQADataset::from_json(MINI_HOTPOTQA).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d.examples[0].supporting_facts.len(), 1);
    }

    #[test]
    fn builds_labeled_corpus_with_gold_chunk_ids() {
        let d = HotpotQADataset::from_json(MINI_HOTPOTQA).unwrap();
        let chunker = chunker();
        let corpus = d
            .to_labeled_corpus(&chunker, |_| None, default_regime)
            .unwrap();
        assert_eq!(corpus.queries.len(), 2);
        // Penguin + Albatross + Eiffel Tower + Statue of Liberty = 4
        // unique titles across the two examples.
        assert_eq!(corpus.docs.len(), 4);
        // Every example should have at least one gold chunk.
        for q in &corpus.queries {
            assert!(
                !q.gold_chunk_ids.is_empty(),
                "query {} got no gold chunks",
                q.id
            );
        }
    }

    #[test]
    fn default_regime_mapping_is_sensible() {
        let easy = HotpotQAExample {
            id: "x".into(),
            question: "q".into(),
            answer: "".into(),
            kind: "comparison".into(),
            level: "easy".into(),
            supporting_facts: vec![],
            context: vec![],
        };
        assert_eq!(default_regime(&easy), RetrievalRegime::Easy);

        let hard_bridge = HotpotQAExample {
            kind: "bridge".into(),
            level: "hard".into(),
            ..easy.clone()
        };
        assert_eq!(
            default_regime(&hard_bridge),
            RetrievalRegime::DistractorHeavy
        );
    }
}
