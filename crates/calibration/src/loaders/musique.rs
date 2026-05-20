//! MuSiQue loader.
//!
//! MuSiQue examples (the `_ans` and `_full` variants) have this shape:
//!
//! ```json
//! {
//!   "id": "...",
//!   "question": "...",
//!   "answer": "...",
//!   "answerable": true,
//!   "question_decomposition": [...],
//!   "paragraphs": [
//!     {"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true},
//!     {"idx": 1, "title": "...", "paragraph_text": "...", "is_supporting": false},
//!     ...
//!   ]
//! }
//! ```
//!
//! The conversion to [`LabeledCorpus`][lc] mirrors the HotpotQA loader:
//! each unique `(title, paragraph_text)` becomes a [`Document`], and
//! each query's gold chunks are derived from the supporting paragraphs.
//!
//! ## Regime labeling
//!
//! The MuSiQue regime heuristic:
//!
//! | answerable | hop count | regime          |
//! |------------|-----------|------------------|
//! | false      | —         | Sparse           |
//! | true       | 2         | Ambiguous        |
//! | true       | 3+        | DistractorHeavy  |
//!
//! Unanswerable MuSiQue questions are the canonical Sparse regime —
//! the corpus does not contain the answer. Multi-hop questions with
//! more hops have more chances to fall off the gold path, and tend to
//! retrieve more distractor paragraphs in our HotpotQA traces too;
//! the heuristic reflects that.
//!
//! [lc]: crate::dataset::LabeledCorpus

use std::collections::BTreeMap;

use neorag_core::{ChunkId, Chunker, Document, Embedding, Error, Result, RetrievalRegime};
use serde::{Deserialize, Serialize};

use crate::dataset::{LabeledCorpus, LabeledQuery};

/// One paragraph in a MuSiQue example.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MuSiQueParagraph {
    /// Paragraph index within the example.
    pub idx: usize,
    /// Paragraph title.
    pub title: String,
    /// Paragraph body.
    pub paragraph_text: String,
    /// True iff this paragraph is required to answer the question.
    pub is_supporting: bool,
}

/// One MuSiQue example.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MuSiQueExample {
    /// Stable example id.
    pub id: String,
    /// Question text.
    pub question: String,
    /// Gold answer string.
    #[serde(default)]
    pub answer: String,
    /// True if the question is answerable from the provided context.
    /// MuSiQue's "Full" subset includes unanswerable questions; Ans
    /// only includes answerable.
    #[serde(default = "default_true")]
    pub answerable: bool,
    /// Decomposition steps (we keep the count as a hop estimate).
    #[serde(default)]
    pub question_decomposition: Vec<serde_json::Value>,
    /// Paragraphs in the example.
    pub paragraphs: Vec<MuSiQueParagraph>,
}

fn default_true() -> bool {
    true
}

/// A loaded MuSiQue dataset.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MuSiQueDataset {
    /// All loaded examples.
    pub examples: Vec<MuSiQueExample>,
}

impl MuSiQueDataset {
    /// Parse from a JSON string (an array of examples). MuSiQue is
    /// frequently distributed as JSONL; for that format use
    /// [`Self::from_jsonl`] instead.
    pub fn from_json(s: &str) -> Result<Self> {
        let examples: Vec<MuSiQueExample> =
            serde_json::from_str(s).map_err(|e| Error::msg(format!("musique parse: {e}")))?;
        Ok(Self { examples })
    }

    /// Parse from a JSONL string (one example per line).
    pub fn from_jsonl(s: &str) -> Result<Self> {
        let mut examples = Vec::new();
        for (i, line) in s.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ex: MuSiQueExample = serde_json::from_str(line).map_err(|e| {
                Error::msg(format!("musique jsonl parse at line {}: {e}", i + 1))
            })?;
            examples.push(ex);
        }
        Ok(Self { examples })
    }

    /// Parse from a file at `path`. Auto-detects array vs JSONL based on
    /// the first non-whitespace character.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let first = text.trim_start().chars().next().unwrap_or(' ');
        if first == '[' {
            Self::from_json(&text)
        } else {
            Self::from_jsonl(&text)
        }
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
    pub fn to_labeled_corpus(
        &self,
        chunker: &dyn Chunker,
        mut embed_query: impl FnMut(&str) -> Option<Embedding>,
        mut label_regime: impl FnMut(&MuSiQueExample) -> RetrievalRegime,
    ) -> Result<LabeledCorpus> {
        // Deduplicate (title → text). MuSiQue paragraphs are short enough
        // that the same title typically carries the same body; if it
        // doesn't, the first occurrence wins (later ones are dropped).
        // We keep a per-example record of `(title → chunk_id)` so we
        // can resolve gold chunks even when text duplicates exist.
        let mut docs_by_title: BTreeMap<String, String> = BTreeMap::new();
        for ex in &self.examples {
            for p in &ex.paragraphs {
                docs_by_title
                    .entry(p.title.clone())
                    .or_insert_with(|| p.paragraph_text.clone());
            }
        }

        let mut docs = Vec::new();
        let mut title_to_chunks: BTreeMap<String, Vec<ChunkId>> = BTreeMap::new();
        for (title, body) in &docs_by_title {
            let doc = Document::new(title.as_str(), body.as_str());
            let chunks = chunker.chunk(&doc)?;
            title_to_chunks.insert(
                title.clone(),
                chunks.iter().map(|c| c.id.clone()).collect(),
            );
            docs.push(doc);
        }

        let mut queries = Vec::with_capacity(self.examples.len());
        for ex in &self.examples {
            let mut q = LabeledQuery::new(ex.id.clone(), ex.question.clone(), label_regime(ex));
            if let Some(e) = embed_query(&ex.question) {
                q = q.with_embedding(e);
            }
            // For MuSiQue, gold = all chunks of every supporting paragraph.
            let mut gold = Vec::new();
            for p in &ex.paragraphs {
                if !p.is_supporting {
                    continue;
                }
                if let Some(ids) = title_to_chunks.get(&p.title) {
                    for id in ids {
                        if !gold.contains(id) {
                            gold.push(id.clone());
                        }
                    }
                }
            }
            q.gold_chunk_ids = gold;
            queries.push(q);
        }

        Ok(LabeledCorpus { docs, queries })
    }
}

/// The standard MuSiQue regime mapping. See module-level docs.
pub fn default_regime(ex: &MuSiQueExample) -> RetrievalRegime {
    if !ex.answerable {
        return RetrievalRegime::Sparse;
    }
    let hops = ex.question_decomposition.len();
    if hops >= 3 {
        RetrievalRegime::DistractorHeavy
    } else {
        RetrievalRegime::Ambiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
    use neorag_core::TokenizerBackend;
    use std::sync::Arc;

    const MINI_MUSIQUE: &str = r#"[
        {
            "id": "m1",
            "question": "Where was the inventor of the safety lamp born?",
            "answer": "Penzance",
            "answerable": true,
            "question_decomposition": [{}, {}],
            "paragraphs": [
                {"idx": 0, "title": "Humphry Davy", "paragraph_text": "Humphry Davy invented the safety lamp.", "is_supporting": true},
                {"idx": 1, "title": "Penzance", "paragraph_text": "Penzance is a town in Cornwall.", "is_supporting": true},
                {"idx": 2, "title": "London", "paragraph_text": "London is the capital of England.", "is_supporting": false}
            ]
        },
        {
            "id": "m2",
            "question": "What was Beethoven's third symphony?",
            "answer": "Eroica",
            "answerable": false,
            "question_decomposition": [{}, {}, {}],
            "paragraphs": [
                {"idx": 0, "title": "Some town", "paragraph_text": "Some town has a population.", "is_supporting": false}
            ]
        }
    ]"#;

    fn chunker() -> SentenceChunker {
        let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
        SentenceChunker::new(tok, 40, 60, 0).unwrap()
    }

    #[test]
    fn parses_canonical_json() {
        let d = MuSiQueDataset::from_json(MINI_MUSIQUE).unwrap();
        assert_eq!(d.len(), 2);
        assert!(d.examples[1].answerable == false);
    }

    #[test]
    fn unanswerable_classified_as_sparse() {
        let d = MuSiQueDataset::from_json(MINI_MUSIQUE).unwrap();
        assert_eq!(default_regime(&d.examples[1]), RetrievalRegime::Sparse);
    }

    #[test]
    fn builds_labeled_corpus_with_gold_chunks_from_supporting_paragraphs() {
        let d = MuSiQueDataset::from_json(MINI_MUSIQUE).unwrap();
        let chunker = chunker();
        let corpus = d
            .to_labeled_corpus(&chunker, |_| None, default_regime)
            .unwrap();
        assert_eq!(corpus.queries.len(), 2);
        // m1 has two supporting paragraphs → at least 2 gold chunks.
        let m1 = corpus.queries.iter().find(|q| q.id == "m1").unwrap();
        assert!(m1.gold_chunk_ids.len() >= 2);
        // m2 is unanswerable → no supporting paragraphs → no gold chunks.
        let m2 = corpus.queries.iter().find(|q| q.id == "m2").unwrap();
        assert!(m2.gold_chunk_ids.is_empty());
    }
}
