//! Labeled dataset types.
//!
//! A [`LabeledCorpus`] is the input to every calibration run. It carries:
//!
//! - the documents (chunked and ready to index),
//! - the queries to evaluate,
//! - for each query: ground-truth regime label + the set of "gold" chunk
//!   ids the retriever ought to surface.
//!
//! The gold-chunk-id label is what lets us measure intervention utility
//! without invoking an LLM-as-judge: an intervention is *useful* if it
//! lifts gold-chunk recall, *harmful* if it removes a gold chunk from
//! the top-k, *neutral* if it leaves recall unchanged. The principle is
//! the same as a classic IR test set; only the labels are required up
//! front.

use redhop::core::{ChunkId, Document, Embedding, RetrievalRegime};
use serde::{Deserialize, Serialize};

/// One labeled query: text, optional precomputed embedding, ground-truth
/// regime, and the set of chunk ids that count as gold evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledQuery {
    /// Stable identifier; used as a key in per-query metric tables.
    pub id: String,
    /// Query text.
    pub text: String,
    /// Pre-computed query embedding, optional. Required for the semantic
    /// diagnostics tier; if absent, the runner skips semantic metrics
    /// silently.
    pub embedding: Option<Embedding>,
    /// Ground-truth regime label. Used by
    /// [`crate::reliability::reliability_diagram`].
    pub true_regime: RetrievalRegime,
    /// Chunk ids that count as "right answer" evidence. Recall is
    /// measured against this set.
    pub gold_chunk_ids: Vec<ChunkId>,
}

impl LabeledQuery {
    /// Convenience constructor.
    pub fn new(
        id: impl Into<String>,
        text: impl Into<String>,
        true_regime: RetrievalRegime,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            embedding: None,
            true_regime,
            gold_chunk_ids: Vec::new(),
        }
    }

    /// Builder: attach an embedding.
    pub fn with_embedding(mut self, e: Embedding) -> Self {
        self.embedding = Some(e);
        self
    }

    /// Builder: attach the gold chunk ids.
    pub fn with_gold(mut self, ids: impl IntoIterator<Item = ChunkId>) -> Self {
        self.gold_chunk_ids = ids.into_iter().collect();
        self
    }
}

/// A complete labeled corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledCorpus {
    /// Documents to be chunked and indexed.
    pub docs: Vec<Document>,
    /// Queries to evaluate.
    pub queries: Vec<LabeledQuery>,
}

impl LabeledCorpus {
    /// Construct an empty corpus.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            queries: Vec::new(),
        }
    }

    /// True iff there are no queries to evaluate.
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// Number of queries.
    pub fn len(&self) -> usize {
        self.queries.len()
    }
}

impl Default for LabeledCorpus {
    fn default() -> Self {
        Self::new()
    }
}
