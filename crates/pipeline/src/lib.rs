//! # neorag-pipeline
//!
//! Top-level facade composing chunking, retrieval, optional reranking, and
//! diagnostics into a single ergonomic API.
//!
//! Most users should interact with NeoRAG through [`NeoRAG`] and its builder
//! [`NeoRAGBuilder`]; everything else in the workspace is reachable from here
//! through the underlying traits, which keeps every component swappable
//! without forking this facade.
//!
//! ## Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
//! use neorag_core::{Document, TokenizerBackend};
//! use neorag_pipeline::NeoRAG;
//! use neorag_retrieval::Bm25Retriever;
//! # async fn run() -> anyhow::Result<()> {
//! let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
//! let chunker = Arc::new(SentenceChunker::new(tok.clone(), 256, 384, 0)?);
//! let retriever = Bm25Retriever::new()?;
//! let mut rag = NeoRAG::builder()
//!     .with_chunker(chunker)
//!     .with_retriever(Arc::new(retriever))
//!     .build()?;
//! rag.ingest(vec![Document::new("doc1", "hello world")]).await?;
//! let results = rag.retrieve("hello", 5).await?;
//! let report = rag.diagnose(&"hello".into(), &results)?;
//! # let _ = report;
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

use neorag_core::{
    Budget, Chunker, DiagnosticsEngine, DiagnosticsReport, Document, Error, Query,
    RegimeClassifier, Reranker, RerankerLevel, Result, RetrievalResult, RetrievalState, Retriever,
};
use neorag_diagnostics::DefaultDiagnosticsEngine;
use neorag_orchestration::{
    compute_confidence, AdaptiveOrchestrator, ConservativeRulePolicy, DefaultActuator, Policy,
    RuleBasedClassifier,
};

/// Builder for [`NeoRAG`].
///
/// The builder enforces the *required* components (chunker, retriever) at
/// `build()` time. Optional components (reranker, custom diagnostics engine)
/// default to sensible no-ops.
pub struct NeoRAGBuilder {
    chunker: Option<Arc<dyn Chunker>>,
    retriever: Option<Arc<dyn Retriever>>,
    reranker: Option<Arc<dyn Reranker>>,
    diagnostics: Option<Arc<dyn DiagnosticsEngine>>,
    classifier: Option<Arc<dyn RegimeClassifier>>,
    /// Adaptive-only: rerankers keyed by their escalation tier. Used by
    /// the [`AdaptiveOrchestrator`] through [`NeoRAG::adaptive_run`]; the
    /// static `retrieve` path uses the single `reranker` field above.
    rerankers_cascade: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
    /// Adaptive-only: the policy. Defaults to [`ConservativeRulePolicy`].
    policy: Option<Arc<dyn Policy>>,
    /// Adaptive-only: budget override.
    adaptive_budget: Option<Budget>,
    candidate_k: usize,
}

impl Default for NeoRAGBuilder {
    fn default() -> Self {
        Self {
            chunker: None,
            retriever: None,
            reranker: None,
            diagnostics: None,
            classifier: None,
            rerankers_cascade: Vec::new(),
            policy: None,
            adaptive_budget: None,
            candidate_k: 32,
        }
    }
}

impl NeoRAGBuilder {
    /// Set the chunker (required).
    pub fn with_chunker(mut self, c: Arc<dyn Chunker>) -> Self {
        self.chunker = Some(c);
        self
    }

    /// Set the retriever (required).
    ///
    /// The retriever is wrapped in an `Arc` because [`NeoRAG`] needs shared
    /// ownership: indexing and retrieval may race in concurrent callers.
    /// If you also need mutable access for indexing, keep a separate handle
    /// to a `Bm25Retriever` (etc.) outside the facade and call its
    /// [`Retriever::index`] directly before wrapping it.
    pub fn with_retriever(mut self, r: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(r);
        self
    }

    /// Optional: attach a reranker that runs over the retriever's output.
    pub fn with_reranker(mut self, r: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(r);
        self
    }

    /// Optional: replace the default diagnostics engine.
    pub fn with_diagnostics(mut self, d: Arc<dyn DiagnosticsEngine>) -> Self {
        self.diagnostics = Some(d);
        self
    }

    /// Optional: attach a regime classifier.
    ///
    /// Configuring a classifier turns on regime annotation in
    /// [`NeoRAG::retrieve_with_state`]. The static `retrieve` and
    /// `diagnose` APIs are unaffected — they continue to behave exactly as
    /// they did before Phase 7. This is the read-only on-ramp to the
    /// adaptive layer.
    pub fn with_classifier(mut self, c: Arc<dyn RegimeClassifier>) -> Self {
        self.classifier = Some(c);
        self
    }

    /// Adaptive-only: register a reranker at a specific escalation tier.
    ///
    /// Multiple rerankers may be registered at distinct tiers; the policy
    /// chooses which tier to escalate to. Used exclusively by
    /// [`NeoRAG::adaptive_run`]; the static [`NeoRAG::retrieve`] path
    /// ignores this list and uses the single reranker (if any) configured
    /// via [`with_reranker`][`NeoRAGBuilder::with_reranker`].
    pub fn with_reranker_at(mut self, level: RerankerLevel, r: Arc<dyn Reranker>) -> Self {
        self.rerankers_cascade.push((level, r));
        self
    }

    /// Adaptive-only: override the policy. Defaults to
    /// [`ConservativeRulePolicy`].
    pub fn with_policy(mut self, p: Arc<dyn Policy>) -> Self {
        self.policy = Some(p);
        self
    }

    /// Adaptive-only: override the per-session compute budget.
    pub fn with_adaptive_budget(mut self, budget: Budget) -> Self {
        self.adaptive_budget = Some(budget);
        self
    }

    /// Optional: how many candidates to pull from the retriever before
    /// passing into the reranker. Ignored when no reranker is configured.
    pub fn with_candidate_k(mut self, k: usize) -> Self {
        self.candidate_k = k.max(1);
        self
    }

    /// Finalize the configuration and build the facade.
    pub fn build(self) -> Result<NeoRAG> {
        let chunker = self
            .chunker
            .ok_or(Error::MissingComponent("chunker"))?;
        let retriever = self
            .retriever
            .ok_or(Error::MissingComponent("retriever"))?;
        let diagnostics = self
            .diagnostics
            .unwrap_or_else(|| Arc::new(DefaultDiagnosticsEngine::new()));
        Ok(NeoRAG {
            chunker,
            retriever,
            reranker: self.reranker,
            diagnostics,
            classifier: self.classifier,
            rerankers_cascade: self.rerankers_cascade,
            policy: self.policy,
            adaptive_budget: self.adaptive_budget,
            candidate_k: self.candidate_k,
        })
    }
}

/// The top-level NeoRAG facade.
///
/// Holds a chunker, a retriever, optionally a reranker, and a diagnostics
/// engine. All components are accessed through traits, so users can swap
/// any of them — including replacing the entire retriever with a remote
/// service that implements [`Retriever`] — without touching this struct.
pub struct NeoRAG {
    chunker: Arc<dyn Chunker>,
    retriever: Arc<dyn Retriever>,
    reranker: Option<Arc<dyn Reranker>>,
    diagnostics: Arc<dyn DiagnosticsEngine>,
    classifier: Option<Arc<dyn RegimeClassifier>>,
    rerankers_cascade: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
    policy: Option<Arc<dyn Policy>>,
    adaptive_budget: Option<Budget>,
    candidate_k: usize,
}

impl NeoRAG {
    /// Start building a new pipeline.
    pub fn builder() -> NeoRAGBuilder {
        NeoRAGBuilder::default()
    }

    /// Ingest a batch of documents: chunk them, then hand the chunks to the
    /// retriever's own index path.
    ///
    /// Note: this requires a `&mut self` cast through `Arc::get_mut`-style
    /// gymnastics if the retriever is shared; in practice we keep a clone
    /// of the inner `Arc<dyn Retriever>` and call `index` on it through a
    /// small trick — see implementation.
    pub async fn ingest(&mut self, docs: Vec<Document>) -> Result<()> {
        let chunks = self.chunker.chunk_batch(&docs)?;
        // The Retriever::index signature takes `&mut self`. We obtain a
        // unique mutable view by routing through `Arc::get_mut`, which is
        // only sound if no other clones of `self.retriever` exist. In the
        // typical builder-then-use lifecycle that is the case; concurrent
        // callers wishing to index while retrieving should construct their
        // own retriever and call `index` on it directly before sharing.
        let retriever = Arc::get_mut(&mut self.retriever).ok_or_else(|| {
            Error::Storage(
                "retriever is shared via Arc and cannot be indexed through the facade; call index() on it directly before sharing"
                    .into(),
            )
        })?;
        retriever.index(&chunks).await
    }

    /// Run a query through retrieval (and optional reranking).
    pub async fn retrieve(
        &self,
        query: impl Into<Query>,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let query = query.into();
        let k = if self.reranker.is_some() {
            self.candidate_k.max(top_k)
        } else {
            top_k
        };
        let candidates = self.retriever.retrieve(&query, k).await?;
        if let Some(rr) = &self.reranker {
            rr.rerank(&query, candidates, top_k).await
        } else {
            Ok(candidates)
        }
    }

    /// Compute diagnostics for a query and its results.
    pub fn diagnose(
        &self,
        query: &Query,
        results: &[RetrievalResult],
    ) -> Result<DiagnosticsReport> {
        self.diagnostics.diagnose(query, results)
    }

    /// Run retrieval and return a full [`RetrievalState`] — candidates,
    /// diagnostics, confidence profile, and (if a classifier is
    /// configured) regime distribution.
    ///
    /// This is the **read-only on-ramp to the adaptive layer**: it is the
    /// same call shape that the Phase 8 adaptive orchestrator will use
    /// internally on every iteration. Today the state is observed and
    /// returned; no part of the pipeline mutates retrieval based on it.
    ///
    /// Static `retrieve` and `diagnose` keep their existing contracts and
    /// are unaffected by whether a classifier is configured.
    pub async fn retrieve_with_state(
        &self,
        query: impl Into<Query>,
        top_k: usize,
    ) -> Result<RetrievalState> {
        let query = query.into();
        let candidates = self.retrieve(query.clone(), top_k).await?;
        let diagnostics = self.diagnostics.diagnose(&query, &candidates)?;
        let confidence = compute_confidence(&candidates);
        let mut state = RetrievalState::new(query, candidates, diagnostics, confidence);
        if let Some(cls) = &self.classifier {
            let dist = cls.classify(&state.diagnostics, &state.confidence);
            state = state.with_regime(dist);
        }
        Ok(state)
    }

    /// Run the **adaptive** retrieval loop. This is the Phase 8 entrypoint.
    ///
    /// The orchestrator behind this call is conservative by design: easy
    /// queries take exactly one terminal `Stop` action, sparse queries
    /// abstain immediately, distractor-heavy queries escalate the reranker
    /// at most once, and ambiguous queries expand top-k at most once.
    /// Inspect [`RetrievalState::history`] for the full audit trail.
    ///
    /// Requires a classifier (configure with
    /// [`NeoRAGBuilder::with_classifier`]). Falls back to the default
    /// [`ConservativeRulePolicy`] if no policy was set.
    pub async fn adaptive_run(&self, query: impl Into<Query>) -> Result<RetrievalState> {
        let classifier = self
            .classifier
            .clone()
            .ok_or(Error::MissingComponent("classifier"))?;
        let policy = self
            .policy
            .clone()
            .unwrap_or_else(|| Arc::new(ConservativeRulePolicy::new()));
        let actuator = Arc::new(DefaultActuator::new(
            self.retriever.clone(),
            self.rerankers_cascade.clone(),
        ));
        let mut orchestrator = AdaptiveOrchestrator::new(
            self.diagnostics.clone(),
            classifier,
            policy,
            actuator,
        )
        .with_initial_top_k(self.candidate_k);
        if let Some(budget) = &self.adaptive_budget {
            orchestrator = orchestrator.with_budget(budget.clone());
        }
        orchestrator.run(query.into()).await
    }

    /// Names of the configured components, for logging / diagnostics.
    pub fn component_names(&self) -> ComponentNames {
        ComponentNames {
            chunker: self.chunker.name(),
            retriever: self.retriever.name(),
            reranker: self.reranker.as_ref().map(|r| r.name()),
            diagnostics: self.diagnostics.name(),
            classifier: self.classifier.as_ref().map(|c| c.name()),
            rerankers_cascade_size: self.rerankers_cascade.len(),
            policy: self.policy.as_ref().map(|p| p.name()),
        }
    }

    /// Construct a fully-default adaptive setup from the configured
    /// chunker + retriever. Adds a `LayeredDiagnosticsEngine` (lexical +
    /// semantic) and a `RuleBasedClassifier` if none were set. Convenience
    /// for callers that want the canonical adaptive configuration without
    /// wiring every component by hand.
    pub fn defaults_for_adaptive() -> NeoRAGBuilder {
        use neorag_diagnostics::{
            LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
        };
        let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
        let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
        let layered = LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic);
        NeoRAGBuilder::default()
            .with_diagnostics(Arc::new(layered))
            .with_classifier(Arc::new(RuleBasedClassifier::new()))
    }
}

/// Names of the components in a [`NeoRAG`] pipeline.
#[derive(Debug, Clone)]
pub struct ComponentNames {
    /// Chunker name.
    pub chunker: &'static str,
    /// Retriever name.
    pub retriever: &'static str,
    /// Reranker name, if any.
    pub reranker: Option<&'static str>,
    /// Diagnostics engine name.
    pub diagnostics: &'static str,
    /// Regime classifier name, if any.
    pub classifier: Option<&'static str>,
    /// Number of rerankers registered in the adaptive cascade.
    pub rerankers_cascade_size: usize,
    /// Policy name, if any.
    pub policy: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
    use neorag_core::TokenizerBackend;
    use neorag_retrieval::Bm25Retriever;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn end_to_end_bm25() {
        rt().block_on(async {
            let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
            let chunker = Arc::new(SentenceChunker::new(tok.clone(), 16, 24, 0).unwrap());
            let retriever = Arc::new(Bm25Retriever::new().unwrap());

            let mut rag = NeoRAG::builder()
                .with_chunker(chunker)
                .with_retriever(retriever)
                .build()
                .unwrap();

            rag.ingest(vec![
                Document::new(
                    "tokio",
                    "Tokio is an asynchronous runtime for Rust. It powers async applications.",
                ),
                Document::new(
                    "django",
                    "Django is a high-level Python web framework. It encourages rapid development.",
                ),
            ])
            .await
            .unwrap();

            let results = rag.retrieve("rust async runtime", 3).await.unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].chunk.source, "tokio");

            let report = rag
                .diagnose(&Query::new("rust async runtime"), &results)
                .unwrap();
            assert!(report.lexical_grounding.is_some());
            assert!(report.retrieval_confidence.is_some());
        });
    }

    #[test]
    fn build_fails_without_required_components() {
        let r = NeoRAG::builder().build();
        assert!(r.is_err());
    }
}
