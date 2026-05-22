//! Actuators apply [`RetrievalAction`]s to a [`RetrievalState`].
//!
//! The actuator is the *only* component allowed to call the underlying
//! retriever and rerankers. Splitting it from the [`Policy`] is what lets
//! policies stay pure functions of state — every side effect happens
//! here, in one place, with measured latency and cost.
//!
//! ## What an actuator does
//!
//! - On `ExpandTopK { to }`: calls `retriever.retrieve(query, to)` and
//!   replaces the candidates.
//! - On `EscalateReranker { to }`: looks up the reranker registered at
//!   that level and applies it to the current candidates.
//! - On `Stop` / `Abstain`: no-op, terminal.
//!
//! After the call the actuator returns an [`ActuationOutcome`] with the
//! work it performed; the orchestrator records this into a
//! [`TakenAction`] alongside the policy's [`PolicyDecision`].
//!
//! [`Policy`]: crate::policy::Policy
//! [`PolicyDecision`]: crate::policy::PolicyDecision

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use redhop_core::{
    ActionCost, Error, Reranker, RerankerLevel, Result, RetrievalAction, RetrievalResult,
    RetrievalState, Retriever,
};

/// Output of one actuation.
#[derive(Debug, Clone)]
pub struct ActuationOutcome {
    /// New candidates, if the action mutated retrieval. `None` for
    /// terminal actions and for actions that left the candidates
    /// untouched.
    pub new_candidates: Option<Vec<RetrievalResult>>,
    /// New reranker level after this action, if changed.
    pub new_reranker_level: Option<RerankerLevel>,
    /// New `current_top_k` after this action, if changed.
    pub new_top_k: Option<usize>,
    /// Work performed.
    pub cost: ActionCost,
    /// Wall-clock latency.
    pub latency_ms: u64,
    /// True iff this is a terminal action and the loop should exit.
    pub terminal: bool,
}

impl ActuationOutcome {
    /// Outcome representing "nothing happened" — used by terminal
    /// actions.
    pub fn terminal(latency_ms: u64) -> Self {
        Self {
            new_candidates: None,
            new_reranker_level: None,
            new_top_k: None,
            cost: ActionCost::default(),
            latency_ms,
            terminal: true,
        }
    }
}

/// Trait shared by all actuators.
#[async_trait]
pub trait Actuator: Send + Sync {
    /// Apply the action to the state. Reads `state.query` (and other
    /// fields needed by the specific action) but does *not* mutate
    /// `state`; the orchestrator merges the [`ActuationOutcome`] into
    /// the state and records a [`TakenAction`][ta].
    ///
    /// [ta]: redhop_core::TakenAction
    async fn apply(
        &self,
        action: &RetrievalAction,
        state: &RetrievalState,
    ) -> Result<ActuationOutcome>;

    /// Initial retrieval. Called once before the first iteration. Kept
    /// separate from `apply` because initial retrieval has no action to
    /// pair with it; folding it into the action enum would complicate
    /// the policy for no benefit.
    async fn initial_retrieve(
        &self,
        query: &redhop_core::Query,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>>;

    /// Human-readable name.
    fn name(&self) -> &'static str;
}

/// Default actuator: retriever + a tiered reranker registry.
///
/// The registry holds at most one reranker per [`RerankerLevel`]. When
/// the policy asks for `EscalateReranker { to }` the actuator looks up
/// the reranker at that level and applies it. If no reranker is
/// registered at the requested level the actuator errors out — silently
/// failing here would mask a policy/configuration mismatch and that's
/// exactly the kind of opacity we are trying to avoid.
pub struct DefaultActuator {
    retriever: Arc<dyn Retriever>,
    rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
}

impl DefaultActuator {
    /// Construct from a retriever and a list of `(level, reranker)`
    /// pairs.
    pub fn new(
        retriever: Arc<dyn Retriever>,
        rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
    ) -> Self {
        Self {
            retriever,
            rerankers,
        }
    }

    /// Convenience: build an actuator with no rerankers configured. The
    /// policy will then only ever choose `Stop`, `Abstain`, or
    /// `ExpandTopK`; `EscalateReranker` becomes a configuration error
    /// the actuator surfaces rather than a silent no-op.
    pub fn retrieve_only(retriever: Arc<dyn Retriever>) -> Self {
        Self::new(retriever, Vec::new())
    }

    fn lookup_reranker(&self, level: RerankerLevel) -> Option<&Arc<dyn Reranker>> {
        self.rerankers
            .iter()
            .find_map(|(lvl, r)| if *lvl == level { Some(r) } else { None })
    }
}

#[async_trait]
impl Actuator for DefaultActuator {
    async fn initial_retrieve(
        &self,
        query: &redhop_core::Query,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        self.retriever.retrieve(query, top_k).await
    }

    async fn apply(
        &self,
        action: &RetrievalAction,
        state: &RetrievalState,
    ) -> Result<ActuationOutcome> {
        let start = Instant::now();
        match action {
            RetrievalAction::Stop { .. } | RetrievalAction::Abstain { .. } => {
                Ok(ActuationOutcome::terminal(elapsed_ms(start)))
            }
            RetrievalAction::ExpandTopK { from, to } => {
                let prev_len = state.candidates.len();
                let candidates = self.retriever.retrieve(&state.query, *to).await?;
                let new_len = candidates.len();
                Ok(ActuationOutcome {
                    new_candidates: Some(candidates),
                    new_reranker_level: None,
                    new_top_k: Some(*to),
                    cost: ActionCost {
                        retrieval_calls: 1,
                        rerank_calls: 0,
                        chunks_delta: new_len as i32 - prev_len as i32,
                    },
                    latency_ms: elapsed_ms(start),
                    terminal: false,
                    // Note: `from` is captured in the action enum itself
                    // and re-emitted via the orchestrator's TakenAction;
                    // we don't repeat it here.
                })
                .map(|mut o| {
                    let _ = from; // referenced for completeness via the action variant
                    o.new_top_k = Some(*to);
                    o
                })
            }
            RetrievalAction::EscalateReranker { from: _from, to } => {
                let reranker = self.lookup_reranker(*to).ok_or_else(|| {
                    Error::Reranking(format!(
                        "no reranker registered at level {}; policy proposed an escalation the actuator cannot execute",
                        to
                    ))
                })?;
                let prev_len = state.candidates.len();
                let candidates = reranker
                    .rerank(
                        &state.query,
                        state.candidates.clone(),
                        state.current_top_k.max(prev_len),
                    )
                    .await?;
                let new_len = candidates.len();
                Ok(ActuationOutcome {
                    new_candidates: Some(candidates),
                    new_reranker_level: Some(*to),
                    new_top_k: None,
                    cost: ActionCost {
                        retrieval_calls: 0,
                        rerank_calls: 1,
                        chunks_delta: new_len as i32 - prev_len as i32,
                    },
                    latency_ms: elapsed_ms(start),
                    terminal: false,
                })
            }
        }
    }

    fn name(&self) -> &'static str {
        "default"
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use redhop_core::{
        Chunk, ChunkId, Query, RetrievalMethod, Score, ScoreBreakdown, TokenCount,
    };
    use std::sync::Mutex;

    struct MockRetriever {
        // (top_k, result_count_to_return)
        results_to_return: Mutex<Vec<RetrievalResult>>,
        calls: Mutex<u32>,
    }
    impl MockRetriever {
        fn new(n: usize) -> Self {
            let results = (0..n).map(|i| mk_result(i)).collect();
            Self {
                results_to_return: Mutex::new(results),
                calls: Mutex::new(0),
            }
        }
    }
    fn mk_result(i: usize) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(
                ChunkId::new(format!("c{i}")),
                format!("text {i}"),
                "doc",
                TokenCount(1),
            ),
            score: Score {
                value: (10 - i as i32) as f32,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }
    #[async_trait]
    impl Retriever for MockRetriever {
        async fn retrieve(&self, _q: &Query, top_k: usize) -> Result<Vec<RetrievalResult>> {
            *self.calls.lock().unwrap() += 1;
            let r = self.results_to_return.lock().unwrap();
            Ok(r.iter().take(top_k).cloned().collect())
        }
        async fn index(&mut self, _c: &[Chunk]) -> Result<()> {
            Ok(())
        }
        fn name(&self) -> &'static str {
            "mock"
        }
    }

    struct ReverseReranker;
    #[async_trait]
    impl Reranker for ReverseReranker {
        async fn rerank(
            &self,
            _q: &Query,
            mut candidates: Vec<RetrievalResult>,
            top_k: usize,
        ) -> Result<Vec<RetrievalResult>> {
            candidates.reverse();
            candidates.truncate(top_k);
            Ok(candidates)
        }
        fn name(&self) -> &'static str {
            "reverse"
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn stop_is_no_op_terminal() {
        rt().block_on(async {
            let act = DefaultActuator::retrieve_only(Arc::new(MockRetriever::new(5)));
            let state = RetrievalState::new(
                Query::new("q"),
                vec![mk_result(0)],
                Default::default(),
                Default::default(),
            );
            let outcome = act
                .apply(
                    &RetrievalAction::Stop {
                        reason: redhop_core::StopReason::Confident,
                    },
                    &state,
                )
                .await
                .unwrap();
            assert!(outcome.terminal);
            assert!(outcome.new_candidates.is_none());
            assert_eq!(outcome.cost.retrieval_calls, 0);
        });
    }

    #[test]
    fn expand_top_k_re_retrieves() {
        rt().block_on(async {
            let retriever = Arc::new(MockRetriever::new(10));
            let act = DefaultActuator::retrieve_only(retriever.clone());
            let mut state = RetrievalState::new(
                Query::new("q"),
                vec![mk_result(0)],
                Default::default(),
                Default::default(),
            );
            state.current_top_k = 1;
            let outcome = act
                .apply(
                    &RetrievalAction::ExpandTopK { from: 1, to: 5 },
                    &state,
                )
                .await
                .unwrap();
            assert!(!outcome.terminal);
            let nc = outcome.new_candidates.unwrap();
            assert_eq!(nc.len(), 5);
            assert_eq!(outcome.cost.retrieval_calls, 1);
            assert_eq!(outcome.new_top_k, Some(5));
        });
    }

    #[test]
    fn escalate_reranker_runs_registered_reranker() {
        rt().block_on(async {
            let act = DefaultActuator::new(
                Arc::new(MockRetriever::new(3)),
                vec![(RerankerLevel::Lexical, Arc::new(ReverseReranker))],
            );
            let mut state = RetrievalState::new(
                Query::new("q"),
                vec![mk_result(0), mk_result(1), mk_result(2)],
                Default::default(),
                Default::default(),
            );
            state.current_top_k = 3;
            let outcome = act
                .apply(
                    &RetrievalAction::EscalateReranker {
                        from: RerankerLevel::None,
                        to: RerankerLevel::Lexical,
                    },
                    &state,
                )
                .await
                .unwrap();
            let nc = outcome.new_candidates.unwrap();
            assert_eq!(nc[0].chunk.id.as_str(), "c2");
            assert_eq!(outcome.cost.rerank_calls, 1);
            assert_eq!(outcome.new_reranker_level, Some(RerankerLevel::Lexical));
        });
    }

    #[test]
    fn escalate_without_registered_reranker_errors() {
        rt().block_on(async {
            let act = DefaultActuator::retrieve_only(Arc::new(MockRetriever::new(3)));
            let state = RetrievalState::new(
                Query::new("q"),
                vec![mk_result(0)],
                Default::default(),
                Default::default(),
            );
            let r = act
                .apply(
                    &RetrievalAction::EscalateReranker {
                        from: RerankerLevel::None,
                        to: RerankerLevel::Semantic,
                    },
                    &state,
                )
                .await;
            assert!(r.is_err());
        });
    }
}
