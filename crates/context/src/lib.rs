//! # neorag-context
//!
//! Finite-attention-aware context construction. Given a query and the
//! chunks a retriever returned, build the *prompt context* a downstream
//! LLM actually sees — under a token budget, optimizing for
//! answer-bearing evidence density rather than raw top-k stuffing.
//!
//! The empirical motivation, from NeoRAG's own experiments:
//!
//! - Answer-bearing density matters strongly; distractors hurt strongly.
//! - More retrieval is often redundant.
//! - Continuity / topology is weak.
//!
//! So the highest-leverage operation is not "retrieve better" but
//! "**allocate the finite attention budget to the densest evidence and
//! stop wasting tokens on distractors and duplicates.**"
//!
//! This crate is deliberately **minimal**: one function,
//! [`build_context`], a small strategy enum, and an economics readout.
//! It is *assembly over already-retrieved chunks* — not a retriever, not
//! a reranker, not a framework. It reuses the same grounding/density
//! primitives the diagnostics tier uses.
//!
//! ```no_run
//! use neorag_context::{build_context, ContextConfig, ContextStrategy};
//! # use neorag_core::{Query, RetrievalResult};
//! # fn demo(query: &Query, chunks: &[RetrievalResult]) {
//! let ctx = build_context(
//!     query,
//!     chunks,
//!     &ContextConfig {
//!         token_budget: 1200,
//!         strategy: ContextStrategy::MaxDensity,
//!         ..Default::default()
//!     },
//! );
//! // ctx.chunks is the ordered, pruned context; ctx.economics reports
//! // evidence density, distractor ratio, redundancy, evidence-per-token,
//! // and estimated wasted tokens.
//! # let _ = ctx;
//! # }
//! ```
//!
//! ## The second-hop tax (and its mitigation)
//!
//! Strategies that prune by *query relevance* (distractor filtering,
//! max-density) drop chunks with low query relevance. On multi-hop
//! questions the second-hop evidence is *relevant to the bridge entity,
//! not the query* — exactly the chunk these strategies discard. This is
//! the **second-hop tax**: the same failure geometry that limits
//! ExpandTopK, query-passage reranking, and max-density pruning. It is
//! measured directly (n=1327, CI-backed) in the project's second-hop-tax
//! findings: a relevance filter at threshold 0.30 keeps only 44% of
//! second hops.
//!
//! [`ContextStrategy::ReasoningPreserving`] is the mitigation: keep
//! query-relevant seeds, *rescue* low-relevance chunks lexically linked
//! to a seed (the bridge entity), drop only unlinked junk. Measured to
//! recover much of the tax (+23 pts of second-hop retention at threshold
//! 0.30) at a modest junk-suppression cost. This is the project's
//! frontier in one function: *reasoning-aware evidence allocation, not
//! relevance optimization.* The economics readout stays honest about
//! what was dropped regardless of strategy.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::HashSet;

use neorag_core::{Chunk, Embedding, Query, RetrievalResult};
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

/// How to allocate the token budget across retrieved chunks.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum ContextStrategy {
    /// Baseline: take chunks in retrieval order until the budget is hit.
    /// This is what most RAG stacks do ("stuff the top-k").
    RawTopK,
    /// Drop chunks whose query grounding is below the distractor cutoff,
    /// then fill in retrieval order.
    DistractorFiltered,
    /// Add chunks in retrieval order but skip any chunk too similar to an
    /// already-selected one (suppresses redundancy). Requires embeddings.
    RedundancyPruned,
    /// Greedily select the highest evidence-density chunks first
    /// (most query-relevant tokens per chunk token), maximizing
    /// evidence-per-token within the budget.
    MaxDensity,
    /// Reasoning-aware selection that resists the **second-hop tax**.
    ///
    /// The other strategies prune by *query relevance*, which discards
    /// the multi-hop second hop (low-relevance-to-query but
    /// reasoning-critical). This strategy keeps two classes of chunk:
    ///
    /// 1. **seeds** — chunks above the query-grounding bar
    ///    (`distractor_min_grounding`), the clearly-relevant evidence;
    /// 2. **rescued second hops** — chunks *below* the bar that are
    ///    lexically *linked* to a seed (term-set Jaccard ≥
    ///    `link_min_jaccard`), i.e. they share the bridge entity with
    ///    relevant evidence.
    ///
    /// Only **true junk** — low query relevance *and* unlinked to any
    /// seed — is dropped. This is a single linkage step at assembly
    /// time, not graph traversal: no graph is built, no iteration, no
    /// topology. It is the minimal operation that distinguishes a
    /// distractor from a reasoning-critical second hop.
    ///
    /// Evidence (this strategy exists because the failure was measured):
    /// `docs/findings/SECOND_HOP_TAX.md` (n=1327, the tax + this
    /// mitigation's retention gain) and `docs/findings/REASONING_PRESERVATION.md`
    /// (n=300 end-to-end QA, +0.035 CI-significant, gain causally
    /// localized to gold reachability).
    ReasoningPreserving,
}

/// Configuration for [`build_context`].
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum total tokens in the assembled context.
    pub token_budget: usize,
    /// Allocation strategy.
    pub strategy: ContextStrategy,
    /// Per-chunk query-grounding cutoff below which a chunk is treated as
    /// a distractor (used by `DistractorFiltered`). In `[0, 1]`.
    pub distractor_min_grounding: f32,
    /// Cosine above which a chunk is "redundant" with an already-selected
    /// chunk (used by `RedundancyPruned`). In `[0, 1]`.
    pub redundancy_max_cosine: f32,
    /// Term-set Jaccard at/above which a low-relevance chunk is treated
    /// as *linked* to a seed (and therefore rescued as a possible second
    /// hop) rather than dropped as junk. Used by `ReasoningPreserving`.
    pub link_min_jaccard: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            token_budget: 2048,
            strategy: ContextStrategy::MaxDensity,
            distractor_min_grounding: 0.10,
            redundancy_max_cosine: 0.92,
            link_min_jaccard: 0.12,
        }
    }
}

/// Economics of an assembled context — what the token budget actually
/// bought.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextEconomics {
    /// Fraction of context tokens that are query-relevant (answer-bearing
    /// density proxy). Higher is better.
    pub evidence_density: f32,
    /// Fraction of selected chunks below the distractor grounding cutoff.
    /// Lower is better.
    pub distractor_ratio: f32,
    /// Mean pairwise cosine among selected chunk embeddings (semantic
    /// redundancy). Lower is better. `None` when embeddings are absent.
    pub redundancy: Option<f32>,
    /// Query-relevant tokens per total token — same as `evidence_density`
    /// but named for the cost framing.
    pub evidence_per_token: f32,
    /// Tokens used / budget.
    pub budget_utilization: f32,
    /// Estimated tokens spent on distractor chunks (wasted attention).
    pub estimated_waste_tokens: usize,
}

/// The result of context construction.
#[derive(Debug, Clone)]
pub struct BuiltContext {
    /// Selected chunks, in presentation order.
    pub chunks: Vec<Chunk>,
    /// Total tokens across the selected chunks.
    pub total_tokens: usize,
    /// Chunks dropped because they were distractors.
    pub n_dropped_distractor: usize,
    /// Chunks dropped because they were redundant.
    pub n_dropped_redundant: usize,
    /// Chunks dropped because the budget was exhausted.
    pub n_dropped_budget: usize,
    /// Economics readout.
    pub economics: ContextEconomics,
}

impl BuiltContext {
    /// True iff the assembled context contains a chunk with the given id.
    pub fn contains(&self, id: &neorag_core::ChunkId) -> bool {
        self.chunks.iter().any(|c| &c.id == id)
    }
}

/// Build a finite-attention-aware context from retrieved chunks.
pub fn build_context(
    query: &Query,
    retrieved: &[RetrievalResult],
    cfg: &ContextConfig,
) -> BuiltContext {
    let q_terms = terms(&query.text);

    // Per-chunk grounding + density, computed once.
    let mut items: Vec<Item> = retrieved
        .iter()
        .map(|r| {
            let c_terms = terms(&r.chunk.text);
            let grounding = grounding(&q_terms, &c_terms);
            let tok = r.chunk.token_count.value().max(1);
            // Density = query-relevant tokens / chunk tokens.
            let relevant = r
                .chunk
                .text
                .unicode_words()
                .filter(|w| q_terms.contains(&w.to_lowercase()))
                .count();
            Item {
                chunk: r.chunk.clone(),
                embedding: r.chunk.embedding.clone(),
                tokens: tok,
                grounding,
                density: relevant as f32 / tok as f32,
                is_distractor: grounding < cfg.distractor_min_grounding,
                c_terms,
            }
        })
        .collect();

    let n_distractor_total = items.iter().filter(|i| i.is_distractor).count();
    let mut n_dropped_distractor = 0;
    let mut n_dropped_redundant = 0;

    // Ordering / filtering per strategy.
    match cfg.strategy {
        ContextStrategy::RawTopK => { /* keep retrieval order */ }
        ContextStrategy::DistractorFiltered => {
            let before = items.len();
            items.retain(|i| !i.is_distractor);
            n_dropped_distractor = before - items.len();
        }
        ContextStrategy::MaxDensity => {
            items.sort_by(|a, b| {
                b.density
                    .partial_cmp(&a.density)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        ContextStrategy::RedundancyPruned => { /* handled in the fill loop */ }
        ContextStrategy::ReasoningPreserving => {
            // Seeds = clearly query-relevant chunks. Rescued = below-bar
            // chunks lexically linked to a seed (shared bridge entity).
            // True junk (below bar, unlinked) is dropped.
            let before = items.len();
            let seed_terms: Vec<HashSet<String>> = items
                .iter()
                .filter(|i| !i.is_distractor)
                .map(|i| i.c_terms.clone())
                .collect();
            items.retain(|i| {
                if !i.is_distractor {
                    return true; // seed
                }
                // Rescue if linked to any seed.
                seed_terms
                    .iter()
                    .any(|s| jaccard(&i.c_terms, s) >= cfg.link_min_jaccard)
            });
            // Order: seeds (by grounding desc) first, then rescued.
            items.sort_by(|a, b| {
                let a_seed = !a.is_distractor;
                let b_seed = !b.is_distractor;
                b_seed
                    .cmp(&a_seed)
                    .then(b.grounding.partial_cmp(&a.grounding).unwrap_or(std::cmp::Ordering::Equal))
            });
            n_dropped_distractor = before - items.len(); // true junk dropped
        }
    }

    // Fill under the token budget.
    let mut selected: Vec<Item> = Vec::new();
    let mut total = 0usize;
    let mut n_dropped_budget = 0;
    for item in items.into_iter() {
        if total + item.tokens > cfg.token_budget {
            n_dropped_budget += 1;
            continue;
        }
        if cfg.strategy == ContextStrategy::RedundancyPruned {
            if let Some(e) = &item.embedding {
                let redundant = selected.iter().any(|s| {
                    s.embedding
                        .as_ref()
                        .map(|se| cosine(e, se) > cfg.redundancy_max_cosine)
                        .unwrap_or(false)
                });
                if redundant {
                    n_dropped_redundant += 1;
                    continue;
                }
            }
        }
        total += item.tokens;
        selected.push(item);
    }

    let economics = economics(&q_terms, &selected, n_distractor_total, cfg);
    BuiltContext {
        chunks: selected.iter().map(|i| i.chunk.clone()).collect(),
        total_tokens: total,
        n_dropped_distractor,
        n_dropped_redundant,
        n_dropped_budget,
        economics,
    }
}

struct Item {
    chunk: Chunk,
    embedding: Option<Embedding>,
    tokens: usize,
    grounding: f32,
    density: f32,
    is_distractor: bool,
    c_terms: HashSet<String>,
}

/// Term-set Jaccard. The chunk↔chunk linkage signal: a multi-hop second
/// hop shares the bridge entity (often a multi-word proper noun) with a
/// relevant chunk, producing meaningful Jaccard even when its
/// query overlap is near zero.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn economics(
    q_terms: &HashSet<String>,
    selected: &[Item],
    _n_distractor_total: usize,
    cfg: &ContextConfig,
) -> ContextEconomics {
    if selected.is_empty() {
        return ContextEconomics {
            budget_utilization: 0.0,
            ..Default::default()
        };
    }
    let total_tokens: usize = selected.iter().map(|i| i.tokens).sum();
    let relevant_tokens: usize = selected
        .iter()
        .map(|i| {
            i.chunk
                .text
                .unicode_words()
                .filter(|w| q_terms.contains(&w.to_lowercase()))
                .count()
        })
        .sum();
    let density = if total_tokens > 0 {
        relevant_tokens as f32 / total_tokens as f32
    } else {
        0.0
    };
    let n_distractor = selected.iter().filter(|i| i.is_distractor).count();
    let waste_tokens: usize = selected
        .iter()
        .filter(|i| i.is_distractor)
        .map(|i| i.tokens)
        .sum();

    // Redundancy: mean pairwise cosine among selected embeddings.
    let embs: Vec<&Embedding> = selected.iter().filter_map(|i| i.embedding.as_ref()).collect();
    let redundancy = if embs.len() >= 2 {
        let mut acc = 0.0;
        let mut n = 0;
        for i in 0..embs.len() {
            for j in (i + 1)..embs.len() {
                acc += cosine(embs[i], embs[j]);
                n += 1;
            }
        }
        Some(acc / n as f32)
    } else {
        None
    };

    ContextEconomics {
        evidence_density: density,
        distractor_ratio: n_distractor as f32 / selected.len() as f32,
        redundancy,
        evidence_per_token: density,
        budget_utilization: total_tokens as f32 / cfg.token_budget.max(1) as f32,
        estimated_waste_tokens: waste_tokens,
    }
}

fn terms(text: &str) -> HashSet<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() > 1)
        .collect()
}

fn grounding(q: &HashSet<String>, c: &HashSet<String>) -> f32 {
    if q.is_empty() {
        return 0.0;
    }
    q.intersection(c).count() as f32 / q.len() as f32
}

fn cosine(a: &Embedding, b: &Embedding) -> f32 {
    let (a, b) = (a.as_slice(), b.as_slice());
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-9);
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use neorag_core::{ChunkId, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    fn rr(id: &str, text: &str, emb: Option<Vec<f32>>) -> RetrievalResult {
        let mut c = Chunk::new(
            ChunkId::new(id),
            text,
            "doc",
            TokenCount(text.split_whitespace().count()),
        );
        if let Some(e) = emb {
            c = c.with_embedding(Embedding::from(e));
        }
        RetrievalResult {
            chunk: c,
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Dense,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    #[test]
    fn raw_topk_respects_budget() {
        let q = Query::new("rust memory safety");
        let chunks = vec![
            rr("a", "rust memory safety ownership", None), // 4 tokens
            rr("b", "more rust safety discussion here", None), // 5 tokens
            rr("c", "totally unrelated cooking content", None), // 4 tokens
        ];
        let ctx = build_context(
            &q,
            &chunks,
            &ContextConfig {
                token_budget: 6,
                strategy: ContextStrategy::RawTopK,
                ..Default::default()
            },
        );
        assert!(ctx.total_tokens <= 6);
        assert_eq!(ctx.chunks[0].id.as_str(), "a"); // retrieval order preserved
    }

    #[test]
    fn distractor_filtered_drops_low_grounding() {
        let q = Query::new("rust memory safety");
        let chunks = vec![
            rr("a", "rust memory safety ownership", None),
            rr("c", "totally unrelated cooking recipe bread", None),
        ];
        let ctx = build_context(
            &q,
            &chunks,
            &ContextConfig {
                token_budget: 100,
                strategy: ContextStrategy::DistractorFiltered,
                distractor_min_grounding: 0.3,
                ..Default::default()
            },
        );
        // The cooking chunk shares 0 query terms → dropped.
        assert!(ctx.chunks.iter().all(|c| c.id.as_str() != "c"));
        assert_eq!(ctx.n_dropped_distractor, 1);
    }

    #[test]
    fn max_density_prefers_dense_chunks() {
        let q = Query::new("rust safety");
        let chunks = vec![
            // low density: 2 relevant of 8 tokens
            rr("dilute", "rust safety amid lots of extra filler words here now", None),
            // high density: 2 relevant of 2 tokens
            rr("dense", "rust safety", None),
        ];
        let ctx = build_context(
            &q,
            &chunks,
            &ContextConfig {
                token_budget: 2, // only room for the dense one
                strategy: ContextStrategy::MaxDensity,
                ..Default::default()
            },
        );
        assert_eq!(ctx.chunks.len(), 1);
        assert_eq!(ctx.chunks[0].id.as_str(), "dense");
    }

    #[test]
    fn redundancy_pruned_skips_near_duplicates() {
        let q = Query::new("rust");
        // Two near-identical embeddings + one different.
        let chunks = vec![
            rr("a", "rust one", Some(vec![1.0, 0.0, 0.0])),
            rr("b", "rust two", Some(vec![0.99, 0.01, 0.0])), // ~dup of a
            rr("c", "rust three", Some(vec![0.0, 1.0, 0.0])), // different
        ];
        let ctx = build_context(
            &q,
            &chunks,
            &ContextConfig {
                token_budget: 100,
                strategy: ContextStrategy::RedundancyPruned,
                redundancy_max_cosine: 0.9,
                ..Default::default()
            },
        );
        // b should be pruned as redundant with a.
        assert!(ctx.chunks.iter().any(|c| c.id.as_str() == "a"));
        assert!(ctx.chunks.iter().any(|c| c.id.as_str() == "c"));
        assert!(ctx.chunks.iter().all(|c| c.id.as_str() != "b"));
        assert_eq!(ctx.n_dropped_redundant, 1);
    }

    // Bridge question: hop1 names the inventor (query-relevant); hop2 is
    // ABOUT the inventor (low query relevance, but shares the bridge
    // entity "humphry davy"); junk is unrelated to both.
    fn bridge_chunks() -> Vec<RetrievalResult> {
        vec![
            // seed: shares {safety, lamp, was, the} with the query
            rr("hop1", "the safety lamp was invented by humphry davy", None),
            // second hop: shares almost nothing with the query, but shares
            // the bridge entity "humphry davy" with hop1
            rr("hop2", "humphry davy born penzance cornwall england chemist", None),
            // true junk: unrelated to query AND to hop1
            rr("junk", "photosynthesis converts sunlight chemical energy green plants", None),
        ]
    }

    fn bridge_cfg(s: ContextStrategy) -> ContextConfig {
        ContextConfig {
            token_budget: 1000,
            strategy: s,
            distractor_min_grounding: 0.25,
            link_min_jaccard: 0.05,
            redundancy_max_cosine: 1.0,
        }
    }

    #[test]
    fn reasoning_preserving_rescues_linked_second_hop_drops_true_junk() {
        let q = Query::new("what nationality was the safety lamp inventor");
        let ctx = build_context(&q, &bridge_chunks(), &bridge_cfg(ContextStrategy::ReasoningPreserving));
        let ids: Vec<&str> = ctx.chunks.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"hop1"), "seed hop1 should be kept");
        assert!(ids.contains(&"hop2"), "linked second hop should be rescued");
        assert!(!ids.contains(&"junk"), "true junk should be dropped");
    }

    #[test]
    fn distractor_filter_drops_the_second_hop_that_reasoning_preserving_keeps() {
        // The two strategies differ exactly on the reasoning-critical chunk.
        let q = Query::new("what nationality was the safety lamp inventor");
        let chunks = bridge_chunks();
        let filtered = build_context(&q, &chunks, &bridge_cfg(ContextStrategy::DistractorFiltered));
        let preserving = build_context(&q, &chunks, &bridge_cfg(ContextStrategy::ReasoningPreserving));
        assert!(!filtered.chunks.iter().any(|c| c.id.as_str() == "hop2"),
            "distractor filter should drop the low-relevance second hop");
        assert!(preserving.chunks.iter().any(|c| c.id.as_str() == "hop2"),
            "reasoning-preserving should rescue it");
    }

    #[test]
    fn economics_reports_density_and_waste() {
        let q = Query::new("rust memory safety");
        let chunks = vec![
            rr("a", "rust memory safety", None), // all relevant
            rr("c", "cooking bread recipe flour", None), // distractor
        ];
        let ctx = build_context(
            &q,
            &chunks,
            &ContextConfig {
                token_budget: 100,
                strategy: ContextStrategy::RawTopK,
                distractor_min_grounding: 0.3,
                ..Default::default()
            },
        );
        // Both kept (raw). One is a distractor → waste tokens > 0.
        assert!(ctx.economics.estimated_waste_tokens > 0);
        assert!(ctx.economics.distractor_ratio > 0.0);
        assert!(ctx.economics.evidence_density > 0.0);
    }
}
