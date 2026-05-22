//! # redhop-context
//!
//! Finite-attention-aware context construction. Given a query and the
//! chunks a retriever returned, build the *prompt context* a downstream
//! LLM actually sees — under a token budget, optimizing for
//! answer-bearing evidence density rather than raw top-k stuffing.
//!
//! The empirical motivation, from RedHop's own experiments:
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
//! use redhop_context::{build_context, ContextConfig};
//! # use redhop_core::{Query, RetrievalResult};
//! # fn demo(query: &Query, chunks: &[RetrievalResult]) {
//! // Default strategy is reasoning-preserving and safe.
//! let ctx = build_context(
//!     query,
//!     chunks,
//!     &ContextConfig { token_budget: 1200, ..Default::default() },
//! );
//! let prompt = ctx.text();        // drop-in for llm.generate(prompt)
//! let report = &ctx.report;       // evidence density, distractor ratio,
//!                                 // second-hop rescues, removed chunks, …
//! # let _ = (prompt, report);
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
use std::sync::OnceLock;

use redhop_core::{Chunk, Embedding, Query, RetrievalResult};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

/// How to allocate the token budget across retrieved chunks.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
            // Safe-by-default: ReasoningPreserving keeps relevant evidence,
            // removes only unlinked junk, and never aggressively prunes by
            // relevance (which the second-hop-tax findings show is harmful
            // on multi-hop). See docs/findings/REASONING_PRESERVATION.md.
            strategy: ContextStrategy::ReasoningPreserving,
            // A low absolute bar: only near-zero-overlap junk is below it.
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

/// Per-reason breakdown of chunks removed during assembly. `total` is the
/// sum of all removal reasons and always equals `n_input - n_selected`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RemovedBreakdown {
    /// Removed because below the query-grounding bar (and, for
    /// `ReasoningPreserving`, unlinked to any seed).
    pub distractor: usize,
    /// Removed because too similar to an already-selected chunk.
    pub redundant: usize,
    /// Removed because the token budget was exhausted.
    pub budget: usize,
    /// Total removed (`distractor + redundant + budget`).
    pub total: usize,
}

/// Full observability trace for one context assembly — the telemetry every
/// strategy emits. Superset of [`ContextEconomics`]; serializable for
/// benchmark/JSON output and deployment dashboards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextReport {
    /// Strategy that produced this context.
    pub strategy: ContextStrategy,
    /// Configured token budget.
    pub token_budget: usize,
    /// Tokens actually used.
    pub total_tokens: usize,
    /// `total_tokens / token_budget`. Always `<= 1.0` (the budget is a hard cap).
    pub token_utilization: f32,
    /// Chunks supplied to the assembler.
    pub n_input_chunks: usize,
    /// Chunks present in the assembled context.
    pub n_selected: usize,
    /// Fraction of *input* chunks below the grounding bar — an estimate of
    /// how distractor-heavy the retrieval was before assembly.
    pub input_distractor_ratio: f32,
    /// Seeds (query-relevant chunks) kept / seeds in the input. A
    /// label-free proxy for gold retention: the fraction of clearly-relevant
    /// evidence that survived assembly. `1.0` when nothing relevant was dropped.
    pub retained_evidence_ratio: f32,
    /// Low-relevance chunks deliberately *rescued* because they were linked
    /// to a seed (the second hop). Non-zero only for `ReasoningPreserving`.
    pub second_hop_rescue_count: usize,
    /// Reasoning-preserved chunks a plain distractor filter would have
    /// dropped (== `second_hop_rescue_count` for `ReasoningPreserving`, `0`
    /// otherwise). The measurable "what reasoning-preservation bought".
    pub reasoning_preservation_delta: usize,
    /// Per-reason removal breakdown.
    pub removed: RemovedBreakdown,
    /// Token/evidence economics of the assembled context.
    pub economics: ContextEconomics,
}

impl ContextReport {
    /// Render a human-readable "Context Optimization Report" — makes the
    /// invisible visible. Pass the `analyze_context` report as `before` to
    /// show the token/density deltas; pass `None` for the assembled view only.
    pub fn render(&self, before: Option<&ContextReport>) -> String {
        let mut s = String::new();
        s.push_str("Context Optimization Report\n");
        s.push_str("───────────────────────────\n");
        s.push_str(&format!("Strategy: {:?}\n\n", self.strategy));

        let in_chunks = before.map(|b| b.n_input_chunks).unwrap_or(self.n_input_chunks);
        s.push_str(&format!("Input chunks:        {in_chunks}\n"));
        s.push_str(&format!("Output chunks:       {}\n", self.n_selected));
        if let Some(b) = before {
            // Negative = fewer tokens than the raw input (the usual, good case).
            let pct = if b.total_tokens > 0 {
                100.0 * (self.total_tokens as f32 - b.total_tokens as f32) / b.total_tokens as f32
            } else {
                0.0
            };
            s.push_str(&format!(
                "Tokens:              {} → {}  ({pct:+.0}%)\n",
                b.total_tokens, self.total_tokens
            ));
        } else {
            s.push_str(&format!("Tokens:              {}\n", self.total_tokens));
        }
        s.push_str(&format!("Distractors pruned:  {}\n", self.removed.distractor));
        if self.removed.redundant > 0 {
            s.push_str(&format!("Duplicates pruned:   {}\n", self.removed.redundant));
        }
        if self.removed.budget > 0 {
            s.push_str(&format!("Budget-trimmed:      {}\n", self.removed.budget));
        }
        s.push_str(&format!("Reasoning rescues:   {}\n\n", self.second_hop_rescue_count));

        if let Some(b) = before {
            s.push_str(&format!(
                "Evidence density:    {:.2} → {:.2}\n",
                b.economics.evidence_density, self.economics.evidence_density
            ));
        } else {
            s.push_str(&format!("Evidence density:    {:.2}\n", self.economics.evidence_density));
        }
        s.push_str(&format!(
            "Retained evidence:   {:.0}%\n",
            self.retained_evidence_ratio * 100.0
        ));
        s.push_str(&format!("Token utilization:   {:.0}%\n", self.token_utilization * 100.0));
        s.push_str(&format!(
            "Estimated waste:     {} tokens on distractors\n",
            self.economics.estimated_waste_tokens
        ));

        // Warnings — surface what the optimizer did and didn't do.
        let mut warnings: Vec<String> = Vec::new();
        if self.second_hop_rescue_count > 0 {
            warnings.push(format!(
                "rescued {} low-relevance linked chunk(s) (possible second hops)",
                self.second_hop_rescue_count
            ));
        }
        if self.removed.redundant > 0 {
            warnings.push(format!("{} near-duplicate chunk(s) pruned", self.removed.redundant));
        }
        if self.removed.budget > 0 {
            warnings.push(format!(
                "{} chunk(s) dropped for token budget — consider raising it",
                self.removed.budget
            ));
        }
        if self.economics.distractor_ratio > 0.05 && self.removed.distractor == 0 {
            warnings.push(format!(
                "context still contains distractors (ratio {:.2}); strategy did not filter",
                self.economics.distractor_ratio
            ));
        }
        if !warnings.is_empty() {
            s.push_str("\nWarnings:\n");
            for w in warnings {
                s.push_str(&format!("- {w}\n"));
            }
        }
        s
    }
}

/// The result of context construction: the selected chunks plus the
/// [`ContextReport`] telemetry.
#[derive(Debug, Clone)]
pub struct BuiltContext {
    /// Selected chunks, in presentation order.
    pub chunks: Vec<Chunk>,
    /// Observability trace for this assembly.
    pub report: ContextReport,
}

impl BuiltContext {
    /// True iff the assembled context contains a chunk with the given id.
    pub fn contains(&self, id: &redhop_core::ChunkId) -> bool {
        self.chunks.iter().any(|c| &c.id == id)
    }

    /// Total tokens across the selected chunks.
    pub fn total_tokens(&self) -> usize {
        self.report.total_tokens
    }

    /// The assembled context as a single prompt string: each chunk on its
    /// own block, in presentation order. The drop-in for `llm.generate(...)`.
    pub fn text(&self) -> String {
        self.chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
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
    let mut items = characterize(&q_terms, retrieved, cfg);

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
            // Every below-bar chunk that survived was kept because it linked
            // to a seed → mark it rescued (reasoning evidence, not junk).
            for i in items.iter_mut().filter(|i| i.is_distractor) {
                i.rescued = true;
            }
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
    let report = make_report(
        cfg,
        retrieved.len(),
        n_distractor_total,
        &selected,
        total,
        RemovedBreakdown {
            distractor: n_dropped_distractor,
            redundant: n_dropped_redundant,
            budget: n_dropped_budget,
            total: n_dropped_distractor + n_dropped_redundant + n_dropped_budget,
        },
        economics,
    );
    BuiltContext {
        chunks: selected.iter().map(|i| i.chunk.clone()).collect(),
        report,
    }
}

/// Assemble the [`ContextReport`] from the selection result.
fn make_report(
    cfg: &ContextConfig,
    n_input: usize,
    n_input_distractor: usize,
    selected: &[Item],
    total_tokens: usize,
    removed: RemovedBreakdown,
    economics: ContextEconomics,
) -> ContextReport {
    let n_input_seed = n_input - n_input_distractor;
    let seeds_kept = selected.iter().filter(|i| !i.is_distractor).count();
    let retained_evidence_ratio = if n_input_seed == 0 {
        1.0
    } else {
        seeds_kept as f32 / n_input_seed as f32
    };
    // Rescued = below-bar chunks kept *on purpose* because linked to a seed
    // (flagged during the ReasoningPreserving pass; 0 for other strategies).
    let rescued = selected.iter().filter(|i| i.rescued).count();
    ContextReport {
        strategy: cfg.strategy,
        token_budget: cfg.token_budget,
        total_tokens,
        token_utilization: total_tokens as f32 / cfg.token_budget.max(1) as f32,
        n_input_chunks: n_input,
        n_selected: selected.len(),
        input_distractor_ratio: if n_input == 0 {
            0.0
        } else {
            n_input_distractor as f32 / n_input as f32
        },
        retained_evidence_ratio,
        second_hop_rescue_count: rescued,
        reasoning_preservation_delta: rescued,
        removed,
        economics,
    }
}

/// Reasoning-preserving filtering **without** budget truncation: remove
/// junk under the configured strategy but keep everything that survives the
/// filter, regardless of token count. The convenience entry point for
/// "clean up this retrieval, I'll manage the budget myself". Equivalent to
/// [`build_context`] with an unbounded token budget.
pub fn filter_context(
    query: &Query,
    retrieved: &[RetrievalResult],
    cfg: &ContextConfig,
) -> BuiltContext {
    let mut unbounded = cfg.clone();
    unbounded.token_budget = usize::MAX;
    build_context(query, retrieved, &unbounded)
}

/// Characterize a retrieved set **without** modifying it: report distractor
/// load, evidence density, redundancy, and how many low-relevance chunks
/// are rescuable second-hop *candidates* (linked to a seed). Pure
/// observability — nothing is dropped or reordered. Use it to decide
/// whether (and how aggressively) to filter.
pub fn analyze_context(
    query: &Query,
    retrieved: &[RetrievalResult],
    cfg: &ContextConfig,
) -> ContextReport {
    let q_terms = terms(&query.text);
    let items = characterize(&q_terms, retrieved, cfg);
    let n_input = items.len();
    let n_input_distractor = items.iter().filter(|i| i.is_distractor).count();
    let total_tokens: usize = items.iter().map(|i| i.tokens).sum();
    // Second-hop candidates: below-bar chunks linked to a seed (what
    // ReasoningPreserving *would* rescue).
    let seed_terms: Vec<&HashSet<String>> =
        items.iter().filter(|i| !i.is_distractor).map(|i| &i.c_terms).collect();
    let candidates = items
        .iter()
        .filter(|i| i.is_distractor)
        .filter(|i| seed_terms.iter().any(|s| jaccard(&i.c_terms, s) >= cfg.link_min_jaccard))
        .count();
    let economics = economics(&q_terms, &items, n_input_distractor, cfg);
    ContextReport {
        strategy: cfg.strategy,
        token_budget: cfg.token_budget,
        total_tokens,
        token_utilization: total_tokens as f32 / cfg.token_budget.max(1) as f32,
        n_input_chunks: n_input,
        n_selected: n_input,
        input_distractor_ratio: if n_input == 0 {
            0.0
        } else {
            n_input_distractor as f32 / n_input as f32
        },
        retained_evidence_ratio: 1.0,
        second_hop_rescue_count: candidates,
        reasoning_preservation_delta: candidates,
        removed: RemovedBreakdown::default(),
        economics,
    }
}

/// Economics of a chunk set as-is, without assembling (no filtering, no
/// budget). Reports evidence density, distractor ratio, redundancy, and
/// estimated wasted tokens over exactly the chunks given.
pub fn context_economics(
    query: &Query,
    chunks: &[RetrievalResult],
    cfg: &ContextConfig,
) -> ContextEconomics {
    let q_terms = terms(&query.text);
    let items = characterize(&q_terms, chunks, cfg);
    let n_distractor = items.iter().filter(|i| i.is_distractor).count();
    economics(&q_terms, &items, n_distractor, cfg)
}

/// Query grounding of a chunk's text: the relevance signal the strategies use —
/// stopword-removed, Snowball-stemmed query-term overlap, in `[0, 1]`. Exposed
/// for observability ("how relevant is this chunk to the query?") and so
/// external/eval code can reuse the library's exact notion of relevance instead
/// of reimplementing (and drifting from) it.
pub fn grounding_score(query: &str, text: &str) -> f32 {
    grounding(&terms(query), &terms(text))
}

/// Linkage strength between two chunks' text: term-set Jaccard over the same
/// normalized terms — the chunk↔chunk bridge signal `ReasoningPreserving` uses
/// to decide whether a low-relevance chunk is a rescuable second hop. In `[0, 1]`.
pub fn link_strength(a: &str, b: &str) -> f32 {
    jaccard(&terms(a), &terms(b))
}

struct Item {
    chunk: Chunk,
    embedding: Option<Embedding>,
    tokens: usize,
    grounding: f32,
    density: f32,
    is_distractor: bool,
    /// True iff this chunk is below the grounding bar but was *deliberately
    /// kept* because it is linked to a seed (a rescued second hop). Such a
    /// chunk is reasoning-critical evidence, not junk, so the economics must
    /// not count it as a distractor.
    rescued: bool,
    c_terms: HashSet<String>,
}

/// Compute per-chunk grounding, density, and the distractor flag once.
/// Shared by `build_context`, `filter_context`, `analyze_context`, and
/// `context_economics` so they all agree on what counts as a distractor.
fn characterize(
    q_terms: &HashSet<String>,
    retrieved: &[RetrievalResult],
    cfg: &ContextConfig,
) -> Vec<Item> {
    retrieved
        .iter()
        .map(|r| {
            let c_terms = terms(&r.chunk.text);
            let grounding = grounding(q_terms, &c_terms);
            let tok = r.chunk.token_count.value().max(1);
            let relevant = r
                .chunk
                .text
                .unicode_words()
                .filter_map(normalize).filter(|t| q_terms.contains(t))
                .count();
            Item {
                chunk: r.chunk.clone(),
                embedding: r.chunk.embedding.clone(),
                tokens: tok,
                grounding,
                density: relevant as f32 / tok as f32,
                is_distractor: grounding < cfg.distractor_min_grounding,
                rescued: false,
                c_terms,
            }
        })
        .collect()
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
                .filter_map(normalize).filter(|t| q_terms.contains(t))
                .count()
        })
        .sum();
    let density = if total_tokens > 0 {
        relevant_tokens as f32 / total_tokens as f32
    } else {
        0.0
    };
    // A rescued second hop is below the grounding bar but is reasoning
    // evidence, not junk — exclude it from the distractor ratio and waste.
    let is_true_distractor = |i: &&Item| i.is_distractor && !i.rescued;
    let n_distractor = selected.iter().filter(is_true_distractor).count();
    let waste_tokens: usize = selected
        .iter()
        .filter(is_true_distractor)
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

// High-frequency English function words that inflate raw lexical overlap.
// Dropping them sharpens the grounding/linkage signal — validated by the
// signal_ablation harness (gold-vs-distractor AUC 0.935→0.968 HotpotQA,
// 0.672→0.734 MuSiQue), CI-clear on both datasets.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "as", "by", "with", "from", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their", "which", "who", "whom",
    "what", "when", "where", "how", "why", "into", "than", "then", "there", "here", "out", "up",
    "down", "over", "under", "do", "does", "did", "has", "have", "had", "not", "no", "can", "will",
    "would", "should", "could", "may", "might", "about", "between", "during", "such", "also",
];

fn stopword_set() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| STOPWORDS.iter().copied().collect())
}

/// Normalize a surface token into its matching term, or `None` if it carries
/// no signal (too short, or a stopword). Lowercases, drops stopwords, and
/// applies Snowball (Porter2) stemming so morphological variants
/// ("invented"/"invention"/"invents") match. Snowball stemming was validated
/// over a crude stand-in in the ablation harness (AUC 0.973→0.975 HotpotQA,
/// 0.762→0.768 MuSiQue).
fn normalize(w: &str) -> Option<String> {
    let lower = w.to_lowercase();
    if lower.chars().count() <= 1 || stopword_set().contains(lower.as_str()) {
        return None;
    }
    thread_local!(static STEMMER: Stemmer = Stemmer::create(Algorithm::English));
    Some(STEMMER.with(|s| s.stem(&lower).into_owned()))
}

fn terms(text: &str) -> HashSet<String> {
    text.unicode_words().filter_map(normalize).collect()
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
    use redhop_core::{ChunkId, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

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
        assert!(ctx.total_tokens() <= 6);
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
        assert_eq!(ctx.report.removed.distractor, 1);
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
        assert_eq!(ctx.report.removed.redundant, 1);
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
        assert!(ctx.report.economics.estimated_waste_tokens > 0);
        assert!(ctx.report.economics.distractor_ratio > 0.0);
        assert!(ctx.report.economics.evidence_density > 0.0);
    }

    // ───────────────────────── invariant / regression contracts ─────────────
    // These turn the findings into enforced guarantees (Phase goal 5).

    fn all_strategies() -> [ContextStrategy; 5] {
        [
            ContextStrategy::RawTopK,
            ContextStrategy::DistractorFiltered,
            ContextStrategy::RedundancyPruned,
            ContextStrategy::MaxDensity,
            ContextStrategy::ReasoningPreserving,
        ]
    }

    #[test]
    fn invariant_default_strategy_is_not_aggressive_relevance_filtering() {
        // The default must never be a relevance-only pruner — the findings
        // show those tax the second hop on multi-hop QA.
        let s = ContextConfig::default().strategy;
        assert_ne!(s, ContextStrategy::DistractorFiltered);
        assert_ne!(s, ContextStrategy::MaxDensity);
        assert_eq!(s, ContextStrategy::ReasoningPreserving);
    }

    #[test]
    fn invariant_no_strategy_exceeds_token_budget() {
        let q = Query::new("rust memory safety guarantees");
        let chunks = vec![
            rr("a", "rust memory safety guarantees ownership borrow checker", None),
            rr("b", "rust prevents data races at compile time safety", None),
            rr("c", "cooking bread recipe flour yeast water salt oven", None),
            rr("d", "memory safety without garbage collection in rust", None),
        ];
        for s in all_strategies() {
            for budget in [1usize, 3, 5, 8, 50] {
                let ctx = build_context(
                    &q,
                    &chunks,
                    &ContextConfig { token_budget: budget, strategy: s, ..Default::default() },
                );
                assert!(
                    ctx.total_tokens() <= budget,
                    "{s:?} exceeded budget {budget}: used {}",
                    ctx.total_tokens()
                );
                assert!(ctx.report.token_utilization <= 1.0 + 1e-6, "{s:?} util > 1");
            }
        }
    }

    #[test]
    fn invariant_removed_breakdown_accounts_for_every_chunk() {
        // Observability contract: removed.total == input - selected, always.
        let q = Query::new("rust memory safety");
        let chunks = vec![
            rr("a", "rust memory safety", Some(vec![1.0, 0.0])),
            rr("b", "rust memory safety again", Some(vec![0.99, 0.01])),
            rr("c", "cooking bread recipe flour", Some(vec![0.0, 1.0])),
        ];
        for s in all_strategies() {
            let ctx = build_context(
                &q,
                &chunks,
                &ContextConfig {
                    token_budget: 6,
                    strategy: s,
                    distractor_min_grounding: 0.3,
                    redundancy_max_cosine: 0.9,
                    ..Default::default()
                },
            );
            let r = &ctx.report;
            assert_eq!(r.n_input_chunks, chunks.len());
            assert_eq!(r.n_selected, ctx.chunks.len());
            assert_eq!(
                r.removed.total,
                r.removed.distractor + r.removed.redundant + r.removed.budget,
                "{s:?} removal subtotals don't sum"
            );
            assert_eq!(
                r.n_input_chunks,
                r.n_selected + r.removed.total,
                "{s:?} chunks unaccounted for"
            );
        }
    }

    #[test]
    fn invariant_reasoning_preserving_keeps_linked_second_hop_aggressive_filter_drops() {
        // The core guarantee: at a threshold where DistractorFiltered drops
        // the linked second hop, ReasoningPreserving must keep it.
        let q = Query::new("what nationality was the safety lamp inventor");
        let chunks = bridge_chunks();
        let aggressive = ContextConfig {
            token_budget: 1000,
            distractor_min_grounding: 0.25,
            link_min_jaccard: 0.05,
            redundancy_max_cosine: 1.0,
            strategy: ContextStrategy::DistractorFiltered,
        };
        let filtered = build_context(&q, &chunks, &aggressive);
        let preserving = build_context(
            &q,
            &chunks,
            &ContextConfig { strategy: ContextStrategy::ReasoningPreserving, ..aggressive },
        );
        assert!(!filtered.chunks.iter().any(|c| c.id.as_str() == "hop2"));
        assert!(preserving.chunks.iter().any(|c| c.id.as_str() == "hop2"));
        // …and the rescue is recorded in telemetry.
        assert!(preserving.report.second_hop_rescue_count >= 1);
        assert_eq!(filtered.report.second_hop_rescue_count, 0);
    }

    #[test]
    fn filter_context_does_not_truncate_to_budget() {
        // filter_context removes junk but never drops for budget.
        let q = Query::new("rust safety");
        let chunks = vec![
            rr("a", "rust safety ownership", None),
            rr("b", "rust safety borrow checker", None),
            rr("c", "cooking bread recipe flour", None), // junk
        ];
        let ctx = filter_context(
            &q,
            &chunks,
            &ContextConfig {
                token_budget: 1, // would truncate under build_context
                strategy: ContextStrategy::DistractorFiltered,
                distractor_min_grounding: 0.3,
                ..Default::default()
            },
        );
        assert_eq!(ctx.report.removed.budget, 0, "filter_context must not drop for budget");
        assert!(ctx.chunks.iter().all(|c| c.id.as_str() != "c"), "junk still filtered");
        assert_eq!(ctx.chunks.len(), 2);
    }

    #[test]
    fn public_grounding_and_link_primitives() {
        // Query-relevant chunk grounds high; off-topic grounds ~0.
        let q = "what nationality was the safety lamp inventor";
        assert!(grounding_score(q, "the safety lamp inventor was famous") > 0.3);
        assert_eq!(grounding_score(q, "photosynthesis converts sunlight"), 0.0);
        // Stemming makes morphological variants match (invented↔inventor↔invent).
        assert!(grounding_score("who invented it", "the invention of the lamp") > 0.0);
        // Linkage: chunks sharing the bridge entity link; unrelated ones don't.
        let hop1 = "the safety lamp was invented by Humphry Davy";
        let hop2 = "Humphry Davy was a British chemist";
        let junk = "photosynthesis in green plants";
        assert!(link_strength(hop1, hop2) > link_strength(hop1, junk));
    }

    #[test]
    fn rescued_second_hop_is_not_counted_as_a_distractor() {
        // The metric-correctness contract: a deliberately-rescued second hop
        // is reasoning evidence, not junk — economics must not count it.
        let q = Query::new("what nationality was the safety lamp inventor");
        let cfg = ContextConfig {
            token_budget: 1000,
            strategy: ContextStrategy::ReasoningPreserving,
            distractor_min_grounding: 0.25,
            link_min_jaccard: 0.05,
            redundancy_max_cosine: 1.0,
        };
        let rp = build_context(&q, &bridge_chunks(), &cfg);
        // hop2 is below the bar but rescued → kept, and NOT a distractor.
        assert!(rp.chunks.iter().any(|c| c.id.as_str() == "hop2"));
        assert!(rp.report.second_hop_rescue_count >= 1);
        assert_eq!(
            rp.report.economics.distractor_ratio, 0.0,
            "rescued hop must not inflate distractor_ratio"
        );
        assert_eq!(rp.report.economics.estimated_waste_tokens, 0);

        // RawTopK keeps the same below-bar chunk as *unrescued* junk → it
        // SHOULD count as a distractor.
        let raw = build_context(
            &q,
            &bridge_chunks(),
            &ContextConfig { strategy: ContextStrategy::RawTopK, ..cfg },
        );
        assert!(raw.report.economics.distractor_ratio > 0.0);
        assert_eq!(raw.report.second_hop_rescue_count, 0);
    }

    #[test]
    fn render_shows_token_delta_and_rescue() {
        let q = Query::new("what nationality was the safety lamp inventor");
        let cfg = ContextConfig {
            token_budget: 1000,
            strategy: ContextStrategy::ReasoningPreserving,
            distractor_min_grounding: 0.25,
            link_min_jaccard: 0.05,
            redundancy_max_cosine: 1.0,
        };
        let before = analyze_context(&q, &bridge_chunks(), &cfg);
        let after = build_context(&q, &bridge_chunks(), &cfg);
        let s = after.report.render(Some(&before));
        assert!(s.contains("Context Optimization Report"));
        assert!(s.contains("ReasoningPreserving"));
        assert!(s.contains("Reasoning rescues:"));
        // before has 3 chunks, after drops the junk → token delta is negative.
        assert!(s.contains('→'));
        assert!(after.report.render(None).contains("Tokens:"));
    }

    #[test]
    fn analyze_context_is_non_destructive_and_flags_candidates() {
        let q = Query::new("what nationality was the safety lamp inventor");
        let report = analyze_context(
            &q,
            &bridge_chunks(),
            &ContextConfig {
                distractor_min_grounding: 0.25,
                link_min_jaccard: 0.05,
                ..Default::default()
            },
        );
        // Nothing removed; all input present.
        assert_eq!(report.removed.total, 0);
        assert_eq!(report.n_selected, report.n_input_chunks);
        // hop2 is a rescuable second-hop candidate.
        assert!(report.second_hop_rescue_count >= 1);
        assert!(report.input_distractor_ratio > 0.0);
    }
}
