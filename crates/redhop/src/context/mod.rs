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
//! use redhop::context::{build_context, ContextConfig};
//! # use redhop::core::{Query, RetrievalResult};
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
//! frontier in one function: *reasoning-preserving evidence allocation, not
//! relevance optimization.* The economics readout stays honest about
//! what was dropped regardless of strategy.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::HashSet;

use crate::core::{Chunk, Embedding, Query, RetrievalResult};
use serde::{Deserialize, Serialize};

pub mod eval;

/// Term-set Jaccard at/above which two chunks are treated as near-exact
/// lexical duplicates by `RedundancyPruned` when embeddings are absent (the
/// local/BM25 path). High by design — only near-identical chunks are dropped.
const NEAR_DUP_JACCARD: f32 = 0.9;

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
    /// Size-gated policy: **decide whether to prune at all** from the input
    /// size, because that — not the pruning algorithm — is what the
    /// measurements show matters.
    ///
    /// - **Small input** (total input tokens ≤ `auto_passthrough_max_tokens`):
    ///   pass the context through unpruned (behaves like
    ///   [`ContextStrategy::RawTopK`]). Under headroom, relevance-based
    ///   pruning is a wash-to-harmful — it can only remove information the
    ///   model would have tolerated, and on multi-hop it risks the
    ///   second-hop tax. So the right move is to not touch it.
    /// - **Large input** (above the gate): the context is big enough that
    ///   attention *dilutes* (lost-in-the-middle); stuffing it all in collapses
    ///   accuracy, and pruning to the budget *recovers* it. Behaves like
    ///   [`ContextStrategy::ReasoningPreserving`] (drop unlinked junk, keep
    ///   seeds + linked hops, fill the budget).
    ///
    /// The load-bearing decision is the **gate (input size)**, not the pruner:
    /// in the dilution regime any sensible pruner captures the recoverable gain
    /// (`ReasoningPreserving` ties naive density-truncation downstream). See
    /// `docs/findings/CONTEXT_DILUTION.md` (the size crossover, n=200, CIs) and
    /// `docs/findings/REASONING_PRESERVATION.md` (why pruning hurts under
    /// headroom). The gain is also model-dependent (large on dilution-sensitive
    /// models, ~0 on models that tolerate the bloat).
    ///
    /// The gate default (`auto_passthrough_max_tokens` ≈ 1500) is calibrated by
    /// a size sweep: on gpt-4o-mini, pruning recovers accuracy at every size
    /// from ~1.5k tokens up (monotonic, CI-significant), with no harmful regime
    /// above the gate. Tune it up to be more conservative (prune less).
    Auto,
}

impl ContextStrategy {
    /// Resolve a meta-strategy ([`Auto`]) to the concrete strategy it acts as
    /// for an input of `total_input_tokens`. Concrete strategies return self.
    fn resolve(self, total_input_tokens: usize, cfg: &ContextConfig) -> ContextStrategy {
        match self {
            ContextStrategy::Auto => {
                if total_input_tokens <= cfg.auto_passthrough_max_tokens {
                    ContextStrategy::RawTopK
                } else {
                    ContextStrategy::ReasoningPreserving
                }
            }
            concrete => concrete,
        }
    }
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
    /// Input-size gate for [`ContextStrategy::Auto`]: at or below this many
    /// total input tokens, Auto passes the context through unpruned; above it,
    /// Auto prunes (the dilution regime). Provisional default pending the size
    /// sweep — see `docs/findings/CONTEXT_DILUTION.md`.
    pub auto_passthrough_max_tokens: usize,
    /// Grounding ceiling for the `low_confidence_retrieval` signal: if every
    /// selected chunk is at or below this score the retrieval is flagged as
    /// low-confidence in the [`ContextReport`]. Defaults to the same value as
    /// `distractor_min_grounding` (the existing "is this a distractor" bar) —
    /// so the signal fires exactly when the assembled context is all
    /// distractor-level relevance.
    pub low_confidence_max_grounding: f32,
    /// Lexical analyzer driving term extraction in the grounding scorer.
    /// Defaults to [`crate::analyzer::default_english`] — preserves the 0.1.4
    /// behavior. Should match the analyzer used by the BM25 retriever
    /// (`Document::with_analyzer` sets both in lockstep) so the two layers
    /// agree on what counts as "the same term".
    pub analyzer: std::sync::Arc<dyn crate::analyzer::Analyzer>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            // 8192-token assembled-context budget by default. Most production
            // LLMs have ≥32k usable context, so anything smaller leaves
            // capacity on the table for the average user. Set explicitly via
            // `ContextConfig { token_budget: ..., ..Default::default() }` or
            // `Document::context(..., budget=...)` if you need a different
            // ceiling. Python bindings have advertised this default since the
            // wheel shipped; this aligns Rust/Node with that contract.
            token_budget: 8192,
            // Safe-by-default: ReasoningPreserving keeps relevant evidence,
            // removes only unlinked junk, and never aggressively prunes by
            // relevance (which the second-hop-tax findings show is harmful
            // on multi-hop). See docs/findings/REASONING_PRESERVATION.md.
            strategy: ContextStrategy::ReasoningPreserving,
            // A low absolute bar: only near-zero-overlap junk is below it.
            distractor_min_grounding: 0.10,
            redundancy_max_cosine: 0.92,
            link_min_jaccard: 0.12,
            // Gate calibrated by the size sweep (CONTEXT_DILUTION.md): pruning
            // recovers accuracy at every measured size from ~1.5k tokens up
            // (monotonic, all CI-significant on gpt-4o-mini), with no harmful
            // regime above it. The crossover sits below ~1.5k (between the
            // ~0.5k/8-distractor regime where pruning is neutral and ~1.5k/50
            // where it already helps). 1500 is the conservative low edge of the
            // measured-benefit range: prune where we have evidence it helps,
            // pass through below it where evidence is absent.
            auto_passthrough_max_tokens: 1_500,
            // Same bar as `distractor_min_grounding`: if every selected chunk
            // is at-or-below distractor relevance, the retrieval itself was
            // weak (issue #1) and the caller should know programmatically.
            low_confidence_max_grounding: 0.10,
            // English Snowball Porter2 by default. Process-wide cached Arc so
            // `Default::default()` is cheap.
            analyzer: crate::analyzer::default_english(),
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

/// The runtime's decision for one assembly, made interpretable. For the
/// [`ContextStrategy::Auto`] policy this is the headline: did RedHop
/// deliberately leave the context intact, or did it intervene?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoDecision {
    /// A concrete strategy was requested — no policy decision was made.
    NotAuto,
    /// Auto deliberately left the context intact (small/undiluted input;
    /// pruning is measured to be wash-to-harmful under headroom).
    Passthrough,
    /// Auto chose to intervene (large/diluted input; pruning recovers signal
    /// density). The context was pruned to budget.
    Prune,
}

/// Full observability trace for one context assembly — the telemetry every
/// strategy emits, and the **explanation layer** for the runtime's decision.
/// Superset of [`ContextEconomics`]; serializable for benchmark/JSON output
/// and deployment dashboards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextReport {
    /// Strategy that actually produced this context (Auto is resolved to a
    /// concrete strategy here).
    pub strategy: ContextStrategy,
    /// Strategy the caller requested — may be [`ContextStrategy::Auto`]. The
    /// difference from `strategy` is what makes an Auto decision visible.
    pub requested_strategy: ContextStrategy,
    /// Configured token budget.
    pub token_budget: usize,
    /// Total tokens in the *input* (retrieved) set, before assembly. The
    /// "before" side of the token economics, and what the Auto gate compares.
    pub input_tokens: usize,
    /// The [`ContextStrategy::Auto`] passthrough gate in effect (input at or
    /// below this many tokens passes through; above it prunes).
    pub auto_gate_tokens: usize,
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
    #[serde(default)]
    pub removed: RemovedBreakdown,
    /// Structural chunks added by context expansion (adjacent neighbors and
    /// section headings attached to selected seeds). These are justified by
    /// document structure, not query relevance, so they are not counted as
    /// distractors. `0` when expansion is off.
    pub n_expanded: usize,
    /// True when retrieval produced nothing above the
    /// `low_confidence_max_grounding` threshold — either the assembled
    /// context is empty, or every chunk in it sits at or below distractor
    /// relevance. Lets callers detect "no good match found" programmatically
    /// instead of guessing from `n_selected == 0` (issue #1).
    #[serde(default)]
    pub low_confidence_retrieval: bool,
    /// The grounding ceiling that the `low_confidence_retrieval` signal
    /// applied. Copies `ContextConfig::low_confidence_max_grounding` so the
    /// report is self-describing.
    #[serde(default)]
    pub low_confidence_threshold: f32,
    /// Token/evidence economics of the assembled context.
    #[serde(default)]
    pub economics: ContextEconomics,
}

impl ContextReport {
    /// The runtime's decision for this assembly — the headline for the
    /// [`ContextStrategy::Auto`] policy. Lets callers branch on *what RedHop
    /// did and why* without parsing the rendered text.
    pub fn auto_decision(&self) -> AutoDecision {
        if self.requested_strategy != ContextStrategy::Auto {
            AutoDecision::NotAuto
        } else if self.strategy == ContextStrategy::RawTopK {
            AutoDecision::Passthrough
        } else {
            AutoDecision::Prune
        }
    }

    /// Token change from input to assembled context, as a signed percentage
    /// (negative = fewer tokens, the usual good case).
    fn token_savings_pct(&self) -> f32 {
        if self.input_tokens > 0 {
            100.0 * (self.total_tokens as f32 - self.input_tokens as f32) / self.input_tokens as f32
        } else {
            0.0
        }
    }

    /// Render the **RedHop Decision Report** — an interpretable explanation of
    /// what the runtime did and why, not a metric dump. Three layers: a
    /// human-readable decision, the token/evidence economics, and technical
    /// diagnostics for deep debugging.
    ///
    /// `before` is accepted for backward compatibility; the report is now
    /// self-contained (it carries `input_tokens`), so `None` renders the full
    /// view. When supplied, `before` is used only to show the density delta.
    pub fn render(&self, before: Option<&ContextReport>) -> String {
        let mut s = String::new();
        s.push_str("RedHop Decision Report\n");
        s.push_str("══════════════════════\n\n");

        // ── Layer 1: the decision, in plain language ───────────────────────
        self.render_decision(&mut s);

        // ── Layer 2: economics ─────────────────────────────────────────────
        s.push_str("\nEconomics\n─────────\n");
        s.push_str(&format!("  Retrieved tokens:   {}\n", self.input_tokens));
        s.push_str(&format!(
            "  Final tokens:       {}  ({:+.0}%)\n",
            self.total_tokens,
            self.token_savings_pct()
        ));
        s.push_str(&format!(
            "  Token budget:       {} ({:.0}% used)\n",
            self.token_budget,
            self.token_utilization * 100.0
        ));
        match before {
            Some(b) => s.push_str(&format!(
                "  Evidence density:   {:.2} → {:.2}\n",
                b.economics.evidence_density, self.economics.evidence_density
            )),
            None => s.push_str(&format!(
                "  Evidence density:   {:.2}\n",
                self.economics.evidence_density
            )),
        }
        s.push_str(&format!(
            "  Retained evidence:  {:.0}%\n",
            self.retained_evidence_ratio * 100.0
        ));

        // ── Layer 3: technical diagnostics ─────────────────────────────────
        s.push_str("\nDiagnostics\n───────────\n");
        s.push_str(&format!(
            "  Chunks:             {} → {}\n",
            self.n_input_chunks, self.n_selected
        ));
        s.push_str(&format!(
            "  Input distractors:  {:.0}% of retrieved chunks\n",
            self.input_distractor_ratio * 100.0
        ));
        s.push_str(&format!(
            "  Distractors pruned: {}\n",
            self.removed.distractor
        ));
        if self.removed.redundant > 0 {
            s.push_str(&format!(
                "  Duplicates pruned:  {}\n",
                self.removed.redundant
            ));
        }
        if self.removed.budget > 0 {
            s.push_str(&format!("  Budget-trimmed:     {}\n", self.removed.budget));
        }
        s.push_str(&format!(
            "  Second-hop rescues: {}\n",
            self.second_hop_rescue_count
        ));
        if self.n_expanded > 0 {
            s.push_str(&format!(
                "  Structural expand:  +{} (neighbors / headings, for continuity)\n",
                self.n_expanded
            ));
        }
        s.push_str(&format!(
            "  Estimated waste:    {} tokens on distractors\n",
            self.economics.estimated_waste_tokens
        ));

        // Warnings — things worth a second look.
        let mut warnings: Vec<String> = Vec::new();
        if self.removed.budget > 0 {
            warnings.push(format!(
                "{} chunk(s) dropped for token budget — consider raising it",
                self.removed.budget
            ));
        }
        // Note leftover distractors only when it wasn't a *deliberate* Auto
        // passthrough — there, keeping them is the explained, intended choice,
        // not something to warn about.
        if self.economics.distractor_ratio > 0.05
            && self.removed.distractor == 0
            && self.auto_decision() != AutoDecision::Passthrough
        {
            warnings.push(format!(
                "context still contains distractors (ratio {:.2}); strategy did not filter",
                self.economics.distractor_ratio
            ));
        }
        if self.low_confidence_retrieval {
            warnings.push(format!(
                "low-confidence retrieval: no chunks above grounding {:.2} — \
                 the query may share little vocabulary with the corpus",
                self.low_confidence_threshold
            ));
        }
        if !warnings.is_empty() {
            s.push_str("\nWarnings\n────────\n");
            for w in warnings {
                s.push_str(&format!("  - {w}\n"));
            }
        }
        s
    }

    /// Layer 1: the decision narrative ("what / why / result").
    fn render_decision(&self, s: &mut String) {
        let why = |s: &mut String, lines: &[String]| {
            s.push_str("  Why:\n");
            for l in lines {
                s.push_str(&format!("    - {l}\n"));
            }
        };
        let result = |s: &mut String, lines: &[String]| {
            s.push_str("  Result:\n");
            for l in lines {
                s.push_str(&format!("    - {l}\n"));
            }
        };

        match self.auto_decision() {
            AutoDecision::Passthrough => {
                s.push_str("Decision: Auto → passthrough (left the context intact)\n\n");
                why(
                    s,
                    &[
                        format!(
                            "input is small: {} tokens ≤ {} gate",
                            self.input_tokens, self.auto_gate_tokens
                        ),
                        "under headroom, pruning is measured to be wash-to-harmful".into(),
                        "intervention predicted to add no signal density here".into(),
                    ],
                );
                let kept = if self.removed.budget > 0 {
                    format!(
                        "kept {}/{} chunks (budget cap only; no relevance pruning)",
                        self.n_selected, self.n_input_chunks
                    )
                } else {
                    format!(
                        "kept all {} retrieved chunks — full evidence preserved",
                        self.n_input_chunks
                    )
                };
                result(s, &[kept, "avoided unnecessary intervention".into()]);
            }
            AutoDecision::Prune => {
                s.push_str("Decision: Auto → pruning (intervened on a diluted context)\n\n");
                why(
                    s,
                    &[
                        format!(
                            "input is large: {} tokens > {} gate",
                            self.input_tokens, self.auto_gate_tokens
                        ),
                        "large/diluted contexts dilute attention; pruning recovers signal density"
                            .into(),
                    ],
                );
                let mut r = vec![
                    format!(
                        "{} → {} tokens ({:+.0}%), {} distractor chunk(s) removed",
                        self.input_tokens,
                        self.total_tokens,
                        self.token_savings_pct(),
                        self.removed.distractor
                    ),
                    format!(
                        "retained {:.0}% of query-relevant evidence",
                        self.retained_evidence_ratio * 100.0
                    ),
                ];
                if self.second_hop_rescue_count > 0 {
                    r.push(format!(
                        "preserved {} second-hop link(s) a plain filter would drop",
                        self.second_hop_rescue_count
                    ));
                }
                result(s, &r);
            }
            AutoDecision::NotAuto => {
                s.push_str(&format!(
                    "Decision: {:?} (explicitly requested — no Auto policy decision)\n",
                    self.strategy
                ));
            }
        }
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
    pub fn contains(&self, id: &crate::core::ChunkId) -> bool {
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
    let analyzer = cfg.analyzer.as_ref();
    let q_terms = terms(&query.text, analyzer);

    // Per-chunk grounding + density, computed once.
    let mut items = characterize(&q_terms, retrieved, cfg);

    let n_distractor_total = items.iter().filter(|i| i.is_distractor).count();
    let mut n_dropped_distractor = 0;
    let mut n_dropped_redundant = 0;

    // Resolve the meta-strategy (Auto) from input size before acting.
    let total_input_tokens: usize = items.iter().map(|i| i.tokens).sum();
    let strategy = cfg.strategy.resolve(total_input_tokens, cfg);

    // Ordering / filtering per (resolved) strategy.
    match strategy {
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
        // Auto is resolved to a concrete strategy above; never reaches here.
        ContextStrategy::Auto => unreachable!("Auto resolved before match"),
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
                b_seed.cmp(&a_seed).then(
                    b.grounding
                        .partial_cmp(&a.grounding)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
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
        if strategy == ContextStrategy::RedundancyPruned {
            let redundant = match &item.embedding {
                // Embeddings present: semantic redundancy via cosine.
                Some(e) => selected.iter().any(|s| {
                    s.embedding
                        .as_ref()
                        .map(|se| cosine(e, se) > cfg.redundancy_max_cosine)
                        .unwrap_or(false)
                }),
                // No embeddings (the local/BM25 path): fall back to a near-exact
                // *lexical* duplicate test (term-set Jaccard). This makes
                // RedundancyPruned actually dedup duplicated/messy corpora
                // without requiring an embedding model. Measured on CUAD `dup`:
                // see docs/findings/DOCUMENT_EVAL_CUAD.md.
                None => selected
                    .iter()
                    .any(|s| jaccard(&item.c_terms, &s.c_terms) >= NEAR_DUP_JACCARD),
            };
            if redundant {
                n_dropped_redundant += 1;
                continue;
            }
        }
        total += item.tokens;
        selected.push(item);
    }

    let economics = economics(&q_terms, &selected, n_distractor_total, cfg);
    let report = make_report(
        cfg,
        strategy,
        total_input_tokens,
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

/// Deterministic structural companions for context expansion. For each seed
/// chunk, the adjacent neighbors and section heading to attach **if** that seed
/// is selected. Derived from document structure (chunk order, headings) — not
/// query relevance — so the companions sidestep the distractor gate.
#[derive(Debug, Clone, Default)]
pub struct ExpansionPlan {
    /// Seed chunk id → its ordered companion chunks (neighbors + heading).
    pub companions: std::collections::HashMap<String, Vec<Chunk>>,
    /// Document position of any chunk id (seed or companion), used to emit the
    /// final context in reading order.
    pub position: std::collections::HashMap<String, usize>,
}

/// [`build_context`] plus deterministic structural expansion: run the normal
/// relevance-and-budget selection, then attach each selected seed's adjacent
/// neighbors and section heading (from `plan`) — fit to the remaining budget,
/// deduplicated, and emitted in document order so each seed reads as a
/// contiguous window. Companions are justified by structure, so they are exempt
/// from the distractor filter but still bounded by `token_budget`.
pub fn build_context_expanded(
    query: &Query,
    retrieved: &[RetrievalResult],
    cfg: &ContextConfig,
    plan: &ExpansionPlan,
) -> BuiltContext {
    let mut built = build_context(query, retrieved, cfg);

    let mut present: HashSet<String> = built
        .chunks
        .iter()
        .map(|c| c.id.as_str().to_string())
        .collect();
    let mut used = built.report.total_tokens;
    let mut added: Vec<Chunk> = Vec::new();

    // Attach companions for each selected seed (in selection order). Neighbors
    // are pre-ordered by the planner; we keep them while the budget allows.
    for seed in &built.chunks {
        let Some(comps) = plan.companions.get(seed.id.as_str()) else {
            continue;
        };
        for c in comps {
            let id = c.id.as_str().to_string();
            if present.contains(&id) {
                continue;
            }
            let t = c.token_count.value().max(1);
            if used + t > cfg.token_budget {
                continue;
            }
            used += t;
            present.insert(id);
            added.push(c.clone());
        }
    }

    if added.is_empty() {
        return built;
    }

    let n_expanded = added.len();
    let mut all = built.chunks;
    all.append(&mut added);
    // Reading order: emit seeds and their companions by document position, so a
    // seed and its neighbors form one contiguous window.
    all.sort_by_key(|c| {
        plan.position
            .get(c.id.as_str())
            .copied()
            .unwrap_or(usize::MAX)
    });

    built.report.total_tokens = used;
    built.report.token_utilization = used as f32 / cfg.token_budget.max(1) as f32;
    built.report.n_selected = all.len();
    built.report.n_expanded = n_expanded;
    built.chunks = all;
    built
}

/// Assemble the [`ContextReport`] from the selection result.
#[allow(clippy::too_many_arguments)]
fn make_report(
    cfg: &ContextConfig,
    strategy: ContextStrategy,
    input_tokens: usize,
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
    // Low-confidence retrieval fires when there's nothing above the bar in
    // the assembled context — either selection was empty, or every selected
    // chunk sits at distractor relevance. Issue #1: a programmatic signal so
    // callers don't have to infer "no good match found" from `n_selected==0`.
    let low_confidence_retrieval = selected.is_empty()
        || selected
            .iter()
            .all(|i| i.grounding <= cfg.low_confidence_max_grounding);
    ContextReport {
        strategy,
        requested_strategy: cfg.strategy,
        token_budget: cfg.token_budget,
        input_tokens,
        auto_gate_tokens: cfg.auto_passthrough_max_tokens,
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
        n_expanded: 0,
        low_confidence_retrieval,
        low_confidence_threshold: cfg.low_confidence_max_grounding,
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
    let q_terms = terms(&query.text, cfg.analyzer.as_ref());
    let items = characterize(&q_terms, retrieved, cfg);
    let n_input = items.len();
    let n_input_distractor = items.iter().filter(|i| i.is_distractor).count();
    let total_tokens: usize = items.iter().map(|i| i.tokens).sum();
    // Second-hop candidates: below-bar chunks linked to a seed (what
    // ReasoningPreserving *would* rescue).
    let seed_terms: Vec<&HashSet<String>> = items
        .iter()
        .filter(|i| !i.is_distractor)
        .map(|i| &i.c_terms)
        .collect();
    let candidates = items
        .iter()
        .filter(|i| i.is_distractor)
        .filter(|i| {
            seed_terms
                .iter()
                .any(|s| jaccard(&i.c_terms, s) >= cfg.link_min_jaccard)
        })
        .count();
    let economics = economics(&q_terms, &items, n_input_distractor, cfg);
    // For analyze() the input set *is* the selected set (no filtering yet),
    // so the same rule applies: low-confidence when nothing is above the bar.
    let low_confidence_retrieval = items.is_empty()
        || items
            .iter()
            .all(|i| i.grounding <= cfg.low_confidence_max_grounding);
    ContextReport {
        // For Auto, report the action the gate would take on this input
        // (passthrough vs prune) — the diagnostic answer to "should I prune?".
        strategy: cfg.strategy.resolve(total_tokens, cfg),
        requested_strategy: cfg.strategy,
        token_budget: cfg.token_budget,
        input_tokens: total_tokens,
        auto_gate_tokens: cfg.auto_passthrough_max_tokens,
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
        n_expanded: 0,
        low_confidence_retrieval,
        low_confidence_threshold: cfg.low_confidence_max_grounding,
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
    let q_terms = terms(&query.text, cfg.analyzer.as_ref());
    let items = characterize(&q_terms, chunks, cfg);
    let n_distractor = items.iter().filter(|i| i.is_distractor).count();
    economics(&q_terms, &items, n_distractor, cfg)
}

/// Query grounding of a chunk's text: the relevance signal the strategies use —
/// stopword-removed, Snowball-stemmed query-term overlap, in `[0, 1]`. Exposed
/// for observability ("how relevant is this chunk to the query?") and so
/// external/eval code can reuse the library's exact notion of relevance instead
/// of reimplementing (and drifting from) it.
///
/// Uses the default English analyzer ([`crate::analyzer::default_english`]).
/// For non-English content, use `Document::context(...).report` which carries
/// the configured analyzer end-to-end.
pub fn grounding_score(query: &str, text: &str) -> f32 {
    let a = crate::analyzer::default_english();
    grounding(&terms(query, a.as_ref()), &terms(text, a.as_ref()))
}

/// Linkage strength between two chunks' text: term-set Jaccard over the same
/// normalized terms — the chunk↔chunk bridge signal `ReasoningPreserving` uses
/// to decide whether a low-relevance chunk is a rescuable second hop. In `[0, 1]`.
///
/// Uses the default English analyzer. See [`grounding_score`] for the
/// non-English path.
pub fn link_strength(a: &str, b: &str) -> f32 {
    let an = crate::analyzer::default_english();
    jaccard(&terms(a, an.as_ref()), &terms(b, an.as_ref()))
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
    let analyzer = cfg.analyzer.as_ref();
    retrieved
        .iter()
        .map(|r| {
            let c_terms = terms(&r.chunk.text, analyzer);
            let grounding = grounding(q_terms, &c_terms);
            let tok = r.chunk.token_count.value().max(1);
            // Count duplicates too — `c_terms` is a HashSet (no dups) but
            // `density` wants term-occurrence frequency over the chunk text.
            let relevant = analyzer
                .tokens(&r.chunk.text)
                .into_iter()
                .filter(|t| q_terms.contains(t))
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
    let analyzer = cfg.analyzer.as_ref();
    let relevant_tokens: usize = selected
        .iter()
        .map(|i| {
            // Count duplicates too — `density` wants term-occurrence
            // frequency over the chunk text, not unique-term coverage.
            analyzer
                .tokens(&i.chunk.text)
                .into_iter()
                .filter(|t| q_terms.contains(t))
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
    let embs: Vec<&Embedding> = selected
        .iter()
        .filter_map(|i| i.embedding.as_ref())
        .collect();
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
//
// Crate-public so `crate::retrieval::bm25` can wire the same list into the
// Tantivy analyzer — without that, BM25 keeps "the"/"is"/"what" while the
// grounding scorer drops them, and the two layers disagree on what a query
// even consists of.
pub(crate) const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "as", "by", "with", "from", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their", "which", "who", "whom",
    "what", "when", "where", "how", "why", "into", "than", "then", "there", "here", "out", "up",
    "down", "over", "under", "do", "does", "did", "has", "have", "had", "not", "no", "can", "will",
    "would", "should", "could", "may", "might", "about", "between", "during", "such", "also",
];

/// Extract the analyzer's terms from a piece of text — used by the grounding
/// scorer to compute overlap-based relevance. Delegating to
/// [`crate::analyzer::Analyzer::tokens`] guarantees parity with whatever
/// pipeline BM25 retrieval uses (Snowball stemming, stopword filtering,
/// ASCII folding, camelCase splitting, …) so the retriever and the scorer
/// never disagree on what "the same term" is.
fn terms(text: &str, analyzer: &dyn crate::analyzer::Analyzer) -> HashSet<String> {
    analyzer.tokens(text).into_iter().collect()
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
    use crate::core::{ChunkId, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

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
    fn expansion_attaches_neighbors_in_document_order() {
        // The hit is chunk 2; its neighbors (1 and 3) carry no query terms, so a
        // relevance filter would drop them — expansion keeps them by structure.
        let hit = rr("2", "the governing law is delaware", None);
        let mut plan = ExpansionPlan::default();
        plan.position.insert("1".into(), 1);
        plan.position.insert("2".into(), 2);
        plan.position.insert("3".into(), 3);
        plan.companions.insert(
            "2".into(),
            vec![
                Chunk::new(
                    ChunkId::new("1"),
                    "preceding clause text",
                    "doc",
                    TokenCount(3),
                ),
                Chunk::new(
                    ChunkId::new("3"),
                    "following clause text",
                    "doc",
                    TokenCount(3),
                ),
            ],
        );
        let cfg = ContextConfig {
            token_budget: 100,
            strategy: ContextStrategy::ReasoningPreserving,
            ..Default::default()
        };
        let ctx = build_context_expanded(&Query::new("governing law"), &[hit], &cfg, &plan);
        let ids: Vec<&str> = ctx.chunks.iter().map(|c| c.id.as_str()).collect();
        // Seed + both neighbors, emitted in document order 1,2,3 (a window).
        assert_eq!(ids, vec!["1", "2", "3"]);
        assert_eq!(ctx.report.n_expanded, 2);
    }

    #[test]
    fn expansion_respects_token_budget() {
        let hit = rr("2", "alpha beta gamma", None); // 3 tokens
        let mut plan = ExpansionPlan::default();
        plan.position.insert("2".into(), 2);
        plan.position.insert("3".into(), 3);
        plan.companions.insert(
            "2".into(),
            vec![Chunk::new(
                ChunkId::new("3"),
                "x ".repeat(50).trim(),
                "doc",
                TokenCount(50),
            )],
        );
        // Budget only fits the seed; the 50-token neighbor must be skipped.
        let cfg = ContextConfig {
            token_budget: 10,
            strategy: ContextStrategy::RawTopK,
            ..Default::default()
        };
        let ctx = build_context_expanded(&Query::new("alpha"), &[hit], &cfg, &plan);
        assert_eq!(ctx.report.n_expanded, 0);
        assert_eq!(ctx.chunks.len(), 1);
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
            rr(
                "dilute",
                "rust safety amid lots of extra filler words here now",
                None,
            ),
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
            rr(
                "hop2",
                "humphry davy born penzance cornwall england chemist",
                None,
            ),
            // true junk: unrelated to query AND to hop1
            rr(
                "junk",
                "photosynthesis converts sunlight chemical energy green plants",
                None,
            ),
        ]
    }

    fn bridge_cfg(s: ContextStrategy) -> ContextConfig {
        ContextConfig {
            token_budget: 1000,
            strategy: s,
            distractor_min_grounding: 0.25,
            link_min_jaccard: 0.05,
            auto_passthrough_max_tokens: 8_000,
            redundancy_max_cosine: 1.0,
            low_confidence_max_grounding: 0.10,
            analyzer: crate::analyzer::default_english(),
        }
    }

    #[test]
    fn reasoning_preserving_rescues_linked_second_hop_drops_true_junk() {
        let q = Query::new("what nationality was the safety lamp inventor");
        let ctx = build_context(
            &q,
            &bridge_chunks(),
            &bridge_cfg(ContextStrategy::ReasoningPreserving),
        );
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
        let filtered = build_context(
            &q,
            &chunks,
            &bridge_cfg(ContextStrategy::DistractorFiltered),
        );
        let preserving = build_context(
            &q,
            &chunks,
            &bridge_cfg(ContextStrategy::ReasoningPreserving),
        );
        assert!(
            !filtered.chunks.iter().any(|c| c.id.as_str() == "hop2"),
            "distractor filter should drop the low-relevance second hop"
        );
        assert!(
            preserving.chunks.iter().any(|c| c.id.as_str() == "hop2"),
            "reasoning-preserving should rescue it"
        );
    }

    #[test]
    fn economics_reports_density_and_waste() {
        let q = Query::new("rust memory safety");
        let chunks = vec![
            rr("a", "rust memory safety", None),         // all relevant
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
            rr(
                "a",
                "rust memory safety guarantees ownership borrow checker",
                None,
            ),
            rr("b", "rust prevents data races at compile time safety", None),
            rr(
                "c",
                "cooking bread recipe flour yeast water salt oven",
                None,
            ),
            rr(
                "d",
                "memory safety without garbage collection in rust",
                None,
            ),
        ];
        for s in all_strategies() {
            for budget in [1usize, 3, 5, 8, 50] {
                let ctx = build_context(
                    &q,
                    &chunks,
                    &ContextConfig {
                        token_budget: budget,
                        strategy: s,
                        ..Default::default()
                    },
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
            auto_passthrough_max_tokens: 8_000,
            redundancy_max_cosine: 1.0,
            low_confidence_max_grounding: 0.10,
            analyzer: crate::analyzer::default_english(),
            strategy: ContextStrategy::DistractorFiltered,
        };
        let filtered = build_context(&q, &chunks, &aggressive);
        let preserving = build_context(
            &q,
            &chunks,
            &ContextConfig {
                strategy: ContextStrategy::ReasoningPreserving,
                ..aggressive
            },
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
        assert_eq!(
            ctx.report.removed.budget, 0,
            "filter_context must not drop for budget"
        );
        assert!(
            ctx.chunks.iter().all(|c| c.id.as_str() != "c"),
            "junk still filtered"
        );
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
            auto_passthrough_max_tokens: 8_000,
            redundancy_max_cosine: 1.0,
            low_confidence_max_grounding: 0.10,
            analyzer: crate::analyzer::default_english(),
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
            &ContextConfig {
                strategy: ContextStrategy::RawTopK,
                ..cfg
            },
        );
        assert!(raw.report.economics.distractor_ratio > 0.0);
        assert_eq!(raw.report.second_hop_rescue_count, 0);
    }

    #[test]
    fn auto_gates_on_input_size() {
        // bridge_chunks total ≈ 22 tokens. Auto must pass through under a high
        // gate (don't prune small contexts) and prune under a low gate (the
        // dilution regime). The deciding variable is input size, not the pruner.
        let q = Query::new("what nationality was the safety lamp inventor");

        // High gate → passthrough: behaves as RawTopK, keeps the junk, no rescue.
        let big_gate = ContextConfig {
            auto_passthrough_max_tokens: 8_000,
            ..bridge_cfg(ContextStrategy::Auto)
        };
        let pass = build_context(&q, &bridge_chunks(), &big_gate);
        assert_eq!(pass.report.strategy, ContextStrategy::RawTopK);
        assert!(pass.chunks.iter().any(|c| c.id.as_str() == "junk"));
        assert_eq!(pass.report.second_hop_rescue_count, 0);

        // Low gate → prune: behaves as ReasoningPreserving, drops junk, rescues
        // the linked second hop.
        let small_gate = ContextConfig {
            auto_passthrough_max_tokens: 10,
            ..bridge_cfg(ContextStrategy::Auto)
        };
        let prune = build_context(&q, &bridge_chunks(), &small_gate);
        assert_eq!(prune.report.strategy, ContextStrategy::ReasoningPreserving);
        assert!(!prune.chunks.iter().any(|c| c.id.as_str() == "junk"));
        assert!(prune.chunks.iter().any(|c| c.id.as_str() == "hop2"));
        assert!(prune.report.second_hop_rescue_count >= 1);

        // analyze_context reports the action the gate *would* take (diagnostic).
        assert_eq!(
            analyze_context(&q, &bridge_chunks(), &big_gate).strategy,
            ContextStrategy::RawTopK
        );
        assert_eq!(
            analyze_context(&q, &bridge_chunks(), &small_gate).strategy,
            ContextStrategy::ReasoningPreserving
        );
    }

    #[test]
    fn redundancy_pruned_dedups_lexically_without_embeddings() {
        // The local/BM25 path has no embeddings; RedundancyPruned must still
        // drop near-exact duplicate chunks (the CUAD `dup` regime fix).
        let q = Query::new("rust memory safety");
        let dup = "rust guarantees memory safety without a garbage collector";
        let chunks = vec![
            rr("a", dup, None),
            rr("b", dup, None), // exact lexical duplicate, no embeddings
            rr(
                "c",
                "python is a dynamically typed scripting language",
                None,
            ),
        ];
        let cfg = ContextConfig {
            strategy: ContextStrategy::RedundancyPruned,
            token_budget: 1000,
            ..Default::default()
        };
        let out = build_context(&q, &chunks, &cfg);
        assert_eq!(
            out.report.removed.redundant, 1,
            "the duplicate should be dropped"
        );
        assert_eq!(out.chunks.iter().filter(|c| c.text == dup).count(), 1);
    }

    #[test]
    fn render_shows_token_delta_and_rescue() {
        let q = Query::new("what nationality was the safety lamp inventor");
        let cfg = ContextConfig {
            token_budget: 1000,
            strategy: ContextStrategy::ReasoningPreserving,
            distractor_min_grounding: 0.25,
            link_min_jaccard: 0.05,
            auto_passthrough_max_tokens: 8_000,
            redundancy_max_cosine: 1.0,
            low_confidence_max_grounding: 0.10,
            analyzer: crate::analyzer::default_english(),
        };
        let before = analyze_context(&q, &bridge_chunks(), &cfg);
        let after = build_context(&q, &bridge_chunks(), &cfg);
        let s = after.report.render(Some(&before));
        assert!(s.contains("RedHop Decision Report"));
        // Explicit strategy → not an Auto policy decision.
        assert_eq!(after.report.auto_decision(), AutoDecision::NotAuto);
        assert!(s.contains("ReasoningPreserving"));
        assert!(s.contains("Second-hop rescues:"));
        assert!(s.contains("Economics") && s.contains("Diagnostics"));
        // chunks delta uses '→'
        assert!(s.contains('→'));
        let solo = after.report.render(None);
        assert!(solo.contains("Final tokens:") && solo.contains("Retrieved tokens:"));
    }

    #[test]
    fn auto_decision_is_explained_in_the_report() {
        let q = Query::new("what nationality was the safety lamp inventor");

        // High gate → Auto passes through; the report must SAY so and why.
        let pass_cfg = ContextConfig {
            auto_passthrough_max_tokens: 8_000,
            ..bridge_cfg(ContextStrategy::Auto)
        };
        let pass = build_context(&q, &bridge_chunks(), &pass_cfg);
        assert_eq!(pass.report.auto_decision(), AutoDecision::Passthrough);
        let r = pass.report.render(None);
        assert!(
            r.contains("Auto → passthrough"),
            "missing passthrough decision:\n{r}"
        );
        assert!(r.contains("left the context intact"));
        assert!(r.contains("Why:") && r.contains("Result:"));

        // Low gate → Auto prunes; the report must SAY so and why.
        let prune_cfg = ContextConfig {
            auto_passthrough_max_tokens: 10,
            ..bridge_cfg(ContextStrategy::Auto)
        };
        let prune = build_context(&q, &bridge_chunks(), &prune_cfg);
        assert_eq!(prune.report.auto_decision(), AutoDecision::Prune);
        let r = prune.report.render(None);
        assert!(
            r.contains("Auto → pruning"),
            "missing pruning decision:\n{r}"
        );
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
                auto_passthrough_max_tokens: 8_000,
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

    /// Issue #1 — observability: when retrieval finds nothing with meaningful
    /// query overlap, the report flags it explicitly rather than returning a
    /// silent (and weak) context.
    #[test]
    fn low_confidence_retrieval_fires_when_nothing_is_grounded() {
        let q = Query::new("refund window cancellation policy");
        let cfg = ContextConfig {
            // Off the auto-passthrough path so the strategies actually run.
            auto_passthrough_max_tokens: 0,
            ..Default::default()
        };

        // Off-topic chunks that share no terms with the query → all grounding
        // scores are 0, well below the 0.10 ceiling.
        let weak = vec![
            rr("1", "photosynthesis converts sunlight into glucose", None),
            rr("2", "tectonic plates drift over millions of years", None),
        ];
        let r = build_context(&q, &weak, &cfg).report;
        assert!(
            r.low_confidence_retrieval,
            "expected low_confidence_retrieval=true on off-topic corpus, got report: {r:?}"
        );
        assert_eq!(r.low_confidence_threshold, cfg.low_confidence_max_grounding);
        assert!(
            r.render(None).contains("low-confidence retrieval"),
            "render() should include the low-confidence warning line"
        );

        // analyze_context surfaces the same signal without selection.
        let a = analyze_context(&q, &weak, &cfg);
        assert!(
            a.low_confidence_retrieval,
            "analyze_context must flag the same case"
        );

        // On-topic chunks: grounding is high → signal stays off.
        let strong = vec![
            rr("3", "the refund window is thirty days from purchase", None),
            rr(
                "4",
                "cancellation policy: written notice within seven days",
                None,
            ),
        ];
        let r2 = build_context(&q, &strong, &cfg).report;
        assert!(
            !r2.low_confidence_retrieval,
            "expected low_confidence_retrieval=false on on-topic corpus, got report: {r2:?}"
        );
    }
}
