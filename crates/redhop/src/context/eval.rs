//! In-process evaluation of a `BuiltContext` against optional gold signals.
//!
//! The user-facing answer to "how good was that context assembly?", computed
//! from the same primitives the runtime already uses to make decisions —
//! grounding scores, the existing [`ContextReport`] fields — with **no LLM
//! judge in the loop**. The LLM-judge metrics that RAGAS/Trulens/Phoenix
//! provide (faithfulness, answer-relevance) are still better signals for the
//! generated answer; this module gives you a fast, cheap, deterministic
//! retrieval-and-assembly score from inside the runtime itself.
//!
//! The differentiator: a low [`EvalReport::overall`] and a `true`
//! [`ContextReport::low_confidence_retrieval`] correlate *by construction* —
//! both are computed from the same grounding signal. If the runtime says
//! "this is low-confidence" and the eval says "this scored 0.2", you are not
//! looking at two independent measurements; you're looking at one signal
//! refracted twice. That's a feature: there is no discrepancy to debug.
//!
//! ## Usage
//!
//! ```no_run
//! # use redhop::{Document, Query};
//! # use redhop::context::eval::{evaluate, EvalGold};
//! # fn demo() -> redhop::Result<()> {
//! let mut doc = Document::from_text("policy.md", "the refund window is thirty days")?;
//! let ctx = doc.context("refund window")?;
//!
//! // Self-evaluation only — no gold provided.
//! let score = evaluate(&Query::new("refund window"), &ctx, EvalGold::None);
//! println!("overall={:.2}  density={:.2}", score.overall, score.evidence_density);
//!
//! // Gold-relative evaluation when you have ground truth.
//! let score = evaluate(
//!     &Query::new("refund window"),
//!     &ctx,
//!     EvalGold::Both {
//!         gold_chunk_ids: &["chunk_42"],
//!         gold_answer: "thirty days",
//!     },
//! );
//! println!("recall={:?}  precision={:?}", score.context_recall, score.context_precision);
//! # Ok(()) }
//! ```

use crate::analyzer::default_english;
use crate::context::{grounding_score, BuiltContext};
use crate::core::Query;
use std::collections::HashSet;

/// Optional ground-truth signals the caller supplies to `evaluate`. Each
/// variant unlocks a different gold-relative metric on the returned
/// [`EvalReport`]; the self-evaluation metrics (grounding, evidence density,
/// second-hop rescues, …) are always populated regardless.
#[derive(Debug, Clone, Copy)]
pub enum EvalGold<'a> {
    /// No ground truth provided. `context_recall`, `context_precision`,
    /// and `answer_token_recall` will all be `None` in the report.
    None,
    /// IDs of chunks that should appear in the assembled context. Unlocks
    /// `context_recall` and `context_precision`.
    Chunks(&'a [&'a str]),
    /// Ground-truth answer text. Unlocks `answer_token_recall` (fraction of
    /// content terms in the gold answer that appear in the assembled
    /// context).
    Answer(&'a str),
    /// Both signals at once.
    Both {
        /// Same shape as `EvalGold::Chunks(...)`.
        gold_chunk_ids: &'a [&'a str],
        /// Same shape as `EvalGold::Answer(...)`.
        gold_answer: &'a str,
    },
}

/// In-process evaluation report for a single (query, BuiltContext) pair.
///
/// Self-evaluation fields are always populated; gold-relative fields are
/// `Some` only when [`EvalGold`] provides the relevant ground truth. The
/// composite `overall` score blends whichever fields are available.
#[derive(Debug, Clone)]
pub struct EvalReport {
    // ── gold-relative metrics (Some iff caller provided the relevant gold) ──
    /// `|selected ∩ gold| / |gold|`. Fraction of the gold chunks that
    /// survived the full pipeline (retrieval + assembly). `None` if the
    /// caller didn't pass `EvalGold::Chunks` / `EvalGold::Both`.
    pub context_recall: Option<f32>,
    /// `|selected ∩ gold| / |selected|`. Fraction of the selected chunks
    /// that were gold. `None` under the same conditions as
    /// `context_recall`.
    pub context_precision: Option<f32>,
    /// Fraction of content terms in the gold answer that appear in the
    /// assembled context. Uses the same Snowball-stemmed,
    /// stopword-filtered term extraction the runtime's grounding scorer
    /// uses, so a token reachable via stemming counts. `None` if no gold
    /// answer was provided.
    pub answer_token_recall: Option<f32>,

    // ── self-evaluation metrics (always populated) ──
    /// Mean grounding score of the selected chunks against the query.
    /// Always in `[0, 1]`. The "how relevant is what we selected, on
    /// average" number. Computed with the runtime's default-English
    /// analyzer — same primitive `ContextStrategy::DistractorFiltered`
    /// uses.
    pub mean_grounding: f32,
    /// `ContextEconomics::evidence_density` — fraction of context tokens
    /// that are query-relevant.
    pub evidence_density: f32,
    /// `ContextReport::retained_evidence_ratio` — fraction of input
    /// evidence that made it through assembly.
    pub retained_evidence_ratio: f32,
    /// `ContextReport::second_hop_rescue_count` — how many bridge
    /// passages the reasoning-preserving rescue saved from a relevance
    /// filter.
    pub second_hop_rescues: usize,
    /// `ContextReport::low_confidence_retrieval` — true when every
    /// selected chunk is at-or-below the grounding ceiling, i.e. the
    /// retrieval itself was weak.
    pub low_confidence: bool,
    /// `ContextEconomics::estimated_waste_tokens` — tokens spent on
    /// below-bar chunks.
    pub estimated_waste_tokens: usize,

    /// A single composite score in `[0, 1]` blending whichever fields
    /// above are available, with gold-relative metrics weighted heavier
    /// when present. Use this as the headline number; use the individual
    /// fields above to debug why it landed where it did.
    pub overall: f32,
}

/// Evaluate an assembled `BuiltContext` against optional ground truth.
///
/// See the module-level docs for the design rationale and a usage example.
/// All metrics are deterministic, in-process, and require no LLM call.
///
/// **Self-eval vs gold-conditional — what evaluate actually tells you.**
/// Without ground truth, `evaluate` populates *self-eval* fields
/// (`mean_grounding`, `evidence_density`, `second_hop_rescues`,
/// `low_confidence`, `estimated_waste_tokens`). These measure how
/// **focused** the assembled context is relative to the query —
/// whether the chunks share query vocabulary, whether the budget is
/// being spent on relevant tokens. They do **not** tell you whether
/// the **correct** answer-bearing chunk is in the context: a dense,
/// on-topic context can still be confidently wrong. To measure
/// correctness, pass `gold` (chunk ids or answer text), which unlocks
/// `context_recall` / `context_precision` (chunk-level) and
/// `answer_token_recall` (token-level). For A/B comparisons of
/// `Stripper` / `Vocabulary` chains, supplying `gold` is what makes
/// the comparison meaningful.
pub fn evaluate(query: &Query, ctx: &BuiltContext, gold: EvalGold<'_>) -> EvalReport {
    let analyzer = default_english();

    // ── Self-evaluation metrics ──
    let mean_grounding = if ctx.chunks.is_empty() {
        0.0
    } else {
        ctx.chunks
            .iter()
            .map(|c| grounding_score(query.text.as_str(), c.text.as_str()))
            .sum::<f32>()
            / ctx.chunks.len() as f32
    };
    let evidence_density = ctx.report.economics.evidence_density;
    let retained_evidence_ratio = ctx.report.retained_evidence_ratio;
    let second_hop_rescues = ctx.report.second_hop_rescue_count;
    let low_confidence = ctx.report.low_confidence_retrieval;
    let estimated_waste_tokens = ctx.report.economics.estimated_waste_tokens;

    // ── Gold-relative metrics ──
    let (gold_chunks, gold_answer) = match gold {
        EvalGold::None => (None, None),
        EvalGold::Chunks(ids) => (Some(ids), None),
        EvalGold::Answer(a) => (None, Some(a)),
        EvalGold::Both {
            gold_chunk_ids,
            gold_answer,
        } => (Some(gold_chunk_ids), Some(gold_answer)),
    };

    let (context_recall, context_precision) = match gold_chunks {
        None => (None, None),
        Some([]) => (Some(1.0), None),
        Some(gold) => {
            let gold_set: HashSet<&str> = gold.iter().copied().collect();
            let selected_ids: HashSet<&str> = ctx.chunks.iter().map(|c| c.id.as_str()).collect();
            let intersection = gold_set.intersection(&selected_ids).count();
            let recall = intersection as f32 / gold_set.len() as f32;
            let precision = if selected_ids.is_empty() {
                0.0
            } else {
                intersection as f32 / selected_ids.len() as f32
            };
            (Some(recall), Some(precision))
        }
    };

    let answer_token_recall = gold_answer.map(|answer| {
        // Use the runtime's term-extraction primitive (Snowball-stemmed,
        // stopword-filtered) so stemming-related matches count. Reuse
        // grounding_score with (gold_answer, assembled_text) — the same
        // signal the runtime uses to score query→passage, repurposed as
        // gold→context.
        if ctx.chunks.is_empty() {
            return 0.0;
        }
        let assembled: String = ctx
            .chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        // grounding_score is symmetric in the analyzer's term-set Jaccard
        // sense — using it here measures "what fraction of the gold-answer
        // terms appear in the assembled context."
        let _ = analyzer; // keep clippy quiet — analyzer is materialized to
                          // pin the same default that `grounding_score` uses.
        grounding_score(answer, &assembled)
    });

    // ── Composite overall score ──
    // Weighted blend with gold-relative metrics dominating when present,
    // self-eval metrics dominating when not. Capped to [0, 1].
    let (mut score, mut weight) = (0.0_f32, 0.0_f32);
    let mut add = |value: f32, w: f32| {
        score += value.clamp(0.0, 1.0) * w;
        weight += w;
    };
    if let Some(r) = context_recall {
        add(r, 3.0);
    }
    if let Some(p) = context_precision {
        add(p, 2.0);
    }
    if let Some(r) = answer_token_recall {
        add(r, 2.0);
    }
    add(mean_grounding, 1.0);
    add(evidence_density, 1.0);
    add(retained_evidence_ratio, 1.0);
    if low_confidence {
        // low_confidence is a flag, not a [0,1] score — treat it as a hard
        // 0.2 multiplier on the blended score so a low-confidence retrieval
        // can't score above ~0.2 regardless of how the gold-free metrics
        // shake out. This matches the runtime's own "the retrieval was
        // weak" verdict.
        score *= 0.2;
    }
    let overall = if weight > 0.0 {
        (score / weight).clamp(0.0, 1.0)
    } else {
        0.0
    };

    EvalReport {
        context_recall,
        context_precision,
        answer_token_recall,
        mean_grounding,
        evidence_density,
        retained_evidence_ratio,
        second_hop_rescues,
        low_confidence,
        estimated_waste_tokens,
        overall,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{build_context, ContextConfig, ContextStrategy};
    use crate::core::{
        Chunk, ChunkId, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown, TokenCount,
    };

    fn rr(id: &str, text: &str) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(
                ChunkId::new(id),
                text,
                "doc",
                TokenCount(text.split_whitespace().count()),
            ),
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    fn build(query: &str, chunks: &[RetrievalResult]) -> BuiltContext {
        let cfg = ContextConfig {
            token_budget: 1000,
            strategy: ContextStrategy::RawTopK,
            ..Default::default()
        };
        build_context(&Query::new(query), chunks, &cfg)
    }

    #[test]
    fn self_eval_works_without_any_gold() {
        let ctx = build(
            "refund window",
            &[rr("a", "the refund window is thirty days")],
        );
        let r = evaluate(&Query::new("refund window"), &ctx, EvalGold::None);
        assert!(r.context_recall.is_none());
        assert!(r.context_precision.is_none());
        assert!(r.answer_token_recall.is_none());
        // Self-eval populated.
        assert!(
            r.mean_grounding > 0.0,
            "grounding should be positive for a matching chunk"
        );
        assert!(r.overall > 0.0);
        assert!(r.overall <= 1.0);
    }

    #[test]
    fn perfect_recall_when_all_gold_in_selected() {
        let ctx = build(
            "refund window",
            &[
                rr("hit1", "the refund window is thirty days"),
                rr("hit2", "refund policy details and timing"),
            ],
        );
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            EvalGold::Chunks(&["hit1", "hit2"]),
        );
        assert_eq!(r.context_recall, Some(1.0));
        assert_eq!(r.context_precision, Some(1.0));
    }

    #[test]
    fn partial_recall_when_some_gold_missing_from_selection() {
        let ctx = build(
            "refund",
            &[
                rr("hit", "refund refund refund"),        // 1 gold present
                rr("noise", "totally unrelated cooking"), // 1 distractor
            ],
        );
        let r = evaluate(
            &Query::new("refund"),
            &ctx,
            EvalGold::Chunks(&["hit", "missing"]), // "missing" never indexed
        );
        assert_eq!(r.context_recall, Some(0.5)); // 1 of 2 gold chunks present
        assert_eq!(r.context_precision, Some(0.5)); // 1 of 2 selected chunks is gold
    }

    #[test]
    fn answer_token_recall_uses_stemming() {
        // Gold answer says "refunds within thirty days"; assembled context
        // contains "refund window is thirty days". Stemming should map
        // refunds → refund, so the term overlap is non-trivial.
        let ctx = build(
            "refund window",
            &[rr("a", "the refund window is thirty days from purchase")],
        );
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            EvalGold::Answer("refunds within thirty days"),
        );
        let recall = r.answer_token_recall.expect("answer recall populated");
        assert!(
            recall >= 0.5,
            "answer-token recall should be substantial when stemming matches; got {recall}"
        );
    }

    #[test]
    fn low_confidence_caps_overall_score() {
        // Pure-noise corpus + a query that shares no terms. Should land in
        // low-confidence territory and the overall score should be capped.
        let ctx = build(
            "quantum chromodynamics gluon coupling",
            &[
                rr("a", "the refund window is thirty days"),
                rr("b", "shipping policy and delivery times"),
            ],
        );
        let r = evaluate(
            &Query::new("quantum chromodynamics gluon coupling"),
            &ctx,
            EvalGold::None,
        );
        assert!(
            r.low_confidence,
            "off-topic query against a refund corpus should flag low_confidence"
        );
        assert!(
            r.overall <= 0.25,
            "low_confidence should keep overall ≤ 0.25 (capped at 0.2× the blended score); got {}",
            r.overall
        );
    }

    #[test]
    fn empty_gold_chunks_returns_perfect_recall() {
        // If the caller explicitly passes EvalGold::Chunks(&[]), that's "no
        // chunks need to be retrieved" — vacuously perfect. Not the same as
        // EvalGold::None (which means "no gold available; skip the metric").
        let ctx = build("query", &[rr("a", "some text")]);
        let r = evaluate(&Query::new("query"), &ctx, EvalGold::Chunks(&[]));
        assert_eq!(r.context_recall, Some(1.0));
        // precision is undefined when gold is empty — we report None here so
        // a caller doesn't accidentally treat 0 as a real precision number.
        assert!(r.context_precision.is_none());
    }

    #[test]
    fn answer_only_gold_leaves_chunk_metrics_none() {
        // EvalGold::Answer alone must populate `answer_token_recall` and
        // leave context_recall / context_precision as None — the caller
        // didn't give us chunk-level gold so we have nothing to compute
        // those from. Guards against a refactor that accidentally treats
        // "answer present" as "chunks present too".
        let ctx = build(
            "refund window",
            &[rr("a", "the refund window is thirty days")],
        );
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            EvalGold::Answer("thirty days"),
        );
        assert!(r.context_recall.is_none());
        assert!(r.context_precision.is_none());
        let atr = r
            .answer_token_recall
            .expect("answer_token_recall should be populated when Answer gold is given");
        assert!(
            atr > 0.0,
            "stemmed gold answer terms appear in context; got {atr}"
        );
    }

    #[test]
    fn both_gold_signals_populate_all_three_metrics() {
        // EvalGold::Both must populate ALL three gold-relative metrics.
        // Catches a refactor that splits the gold extraction wrong (e.g.,
        // reads `gold_chunk_ids` but forgets `gold_answer` from the same
        // variant).
        let ctx = build(
            "refund window",
            &[
                rr("hit", "the refund window is thirty days"),
                rr("noise", "shipping policy details"),
            ],
        );
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            EvalGold::Both {
                gold_chunk_ids: &["hit"],
                gold_answer: "thirty days",
            },
        );
        assert_eq!(r.context_recall, Some(1.0)); // "hit" is in the selection
        assert!(r.context_precision.is_some());
        let atr = r
            .answer_token_recall
            .expect("answer_token_recall should be populated under Both");
        assert!(atr > 0.0);
        // overall should reflect all three signals being available.
        assert!(r.overall > 0.0);
        assert!(r.overall <= 1.0);
    }

    #[test]
    fn precision_distinct_from_recall_with_asymmetric_sets() {
        // 3 selected, 2 gold, 1 hit → recall=0.5, precision≈0.33. The
        // earlier tests had |selected| == |gold| so both metrics coincided;
        // this one exercises the asymmetric case.
        let ctx = build(
            "policy",
            &[
                rr("hit", "policy section about refunds"),
                rr("noise_a", "totally unrelated cooking recipe"),
                rr("noise_b", "more cooking instructions"),
            ],
        );
        let r = evaluate(
            &Query::new("policy"),
            &ctx,
            EvalGold::Chunks(&["hit", "missing"]),
        );
        assert_eq!(r.context_recall, Some(0.5)); // 1 of 2 gold present
        let p = r.context_precision.expect("precision populated");
        // 1 of 3 selected is gold → 1/3.
        assert!(
            (p - (1.0 / 3.0)).abs() < 1e-5,
            "expected precision ≈ 1/3; got {p}"
        );
    }

    #[test]
    fn empty_built_context_is_handled_gracefully() {
        // If the strategy / budget produced an empty selection, every
        // self-eval metric must be defined (no NaN, no panic). With chunk
        // gold provided, recall = 0/|gold|. Without gold, the function
        // must still produce a sensible report.
        let cfg = ContextConfig {
            // Zero budget forces an empty selection.
            token_budget: 0,
            strategy: ContextStrategy::RawTopK,
            ..Default::default()
        };
        let chunks = vec![rr("a", "some text")];
        let ctx = build_context(&Query::new("query"), &chunks, &cfg);
        assert!(
            ctx.chunks.is_empty(),
            "test premise: zero-budget should empty the selection; got {} chunks",
            ctx.chunks.len()
        );

        // No gold — every self-eval field must be finite.
        let r = evaluate(&Query::new("query"), &ctx, EvalGold::None);
        assert!(r.mean_grounding.is_finite());
        assert!(r.overall.is_finite());
        assert!((0.0..=1.0).contains(&r.overall));

        // Chunk-gold provided — recall must be 0 (nothing selected), not NaN.
        let r = evaluate(&Query::new("query"), &ctx, EvalGold::Chunks(&["expected"]));
        assert_eq!(r.context_recall, Some(0.0));
        // precision on an empty selection is reported as 0.0 (not NaN) so
        // callers can treat the field as always-finite when chunk gold is
        // present.
        assert_eq!(r.context_precision, Some(0.0));
    }
}
