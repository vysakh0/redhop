//! In-process evaluation of a `BuiltContext` (and optionally the LLM's
//! answer) against optional gold signals.
//!
//! The user-facing answer to "how good was that context assembly — and did
//! the LLM stay on it?", computed from the same primitives the runtime
//! already uses to make decisions — grounding scores, the existing
//! [`ContextReport`] fields — with **no LLM judge in the loop** by default.
//!
//! There are two tiers of answer-quality metrics in this module:
//!
//! - **Tier 1 — lexical** (this module, deterministic). Token-overlap proxies
//!   for faithfulness / relevancy / correctness. Cheap, no LLM, no API key,
//!   runs in CI. Named with the `_lexical` suffix so callers don't confuse
//!   them with the real (LLM-judged) thing.
//! - **Tier 2 — judged** (separate module, opt-in). LLM-scored
//!   faithfulness / relevancy / correctness — what Ragas calls these by
//!   default. Same `EvalReport` shape; fields named with the `_judged`
//!   suffix. See `crate::judge` (Phase 2).
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
//! // Self-evaluation only — no answer, no gold.
//! let score = evaluate(&Query::new("refund window"), &ctx, None, EvalGold::None, None);
//! println!("overall={:.2}  density={:.2}", score.overall, score.evidence_density);
//!
//! // With the LLM's answer — unlocks Tier-1 answer-quality proxies.
//! let score = evaluate(
//!     &Query::new("refund window"),
//!     &ctx,
//!     Some("Thirty days from purchase."),
//!     EvalGold::Answer("thirty days"),
//!     None,  // no judge — Tier-2 fields stay None
//! );
//! println!(
//!     "faithfulness_lexical={:?}  correctness_lexical={:?}",
//!     score.faithfulness_lexical, score.correctness_lexical,
//! );
//! # Ok(()) }
//! ```

use crate::analyzer::{default_english, Analyzer};
use crate::context::{grounding_score, BuiltContext};
use crate::core::Query;
use crate::judge::{Judge, JudgeRequest};
use std::collections::HashSet;

// ── Tier-2 judge prompts ─────────────────────────────────────────────────
//
// Calibrated to be short, low-token, and parseable by `judge::parse_score`.
// Every prompt asks for a single numeric reply ("Reply with the number
// only.") so small models (gpt-4o-mini, claude-haiku, llama-8b) stay on
// task. The system messages name a strict rubric so different vendors
// interpret the task the same way.

const FAITHFULNESS_SYSTEM: &str = "You are a strict, careful judge. Your job is to determine \
    whether an ANSWER is supported by a given CONTEXT, with no claims beyond what the \
    CONTEXT actually says. A claim is unsupported if it adds details not in the CONTEXT, \
    even if those details are plausible.";

const RELEVANCY_SYSTEM: &str = "You are a strict judge of whether an answer directly \
    addresses a question. An off-topic, evasive, or partial-only answer should score below 1.0.";

const CORRECTNESS_SYSTEM: &str = "You are a strict judge of factual correctness comparing a \
    generated answer to a reference answer. Score on whether the generated answer conveys \
    the same facts as the reference, allowing paraphrase but penalizing missing or \
    contradictory facts.";

fn faithfulness_prompt(context: &str, answer: &str) -> String {
    format!(
        "CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nIs every claim in the ANSWER supported \
         by the CONTEXT? Reply with a single number from 0 (any claim fabricated or \
         unsupported) to 1 (every claim supported). Reply with the number only."
    )
}

fn relevancy_prompt(question: &str, answer: &str) -> String {
    format!(
        "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nDoes the ANSWER directly address the \
         QUESTION? Reply with a single number from 0 (does not address) to 1 (fully \
         addresses). Reply with the number only."
    )
}

fn correctness_prompt(gold: &str, answer: &str) -> String {
    format!(
        "REFERENCE ANSWER:\n{gold}\n\nGENERATED ANSWER:\n{answer}\n\nDoes the GENERATED \
         ANSWER convey the same facts as the REFERENCE ANSWER (paraphrase is fine; missing \
         or contradicted facts hurt the score)? Reply with a single number from 0 \
         (incorrect) to 1 (matches). Reply with the number only."
    )
}

/// Call the judge with the given system + prompt, returning a normalized
/// `[0, 1]` score (or `None` if the judge or the parser errors). The
/// raw error is dropped because eval is best-effort — a single failed
/// judge call shouldn't crash a whole evaluate(); the missing metric
/// surfaces as `None` and the caller decides whether to retry or
/// ignore.
fn judge_score(judge: &dyn Judge, system: &str, prompt: &str) -> Option<f32> {
    let req = JudgeRequest {
        prompt,
        system: Some(system),
    };
    match judge.score(&req) {
        Ok(resp) => Some(resp.score.clamp(0.0, 1.0)),
        Err(_) => None,
    }
}

/// Naive sentence splitter for the lexical-faithfulness metric. Splits on
/// `.`, `?`, `!` followed by whitespace, plus newlines. Drops empty
/// segments. Good enough for the token-overlap proxy — we're computing
/// "what fraction of these segments share vocabulary with context", not
/// extracting structured claims. A misclassified sentence boundary at
/// worst shifts a single token across segments; the aggregate fraction is
/// robust.
fn split_sentences(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        // Newline is an unconditional sentence break (bullet lists, line-
        // broken prose). `.` `?` `!` only split when followed by
        // whitespace or end-of-input — that's how we avoid splitting on
        // "U.S.A." or "3.14".
        let split_here = match c {
            b'\n' => true,
            b'.' | b'?' | b'!' => {
                i + 1 == bytes.len() || bytes[i + 1].is_ascii_whitespace()
            }
            _ => false,
        };
        if split_here {
            let seg = text[start..=i].trim();
            if !seg.is_empty() {
                sentences.push(seg);
            }
            start = i + 1;
        }
        i += 1;
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

/// Sentence-level lexical faithfulness proxy. For each sentence in
/// `answer`, compute the fraction of its analyzer-extracted terms that
/// appear in `context_text`'s term set. A sentence is "supported" if a
/// majority (>= half) of its content terms appear in the context. Returns
/// the fraction of supported sentences over total. Empty answer ⇒ NaN
/// would be misleading; we return `None` at the caller-level instead.
fn faithfulness_lexical_score(answer: &str, context_text: &str, analyzer: &dyn Analyzer) -> f32 {
    let sentences = split_sentences(answer);
    if sentences.is_empty() {
        return 0.0;
    }
    let ctx_terms: HashSet<String> = analyzer.tokens(context_text).into_iter().collect();
    let mut supported = 0_usize;
    for sentence in &sentences {
        let s_terms: Vec<String> = analyzer.tokens(sentence);
        if s_terms.is_empty() {
            // A sentence with no content tokens (stopwords / punctuation
            // only) is vacuously consistent with any context — don't
            // penalize it.
            supported += 1;
            continue;
        }
        let unique: HashSet<&str> = s_terms.iter().map(|s| s.as_str()).collect();
        let overlap = unique
            .iter()
            .filter(|t| ctx_terms.contains(**t))
            .count();
        // Majority rule: at least half the content terms must appear in
        // context. This is a deliberately loose bar — strict-equal would
        // miss any paraphrase; this catches "the answer mentions tokens
        // never in the context."
        let threshold = (unique.len() + 1) / 2;
        if overlap >= threshold {
            supported += 1;
        }
    }
    supported as f32 / sentences.len() as f32
}

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

    // ── Tier-1 answer-quality metrics (lexical proxies — not LLM-judged) ──
    // Populated when the caller supplies the LLM's `answer` text. Token-
    // overlap proxies — fast, deterministic, but weaker signals than
    // LLM-judged faithfulness/relevancy. Use `_lexical` results in CI for
    // regression detection; reach for the `_judged` fields (Phase 2) when
    // you need real hallucination detection.
    /// Sentence-level token-overlap proxy for faithfulness: what fraction
    /// of the answer's sentences have at least half their content terms
    /// also present in the assembled context. **Not real faithfulness** —
    /// an LLM judge is the right tool for that. This catches the obvious
    /// "answer mentions things never in the context" failure mode; it
    /// won't catch a confidently-wrong paraphrase. `None` if no `answer`
    /// was supplied to `evaluate`.
    pub faithfulness_lexical: Option<f32>,
    /// Token-overlap between the query and the answer, in `[0, 1]`. Same
    /// scorer as `mean_grounding` (Snowball-stemmed, stopword-filtered)
    /// applied to the (query, answer) pair. A proxy for "did the answer
    /// address the question". `None` if no `answer` was supplied.
    pub relevancy_lexical: Option<f32>,
    /// Token-overlap between the LLM's answer and the gold answer, in
    /// `[0, 1]`. A proxy for "did the LLM produce the right tokens";
    /// strict and easily fooled by paraphrase, but works as a regression
    /// signal on stable QA pairs. `None` unless BOTH `answer` and
    /// `gold_answer` were supplied.
    pub correctness_lexical: Option<f32>,

    // ── Tier-2 LLM-judged answer-quality metrics ──
    // Populated when the caller supplies BOTH an `answer` AND a `judge`.
    // Each metric is one LLM call (cached by `CachedJudge` so re-runs are
    // free). Stronger signals than the `_lexical` proxies above — catch
    // paraphrase-correct answers, paraphrase-incorrect answers,
    // confidently-wrong fabrications. A judge call that errors leaves
    // the corresponding metric as `None`.
    /// LLM-judged faithfulness: "is every claim in the answer supported
    /// by the assembled context?" Score in `[0, 1]`. This is the real
    /// hallucination-detection signal — strictly stronger than
    /// `faithfulness_lexical`. `None` unless `answer` + `judge` were
    /// supplied (and the judge call succeeded).
    pub faithfulness_judged: Option<f32>,
    /// LLM-judged relevancy: "does the answer directly address the
    /// question?" Score in `[0, 1]`. Stronger than
    /// `relevancy_lexical` (catches answers that share query terms
    /// without actually answering). `None` unless `answer` + `judge`
    /// were supplied.
    pub relevancy_judged: Option<f32>,
    /// LLM-judged correctness: "does the generated answer convey the
    /// same facts as the gold answer?" Score in `[0, 1]`. Allows
    /// paraphrase — strictly stronger than `correctness_lexical` which
    /// only counts shared tokens. `None` unless `answer` + `gold_answer`
    /// + `judge` were all supplied.
    pub correctness_judged: Option<f32>,

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

/// Evaluate an assembled `BuiltContext` (and optionally the LLM's answer)
/// against optional ground truth.
///
/// See the module-level docs for the design rationale and a usage example.
/// All metrics here are deterministic, in-process, and require no LLM call
/// — this is Tier 1. For the LLM-judged answer-quality metrics (real
/// faithfulness / relevancy / correctness), see `crate::judge` and
/// `evaluate_with_judge` (Phase 2 of the eval roadmap).
///
/// **Inputs and what each one unlocks:**
///
/// | input | unlocks |
/// |---|---|
/// | `query`, `ctx` (always) | `mean_grounding`, `evidence_density`, `retained_evidence_ratio`, `second_hop_rescues`, `low_confidence`, `estimated_waste_tokens` |
/// | `answer` ≠ `None` | `faithfulness_lexical`, `relevancy_lexical` |
/// | `gold` contains chunks | `context_recall`, `context_precision` |
/// | `gold` contains an answer | `answer_token_recall` |
/// | `answer` + `gold` contains an answer | `correctness_lexical` |
/// | `answer` + `judge` | `faithfulness_judged`, `relevancy_judged` |
/// | `answer` + `gold` contains an answer + `judge` | `correctness_judged` |
///
/// **Self-eval vs gold-conditional — what evaluate actually tells you.**
/// Without ground truth, `evaluate` populates *self-eval* fields. These
/// measure how **focused** the assembled context is relative to the
/// query — whether the chunks share query vocabulary, whether the budget
/// is being spent on relevant tokens. They do **not** tell you whether
/// the **correct** answer-bearing chunk is in the context. To measure
/// correctness, pass `gold` and/or `answer`.
pub fn evaluate(
    query: &Query,
    ctx: &BuiltContext,
    answer: Option<&str>,
    gold: EvalGold<'_>,
    judge: Option<&dyn Judge>,
) -> EvalReport {
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

    let answer_token_recall = gold_answer.map(|gold| {
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
        grounding_score(gold, &assembled)
    });

    // ── Tier-1 answer-quality metrics (lexical proxies) ──
    let assembled_for_answer: Option<String> = answer.map(|_| {
        ctx.chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    });
    let faithfulness_lexical = match (answer, assembled_for_answer.as_deref()) {
        (Some(a), Some(assembled)) if !a.trim().is_empty() => {
            Some(faithfulness_lexical_score(a, assembled, analyzer.as_ref()))
        }
        _ => None,
    };
    let relevancy_lexical = answer
        .filter(|a| !a.trim().is_empty())
        .map(|a| grounding_score(query.text.as_str(), a));
    let correctness_lexical = match (answer, gold_answer) {
        (Some(a), Some(gold)) if !a.trim().is_empty() && !gold.trim().is_empty() => {
            Some(grounding_score(gold, a))
        }
        _ => None,
    };

    // ── Tier-2 judged metrics ──
    // Reuses the assembled-context string computed above for the
    // lexical-faithfulness path so we don't re-concatenate.
    let (faithfulness_judged, relevancy_judged, correctness_judged) = match (
        answer.filter(|a| !a.trim().is_empty()),
        judge,
    ) {
        (Some(a), Some(j)) => {
            let assembled = assembled_for_answer.as_deref().unwrap_or("");
            let f = judge_score(j, FAITHFULNESS_SYSTEM, &faithfulness_prompt(assembled, a));
            let r = judge_score(j, RELEVANCY_SYSTEM, &relevancy_prompt(query.text.as_str(), a));
            let c = gold_answer
                .filter(|g| !g.trim().is_empty())
                .and_then(|g| judge_score(j, CORRECTNESS_SYSTEM, &correctness_prompt(g, a)));
            (f, r, c)
        }
        _ => (None, None, None),
    };

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
    // Tier-1 answer-quality metrics weighted lighter than gold-relative
    // ones (they're lexical proxies, not direct correctness signals) but
    // heavier than self-eval (they incorporate the LLM's output, which is
    // closer to "did the system actually answer" than self-eval alone).
    if let Some(f) = faithfulness_lexical {
        add(f, 2.0);
    }
    if let Some(r) = relevancy_lexical {
        add(r, 1.5);
    }
    if let Some(c) = correctness_lexical {
        add(c, 2.5);
    }
    // Tier-2 judged metrics dominate when present (stronger signals than
    // lexical proxies — they catch the failures lexical misses). When a
    // _judged metric is present, its _lexical counterpart effectively
    // becomes redundant; we still include both in the blend because
    // _lexical is cheap and the disagreement between them is itself a
    // signal the caller may want to see.
    if let Some(f) = faithfulness_judged {
        add(f, 4.0);
    }
    if let Some(r) = relevancy_judged {
        add(r, 3.0);
    }
    if let Some(c) = correctness_judged {
        add(c, 4.0);
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
        faithfulness_lexical,
        relevancy_lexical,
        correctness_lexical,
        faithfulness_judged,
        relevancy_judged,
        correctness_judged,
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
        let r = evaluate(&Query::new("refund window"), &ctx, None, EvalGold::None, None);
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
            None,
            EvalGold::Chunks(&["hit1", "hit2"]),
        None,
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
            None,
            EvalGold::Chunks(&["hit", "missing"]), // "missing" never indexed
            None,
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
            None,
            EvalGold::Answer("refunds within thirty days"),
        None,
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
            None,
            EvalGold::None,
        None,
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
        let r = evaluate(&Query::new("query"), &ctx, None, EvalGold::Chunks(&[]), None);
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
            None,
            EvalGold::Answer("thirty days"),
        None,
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
            None,
            EvalGold::Both {
                gold_chunk_ids: &["hit"],
                gold_answer: "thirty days",
            },
            None,
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
            None,
            EvalGold::Chunks(&["hit", "missing"]),
        None,
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
        let r = evaluate(&Query::new("query"), &ctx, None, EvalGold::None, None);
        assert!(r.mean_grounding.is_finite());
        assert!(r.overall.is_finite());
        assert!((0.0..=1.0).contains(&r.overall));

        // Chunk-gold provided — recall must be 0 (nothing selected), not NaN.
        let r = evaluate(&Query::new("query"), &ctx, None, EvalGold::Chunks(&["expected"]), None);
        assert_eq!(r.context_recall, Some(0.0));
        // precision on an empty selection is reported as 0.0 (not NaN) so
        // callers can treat the field as always-finite when chunk gold is
        // present.
        assert_eq!(r.context_precision, Some(0.0));
    }

    // ── Tier-1 lexical answer-quality metrics ──────────────────────────────

    #[test]
    fn tier1_metrics_none_when_no_answer_provided() {
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let r = evaluate(&Query::new("refund window"), &ctx, None, EvalGold::None, None);
        assert!(r.faithfulness_lexical.is_none());
        assert!(r.relevancy_lexical.is_none());
        assert!(r.correctness_lexical.is_none());
    }

    #[test]
    fn faithfulness_lexical_high_when_answer_grounded_in_context() {
        // Answer is essentially a paraphrase of the context — every
        // content term in the answer appears (after stemming) in the
        // chunks. Faithfulness should be high.
        let ctx = build(
            "refund window",
            &[rr(
                "a",
                "the refund window is thirty days from purchase. customers may return items.",
            )],
        );
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("The refund window is thirty days from purchase."),
            EvalGold::None,
        None,
        );
        let f = r.faithfulness_lexical.expect("populated");
        assert!(
            f >= 0.9,
            "answer paraphrasing the context should score near 1.0; got {f}"
        );
    }

    #[test]
    fn faithfulness_lexical_low_when_answer_fabricated() {
        // Answer mentions tokens never in the context (a classic
        // hallucination shape). Lexical faithfulness should drop.
        let ctx = build(
            "refund window",
            &[rr(
                "a",
                "the refund window is thirty days from purchase",
            )],
        );
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            // Three sentences, all about unrelated tokens (quantum
            // mechanics) — context never mentions any of these terms.
            Some("Quantum chromodynamics couples gluons. \
                  Schrödinger equations describe quantum states. \
                  Heisenberg uncertainty bounds measurement."),
            EvalGold::None,
        None,
        );
        let f = r.faithfulness_lexical.expect("populated");
        assert!(
            f <= 0.5,
            "fabricated answer with no context overlap should score low; got {f}"
        );
    }

    #[test]
    fn relevancy_lexical_uses_query_answer_overlap() {
        let ctx = build("refund window", &[rr("a", "any text — doesn't matter")]);
        // Answer that directly echoes query terms (refund window) — high relevancy.
        let on_topic = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("The refund window is thirty days."),
            EvalGold::None,
        None,
        );
        // Answer with zero query-term overlap — low relevancy.
        let off_topic = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Photosynthesis converts sunlight into glucose."),
            EvalGold::None,
        None,
        );
        let r_on = on_topic.relevancy_lexical.expect("populated");
        let r_off = off_topic.relevancy_lexical.expect("populated");
        assert!(
            r_on > r_off,
            "on-topic answer must score higher; on={r_on}, off={r_off}"
        );
    }

    #[test]
    fn correctness_lexical_requires_both_answer_and_gold_answer() {
        let ctx = build("refund window", &[rr("a", "anything")]);
        // Answer only — no correctness yet.
        let r_answer_only = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days from purchase."),
            EvalGold::None,
        None,
        );
        assert!(r_answer_only.correctness_lexical.is_none());

        // Gold answer only — also no correctness (no LLM output to compare).
        let r_gold_only = evaluate(
            &Query::new("refund window"),
            &ctx,
            None,
            EvalGold::Answer("thirty days"),
        None,
        );
        assert!(r_gold_only.correctness_lexical.is_none());

        // Both — correctness populated.
        let r_both = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days from purchase."),
            EvalGold::Answer("thirty days"),
        None,
        );
        let c = r_both
            .correctness_lexical
            .expect("populated when both answer and gold_answer supplied");
        assert!(c > 0.0, "correctness should be positive on overlap; got {c}");
    }

    #[test]
    fn empty_answer_string_treated_as_no_answer() {
        // An empty/whitespace-only answer is the same as "no answer" — none
        // of the Tier-1 metrics should fire.
        let ctx = build("refund window", &[rr("a", "refund thirty days")]);
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("   "),
            EvalGold::Answer("thirty days"),
        None,
        );
        assert!(r.faithfulness_lexical.is_none());
        assert!(r.relevancy_lexical.is_none());
        assert!(r.correctness_lexical.is_none());
    }

    #[test]
    fn split_sentences_basic_punctuation() {
        let sents = split_sentences("First sentence. Second one! Third? Final.");
        assert_eq!(sents.len(), 4);
        assert_eq!(sents[0], "First sentence.");
        assert_eq!(sents[3], "Final.");
    }

    #[test]
    fn split_sentences_newlines_split_too() {
        let sents = split_sentences("Bullet one\nBullet two\nBullet three");
        assert_eq!(sents.len(), 3);
    }

    #[test]
    fn split_sentences_single_sentence_no_terminator() {
        let sents = split_sentences("just one fragment with no period");
        assert_eq!(sents.len(), 1);
    }

    #[test]
    fn split_sentences_empty_input() {
        let sents = split_sentences("");
        assert!(sents.is_empty());
    }

    // ── Tier-2 judged metrics (Phase 3) ─────────────────────────────────────

    use crate::judge::{CallableJudge, Judge, JudgeRequest, JudgeResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Stub judge that returns a fixed score and counts invocations.
    fn const_judge(score: f32, counter: Arc<AtomicUsize>) -> CallableJudge<impl Fn(&JudgeRequest<'_>) -> crate::core::Result<JudgeResponse> + Send + Sync> {
        CallableJudge::with_name("stub", move |_req| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(JudgeResponse {
                score,
                raw_text: format!("{score}"),
                model: "stub".into(),
            })
        })
    }

    /// Stub judge that returns different scores per system prompt — so
    /// faithfulness / relevancy / correctness can be distinguished.
    fn keyed_judge(counter: Arc<AtomicUsize>) -> CallableJudge<impl Fn(&JudgeRequest<'_>) -> crate::core::Result<JudgeResponse> + Send + Sync> {
        CallableJudge::with_name("keyed", move |req| {
            counter.fetch_add(1, Ordering::SeqCst);
            let score = match req.system.unwrap_or("") {
                s if s.contains("supported by a given CONTEXT") => 0.7,  // faithfulness
                s if s.contains("addresses a question") => 0.9,           // relevancy
                s if s.contains("factual correctness") => 0.6,            // correctness
                _ => 0.0,
            };
            Ok(JudgeResponse {
                score,
                raw_text: format!("{score}"),
                model: "keyed".into(),
            })
        })
    }

    #[test]
    fn tier2_metrics_none_when_no_judge() {
        // Even with answer + gold supplied, _judged fields stay None
        // unless a judge is passed.
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days."),
            EvalGold::Answer("thirty days"),
            None,
        );
        assert!(r.faithfulness_judged.is_none());
        assert!(r.relevancy_judged.is_none());
        assert!(r.correctness_judged.is_none());
    }

    #[test]
    fn tier2_metrics_none_when_no_answer_even_with_judge() {
        // Judge needs an answer to score — without one, no judged
        // metrics are populated even if a judge is available.
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = const_judge(1.0, counter.clone());
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            None,
            EvalGold::Answer("thirty days"),
            Some(&judge),
        );
        assert!(r.faithfulness_judged.is_none());
        assert!(r.relevancy_judged.is_none());
        assert!(r.correctness_judged.is_none());
        // And critically the judge wasn't called.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn tier2_faithfulness_and_relevancy_fire_with_answer_and_judge() {
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = keyed_judge(counter.clone());
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days from purchase."),
            EvalGold::None,
            Some(&judge),
        );
        assert_eq!(r.faithfulness_judged, Some(0.7));
        assert_eq!(r.relevancy_judged, Some(0.9));
        // correctness_judged requires a gold answer; without it stays None.
        assert!(r.correctness_judged.is_none());
        // Exactly 2 judge calls (faithfulness, relevancy).
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn tier2_correctness_judged_requires_gold_answer() {
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = keyed_judge(counter.clone());
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days from purchase."),
            EvalGold::Answer("thirty days"),
            Some(&judge),
        );
        assert_eq!(r.correctness_judged, Some(0.6));
        // 3 judge calls now (faithfulness, relevancy, correctness).
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn tier2_judge_error_leaves_metric_none_doesnt_panic() {
        // A judge that always errors should not crash evaluate; the
        // metric just stays None.
        let ctx = build("refund window", &[rr("a", "refund window")]);
        let judge = CallableJudge::with_name("err", |_req| {
            Err(crate::core::Error::Other("transport".into()))
        });
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days."),
            EvalGold::Answer("thirty days"),
            Some(&judge),
        );
        assert!(r.faithfulness_judged.is_none());
        assert!(r.relevancy_judged.is_none());
        assert!(r.correctness_judged.is_none());
        // The lexical and self-eval fields should still be populated.
        assert!(r.faithfulness_lexical.is_some());
        assert!(r.relevancy_lexical.is_some());
        assert!(r.mean_grounding >= 0.0);
    }

    #[test]
    fn tier2_judged_metrics_lift_overall_score() {
        // A judge returning high scores should produce a higher overall
        // than the same call without a judge — verifies the judged
        // fields are actually wired into the composite.
        let ctx = build("refund window", &[rr("a", "the refund window is thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let high_judge = const_judge(1.0, counter.clone());
        let no_judge_overall = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("The refund window is thirty days."),
            EvalGold::Answer("thirty days"),
            None,
        )
        .overall;
        let judge_overall = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("The refund window is thirty days."),
            EvalGold::Answer("thirty days"),
            Some(&high_judge),
        )
        .overall;
        assert!(
            judge_overall > no_judge_overall,
            "perfect judged metrics must raise overall; judged={judge_overall}, no_judge={no_judge_overall}"
        );
    }
}
