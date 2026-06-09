//! In-process evaluation of a `BuiltContext` (and optionally the LLM's
//! answer) against optional gold signals.
//!
//! The user-facing answer to "how good was that context assembly — and did
//! the LLM stay on it?", computed from the same primitives the runtime
//! already uses to make decisions — grounding scores, the existing
//! [`ContextReport`] fields — with **no LLM judge in the loop** by default.
//!
//! There are two flavors of answer-quality metrics in this module:
//!
//! - **Lexical** — deterministic token-overlap proxies for faithfulness /
//!   relevancy / correctness. Cheap, no LLM, no API key, runs in CI. Named
//!   with the `_lexical` suffix so callers don't confuse them with the
//!   real (LLM-judged) thing.
//! - **Judged** — opt-in, requires a [`crate::judge::Judge`]. LLM-scored
//!   faithfulness / relevancy / correctness. Same `EvalReport` shape;
//!   fields named with the `_judged` suffix.
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
//! # use redhop::context::eval::{evaluate, EvalConfig, EvalGold};
//! # fn demo() -> redhop::Result<()> {
//! let mut doc = Document::from_text("policy.md", "the refund window is thirty days")?;
//! let ctx = doc.context("refund window")?;
//!
//! // Self-evaluation only — no answer, no gold.
//! let score = evaluate(
//!     &Query::new("refund window"),
//!     &ctx,
//!     None,
//!     EvalGold::None,
//!     None,
//!     EvalConfig::default(),
//! );
//! println!("overall={:.2}  density={:.2}", score.overall, score.evidence_density);
//!
//! // With the LLM's answer — unlocks lexical answer-quality proxies.
//! let score = evaluate(
//!     &Query::new("refund window"),
//!     &ctx,
//!     Some("Thirty days from purchase."),
//!     EvalGold::Answer("thirty days"),
//!     None,  // no judge — judged fields stay None
//!     EvalConfig::default(),
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

// ── judged judge prompts ─────────────────────────────────────────────────
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

// ── judged claim-decomposition (, opt-in via EvalConfig) ─────────
// Two-pass approach: instead of one yes/no judge call per (answer,
// context), decompose the answer into atomic claims and verify each
// against the context. More accurate; costs +N LLM calls per
// evaluate. Opt in via `EvalConfig::decompose_faithfulness=true`.

const CLAIM_EXTRACTION_SYSTEM: &str = "You are a precise reader. Decompose answers into atomic, \
    independently-verifiable factual claims.";

const CLAIM_VERIFICATION_SYSTEM: &str = "You are a strict, careful judge. Determine whether a \
    single CLAIM is supported by a given CONTEXT, with no inference beyond what the CONTEXT \
    actually says.";

fn claim_extraction_prompt(answer: &str) -> String {
    // Few-shot prompt with TWO examples. The simple example shows
    // pronoun resolution + dropping filler. The second example shows
    // that compound attributions ("X did Y in Z") must stay as a SINGLE
    // claim — splitting them into innocent components lets a true part
    // ("Z is a real thing") dilute a false core claim ("X did Y").
    // The HotpotQA-traced regression that motivated this is captured
    // in docs/findings/EVAL_JUDGED_CALIBRATION.md.
    format!(
        "Decompose an ANSWER into atomic factual claims for verification. \
         Each claim must be self-contained — resolve pronouns, drop \
         conversational filler. Keep COMPOUND attributions ('X did Y in Z') \
         as a single claim — do NOT split into innocent components, which \
         would dilute a false core claim. Output one claim per line, no \
         numbering, no introduction. If the ANSWER makes no verifiable \
         factual claims (refusal, pure opinion), output nothing.\n\n\
         EXAMPLE 1 — simple\n\
         ANSWER: He was a German-born theoretical physicist best known for \
         developing the theory of relativity. He also made important \
         contributions to quantum mechanics.\n\
         CLAIMS:\n\
         Albert Einstein was a German-born theoretical physicist.\n\
         Albert Einstein was best known for developing the theory of relativity.\n\
         Albert Einstein made important contributions to quantum mechanics.\n\n\
         EXAMPLE 2 — apposition with compound attribution\n\
         ANSWER: The singer of 'A', Jim Cummings, voiced Tails in 'Sonic'.\n\
         CLAIMS:\n\
         Jim Cummings is the singer of 'A'.\n\
         Jim Cummings voiced Tails in 'Sonic'.\n\n\
         NOW DO THIS ONE\n\
         ANSWER: {answer}\n\
         CLAIMS:"
    )
}

fn claim_verification_prompt(context: &str, claim: &str) -> String {
    format!(
        "CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nIs the CLAIM supported by the CONTEXT? \
         Reply with a single number from 0 (unsupported or contradicted) to 1 (fully \
         supported). Reply with the number only."
    )
}

/// Batched verification — score N claims in a single LLM call instead
/// of N. The prompt asks for one line per claim in `N: SCORE` format;
/// the parser is robust to commentary, missing lines, and number
/// shapes. Falls back to NaN per claim when a line is unparseable,
/// and the caller treats NaN as "unsupported / 0".
fn claim_verification_batched_prompt(context: &str, claims: &[String]) -> String {
    let mut numbered = String::new();
    for (i, claim) in claims.iter().enumerate() {
        numbered.push_str(&format!("{}. {claim}\n", i + 1));
    }
    // The rubric, the outside-knowledge ban, and the two worked
    // examples (one positive, one negative) all fix a class of failure
    // where the verifier was returning "supported" for claims the
    // CONTEXT didn't actually mention — relying on world knowledge
    // instead. The comparative-claims rule fixes the specific HotpotQA
    // case where "X is older than Y" was passing even when the context
    // only had X's birth date. Output format unchanged for parser
    // compatibility; the example shows REASONING is suppressed in the
    // final output.
    format!(
        "Judge whether each CLAIM is supported by the CONTEXT. The CONTEXT \
         is the ONLY source of truth — outside / world knowledge does NOT \
         count as support. If the CONTEXT does not mention a fact the claim \
         asserts, the score is 0, even if you know it to be true.\n\n\
         Scoring rubric:\n\
         \u{2003}1.0 — every part of the claim is explicitly stated or directly entailed by the CONTEXT.\n\
         \u{2003}0.5 — partial support: some parts in the CONTEXT, others absent.\n\
         \u{2003}0.0 — at least one part is NOT in the CONTEXT or is contradicted.\n\n\
         For comparative claims (older than, before, larger than): the \
         CONTEXT must contain enough information about BOTH sides to make \
         the comparison. If one side's underlying attribute is missing, \
         score 0.\n\n\
         EXAMPLE 1\n\
         CONTEXT: Annie Morton (born October 8, 1970) is an American model.\n\
         CLAIM: Annie Morton is older than Terry Richardson.\n\
         REASONING: Context gives Annie Morton's birth date but says nothing about Terry Richardson's date of birth. Comparison cannot be made from CONTEXT alone.\n\
         SCORE: 0\n\n\
         EXAMPLE 2\n\
         CONTEXT: WINNER is a South Korean boy group formed in 2014 by YG Entertainment.\n\
         CLAIM: WINNER was formed by YG Entertainment.\n\
         REASONING: Explicitly stated in CONTEXT.\n\
         SCORE: 1\n\n\
         CONTEXT:\n{context}\n\n\
         Claims to verify against the CONTEXT:\n{numbered}\n\
         Now for EACH numbered claim above, output ONE line in this exact format:\n\
         N: SCORE\n\
         where N is the claim number and SCORE is from 0 to 1. No commentary, \
         no reasoning in the output — score only. One line per claim, in order.\n\n\
         Example output for 3 claims:\n\
         1: 1.0\n\
         2: 0.0\n\
         3: 0.5"
    )
}

/// Parse the batched-verification response. Returns one score per
/// claim in input order; lines that fail to parse become NaN. Robust
/// to: leading commentary the LLM might add, missing claim lines
/// (filled as NaN), out-of-order responses (reorders by `N`),
/// trailing junk after the score.
fn parse_batched_scores(text: &str, n_claims: usize) -> Vec<f32> {
    let mut scores: Vec<f32> = vec![f32::NAN; n_claims];
    for line in text.lines() {
        let line = line.trim();
        // Split on the first `:` — `1: 0.9` → ("1", "0.9")
        let Some((idx_str, score_str)) = line.split_once(':') else {
            continue;
        };
        let idx_str = idx_str.trim_matches(|c: char| !c.is_ascii_digit());
        let Ok(idx) = idx_str.parse::<usize>() else {
            continue;
        };
        if idx == 0 || idx > n_claims {
            continue;
        }
        // parse_score handles "0.9", "yes", "1", "90%", "Score: 0.9", etc.
        if let Ok(score) = crate::judge::parse_score(score_str.trim()) {
            scores[idx - 1] = score;
        }
    }
    scores
}

/// Parse the LLM's claim-extraction response into individual claims.
/// Tolerates the three common LLM output shapes: bare one-per-line,
/// numbered ("1. claim"), and bulleted ("- claim" / "* claim").
/// Whitespace-only lines are dropped. No claims (empty / refusal
/// response) returns an empty vec.
fn parse_claims(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|raw| {
            let s = raw.trim();
            if s.is_empty() {
                return None;
            }
            // Strip numbered "1. " / "1) " prefix.
            let s = if let Some(rest) = s
                .strip_prefix(|c: char| c.is_ascii_digit())
                .and_then(|r| r.strip_prefix(|c: char| c == '.' || c == ')'))
                .map(|r| r.trim_start())
            {
                rest
            } else {
                s
            };
            // Strip bullet "- " / "* " prefix.
            let s = s.trim_start_matches(['-', '*']).trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        })
        .collect()
}

/// Use the judge to decompose an answer into claims and verify each
/// against the context. Returns `(score, n_extracted, n_supported)`
/// where score is the mean of per-claim verification scores. A claim
/// is "supported" if its verification score is ≥ 0.5 (mirrors the
/// half-supported threshold used by `faithfulness_lexical_score`).
/// Returns `None` when claim extraction errors or yields zero claims —
/// we don't fall back to single-prompt scoring because mixing the two
/// scoring modes would conflate signals.
fn faithfulness_judged_decomposed(
    judge: &dyn Judge,
    context: &str,
    answer: &str,
) -> Option<(f32, usize, usize)> {
    // Step 1: extract claims (1 LLM call).
    let extraction_req = JudgeRequest {
        prompt: &claim_extraction_prompt(answer),
        system: Some(CLAIM_EXTRACTION_SYSTEM),
    };
    let extracted = match judge.score(&extraction_req) {
        Ok(resp) => resp.raw_text,
        Err(_) => return None,
    };
    let claims = parse_claims(&extracted);
    if claims.is_empty() {
        return None;
    }

    // Step 2: verify all claims in ONE LLM call (batched). This is the
    // A single batched LLM call for
    // statement-set and so should we; N→1 LLM calls per evaluate is a
    // real cost win at scale. Per-claim failures (parse errors,
    // missing lines from the LLM) become NaN, which we treat as
    // "unsupported" (score 0) so the metric is still well-defined.
    let batched_req = JudgeRequest {
        prompt: &claim_verification_batched_prompt(context, &claims),
        system: Some(CLAIM_VERIFICATION_SYSTEM),
    };
    let batched_text = match judge.score(&batched_req) {
        Ok(resp) => resp.raw_text,
        Err(_) => {
            // Total verification failure — return None rather than
            // reporting an all-zero faithfulness, which would be
            // misleading.
            return None;
        }
    };
    let per_claim_scores = parse_batched_scores(&batched_text, claims.len());
    let mut sum = 0.0_f32;
    let mut supported = 0_usize;
    for s in &per_claim_scores {
        let s = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
        sum += s;
        if s >= 0.5 {
            supported += 1;
        }
    }
    let n = claims.len();
    Some((sum / n as f32, n, supported))
}

const RELEVANCY_SYSTEM: &str = "You are a strict judge of whether an answer directly \
    addresses a question. An off-topic, evasive, or partial-only answer should score below \
    1.0. A NONCOMMITTAL answer (\"I don't know\", \"I'm not sure\", \"As an AI I cannot\", \
    \"It depends\", refusal of any form) MUST score 0 regardless of vocabulary overlap with \
    the question.";

const CORRECTNESS_SYSTEM: &str = "You are a strict judge of factual correctness comparing a \
    generated answer to a reference answer. Score on whether the generated answer conveys \
    the same facts as the reference, allowing paraphrase but penalizing missing or \
    contradictory facts.";

const CORRECTNESS_CLASSIFICATION_SYSTEM: &str = "You are a strict judge classifying which \
    factual claims from a GENERATED ANSWER are present in a REFERENCE ANSWER and vice \
    versa. Be precise — paraphrased claims that convey the same fact count as matches.";

fn correctness_classification_prompt(answer_claims: &[String], gold_claims: &[String]) -> String {
    let mut prompt = String::new();
    prompt.push_str("Compare two lists of factual claims.\n\n");
    prompt.push_str("CLAIMS FROM THE GENERATED ANSWER:\n");
    for (i, c) in answer_claims.iter().enumerate() {
        prompt.push_str(&format!("A{}. {c}\n", i + 1));
    }
    prompt.push_str("\nCLAIMS FROM THE REFERENCE ANSWER:\n");
    for (i, c) in gold_claims.iter().enumerate() {
        prompt.push_str(&format!("R{}. {c}\n", i + 1));
    }
    prompt.push_str(
        "\nFor EACH generated claim (A1, A2, …), classify it as one of:\n\
         - TP: directly supported by one or more REFERENCE claims (paraphrase ok)\n\
         - FP: not supported by any REFERENCE claim, OR contradicts a REFERENCE claim\n\n\
         For EACH reference claim (R1, R2, …) that is NOT covered by any generated claim, \
         emit FN.\n\n\
         Output one verdict per line in this EXACT format (no commentary):\n\
         A1: TP\n\
         A2: FP\n\
         …\n\
         FN: R2\n\
         FN: R3\n\
         …",
    );
    prompt
}

/// Parsed TP/FP/FN counts from a correctness-classification
/// response. Lines that don't parse are skipped silently — the F-beta
/// score is computed from whatever the LLM returned.
fn parse_correctness_classification(
    text: &str,
    n_answer: usize,
    n_gold: usize,
) -> (usize, usize, usize) {
    let mut answer_verdicts: Vec<Option<bool>> = vec![None; n_answer]; // Some(true)=TP, Some(false)=FP
    let mut gold_missed: Vec<bool> = vec![false; n_gold];
    for line in text.lines() {
        let line = line.trim();
        let Some((tag, verdict)) = line.split_once(':') else {
            continue;
        };
        let tag = tag.trim();
        let verdict = verdict.trim().to_uppercase();
        if let Some(idx_str) = tag.strip_prefix('A').or_else(|| tag.strip_prefix('a')) {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx > 0 && idx <= n_answer {
                    if verdict.contains("TP") {
                        answer_verdicts[idx - 1] = Some(true);
                    } else if verdict.contains("FP") {
                        answer_verdicts[idx - 1] = Some(false);
                    }
                }
            }
        } else if tag.eq_ignore_ascii_case("FN") {
            // verdict is the reference-claim id, e.g. "R3".
            let rest = verdict.trim_start_matches(|c: char| c == 'R' || c == 'r');
            if let Ok(idx) = rest.parse::<usize>() {
                if idx > 0 && idx <= n_gold {
                    gold_missed[idx - 1] = true;
                }
            }
        }
    }
    let tp = answer_verdicts.iter().filter(|v| **v == Some(true)).count();
    let fp = answer_verdicts.iter().filter(|v| **v == Some(false)).count();
    let fn_ = gold_missed.iter().filter(|m| **m).count();
    (tp, fp, fn_)
}

/// F-beta score from TP / FP / FN counts. β > 1 favors recall, β < 1
/// favors precision, β = 1 is the standard F1. Returns 0.0 when all
/// three counts are zero (degenerate — no data to score).
fn f_beta(tp: usize, fp: usize, fn_: usize, beta: f32) -> f32 {
    if tp == 0 && fp == 0 && fn_ == 0 {
        return 0.0;
    }
    let tp = tp as f32;
    let fp = fp as f32;
    let fn_ = fn_ as f32;
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let beta_sq = beta * beta;
    let denom = beta_sq * precision + recall;
    if denom == 0.0 {
        0.0
    } else {
        (1.0 + beta_sq) * (precision * recall) / denom
    }
}

/// two-pass 2-pass correctness: extract claims from both
/// answer and gold, classify each as TP/FP/FN, return F1 (β=1). One
/// extraction call for the answer + one for the gold + one
/// classification call = 3 LLM calls per evaluate. Returns
/// `(score, tp, fp, fn)`; `None` when either extraction yields zero
/// claims (degenerate input).
fn correctness_judged_decomposed(
    judge: &dyn Judge,
    answer: &str,
    gold: &str,
) -> Option<(f32, usize, usize, usize)> {
    // Extract claims from the answer (1 LLM call).
    let answer_req = JudgeRequest {
        prompt: &claim_extraction_prompt(answer),
        system: Some(CLAIM_EXTRACTION_SYSTEM),
    };
    let answer_claims = parse_claims(&judge.score(&answer_req).ok()?.raw_text);
    if answer_claims.is_empty() {
        return None;
    }
    // Extract claims from the gold answer (1 LLM call).
    let gold_req = JudgeRequest {
        prompt: &claim_extraction_prompt(gold),
        system: Some(CLAIM_EXTRACTION_SYSTEM),
    };
    let gold_claims = parse_claims(&judge.score(&gold_req).ok()?.raw_text);
    if gold_claims.is_empty() {
        return None;
    }
    // Classify (1 LLM call).
    let classify_req = JudgeRequest {
        prompt: &correctness_classification_prompt(&answer_claims, &gold_claims),
        system: Some(CORRECTNESS_CLASSIFICATION_SYSTEM),
    };
    let text = judge.score(&classify_req).ok()?.raw_text;
    let (tp, fp, fn_) = parse_correctness_classification(&text, answer_claims.len(), gold_claims.len());
    Some((f_beta(tp, fp, fn_, 1.0), tp, fp, fn_))
}

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
         QUESTION? Score 0 if the answer is noncommittal/evasive (\"I don't know\", \"It \
         depends\", any refusal) — vocabulary overlap with the question doesn't count. \
         Otherwise score from 0 (does not address) to 1 (fully addresses). Reply with the \
         number only."
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

/// Mode-and-flag knobs for [`evaluate`] that aren't (query, ctx, answer,
/// gold, judge). Default = behavior shipped through /// (single-prompt faithfulness when a judge is supplied). Set
/// `decompose_faithfulness=true` to opt into two-pass claim
/// extraction + per-claim verification — more accurate, costs +N LLM
/// calls per evaluate where N is the number of claims in the answer.
#[derive(Debug, Clone, Default)]
pub struct EvalConfig {
    /// When `true` AND `judge` + `answer` are both supplied, compute
    /// `faithfulness_judged` via 2-pass claim decomposition:
    /// (1) extract atomic claims from the answer, (2) verify ALL
    /// claims in a single batched LLM call (fix;     /// per-claim is no longer the path here), return the mean score.
    /// Also populates `n_faithfulness_claims_extracted` and
    /// `n_faithfulness_claims_supported` on the report.
    ///
    /// When `false` (default), `faithfulness_judged` is computed via
    /// a single prompt over (answer, context). Faster, cheaper, but
    /// can miss partial-truth answers where some claims check out and
    /// others don't.
    pub decompose_faithfulness: bool,

    /// When `true` AND `judge` + `answer` + `gold_answer` are all
    /// supplied, compute `correctness_judged` via 3-pass classification
    /// (two-pass): extract claims from the answer, extract claims
    /// from the gold, classify each as TP / FP / FN, return F1. Also
    /// populates `n_correctness_tp`, `n_correctness_fp`, and
    /// `n_correctness_fn` on the report so callers can debug which
    /// facts were missed vs hallucinated.
    ///
    /// When `false` (default), `correctness_judged` is a single
    /// prompt comparing the two answers directly. Faster, cheaper,
    /// but doesn't surface the TP/FP/FN breakdown.
    pub decompose_correctness: bool,
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

    // ── Lexical answer-quality metrics (deterministic proxies, no LLM) ──
    // Populated when the caller supplies the LLM's `answer` text. Token-
    // overlap proxies — fast, deterministic, but weaker signals than
    // LLM-judged faithfulness/relevancy. Use `_lexical` results in CI for
    // regression detection; reach for the `_judged` fields when you need
    // real hallucination detection.
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

    // ── LLM-judged answer-quality metrics ──
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

    // ── claim-decomposition diagnostics ──
    // Populated only when `EvalConfig::decompose_faithfulness=true`
    // AND the extraction-pass returned at least one claim. Surfaces
    // "this faithfulness was computed from K claims, M supported"
    // so a caller can spot a degenerate case (zero claims extracted,
    // wildly few supported) before reading too much into the score.
    /// Number of atomic claims the judge extracted from the answer.
    /// `None` when decomposition was disabled, errored, or returned
    /// zero claims.
    pub n_faithfulness_claims_extracted: Option<usize>,
    /// Number of those claims the judge scored ≥ 0.5 against the
    /// context (the per-claim "supported" threshold). `None` under
    /// the same conditions as `n_faithfulness_claims_extracted`.
    pub n_faithfulness_claims_supported: Option<usize>,

    // ── correctness-classification diagnostics ──
    // Populated only when `EvalConfig::decompose_correctness=true`
    // AND extraction yielded ≥1 claim on both the answer AND the
    // gold answer. Surfaces "the LLM's answer covered K of the gold's
    // M facts, plus J extra facts not in the gold" so callers can
    // debug what's missing vs fabricated.
    /// True-positives: claims in the answer directly supported by a
    /// reference claim. `None` when classification was disabled,
    /// errored, or returned zero claims on either side.
    pub n_correctness_tp: Option<usize>,
    /// False-positives: claims in the answer NOT supported by any
    /// reference claim (or contradicted). `None` under the same
    /// conditions as `n_correctness_tp`.
    pub n_correctness_fp: Option<usize>,
    /// False-negatives: claims in the reference NOT covered by the
    /// answer. `None` under the same conditions as `n_correctness_tp`.
    pub n_correctness_fn: Option<usize>,

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
/// — this is lexical metrics. For the LLM-judged answer-quality metrics (real
/// faithfulness / relevancy / correctness), see `crate::judge` and
/// `evaluate_with_judge` ().
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
    config: EvalConfig,
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

    // ── lexical answer-quality metrics (lexical proxies) ──
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

    // ── judged judged metrics ──
    // Reuses the assembled-context string computed above for the
    // lexical-faithfulness path so we don't re-concatenate.
    let (
        faithfulness_judged,
        relevancy_judged,
        correctness_judged,
        n_faithfulness_claims_extracted,
        n_faithfulness_claims_supported,
    ) = match (answer.filter(|a| !a.trim().is_empty()), judge) {
        (Some(a), Some(j)) => {
            let assembled = assembled_for_answer.as_deref().unwrap_or("");
            // Faithfulness has two paths: single-prompt (fast, default) or
            // claim-decomposed (, opt-in). They populate the SAME
            // faithfulness_judged field; the n_claims_* fields disambiguate
            // which mode produced the score.
            let (f, n_ex, n_sup) = if config.decompose_faithfulness {
                match faithfulness_judged_decomposed(j, assembled, a) {
                    Some((score, n_ex, n_sup)) => (Some(score), Some(n_ex), Some(n_sup)),
                    None => (None, None, None),
                }
            } else {
                (
                    judge_score(j, FAITHFULNESS_SYSTEM, &faithfulness_prompt(assembled, a)),
                    None,
                    None,
                )
            };
            let r = judge_score(j, RELEVANCY_SYSTEM, &relevancy_prompt(query.text.as_str(), a));
            let c = gold_answer.filter(|g| !g.trim().is_empty()).and_then(|g| {
                if config.decompose_correctness {
                    None  // populated separately via correctness_judged_decomposed below
                } else {
                    judge_score(j, CORRECTNESS_SYSTEM, &correctness_prompt(g, a))
                }
            });
            (f, r, c, n_ex, n_sup)
        }
        _ => (None, None, None, None, None),
    };

    // ── decomposed correctness (two-pass TP/FP/FN) ──
    let (correctness_judged, n_correctness_tp, n_correctness_fp, n_correctness_fn) =
        match (answer.filter(|a| !a.trim().is_empty()), judge, gold_answer) {
            (Some(a), Some(j), Some(g))
                if config.decompose_correctness && !g.trim().is_empty() =>
            {
                match correctness_judged_decomposed(j, a, g) {
                    Some((score, tp, fp, fn_)) => {
                        (Some(score), Some(tp), Some(fp), Some(fn_))
                    }
                    None => (None, None, None, None),
                }
            }
            _ => (correctness_judged, None, None, None),
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
    // lexical answer-quality metrics weighted lighter than gold-relative
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
    // judged judged metrics dominate when present (stronger signals than
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
        n_faithfulness_claims_extracted,
        n_faithfulness_claims_supported,
        n_correctness_tp,
        n_correctness_fp,
        n_correctness_fn,
        mean_grounding,
        evidence_density,
        retained_evidence_ratio,
        second_hop_rescues,
        low_confidence,
        estimated_waste_tokens,
        overall,
    }
}

/// Aggregated statistics across a list of [`EvalReport`]s — the
/// "what's my average score across the test set" answer. Each `Option`
/// field is `None` only when *no* report in the input had that metric
/// populated; otherwise it's the mean over the subset of reports that
/// did. The `n_with_<metric>` counters surface how big each subset was
/// so callers can spot "this number is computed from 3 out of 200
/// reports" before reading too much into it.
#[derive(Debug, Clone)]
pub struct EvalSummary {
    /// Total number of reports in the input.
    pub n: usize,

    // ── always-present metrics: mean + median over all n reports ──
    /// Mean of `overall` across all `n` reports (0.0 if `n == 0`).
    pub mean_overall: f32,
    /// Median of `overall` across all `n` reports (0.0 if `n == 0`).
    pub median_overall: f32,
    /// Mean of `mean_grounding` over all `n` reports.
    pub mean_grounding: f32,
    /// Mean of `evidence_density` over all `n` reports.
    pub mean_evidence_density: f32,
    /// Fraction of reports where `low_confidence` was `true`. In `[0, 1]`.
    pub low_confidence_rate: f32,

    // ── conditionally-populated metrics: mean over the subset ──
    // For each Optional<f32> field on EvalReport, summarize emits the
    // mean over reports where it was Some(_) plus a counter of how
    // many that was. The convention "None if zero present" lets callers
    // distinguish "all reports had 0 score" from "no reports had the
    // metric".
    pub mean_context_recall: Option<f32>,
    /// Number of reports where `context_recall` was populated.
    pub n_with_context_recall: usize,
    pub mean_context_precision: Option<f32>,
    pub n_with_context_precision: usize,
    pub mean_answer_token_recall: Option<f32>,
    pub n_with_answer_token_recall: usize,
    pub mean_faithfulness_lexical: Option<f32>,
    pub n_with_faithfulness_lexical: usize,
    pub mean_relevancy_lexical: Option<f32>,
    pub n_with_relevancy_lexical: usize,
    pub mean_correctness_lexical: Option<f32>,
    pub n_with_correctness_lexical: usize,
    pub mean_faithfulness_judged: Option<f32>,
    pub n_with_faithfulness_judged: usize,
    pub mean_relevancy_judged: Option<f32>,
    pub n_with_relevancy_judged: usize,
    pub mean_correctness_judged: Option<f32>,
    pub n_with_correctness_judged: usize,
}

/// Aggregate a list of [`EvalReport`]s into an [`EvalSummary`]. Useful
/// at the end of a test-set sweep — collect one report per `(query,
/// answer, gold)` triple, then call `summarize(&reports)` to get the
/// "how did the system do, overall" numbers.
///
/// On an empty input, returns a zero-shaped summary (`n=0`, all means
/// zero, all conditional fields `None`).
pub fn summarize(reports: &[EvalReport]) -> EvalSummary {
    let n = reports.len();
    if n == 0 {
        return EvalSummary {
            n: 0,
            mean_overall: 0.0,
            median_overall: 0.0,
            mean_grounding: 0.0,
            mean_evidence_density: 0.0,
            low_confidence_rate: 0.0,
            mean_context_recall: None,
            n_with_context_recall: 0,
            mean_context_precision: None,
            n_with_context_precision: 0,
            mean_answer_token_recall: None,
            n_with_answer_token_recall: 0,
            mean_faithfulness_lexical: None,
            n_with_faithfulness_lexical: 0,
            mean_relevancy_lexical: None,
            n_with_relevancy_lexical: 0,
            mean_correctness_lexical: None,
            n_with_correctness_lexical: 0,
            mean_faithfulness_judged: None,
            n_with_faithfulness_judged: 0,
            mean_relevancy_judged: None,
            n_with_relevancy_judged: 0,
            mean_correctness_judged: None,
            n_with_correctness_judged: 0,
        };
    }

    let n_f = n as f32;
    let mean_overall: f32 = reports.iter().map(|r| r.overall).sum::<f32>() / n_f;
    let mean_grounding: f32 = reports.iter().map(|r| r.mean_grounding).sum::<f32>() / n_f;
    let mean_evidence_density: f32 =
        reports.iter().map(|r| r.evidence_density).sum::<f32>() / n_f;
    let low_confidence_rate: f32 =
        reports.iter().filter(|r| r.low_confidence).count() as f32 / n_f;

    // Median of overall — sort a copy.
    let mut overalls: Vec<f32> = reports.iter().map(|r| r.overall).collect();
    overalls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_overall = if n % 2 == 1 {
        overalls[n / 2]
    } else {
        // For even n, take the lower-of-the-two-middles (consistent with
        // most stats libraries' "lower" tie-break). Avoids averaging
        // which would imply more precision than we have.
        overalls[n / 2 - 1]
    };

    fn mean_of_some<F>(reports: &[EvalReport], pick: F) -> (Option<f32>, usize)
    where
        F: Fn(&EvalReport) -> Option<f32>,
    {
        let mut sum = 0.0_f32;
        let mut count = 0_usize;
        for r in reports {
            if let Some(v) = pick(r) {
                sum += v;
                count += 1;
            }
        }
        if count == 0 {
            (None, 0)
        } else {
            (Some(sum / count as f32), count)
        }
    }

    let (mean_context_recall, n_with_context_recall) = mean_of_some(reports, |r| r.context_recall);
    let (mean_context_precision, n_with_context_precision) =
        mean_of_some(reports, |r| r.context_precision);
    let (mean_answer_token_recall, n_with_answer_token_recall) =
        mean_of_some(reports, |r| r.answer_token_recall);
    let (mean_faithfulness_lexical, n_with_faithfulness_lexical) =
        mean_of_some(reports, |r| r.faithfulness_lexical);
    let (mean_relevancy_lexical, n_with_relevancy_lexical) =
        mean_of_some(reports, |r| r.relevancy_lexical);
    let (mean_correctness_lexical, n_with_correctness_lexical) =
        mean_of_some(reports, |r| r.correctness_lexical);
    let (mean_faithfulness_judged, n_with_faithfulness_judged) =
        mean_of_some(reports, |r| r.faithfulness_judged);
    let (mean_relevancy_judged, n_with_relevancy_judged) =
        mean_of_some(reports, |r| r.relevancy_judged);
    let (mean_correctness_judged, n_with_correctness_judged) =
        mean_of_some(reports, |r| r.correctness_judged);

    EvalSummary {
        n,
        mean_overall,
        median_overall,
        mean_grounding,
        mean_evidence_density,
        low_confidence_rate,
        mean_context_recall,
        n_with_context_recall,
        mean_context_precision,
        n_with_context_precision,
        mean_answer_token_recall,
        n_with_answer_token_recall,
        mean_faithfulness_lexical,
        n_with_faithfulness_lexical,
        mean_relevancy_lexical,
        n_with_relevancy_lexical,
        mean_correctness_lexical,
        n_with_correctness_lexical,
        mean_faithfulness_judged,
        n_with_faithfulness_judged,
        mean_relevancy_judged,
        n_with_relevancy_judged,
        mean_correctness_judged,
        n_with_correctness_judged,
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
        let r = evaluate(&Query::new("refund window"), &ctx, None, EvalGold::None, None, EvalConfig::default());
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
            EvalConfig::default(),
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
            EvalConfig::default(),
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
            EvalConfig::default(),
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
            EvalConfig::default(),
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
        let r = evaluate(&Query::new("query"), &ctx, None, EvalGold::Chunks(&[]), None, EvalConfig::default());
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
            EvalConfig::default(),
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
            EvalConfig::default(),
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
            EvalConfig::default(),
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
        let r = evaluate(&Query::new("query"), &ctx, None, EvalGold::None, None, EvalConfig::default());
        assert!(r.mean_grounding.is_finite());
        assert!(r.overall.is_finite());
        assert!((0.0..=1.0).contains(&r.overall));

        // Chunk-gold provided — recall must be 0 (nothing selected), not NaN.
        let r = evaluate(&Query::new("query"), &ctx, None, EvalGold::Chunks(&["expected"]), None, EvalConfig::default());
        assert_eq!(r.context_recall, Some(0.0));
        // precision on an empty selection is reported as 0.0 (not NaN) so
        // callers can treat the field as always-finite when chunk gold is
        // present.
        assert_eq!(r.context_precision, Some(0.0));
    }

    // ── lexical lexical answer-quality metrics ──────────────────────────────

    #[test]
    fn lexical_metrics_none_when_no_answer_provided() {
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let r = evaluate(&Query::new("refund window"), &ctx, None, EvalGold::None, None, EvalConfig::default());
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
            EvalConfig::default(),
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
            EvalConfig::default(),
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
            EvalConfig::default(),
        );
        // Answer with zero query-term overlap — low relevancy.
        let off_topic = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Photosynthesis converts sunlight into glucose."),
            EvalGold::None,
        None,
            EvalConfig::default(),
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
            EvalConfig::default(),
        );
        assert!(r_answer_only.correctness_lexical.is_none());

        // Gold answer only — also no correctness (no LLM output to compare).
        let r_gold_only = evaluate(
            &Query::new("refund window"),
            &ctx,
            None,
            EvalGold::Answer("thirty days"),
        None,
            EvalConfig::default(),
        );
        assert!(r_gold_only.correctness_lexical.is_none());

        // Both — correctness populated.
        let r_both = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days from purchase."),
            EvalGold::Answer("thirty days"),
        None,
            EvalConfig::default(),
        );
        let c = r_both
            .correctness_lexical
            .expect("populated when both answer and gold_answer supplied");
        assert!(c > 0.0, "correctness should be positive on overlap; got {c}");
    }

    #[test]
    fn empty_answer_string_treated_as_no_answer() {
        // An empty/whitespace-only answer is the same as "no answer" — none
        // of the lexical metrics should fire.
        let ctx = build("refund window", &[rr("a", "refund thirty days")]);
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("   "),
            EvalGold::Answer("thirty days"),
        None,
            EvalConfig::default(),
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

    // ── : summarize(reports) ─────────────────────────────────────────

    fn synthetic_report(overall: f32, faith_lex: Option<f32>) -> EvalReport {
        EvalReport {
            context_recall: None,
            context_precision: None,
            answer_token_recall: None,
            faithfulness_lexical: faith_lex,
            relevancy_lexical: None,
            correctness_lexical: None,
            faithfulness_judged: None,
            relevancy_judged: None,
            correctness_judged: None,
            n_faithfulness_claims_extracted: None,
            n_faithfulness_claims_supported: None,
            n_correctness_tp: None,
            n_correctness_fp: None,
            n_correctness_fn: None,
            mean_grounding: 0.5,
            evidence_density: 0.5,
            retained_evidence_ratio: 0.5,
            second_hop_rescues: 0,
            low_confidence: false,
            estimated_waste_tokens: 0,
            overall,
        }
    }

    #[test]
    fn summarize_empty_returns_zeros_and_none() {
        let s = summarize(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.mean_overall, 0.0);
        assert_eq!(s.median_overall, 0.0);
        assert!(s.mean_faithfulness_lexical.is_none());
        assert!(s.mean_faithfulness_judged.is_none());
    }

    #[test]
    fn summarize_aggregates_overall_and_self_eval_always() {
        let reports = vec![
            synthetic_report(0.2, None),
            synthetic_report(0.4, None),
            synthetic_report(0.9, None),
        ];
        let s = summarize(&reports);
        assert_eq!(s.n, 3);
        assert!((s.mean_overall - 0.5).abs() < 1e-5);
        assert_eq!(s.median_overall, 0.4);
        assert_eq!(s.low_confidence_rate, 0.0); // no reports flagged
    }

    #[test]
    fn summarize_optional_metrics_aggregate_over_present_subset() {
        // 3 reports — only 2 of them have faithfulness_lexical populated.
        let reports = vec![
            synthetic_report(0.5, Some(0.8)),
            synthetic_report(0.5, None),
            synthetic_report(0.5, Some(0.6)),
        ];
        let s = summarize(&reports);
        assert_eq!(s.n_with_faithfulness_lexical, 2);
        // Mean is over the 2 reports that have it — (0.8 + 0.6) / 2 = 0.7
        let f = s.mean_faithfulness_lexical.expect("at least one report has it");
        assert!((f - 0.7).abs() < 1e-5);
    }

    #[test]
    fn summarize_none_when_no_report_has_field() {
        // 0 reports with faithfulness_lexical → mean is None.
        let reports = vec![
            synthetic_report(0.5, None),
            synthetic_report(0.5, None),
        ];
        let s = summarize(&reports);
        assert!(s.mean_faithfulness_lexical.is_none());
        assert_eq!(s.n_with_faithfulness_lexical, 0);
    }

    // ── judged judged metrics () ─────────────────────────────────────

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
    fn judged_metrics_none_when_no_judge() {
        // Even with answer + gold supplied, _judged fields stay None
        // unless a judge is passed.
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days."),
            EvalGold::Answer("thirty days"),
            None,
            EvalConfig::default(),
        );
        assert!(r.faithfulness_judged.is_none());
        assert!(r.relevancy_judged.is_none());
        assert!(r.correctness_judged.is_none());
    }

    #[test]
    fn judged_metrics_none_when_no_answer_even_with_judge() {
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
            EvalConfig::default(),
        );
        assert!(r.faithfulness_judged.is_none());
        assert!(r.relevancy_judged.is_none());
        assert!(r.correctness_judged.is_none());
        // And critically the judge wasn't called.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn judged_faithfulness_and_relevancy_fire_with_answer_and_judge() {
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = keyed_judge(counter.clone());
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days from purchase."),
            EvalGold::None,
            Some(&judge),
            EvalConfig::default(),
        );
        assert_eq!(r.faithfulness_judged, Some(0.7));
        assert_eq!(r.relevancy_judged, Some(0.9));
        // correctness_judged requires a gold answer; without it stays None.
        assert!(r.correctness_judged.is_none());
        // Exactly 2 judge calls (faithfulness, relevancy).
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn judged_correctness_judged_requires_gold_answer() {
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = keyed_judge(counter.clone());
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Thirty days from purchase."),
            EvalGold::Answer("thirty days"),
            Some(&judge),
            EvalConfig::default(),
        );
        assert_eq!(r.correctness_judged, Some(0.6));
        // 3 judge calls now (faithfulness, relevancy, correctness).
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn judged_judge_error_leaves_metric_none_doesnt_panic() {
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
            EvalConfig::default(),
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
    fn judged_judged_metrics_lift_overall_score() {
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
            EvalConfig::default(),
        )
        .overall;
        let judge_overall = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("The refund window is thirty days."),
            EvalGold::Answer("thirty days"),
            Some(&high_judge),
            EvalConfig::default(),
        )
        .overall;
        assert!(
            judge_overall > no_judge_overall,
            "perfect judged metrics must raise overall; judged={judge_overall}, no_judge={no_judge_overall}"
        );
    }

    // ── : claim decomposition ────────────────────────────────────────

    /// Stub judge that returns a different response depending on which
    /// scoring task is asked: the extraction prompt gets a multi-claim
    /// list as `raw_text`, verification prompts get a numeric score.
    fn decomposer_judge(
        counter: Arc<AtomicUsize>,
        n_claims: usize,
        verification_score: f32,
    ) -> CallableJudge<impl Fn(&JudgeRequest<'_>) -> crate::core::Result<JudgeResponse> + Send + Sync>
    {
        CallableJudge::with_name("decomposer", move |req| {
            counter.fetch_add(1, Ordering::SeqCst);
            let system = req.system.unwrap_or("");
            let prompt = req.prompt;
            if system.contains("Decompose answers") {
                // Extraction: return n_claims one-per-line.
                let claims = (0..n_claims)
                    .map(|i| format!("claim {i}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(JudgeResponse {
                    score: 0.0, // unused on extraction calls
                    raw_text: claims,
                    model: "decomposer".into(),
                })
            } else if prompt.contains("N: SCORE") || prompt.contains("Claims to verify") {
                // BATCHED verification: return per-claim scores
                // as `1: <score>\n2: <score>\n...` in the format the
                // parser expects.
                let body = (1..=n_claims)
                    .map(|i| format!("{i}: {verification_score}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(JudgeResponse {
                    score: 0.0,
                    raw_text: body,
                    model: "decomposer".into(),
                })
            } else {
                // Fallback for non-decomposition arms (relevancy,
                // correctness). Numeric score.
                Ok(JudgeResponse {
                    score: verification_score,
                    raw_text: format!("{verification_score}"),
                    model: "decomposer".into(),
                })
            }
        })
    }

    #[test]
    fn decomposition_off_by_default() {
        // Without EvalConfig::decompose_faithfulness=true the legacy
        // single-prompt path runs (1 LLM call for faithfulness).
        let ctx = build("refund window", &[rr("a", "refund window thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = const_judge(0.9, counter.clone());
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("The refund window is thirty days."),
            EvalGold::None,
            Some(&judge),
            EvalConfig::default(),
        );
        assert_eq!(r.faithfulness_judged, Some(0.9));
        assert!(r.n_faithfulness_claims_extracted.is_none());
        assert!(r.n_faithfulness_claims_supported.is_none());
        // 2 calls: faithfulness + relevancy. No correctness (no gold).
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn decomposition_extracts_and_verifies_each_claim() {
        let ctx = build("refund window", &[rr("a", "the refund window is thirty days")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = decomposer_judge(counter.clone(), 3, 0.8);
        let r = evaluate(
            &Query::new("refund window"),
            &ctx,
            Some("Three claims here."),
            EvalGold::None,
            Some(&judge),
            EvalConfig {
                decompose_faithfulness: true,
                decompose_correctness: false,
            },
        );
        // 3 claims, each verified at 0.8 → mean = 0.8.
        assert_eq!(r.faithfulness_judged, Some(0.8));
        assert_eq!(r.n_faithfulness_claims_extracted, Some(3));
        // 0.8 >= 0.5 threshold → all supported.
        assert_eq!(r.n_faithfulness_claims_supported, Some(3));
        // Calls (batched): 1 extraction + 1 batched verification
        // + 1 relevancy = 3 LLM calls (was 5 pre-batching).
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn decomposition_supported_threshold_at_half() {
        // Verification score below 0.5 → claim counts as unsupported.
        let ctx = build("q", &[rr("a", "any text")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = decomposer_judge(counter.clone(), 4, 0.3);
        let r = evaluate(
            &Query::new("q"),
            &ctx,
            Some("Four claims here."),
            EvalGold::None,
            Some(&judge),
            EvalConfig {
                decompose_faithfulness: true,
                decompose_correctness: false,
            },
        );
        assert_eq!(r.faithfulness_judged, Some(0.3));
        assert_eq!(r.n_faithfulness_claims_extracted, Some(4));
        assert_eq!(r.n_faithfulness_claims_supported, Some(0));
    }

    #[test]
    fn decomposition_zero_claims_returns_none() {
        // Extraction returns an empty list (e.g. answer is a refusal /
        // pure opinion). faithfulness_judged stays None; we don't
        // silently fall back to single-prompt scoring.
        let ctx = build("q", &[rr("a", "anything")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = decomposer_judge(counter.clone(), 0, 0.5);
        let r = evaluate(
            &Query::new("q"),
            &ctx,
            Some("I cannot answer that."),
            EvalGold::None,
            Some(&judge),
            EvalConfig {
                decompose_faithfulness: true,
                decompose_correctness: false,
            },
        );
        assert!(r.faithfulness_judged.is_none());
        assert!(r.n_faithfulness_claims_extracted.is_none());
        assert!(r.n_faithfulness_claims_supported.is_none());
        // 1 extraction call + 1 relevancy = 2.
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn parse_batched_scores_handles_clean_output() {
        let scores = parse_batched_scores("1: 0.9\n2: 0.5\n3: 0.1", 3);
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 0.9).abs() < 1e-5);
        assert!((scores[1] - 0.5).abs() < 1e-5);
        assert!((scores[2] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn parse_batched_scores_tolerates_commentary_and_out_of_order() {
        // LLM might add a preamble and order differently.
        let scores = parse_batched_scores(
            "Here are the verdicts:\n3: 0.7\n1: 0.9\n2: 0.4\n\nDone.",
            3,
        );
        // Reordered correctly by N.
        assert!((scores[0] - 0.9).abs() < 1e-5);
        assert!((scores[1] - 0.4).abs() < 1e-5);
        assert!((scores[2] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn parse_batched_scores_missing_lines_are_nan() {
        // Only 2 of 3 claims got scored; the missing one is NaN.
        let scores = parse_batched_scores("1: 0.8\n3: 0.6", 3);
        assert!((scores[0] - 0.8).abs() < 1e-5);
        assert!(scores[1].is_nan());
        assert!((scores[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn parse_batched_scores_handles_percentage_and_yes_no() {
        // Each score handed to `judge::parse_score`, so the same input
        // shapes are accepted.
        let scores = parse_batched_scores("1: yes\n2: 80%\n3: no", 3);
        assert_eq!(scores[0], 1.0);
        assert!((scores[1] - 0.8).abs() < 1e-5);
        assert_eq!(scores[2], 0.0);
    }

    #[test]
    fn parse_claims_handles_common_formats() {
        // Bare one-per-line.
        let bare = parse_claims("claim a\nclaim b\nclaim c");
        assert_eq!(bare.len(), 3);
        // Numbered.
        let numbered = parse_claims("1. claim a\n2. claim b\n3) claim c");
        assert_eq!(numbered.len(), 3);
        assert_eq!(numbered[0], "claim a");
        // Bulleted.
        let bullets = parse_claims("- claim a\n* claim b\n  - claim c");
        assert_eq!(bullets.len(), 3);
        // Empty lines + whitespace dropped.
        let mixed = parse_claims("\n\n   \nclaim a\n\nclaim b\n");
        assert_eq!(mixed.len(), 2);
        // Empty input → empty.
        assert!(parse_claims("").is_empty());
        assert!(parse_claims("\n\n   ").is_empty());
    }

    // ── Decomposed correctness (TP/FP/FN classification) ────────────────────

    #[test]
    fn parse_correctness_classification_clean_output() {
        // 3 answer claims, 2 reference claims; answer has 1 TP, 2 FP;
        // reference has 1 FN.
        let text = "A1: TP\nA2: FP\nA3: FP\nFN: R2";
        let (tp, fp, fn_) = parse_correctness_classification(text, 3, 2);
        assert_eq!(tp, 1);
        assert_eq!(fp, 2);
        assert_eq!(fn_, 1);
    }

    #[test]
    fn parse_correctness_classification_tolerates_commentary() {
        let text = "Verdict:\nA1: TP\nA2: TP\nFN: R3\nFN: R5";
        let (tp, fp, fn_) = parse_correctness_classification(text, 2, 5);
        assert_eq!(tp, 2);
        assert_eq!(fp, 0);
        assert_eq!(fn_, 2);
    }

    #[test]
    fn f_beta_known_values() {
        // F1: tp=2, fp=1, fn=1 → precision=2/3, recall=2/3, f1=2/3
        let s = f_beta(2, 1, 1, 1.0);
        assert!((s - 2.0 / 3.0).abs() < 1e-5, "expected 2/3, got {s}");
        // All zeros → 0.0 (degenerate)
        assert_eq!(f_beta(0, 0, 0, 1.0), 0.0);
        // Perfect: tp=3, fp=0, fn=0 → 1.0
        assert_eq!(f_beta(3, 0, 0, 1.0), 1.0);
    }

    /// Stub judge that mimics a classifier: extraction returns N claims,
    /// classification returns a known TP/FP/FN verdict pattern.
    fn correctness_classifier_judge(
        counter: Arc<AtomicUsize>,
        n_answer_claims: usize,
        n_gold_claims: usize,
        n_tp: usize,
        n_fn: usize,
    ) -> CallableJudge<impl Fn(&JudgeRequest<'_>) -> crate::core::Result<JudgeResponse> + Send + Sync>
    {
        CallableJudge::with_name("classifier", move |req| {
            counter.fetch_add(1, Ordering::SeqCst);
            let system = req.system.unwrap_or("");
            let prompt = req.prompt;
            if system.contains("Decompose answers") {
                // Extraction. Decide which side we're on by looking at
                // the prompt body — the user's `answer` and `gold` are
                // disambiguated by their content.
                let n = if prompt.contains("ANSWER:") && prompt.contains("THE_ANSWER") {
                    n_answer_claims
                } else if prompt.contains("THE_GOLD") {
                    n_gold_claims
                } else {
                    // First call is always for the answer (in eval order).
                    n_answer_claims
                };
                let claims = (0..n)
                    .map(|i| format!("claim {i}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                return Ok(JudgeResponse {
                    score: 0.0,
                    raw_text: claims,
                    model: "classifier".into(),
                });
            }
            if system.contains("classifying") {
                // Classification: emit n_tp TPs, rest FPs, n_fn FNs.
                let mut out = String::new();
                for i in 1..=n_answer_claims {
                    if i <= n_tp {
                        out.push_str(&format!("A{i}: TP\n"));
                    } else {
                        out.push_str(&format!("A{i}: FP\n"));
                    }
                }
                for i in 1..=n_fn {
                    out.push_str(&format!("FN: R{i}\n"));
                }
                return Ok(JudgeResponse {
                    score: 0.0,
                    raw_text: out,
                    model: "classifier".into(),
                });
            }
            // Other arms (relevancy, etc.) — neutral score.
            Ok(JudgeResponse {
                score: 0.5,
                raw_text: "0.5".into(),
                model: "classifier".into(),
            })
        })
    }

    #[test]
    fn decomposed_correctness_off_by_default() {
        let ctx = build("q", &[rr("a", "ctx")]);
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = const_judge(0.8, counter);
        let r = evaluate(
            &Query::new("q"),
            &ctx,
            Some("answer"),
            EvalGold::Answer("gold"),
            Some(&judge),
            EvalConfig::default(),
        );
        // Single-prompt correctness fires; TP/FP/FN stay None.
        assert_eq!(r.correctness_judged, Some(0.8));
        assert!(r.n_correctness_tp.is_none());
        assert!(r.n_correctness_fp.is_none());
        assert!(r.n_correctness_fn.is_none());
    }

    #[test]
    fn decomposed_correctness_classifies_and_returns_f1() {
        let ctx = build("q", &[rr("a", "ctx")]);
        let counter = Arc::new(AtomicUsize::new(0));
        // Answer has 3 claims (2 TP, 1 FP); gold has 2 claims (1 FN).
        let judge =
            correctness_classifier_judge(counter.clone(), 3, 2, 2, 1);
        let r = evaluate(
            &Query::new("q"),
            &ctx,
            Some("THE_ANSWER content"),
            EvalGold::Answer("THE_GOLD content"),
            Some(&judge),
            EvalConfig {
                decompose_faithfulness: false,
                decompose_correctness: true,
            },
        );
        assert_eq!(r.n_correctness_tp, Some(2));
        assert_eq!(r.n_correctness_fp, Some(1));
        assert_eq!(r.n_correctness_fn, Some(1));
        // F1 with tp=2, fp=1, fn=1 → 2/3
        let score = r.correctness_judged.expect("populated");
        assert!((score - 2.0 / 3.0).abs() < 1e-3, "expected ~0.667, got {score}");
    }

    #[test]
    fn relevancy_judged_prompt_calls_out_noncommittal_answers() {
        // The judge sees the relevancy prompt and we can inspect it via
        // raw_text; confirm the prompt actually mentions noncommittal-
        // answer handling so an LLM can use the guidance.
        let captured = Arc::new(std::sync::Mutex::new(String::new()));
        let c = captured.clone();
        let judge = CallableJudge::with_name("inspect", move |req| {
            if req
                .system
                .map(|s| s.contains("addresses a question"))
                .unwrap_or(false)
            {
                *c.lock().unwrap() = req.prompt.to_string();
            }
            Ok(JudgeResponse {
                score: 0.5,
                raw_text: "0.5".into(),
                model: "inspect".into(),
            })
        });
        let ctx = build("q", &[rr("a", "ctx")]);
        evaluate(
            &Query::new("q"),
            &ctx,
            Some("answer"),
            EvalGold::None,
            Some(&judge),
            EvalConfig::default(),
        );
        let prompt = captured.lock().unwrap().clone();
        assert!(
            prompt.to_lowercase().contains("noncommittal"),
            "relevancy prompt should explicitly mention noncommittal answers"
        );
    }
}
