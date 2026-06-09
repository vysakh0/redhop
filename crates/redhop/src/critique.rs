//! LLM-judged aspect critique — per-aspect "score this answer on
//! these qualitative dimensions."
//!
//! `redhop::evaluate` covers the closed set of metrics every RAG
//! pipeline cares about (faithfulness / relevancy / correctness +
//! retrieval + self-eval). `critique` is the open-ended companion:
//! callers define their own aspects (harmfulness, conciseness,
//! coherence, "matches our brand voice", …) and get back one judged
//! score per aspect. The Judge from `crate::judge` does the heavy
//! lifting; this module is just the per-aspect prompting + result
//! collection.
//!
//! ## Why a separate function instead of folding into `evaluate`
//!
//! `evaluate` returns a fixed-shape struct with known fields. Aspects
//! are arbitrary strings the caller picks. Mixing them would either
//! force a `HashMap<String, f32>` onto `EvalReport` (ugly for
//! callers who never use aspects) or constrain aspect names to a
//! closed enum (defeats the purpose). A separate `critique` keeps
//! both shapes clean: `EvalReport` stays a struct, `CritiqueReport`
//! is a named-pair list.
//!
//! ## Usage
//!
//! ```no_run
//! # use redhop::judge::CallableJudge;
//! # use redhop::critique::{critique, Aspect, CritiqueInputs};
//! # fn demo() -> redhop::Result<()> {
//! # let judge = CallableJudge::with_name("stub", |_| {
//! #     Ok(redhop::judge::JudgeResponse {
//! #         score: 1.0,
//! #         raw_text: "1".into(),
//! #         model: "stub".into(),
//! #     })
//! # });
//! let aspects = [
//!     Aspect {
//!         name: "harmfulness",
//!         definition: "Does the ANSWER contain harmful, offensive, or unsafe content?",
//!         high_is_good: false,  // a "yes" answer here is BAD
//!     },
//!     Aspect {
//!         name: "conciseness",
//!         definition: "Is the ANSWER free of unnecessary repetition or filler?",
//!         high_is_good: true,
//!     },
//! ];
//! let report = critique(
//!     CritiqueInputs {
//!         answer: "Thirty days from purchase.",
//!         aspects: &aspects,
//!         context: Some("the refund window is thirty days"),
//!         query: Some("what's the refund window?"),
//!     },
//!     &judge,
//! );
//! for (name, score) in &report.scores {
//!     println!("{name} = {:?}", score);
//! }
//! # Ok(()) }
//! ```

use crate::judge::{Judge, JudgeRequest};

/// One qualitative dimension to score the answer on. The judge sees
/// `name` only as part of the rendered prompt body (for context) —
/// it's the `definition` that does the work. `high_is_good` controls
/// whether 1.0 means "the property is present" (yes) or "the property
/// is absent" (no); think of it as the polarity of the rubric.
#[derive(Debug, Clone, Copy)]
pub struct Aspect<'a> {
    /// Short label, used as the key in the returned report. Choose
    /// something display-friendly (e.g. `"harmfulness"`, not `"asp_1"`).
    pub name: &'a str,
    /// Sentence (or paragraph) describing what the judge is scoring.
    /// Write it as a question the LLM can answer 0–1: "Is the ANSWER
    /// X?" or "Does the ANSWER satisfy Y?". The judge gets this
    /// verbatim alongside the answer + optional context + query.
    pub definition: &'a str,
    /// When `true`, the LLM's raw score is preserved; 1.0 means "the
    /// property is satisfied" and the metric is "more is better". When
    /// `false`, the score is INVERTED to `1.0 - raw` before being
    /// returned, so high values still mean "good answer" across
    /// aspects with opposite polarity (e.g. harmfulness, where the
    /// LLM saying "yes, harmful" should produce a LOW final score).
    pub high_is_good: bool,
}

/// Inputs to [`critique`]. Bundled so adding optional fields (more
/// gold signals, batching options) doesn't churn the call signature.
#[derive(Debug, Clone, Copy)]
pub struct CritiqueInputs<'a> {
    /// The LLM's answer text.
    pub answer: &'a str,
    /// The aspects to score. Empty slice → empty report.
    pub aspects: &'a [Aspect<'a>],
    /// Optional context — when present, the prompt includes it so
    /// the judge can reference it (useful for aspects like "does the
    /// answer stick to the source").
    pub context: Option<&'a str>,
    /// Optional query — when present, the prompt includes it (useful
    /// for "does the answer address the question").
    pub query: Option<&'a str>,
}

/// Result of [`critique`]: one (aspect-name, score) pair per input
/// aspect, in the same order. Score is `None` if the judge call
/// errored for that aspect (transport / parse failure); same
/// best-effort semantics as `evaluate`'s `_judged` fields.
#[derive(Debug, Clone)]
pub struct CritiqueReport {
    /// `Vec<(aspect_name, Option<score>)>`. The score is in `[0, 1]`,
    /// already polarity-corrected — 1.0 means "good" regardless of
    /// the aspect's `high_is_good` flag.
    pub scores: Vec<(String, Option<f32>)>,
}

impl CritiqueReport {
    /// Number of aspects scored.
    pub fn len(&self) -> usize {
        self.scores.len()
    }
    /// True iff no aspects were scored.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
    /// Look up an aspect's score by name. Returns `None` either when
    /// the aspect isn't in the report OR when the judge call errored
    /// for it; callers who need to distinguish those cases should
    /// iterate `self.scores` directly.
    pub fn get(&self, name: &str) -> Option<f32> {
        self.scores
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, s)| *s)
    }
}

const CRITIQUE_SYSTEM: &str = "You are a strict, careful judge of a specific qualitative \
    property of a generated ANSWER. You will be told what property to score and given the \
    relevant material; reply with a single number from 0 to 1, where the meaning of the \
    scale is described in the user prompt. Reply with the number only.";

fn critique_prompt(
    aspect_def: &str,
    answer: &str,
    context: Option<&str>,
    query: Option<&str>,
) -> String {
    let mut prompt = String::new();
    if let Some(q) = query {
        prompt.push_str("QUESTION:\n");
        prompt.push_str(q);
        prompt.push_str("\n\n");
    }
    if let Some(c) = context {
        prompt.push_str("CONTEXT:\n");
        prompt.push_str(c);
        prompt.push_str("\n\n");
    }
    prompt.push_str("ANSWER:\n");
    prompt.push_str(answer);
    prompt.push_str("\n\nQUESTION TO SCORE:\n");
    prompt.push_str(aspect_def);
    prompt.push_str(
        "\n\nReply with a single number from 0 (the property is fully absent / the \
         answer fails this criterion) to 1 (the property is fully present / the answer \
         passes this criterion), or a partial value in between. Reply with the number only.",
    );
    prompt
}

/// Score an answer on multiple user-defined aspects with a single
/// [`Judge`]. Each aspect produces one Judge call; aspects with the
/// same `(answer, definition)` produce identical cache keys when the
/// underlying Judge is `CachedJudge`-wrapped, so repeating an eval
/// run is free.
///
/// On a Judge error for any aspect, that aspect's score is `None` —
/// other aspects are unaffected. Empty `aspects` slice returns an
/// empty report (and makes zero Judge calls).
pub fn critique(inputs: CritiqueInputs<'_>, judge: &dyn Judge) -> CritiqueReport {
    let scores = inputs
        .aspects
        .iter()
        .map(|aspect| {
            let prompt = critique_prompt(
                aspect.definition,
                inputs.answer,
                inputs.context,
                inputs.query,
            );
            let req = JudgeRequest {
                prompt: &prompt,
                system: Some(CRITIQUE_SYSTEM),
            };
            let score = match judge.score(&req) {
                Ok(resp) => {
                    let raw = resp.score.clamp(0.0, 1.0);
                    Some(if aspect.high_is_good { raw } else { 1.0 - raw })
                }
                Err(_) => None,
            };
            (aspect.name.to_string(), score)
        })
        .collect();
    CritiqueReport { scores }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Error, Result};
    use crate::judge::{CallableJudge, JudgeResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn const_judge(
        score: f32,
        counter: Arc<AtomicUsize>,
    ) -> CallableJudge<impl Fn(&JudgeRequest<'_>) -> Result<JudgeResponse> + Send + Sync> {
        CallableJudge::with_name("stub", move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(JudgeResponse {
                score,
                raw_text: format!("{score}"),
                model: "stub".into(),
            })
        })
    }

    #[test]
    fn empty_aspects_returns_empty_report_and_zero_calls() {
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = const_judge(1.0, counter.clone());
        let report = critique(
            CritiqueInputs {
                answer: "any text",
                aspects: &[],
                context: None,
                query: None,
            },
            &judge,
        );
        assert!(report.is_empty());
        assert_eq!(report.len(), 0);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn each_aspect_produces_one_judge_call() {
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = const_judge(0.7, counter.clone());
        let aspects = [
            Aspect {
                name: "conciseness",
                definition: "Is the answer concise?",
                high_is_good: true,
            },
            Aspect {
                name: "tone",
                definition: "Is the tone professional?",
                high_is_good: true,
            },
            Aspect {
                name: "coherence",
                definition: "Is the answer coherent?",
                high_is_good: true,
            },
        ];
        let report = critique(
            CritiqueInputs {
                answer: "a",
                aspects: &aspects,
                context: None,
                query: None,
            },
            &judge,
        );
        assert_eq!(report.len(), 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        // All should report the stub score, polarity-uncorrected because
        // high_is_good=true everywhere.
        for (name, score) in &report.scores {
            assert!(score.is_some(), "{name} should have a score");
            assert!((score.unwrap() - 0.7).abs() < 1e-5);
        }
    }

    #[test]
    fn high_is_good_false_inverts_score() {
        // "harmfulness" — the LLM saying "yes, very harmful" → 1.0 raw
        // — should produce a LOW final score because high should mean
        // "good answer" across all aspects in the report.
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = const_judge(0.9, counter); // LLM says "very harmful = 0.9"
        let aspects = [Aspect {
            name: "harmfulness",
            definition: "Does the ANSWER contain harmful content?",
            high_is_good: false,
        }];
        let report = critique(
            CritiqueInputs {
                answer: "anything",
                aspects: &aspects,
                context: None,
                query: None,
            },
            &judge,
        );
        let score = report.get("harmfulness").expect("scored");
        // 1.0 - 0.9 = 0.1
        assert!((score - 0.1).abs() < 1e-5, "expected inverted ≈ 0.1, got {score}");
    }

    #[test]
    fn judge_error_leaves_only_that_aspect_none() {
        // A judge that errors on the SECOND call only — first and third
        // aspects still get scored.
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let judge = CallableJudge::with_name("flaky", move |_| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            if n == 1 {
                Err(Error::Other("transient".into()))
            } else {
                Ok(JudgeResponse {
                    score: 0.5,
                    raw_text: "0.5".into(),
                    model: "flaky".into(),
                })
            }
        });
        let aspects = [
            Aspect {
                name: "a",
                definition: "first",
                high_is_good: true,
            },
            Aspect {
                name: "b",
                definition: "second",
                high_is_good: true,
            },
            Aspect {
                name: "c",
                definition: "third",
                high_is_good: true,
            },
        ];
        let report = critique(
            CritiqueInputs {
                answer: "x",
                aspects: &aspects,
                context: None,
                query: None,
            },
            &judge,
        );
        assert!(report.get("a").is_some());
        assert!(report.scores[1].1.is_none(), "b should be None on transient error");
        assert!(report.get("c").is_some());
    }

    #[test]
    fn critique_prompt_includes_context_and_query_when_provided() {
        // The judge captures whatever prompt it gets so we can inspect it.
        let captured = Arc::new(std::sync::Mutex::new(String::new()));
        let c = captured.clone();
        let judge = CallableJudge::with_name("capture", move |req| {
            *c.lock().expect("lock") = req.prompt.to_string();
            Ok(JudgeResponse {
                score: 1.0,
                raw_text: "1".into(),
                model: "capture".into(),
            })
        });
        let aspect = Aspect {
            name: "x",
            definition: "Some property",
            high_is_good: true,
        };
        critique(
            CritiqueInputs {
                answer: "ANSWER_TOKEN",
                aspects: &[aspect],
                context: Some("CTX_TOKEN"),
                query: Some("QUERY_TOKEN"),
            },
            &judge,
        );
        let prompt = captured.lock().expect("lock").clone();
        assert!(prompt.contains("QUERY_TOKEN"), "query must appear in prompt");
        assert!(prompt.contains("CTX_TOKEN"), "context must appear in prompt");
        assert!(prompt.contains("ANSWER_TOKEN"), "answer must appear in prompt");
        assert!(prompt.contains("Some property"), "aspect definition must appear");
    }

    #[test]
    fn get_returns_none_for_missing_aspect_name() {
        let counter = Arc::new(AtomicUsize::new(0));
        let judge = const_judge(0.5, counter);
        let aspects = [Aspect {
            name: "present",
            definition: "x",
            high_is_good: true,
        }];
        let report = critique(
            CritiqueInputs {
                answer: "a",
                aspects: &aspects,
                context: None,
                query: None,
            },
            &judge,
        );
        assert!(report.get("missing").is_none());
        assert_eq!(report.get("present"), Some(0.5));
    }
}
