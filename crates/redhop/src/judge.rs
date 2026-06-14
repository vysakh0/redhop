//! LLM-judge scaffolding for judged answer-quality metrics.
//!
//! This module is the bridge layer between RedHop's deterministic eval
//! (lexical metrics, in `crate::context::eval`) and an external LLM that scores
//! generated answers (judged metrics). The design is deliberately boring: a
//! [`Judge`] is anything that maps a prompt to a numeric score; we ship
//! a wrapper that caches identical prompts and an adapter that wraps a
//! caller-supplied closure. **We do not ship a built-in HTTP client to
//! OpenAI / Anthropic / etc.** Users bring their own LLM client (the
//! `openai` Python SDK, `litellm`, the `anthropic` crate, etc.) and wrap
//! it with [`CallableJudge::new`] in three lines.
//!
//! Why no built-in HTTP client:
//!
//! - **Bounded architecture.** RedHop is a context runtime; we don't
//!   want to own retry policy, rate-limit handling, multi-region failover,
//!   OAuth flows, etc. The ecosystem has good LLM clients already.
//! - **Auth surface.** We never want to handle API keys, and a built-in
//!   client invites users to pass them through us.
//! - **Vendor lock-in optics.** Hardcoding OpenAI would imply a
//!   recommendation; the trait approach keeps every vendor equally
//!   accessible.
//!
//! What we DO ship: the trait, a stable cache (so judged metrics are
//! deterministic across re-runs without paying for the LLM twice), and
//! conversion adapters at the Python/Node binding layer so a
//! Python `def score(prompt, system) -> float:` becomes a [`Judge`]
//! transparently.
//!
//! ## Usage (Rust)
//!
//! ```no_run
//! use redhop::judge::{Judge, JudgeRequest, JudgeResponse, CachedJudge, CallableJudge};
//!
//! // Wrap any closure that maps (prompt, system) → score in [0,1].
//! // In a real implementation you'd call OpenAI here; this stub returns 1.0.
//! let raw = CallableJudge::new(|req: &JudgeRequest<'_>| {
//!     Ok(JudgeResponse {
//!         score: 1.0,
//!         raw_text: "1".into(),
//!         model: "stub".into(),
//!     })
//! });
//!
//! // Wrap with a memory cache so identical prompts don't re-call the LLM.
//! let judge = CachedJudge::new(raw);
//!
//! let req = JudgeRequest {
//!     prompt: "Is 'the refund window is 30 days' supported by the context 'refund window: 30 days'? Reply 0 or 1.",
//!     system: Some("You are a strict faithfulness scorer."),
//! };
//! let resp = judge.score(&req).unwrap();
//! assert_eq!(resp.score, 1.0);
//! ```
//!
//! A separate module plumbs an `Option<&dyn Judge>` parameter into
//! [`crate::evaluate`] and implements the `faithfulness_judged` /
//! `relevancy_judged` / `correctness_judged` metrics on top of it.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::core::{Error, Result};

/// A single judge call: the prompt to score and any system instruction.
///
/// Kept intentionally narrow — adding a `temperature` / `max_tokens` /
/// `tools` field here would push the trait toward a generic LLM client,
/// which it isn't. The judge's job is "look at this prompt and return a
/// score"; if a vendor needs prompt-side variations, the caller can
/// embed them in `prompt` or wrap the underlying client.
#[derive(Debug, Clone, Copy)]
pub struct JudgeRequest<'a> {
    /// The user-facing prompt sent to the LLM. RedHop's judged metrics
    /// construct prompts of the shape "Is this sentence
    /// supported by this context? Reply 0..1."
    pub prompt: &'a str,
    /// Optional system instruction. RedHop's judged metrics pass a
    /// stable system prompt naming the scoring rubric so different
    /// vendors interpret the task the same way.
    pub system: Option<&'a str>,
}

/// One scored response from a [`Judge`].
///
/// `raw_text` is preserved so a caller can debug a judge that
/// systematically misparses, and so the [`CachedJudge`] persistence (if
/// enabled) can round-trip the full record. The `score` is normalized
/// to `[0, 1]` by the underlying judge — callers don't need to bound it
/// again.
#[derive(Debug, Clone)]
pub struct JudgeResponse {
    /// The judged score, normalized to `[0, 1]`. The underlying judge
    /// is responsible for the normalization; this trait stays vendor-
    /// agnostic on whether the raw LLM output is "yes/no", "0..10", or
    /// a float.
    pub score: f32,
    /// The raw text the LLM produced before normalization. Useful for
    /// debugging "why did this score 0.4?".
    pub raw_text: String,
    /// Name of the model that produced this score, for audit /
    /// observability. The caller's judge implementation populates this.
    pub model: String,
}

/// LLM-judge trait. Implementors map a prompt to a numeric score.
///
/// The trait is intentionally synchronous — most real LLM clients
/// expose both sync and async surfaces; the synchronous one is the
/// simpler floor and matches Python's blocking call style. Async
/// callers can wrap their async call in a `tokio::runtime::Handle::current().block_on(...)`
/// inside their `score()` impl, or run the eval loop in a single
/// runtime context. RedHop's eval is itself synchronous; making the
/// judge async would force the whole eval surface async.
pub trait Judge: Send + Sync {
    /// Score one prompt. Returns `Err` on transport / authentication /
    /// parse errors — the caller decides whether to retry, surface,
    /// or leave the metric as `None`.
    fn score(&self, req: &JudgeRequest<'_>) -> Result<JudgeResponse>;

    /// Score multiple prompts. Default impl serializes; vendors with
    /// native batching (OpenAI's batch API, etc.) can override.
    /// Returns the same number of responses as inputs, in order. On
    /// error, stops at the first failure.
    fn batch_score(&self, reqs: &[JudgeRequest<'_>]) -> Result<Vec<JudgeResponse>> {
        reqs.iter().map(|r| self.score(r)).collect()
    }

    /// A stable identifier for logging / observability. Default is
    /// `"unnamed-judge"`; implementors should override.
    fn name(&self) -> &str {
        "unnamed-judge"
    }
}

/// Adapter that wraps a closure as a [`Judge`].
///
/// This is the path users will reach for from Python / Node: their LLM
/// client (OpenAI SDK, LiteLLM, the `anthropic` Python package, etc.)
/// gives them a `(prompt, system) -> float` function; they wrap it
/// in `CallableJudge`. The Python/Node bindings construct a
/// `CallableJudge` over a user-supplied callable transparently.
pub struct CallableJudge<F>
where
    F: Fn(&JudgeRequest<'_>) -> Result<JudgeResponse> + Send + Sync,
{
    f: F,
    name: String,
}

impl<F> CallableJudge<F>
where
    F: Fn(&JudgeRequest<'_>) -> Result<JudgeResponse> + Send + Sync,
{
    /// Wrap `f` as a [`Judge`] named `"callable"`. Use
    /// [`Self::with_name`] to set a custom name.
    pub fn new(f: F) -> Self {
        Self {
            f,
            name: "callable".to_string(),
        }
    }

    /// Wrap `f` as a [`Judge`] with the given name (used in logs and
    /// the cache key namespacing).
    pub fn with_name(name: impl Into<String>, f: F) -> Self {
        Self {
            f,
            name: name.into(),
        }
    }
}

impl<F> Judge for CallableJudge<F>
where
    F: Fn(&JudgeRequest<'_>) -> Result<JudgeResponse> + Send + Sync,
{
    fn score(&self, req: &JudgeRequest<'_>) -> Result<JudgeResponse> {
        (self.f)(req)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// In-memory cache wrapper around any [`Judge`]. Identical
/// `(prompt, system)` pairs hit the cache instead of re-calling the
/// inner judge.
///
/// Cache key: hash of `(inner.name, prompt, system.unwrap_or(""))`.
/// Including the inner judge's `name()` in the key prevents
/// cross-contamination if a caller swaps the underlying model — a
/// score from `gpt-4o-mini` shouldn't reappear as a `claude-haiku`
/// score on a re-run.
///
/// **Persistence note (deliberate non-feature).** This cache is
/// memory-only. A future enhancement could persist to JSON Lines or
/// SQLite on `Drop`, but for the first version we keep the surface
/// small. Users who want persistence across processes can save the
/// `EvalReport` after every run and skip the cache layer entirely.
pub struct CachedJudge {
    inner: Box<dyn Judge>,
    cache: Mutex<HashMap<u64, JudgeResponse>>,
    hits: Mutex<usize>,
    misses: Mutex<usize>,
}

impl CachedJudge {
    /// Wrap `inner` with a fresh in-memory cache. Takes ownership; the
    /// caller's `J` is boxed up internally so this works with any
    /// concrete `Judge` (e.g. a `CallableJudge<F>` whose `F` is a
    /// closure type unnameable in user code).
    pub fn new<J: Judge + 'static>(inner: J) -> Self {
        Self {
            inner: Box::new(inner),
            cache: Mutex::new(HashMap::new()),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// Same as [`Self::new`] but for an already-boxed inner judge — the
    /// shape Python/Node bindings reach for since they hold judges as
    /// trait objects.
    pub fn from_boxed(inner: Box<dyn Judge>) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// Cache hit count since construction.
    pub fn hits(&self) -> usize {
        *self.hits.lock().expect("cache hits mutex poisoned")
    }

    /// Cache miss count (= LLM calls actually made) since construction.
    pub fn misses(&self) -> usize {
        *self.misses.lock().expect("cache misses mutex poisoned")
    }

    /// Number of distinct prompts cached.
    pub fn len(&self) -> usize {
        self.cache.lock().expect("cache mutex poisoned").len()
    }

    /// `true` iff no prompts have been cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn cache_key(name: &str, prompt: &str, system: Option<&str>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        name.hash(&mut h);
        prompt.hash(&mut h);
        system.unwrap_or("").hash(&mut h);
        h.finish()
    }
}

impl Judge for CachedJudge {
    fn score(&self, req: &JudgeRequest<'_>) -> Result<JudgeResponse> {
        let key = Self::cache_key(self.inner.name(), req.prompt, req.system);
        {
            let cache = self.cache.lock().expect("cache mutex poisoned");
            if let Some(hit) = cache.get(&key) {
                *self.hits.lock().expect("cache hits mutex poisoned") += 1;
                return Ok(hit.clone());
            }
        }
        let resp = self.inner.score(req)?;
        *self.misses.lock().expect("cache misses mutex poisoned") += 1;
        self.cache
            .lock()
            .expect("cache mutex poisoned")
            .insert(key, resp.clone());
        Ok(resp)
    }

    fn name(&self) -> &str {
        // Forward the inner name — the cache is an internal detail.
        // Observability that should distinguish cached vs uncached
        // judges can read `hits()` / `misses()` directly.
        self.inner.name()
    }
}

/// Parse an LLM's text response into a `[0, 1]` score. Handles the
/// three common judged output shapes: a plain float (`"0.8"`), a 0/1
/// classification (`"yes"`, `"no"`, `"1"`, `"0"`), and a percentage
/// (`"80%"`, `"80"`). Returns `Err(Error::Other)` when the response
/// has no parseable numeric content, with the raw text in the
/// message so users can debug their prompts.
///
/// This is a public helper because every judge implementation will
/// need it. Sharing the parser ensures that a "yes"-style verdict and
/// a "0.92" verdict normalize identically across vendors.
pub fn parse_score(text: &str) -> Result<f32> {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    // Yes/no classifications come first because they're unambiguous.
    if matches!(lower.as_str(), "yes" | "true") {
        return Ok(1.0);
    }
    if matches!(lower.as_str(), "no" | "false") {
        return Ok(0.0);
    }

    // Strip a trailing `%` for percentage-style outputs.
    let stripped = trimmed.trim_end_matches('%').trim();

    // Find the first contiguous run of digits / `.` / `-` / `+`. This
    // tolerates leading explanatory text — a model that returns
    // "Score: 0.8" still parses cleanly.
    let mut start: Option<usize> = None;
    let mut end: usize = 0;
    for (i, c) in stripped.char_indices() {
        if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' {
            if start.is_none() {
                start = Some(i);
            }
            end = i + c.len_utf8();
        } else if start.is_some() {
            break;
        }
    }
    let start = start
        .ok_or_else(|| Error::Other(format!("judge response has no numeric content: {text:?}")))?;
    let n: f32 = stripped[start..end]
        .parse()
        .map_err(|e| Error::Other(format!("judge response parse failed ({e}): {text:?}")))?;

    // Heuristic: a value > 1 is interpreted as a percentage if it's
    // <= 100, else clamped. < 0 clamps to 0.
    let score = if n > 1.0 {
        if n <= 100.0 {
            n / 100.0
        } else {
            1.0
        }
    } else if n < 0.0 {
        0.0
    } else {
        n
    };
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn stub_judge<F>(f: F) -> CallableJudge<F>
    where
        F: Fn(&JudgeRequest<'_>) -> Result<JudgeResponse> + Send + Sync,
    {
        CallableJudge::with_name("stub", f)
    }

    #[test]
    fn callable_judge_invokes_closure() {
        let j = stub_judge(|req| {
            Ok(JudgeResponse {
                score: if req.prompt.contains("yes") { 1.0 } else { 0.0 },
                raw_text: req.prompt.into(),
                model: "stub".into(),
            })
        });
        let r = j
            .score(&JudgeRequest {
                prompt: "yes please",
                system: None,
            })
            .unwrap();
        assert_eq!(r.score, 1.0);
    }

    #[test]
    fn batch_score_default_calls_score_in_order() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let j = stub_judge(move |req| {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(JudgeResponse {
                score: req.prompt.len() as f32 / 100.0,
                raw_text: req.prompt.into(),
                model: "stub".into(),
            })
        });
        let reqs = [
            JudgeRequest {
                prompt: "a",
                system: None,
            },
            JudgeRequest {
                prompt: "bb",
                system: None,
            },
            JudgeRequest {
                prompt: "ccc",
                system: None,
            },
        ];
        let out = j.batch_score(&reqs).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        // Order preserved.
        assert!(out[0].score < out[1].score && out[1].score < out[2].score);
    }

    #[test]
    fn cached_judge_skips_inner_on_repeat_prompt() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let inner = stub_judge(move |_req| {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(JudgeResponse {
                score: 0.7,
                raw_text: "0.7".into(),
                model: "stub".into(),
            })
        });
        let j = CachedJudge::new(inner);
        let req = JudgeRequest {
            prompt: "same prompt",
            system: None,
        };
        let r1 = j.score(&req).unwrap();
        let r2 = j.score(&req).unwrap();
        // Inner judge called once; second call served from cache.
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "second call must hit cache"
        );
        assert_eq!(r1.score, r2.score);
        assert_eq!(j.hits(), 1);
        assert_eq!(j.misses(), 1);
        assert_eq!(j.len(), 1);
    }

    #[test]
    fn cached_judge_distinct_system_prompts_are_distinct_keys() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let inner = stub_judge(move |_req| {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(JudgeResponse {
                score: 0.5,
                raw_text: "0.5".into(),
                model: "stub".into(),
            })
        });
        let j = CachedJudge::new(inner);
        j.score(&JudgeRequest {
            prompt: "p",
            system: Some("sys-a"),
        })
        .unwrap();
        j.score(&JudgeRequest {
            prompt: "p",
            system: Some("sys-b"),
        })
        .unwrap();
        // Same prompt, different system — must NOT collide.
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "distinct system prompts must be distinct cache keys"
        );
        assert_eq!(j.len(), 2);
    }

    #[test]
    fn cached_judge_propagates_inner_errors() {
        let inner = stub_judge(|_req| Err(Error::Other("boom".into())));
        let j = CachedJudge::new(inner);
        let err = j
            .score(&JudgeRequest {
                prompt: "p",
                system: None,
            })
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("boom"),
            "underlying error should propagate: {msg}"
        );
    }

    #[test]
    fn parse_score_handles_float_yes_no_percent() {
        assert_eq!(parse_score("0.8").unwrap(), 0.8);
        assert_eq!(parse_score("1").unwrap(), 1.0);
        assert_eq!(parse_score("0").unwrap(), 0.0);
        assert_eq!(parse_score("yes").unwrap(), 1.0);
        assert_eq!(parse_score("YES").unwrap(), 1.0);
        assert_eq!(parse_score("no").unwrap(), 0.0);
        assert_eq!(parse_score("true").unwrap(), 1.0);
        assert_eq!(parse_score("false").unwrap(), 0.0);
        assert_eq!(parse_score("80%").unwrap(), 0.8);
        assert_eq!(parse_score("Score: 0.42").unwrap(), 0.42);
    }

    #[test]
    fn parse_score_clamps_out_of_range() {
        assert_eq!(parse_score("150").unwrap(), 1.0);
        assert_eq!(parse_score("-0.5").unwrap(), 0.0);
        // 0..1 floats pass through unchanged.
        assert!((parse_score("0.9").unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn parse_score_errors_on_non_numeric() {
        assert!(parse_score("totally unclear").is_err());
        assert!(parse_score("").is_err());
    }
}
