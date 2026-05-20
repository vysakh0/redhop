//! Phase 6 falsification test.
//!
//! Claim: the semantic-tier diagnostics distinguish retrieval regimes that
//! the lexical tier alone confuses — specifically the *paraphrase regime*,
//! where retrieved chunks are semantically aligned with the query but share
//! no lexical terms with it.
//!
//! Falsification design — four regimes, each represented by a hand-built
//! query/results pair:
//!
//! | Regime               | Lexical signal           | Semantic signal      |
//! |----------------------|--------------------------|----------------------|
//! | aligned              | high grounding           | high grounding       |
//! | paraphrase           | **zero** grounding       | **high** grounding   |
//! | wrong-overlap        | high grounding           | **low** grounding    |
//! | sparse               | zero grounding           | low grounding        |
//!
//! A useful semantic tier must (a) correctly mark the paraphrase regime as
//! grounded where the lexical tier could not, (b) correctly mark the
//! wrong-overlap regime as ungrounded where the lexical tier was fooled,
//! and (c) not invent grounding signal in the sparse regime.
//!
//! Embeddings are produced by a deterministic fixed-dimensional hash so the
//! test is hermetic. The same embedder is used for both the query and the
//! chunks, so the cosines are computed against a consistent vector space.

use neorag_core::{
    Chunk, ChunkId, DiagnosticsEngine, Embedding, Query, RetrievalMethod, RetrievalResult, Score,
    ScoreBreakdown, TokenCount,
};
use neorag_diagnostics::{DefaultDiagnosticsEngine, SemanticDiagnosticsEngine};

const DIM: usize = 128;

/// Topic-bucket fake embedder, with cleanly partitioned dimensions and an
/// explicit weight ratio between topic signal and noise.
///
/// Design:
///
/// - Three topic vocabularies each map to a single distinct slot in the
///   first 3 dimensions of the vector. All vocabulary in the same topic
///   contributes to the same slot with weight `TOPIC_WEIGHT`, so two
///   sentences in the same topic expressed with disjoint vocabulary
///   produce identical topic vectors before normalization.
/// - Stopwords are dropped — they would inflate "noise overlap" between
///   semantically unrelated sentences and pollute the test.
/// - Non-stopword non-topic words are *hashed* into a wide noise band
///   (`[NOISE_START, DIM)`) with weight `1.0`. Different noise words
///   land in different slots with overwhelming probability, so noise
///   contributes a small, well-controlled signal.
///
/// The 4× weight ratio ensures topic-orthogonal pairs cosine below
/// noise-overlap pairs only when the noise overlap is unusually heavy,
/// which is exactly what the wrong-overlap regime is supposed to be.
fn embed(text: &str) -> Embedding {
    const TOPIC_WEIGHT: f32 = 4.0;
    const NOISE_START: usize = 10;
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
        "this", "that", "with", "as", "be", "by", "on", "at", "it",
    ];
    const TOPIC_FELINE: &[&str] = &[
        "cat", "cats", "kitten", "kittens", "feline", "felines", "purr", "purrs",
        "mews", "tabby",
    ];
    const TOPIC_RUNTIME: &[&str] = &[
        "tokio", "executor", "executors", "scheduler", "schedulers", "future",
        "futures", "async", "runtime", "runtimes", "await",
    ];
    const TOPIC_DATABASE: &[&str] = &[
        "postgres", "postgresql", "database", "databases", "sql", "transaction",
        "transactions", "acid", "row", "rows",
    ];

    fn hash_word(w: &str) -> u64 {
        // FNV-1a, 64-bit.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in w.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    let mut v = vec![0f32; DIM];
    for w in text.split(|c: char| !c.is_alphanumeric()) {
        let w = w.to_lowercase();
        if w.is_empty() || STOPWORDS.contains(&w.as_str()) {
            continue;
        }
        if TOPIC_FELINE.contains(&w.as_str()) {
            v[0] += TOPIC_WEIGHT;
        } else if TOPIC_RUNTIME.contains(&w.as_str()) {
            v[1] += TOPIC_WEIGHT;
        } else if TOPIC_DATABASE.contains(&w.as_str()) {
            v[2] += TOPIC_WEIGHT;
        } else {
            let slot = NOISE_START + (hash_word(&w) as usize) % (DIM - NOISE_START);
            v[slot] += 1.0;
        }
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    Embedding(v)
}

fn mk(id: &str, text: &str) -> RetrievalResult {
    let chunk = Chunk::new(ChunkId::new(id), text, "doc", TokenCount(1))
        .with_embedding(embed(text));
    RetrievalResult {
        chunk,
        score: Score {
            value: 1.0,
            method: RetrievalMethod::Dense,
        },
        breakdown: ScoreBreakdown::default(),
    }
}

fn q(text: &str) -> Query {
    Query::new(text).with_embedding(embed(text))
}

#[test]
fn lexical_tier_misses_paraphrase_regime() {
    // Query uses one vocabulary, chunks use a synonym vocabulary in the
    // same topic. Lexical grounding will be zero; semantic grounding must
    // be high.
    let query = q("feline kitten purrs");
    let results = vec![
        mk("c1", "the cat mews loudly"),
        mk("c2", "tabby cats purr a great deal"),
    ];

    let lexical = DefaultDiagnosticsEngine::new();
    let lex_report = lexical.diagnose(&query, &results).unwrap();
    assert!(
        lex_report.lexical_grounding.unwrap_or(0.0) < 0.05,
        "lexical grounding should be near zero (got {:?})",
        lex_report.lexical_grounding
    );

    let semantic = SemanticDiagnosticsEngine::new();
    let sem_report = semantic.diagnose(&query, &results).unwrap();
    assert!(
        sem_report.semantic_grounding.unwrap_or(0.0) > 0.85,
        "semantic grounding should be high in paraphrase regime (got {:?})",
        sem_report.semantic_grounding
    );
}

#[test]
fn semantic_tier_catches_wrong_overlap_regime() {
    // Construct a chunk that shares a query *non-topic* word but is
    // topically wrong. The lexical tier credits the overlap; the semantic
    // tier should not.
    let query = q("cat purrs frequently and softly");
    let results = vec![
        // Shares "frequently" / "softly" but is about a database — wrong topic.
        mk("c1", "postgres transactions commit frequently and softly"),
    ];

    let lexical = DefaultDiagnosticsEngine::new();
    let lex = lexical.diagnose(&query, &results).unwrap();
    let semantic = SemanticDiagnosticsEngine::new();
    let sem = semantic.diagnose(&query, &results).unwrap();

    assert!(
        sem.semantic_grounding.unwrap() < lex.lexical_grounding.unwrap(),
        "semantic should rate this lower than lexical: lex={:?}, sem={:?}",
        lex.lexical_grounding,
        sem.semantic_grounding
    );
}

#[test]
fn aligned_regime_both_tiers_agree_high() {
    let query = q("tokio async runtime executor");
    let results = vec![
        mk("c1", "tokio is an async runtime"),
        mk("c2", "the tokio executor schedules futures"),
    ];
    let lex = DefaultDiagnosticsEngine::new()
        .diagnose(&query, &results)
        .unwrap();
    let sem = SemanticDiagnosticsEngine::new()
        .diagnose(&query, &results)
        .unwrap();
    assert!(lex.lexical_grounding.unwrap() > 0.3);
    assert!(sem.semantic_grounding.unwrap() > 0.85);
}

#[test]
fn sparse_regime_both_tiers_disagree_with_corpus() {
    // Off-corpus query against unrelated chunks. Both tiers must be honest:
    // neither should invent grounding.
    let query = q("ancient roman aqueducts");
    let results = vec![
        mk("c1", "tokio is an async runtime"),
        mk("c2", "postgres ACID transactions"),
    ];
    let lex = DefaultDiagnosticsEngine::new()
        .diagnose(&query, &results)
        .unwrap();
    let sem = SemanticDiagnosticsEngine::new()
        .diagnose(&query, &results)
        .unwrap();
    assert!(lex.lexical_grounding.unwrap_or(0.0) < 0.05);
    assert!(
        sem.semantic_grounding.unwrap_or(1.0) < 0.55,
        "semantic should not invent grounding for sparse regime (got {:?})",
        sem.semantic_grounding
    );
}

#[test]
fn regime_discrimination_summary() {
    // Single-shot summary: the *combined* signal (max(lexical, semantic))
    // should separate hard regimes from easy regimes:
    //
    //   aligned ≥ paraphrase  ≫  wrong_overlap , sparse
    //
    // Note we use ≥ rather than > for the aligned vs paraphrase comparison.
    // The headline claim is *not* that aligned beats paraphrase — once the
    // semantic tier is in play they correctly converge — but that the
    // semantic tier lifts the paraphrase regime out of the wrong_overlap /
    // sparse floor where the lexical tier alone strands it.
    let aligned = (
        q("tokio async runtime"),
        vec![mk("a", "tokio runtime async executor")],
    );
    let paraphrase = (
        q("feline kitten"),
        vec![mk("a", "the tabby cat purrs")],
    );
    let wrong_overlap = (
        q("cat purrs frequently"),
        vec![mk("a", "postgres commits frequently")],
    );
    let sparse = (
        q("ancient roman aqueducts"),
        vec![mk("a", "tokio runtime futures")],
    );

    let lex = DefaultDiagnosticsEngine::new();
    let sem = SemanticDiagnosticsEngine::new();

    let combined = |query: &Query, results: &[RetrievalResult]| -> f32 {
        let l = lex.diagnose(query, results).unwrap().lexical_grounding.unwrap_or(0.0);
        let s = sem.diagnose(query, results).unwrap().semantic_grounding.unwrap_or(0.0);
        l.max(s)
    };

    let s_aligned = combined(&aligned.0, &aligned.1);
    let s_paraphrase = combined(&paraphrase.0, &paraphrase.1);
    let s_wrong = combined(&wrong_overlap.0, &wrong_overlap.1);
    let s_sparse = combined(&sparse.0, &sparse.1);

    println!(
        "regime scores: aligned={:.3} paraphrase={:.3} wrong_overlap={:.3} sparse={:.3}",
        s_aligned, s_paraphrase, s_wrong, s_sparse
    );

    assert!(
        s_aligned >= s_paraphrase,
        "aligned ({}) should be >= paraphrase ({})",
        s_aligned,
        s_paraphrase
    );
    assert!(
        s_paraphrase > s_wrong + 0.2,
        "paraphrase ({}) should clearly beat wrong_overlap ({})",
        s_paraphrase,
        s_wrong
    );
    assert!(
        s_paraphrase > s_sparse + 0.2,
        "paraphrase ({}) should clearly beat sparse ({})",
        s_paraphrase,
        s_sparse
    );
}
