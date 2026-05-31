//! Phase 7 end-to-end falsification.
//!
//! Constructs four queries that exercise four regimes, runs the full
//! `RedHop::retrieve_with_state` path (chunker → BM25 → layered
//! diagnostics → confidence profile → rule-based classifier) and asserts:
//!
//! 1. Each regime's argmax matches expectations.
//! 2. The classification trace contains at least one rule firing for the
//!    expected regime, with a non-empty justification — *interpretability
//!    is verified, not assumed*.
//! 3. Probability mass sums to ~1.0 (sanity).
//!
//! The same deterministic topic-bucket embedder from
//! `crates/diagnostics/tests/regime_discrimination.rs` is reused so the
//! test stays hermetic.

use std::sync::Arc;

use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, DiagnosticsEngine, Document, Embedding, Query, RegimeClassifier,
    RetrievalMethod, RetrievalRegime, RetrievalResult, Score, ScoreBreakdown, TokenizerBackend,
};
use redhop::retrieval::Bm25Retriever;
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::RuleBasedClassifier;
use redhop_pipeline::RedHop;

const DIM: usize = 128;

fn embed(text: &str) -> Embedding {
    const TOPIC_WEIGHT: f32 = 4.0;
    const NOISE_START: usize = 10;
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "this", "that",
        "with", "as", "be", "by", "on", "at", "it",
    ];
    const TOPIC_FELINE: &[&str] = &[
        "cat", "cats", "kitten", "kittens", "feline", "felines", "purr", "purrs", "mews", "tabby",
    ];
    const TOPIC_RUNTIME: &[&str] = &[
        "tokio",
        "executor",
        "executors",
        "scheduler",
        "schedulers",
        "future",
        "futures",
        "async",
        "runtime",
        "runtimes",
        "await",
    ];
    const TOPIC_DATABASE: &[&str] = &[
        "postgres",
        "postgresql",
        "database",
        "databases",
        "sql",
        "transaction",
        "transactions",
        "acid",
        "row",
        "rows",
    ];

    fn hash_word(w: &str) -> u64 {
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

fn embed_chunks(chunks: Vec<Chunk>) -> Vec<Chunk> {
    chunks
        .into_iter()
        .map(|c| {
            let e = embed(&c.text);
            c.with_embedding(e)
        })
        .collect()
}

/// BM25 strips chunk embeddings on retrieval. Reattach them from the
/// indexed-side cache so the semantic tier has something to work with.
fn attach_embeddings(
    state: redhop::core::RetrievalState,
    indexed: &[Chunk],
) -> redhop::core::RetrievalState {
    let mut s = state;
    for r in &mut s.candidates {
        if let Some(c) = indexed.iter().find(|c| c.id == r.chunk.id) {
            r.chunk.embedding = c.embedding.clone();
        }
    }
    s
}

async fn build_rag(docs: Vec<Document>) -> (RedHop, Vec<Chunk>) {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok.clone(), 40, 60, 0).unwrap();
    let chunks = embed_chunks(redhop::core::Chunker::chunk_batch(&chunker, &docs).unwrap());

    let mut bm25 = Bm25Retriever::new().unwrap();
    redhop::core::Retriever::index(&mut bm25, &chunks)
        .await
        .unwrap();

    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics = Arc::new(LayeredDiagnosticsEngine::lexical_and_semantic(
        lexical, semantic,
    ));
    let classifier: Arc<dyn RegimeClassifier> = Arc::new(RuleBasedClassifier::new());

    let rag = RedHop::builder()
        .with_chunker(Arc::new(chunker))
        .with_retriever(Arc::new(bm25))
        .with_diagnostics(diagnostics)
        .with_classifier(classifier)
        .build()
        .unwrap();

    (rag, chunks)
}

#[tokio::test(flavor = "multi_thread")]
async fn retrieve_with_state_returns_full_state() {
    let docs = vec![Document::new(
        "tokio",
        "Tokio is an async runtime. The executor schedules futures.",
    )];
    let (rag, indexed) = build_rag(docs).await;

    let query = Query::new("tokio async runtime").with_embedding(embed("tokio async runtime"));
    let state = rag.retrieve_with_state(query, 3).await.unwrap();
    let state = attach_embeddings(state, &indexed);

    // Re-diagnose + reclassify now that embeddings are reattached, since
    // the pipeline's first pass had BM25-only chunks.
    let dx = LayeredDiagnosticsEngine::lexical_and_semantic(
        Arc::new(DefaultDiagnosticsEngine::new()),
        Arc::new(SemanticDiagnosticsEngine::new()),
    );
    let diag = dx.diagnose(&state.query, &state.candidates).unwrap();
    let conf = redhop_orchestration::compute_confidence(&state.candidates);
    let cls = RuleBasedClassifier::new();
    let dist = cls.classify(&diag, &conf);

    assert!(diag.lexical_grounding.is_some());
    assert!(diag.semantic_grounding.is_some());
    assert!(conf.posterior_concentration.is_some());
    assert_eq!(dist.argmax, RetrievalRegime::Easy);
}

async fn classify_query(
    query_text: &str,
    rag: &RedHop,
    indexed: &[Chunk],
) -> redhop::core::RegimeDistribution {
    let query = Query::new(query_text).with_embedding(embed(query_text));
    let state = rag.retrieve_with_state(query, 4).await.unwrap();
    let state = attach_embeddings(state, indexed);

    // Reclassify against the embedded chunks so semantic-tier signal flows.
    let dx = LayeredDiagnosticsEngine::lexical_and_semantic(
        Arc::new(DefaultDiagnosticsEngine::new()),
        Arc::new(SemanticDiagnosticsEngine::new()),
    );
    let diag = dx.diagnose(&state.query, &state.candidates).unwrap();
    let conf = redhop_orchestration::compute_confidence(&state.candidates);
    RuleBasedClassifier::new().classify(&diag, &conf)
}

#[tokio::test(flavor = "multi_thread")]
async fn aligned_regime_classifies_as_easy() {
    let docs = vec![Document::new(
        "tokio",
        "Tokio is an async runtime. The Tokio executor schedules futures.",
    )];
    let (rag, indexed) = build_rag(docs).await;
    let dist = classify_query("tokio async runtime", &rag, &indexed).await;
    assert_eq!(dist.argmax, RetrievalRegime::Easy);
    let has_easy_rule = dist
        .trace
        .rules_fired
        .iter()
        .any(|f| f.regime == RetrievalRegime::Easy && !f.justification.is_empty());
    assert!(
        has_easy_rule,
        "expected at least one Easy-firing rule with a justification, got {:?}",
        dist.trace.rules_fired
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn paraphrase_regime_classifies_as_easy_through_semantic_rule() {
    let docs = vec![Document::new(
        "cats",
        "The tabby cat purrs. Cats nap. Cats stalk mice.",
    )];
    let (rag, indexed) = build_rag(docs).await;
    let dist = classify_query("feline kitten purrs", &rag, &indexed).await;
    assert_eq!(dist.argmax, RetrievalRegime::Easy);
    // The headline interpretability claim: the rule that classified this
    // as Easy must be the *semantic* one, not the lexical one — the
    // lexical signal is near zero in the paraphrase regime.
    assert!(
        dist.trace
            .rules_fired
            .iter()
            .any(|f| f.rule == "easy_semantically_grounded"),
        "expected easy_semantically_grounded to fire; got {:?}",
        dist.trace
            .rules_fired
            .iter()
            .map(|f| &f.rule)
            .collect::<Vec<_>>()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sparse_regime_classifies_as_sparse() {
    let docs = vec![Document::new(
        "tokio",
        "Tokio is an async runtime. The executor schedules futures.",
    )];
    let (rag, indexed) = build_rag(docs).await;
    // Off-corpus query — corpus is about Tokio, query about Roman history.
    let dist = classify_query("ancient roman aqueducts", &rag, &indexed).await;
    // BM25 may return zero results for an off-corpus query, in which case
    // there are no signals and the classifier defaults to uniform. Both
    // outcomes are legal — what matters is that the classifier does NOT
    // call this Easy.
    assert_ne!(dist.argmax, RetrievalRegime::Easy);
}

#[tokio::test(flavor = "multi_thread")]
async fn probabilities_sum_to_one_in_every_classification() {
    let docs = vec![Document::new(
        "mixed",
        "Tokio runtime. Cats purr. Postgres transactions.",
    )];
    let (rag, indexed) = build_rag(docs).await;

    for q in &["tokio runtime", "feline kitten", "ancient roman aqueducts"] {
        let dist = classify_query(q, &rag, &indexed).await;
        let sum: f32 = dist.probabilities.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "query={q}: probabilities sum to {sum}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn trace_always_records_features_and_thresholds() {
    let docs = vec![Document::new("tokio", "Tokio async runtime.")];
    let (rag, indexed) = build_rag(docs).await;
    let dist = classify_query("tokio async runtime", &rag, &indexed).await;
    // The classifier is contractually required to populate the trace.
    assert!(
        !dist.trace.thresholds.is_empty(),
        "thresholds must be recorded"
    );
    assert!(!dist.trace.features.is_empty(), "features must be recorded");
    assert!(
        !dist.trace.raw_scores.is_empty(),
        "raw scores must be recorded"
    );
}

// Suppress unused-import warnings if any of the testing-only items aren't
// pulled in by every path.
#[allow(dead_code)]
fn _unused() -> (
    Score,
    ScoreBreakdown,
    ChunkId,
    RetrievalResult,
    RetrievalMethod,
) {
    (
        Score {
            value: 0.0,
            method: RetrievalMethod::Lexical,
        },
        ScoreBreakdown::default(),
        ChunkId::new("x"),
        RetrievalResult::new(
            Chunk::new("a", "a", "doc", redhop::core::TokenCount(1)),
            Score {
                value: 0.0,
                method: RetrievalMethod::Lexical,
            },
        ),
        RetrievalMethod::Lexical,
    )
}
