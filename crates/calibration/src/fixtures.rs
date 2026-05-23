//! Synthetic calibration fixtures.
//!
//! The fixture here exists to demonstrate the harness end-to-end. It is
//! NOT a substitute for real labeled data. Headline calibration numbers
//! should always come from your own [`LabeledCorpus`] built from
//! HotpotQA traces or your domain workload.
//!
//! The fixture uses a deterministic topic-bucket embedder (3 topical
//! bands + a wide noise band) so the test is hermetic. The same
//! embedder is used throughout the test/example suite — see
//! `crates/diagnostics/tests/regime_discrimination.rs`.

use redhop_core::{ChunkId, Document, Embedding, RetrievalRegime};

use crate::dataset::{LabeledCorpus, LabeledQuery};

/// Topic-bucket embedder.
pub fn embed(text: &str) -> Embedding {
    const DIM: usize = 128;
    const TOPIC_WEIGHT: f32 = 4.0;
    const NOISE_START: usize = 10;
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "this", "that",
        "with", "as", "be", "by", "on", "at", "it",
    ];
    const TOPIC_FELINE: &[&str] = &[
        "cat", "cats", "kitten", "kittens", "feline", "felines", "purr", "purrs", "mews", "tabby",
        "whiskers", "paws", "claws",
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
        "task",
        "spawn",
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
        "index",
        "mvcc",
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

/// Build the synthetic calibration corpus.
///
/// Returns a [`LabeledCorpus`] with:
/// - 15 documents across 3 topics (cats, runtime, database) plus 3
///   noise-only distractor documents.
/// - 25 queries: 5 per regime (Easy / Paraphrase / DistractorHeavy /
///   Ambiguous / Sparse). Each query has gold chunk ids referring to
///   the chunker output for the relevant documents.
///
/// **Gold chunk id convention.** Documents are chunked by the
/// `SentenceChunker`, which assigns ids of the form
/// `"<source>::sent::<idx>"`. The fixture pins the chunker config
/// (`target=40, max=60, overlap=0`) so the ids are deterministic.
pub fn synthetic_dataset() -> LabeledCorpus {
    let docs = vec![
        // Cats (5)
        Document::new(
            "cats-1",
            "The tabby cat purrs softly. Cats nap in warm sun. Cats stalk mice for fun.",
        ),
        Document::new(
            "cats-2",
            "Kittens are playful. Kittens explore boxes. Kittens mew when hungry.",
        ),
        Document::new(
            "cats-3",
            "Cat whiskers sense air currents. Whiskers help cats navigate dark spaces.",
        ),
        Document::new(
            "cats-4",
            "Cats use claws to climb trees. Claws retract when not in use.",
        ),
        Document::new(
            "cats-5",
            "A feline's paws are silent. Paws have soft pads. Pads cushion every step.",
        ),
        // Tokio runtime (5)
        Document::new(
            "tokio-1",
            "Tokio is an async runtime. The runtime drives futures to completion.",
        ),
        Document::new(
            "tokio-2",
            "The Tokio executor schedules tasks. Spawned tasks run concurrently.",
        ),
        Document::new(
            "tokio-3",
            "Futures in Rust are lazy. Futures must be polled by an executor.",
        ),
        Document::new(
            "tokio-4",
            "Async functions in Tokio await IO. The scheduler manages context switches.",
        ),
        Document::new(
            "tokio-5",
            "Tokio's work-stealing scheduler balances load across worker threads.",
        ),
        // Postgres (5)
        Document::new(
            "pg-1",
            "Postgres provides ACID transactions. Transactions guarantee atomicity.",
        ),
        Document::new(
            "pg-2",
            "SQL queries hit Postgres indexes. Indexes speed up row lookups.",
        ),
        Document::new(
            "pg-3",
            "MVCC lets Postgres serve concurrent reads without blocking writes.",
        ),
        Document::new(
            "pg-4",
            "Postgres stores rows on disk. Each row has a transaction id.",
        ),
        Document::new(
            "pg-5",
            "Database administrators tune index strategies. ACID guarantees protect data.",
        ),
        // Distractor / noise (3)
        Document::new(
            "noise-1",
            "Breakfast was tasty. The weather is warm. Trees grow tall in summer.",
        ),
        Document::new(
            "noise-2",
            "The mountain stands alone. Rivers flow downhill. Stars shine at night.",
        ),
        Document::new(
            "noise-3",
            "Coffee tastes bitter. Bread rises slowly. Bakeries open early.",
        ),
    ];

    let q = |id: &str, text: &str, regime: RetrievalRegime, gold: &[&str]| -> LabeledQuery {
        let mut lq = LabeledQuery::new(id, text, regime).with_embedding(embed(text));
        lq.gold_chunk_ids = gold.iter().map(|g| ChunkId::new(*g)).collect();
        lq
    };

    // Gold chunk ids: with SentenceChunker (target=40, max=60, overlap=0)
    // and the docs above, a single document fits in one chunk (well below
    // 40 tokens). So `<source>::sent::0` is the only chunk per doc.
    let queries = vec![
        // ── Easy (5): high lexical + high semantic match against a single topical doc.
        q(
            "easy-1",
            "tokio async runtime",
            RetrievalRegime::Easy,
            &["tokio-1::sent::0"],
        ),
        q(
            "easy-2",
            "tokio executor scheduler",
            RetrievalRegime::Easy,
            &["tokio-2::sent::0", "tokio-5::sent::0"],
        ),
        q(
            "easy-3",
            "postgres ACID transactions",
            RetrievalRegime::Easy,
            &["pg-1::sent::0"],
        ),
        q(
            "easy-4",
            "cat whiskers paws",
            RetrievalRegime::Easy,
            &["cats-3::sent::0", "cats-5::sent::0"],
        ),
        q(
            "easy-5",
            "futures await scheduler",
            RetrievalRegime::Easy,
            &["tokio-3::sent::0", "tokio-4::sent::0"],
        ),
        // ── Easy via paraphrase (5): low lexical overlap, high semantic.
        // True regime is still Easy — only the path to it differs.
        q(
            "para-1",
            "feline kitten purrs",
            RetrievalRegime::Easy,
            &["cats-1::sent::0", "cats-2::sent::0"],
        ),
        q(
            "para-2",
            "asynchronous workload runner",
            RetrievalRegime::Easy,
            &["tokio-1::sent::0", "tokio-2::sent::0"],
        ),
        q(
            "para-3",
            "relational store rows acid",
            RetrievalRegime::Easy,
            &["pg-1::sent::0", "pg-4::sent::0"],
        ),
        q(
            "para-4",
            "kitten claws climb",
            RetrievalRegime::Easy,
            &["cats-2::sent::0", "cats-4::sent::0"],
        ),
        q(
            "para-5",
            "spawned async task",
            RetrievalRegime::Easy,
            &["tokio-2::sent::0"],
        ),
        // ── DistractorHeavy (5): query terms overlap noise docs heavily.
        q(
            "dist-1",
            "the tasty warm rows",
            RetrievalRegime::DistractorHeavy,
            &["pg-4::sent::0"],
        ),
        q(
            "dist-2",
            "early bitter task",
            RetrievalRegime::DistractorHeavy,
            &["tokio-2::sent::0"],
        ),
        q(
            "dist-3",
            "tall mountain claws",
            RetrievalRegime::DistractorHeavy,
            &["cats-4::sent::0"],
        ),
        q(
            "dist-4",
            "stars summer index",
            RetrievalRegime::DistractorHeavy,
            &["pg-2::sent::0"],
        ),
        q(
            "dist-5",
            "coffee tabby softly",
            RetrievalRegime::DistractorHeavy,
            &["cats-1::sent::0"],
        ),
        // ── Ambiguous (5): terms span multiple topics.
        q(
            "amb-1",
            "cat scheduler",
            RetrievalRegime::Ambiguous,
            &["cats-1::sent::0", "tokio-2::sent::0"],
        ),
        q(
            "amb-2",
            "feline transactions",
            RetrievalRegime::Ambiguous,
            &["cats-1::sent::0", "pg-1::sent::0"],
        ),
        q(
            "amb-3",
            "kitten async executor",
            RetrievalRegime::Ambiguous,
            &["cats-2::sent::0", "tokio-2::sent::0"],
        ),
        q(
            "amb-4",
            "tokio mvcc claws",
            RetrievalRegime::Ambiguous,
            &["tokio-1::sent::0", "pg-3::sent::0", "cats-4::sent::0"],
        ),
        q(
            "amb-5",
            "purrs scheduler postgres",
            RetrievalRegime::Ambiguous,
            &["cats-1::sent::0", "tokio-2::sent::0", "pg-1::sent::0"],
        ),
        // ── Sparse (5): off-corpus.
        q(
            "sparse-1",
            "ancient roman aqueducts",
            RetrievalRegime::Sparse,
            &[],
        ),
        q(
            "sparse-2",
            "ottoman empire architecture",
            RetrievalRegime::Sparse,
            &[],
        ),
        q(
            "sparse-3",
            "renaissance painting techniques",
            RetrievalRegime::Sparse,
            &[],
        ),
        q(
            "sparse-4",
            "deep sea hydrothermal vents",
            RetrievalRegime::Sparse,
            &[],
        ),
        q(
            "sparse-5",
            "medieval guild apprenticeship",
            RetrievalRegime::Sparse,
            &[],
        ),
    ];

    LabeledCorpus { docs, queries }
}
