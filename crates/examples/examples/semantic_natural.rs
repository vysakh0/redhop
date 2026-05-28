//! Phase-2 of the semantic-mismatch study, on NATURAL data (HotpotQA distractor).
//!
//! Does the lexical↔semantic boundary from the controlled probe hold on real
//! questions, and does a confidence gate capture dense's wins cheaply?
//!
//! Per item we retrieve the supporting paragraphs from its 10-paragraph pool
//! (2 gold + 8 distractors) with BM25 / dense (BGE exact cosine) / hybrid (RRF),
//! and bin the item by query↔gold lexical overlap:
//!   - semantic-heavy  = low overlap (the regime where lexis and meaning diverge)
//!   - lexical-friendly = high overlap
//!
//! Tier-1 (here, free): gold-paragraph recall@K per subset per mode, plus
//! escalation-trigger probes — does any deterministic BM25 score-distribution
//! signal (top1−top2 margin, top-k entropy) separate semantic-heavy from
//! lexical-friendly so we can escalate to dense only when it pays? Per-item
//! signals + Tier-3 contexts are emitted to exports/ for downstream scoring.
//!
//! Run:  cargo run -p redhop-examples --example semantic_natural --features onnx --release

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::RwLock;
use redhop_calibration::loaders::hotpotqa::HotpotQADataset;
use redhop::context::grounding_score;
use redhop::core::{Chunk, ChunkId, EmbeddingProvider, Query, Retriever, TokenCount, VectorIndex};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::retrieval::{Bm25Retriever, DenseRetriever, HybridRetriever};
use redhop::storage::{ChunkStore, FlatVectorIndex};

const DIM: usize = 384;
const SAMPLE: usize = 400;
const TOP_K: usize = 3;
const DEFAULT_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const DEFAULT_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";

#[derive(Default, Clone)]
struct Rec {
    n: usize,
    recall: f64,
}
impl Rec {
    fn add(&mut self, r: f64) {
        self.n += 1;
        self.recall += r;
    }
    fn mean(&self) -> f64 {
        self.recall / self.n.max(1) as f64
    }
}

fn tc(s: &str) -> usize {
    s.split_whitespace().count().max(1)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("REDHOP_BGE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let tokenizer =
        std::env::var("REDHOP_BGE_TOKENIZER").unwrap_or_else(|_| DEFAULT_TOKENIZER.into());

    let mut ds = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    ds.examples.truncate(SAMPLE);

    println!("loading BGE-small ONNX...");
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &model,
        &tokenizer,
        EmbedderConfig::bge(DIM),
    )?);

    // First pass: gather per-item paragraph chunks + gold ids + overlap, and
    // collect overlaps to pick the median split point.
    struct ItemData {
        question: String,
        answer: String,
        chunks: Vec<Chunk>,
        gold: HashSet<String>,
        overlap: f32,
    }
    let mut items: Vec<ItemData> = Vec::new();

    for (i, ex) in ds.examples.iter().enumerate() {
        let gold_titles: HashSet<&str> = ex
            .supporting_facts
            .iter()
            .map(|(t, _)| t.as_str())
            .collect();
        let mut chunks = Vec::new();
        let mut gold = HashSet::new();
        let mut gold_text = String::new();
        for (p, (title, sents)) in ex.context.iter().enumerate() {
            let text = sents.join(" ");
            let id = format!("{i}::p{p}");
            if gold_titles.contains(title.as_str()) {
                gold.insert(id.clone());
                gold_text.push_str(&text);
                gold_text.push(' ');
            }
            chunks.push(Chunk::new(
                ChunkId::new(id),
                &text,
                "hotpot",
                TokenCount(tc(&text)),
            ));
        }
        if gold.is_empty() || chunks.len() < 3 {
            continue;
        }
        let overlap = grounding_score(&ex.question, &gold_text);
        items.push(ItemData {
            question: ex.question.clone(),
            answer: ex.answer.clone(),
            chunks,
            gold,
            overlap,
        });
    }

    // Median overlap → split semantic-heavy (low) vs lexical-friendly (high).
    let mut ovs: Vec<f32> = items.iter().map(|it| it.overlap).collect();
    ovs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ovs[ovs.len() / 2];
    println!(
        "items: {}  median query↔gold overlap: {:.3}  (split point)\n",
        items.len(),
        median
    );

    // Embed all chunk texts + queries in modest batches (one giant ONNX batch
    // OOMs — pad-to-max blows up at thousands of sequences).
    async fn embed_batched(
        bge: &Arc<dyn EmbeddingProvider>,
        texts: &[String],
    ) -> anyhow::Result<Vec<redhop::core::Embedding>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(64) {
            out.extend(bge.embed(batch).await?);
        }
        Ok(out)
    }
    let mut all_texts: Vec<String> = Vec::new();
    for it in &items {
        all_texts.extend(it.chunks.iter().map(|c| c.text.clone()));
    }
    let chunk_vecs = embed_batched(&bge, &all_texts).await?;
    let q_vecs = embed_batched(
        &bge,
        &items
            .iter()
            .map(|it| it.question.clone())
            .collect::<Vec<_>>(),
    )
    .await?;

    // Per-(subset, mode) recall, BM25 score-signal rows, and Tier-3 emit.
    let modes = ["bm25", "dense", "hybrid"];
    let mut stats: std::collections::HashMap<(&str, &str), Rec> = std::collections::HashMap::new();
    let mut emit = String::new();
    let mut sig = String::new();

    // One record per item for the escalation-trigger sweeps.
    struct Row {
        subset: &'static str,
        bm25_r: f64,
        dense_r: f64,
        margin: f32,  // (top1-top2)/top1: 1 = confident, 0 = ambiguous
        entropy: f32, // normalized entropy of top-k BM25 scores: high = flat/uncertain
    }
    let mut rows: Vec<Row> = Vec::new();

    let mut off = 0usize;
    for (qi, it) in items.iter().enumerate() {
        let n = it.chunks.len();
        let mut chunks = it.chunks.clone();
        for (c, v) in chunks.iter_mut().zip(&chunk_vecs[off..off + n]) {
            c.embedding = Some(v.clone());
        }
        off += n;
        let subset = if it.overlap <= median {
            "semantic"
        } else {
            "lexical"
        };

        let mut bm25 = Bm25Retriever::new()?;
        Retriever::index(&mut bm25, &chunks).await?;
        let bm25: Arc<dyn Retriever> = Arc::new(bm25);
        let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
        let mut dr = DenseRetriever::new(index, Arc::new(ChunkStore::new()));
        dr.index(&chunks).await?;
        let dense: Arc<dyn Retriever> = Arc::new(dr);
        let hybrid: Arc<dyn Retriever> =
            Arc::new(HybridRetriever::rrf(vec![bm25.clone(), dense.clone()], n));

        let q = Query::new(&it.question).with_embedding(q_vecs[qi].clone());
        let recall_of = |res: &[redhop::core::RetrievalResult]| -> f64 {
            let got = res
                .iter()
                .take(TOP_K)
                .filter(|r| it.gold.contains(r.chunk.id.as_str()))
                .count();
            got as f64 / it.gold.len() as f64
        };

        let mut ctx_by_mode = std::collections::HashMap::new();
        let mut recalls: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        let mut bm25_scores: Vec<f32> = Vec::new();
        for mode in modes {
            let retr = match mode {
                "bm25" => &bm25,
                "dense" => &dense,
                _ => &hybrid,
            };
            let res = retr.retrieve(&q, TOP_K).await?;
            let r = recall_of(&res);
            recalls.insert(mode, r);
            stats.entry((subset, mode)).or_default().add(r);
            stats.entry(("ALL", mode)).or_default().add(r);
            if mode == "bm25" {
                bm25_scores = res.iter().map(|x| x.score.value).collect();
            }
            ctx_by_mode.insert(
                mode,
                res.iter()
                    .take(TOP_K)
                    .map(|r| r.chunk.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            );
        }

        // BM25 score-distribution signals — the candidate escalation triggers.
        let top1 = bm25_scores.first().copied().unwrap_or(0.0);
        let top2 = bm25_scores.get(1).copied().unwrap_or(0.0);
        let margin = if top1 > 0.0 {
            (top1 - top2) / top1
        } else {
            0.0
        };
        let ksum: f32 = bm25_scores.iter().take(TOP_K).sum();
        let kn = bm25_scores.iter().take(TOP_K).count().max(1) as f32;
        let entropy = if ksum > 0.0 && kn > 1.0 {
            let h: f32 = bm25_scores
                .iter()
                .take(TOP_K)
                .map(|s| {
                    let p = s / ksum;
                    if p > 0.0 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum();
            h / kn.ln()
        } else {
            0.0
        };
        rows.push(Row {
            subset,
            bm25_r: recalls["bm25"],
            dense_r: recalls["dense"],
            margin,
            entropy,
        });

        sig.push_str(&serde_json::to_string(&serde_json::json!({
            "id": qi, "subset": subset, "margin": margin, "entropy": entropy,
            "bm25_recall": recalls["bm25"], "dense_recall": recalls["dense"],
        }))?);
        sig.push('\n');
        for mode in modes {
            emit.push_str(&serde_json::to_string(&serde_json::json!({
                "id": qi, "subset": subset, "mode": mode,
                "question": it.question, "gold_answer": it.answer,
                "context": ctx_by_mode[mode],
            }))?);
            emit.push('\n');
        }
    }

    // ── Tier-1 table ──
    println!("Gold-paragraph recall@{TOP_K} by subset and mode");
    println!(
        "  {:<22} {:>8} {:>8} {:>8}",
        "subset", "bm25", "dense", "hybrid"
    );
    println!("  {}", "─".repeat(50));
    for subset in ["lexical", "semantic", "ALL"] {
        let g = |m: &str| stats.get(&(subset, m)).map(|r| r.mean()).unwrap_or(0.0);
        let nn = stats.get(&(subset, "bm25")).map(|r| r.n).unwrap_or(0);
        println!(
            "  {:<14} (n={:>4}) {:>8.2} {:>8.2} {:>8.2}",
            subset,
            nn,
            g("bm25"),
            g("dense"),
            g("hybrid")
        );
    }

    // ── The crux: does BM25 uncertainty separate the subsets? ──
    let smean = |sub: &str, sel: fn(&Row) -> f32| -> f32 {
        let v: Vec<f32> = rows
            .iter()
            .filter(|r| sub == "ALL" || r.subset == sub)
            .map(sel)
            .collect();
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    println!("\nDoes BM25 uncertainty separate the subsets? (mean signal)");
    println!("  {:<24} {:>10} {:>10}", "signal", "lexical", "semantic");
    println!(
        "  {:<24} {:>10.3} {:>10.3}",
        "margin (1=confident)",
        smean("lexical", |r| r.margin),
        smean("semantic", |r| r.margin)
    );
    println!(
        "  {:<24} {:>10.3} {:>10.3}",
        "entropy (high=flat)",
        smean("lexical", |r| r.entropy),
        smean("semantic", |r| r.entropy)
    );

    // ── Trigger sweeps: escalate to dense when BM25 looks uncertain ──
    let n_sem = rows
        .iter()
        .filter(|r| r.subset == "semantic")
        .count()
        .max(1) as f64;
    let n_lex = rows.iter().filter(|r| r.subset == "lexical").count().max(1) as f64;
    let total = rows.len() as f64;
    let sweep = |name: &str, taus: &[f32], esc: fn(&Row, f32) -> bool| {
        println!("\nTrigger: {name}");
        println!(
            "  {:<8} {:>9} {:>12} {:>12} {:>9}",
            "τ", "escal.%", "sem.capt.%", "lex.false%", "recall"
        );
        for &tau in taus {
            let (mut esc_n, mut sem_c, mut lex_f, mut rec) = (0.0, 0.0, 0.0, 0.0);
            for r in &rows {
                let e = esc(r, tau);
                if e {
                    esc_n += 1.0;
                    if r.subset == "semantic" {
                        sem_c += 1.0;
                    } else {
                        lex_f += 1.0;
                    }
                }
                rec += if e { r.dense_r } else { r.bm25_r };
            }
            println!(
                "  {:<8.2} {:>8.0}% {:>11.0}% {:>11.0}% {:>9.2}",
                tau,
                100.0 * esc_n / total,
                100.0 * sem_c / n_sem,
                100.0 * lex_f / n_lex,
                rec / total
            );
        }
    };
    sweep(
        "low margin → escalate (margin < τ)",
        &[0.10, 0.20, 0.30, 0.50],
        |r, t| r.margin < t,
    );
    sweep(
        "high entropy → escalate (entropy > τ)",
        &[0.50, 0.70, 0.85],
        |r, t| r.entropy > t,
    );
    println!(
        "\n  baselines: always-bm25 {:.2} / always-dense {:.2} (escalate 100%)",
        stats.get(&("ALL", "bm25")).unwrap().mean(),
        stats.get(&("ALL", "dense")).unwrap().mean()
    );

    let dir = redhop_examples::exports_path("semantic_natural_contexts.jsonl");
    std::fs::create_dir_all(dir.parent().unwrap())?;
    std::fs::write(&dir, emit)?;
    std::fs::write(
        redhop_examples::exports_path("semantic_natural_signals.jsonl"),
        sig,
    )?;
    println!("\nTier-3 contexts + signals → exports/semantic_natural_*.jsonl");
    Ok(())
}
