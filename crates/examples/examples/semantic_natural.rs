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
//! Tier-1 (here, free): gold-paragraph recall@K per subset per mode + a
//! confidence-gated escalation probe (use BM25 unless its top hit has low
//! lexical overlap with the query → escalate to dense). Tier-3 contexts are
//! emitted to exports/ for downstream LLM answer scoring.
//!
//! Run:  cargo run -p redhop-examples --example semantic_natural --features onnx --release

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::RwLock;
use redhop_calibration::loaders::hotpotqa::HotpotQADataset;
use redhop_context::grounding_score;
use redhop_core::{Chunk, ChunkId, EmbeddingProvider, Query, Retriever, TokenCount, VectorIndex};
use redhop_embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop_retrieval::{Bm25Retriever, DenseRetriever, HybridRetriever};
use redhop_storage::{ChunkStore, FlatVectorIndex};

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
    ) -> anyhow::Result<Vec<redhop_core::Embedding>> {
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

    // Per-(subset, mode) recall + the gate probe + Tier-3 emit.
    let modes = ["bm25", "dense", "hybrid"];
    let mut stats: std::collections::HashMap<(&str, &str), Rec> = std::collections::HashMap::new();
    // Gate: BM25 unless its top hit has query overlap < τ → escalate to dense.
    let gate_taus = [0.10f32, 0.20, 0.30];
    let mut gate: Vec<(Rec, usize)> = gate_taus.iter().map(|_| (Rec::default(), 0usize)).collect();
    let mut emit = String::new();

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
        let recall_of = |res: &[redhop_core::RetrievalResult]| -> f64 {
            let got = res
                .iter()
                .take(TOP_K)
                .filter(|r| it.gold.contains(r.chunk.id.as_str()))
                .count();
            got as f64 / it.gold.len() as f64
        };

        let mut ctx_by_mode = std::collections::HashMap::new();
        let mut bm25_top_overlap = 0.0f32;
        for mode in modes {
            let retr = match mode {
                "bm25" => &bm25,
                "dense" => &dense,
                _ => &hybrid,
            };
            let res = retr.retrieve(&q, TOP_K).await?;
            stats
                .entry((subset, mode))
                .or_default()
                .add(recall_of(&res));
            stats.entry(("ALL", mode)).or_default().add(recall_of(&res));
            if mode == "bm25" {
                if let Some(top) = res.first() {
                    bm25_top_overlap = grounding_score(&it.question, &top.chunk.text);
                }
            }
            let ctx = res
                .iter()
                .take(TOP_K)
                .map(|r| r.chunk.text.clone())
                .collect::<Vec<_>>()
                .join("\n\n");
            ctx_by_mode.insert(mode, ctx);
        }

        // Gate probe: low BM25 top-hit overlap ⇒ escalate to dense.
        for (gi, &tau) in gate_taus.iter().enumerate() {
            let escalate = bm25_top_overlap < tau;
            let res = if escalate {
                dense.retrieve(&q, TOP_K).await?
            } else {
                bm25.retrieve(&q, TOP_K).await?
            };
            gate[gi].0.add(recall_of(&res));
            if escalate {
                gate[gi].1 += 1;
            }
        }

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

    println!("\nConfidence-gated escalation (BM25 → dense when top-hit overlap < τ)");
    println!("  {:<10} {:>10} {:>14}", "τ", "recall", "escalated%");
    let total = items.len() as f64;
    for (gi, &tau) in gate_taus.iter().enumerate() {
        println!(
            "  {:<10.2} {:>10.2} {:>13.0}%",
            tau,
            gate[gi].0.mean(),
            100.0 * gate[gi].1 as f64 / total
        );
    }
    println!(
        "  (compare to always-bm25 {:.2} / always-dense {:.2})",
        stats.get(&("ALL", "bm25")).unwrap().mean(),
        stats.get(&("ALL", "dense")).unwrap().mean()
    );

    let out = redhop_examples::exports_path("semantic_natural_contexts.jsonl");
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, emit)?;
    println!("\nTier-3 contexts → {}", out.display());
    Ok(())
}
