//! Can dense local-rerank climb past its ~0.80 recall@3 plateau using only what
//! we have (no bigger model, no agentic multi-hop)? Hypothesis: the residual miss
//! is the **second hop** — the bridge passage dense demotes because it is not
//! query-relevant (the documented second-hop tax). The on-thesis fix is
//! reasoning-aware rescue: keep dense's reliable top hit as the seed, then promote
//! pool candidates *linked* to that seed (the same `link_strength` Jaccard the
//! ReasoningPreserving strategy uses) so the bridge can land in the top-3.
//!
//! Arms (all over the SAME BM25 candidate pool, on the global HotpotQA corpus).
//! `dense`: dense cosine top-3 (the 0.80 baseline / local-rerank arm).
//! `dense+rescue(β)`: slot-1 = dense top-1; rank the rest by
//! `dense_cos + β·link_strength(seed, candidate)`, take top-3 (β=0 == dense).
//! Tier-1 only (recall@3), split lexical vs semantic, plus a second-hop recovery
//! diagnostic: of queries where a gold was in the pool but missed by dense@3, how
//! often does rescue recover it (and how often does it hurt a dense hit)?
//!
//! Run:  cargo run -p redhop-examples --example semantic_reasoning_rerank --features onnx --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redhop_calibration::loaders::hotpotqa::HotpotQADataset;
use redhop_context::{grounding_score, link_strength};
use redhop_core::{Chunk, ChunkId, Embedding, EmbeddingProvider, Query, Retriever, TokenCount};
use redhop_embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop_retrieval::Bm25Retriever;

const DIM: usize = 384;
const SAMPLE: usize = 400;
const K_CAND: usize = 50;
const TOP_K: usize = 3;
const BETAS: [f32; 6] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0];
const DEFAULT_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const DEFAULT_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";

fn cosine(a: &Embedding, b: &Embedding) -> f32 {
    let (x, y) = (a.as_slice(), b.as_slice());
    let n = x.len().min(y.len());
    let (mut d, mut nx, mut ny) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        d += x[i] * y[i];
        nx += x[i] * x[i];
        ny += y[i] * y[i];
    }
    d / (nx.sqrt() * ny.sqrt()).max(1e-9)
}

async fn embed_batched(
    bge: &Arc<dyn EmbeddingProvider>,
    texts: &[String],
) -> anyhow::Result<Vec<Embedding>> {
    let mut out = Vec::with_capacity(texts.len());
    for batch in texts.chunks(64) {
        out.extend(bge.embed(batch).await?);
    }
    Ok(out)
}

#[derive(Default, Clone)]
struct Acc {
    n: usize,
    sum: f64,
}
impl Acc {
    fn add(&mut self, v: f64) {
        self.n += 1;
        self.sum += v;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }
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

    // Global corpus: dedupe paragraphs by title (identical to semantic_local_rerank).
    let mut by_title: HashMap<String, String> = HashMap::new();
    for ex in &ds.examples {
        for (title, sents) in &ex.context {
            by_title
                .entry(title.clone())
                .or_insert_with(|| sents.join(" "));
        }
    }
    let titles: Vec<String> = by_title.keys().cloned().collect();
    let title_id: HashMap<&str, usize> = titles
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    let chunks: Vec<Chunk> = titles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let text = &by_title[t];
            Chunk::new(
                ChunkId::new(format!("c{i}")),
                text,
                "pool",
                TokenCount(text.split_whitespace().count().max(1)),
            )
        })
        .collect();

    struct QItem {
        question: String,
        gold: HashSet<String>,
        overlap: f32,
    }
    let mut qs: Vec<QItem> = Vec::new();
    for ex in &ds.examples {
        let gold: HashSet<String> = ex
            .supporting_facts
            .iter()
            .filter_map(|(t, _)| title_id.get(t.as_str()).map(|i| format!("c{i}")))
            .collect();
        if gold.is_empty() {
            continue;
        }
        let gold_text: String = ex
            .supporting_facts
            .iter()
            .filter_map(|(t, _)| by_title.get(t).cloned())
            .collect::<Vec<_>>()
            .join(" ");
        qs.push(QItem {
            question: ex.question.clone(),
            overlap: grounding_score(&ex.question, &gold_text),
            gold,
        });
    }
    let text_by_id: HashMap<String, String> = chunks
        .iter()
        .map(|c| (c.id.as_str().to_string(), c.text.clone()))
        .collect();

    let multi = qs.iter().filter(|q| q.gold.len() >= 2).count();
    println!(
        "global corpus: {} paragraphs; {} queries ({:.0}% have >=2 gold facts); K_cand={K_CAND}, top-{TOP_K}",
        chunks.len(),
        qs.len(),
        100.0 * multi as f64 / qs.len() as f64
    );

    println!("loading BGE-small ONNX + embedding corpus...");
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &model,
        &tokenizer,
        EmbedderConfig::bge(DIM),
    )?);
    let corpus_vecs = embed_batched(
        &bge,
        &chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>(),
    )
    .await?;
    let emb_by_id: HashMap<String, Embedding> = chunks
        .iter()
        .zip(&corpus_vecs)
        .map(|(c, v)| (c.id.as_str().to_string(), v.clone()))
        .collect();
    let q_vecs = embed_batched(
        &bge,
        &qs.iter().map(|q| q.question.clone()).collect::<Vec<_>>(),
    )
    .await?;

    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;

    let median = {
        let mut o: Vec<f32> = qs.iter().map(|q| q.overlap).collect();
        o.sort_by(|a, b| a.partial_cmp(b).unwrap());
        o[o.len() / 2]
    };

    // (subset, arm) -> recall@3.  arm = "dense" or "rescue@β".
    let mut rec: HashMap<(&str, String), Acc> = HashMap::new();
    // second-hop recovery: queries with a gold in the pool but missed by dense@3.
    let (mut miss_total, mut miss_recovered_best, mut hurt_best) = (0usize, 0usize, 0usize);

    for (qi, q) in qs.iter().enumerate() {
        let subset = if q.overlap <= median {
            "semantic"
        } else {
            "lexical"
        };
        let qv = &q_vecs[qi];
        let query = Query::new(&q.question).with_embedding(qv.clone());
        let cand = bm25.retrieve(&query, K_CAND).await?;
        if cand.is_empty() {
            continue;
        }

        // Dense cosine over the pool.
        let mut dense: Vec<(String, f32)> = cand
            .iter()
            .map(|r| {
                let id = r.chunk.id.as_str().to_string();
                let cs = cosine(qv, &emb_by_id[&id]);
                (id, cs)
            })
            .collect();
        dense.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let recall3 = |ids: &[String]| -> f64 {
            let hit = ids
                .iter()
                .take(TOP_K)
                .filter(|id| q.gold.contains(*id))
                .count();
            hit as f64 / q.gold.len() as f64
        };

        let dense_ids: Vec<String> = dense.iter().take(TOP_K).map(|(id, _)| id.clone()).collect();
        let dense_r3 = recall3(&dense_ids);
        rec.entry((subset, "dense".into()))
            .or_default()
            .add(dense_r3);
        rec.entry(("ALL", "dense".into()))
            .or_default()
            .add(dense_r3);

        // Is a gold present in the pool but missed by dense@3? (the recoverable set)
        let gold_in_pool = dense.iter().any(|(id, _)| q.gold.contains(id));
        let recoverable = gold_in_pool && dense_r3 < 1.0;
        if recoverable {
            miss_total += 1;
        }
        let mut best_rescue_r3 = dense_r3;

        // dense + linkage rescue: seed = dense top-1; rank the rest by
        // dense_cos + β·link_strength(seed, candidate). β=0 == dense.
        let seed_id = dense[0].0.clone();
        let seed_text = &text_by_id[&seed_id];
        // min-max normalize dense over the pool so β is on a comparable scale.
        let (dlo, dhi) = dense.iter().fold((f32::MAX, f32::MIN), |(lo, hi), (_, s)| {
            (lo.min(*s), hi.max(*s))
        });
        let dspan = (dhi - dlo).max(1e-6);
        for &beta in &BETAS {
            let mut rest: Vec<(String, f32)> = dense
                .iter()
                .skip(1)
                .map(|(id, cs)| {
                    let dn = (cs - dlo) / dspan;
                    let link = link_strength(seed_text, &text_by_id[id]);
                    (id.clone(), dn + beta * link)
                })
                .collect();
            rest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut ids = vec![seed_id.clone()];
            ids.extend(rest.iter().take(TOP_K - 1).map(|(id, _)| id.clone()));
            let r3 = recall3(&ids);
            let arm = format!("rescue@{beta}");
            rec.entry((subset, arm.clone())).or_default().add(r3);
            rec.entry(("ALL", arm)).or_default().add(r3);
            if beta > 0.0 {
                best_rescue_r3 = best_rescue_r3.max(r3);
            }
        }
        if recoverable {
            if best_rescue_r3 > dense_r3 {
                miss_recovered_best += 1;
            } else if best_rescue_r3 < dense_r3 {
                hurt_best += 1;
            }
        }
    }

    // ── Table ──
    let arms: Vec<String> = std::iter::once("dense".to_string())
        .chain(BETAS.iter().map(|b| format!("rescue@{b}")))
        .collect();
    println!("\nRecall@{TOP_K} by subset and arm (median overlap split {median:.3})");
    print!("  {:<22}", "arm");
    for s in ["lexical", "semantic", "ALL"] {
        print!("{s:>12}");
    }
    println!();
    println!("  {}", "─".repeat(58));
    for arm in &arms {
        print!("  {arm:<22}");
        for s in ["lexical", "semantic", "ALL"] {
            let v = rec.get(&(s, arm.clone())).map(|x| x.mean()).unwrap_or(0.0);
            print!("{v:>12.3}");
        }
        println!();
    }

    println!("\nSecond-hop recovery (queries with a gold in the pool but missed by dense@{TOP_K}): n={miss_total}");
    if miss_total > 0 {
        println!(
            "  rescue recovered (best β): {} ({:.0}%)   rescue hurt: {} ({:.0}%)   net: {:+}",
            miss_recovered_best,
            100.0 * miss_recovered_best as f64 / miss_total as f64,
            hurt_best,
            100.0 * hurt_best as f64 / miss_total as f64,
            miss_recovered_best as i64 - hurt_best as i64,
        );
        println!("  (best-β is an oracle upper bound on what linkage rescue can add; see per-arm table for a FIXED β.)");
    }
    Ok(())
}
