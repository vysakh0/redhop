//! Export the BM25 candidate pool (top-K_cand per query) over the global HotpotQA
//! corpus, so an external reranker (e.g. static embeddings via Model2Vec, tested in
//! Python) can reorder the SAME pool the BGE dense local-rerank used — making any
//! static-embedding result directly comparable to the BGE recall@3 ≈ 0.80 baseline.
//!
//! No ONNX: BM25 only. Same corpus / gold / overlap-split as semantic_local_rerank.
//! Emits exports/rerank_pool.jsonl, one query per line:
//!   {"qid","subset","question","gold":[ids],"candidates":[{"id","text"}, ...]}
//!
//! Run:  cargo run -p redhop-examples --example export_rerank_pool --release

use std::collections::{HashMap, HashSet};
use std::fs;

use redhop_calibration::loaders::hotpotqa::HotpotQADataset;
use redhop::context::grounding_score;
use redhop::core::{Chunk, ChunkId, Query, Retriever, TokenCount};
use redhop::retrieval::Bm25Retriever;

const SAMPLE: usize = 400;
const K_CAND: usize = 50;

fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut ds = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    ds.examples.truncate(SAMPLE);

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

    let median = {
        let mut o: Vec<f32> = qs.iter().map(|q| q.overlap).collect();
        o.sort_by(|a, b| a.partial_cmp(b).unwrap());
        o[o.len() / 2]
    };

    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;

    let mut out = String::new();
    for (qi, q) in qs.iter().enumerate() {
        let subset = if q.overlap <= median {
            "semantic"
        } else {
            "lexical"
        };
        let query = Query::new(&q.question);
        let cand = bm25.retrieve(&query, K_CAND).await?;

        let mut line = String::new();
        line.push_str(&format!(
            "{{\"qid\": {}, \"subset\": {}, \"question\": {}, \"gold\": [",
            jstr(&format!("q{qi}")),
            jstr(subset),
            jstr(&q.question)
        ));
        let golds: Vec<String> = q.gold.iter().map(|g| jstr(g)).collect();
        line.push_str(&golds.join(", "));
        line.push_str("], \"candidates\": [");
        let cands: Vec<String> = cand
            .iter()
            .map(|r| {
                format!(
                    "{{\"id\": {}, \"text\": {}}}",
                    jstr(r.chunk.id.as_str()),
                    jstr(&r.chunk.text)
                )
            })
            .collect();
        line.push_str(&cands.join(", "));
        line.push_str("]}\n");
        out.push_str(&line);
    }

    let path = redhop_examples::exports_path("rerank_pool.jsonl");
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(&path, out)?;
    println!(
        "wrote {} queries (K_cand={K_CAND}) over {} paragraphs; median overlap {:.3} -> {}",
        qs.len(),
        chunks.len(),
        median,
        path.display()
    );
    Ok(())
}
