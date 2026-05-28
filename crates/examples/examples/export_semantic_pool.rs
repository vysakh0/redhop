//! Export the EXACT global HotpotQA pool used by `semantic_local_rerank` into
//! BEIR format, so `semantic-bm25` (dependency-free graph expansion) can be
//! evaluated on the same corpus / queries / gold / overlap-split — apples to
//! apples against bm25 / global dense / local rerank.
//!
//! Emits three BEIR dirs under exports/semantic_pool/{all,lexical,semantic}/,
//! each sharing the same corpus.jsonl but with the subset's queries + qrels.
//! Run:  cargo run -p redhop-examples --example export_semantic_pool --release

use std::collections::{HashMap, HashSet};
use std::fs;

use redhop_calibration::loaders::hotpotqa::HotpotQADataset;
use redhop::context::grounding_score;

const SAMPLE: usize = 400; // identical to semantic_local_rerank

fn jstr(s: &str) -> String {
    // minimal JSON string escaping for jsonl emission
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

fn main() -> anyhow::Result<()> {
    let mut ds = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    ds.examples.truncate(SAMPLE);

    // Global corpus: dedupe paragraphs by title across all items (== the study).
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

    struct QItem {
        question: String,
        gold: HashSet<usize>,
        overlap: f32,
    }
    let mut qs: Vec<QItem> = Vec::new();
    for ex in &ds.examples {
        let gold: HashSet<usize> = ex
            .supporting_facts
            .iter()
            .filter_map(|(t, _)| title_id.get(t.as_str()).copied())
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

    // corpus.jsonl content (shared by all three dirs).
    let mut corpus = String::new();
    for (i, t) in titles.iter().enumerate() {
        let text = &by_title[t];
        // Title left empty on purpose: the local-rerank study indexed paragraph
        // TEXT ONLY (no title), so for an apples-to-apples comparison we must not
        // let the BEIR loader prepend the title (it leaks the answer entity).
        let _ = t;
        corpus.push_str(&format!(
            "{{\"_id\": {}, \"title\": \"\", \"text\": {}}}\n",
            jstr(&i.to_string()),
            jstr(text)
        ));
    }

    let base = redhop_examples::exports_path("semantic_pool");
    for subset in ["all", "lexical", "semantic"] {
        let dir = base.join(subset);
        fs::create_dir_all(dir.join("qrels"))?;
        fs::write(dir.join("corpus.jsonl"), &corpus)?;

        let mut queries = String::new();
        let mut qrels = String::new();
        let mut n = 0usize;
        for (qi, q) in qs.iter().enumerate() {
            let is_sem = q.overlap <= median;
            let keep = match subset {
                "lexical" => !is_sem,
                "semantic" => is_sem,
                _ => true,
            };
            if !keep {
                continue;
            }
            n += 1;
            let qid = format!("q{qi}");
            queries.push_str(&format!(
                "{{\"_id\": {}, \"text\": {}}}\n",
                jstr(&qid),
                jstr(&q.question)
            ));
            for &g in &q.gold {
                qrels.push_str(&format!("{}\t{}\t1\n", qid, g));
            }
        }
        fs::write(dir.join("queries.jsonl"), &queries)?;
        fs::write(dir.join("qrels/test.tsv"), &qrels)?;
        println!("{subset:<9} {n:>4} queries -> {}", dir.display());
    }

    println!(
        "\ncorpus: {} unique paragraphs; {} queries total; median overlap {:.3}",
        titles.len(),
        qs.len(),
        median
    );
    Ok(())
}
