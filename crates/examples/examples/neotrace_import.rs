// Quick end-to-end check: load hotpot_full.neotrace.jsonl and run the
// existing analysis utilities against it.
use redhop_calibration::analysis::{confusion_matrix, regret_summary};
use redhop_calibration::loaders::neotrace::{
    load_corpus, load_outcomes, parse_path, NeoTraceRecord,
};

fn main() {
    let path = redhop_examples::exports_path("neotrace/hotpot_full.neotrace.jsonl");
    let records = parse_path(&path).expect("parse");
    println!("loaded {} records", records.len());

    let corpus =
        load_corpus::<fn(&NeoTraceRecord) -> Option<redhop::core::RetrievalRegime>>(&records, None)
            .unwrap();
    println!(
        "corpus: {} unique queries, {} docs",
        corpus.queries.len(),
        corpus.docs.len()
    );

    // Pair cosine (static) vs cross_encoder (adaptive).
    let outcomes = load_outcomes(&records, "cosine", Some("cross_encoder")).unwrap();
    println!("outcomes: {}", outcomes.len());

    let cm = confusion_matrix(&outcomes);
    println!("regime accuracy: not applicable (no predicted_regime in NeoTrace records yet)");
    println!(
        "  n_predicted = {}, n_unpredicted = {}",
        cm.n_predicted, cm.n_unpredicted
    );

    let r = regret_summary(&outcomes);
    println!("interventions: {}", r.n_interventions);
    println!("mean useful lift: {:+.3}", r.mean_useful_lift);
    println!("mean harmful lift: {:+.3}", r.mean_harmful_lift);
    println!("wasted interventions: {}", r.n_wasted_interventions);
}
