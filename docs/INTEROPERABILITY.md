# NeoRAG Python ↔ Rust Interoperability

**Audience:** anyone touching the Python research repo at
`../neorag` or the Rust calibration engine at this repository.

## TL;DR

The Python repo is the **experimentation lab**. The Rust repo is the
**execution engine + calibration runtime**. They communicate through
**NeoTrace JSONL** — one canonical format, one exporter on the Python
side, one loader on the Rust side. No tighter coupling.

Right now, **5,190 records** of post-pivot experimental data are
already exportable. That's the calibration corpus available
*immediately* — without re-running anything, without writing any new
Python.

## Layer responsibilities

| Concern                                    | Lives in       | Why                                                                          |
| ------------------------------------------ | -------------- | ---------------------------------------------------------------------------- |
| LLM provider glue (Anthropic, HF, mock)    | **Python**     | mature, prompt-caching aware, judge invocation                               |
| PDF / OCR ingestion                        | **Python**     | PyMuPDF + pdfplumber; Rust shouldn't pull in PDF deps                        |
| Dataset downloads, fixture generation      | **Python**     | one-off scripts; not a runtime concern                                       |
| Stats helpers (Wilcoxon, bootstrap, paired-t) | **Python**  | scipy is already used; no point reimplementing                               |
| Visualisation (pyvis, plotly)              | **Python**     | research-only                                                                |
| Retrieval execution (BM25, dense, hybrid)  | **Rust**       | the production runtime path                                                  |
| Adaptive orchestration                     | **Rust**       | closed-loop controller, conservative policy, action traces                   |
| Calibration analyses                       | **Rust**       | sweep, regret, confusion, bootstrap stability                                |
| Production retrievers / production deploy  | **Rust**       | Tantivy + the adaptive controller, eventually with Python / Node FFI         |

This split is deliberate. The Python lab keeps doing what it's good
at: rapid experimentation, dataset curation, judge-model evaluation.
The Rust engine keeps doing what it's good at: deterministic
retrieval, low-overhead instrumentation, embeddable runtime. NeoTrace
is the seam.

## The audit at a glance

A complete inventory lives in this commit's discussion. The headline:

### Class A — directly importable today

5 source families, all exporting cleanly through
`scripts/export_to_neotrace.py`:

| Python source                                    | records | what it carries                                        |
| ------------------------------------------------ | ------- | ------------------------------------------------------ |
| `hotpot_full.json` + `hotpot_llama8b.json` + smoke | 1,407 | HotpotQA dev-distractor on 4 LLMs × 7 retrievers       |
| `musique_full.json` + 3 LLM variants + smoke    | 2,835   | MuSiQue dev on 4 LLMs × 7 retrievers                   |
| `evidence_evidence.json`                         | 130     | in-house multi-hop evidence-quality measurements        |
| `learned_*.json` (4 files)                       | 650     | learned-scorer LOO-doc CV folds                         |
| `judge_multihop.json`                            | 104     | Sonnet-as-judge pairwise verdicts                       |
| `endtoend.json`                                  | 64      | falsification control (TinyLlama, cosine vs trajectory) |

**Total: 5,190 records.** All present, all post-pivot.

### Class B — reusable with adaptation

Code we may want to consult but not run directly:

- `neorag/analysis/answerability.py` — the learned scorer. Its
  coefficients are already serialised into every Class-A trace as
  `scorer_info.coefs`; Rust can either consume them as-is for offline
  reproduction or use them as a regression target.
- `neorag/analysis/evidence.py` — defines the evidence-quality column
  set (`answer_span_density`, `distractor_ratio`, `purity`, …) that
  the canonical schema standardises.
- `neorag/metrics.py` — answer-similarity / keyword-recall reference
  implementations. Useful as ground truth for future Rust
  reimplementations.

### Class C — topology-era, kept as negative controls

These are pre-pivot. The Python `VALIDATION_REPORT.md` already
documents them as pruned, demoted, or null-verdicted:

- `ablations.json` — topology ablation matrix. `position_dispersion`
  metric pruned as circular.
- `operators_topology.json` — Personalised PageRank (already
  identical-to-RWR per VR §4).
- `scaling.json` — length-scaling on the synthetic doc.
- `validation.json` — the circularity audit itself.
- `experiments.json` — first-round MockLLM experiments.
- `organization_*.json` — organization-doesn't-matter trilogy. All
  three null-verdicted in the Python output already.

**Do not calibrate against these.** They embody falsified hypotheses.
They are kept on disk as historical record.

### Class D — Python-only, stays Python forever

- LLM provider glue, prompt caching, judge invocation.
- PDF / OCR parsing.
- Dataset download + chunking scripts.
- Visualisation.
- Statistics support (`scipy.stats`).

## NeoTrace JSONL — the format

See `docs/NEOTRACE_SCHEMA.md` for the full spec. The short version:

- One JSON object per line, one record per `(item_id, method, model)`
  tuple.
- Field set is a *superset* of every existing Python trace format.
- Optional fields use absent-key semantics; the Rust loader tolerates
  sparsity by design.
- Versioned via `schema_version: "neotrace/1"`. Future major versions
  ship parallel loaders, never silent upgrades.

The canonical fields are:

```text
identity:    schema_version, run_id, item_id, method, model, ce_model
query:       question, gold_answer, gold_para, gold_chunk_ids, doc, type, level, n_chunks
retrieval:   retrieved, retrieved_chunk_ids, retrieval_recall, top_k
evidence:    continuity, answer_span_density, answer_bearing_fraction,
             distractor_ratio, query_overlap, entity_overlap, purity
generation:  answer, ans_similarity, ans_kw_recall
judge:       judge_model, judge_score, judge_preferred, judge_reason
regime:      true_regime, predicted_regime, predicted_regime_p
adaptive:    intervened, action_trace[], latency_ms
escape:      extra (arbitrary JSON, never consumed by core calibration)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Python research lab  (../neorag)                                   │
│                                                                     │
│   experiments/results/*.json                                        │
│         │                                                           │
│         ▼                                                           │
│   scripts/export_to_neotrace.py                                     │
│         │                                                           │
│         ▼                                                           │
│   exports/neotrace/*.neotrace.jsonl   ◄──── canonical interchange   │
│                                       │                             │
└───────────────────────────────────────┼─────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────┼─────────────────────────────┐
│  Rust calibration engine  (.)         │                             │
│                                       │                             │
│   neorag-calibration::loaders::neotrace                             │
│         │                                                           │
│         ├──► load_corpus  → LabeledCorpus                           │
│         │                  │                                        │
│         │                  ▼                                        │
│         │                  ThresholdSweep, reliability_diagram,     │
│         │                  Adaptive orchestrator re-runs            │
│         │                                                           │
│         └──► load_outcomes → Vec<QueryOutcome>                      │
│                              │                                      │
│                              ▼                                      │
│                              confusion_matrix, regret_summary,      │
│                              bootstrap_stability (no rerun needed)  │
└─────────────────────────────────────────────────────────────────────┘
```

`load_corpus` is the right entry point when you want the Rust adaptive
controller to **re-run** retrieval against the same queries. It
extracts the labels.

`load_outcomes` is the right entry point when you want to analyze the
Python lab's measurements **directly** — the analyses run without any
retrieval at all. The Python-side `retrieval_recall` becomes
`gold_recall_static` / `gold_recall_adaptive` depending on which
method you pair against which.

## Migration strategy

There is no migration. The Python repo keeps running. The Rust repo
gains a loader. The interaction is one-way: Python writes JSONL, Rust
reads it.

If a future experiment in Python wants to consume Rust-side adaptive
traces (e.g., for a learned-policy training loop), the Rust orchestrator
emits `action_trace[]` entries in the same NeoTrace format. Python can
read those too. The format is symmetric — but the **discipline is
deliberately asymmetric**: Rust does retrieval and adaptive control;
Python does evaluation and judgment.

## How to use it today

### Python side

```bash
cd ../neorag
python scripts/export_to_neotrace.py
# → ../neorag/exports/neotrace/*.neotrace.jsonl
```

5,190 records exported in under a second. The script is idempotent;
re-run after every new experiment.

### Rust side

```rust
use neorag_calibration::loaders::neotrace::{parse_path, load_corpus, load_outcomes};

let records = parse_path("../neorag/exports/neotrace/hotpot_full.neotrace.jsonl")?;

// Option 1: re-run retrieval through the adaptive controller.
let corpus = load_corpus::<fn(_) -> _>(&records, None)?;
// → feed into ThresholdSweep::run, NeoRAG::adaptive_run, etc.

// Option 2: consume Python's measurements directly.
let outcomes = load_outcomes(&records, "cosine", Some("cross_encoder"))?;
// → feed into confusion_matrix, regret_summary, bootstrap_stability
```

The `crates/examples/examples/neotrace_import.rs` example runs
end-to-end against the real HotpotQA export today.

## Immediate experiments unlocked by reuse

Every item below is *available now*, no new code required beyond what
this commit ships:

1. **Cross-LLM regret analysis.** Compare `cosine` vs `cross_encoder`
   regret across `haiku`, `llama-8b`, `qwen-7b`, `mistral-nemo` on the
   same MuSiQue queries. The Python lab already ran the retrieval; we
   just compute regret per LLM.

2. **Method-pair Pareto curves.** Pair every method against every
   other method on the HotpotQA `*_full.json` and check which pairs
   produce a Pareto improvement on recall lift vs cost. Seven methods
   → 21 pairs. Single sweep.

3. **Regime-conditioned utility curves.** HotpotQA records carry
   `level` and `type`; map those to regime labels (the exporter
   already does), then compute mean intervention utility per regime.
   Tells us empirically whether `hard+bridge` queries actually benefit
   from escalation more than `easy+comparison`.

4. **Underconfidence calibration check on real data.** The synthetic
   demo flagged a +0.246 ECE. Run the reliability diagram against
   judge-graded multihop records to confirm whether the classifier is
   underconfident on real workloads or only on the synthetic fixture.

5. **Trajectory-as-control sanity check.** Pair every `trajectory`
   record against the matching `cosine` record. Expected outcome:
   recall lift ≈ 0 (the falsification result). Confirms our pipeline
   reproduces the Python verdict.

6. **Cross-encoder ROI study.** For every `(item, method=cosine)`
   record, find the matching `(item, method=cross_encoder)` record.
   Recall lift histogram answers: *on what fraction of HotpotQA
   queries does cross_encoder actually buy gold-chunk recall?* From
   our smoke test today, the answer on HotpotQA full is ≈37%
   improved, ≈63% no change.

These are all `cargo run -p neorag-examples --example
neotrace_import` variations away from being concrete numbers.

## Boundaries we hold

- **The Python repo is not getting rewritten.** The exporter is a
  one-file Python script. It has no Rust counterpart and won't get
  one.
- **The Rust repo is not gaining LLM glue.** Judge invocation,
  generator selection, prompt caching — all stay Python.
- **No PDF parsing in Rust.** If `data/real/*.pdf` needs to be
  ingested, Python pre-chunks once and ships JSON. Anything else is
  feature creep.
- **No retrieval architecture changes in this commit.** This is
  purely about plumbing existing data into existing analyses.

## Honest assessment

What's reusable:
  - All post-pivot HotpotQA + MuSiQue experimental traces (5,190
    records).
  - The learned-scorer coefficients (already inlined into the
    traces).
  - The hand-curated multi-hop / end-to-end gold sets.
  - The raw HotpotQA + MuSiQue dev corpora (46 MB + 30 MB).

What's *not* reusable:
  - The trajectory / graph operators. They embody a falsified
    hypothesis; their numbers go into the bin labelled "negative
    control" and stay there.
  - The PDF ingestion pipeline. Stays Python forever.
  - The pre-pivot ablation matrix. Documented as pruned in the
    validation report itself.

What's a gap, not a flaw:
  - The Python lab has no notion of `predicted_regime` or
    `action_trace` — those are Rust-side concepts. NeoTrace records
    exported from Python carry only `true_regime`. As soon as the Rust
    adaptive controller is run against the same corpus, the
    `predicted_regime` and `action_trace` fields populate naturally.
    The reliability diagram is dark until that happens.

That's the bridge. The architecture stays asymmetric on purpose: the
Rust engine doesn't pretend to be the lab, and the Python lab doesn't
pretend to be production. The seam is one JSONL file family and one
small loader, both versioned, both tested.
