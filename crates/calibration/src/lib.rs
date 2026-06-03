//! # redhop-calibration
//!
//! Calibration and evaluation harness for RedHop adaptive retrieval.
//!
//! ## What this crate is for
//!
//! The calibration substrate proved that the adaptive controller **can** improve hard
//! regimes and stay neutral on easy ones. It did not answer the
//! follow-up question that actually matters in production:
//!
//! > **When should the controller intervene?**
//!
//! That is a calibration question, not an architectural one. This crate
//! provides the harness for answering it on a real workload.
//!
//! ## What's inside
//!
//! - [`dataset::LabeledQuery`] / [`dataset::LabeledCorpus`] — the data
//!   shape: queries with ground-truth regime labels and gold chunk ids.
//! - [`runner::run_query`] — runs a single labeled query through both
//!   static and adaptive pipelines side-by-side and returns a
//!   [`runner::QueryOutcome`] with per-query metrics.
//! - [`sweep::ThresholdSweep`] — sweeps the policy threshold grid,
//!   producing a [`sweep::SweepReport`] with per-setting aggregates.
//! - [`reliability::reliability_diagram`] — buckets queries by
//!   `p(predicted regime)` and reports the empirical fraction where the
//!   true regime matched. The classic calibration check.
//! - [`fixtures::synthetic_dataset`] — a hand-curated synthetic corpus
//!   for demonstrating the harness. Real users supply their own
//!   [`dataset::LabeledCorpus`] from HotpotQA / judge-model traces.
//! - [`report`] — ASCII pretty-printers; no external plotting deps.
//!
//! ## What this crate is NOT
//!
//! - Not a benchmark harness in the criterion sense. Criterion measures
//!   wall-clock; this crate measures *evidence quality* and
//!   *intervention utility*.
//! - Not a learned-policy training loop. A future iteration may build one on top of
//!   the [`runner::QueryOutcome`] traces this crate emits, but no
//!   training happens here.
//! - Not a replacement for empirical evaluation against real workloads.
//!   The synthetic fixtures here exist to demonstrate the methodology;
//!   the headline calibration numbers come from your own data.
//!
//! ## Methodology
//!
//! ```text
//!   LabeledCorpus
//!         │
//!         ▼
//!   for each LabeledQuery q in corpus:
//!     static_result   ← static_retrieve(q)        // baseline
//!     adaptive_result ← adaptive_run(q)           // closed loop
//!     metrics ← compare(static, adaptive, q.gold) // utility, cost, etc.
//!         │
//!         ▼
//!   ThresholdSweep over min_p_distractor, min_p_ambiguous, …
//!         │
//!         ▼
//!   SweepReport:
//!     intervention_rate, recall_lift, latency_overhead per threshold
//!         │
//!         ▼
//!   Headline answer: at which threshold does adaptive Pareto-dominate static?
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod analysis;
pub mod corruption;
pub mod dataset;
pub mod economics;
pub mod embedder;
pub mod embedder_bench;
pub mod fixtures;
pub mod htmlreport;
pub mod loaders;
pub mod reliability;
pub mod report;
pub mod runner;
pub mod sweep;

pub use dataset::{LabeledCorpus, LabeledQuery};
pub use reliability::{reliability_diagram, ReliabilityBin, ReliabilityDiagram};
pub use runner::{run_query, QueryOutcome};
pub use sweep::{SweepReport, SweepRow, ThresholdSweep};
