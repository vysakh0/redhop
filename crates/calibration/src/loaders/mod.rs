//! Loaders for real-world QA datasets.
//!
//! Each loader turns a workload-specific file format into a RedHop
//! [`LabeledCorpus`][lc]. The conversion is faithful but not lossy in
//! the regime label — the loaders apply a default heuristic mapping
//! from each dataset's native labels to [`RetrievalRegime`][rr] which
//! callers can override.
//!
//! ## What's in scope
//!
//! - [`hotpotqa`] — the canonical HotpotQA JSON format
//!   (`[{_id, question, answer, type, level, supporting_facts, context}]`).
//! - [`musique`] — MuSiQue JSON (multi-hop variants), supporting both
//!   the answerable and unanswerable subsets.
//! - [`jsonl`] — a generic JSONL escape hatch for custom workloads
//!   that don't fit either canonical shape.
//!
//! ## What's deliberately NOT in scope
//!
//! - PDF / OCR parsing. PDF extraction is a separate engineering
//!   problem and would pull heavyweight dependencies into the
//!   calibration crate. Users supply text + queries; we only handle
//!   the LabeledCorpus assembly.
//! - Splitting train/dev/test. Calibration runs against whichever
//!   subset the user loads. Downstream analyses are pure functions of
//!   the loaded set.
//! - Dataset download. We do not fetch from the internet. The user
//!   points the loader at a local file they already have.
//!
//! [lc]: crate::dataset::LabeledCorpus
//! [rr]: redhop_core::RetrievalRegime

pub mod hotpotqa;
pub mod jsonl;
pub mod musique;
pub mod neotrace;
