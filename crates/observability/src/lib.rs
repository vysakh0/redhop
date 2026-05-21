//! # neorag-observability
//!
//! Per-query retrieval traces. This crate answers the operator's
//! question: *"why did retrieval behave this way on this query?"*
//!
//! It is a thin, **core-only** layer that converts a finished
//! [`RetrievalState`] into a serializable [`RetrievalTrace`] and renders
//! it for humans (CLI) or machines (JSON). It introduces **no behavior
//! change** to the controller — a trace is a *view* over the state the
//! orchestrator already produced.
//!
//! Aggregate evaluation reports (regime distributions across thousands
//! of queries, useful-vs-wasted rerank economics, the HTML "moat"
//! report) live in `neorag-calibration`, where the gold-labeled
//! `QueryOutcome` data is. This crate is the *live, per-query* half;
//! that crate is the *offline, aggregate* half. The split keeps this
//! crate's dependency surface to `neorag-core` alone, so a production
//! deployment can emit traces without pulling in the calibration
//! tooling.
//!
//! ## Zero-cost when unused
//!
//! Tracing is opt-in: you call [`RetrievalTrace::from_state`] when you
//! want a trace. The orchestrator does not record anything extra on the
//! hot path; everything a trace needs is already in
//! [`RetrievalState::history`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod render;
pub mod trace;

pub use trace::{RetrievalTrace, TraceIteration};
