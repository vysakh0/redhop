//! Renderers for [`RetrievalTrace`].
//!
//! - [`cli`] — human-readable ASCII, one trace per call.
//! - [`json`] — machine-readable; one JSON object per trace, suitable
//!   for appending to a JSONL stream.
//!
//! All renderers read the same [`RetrievalTrace`] fields; there is no
//! view-specific recomputation, so the CLI, JSON, and (in
//! `neorag-calibration`) HTML views always agree on the numbers.
//!
//! [`RetrievalTrace`]: crate::trace::RetrievalTrace

pub mod cli;
pub mod json;
