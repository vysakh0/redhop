//! Post-sweep analyses.
//!
//! These modules consume [`SweepReport`][sr] / [`QueryOutcome`][qo]
//! traces and produce structured analytic summaries:
//!
//! - [`confusion`] — regime confusion matrix with per-regime
//!   precision / recall / F1.
//! - [`regret`] — intervention regret (predicted gain vs measured gain)
//!   and the empirical distribution of useful vs harmful interventions.
//! - [`stability`] — bootstrap resampling over the query set to
//!   produce confidence intervals on per-threshold metrics.
//!
//! [sr]: crate::sweep::SweepReport
//! [qo]: crate::runner::QueryOutcome

pub mod confusion;
pub mod regret;
pub mod stability;

pub use confusion::{confusion_matrix, RegimeConfusionMatrix, RegimeMetrics};
pub use regret::{regret_summary, InterventionRegret};
pub use stability::{bootstrap_stability, BootstrapStability};
