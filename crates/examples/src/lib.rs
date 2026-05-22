//! Shared helpers for the runnable examples. See `examples/`.

use std::path::PathBuf;

/// Resolve a path under the repo's local **data** directory (datasets like
/// HotpotQA / MuSiQue). Override the base with the `REDHOP_DATA_DIR`
/// environment variable; otherwise defaults to `<repo>/data`, computed
/// relative to this crate so examples never hardcode an absolute machine path.
pub fn data_path(rel: &str) -> PathBuf {
    let base = std::env::var("REDHOP_DATA_DIR")
        .unwrap_or_else(|_| format!("{}/../../data", env!("CARGO_MANIFEST_DIR")));
    PathBuf::from(base).join(rel)
}

/// Resolve a path under the repo's local **exports** directory (experiment
/// outputs). Override with `REDHOP_EXPORTS_DIR`; defaults to `<repo>/exports`.
pub fn exports_path(rel: &str) -> PathBuf {
    let base = std::env::var("REDHOP_EXPORTS_DIR")
        .unwrap_or_else(|_| format!("{}/../../exports", env!("CARGO_MANIFEST_DIR")));
    PathBuf::from(base).join(rel)
}
