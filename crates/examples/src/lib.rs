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

/// Resolve a path under the repo's local **models** directory (ONNX models
/// like BGE / MS-MARCO). Override with `REDHOP_MODELS_DIR`; defaults to
/// `<repo>/models`. Examples that need a model resolve it through this
/// helper so a contributor without the lab corpus can drop the right files
/// under any directory and point the env var at it.
pub fn model_path(rel: &str) -> PathBuf {
    let base = std::env::var("REDHOP_MODELS_DIR")
        .unwrap_or_else(|_| format!("{}/../../models", env!("CARGO_MANIFEST_DIR")));
    PathBuf::from(base).join(rel)
}

/// The BGE-small-en-v1.5 ONNX model + tokenizer pair, the most common
/// embedder across the finding-reproduction examples. Returns
/// `(model_path, tokenizer_path)`. Override the base with
/// `REDHOP_MODELS_DIR` or replace per-pair with `REDHOP_BGE_MODEL` /
/// `REDHOP_BGE_TOKENIZER` (those take priority).
pub fn bge_small_paths() -> (PathBuf, PathBuf) {
    let model = std::env::var("REDHOP_BGE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_path("bge-small-en-v1.5/onnx/model.onnx"));
    let tokenizer = std::env::var("REDHOP_BGE_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_path("bge-small-en-v1.5/tokenizer.json"));
    (model, tokenizer)
}

/// The MS-MARCO MiniLM cross-encoder ONNX model + tokenizer pair.
/// Override with `REDHOP_CE_MODEL` / `REDHOP_CE_TOKENIZER`, or by setting
/// `REDHOP_MODELS_DIR` to a directory containing
/// `ms-marco-MiniLM-L-6-v2/{onnx/model.onnx,tokenizer.json}`.
pub fn ms_marco_paths() -> (PathBuf, PathBuf) {
    let model = std::env::var("REDHOP_CE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_path("ms-marco-MiniLM-L-6-v2/onnx/model.onnx"));
    let tokenizer = std::env::var("REDHOP_CE_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_path("ms-marco-MiniLM-L-6-v2/tokenizer.json"));
    (model, tokenizer)
}
