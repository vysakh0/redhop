# Dense-tier DX: `model="…"` with auto-download (design spec)

> **Status:** Proposed (Option 1, chosen). **Goal:** remove the ONNX-export
> friction from the dense rerank tier so it's a one-liner, while keeping RedHop's
> Rust engine, lightweight default build, and offline guarantees.

## Problem

Today, opting into dense rerank makes the user *produce and locate model files*:

```python
doc = redhop.Document.from_text(
    text, retrieval="rerank",
    embedder_model="bge/model.onnx",        # they must export this
    embedder_tokenizer="bge/tokenizer.json",
    embedder_dim=384, embedder_pooling="cls",
)
```

Most users have never run `optimum-cli export onnx`. Competitors hide this —
`HuggingFaceEmbeddings("BAAI/bge-small-en-v1.5")` auto-downloads and runs in one
line. This friction only hits the *opt-in* tier (the BM25 default needs no model),
but it's the difference between "two lines" and "go learn ONNX export."

## Key finding that makes this cheap

Pre-built ONNX exports of the recommended models **already exist on HuggingFace** —
no user export needed:

- `Qdrant/bge-small-en-v1.5-onnx-Q` (int8, ~35 MB) — quantized, fastest to index
- `onnx-community/bge-small-en-v1.5-ONNX`, `Xenova/bge-small-en-v1.5` (`onnx/` subfolder)

And [`fastembed-rs`](https://github.com/Anush008/fastembed-rs) (Apache-2.0) proves
the exact pattern on **our own stack** (`ort` + `tokenizers`): model-by-name →
auto-download from HF → run, with quantized variants. We adopt the *pattern*, not
the dependency, to keep control of pooling/prefixes and the Decision Report.

## Proposed API

```python
# one-liner: downloads the default (quantized BGE-small) on first use, cached
doc = redhop.Document.from_text(text, retrieval="rerank")

# pick a known model by name
doc = redhop.Document.from_text(text, retrieval="rerank", model="bge-small")

# power user / offline / custom model: explicit paths still work, unchanged
doc = redhop.Document.from_text(
    text, retrieval="rerank",
    embedder_model="/path/model.onnx", embedder_tokenizer="/path/tokenizer.json",
    embedder_dim=384, embedder_pooling="cls",
)
```

- `model="…"` and explicit `embedder_*` paths are mutually exclusive; paths win if both given.
- Default model (no `model`, no paths) = the recommended **quantized** BGE-small —
  per [SPEED_VS_FRAMEWORKS](findings/SPEED_VS_FRAMEWORKS.md), int8 is ~2× faster to
  index and 4× smaller at near-identical quality.

## Model registry

A small static table in `redhop-embeddings` (reuses existing `EmbedderConfig`
presets, which already encode pooling + E5/BGE prefixes):

| name | HF repo | revision (pinned) | onnx path | dim | pooling | prefixes |
| ---- | ------- | ----------------- | --------- | --- | ------- | -------- |
| `bge-small` *(default)* | `Qdrant/bge-small-en-v1.5-onnx-Q` | `<sha>` | `model_optimized.onnx` | 384 | cls | — |
| `bge-small-fp32` | `onnx-community/bge-small-en-v1.5-ONNX` | `<sha>` | `onnx/model.onnx` | 384 | cls | — |
| `bge-base` | `onnx-community/bge-base-en-v1.5-ONNX` | `<sha>` | `onnx/model.onnx` | 768 | cls | — |
| `e5-small` | `…/e5-small-v2-onnx` | `<sha>` | `onnx/model.onnx` | 384 | mean | `query: ` / `passage: ` |
| `minilm` | `…/all-MiniLM-L6-v2-onnx` | `<sha>` | `onnx/model.onnx` | 384 | mean | — |

- **Pin a revision (commit SHA) per entry** for reproducibility + supply-chain safety
  (don't track a moving `main`). Each entry is validated once (recall + pooling correct).
- Registry is data, not new abstraction — the resolver returns an `EmbedderConfig` +
  local file paths, then the existing `OnnxEmbedder::load` runs unchanged.

## Download mechanism

- Add the **`hf-hub`** crate (the standard Rust HF client; what fastembed uses),
  **behind the existing `onnx` feature** (or a dedicated `hub` feature) so the default
  `cargo build` stays dependency-light and offline, per the runtime's design discipline.
- Cache to the standard HF cache (`HF_HOME` / `~/.cache/huggingface/hub`), so it
  shares with any other HF tooling and survives across runs.
- **Respect `HF_HUB_OFFLINE=1`**: use cache only; if the model isn't cached, fail with
  a clear, actionable error (suggest pre-downloading or passing explicit paths).
- Optionally surface a download-progress callback (fastembed does); nice-to-have.

## Offline / air-gapped story (unchanged guarantees)

- The **default build and the BM25 default tier need no network and no model** — this
  proposal only touches the opt-in `onnx`/rerank path.
- Explicit `embedder_*` paths remain the fully-offline escape hatch.
- `HF_HUB_OFFLINE` + a warmed cache = reproducible offline rerank.

## Out of scope (note for later)

- **Cross-encoder rerankers** (e.g. `bge-reranker`): fastembed's `TextRerank` is a
  *cross-encoder*; RedHop's "rerank" is bi-encoder cosine over a BM25 pool. A
  cross-encoder tier is a separate, higher-quality (slower) option — future finding.
- **Static embeddings** (`minishlab/potion-retrieval-32M`, model2vec): ~no inference
  cost, much faster, lower quality. Candidate "ultra-fast" tier; evaluate separately.

## Risks

- **Trusting third-party ONNX repos.** Mitigate by pinning revisions, curating to a
  small known-good set (Qdrant/onnx-community/Xenova), and optionally verifying a
  hash. Document that `model="…"` downloads from these repos.
- **`hf-hub` pulls `reqwest`/`tokio`** — acceptable because it's gated behind `onnx`
  (already non-default); the lightweight default build is unaffected.
- **Per-entry correctness** (pooling/prefix/dim) must be validated once per model —
  wrong pooling silently degrades recall.

## Implementation steps

1. `redhop-embeddings`: add `hf-hub` (feature-gated), a `model_registry` (name →
   repo+revision+path+`EmbedderConfig`), and a `resolve_model(name) -> (paths, config)`
   that downloads-if-absent and respects `HF_HUB_OFFLINE`.
2. `Document`/builder: accept `model: Option<&str>`; resolve to paths+config; keep the
   explicit-paths branch; default to `bge-small` (quantized) when rerank is requested
   with neither. Mutual-exclusion validation + clear errors.
3. Python binding (`from_text`): add `model=` kwarg; document `model=` vs `embedder_*`.
4. Validate each registry entry end-to-end (recall sanity on HotpotQA slice) and pin SHAs.
5. Docs: update the site's Retrieval-options page to lead with `model="bge-small"` and
   demote the manual-export path to "custom/offline."
