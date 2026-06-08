# Per-platform hybrid acceleration — design exercise (REJECTED)

> **Outcome:** plan written, EP wiring built, CoreML measured to
> regress, all EP flags removed from the codebase. Kept as a record
> of why we won't pursue this path again without a new measurement.

## What we tried

In the 0.3.1 audit branch we identified that RedHop's `retrieval="hybrid"`
on Apple Silicon was ~30% slower than sentence-transformers PyTorch MPS
on the same model + same workload
([HYBRID_LATENCY_PROFILE](../findings/HYBRID_LATENCY_PROFILE.md)). The
plan was to close the gap with platform-specific ORT Execution Providers
(EPs):

| Platform | EP we tried/considered | Status |
|---|---|---|
| macOS arm64 | CoreML (Apple Neural Engine + GPU + CPU fallback) | **MEASURED — regressed.** Removed. |
| macOS x86_64 | CoreML (CPU-only path) | Paper-projected only. Removed. |
| Linux x86_64 | OneDNN (Intel oneDNN) | Paper-projected only. Removed. |
| Linux aarch64 | XNNPACK (ARM NEON) | Paper-projected only. Removed. |
| Windows x64 | OneDNN / DirectML | Paper-projected only. Removed. |

## What the measurement said

CoreML on Apple Silicon at ort 2.0.0-rc.10 + bge-small, n=100 multi-hop
hybrid probe:

| | Without EP (CPU) | With CoreML EP |
|---|---:|---:|
| HotpotQA hybrid p50 | 240 ms | **303 ms (worse)** |
| MuSiQue hybrid p50 | 467 ms | **513 ms (worse)** |
| Amortized (3 docs × 5 queries) | 915 ms | 945 ms (essentially tied) |

Likely cause: the bge-small graph has ops the CoreML EP at this ort
version doesn't accelerate well; fallback-to-CPU per-op adds boundary
memory-copy overhead that exceeds any per-op savings. Plus per-Document
CoreML model compile cost dominates the one-shot pattern.

## Why we removed the flags entirely

Initially the code kept `ep-coreml`, `ep-onednn`, `ep-xnnpack`,
`ep-directml`, `ep-cuda` as opt-in Cargo features even though only
`ep-coreml` was measured (and it regressed). On review: keeping
unmeasured-or-regressed feature flags in the public surface is a
maintainer footgun. A future contributor would read the comment
"Apple Silicon, no wheel-size cost" and flip a flag without
re-measuring. Stripped them all.

If a future ort release lands CoreML improvements, or someone benchmarks
OneDNN/XNNPACK on a Linux runner and finds a real win, the right move
is to re-introduce the specific flag **with the measurement attached**,
not to keep paper-projected flags speculatively.

## What this leaves

- ORT's default CPU EP everywhere. Works on every platform, consistent
  numerics across platforms, no surprises.
- The 30% ORT-CPU vs PyTorch-MPS latency gap on Apple Silicon is the
  honest price of pure-Rust ONNX. Documented in HYBRID_LATENCY_PROFILE.
- Real future paths (separate work):
  - **Candle backend** (Hugging Face's Rust ML framework with Metal +
    Accelerate.framework + ARM NEON support) — has been pitched for 0.4.
  - **`Document.from_chunks(chunks_with_precomputed_embeddings)`** —
    users who care about speed and have access to sentence-transformers
    or another fast embedder can compute vectors externally and hand
    them to RedHop. Already supported.

## See also

- [HYBRID_LATENCY_PROFILE](../findings/HYBRID_LATENCY_PROFILE.md) — the
  measurement that motivated and rejected this plan.
- [MULTIHOP_HYBRID_COMPETITORS](../findings/MULTIHOP_HYBRID_COMPETITORS.md)
  — the comparison establishing where RedHop hybrid stands today.
