# Per-platform hybrid acceleration — the plan

> **Goal:** close the 30% ORT-CPU vs PyTorch-MPS gap measured in
> [HYBRID_LATENCY_PROFILE](../findings/HYBRID_LATENCY_PROFILE.md), and
> deliver equivalent wins on Linux and Windows. The current build uses
> ORT's default CPU EP everywhere; faster Execution Providers (EPs)
> exist for every platform but require build-time wiring.

## The EP landscape, by platform

ORT bundles multiple "execution providers" that route the forward
pass through platform-specific accelerators. The `ort = "2.0.0-rc.10"`
crate exposes each as a Cargo feature; the runtime tries them in
priority order and falls back to CPU.

| Platform | Best CPU-only EP | Best GPU EP (opt-in) | Notes |
|---|---|---|---|
| **macOS arm64 (M1/M2/M3+)** | **CoreML** (Neural Engine + CPU) | — | The biggest win here; CoreML routes ONNX through Apple's NE. Measured ~30-50% on small transformer forward passes on M1/M2. |
| **macOS x86_64 (Intel Mac)** | CoreML (CPU-only path) | — | Intel Mac is end-of-life for Apple; modest win, mostly the runtime's vectorized kernels. |
| **Linux x86_64** | **oneDNN / DNNL** (Intel CPU vector) | CUDA, TensorRT (NVIDIA) | DNNL is Intel's open-source CPU accelerator; works on AMD too via fallback. CUDA needs the user to ship a CUDA runtime; heavy. |
| **Linux aarch64** | XNNPACK (ARM NEON) | — | ARM-side ORT support is mature for inference; XNNPACK is the right EP. |
| **Windows x64** | **DNNL** OR **DirectML** | DirectML (NVIDIA/AMD/Intel GPU) | DirectML is the cross-vendor GPU EP unique to Windows; works with any modern GPU through DX12. DNNL is the CPU-only equivalent. |

**Default EP recommendation per published wheel:**

| Wheel target | EP feature flag(s) we'd build with |
|---|---|
| `macosx_11_0_arm64` | `coreml` |
| `macosx_11_0_x86_64` | `coreml` (falls back to CPU on Intel Mac silicon) |
| `manylinux_2_28_x86_64` | `dnnl` |
| `manylinux_2_28_aarch64` | `xnnpack` |
| `win_amd64` | `dnnl` |

GPU EPs (CUDA, DirectML) stay opt-in via separate wheels (`redhop[cuda]`
extras) because they require runtime deps the user must install.

## Architecture

### 1. Cargo features

In `crates/redhop/Cargo.toml`:

```toml
[features]
# Existing
semantic = ["dep:ort", "dep:tokenizers", "dep:ndarray"]
files = [...]

# New: platform-conditional EP features. Each pulls in the matching
# `ort` feature so the EP is linked + available at runtime.
ep-coreml   = ["semantic", "ort/coreml"]
ep-dnnl     = ["semantic", "ort/dnnl"]
ep-xnnpack  = ["semantic", "ort/xnnpack"]
ep-directml = ["semantic", "ort/directml"]
ep-cuda     = ["semantic", "ort/cuda"]   # heavy; CUDA runtime required
```

Default features stay `[]` so users who build from source on
unfamiliar hardware get the CPU EP (always works).

### 2. Session builder — runtime EP registration

`crates/redhop/src/embeddings/onnx.rs` currently does:

```rust
let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(intra_threads)?
    .commit_from_file(model_path)?;
```

Add EP registration before `commit_from_file`. The ORT API is:

```rust
let mut builder = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(intra_threads)?;

// Register every EP that was compiled in, in priority order.
// ORT tries each at session-creation time; first one that successfully
// loads wins for ops it can run. Unsupported ops fall through to CPU
// automatically — no code change needed.
let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();

#[cfg(feature = "ep-coreml")]
{
    use ort::execution_providers::CoreMLExecutionProvider;
    providers.push(CoreMLExecutionProvider::default().build());
}
#[cfg(feature = "ep-dnnl")]
{
    use ort::execution_providers::DnnlExecutionProvider;
    providers.push(DnnlExecutionProvider::default().build());
}
#[cfg(feature = "ep-xnnpack")]
{
    use ort::execution_providers::XNNPACKExecutionProvider;
    providers.push(XNNPACKExecutionProvider::default().build());
}
// DirectML / CUDA wired the same way under their features.

if !providers.is_empty() {
    builder = builder.with_execution_providers(providers)?;
}

let session = builder.commit_from_file(model_path)?;
```

ORT's design handles "EP failed to initialize" gracefully: if CoreML
isn't available at runtime (e.g. the wheel ran on Intel Mac with
ancient macOS), session creation still succeeds via the CPU fallback.
We don't need detection code.

### 3. CI wheel build matrix

`.github/workflows/release-python.yml` already has a matrix entry per
target. Each one's `args:` line passes `--features semantic,files`
today. New plan: add the right EP flag per target.

```yaml
- target: x86_64-apple-darwin
  args: --release --features semantic,files,ep-coreml -m python/Cargo.toml
- target: aarch64-apple-darwin
  args: --release --features semantic,files,ep-coreml -m python/Cargo.toml
- target: x86_64-unknown-linux-gnu
  args: --release --features semantic,files,ep-dnnl -m python/Cargo.toml
- target: aarch64-unknown-linux-gnu
  args: --release --features semantic,files,ep-xnnpack -m python/Cargo.toml
- target: x86_64-pc-windows-msvc
  args: --release --features semantic,files,ep-dnnl -m python/Cargo.toml
```

Same shape in `release-node.yml` for the napi binaries (which need
to expose the EP setting through `python/Cargo.toml`'s `[features]` and
`nodejs/Cargo.toml`'s `[features]` — see step 4).

### 4. Bindings feature pass-through

`python/Cargo.toml` and `nodejs/Cargo.toml` currently pass `files` and
`semantic` through to `redhop`. Add per-EP pass-through:

```toml
# python/Cargo.toml
[features]
default = []
semantic = ["redhop/semantic"]
files = ["redhop/files"]
ep-coreml   = ["semantic", "redhop/ep-coreml"]
ep-dnnl     = ["semantic", "redhop/ep-dnnl"]
ep-xnnpack  = ["semantic", "redhop/ep-xnnpack"]
ep-directml = ["semantic", "redhop/ep-directml"]
ep-cuda     = ["semantic", "redhop/ep-cuda"]
```

A `cargo install redhop --features ep-dnnl` from-source path then
matches what the published wheel gets.

### 5. Per-platform sanity probe

After wheel publish, run a sanity script on each platform that:

1. Loads `redhop` and creates a `Document.from_text(..., retrieval="hybrid", model="bge-small")`.
2. Calls `doc.context("test query")`.
3. Measures p50 latency over 20 warm queries.

Compare to the documented baseline (Mac CPU EP: 240ms / 467ms; goal
post-EP: 70-150ms on Apple Silicon, similar on Linux DNNL, Windows
DNNL). Persist the result as
`reports/hybrid_latency_<platform>_<date>.txt`. If the EP didn't
load (silent CPU fallback), the latency stays at baseline — that's
the signal.

A second script verifies `RetrievalMethod::Rerank` results don't drift
(EP changes only execution, not numerics meaningful for retrieval):

```python
hits = doc.context("test query").chunks
expected = [...]  # snapshot from CPU EP baseline
assert hits == expected, f"EP changed retrieval results: {hits}"
```

EPs *can* introduce tiny floating-point drift (CoreML uses fp16 in
some operators). If retrieval ranking changes, that's a real bug to
investigate before publishing.

## Build-complexity cost

Per-EP build-time deps (all transitive via `ort`):

| Feature | Adds | Wheel size impact |
|---|---|---|
| `ep-coreml` | Apple CoreML framework linker (system, no bundled binary) | ~0 MB |
| `ep-dnnl` | oneDNN static lib (bundled by ort) | ~15-25 MB |
| `ep-xnnpack` | XNNPACK static lib | ~5-10 MB |
| `ep-directml` | DirectML.dll (Windows system, no bundle) | ~0 MB |
| `ep-cuda` | CUDA runtime (user-installed) | ~0 MB but requires user CUDA |

CoreML and DirectML are zero-cost wheel-size-wise (they're system
frameworks on their target OS). DNNL adds ~15-25 MB to Linux + Windows
wheels — significant but worth it for the speedup. XNNPACK is smaller
for ARM-Linux.

CUDA stays separate — published as `redhop[cuda]` so the default wheel
isn't bloated by CUDA runtime requirements.

## Estimated impact, per platform

These are projections from ORT's documented EP speedups on similar
transformer workloads. Will need measurement post-implementation.

| Platform | Current p50 (HotpotQA hybrid) | With EP | Lift |
|---|---:|---:|---|
| macOS arm64 (M1+) | 240 ms | ~100-130 ms | 40-50% |
| macOS x86_64 | 240 ms | ~180-220 ms | 10-20% (Intel Mac is the weakest gain) |
| Linux x86_64 (Intel) | (untested baseline) | DNNL: ~100-150 ms | 40-50% (extrapolated) |
| Linux aarch64 (Graviton) | (untested baseline) | XNNPACK: ~80-120 ms | similar |
| Windows x64 (Intel) | (untested baseline) | DNNL: ~100-150 ms | 40-50% |
| Windows x64 + DirectML GPU | (untested) | ~30-50 ms | 5-10× |

**Honest caveats:** these are paper estimates. The ORT version has to
support each EP cleanly; some EPs have op-coverage gaps that force
chunks of the graph to fall back to CPU, eroding the lift. We need to
verify each EP loads for our specific bge-small ONNX graph (it should
— BERT-family architectures are well-supported across EPs — but
"should" is not "measured").

## What this does NOT do

- **Doesn't unify acceleration via PyTorch.** sentence-transformers
  on PyTorch MPS would be even faster on Mac (171ms vs ~100-130ms for
  CoreML in our projection), but bringing PyTorch into the Rust core
  is a much bigger architectural commitment. The EP path keeps the
  pure-Rust ONNX runtime.
- **Doesn't address the "ORT session has fixed startup cost"
  problem.** Even with the EP, the model load + session init at
  first `Document.from_text(retrieval="hybrid")` call is ~70-100ms
  on cold start. The EP cuts forward-pass time, not session setup.
- **Doesn't help users who never query semantic.** Pure-lexical
  workloads pay no model-load cost; this change is invisible to them.

## Suggested implementation order

1. **`ep-coreml` only** as a first pass. Mac is the biggest measured
   gap, and CoreML is zero wheel-size cost. One CI matrix update; one
   feature-flag block in `onnx.rs`. Verify with the per-platform sanity
   probe. ~1-2 hours of work, completes Task 309's "concrete fix path."
2. **`ep-dnnl` for Linux x86_64 + Windows x64.** Same shape but pulls
   in a system dep that bloats the wheel; verify wheel size + build
   time stay reasonable on CI. ~1-2 hours.
3. **`ep-xnnpack` for Linux aarch64.** Mostly the same as DNNL; lower
   priority because ARM-Linux is a smaller user share. ~30 min.
4. **`ep-directml` for Windows GPU users.** Optional; can ship as
   `redhop[directml]` separate wheel later if there's demand.
5. **CUDA never as default.** Separate `redhop[cuda]` extras wheel
   only.

## What to call out to users

If we ship step 1 (CoreML) in 0.3.2:

> "`retrieval='hybrid'` on Apple Silicon now routes through CoreML
> when available — measured ~40-50% latency drop on bge-small
> (240ms → ~100-130ms p50 on HotpotQA). Other platforms still use
> the CPU EP at default speeds for 0.3.2; Linux/Windows acceleration
> tracked for 0.3.3 (DNNL/XNNPACK pass-through, similar ~40-50% lift
> projected). No code changes needed by users — the EP is selected at
> wheel-build time per platform."

After step 2:

> "0.3.3 enables DNNL on Linux x86_64 + Windows x64. Same ~40-50%
> measured latency cut on bge-small hybrid retrieval, no API change."
