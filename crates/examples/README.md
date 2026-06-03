# redhop-examples

End-to-end examples and finding-reproduction scripts. **Not published** —
this crate lives only in the GitHub source tree.

## Two categories

### 1. Library-only examples

Self-contained, no external data. Build + run on any machine, no env
vars, no downloads. These are the right starting point if you're
exploring the API:

| Example | What it shows |
| --- | --- |
| `quickstart` | the 30-second tour of `Document::from_text` → `context()` |
| `document_dense` | bring-your-own embedder for `RetrievalMode::Dense` (needs `--features onnx` + a BGE-class ONNX model — env-var-overridable) |
| `bench_context_strategies` | the hermetic benchmark used in `docs/findings/` |

### 2. Finding-reproduction scripts

The benchmark and eval scripts under `examples/` (HotpotQA / MuSiQue /
CUAD / neotrace exports / ONNX bakeoffs) are what we used to generate
the numbers in `docs/findings/`. Many of them currently have **hardcoded
data paths** baked in as `const X: &str = "/Users/vysakh/projects/neorag/..."`,
left over from the lab-repo layout. They build fine; they only fail at
runtime if the path doesn't exist on your machine.

To run one of these on a different machine you have two options:

1. **Mirror the lab layout** — clone the same fixture files into
   `/Users/vysakh/projects/neorag/` (paths shown below). The error
   message you get from the failed `read_to_string` / ONNX load tells
   you which file is missing.
2. **Edit the example** to point at your own paths, or wrap the
   constants with `std::env::var("REDHOP_…").unwrap_or_else(...)` and
   contribute the PR — a handful of examples already do this
   (`document_dense`, `semantic_local_rerank`, `semantic_mismatch`).

The env-var convention already in use:

| Var | Purpose | Default (lab path) |
| --- | --- | --- |
| `REDHOP_BGE_MODEL` | BGE-small ONNX graph | `/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx` |
| `REDHOP_BGE_TOKENIZER` | BGE-small tokenizer JSON | `…/tokenizer.json` |
| `REDHOP_CUAD_PATH` | CUAD contracts dataset | (no default; required) |
| `REDHOP_CUAD_PERTURB` | CUAD perturbation mode | (optional) |
| `REDHOP_DOC_STRATEGY` | per-example strategy override | (optional) |
| `REDHOP_FILTER_TAU` | distractor filter threshold | (optional) |

## Known gap

About 15 of the finding-reproduction examples don't yet have env-var
fallbacks for their data paths — they hardcode the lab layout. PRs to
add `std::env::var(...).unwrap_or_else(...)` are welcome; the pattern
to mirror lives in `examples/document_dense.rs`.
