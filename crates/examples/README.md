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
| `bench_context_strategies` | the hermetic benchmark used in `docs/findings/` |

### 2. Finding-reproduction scripts

The benchmark + eval scripts (HotpotQA / MuSiQue / CUAD / neotrace exports
/ ONNX bakeoffs) reproduce the numbers in `docs/findings/`. They expect
the lab corpus + models on disk under a predictable layout, but they no
longer hardcode an absolute path — every path goes through the helpers
in `crates/examples/src/lib.rs`:

| Helper | Default location | Override env var |
| --- | --- | --- |
| `data_path("...")` | `<repo>/data/...` | `REDHOP_DATA_DIR` |
| `exports_path("...")` | `<repo>/exports/...` | `REDHOP_EXPORTS_DIR` |
| `model_path("...")` | `<repo>/models/...` | `REDHOP_MODELS_DIR` |
| `bge_small_paths()` | `<models>/bge-small-en-v1.5/...` | `REDHOP_BGE_MODEL`, `REDHOP_BGE_TOKENIZER` |
| `ms_marco_paths()` | `<models>/ms-marco-MiniLM-L-6-v2/...` | `REDHOP_CE_MODEL`, `REDHOP_CE_TOKENIZER` |

So to reproduce a finding on a fresh machine:

```bash
# put the corpora + models wherever you like, then point env vars at them
export REDHOP_DATA_DIR=$HOME/redhop-corpora/data
export REDHOP_MODELS_DIR=$HOME/redhop-corpora/models
export REDHOP_EXPORTS_DIR=$HOME/redhop-corpora/exports

cargo run -p redhop-examples --example semantic_local_rerank \
    --features onnx --release
```

If the path isn't there, you get a clear `read_to_string` /
`OnnxEmbedder::load` error pointing at the missing file — not a
silent skip.

## Setup for the most common case

The BGE-small ONNX model + tokenizer cover most of the
finding-reproduction examples. One-time download:

```bash
mkdir -p ./models
python -c "
from huggingface_hub import hf_hub_download
for f in ['onnx/model.onnx', 'tokenizer.json']:
    hf_hub_download('BAAI/bge-small-en-v1.5', f,
        local_dir='./models/bge-small-en-v1.5')"
export REDHOP_MODELS_DIR=$PWD/models
```

For the cross-encoder examples (`ce_smoke`, `ce_escalation_economics`)
use `ms-marco-MiniLM-L-6-v2` the same way and set
`REDHOP_CE_MODEL` / `REDHOP_CE_TOKENIZER` (or drop it under
`<REDHOP_MODELS_DIR>/ms-marco-MiniLM-L-6-v2/`).

For HotpotQA / MuSiQue / CUAD, point `REDHOP_DATA_DIR` at a directory
containing `hotpotqa/hotpot_dev_distractor_v1.json`,
`musique/dev.jsonl`, etc. (paths are exactly what the docs/findings
commands reference).
