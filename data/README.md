# data/

Local datasets used by the example/benchmark harnesses, vendored into the repo
so nothing references a machine-specific absolute path. Examples resolve this
directory via `redhop_examples::data_path(...)` (override with `REDHOP_DATA_DIR`).

| path | dataset | source / license |
| ---- | ------- | ---------------- |
| `hotpotqa/hotpot_dev_distractor_v1.json` | HotpotQA dev (distractor) | Yang et al., 2018 — CC BY-SA 4.0 |
| `musique/dev.jsonl` | MuSiQue dev (answerable) | Trivedi et al., 2022 — CC BY 4.0 |

These are third-party research datasets included for reproducibility; their
original licenses apply. Models (BGE / cross-encoder) used by the `--features
onnx` examples are **not** vendored — fetch them separately (see
`docs/findings/EMBEDDING_BAKEOFF.md`).
