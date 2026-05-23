# data/

Local datasets used by the example/benchmark harnesses, vendored into the repo
so nothing references a machine-specific absolute path. Examples resolve this
directory via `redhop_examples::data_path(...)` (override with `REDHOP_DATA_DIR`).

| path | dataset | source / license |
| ---- | ------- | ---------------- |
| `hotpotqa/hotpot_dev_distractor_v1.json` | HotpotQA dev (distractor) | Yang et al., 2018 — CC BY-SA 4.0 |
| `musique/dev.jsonl` | MuSiQue dev (answerable) | Trivedi et al., 2022 — CC BY 4.0 |
| `cuad/cuad_sample.json` | CUAD v1 — 50 contracts, answerable clause QAs only (real long legal documents + gold answer spans) | The Atticus Project, 2021 — CC BY 4.0 |

The CUAD sample is the first 50 contracts of CUADv1 (SQuAD format), keeping only
answerable questions (gold spans present). Source: `TheAtticusProject/cuad`
(`data.zip` → `CUADv1.json`). Used by the real-document Document eval
(`eval_cuad_documents`). Point `REDHOP_CUAD_PATH` at the full `CUADv1.json` to
run the whole set.

These are third-party research datasets included for reproducibility; their
original licenses apply. Models (BGE / cross-encoder) used by the `--features
onnx` examples are **not** vendored — fetch them separately (see
`docs/findings/EMBEDDING_BAKEOFF.md`).
