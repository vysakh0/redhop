# NeoTrace JSONL Schema (v1)

**Status:** stable for v1. Field additions are non-breaking; field removals require a version bump.

NeoTrace is the canonical interchange format between the Python RedHop
experimentation lab and the Rust RedHop calibration engine. One file
holds many records; one record describes the outcome of a single
*(query, retrieval method, generator model)* tuple. The schema is
designed to be a *superset* of what every existing Python experiment
file already records, so the exporter is lossless and the Rust loader
is a straight projection.

## Why JSONL

- Stream-parseable; the Rust loader doesn't have to materialize the
  whole file before yielding records.
- Trivially appendable; long-running Python experiments can flush rows
  as they run without coordinated file rewrites.
- Comments outside records are illegal; every line is a complete
  record. There is no nested top-level "rows" array to drift away from
  per-row schemas.

## Per-record fields

Required fields are in **bold**. Optional fields are marked `?`. Types
are written in Rust syntax; everything serializes to canonical JSON.

```jsonc
{
  // ── Identity ───────────────────────────────────────────────────────
  "schema_version": "neotrace/1",     // string, exact value
  "run_id":         "string",         // unique per experiment run
  "item_id":        "string",         // stable cross-method id
  "method":         "string",         // see "Method codes" below
  "model":          "string",         // generator model id (haiku, llama-8b…)
  "ce_model":       "string?",        // cross-encoder model id if used

  // ── Query + gold ───────────────────────────────────────────────────
  "question":       "string",
  "gold_answer":    "string?",        // present for HotpotQA/MuSiQue
  "gold_para":      "u32[]?",         // gold paragraph/chunk indices
  "gold_chunk_ids": "string[]?",      // when corpus uses string ids
  "doc":            "string?",        // per-doc benchmarks (evidence, learned)
  "type":           "string?",        // HotpotQA: bridge|comparison; in-house: A|B|C
  "level":          "string?",        // HotpotQA: easy|medium|hard
  "n_chunks":       "u32?",           // chunks in this example's pool

  // ── Retrieval ──────────────────────────────────────────────────────
  "retrieved":      "u32[]?",         // top-k indices in the example's pool
  "retrieved_chunk_ids": "string[]?", // OR string ids when applicable
  "retrieval_recall": "f32?",         // |gold ∩ retrieved| / |gold|
  "top_k":          "u32?",           // k used for retrieval

  // ── Evidence-quality metrics (Python lab columns) ─────────────────
  // All are normalized [0, 1] where higher is better unless noted.
  "continuity":              "f32?",
  "answer_span_density":     "f32?",
  "answer_bearing_fraction": "f32?",
  "distractor_ratio":        "f32?",  // lower is better
  "query_overlap":           "f32?",
  "entity_overlap":          "f32?",
  "purity":                  "f32?",

  // ── Generation outcome ─────────────────────────────────────────────
  "answer":          "string?",
  "ans_similarity":  "f32?",          // cosine vs gold_answer embedding
  "ans_kw_recall":   "f32?",          // gold-keyword recall in answer

  // ── LLM-judge (when available) ─────────────────────────────────────
  "judge_model":     "string?",       // e.g. "sonnet"
  "judge_score":     "f32?",          // [0, 1]; semantics judge-specific
  "judge_preferred": "string?",       // for pairwise: method/run id chosen
  "judge_reason":    "string?",       // free-text rationale

  // ── Regime label (for calibration) ─────────────────────────────────
  "true_regime":     "string?",       // easy|saturated|distractor_heavy|ambiguous|sparse

  // ── Adaptive-controller fields (Rust side; absent in Python rows) ─
  "intervened":      "bool?",
  "predicted_regime": "string?",
  "predicted_regime_p": "f32?",
  "action_trace": [                   // optional; mirrors TakenAction
    {
      "action":         "string",     // stop|abstain|expand_top_k|escalate_reranker
      "iteration":      "u32",
      "expected_gain":  "f32",
      "actual_gain":    "f32?",
      "latency_ms":     "u64",
      "retrieval_calls": "u32",
      "rerank_calls":   "u32",
      "rationale":      "string"
    }
  ],

  // ── Performance ────────────────────────────────────────────────────
  "latency_ms":      "u64?",

  // ── Escape hatch ───────────────────────────────────────────────────
  "extra": { /* arbitrary JSON; never consumed by core calibration */ }
}
```

## File header (optional sidecar)

Each `*.jsonl` may be accompanied by a same-name `.meta.json` file
holding aggregates that don't fit per-row:

```jsonc
{
  "schema_version": "neotrace/1",
  "source":      "hotpot_full.json",     // original Python source
  "exported_at": "2026-05-21T14:00:00Z",
  "model":       "haiku",
  "ce_model":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "n_eval":      100,
  "top_k":       4,
  "scorer_info": {                       // frozen learned-scorer artifact
    "coefs":             { "bm25": ..., "cosine": ..., ... },
    "n_train_pairs":     2300,
    "train_positive_rate": 0.18
  },
  "notes":       "auto-generated by export_to_neotrace.py"
}
```

## Method codes

The Python lab uses these names; the schema fixes them as the canonical
set so Rust enums stay stable.

| code            | description                                                          |
| --------------- | -------------------------------------------------------------------- |
| `cosine`        | dense cosine baseline                                                |
| `bm25`          | BM25 lexical baseline                                                |
| `rrf`           | reciprocal rank fusion of cosine + BM25                              |
| `answerability` | learned scorer with hand features (Python `analysis/answerability.py`) |
| `learned`       | learned ranker (legacy alias for answerability)                      |
| `cross_encoder` | cross-encoder reranker over top-N candidates                         |
| `trajectory`    | trajectory-aware (topology-era; pre-pivot)                           |
| `adaptive`      | Rust adaptive orchestrator (new)                                     |
| `static`        | Rust static baseline (new)                                           |

`trajectory` rows are retained for negative-control analysis; they are
**not** considered a recommended production method.

## Regime semantics

Regime is canonical to RedHop; the loader maps it to
[`redhop_core::RetrievalRegime`]. Codes:

| code               | meaning                                                  |
| ------------------ | -------------------------------------------------------- |
| `easy`             | high grounding, retrieval done                           |
| `saturated`        | high redundancy in top-k                                 |
| `distractor_heavy` | many off-topic retrievals                                |
| `ambiguous`        | flat scores or multi-cluster top-k                       |
| `sparse`           | corpus unlikely to contain the answer                    |

`true_regime` is the human/judge-derived ground truth (e.g.
HotpotQA hard-bridge → `distractor_heavy`). `predicted_regime` is what
the Rust classifier emitted at adaptive_run time. Either may be absent.

## Mapping from Python sources

| Python file                          | record shape                            | required transform                                        |
| ------------------------------------ | --------------------------------------- | --------------------------------------------------------- |
| `hotpot_*.json`                      | one record per (item_id, method)        | promote file-level `model`, `ce_model`, `top_k` to row    |
| `musique_*.json`                     | one record per (item_id, method)        | same; `gold_para` already present                          |
| `evidence_evidence.json`             | one record per (doc, question, method)  | `doc` + `question` form a synthetic `item_id`; no `gold_*` |
| `learned_*.json`                     | one record per (doc, question, method)  | same as evidence                                            |
| `judge_multihop.json`                | one record per (item, comparison-side)  | fold pairwise judgments into `judge_*` fields              |
| `endtoend.json`                      | one record per (item, arm)              | `cosine_baseline` → method=`cosine`, `trajectory_aware` → `trajectory` |

Files explicitly **not** mapped: `ablations.json`, `operators_topology.json`,
`scaling.json`, `validation.json`, `experiments.json`, `organization_*.json`.
Per the Python validation report these are topology-era aggregate-only
artifacts; they have no per-query rows that survive the pivot.

## Versioning

The version string is `neotrace/<major>` where `<major>` is incremented
only on a breaking change (field removal, type change, semantic
reinterpretation). Additive changes do not bump the version; consumers
must tolerate unknown fields.

The Rust loader accepts `schema_version: "neotrace/1"` exactly; any
other value is a hard error. Future major versions ship parallel
loaders, never silent upgrades.
