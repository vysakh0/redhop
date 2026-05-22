# Substrate in the Action Path — Dense Retrieval vs BM25

**Setup.** The recalibration experiment showed that BGE in the *sensing*
path (diagnostics only) didn't move the controller's economics, because
the retriever and reranker were embedding-blind. This experiment puts
BGE in the **action path**: same BGE diagnostics, same classifier, same
conservative policy — only the retriever changes.

| arm | retriever | substrate in action path? |
| --- | --------- | ------------------------- |
| A | BM25 | no (embedding-blind) |
| B | dense BGE (FlatVectorIndex) | yes (embedding-driven) |

60 HotpotQA items, top-k=4, lexical escalation reranker.

```bash
cargo run -p redhop-examples --example bge_dense_retrieval --features onnx --release
```

## Result — the opposite of the convenient hypothesis

| metric | BM25 (blind) | dense BGE (action) | Δ |
| ------ | ------------ | ------------------ | - |
| static recall | 0.537 | **0.732** | **+0.195** |
| adaptive recall | 0.600 | **0.778** | +0.178 |
| recall lift (from intervention) | 0.062 | 0.045 | −0.017 |
| intervention rate | 28% | **38%** | **+10 pts** |
| useful % | 53% | **26%** | **−27 pts** |
| rerank calls/query | 0.000 | 0.000 | 0 |
| **harmful lift** | **0.000** | **0.000** | **0** |
| wasted interventions | 8 | 17 | +9 |
| ECE | 0.277 | 0.255 | −0.022 |

The hypothesis was: stronger first-stage retrieval → less to fix →
fewer interventions. **The data falsifies it.** Dense BGE raised static
recall a lot (+19.5 pts, matching the bakeoff's 0.739), but the
controller intervened *more* (28% → 38%) and those interventions were
*half as useful* (53% → 26%), with double the waste.

## What's actually going on

Two effects, separated by the measurement:

1. **Raw recall is dominated by the substrate-in-action-path.** Dense
   BGE retrieval lifts static recall +19.5 pts and adaptive recall to
   0.778. If you care about absolute recall, dense + controller wins
   decisively — but almost all of that win is the *retriever*, not the
   controller.

2. **The controller's marginal value SHRINKS on dense retrieval**, and
   its intervention precision *drops*. The likely mechanism (empirical
   pattern certain; mechanism inferred):

   - Dense retrieval returns a *semantically tight* cluster — chunks all
     similar to the query, with flatter cosine spread. That diagnostic
     signature (flatter scores, higher redundancy) trips the
     `Ambiguous` / distractor pathways *more* than BM25's signature did,
     so the controller fires `ExpandTopK` more often.
   - But on multi-hop HotpotQA, the missing gold is often the *second
     hop* — a chunk **dissimilar** to the query. `ExpandTopK` on a dense
     index just returns *more query-similar* neighbors, which don't
     recover the dissimilar second-hop gold. So the extra interventions
     are **wasted** (recall_lift = 0).

   In short: **the controller's action repertoire (ExpandTopK = more
   neighbors, lexical rerank) is matched to BM25's failure modes, not
   dense retrieval's.** Swapping the retriever changed the failure mode;
   the actions didn't follow.

## The answer to the central questions (measured, on this setup)

> Does stronger semantic retrieval reduce reranking need?

**No** — it *increased* intervention rate (28% → 38%). (Reranking
specifically stayed ~0 on both; the controller prefers ExpandTopK.)

> Does better retrieval sharpen intervention precision?

**No** — precision *dropped* (useful% 53% → 26%).

> Can semantic retrieval reduce escalation cost?

**Not via the retriever alone.** Dense retrieval raised raw recall but
made the controller *less* efficient, because the available actions
don't address dense retrieval's failure modes.

## The systems insight (this is the result)

> **Retriever and action repertoire are coupled. A controller's actions
> must match the *failure modes* of the retriever it sits on. Swapping
> the retriever (even to a strictly better one on raw recall) can make a
> fixed action set mis-fire — more interventions, lower precision —
> until the actions are matched to the new retriever's gaps.**

This generalizes the calibration-coupling lessons:
- substrate ↔ classifier labels (ECE drift) — earlier finding;
- substrate ↔ controller actions — *this* finding, and it's the
  load-bearing one. The actions are where the economics live, and they
  are retriever-specific.

For dense retrieval on multi-hop QA, the action that would actually
help is **not** "more neighbors" — it's something that reaches the
*dissimilar* hop (query decomposition, MMR-style diversity, or a
cross-encoder that re-scores a wider net). That is a future
*experiment*, not a present claim — and deliberately not built here.

## Validated, again

The conservative controller's **zero-harm property held on both
retrievers** (harmful lift = 0.000). Even when its actions were a poor
match for dense retrieval — firing more, wasting more — it never
*reduced* recall. It degraded to "wasteful but safe," exactly the
failure mode the conservative design is meant to guarantee. The
architecture stayed stable across a retriever swap that broke the
controller's value-add.

## Honest limits

- **60-item sample, single run, no CI.** The +0.195 static-recall gap is
  large and consistent with the independent bakeoff (0.739); the
  intervention/useful deltas are directional.
- **Mechanism inferred, not isolated.** The "ExpandTopK can't reach the
  second hop" story is the most plausible reading of the
  more-intervention-less-useful pattern, but I have not instrumented
  per-action gold-recall by hop to prove it. Stated as a hypothesis.
- **Lexical escalation reranker.** The reranker is still embedding-blind.
  A semantic / cross-encoder reranker is the *next* action-path
  experiment (and the one that could actually help dense retrieval).
- **Default policy thresholds.** Tuned implicitly for BM25; a dense-
  specific recalibration might recover some precision, but the
  recalibration experiment showed the easy-threshold lever is a no-op,
  so the fix is more likely in the *action set* than the thresholds.

## Next, now sharply defined

1. **Semantic / cross-encoder escalation** — make `EscalateReranker` use
   a reranker that re-scores against the query over a wider candidate
   net. This is the action that fits dense retrieval's failure mode, and
   the cross-encoder model is downloadable + the `OnnxCrossEncoder`
   backend is run-ready. This is the experiment most likely to show the
   controller adding value *on top of* strong dense retrieval.
2. **Per-hop recall instrumentation** — to confirm the "ExpandTopK
   misses the second hop" mechanism.
