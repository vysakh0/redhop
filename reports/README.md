# reports/

Captured raw outputs of specific experiment runs — the durable, citable
evidence behind the [findings](../docs/findings/README.md). A findings doc
explains *what we learned*; a report here is *the actual output that run
produced*, so claims stay reproducible and regressions stay catchable.

Each report directory contains a `README.md` (run metadata + headline) and
the raw output files.

| Report | Finding | Headline |
| ------ | ------- | -------- |
| [reasoning_preserving_n300](reasoning_preserving_n300/) | [REASONING_PRESERVATION](../docs/findings/REASONING_PRESERVATION.md) | ReasoningPreserving beats aggressive filtering end-to-end; gain causally localized to gold reachability |

New reports should record: date, n, models, exact command, and the
headline result with CIs. Prefer outputs that can be regenerated from a
cached/deterministic source.
