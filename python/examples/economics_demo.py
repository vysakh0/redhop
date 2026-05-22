#!/usr/bin/env python3
"""economics_demo — context economics + non-destructive analysis.

Most RAG stacks have near-zero context observability. This shows what RedHop
exposes: distractor load before filtering, economics of the raw set, and the
optimized report.

    python examples/economics_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redhop  # noqa: E402
from _sample import QUERY, RETRIEVED, DISTRACTOR_MIN_GROUNDING, LINK_MIN_JACCARD  # noqa: E402


def main() -> None:
    kw = dict(distractor_min_grounding=DISTRACTOR_MIN_GROUNDING, link_min_jaccard=LINK_MIN_JACCARD)

    print("── analyze_context (non-destructive: what you have) ──\n")
    print(redhop.analyze_context(QUERY, RETRIEVED, **kw))

    print("\n── context_economics of the raw retrieved set ──")
    econ = redhop.context_economics(QUERY, RETRIEVED, **kw)
    print(json.dumps(econ, indent=2))

    print("\n── build_context (optimized) report as a dict ──")
    ctx = redhop.build_context(QUERY, RETRIEVED, token_budget=12000, **kw)
    report = redhop.report_to_dict(ctx.report)
    for k in ("strategy", "n_input_chunks", "n_selected", "total_tokens",
              "second_hop_rescue_count", "retained_evidence_ratio"):
        print(f"  {k}: {report[k]}")
    waste_before = econ["estimated_waste_tokens"]
    print(f"\n  wasted tokens on distractors: {waste_before} (raw) → "
          f"{report['economics']['estimated_waste_tokens']} (optimized)")


if __name__ == "__main__":
    main()
