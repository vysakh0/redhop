#!/usr/bin/env python3
"""Guard against benchmark-number drift in user-facing docs.

The README and docs/COMPARISON.md quote headline numbers measured in
docs/findings/. Findings get re-run; prose gets edited piecemeal; the
same figure historically appeared in two places with two values (e.g.
raw CUAD as both 81.3% and 82%, hybrid HotpotQA as both 83% and 81%).

This script pins each headline claim to the exact strings that must
appear in each user-facing file. When a finding is re-run:

  1. update the canonical value HERE (one place), citing the finding,
  2. the check then fails on every file still carrying the old number,
     telling you exactly what to update.

Zero dependencies. Run locally:  python3 scripts/check_readme_numbers.py
In CI: invoked by the `docs-numbers` job in .github/workflows/ci.yml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Claim:
    claim_id: str
    source: str  # the finding that owns the canonical value
    # file (repo-relative) -> substrings that MUST be present verbatim
    required: dict[str, list[str]] = field(default_factory=dict)


CLAIMS = [
    Claim(
        "hotpotqa-retention-vs-frameworks",
        source="docs/findings/FRAMEWORK_COMPARISON.md (rerun 2026-06-06)",
        required={
            "README.md": ["| HotpotQA (multi-hop) | **80%** | 71% | 72% |"],
            "docs/COMPARISON.md": ["| HotpotQA (multi-hop) | **80%** | 71% | 72% |"],
        },
    ),
    Claim(
        "musique-retention-vs-frameworks",
        source="docs/findings/MUSIQUE_MULTIHOP.md (n=300, 2026-06-08)",
        required={
            "README.md": [
                "| MuSiQue (compositional multi-hop) | **22%** | 19% | 17% |"
            ],
            "docs/COMPARISON.md": [
                "| MuSiQue (compositional multi-hop) | **22%** | 19% | 17% |"
            ],
        },
    ),
    Claim(
        "cuad-raw-retention-vs-frameworks",
        source="docs/findings/FRAMEWORK_COMPARISON.md",
        required={
            "README.md": [
                "| CUAD (contracts, raw template query) | 82% | 73% | **86%** |"
            ],
            "docs/COMPARISON.md": [
                "| CUAD (contracts, raw template query) | 82% | 73% | **86%** |"
            ],
        },
    ),
    Claim(
        "cuad-recipe-ladder",
        source="docs/findings/CUAD_CLAUSE_EXPANSION.md (three-arm, n=300)",
        required={
            "README.md": ["81.3% → 87.7% → 90.7%"],
            "docs/COMPARISON.md": [
                "| raw 24-word template | — | 81.3% | — |",
                "| + strip the wrapper | `Stripper` | 87.7% | **+6.4** |",
                "| + add workload synonyms | `Vocabulary` (34-key clause dict) | **90.7%** | **+3.0** |",
            ],
        },
    ),
    Claim(
        "cuad-fair-preprocessing",
        source="bench/compare.py (n=300, 2026-06-08)",
        required={
            "docs/COMPARISON.md": [
                "| LlamaIndex | 86% | **94%** |",
                "| RedHop | 82% | 88% |",
                "| LangChain | 73% | 79% |",
            ],
        },
    ),
    Claim(
        "hybrid-head-to-head",
        source="docs/findings/MULTIHOP_HYBRID_COMPETITORS.md (n=100, post 0.3.1 pure-rerank fix)",
        required={
            "README.md": ["71% → 81% (n=100)"],
            "docs/COMPARISON.md": [
                "| HotpotQA | **81%** | 77% | 67% |",
                "| MuSiQue | 34% | **39%** | 31% |",
            ],
        },
    ),
    Claim(
        "document-eval-token-economics",
        source="docs/findings/DOCUMENT_EVAL_CUAD.md (50 contracts, 644 queries)",
        required={
            "README.md": ["−80% prompt", "~1.7ms per query"],
        },
    ),
    Claim(
        "ragas-correlation",
        source="docs/findings/EVAL_JUDGED_CALIBRATION.md (n=200 HotpotQA)",
        required={
            "README.md": ["r=+0.664"],
            "docs/COMPARISON.md": ["r=+0.664"],
        },
    ),
]


def main() -> int:
    failures: list[str] = []
    for claim in CLAIMS:
        for rel_path, substrings in claim.required.items():
            path = ROOT / rel_path
            if not path.is_file():
                failures.append(f"[{claim.claim_id}] missing file: {rel_path}")
                continue
            text = path.read_text(encoding="utf-8")
            for s in substrings:
                if s not in text:
                    failures.append(
                        f"[{claim.claim_id}] {rel_path} no longer contains:\n"
                        f"      {s!r}\n"
                        f"    canonical source: {claim.source}\n"
                        f"    (if the finding was re-run, update the registry in "
                        f"scripts/check_readme_numbers.py and all files it lists)"
                    )
    if failures:
        print(f"docs number-drift check FAILED ({len(failures)} problem(s)):\n")
        for f in failures:
            print(f"  - {f}\n")
        return 1
    n = sum(len(v) for c in CLAIMS for v in c.required.values())
    print(f"docs number-drift check OK: {len(CLAIMS)} claims, {n} pinned strings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
