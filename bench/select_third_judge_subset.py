#!/usr/bin/env python3
"""Pick a subset of the n=200 correlation results to feed into the
third-judge bench. Runs Claude on all contested cases + a sample of
agreement cases as a control.

Output JSON has the same shape as the n=200 bench so
`bench/eval_third_judge.py --in <filtered.json>` Just Works.
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--contested-threshold", type=float, default=0.3)
    p.add_argument("--control-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = json.loads(Path(args.input_path).read_text())
    cases = data["cases"]

    contested = []
    agreement = []
    for c in cases:
        rd = c.get("redhop_decomposed")
        rg = c.get("ragas_faithfulness")
        if rd is None or rg is None:
            continue
        delta = abs(rd - rg)
        if delta >= args.contested_threshold:
            contested.append(c)
        elif delta == 0:
            agreement.append(c)

    random.seed(args.seed)
    control = random.sample(agreement, min(args.control_n, len(agreement)))

    selected = contested + control
    selected_qids = {c["qid"] for c in selected}
    print(f"Selected {len(selected)} cases:")
    print(f"  contested (|delta| >= {args.contested_threshold}): {len(contested)}")
    print(f"  agreement control:                                 {len(control)}")

    out_data = {**data, "n": len(selected), "cases": selected}
    out_path = REPO / args.out
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
