#!/usr/bin/env python3
"""Trace claim extraction + verification on specific HotpotQA cases.

For each target qid this script:
  1. Loads the (question, context, generated answer) from
     `reports/eval_correlation_hotpot_n25.json` and the original
     HotpotQA dev distractor data.
  2. Calls the LLM (OpenRouter or OpenAI) with our current extraction
     prompt → prints the extracted claims.
  3. Calls the LLM with our current batched verification prompt → prints
     per-claim scores.
  4. Compares vs the on-disk score and Claude's score so we can see
     exactly where the pipeline leaked.

Run:
  bench/.venv/bin/python bench/eval_faith_trace.py
  bench/.venv/bin/python bench/eval_faith_trace.py --variant v2

The `--variant` flag lets us try alternate prompt formulations in
parallel without touching Rust. v0 == current Rust prompts (mirrored).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]


# ── LLM client ──────────────────────────────────────────────────────────────

def _make_client():
    """Return a chat-completion callable. Tries OpenRouter then OpenAI."""
    from openai import OpenAI

    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        model = os.environ.get("EVAL_MODEL", "openai/gpt-4o-mini")
    elif os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        model = os.environ.get("EVAL_MODEL", "gpt-4o-mini")
    else:
        print("ERROR: set OPENROUTER_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    return call, model


# ── Prompts — variants ──────────────────────────────────────────────────────

CLAIM_EXTRACTION_SYSTEM = (
    "You decompose answers into atomic factual claims for verification "
    "against a context. No commentary."
)
CLAIM_VERIFICATION_SYSTEM = (
    "You are a strict, careful judge. Determine whether a single CLAIM "
    "is supported by a given CONTEXT, with no inference beyond what the "
    "CONTEXT actually says."
)


def claim_extraction_v0(answer: str) -> str:
    """Mirror of crates/redhop/src/context/eval.rs:claim_extraction_prompt
    (current Rust)."""
    return (
        "Decompose an ANSWER into atomic factual claims. Each claim must be "
        "self-contained — resolve pronouns, drop conversational filler. Output "
        "one claim per line, no numbering, no introduction. If the ANSWER "
        "makes no verifiable factual claims (refusal, pure opinion, "
        "meta-commentary), output nothing.\n\n"
        "EXAMPLE\n"
        "ANSWER: He was a German-born theoretical physicist best known for "
        "developing the theory of relativity. He also made important "
        "contributions to quantum mechanics.\n"
        "CLAIMS:\n"
        "Albert Einstein was a German-born theoretical physicist.\n"
        "Albert Einstein was best known for developing the theory of relativity.\n"
        "Albert Einstein made important contributions to quantum mechanics.\n\n"
        f"NOW DO THIS ONE\n"
        f"ANSWER: {answer}\n"
        f"CLAIMS:"
    )


def claim_extraction_v1(answer: str) -> str:
    """v1: tighter rules. Forces decomposition of comparatives, compound
    relations, definite-article uniqueness, and cross-attributions. Adds
    a comparative-claim example and a compound-attribution example.

    The Einstein example stays for the simple case. Two extra examples
    show how to handle the failure modes we observed on HotpotQA:
      - 'X is older than Y' → 3 atomic claims
      - 'A is the B in C designed by D' → 3 atomic claims, with the
        definite-article uniqueness made explicit
    """
    return (
        "Decompose an ANSWER into ATOMIC factual claims for verification. "
        "RULES:\n"
        "  R1. Each claim must be a single subject-predicate-object fact. "
        "If a sentence has two predicates joined by AND or by a comma, "
        "split into two claims.\n"
        "  R2. Comparisons like 'X is older/larger/before than Y' decompose "
        "into the comparison AND each side's underlying fact when present.\n"
        "  R3. Definite articles ('the X', 'the only Y') assert UNIQUENESS — "
        "extract the uniqueness claim explicitly when it's non-trivial.\n"
        "  R4. Apposition and embedded clauses ('A, who did B, lived in C') "
        "decompose into separate claims for B and C.\n"
        "  R5. Resolve pronouns. Drop hedges and conversational filler. "
        "Output one claim per line, no numbering, no introduction. "
        "If the ANSWER makes no verifiable factual claims (refusal, pure "
        "opinion), output nothing.\n\n"
        "EXAMPLE 1 — simple\n"
        "ANSWER: He was a German-born theoretical physicist best known for "
        "developing the theory of relativity. He also made important "
        "contributions to quantum mechanics.\n"
        "CLAIMS:\n"
        "Albert Einstein was a German-born theoretical physicist.\n"
        "Albert Einstein was best known for developing the theory of relativity.\n"
        "Albert Einstein made important contributions to quantum mechanics.\n\n"
        "EXAMPLE 2 — comparison\n"
        "ANSWER: Annie Morton is older than Terry Richardson.\n"
        "CLAIMS:\n"
        "Annie Morton is older than Terry Richardson.\n"
        "Annie Morton has a date of birth.\n"
        "Terry Richardson has a date of birth.\n"
        "Annie Morton's date of birth is earlier than Terry Richardson's.\n\n"
        "EXAMPLE 3 — compound attribution with definite article\n"
        "ANSWER: Arena of Khazan is the adventure in 'Tunnels & Trolls' "
        "designed by Ken St. Andre.\n"
        "CLAIMS:\n"
        "Arena of Khazan is an adventure in Tunnels & Trolls.\n"
        "Arena of Khazan is the ONLY adventure in Tunnels & Trolls "
        "designed by Ken St. Andre.\n"
        "Arena of Khazan was designed by Ken St. Andre.\n\n"
        "EXAMPLE 4 — apposition / cross-attribution\n"
        "ANSWER: The singer of 'A Rather Blustery Day', Jim Cummings, "
        "voiced Miles 'Tails' Prower in the video game series 'Sonic the Hedgehog.'\n"
        "CLAIMS:\n"
        "Jim Cummings is the singer of 'A Rather Blustery Day'.\n"
        "Jim Cummings voiced Miles 'Tails' Prower.\n"
        "Miles 'Tails' Prower is a character in the video game series 'Sonic the Hedgehog'.\n\n"
        f"NOW DO THIS ONE\n"
        f"ANSWER: {answer}\n"
        f"CLAIMS:"
    )


def claim_verification_batched_v0(context: str, claims: list[str]) -> str:
    """Mirror of the current Rust batched prompt."""
    numbered = "".join(f"{i+1}. {c}\n" for i, c in enumerate(claims))
    return (
        f"CONTEXT:\n{context}\n\nClaims to verify against the CONTEXT:\n{numbered}\n"
        "For EACH numbered claim, output ONE line in this exact format:\n"
        "N: SCORE\n"
        "where N is the claim number and SCORE is a number from 0 (unsupported "
        "or contradicted) to 1 (fully supported). No commentary. No grouping. "
        "One line per claim, in the same order.\n\n"
        "Example output for 3 claims:\n"
        "1: 0.9\n"
        "2: 0.2\n"
        "3: 0.7"
    )


def claim_verification_batched_v1(context: str, claims: list[str]) -> str:
    """v1: stricter rubric. Spells out that 'not mentioned' → 0, that
    real-world knowledge ('I happen to know this is true') doesn't
    count, and that comparative/uniqueness claims need explicit textual
    support on every component."""
    numbered = "".join(f"{i+1}. {c}\n" for i, c in enumerate(claims))
    return (
        f"CONTEXT:\n{context}\n\nClaims to verify against the CONTEXT:\n{numbered}\n"
        "For EACH claim, judge support STRICTLY against the CONTEXT only:\n"
        "  • SCORE 1.0 — every part of the claim is explicitly stated or directly entailed by the CONTEXT.\n"
        "  • SCORE 0.5 — partial support: some parts are in the CONTEXT, others are absent or only weakly implied.\n"
        "  • SCORE 0.0 — at least one part of the claim is NOT in the CONTEXT, is contradicted, or relies on outside knowledge.\n"
        "RULES:\n"
        "  • 'Not mentioned in context' is SCORE 0, not SCORE 1. The CONTEXT is the only source of truth.\n"
        "  • Outside / world knowledge does NOT count as support. If you know it's true but the CONTEXT does not say so, that's SCORE 0.\n"
        "  • For comparative claims (older than / before / largest …), BOTH sides must be grounded in the CONTEXT.\n"
        "  • For uniqueness claims ('the only', 'the first', 'the X designed by Y'), the CONTEXT must rule out alternatives or explicitly assert uniqueness.\n"
        "Output ONE line per claim in this exact format:\n"
        "N: SCORE\n"
        "No commentary. One line per claim, in the same order.\n\n"
        "Example output for 3 claims:\n"
        "1: 1.0\n"
        "2: 0.0\n"
        "3: 0.5"
    )


def claim_extraction_v2(answer: str) -> str:
    """v2: same as v0 (proven adequate on most cases) plus ONE targeted
    rule for comparatives — split 'X is older/larger/before Y' into the
    comparison claim itself. Does NOT add vacuous 'X has a date of
    birth' style claims (v1 lesson). Does NOT decompose cross-attributions
    into innocent parts (v1 case-C regression: 'JC voiced Tails in Sonic'
    must stay as one claim, otherwise 'Tails is in Sonic' (true) dilutes
    the false 'JC voiced Tails')."""
    return (
        "Decompose an ANSWER into atomic factual claims for verification. "
        "Each claim must be self-contained — resolve pronouns, drop "
        "conversational filler. Keep COMPOUND attributions ('X did Y in Z') "
        "as a single claim — do NOT split into innocent components, which "
        "would dilute a false core claim. Output one claim per line, no "
        "numbering, no introduction. If the ANSWER makes no verifiable "
        "factual claims (refusal, pure opinion), output nothing.\n\n"
        "EXAMPLE 1 — simple\n"
        "ANSWER: He was a German-born theoretical physicist best known for "
        "developing the theory of relativity. He also made important "
        "contributions to quantum mechanics.\n"
        "CLAIMS:\n"
        "Albert Einstein was a German-born theoretical physicist.\n"
        "Albert Einstein was best known for developing the theory of relativity.\n"
        "Albert Einstein made important contributions to quantum mechanics.\n\n"
        "EXAMPLE 2 — apposition (do NOT split the compound 'voiced X in Y')\n"
        "ANSWER: The singer of 'A', Jim Cummings, voiced Tails in 'Sonic'.\n"
        "CLAIMS:\n"
        "Jim Cummings is the singer of 'A'.\n"
        "Jim Cummings voiced Tails in 'Sonic'.\n\n"
        f"NOW DO THIS ONE\n"
        f"ANSWER: {answer}\n"
        f"CLAIMS:"
    )


def claim_verification_batched_v2(context: str, claims: list[str]) -> str:
    """v2: keeps the v0 output format ('N: SCORE') for parser
    compatibility but adds an explicit worked example showing that
    'context does not mention X' must score 0, not 1. This is the
    structural fix for case A: the verifier was hallucinating support
    from world knowledge.

    Two worked examples — one positive ('explicitly stated, score 1'),
    one negative ('not in context, score 0'). Plus an explicit
    instruction not to use outside knowledge.
    """
    numbered = "".join(f"{i+1}. {c}\n" for i, c in enumerate(claims))
    return (
        "Judge whether each CLAIM is supported by the CONTEXT. The CONTEXT "
        "is the ONLY source of truth — outside / world knowledge does NOT "
        "count as support. If the CONTEXT does not mention a fact the claim "
        "asserts, the score is 0, even if you know it to be true.\n\n"
        "Scoring rubric:\n"
        "  1.0 — every part of the claim is explicitly stated or directly entailed by the CONTEXT.\n"
        "  0.5 — partial support: some parts in the CONTEXT, others absent.\n"
        "  0.0 — at least one part is NOT in the CONTEXT or is contradicted.\n\n"
        "For comparative claims (older than, before, larger than): the "
        "CONTEXT must contain enough information about BOTH sides to make "
        "the comparison. If one side's underlying attribute is missing, "
        "score 0.\n\n"
        "EXAMPLE 1\n"
        "CONTEXT: Annie Morton (born October 8, 1970) is an American model.\n"
        "CLAIM: Annie Morton is older than Terry Richardson.\n"
        "REASONING: Context gives Annie Morton's birth date but says nothing about Terry Richardson's date of birth. Comparison cannot be made from CONTEXT alone.\n"
        "SCORE: 0\n\n"
        "EXAMPLE 2\n"
        "CONTEXT: WINNER is a South Korean boy group formed in 2014 by YG Entertainment.\n"
        "CLAIM: WINNER was formed by YG Entertainment.\n"
        "REASONING: Explicitly stated in CONTEXT.\n"
        "SCORE: 1\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"Claims to verify against the CONTEXT:\n{numbered}\n"
        "Now for EACH numbered claim above, output ONE line in this exact format:\n"
        "N: SCORE\n"
        "where N is the claim number and SCORE is from 0 to 1. No commentary, "
        "no reasoning in the output — score only. One line per claim, in order.\n\n"
        "Example output for 3 claims:\n"
        "1: 1.0\n"
        "2: 0.0\n"
        "3: 0.5"
    )


def claim_verification_batched_v3(context: str, claims: list[str]) -> str:
    """v3: v2 plus an entailment-positive example. v2's "explicitly
    stated or directly entailed" rubric was being read too literally on
    paraphrase cases (e.g. context says "written by X and Y", claim says
    "X co-wrote" — verifier scored 0). Adding a worked entailment
    example pulls the model back toward accepting paraphrase as
    support, without re-opening the world-knowledge loophole.
    """
    numbered = "".join(f"{i+1}. {c}\n" for i, c in enumerate(claims))
    return (
        "Judge whether each CLAIM is supported by the CONTEXT. The CONTEXT "
        "is the ONLY source of truth — outside / world knowledge does NOT "
        "count as support. If the CONTEXT does not mention a fact the claim "
        "asserts, the score is 0, even if you know it to be true. But "
        "PARAPHRASE counts as support: if the claim restates a fact from "
        "the CONTEXT in different words, that is SCORE 1.\n\n"
        "Scoring rubric:\n"
        "  1.0 — every part of the claim is explicitly stated, paraphrased, or directly entailed by the CONTEXT.\n"
        "  0.5 — partial support: some parts in the CONTEXT, others absent.\n"
        "  0.0 — at least one part is NOT in the CONTEXT or is contradicted.\n\n"
        "For comparative claims (older than, before, larger than): the "
        "CONTEXT must contain enough information about BOTH sides to make "
        "the comparison. If one side's underlying attribute is missing, "
        "score 0.\n\n"
        "EXAMPLE 1 — comparative claim, only one side in CONTEXT\n"
        "CONTEXT: Annie Morton (born October 8, 1970) is an American model.\n"
        "CLAIM: Annie Morton is older than Terry Richardson.\n"
        "REASONING: Context gives Annie Morton's birth date but says nothing about Terry Richardson's date of birth. Comparison cannot be made from CONTEXT alone.\n"
        "SCORE: 0\n\n"
        "EXAMPLE 2 — claim is explicitly stated\n"
        "CONTEXT: WINNER is a South Korean boy group formed in 2014 by YG Entertainment.\n"
        "CLAIM: WINNER was formed by YG Entertainment.\n"
        "REASONING: Explicitly stated in CONTEXT.\n"
        "SCORE: 1\n\n"
        "EXAMPLE 3 — paraphrase / entailment counts as support\n"
        "CONTEXT: The Family Man is a 2000 film written by David Diamond and David Weissman, and starring Nicolas Cage.\n"
        "CLAIM: David Weissman co-wrote The Family Man.\n"
        "REASONING: 'written by David Diamond and David Weissman' directly entails 'David Weissman co-wrote'. Same fact, different surface form.\n"
        "SCORE: 1\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"Claims to verify against the CONTEXT:\n{numbered}\n"
        "Now for EACH numbered claim above, output ONE line in this exact format:\n"
        "N: SCORE\n"
        "where N is the claim number and SCORE is from 0 to 1. No commentary, "
        "no reasoning in the output — score only. One line per claim, in order.\n\n"
        "Example output for 3 claims:\n"
        "1: 1.0\n"
        "2: 0.0\n"
        "3: 0.5"
    )


def claim_verification_batched_v4(context: str, claims: list[str]) -> str:
    """v4: v3 plus a NEGATIVE entailment example. v3's paraphrase rule
    swung the verifier too lenient on wrong-entity attributions
    (e.g. context says X designed game G, claim says X designed
    adventure A for G — different entity; or context says X voiced
    character A in series S, claim says X voiced character B in series
    S — different character). The negative example shows that
    SUBSTITUTING the subject or object of an attribution breaks
    support, even if other parts of the claim are present in the
    CONTEXT.
    """
    numbered = "".join(f"{i+1}. {c}\n" for i, c in enumerate(claims))
    return (
        "Judge whether each CLAIM is supported by the CONTEXT. The CONTEXT "
        "is the ONLY source of truth — outside / world knowledge does NOT "
        "count as support. PARAPHRASE counts as support: same fact in "
        "different words is SCORE 1. But SUBSTITUTION does not: if the "
        "claim swaps the subject, object, or attribute for a similar-but-"
        "different one (different game, different character, different "
        "person), that is SCORE 0 even if the surrounding facts are in "
        "the CONTEXT.\n\n"
        "Scoring rubric:\n"
        "  1.0 — every part of the claim is explicitly stated, paraphrased, or directly entailed by the CONTEXT.\n"
        "  0.5 — partial support: some parts in the CONTEXT, others absent.\n"
        "  0.0 — at least one part is NOT in the CONTEXT, is contradicted, or substitutes a different entity.\n\n"
        "For comparative claims (older than, before, larger than): the "
        "CONTEXT must contain enough information about BOTH sides to make "
        "the comparison. If one side's underlying attribute is missing, "
        "score 0.\n\n"
        "EXAMPLE 1 — comparative, only one side in CONTEXT\n"
        "CONTEXT: Annie Morton (born October 8, 1970) is an American model.\n"
        "CLAIM: Annie Morton is older than Terry Richardson.\n"
        "REASONING: Context gives Annie's birth date but says nothing about Terry's. Comparison cannot be made.\n"
        "SCORE: 0\n\n"
        "EXAMPLE 2 — explicit\n"
        "CONTEXT: WINNER is a South Korean boy group formed in 2014 by YG Entertainment.\n"
        "CLAIM: WINNER was formed by YG Entertainment.\n"
        "SCORE: 1\n\n"
        "EXAMPLE 3 — paraphrase / entailment\n"
        "CONTEXT: The Family Man is a 2000 film written by David Diamond and David Weissman.\n"
        "CLAIM: David Weissman co-wrote The Family Man.\n"
        "REASONING: 'written by Diamond and Weissman' directly entails 'Weissman co-wrote'. Same fact, paraphrased.\n"
        "SCORE: 1\n\n"
        "EXAMPLE 4 — wrong entity substitution\n"
        "CONTEXT: Voice actor Mira Vance is known for voicing the character Echo in the animated series 'Starlight Coast'.\n"
        "CLAIM: Mira Vance voiced the character Lumen in 'Starlight Coast'.\n"
        "REASONING: Context says Echo, claim says Lumen. Different character. Substituting the object of the attribution breaks support.\n"
        "SCORE: 0\n\n"
        "EXAMPLE 5 — attribution to a related but different thing\n"
        "CONTEXT: The role-playing game 'Cinderpeak' was designed by Lana Ortiz. 'Caverns of Ash' is an adventure module for 'Cinderpeak' published by Riverstone.\n"
        "CLAIM: 'Caverns of Ash' was designed by Lana Ortiz.\n"
        "REASONING: Context says Ortiz designed the GAME, not the adventure module. The adventure's designer is not specified.\n"
        "SCORE: 0\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"Claims to verify against the CONTEXT:\n{numbered}\n"
        "Now for EACH numbered claim above, output ONE line in this exact format:\n"
        "N: SCORE\n"
        "where N is the claim number and SCORE is from 0 to 1. No commentary, "
        "no reasoning in the output — score only. One line per claim, in order.\n\n"
        "Example output for 3 claims:\n"
        "1: 1.0\n"
        "2: 0.0\n"
        "3: 0.5"
    )


VARIANTS: dict[str, tuple[Callable[[str], str], Callable[[str, list[str]], str]]] = {
    "v0": (claim_extraction_v0, claim_verification_batched_v0),
    "v1": (claim_extraction_v1, claim_verification_batched_v1),
    "v2": (claim_extraction_v2, claim_verification_batched_v2),
    "v3": (claim_extraction_v2, claim_verification_batched_v3),
    "v4": (claim_extraction_v2, claim_verification_batched_v4),
}


# ── Parsers (same shape as Rust) ────────────────────────────────────────────

def parse_claims(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # strip 1. / 1) prefix
        m = re.match(r"^\d+[.)]\s*(.*)$", s)
        if m:
            s = m.group(1)
        s = s.lstrip("-*").strip()
        if s:
            out.append(s)
    return out


def parse_batched_scores(text: str, n_claims: int) -> list[float]:
    scores = [float("nan")] * n_claims
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        idx_str, score_str = line.split(":", 1)
        digits = re.search(r"\d+", idx_str)
        if not digits:
            continue
        idx = int(digits.group(0))
        if idx == 0 or idx > n_claims:
            continue
        m = re.search(r"[-+]?\d*\.?\d+", score_str)
        if m:
            try:
                scores[idx - 1] = max(0.0, min(1.0, float(m.group(0))))
            except ValueError:
                pass
    return scores


# ── Main ────────────────────────────────────────────────────────────────────

TARGET_QIDS = [
    "5a7bbb64554299042af8f7cc",   # comparative — RD=1.0, Claude=0.0
    "5a8a3e745542996c9b8d5e70",   # compound attribution — RD=1.0, Claude=0.0
    "5ae6050f55429929b0807a5e",   # apposition cross-attr — RD=0.667, Claude=0.0
    # Two cases we GOT RIGHT, to make sure variants don't regress on them:
    "5a8e3ea95542995a26add48d",   # RD=1.0, Claude=1.0 (Adriana Trigiani)
    "5abd94525542992ac4f382d2",   # RD=1.0, Claude=1.0 (2014 S/S / WINNER)
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v0", choices=list(VARIANTS), help="prompt variant")
    p.add_argument(
        "--qids",
        nargs="+",
        default=TARGET_QIDS,
        help="qids to trace (default: 3 failures + 2 controls)",
    )
    p.add_argument(
        "--in",
        dest="input_path",
        default="reports/eval_correlation_hotpot_n25.json",
        help="path to a correlation bench JSON containing the qids' answers",
    )
    args = p.parse_args()

    extract_prompt_fn, verify_prompt_fn = VARIANTS[args.variant]

    n25 = json.loads((REPO / args.input_path).read_text())
    cases_by_qid = {c["qid"]: c for c in n25["cases"]}
    third_judge_path = REPO / "reports/eval_third_judge_eval_correlation_hotpot_n25.json"
    claude_by_qid: dict[str, float] = {}
    if third_judge_path.exists():
        tj = json.loads(third_judge_path.read_text())
        for c, s in zip(n25["cases"], tj["claude_scores"]):
            claude_by_qid[c["qid"]] = s

    hotpot = json.loads((REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())
    hotpot_by_qid = {ex["_id"]: ex for ex in hotpot}

    call, model = _make_client()
    print(f"Model: {model}  Variant: {args.variant}\n")

    for qid in args.qids:
        case = cases_by_qid.get(qid)
        ex = hotpot_by_qid.get(qid)
        if case is None or ex is None:
            print(f"  {qid}: not found, skipping")
            continue
        paras = {title: sents for title, sents in ex["context"]}
        ctx_text = "\n\n".join(" ".join(paras[t]) for t in paras)
        answer = case["answer"]
        rd = case["redhop_decomposed"]
        rg = case["ragas_faithfulness"]
        cl = claude_by_qid.get(qid, float("nan"))

        print("=" * 90)
        print(f"  qid: {qid}")
        print(f"  Q: {ex['question']}")
        print(f"  A: {answer}")
        print(
            f"  scored: RedHop_d={rd:.2f}  Ragas={rg:.2f}  Claude={cl:.2f}"
        )
        print(f"  context: {len(ctx_text)} chars across {len(paras)} paragraphs")
        print("-" * 90)

        # Extraction
        extracted_text = call(CLAIM_EXTRACTION_SYSTEM, extract_prompt_fn(answer))
        claims = parse_claims(extracted_text)
        print(f"  extracted {len(claims)} claim(s):")
        for i, c in enumerate(claims, 1):
            print(f"    {i}. {c}")

        if not claims:
            print("    (no claims — would return None)")
            continue

        # Verification
        verify_text = call(
            CLAIM_VERIFICATION_SYSTEM, verify_prompt_fn(ctx_text, claims)
        )
        scores = parse_batched_scores(verify_text, len(claims))
        print(f"  per-claim verification scores:")
        for i, (c, s) in enumerate(zip(claims, scores), 1):
            s_disp = f"{s:.2f}" if s == s else "NaN"
            tag = "✓" if (s == s and s >= 0.5) else "✗"
            print(f"    {i}. [{tag} {s_disp}] {c}")
        clean = [0.0 if s != s else s for s in scores]
        mean = sum(clean) / len(clean)
        supported = sum(1 for s in clean if s >= 0.5)
        print(f"  → faithfulness({args.variant}) = {mean:.3f}  ({supported}/{len(claims)} supported)")
        print(f"  → delta from Claude: {mean - cl:+.2f}" if cl == cl else "")
        print()


if __name__ == "__main__":
    main()
