"""
Step 4 — Evaluation Harness: Step 8, Error Analysis.

STUB. Filled in after the harness is frozen (eval/PLAN.md §7) and baseline
results exist (Step 5+). Interfaces are documented now so downstream code
(and whoever picks this up) can import against a stable signature before
the implementation lands, per eval/PLAN.md §3's file layout.

Domain-agnostic per eval/PLAN.md §0 -- once implemented, this file must
contain zero domain-specific class/attribute/relation names from any
concrete gold schema, same as schema_ir.py/matching.py/metrics.py.
"""

from __future__ import annotations

from eval.matching import CREConfig, EmbeddingCache
from eval.metrics import ScoringResult
from eval.schema_ir import Schema


def categorize_mismatches(
    gold: Schema,
    induced: Schema,
    result: ScoringResult,
    cfg: CREConfig,
    cache: EmbeddingCache | None = None,
) -> list[dict]:
    """Bucket the FP/FN of a scored run into failure modes -- direction
    reversal (cross-reference against the D3 --allow-inverse variant),
    split-class (an induced class left unmatched that scores highly
    against an ALREADY-matched gold class), missing-attribute,
    taxonomy-flattening, and so on -- for the paper's error-analysis
    table (RQ3/Step 8).

    Expected return shape: one dict per FP/FN item, at minimum
    `{"layer": str, "kind": str, "gold": ..., "induced": ..., "reason": str}`,
    where `kind` is the failure-mode label and `reason` is a short
    human-readable explanation an author can quote directly in the paper.
    """
    raise NotImplementedError(
        "Step 8 (Error Analysis) is a stub -- filled in after the harness "
        "is frozen (PLAN.md §7) and Step 5's first baseline results exist. "
        "See eval/PLAN.md §3 and §9 (Definition of Done)."
    )


def sample_m3_decisions(result: ScoringResult, n: int = 100, seed: int = 42) -> list[dict]:
    """Sample `n` M3 (semantic embedding) match decisions for hand
    verification -- the Definition-of-Done item "100 sampled M3 decisions
    hand-verified, agreement rate recorded" (PLAN.md §9). `result` should
    be a ScoringResult computed at level="M3"; sampling should draw from
    ALL match decisions the run produced (classes, and once
    metrics.py/report.py expose them, per-class-pair attribute matches and
    relation-label matches too -- see report.py's
    result_to_match_decisions() docstring for the current scope), not just
    the class-level ones currently written to the raw match-decisions
    JSON.

    Expected return shape: one dict per sampled decision, at minimum
    `{"layer": str, "gold": str, "induced": str, "score": float,
    "matched": bool}`, ready to be handed to a human reviewer for a
    match/no-match verdict, with the agreement rate against
    `result.<layer>.matched`-derived labels reported as a harness-validity
    number in the paper.
    """
    raise NotImplementedError(
        "Step 8 (Error Analysis) is a stub -- filled in after the harness "
        "is frozen (PLAN.md §7) and Step 5's first baseline results exist. "
        "See eval/PLAN.md §3 and §9 (Definition of Done)."
    )
