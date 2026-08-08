"""
Step 4 — Evaluation Harness: P/R/F1 scoring across the four layers (RQ2):
classes, taxonomy, attributes (effective + declared, micro + macro),
relations (D4 endpoint-conditioned, with an --allow-inverse variant per D3).

Domain-agnostic per eval/PLAN.md §0 — this file must contain zero
domain-specific class/attribute/relation names from any concrete gold
schema. See eval/tests/test_harness.py::test_no_domain_leakage for the
enforced check.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from eval.matching import CREConfig, EmbeddingCache, MatchResult, bipartite_match_by_key, match_sets
from eval.schema_ir import Relation, Schema, effective_attributes

LEVELS = ("M1", "M2", "M3")


# ---------------------------------------------------------------------------
# P/R/F1 with raw counts (D6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PRF1:
    """Precision/recall/F1 plus the raw TP/FP/FN counts they're derived
    from, so any degenerate-case convention (D6) is auditable from the
    counts rather than baked invisibly into a single number.

    `n_a=True` marks a layer that was skipped because the *gold* side was
    empty (D6: "empty gold layer -- skip layer, mark n/a"), distinct from
    an empty *induced* side, which scores 0.0/0.0/0.0 normally rather than
    being skipped.
    """

    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    n_a: bool = False


def prf1(tp: int, fp: int, fn: int) -> PRF1:
    """D6: empty induced set (tp+fp == 0 or tp+fn == 0) -> 0.0, never NaN."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return PRF1(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def prf1_na(fp: int) -> PRF1:
    """D6: empty gold layer -- skip, mark n/a rather than reporting a
    (misleadingly precise-looking) 0.0."""
    return PRF1(tp=0, fp=fp, fn=0, precision=None, recall=None, f1=None, n_a=True)


# ---------------------------------------------------------------------------
# Layer 1 — Classes
# ---------------------------------------------------------------------------

def classes_layer(gold: Schema, induced: Schema, level: str, cfg: CREConfig, cache: EmbeddingCache | None = None) -> tuple[PRF1, MatchResult]:
    """Also returns the underlying class-to-class MatchResult: taxonomy,
    attributes, and relations layers are all conditioned on which gold
    classes matched which induced classes here."""
    gold_names = list(gold.classes.keys())
    induced_names = list(induced.classes.keys())

    if not gold_names:
        return prf1_na(fp=len(induced_names)), MatchResult(matched=(), unmatched_gold=(), unmatched_induced=tuple(induced_names))

    class_match = match_sets(gold_names, induced_names, level, cfg, cache)
    score = prf1(
        tp=len(class_match.matched),
        fp=len(class_match.unmatched_induced),
        fn=len(class_match.unmatched_gold),
    )
    return score, class_match


# ---------------------------------------------------------------------------
# Layer 2 — Taxonomy (child, parent) edges, D4-style endpoint conditioning
# ---------------------------------------------------------------------------

def taxonomy_layer(gold: Schema, induced: Schema, class_match: MatchResult) -> PRF1:
    """(child, parent) edges. An edge is only a candidate TP if BOTH
    endpoints (child and parent) have a class-level match -- same
    endpoint-conditioning principle as D4 for relations. A gold edge whose
    child or parent never matched is an automatic FN; every induced edge
    that doesn't end up a TP is an FP (a plain count-difference, matching
    the "automatically FP/FN" language in D4 -- no separate "eligible vs.
    ineligible FP" distinction is drawn, since an induced edge this harness
    can't verify against gold is wrong by default either way)."""
    gold_to_induced = {g: i for g, i, _ in class_match.matched}

    gold_edges = {(c.name, c.parent) for c in gold.classes.values() if c.parent is not None}
    induced_edges = {(c.name, c.parent) for c in induced.classes.values() if c.parent is not None}

    if not gold_edges:
        return prf1_na(fp=len(induced_edges))

    tp = 0
    for child_g, parent_g in gold_edges:
        child_i = gold_to_induced.get(child_g)
        parent_i = gold_to_induced.get(parent_g)
        if child_i is None or parent_i is None:
            continue  # endpoint(s) never matched -> automatic FN
        induced_parent_actual = induced.classes[child_i].parent
        if induced_parent_actual == parent_i:
            tp += 1

    fn = len(gold_edges) - tp
    fp = len(induced_edges) - tp
    return prf1(tp, fp, fn)


# ---------------------------------------------------------------------------
# Layer 3 — Attributes (D2: effective vs. declared; micro + macro)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttributeScores:
    effective_micro: PRF1
    effective_macro: PRF1
    declared_micro: PRF1
    declared_macro: PRF1


def _attribute_set_for(schema: Schema, class_name: str, effective: bool) -> frozenset[str]:
    if effective:
        return effective_attributes(schema, class_name)
    return schema.classes[class_name].declared_attributes


def _attributes_micro_macro(
    gold: Schema,
    induced: Schema,
    class_match: MatchResult,
    level: str,
    cfg: CREConfig,
    cache: EmbeddingCache | None,
    effective: bool,
) -> tuple[PRF1, PRF1]:
    """Micro: pool (class, attr) TP/FP/FN across every matched class pair
    into one global P/R/F1. Macro: per-matched-class F1, then mean.

    Gold classes that never matched at all contribute their full attribute
    set as automatic FN (mirrors D4's endpoint-conditioning principle,
    extended here to the attribute layer: a class the harness couldn't
    even locate in the induced schema can't have any of its attributes
    credited as recovered). They're also included in the macro average at
    F1=0.0 (D6: no induced attributes to consider for this class ->
    precision 0.0, not skipped) -- otherwise macro-averaging over only the
    "easy" matched classes would silently overstate attribute recall.
    """
    gold_to_induced = {g: i for g, i, _ in class_match.matched}

    total_tp = total_fp = total_fn = 0
    per_class_f1: list[float] = []

    for gold_class_name in gold.classes:
        gold_attrs = list(_attribute_set_for(gold, gold_class_name, effective))
        induced_class_name = gold_to_induced.get(gold_class_name)

        if induced_class_name is None:
            # Class never matched -> every gold attribute is an automatic FN.
            total_fn += len(gold_attrs)
            per_class_f1.append(0.0)
            continue

        induced_attrs = list(_attribute_set_for(induced, induced_class_name, effective))
        attr_match = match_sets(gold_attrs, induced_attrs, level, cfg, cache)
        tp = len(attr_match.matched)
        fp = len(attr_match.unmatched_induced)
        fn = len(attr_match.unmatched_gold)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_class_f1.append(prf1(tp, fp, fn).f1)

    micro = prf1(total_tp, total_fp, total_fn)
    macro_f1 = sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0
    # Macro precision/recall aren't separately meaningful (F1 isn't the
    # harmonic mean of macro-P/macro-R in general) -- report macro P/R as
    # the mean of per-class P/R too, for the secondary-column table, but
    # macro F1 (the primary macro number) is the mean of per-class F1
    # above, not derived from macro-P/macro-R.
    macro = PRF1(tp=total_tp, fp=total_fp, fn=total_fn, precision=None, recall=None, f1=macro_f1)
    return micro, macro


def attributes_layer(
    gold: Schema,
    induced: Schema,
    class_match: MatchResult,
    level: str,
    cfg: CREConfig,
    cache: EmbeddingCache | None = None,
) -> AttributeScores:
    effective_micro, effective_macro = _attributes_micro_macro(
        gold, induced, class_match, level, cfg, cache, effective=True
    )
    declared_micro, declared_macro = _attributes_micro_macro(
        gold, induced, class_match, level, cfg, cache, effective=False
    )
    return AttributeScores(
        effective_micro=effective_micro,
        effective_macro=effective_macro,
        declared_micro=declared_micro,
        declared_macro=declared_macro,
    )


# ---------------------------------------------------------------------------
# Layer 4 — Relations (D4 endpoint conditioning; D3 --allow-inverse)
# ---------------------------------------------------------------------------

def _relations_match(
    gold: Schema,
    induced: Schema,
    class_match: MatchResult,
    level: str,
    cfg: CREConfig,
    cache: EmbeddingCache | None,
) -> tuple[frozenset[Relation], frozenset[Relation], frozenset[Relation]]:
    """Returns (tp_gold, fn_gold, fp_induced) as sets of the actual
    Relation objects (not just counts) so allow-inverse scoring (below) can
    union/intersect results across the forward and reversed passes without
    losing track of which specific relation instance is which.

    D4: bucket gold and induced relations by their (source_class,
    target_class) pair, but only using pairs whose BOTH endpoints have a
    class-level match. Within a bucket, relation *labels* are matched via
    the normal M1/M2/M3 bipartite machinery (this is what lets a
    sub-property-literal induced label like the OWL implementation artifact
    named in D1 still match its gold parent-property label at M2/M3, not
    M1 -- see the T9 toy fixture). Any gold/induced relation whose
    endpoints never matched is an automatic FN/FP, per D4.
    """
    gold_to_induced = {g: i for g, i, _ in class_match.matched}
    induced_to_gold = {i: g for g, i, _ in class_match.matched}

    gold_buckets: dict[tuple[str, str], list[Relation]] = defaultdict(list)
    fn_gold: set[Relation] = set()
    for rel in gold.relations:
        if rel.source in gold_to_induced and rel.target in gold_to_induced:
            gold_buckets[(rel.source, rel.target)].append(rel)
        else:
            fn_gold.add(rel)

    induced_buckets: dict[tuple[str, str], list[Relation]] = defaultdict(list)
    fp_induced: set[Relation] = set()
    for rel in induced.relations:
        g_src = induced_to_gold.get(rel.source)
        g_tgt = induced_to_gold.get(rel.target)
        if g_src is not None and g_tgt is not None:
            induced_buckets[(g_src, g_tgt)].append(rel)
        else:
            fp_induced.add(rel)

    tp_gold: set[Relation] = set()
    for key in set(gold_buckets) | set(induced_buckets):
        g_rels = gold_buckets.get(key, [])
        i_rels = induced_buckets.get(key, [])
        if not g_rels:
            fp_induced.update(i_rels)
            continue
        if not i_rels:
            fn_gold.update(g_rels)
            continue
        bucket_match = bipartite_match_by_key(g_rels, i_rels, key=lambda r: r.label, level=level, cfg=cfg, cache=cache)
        tp_gold.update(g for g, _i, _s in bucket_match.matched)
        fn_gold.update(bucket_match.unmatched_gold)
        fp_induced.update(bucket_match.unmatched_induced)

    return frozenset(tp_gold), frozenset(fn_gold), frozenset(fp_induced)


def relations_layer(gold: Schema, induced: Schema, class_match: MatchResult, level: str, cfg: CREConfig, cache: EmbeddingCache | None = None) -> PRF1:
    """Strict direction (D3 primary metric)."""
    if not gold.relations:
        return prf1_na(fp=len(induced.relations))
    tp_gold, fn_gold, fp_induced = _relations_match(gold, induced, class_match, level, cfg, cache)
    return prf1(len(tp_gold), len(fp_induced), len(fn_gold))


def relations_layer_allow_inverse(gold: Schema, induced: Schema, class_match: MatchResult, level: str, cfg: CREConfig, cache: EmbeddingCache | None = None) -> PRF1:
    """D3 robustness variant: a gold relation counts as recovered if it
    matches in EITHER the induced schema's asserted direction or the fully
    reversed direction (a second, independent pass over a direction-swapped
    copy of the induced relations) -- never both passes summed, which
    would double-count. Symmetrically, an induced relation is only an FP
    under this mode if it fails to match gold in *both* orientations.

    Kept entirely separate from relations_layer() (strict mode) so strict
    mode's logic stays simple and auditable, per the build plan."""
    if not gold.relations:
        return prf1_na(fp=len(induced.relations))

    reversed_relations = frozenset(
        Relation(source=r.target, label=r.label, target=r.source) for r in induced.relations
    )
    reversed_induced = Schema(classes=induced.classes, relations=reversed_relations)

    tp_fwd, _fn_fwd, fp_fwd = _relations_match(gold, induced, class_match, level, cfg, cache)
    tp_rev, _fn_rev, fp_rev = _relations_match(gold, reversed_induced, class_match, level, cfg, cache)

    tp_gold = tp_fwd | tp_rev
    fn_gold = gold.relations - tp_gold

    # Map the reversed pass's FP set (reversed-orientation Relation objects)
    # back to original orientation so it can be intersected with the
    # forward pass's FP set on a like-for-like identity.
    fp_rev_mapped = frozenset(Relation(source=r.target, label=r.label, target=r.source) for r in fp_rev)
    fp_induced = fp_fwd & fp_rev_mapped

    return prf1(len(tp_gold), len(fp_induced), len(fn_gold))


# ---------------------------------------------------------------------------
# Orchestrator -- scores all four layers in one call
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringResult:
    level: str
    classes: PRF1
    taxonomy: PRF1
    attributes: AttributeScores
    relations: PRF1
    relations_allow_inverse: PRF1 | None
    class_match: MatchResult  # exposed for report.py / error_analysis.py raw match-decision dumps


def score_schema(
    gold: Schema,
    induced: Schema,
    level: str,
    cfg: CREConfig,
    cache: EmbeddingCache | None = None,
    allow_inverse: bool = False,
) -> ScoringResult:
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level!r} (expected one of {LEVELS})")

    classes_score, class_match = classes_layer(gold, induced, level, cfg, cache)
    taxonomy_score = taxonomy_layer(gold, induced, class_match)
    attribute_scores = attributes_layer(gold, induced, class_match, level, cfg, cache)
    relations_score = relations_layer(gold, induced, class_match, level, cfg, cache)
    relations_inverse_score = (
        relations_layer_allow_inverse(gold, induced, class_match, level, cfg, cache) if allow_inverse else None
    )

    return ScoringResult(
        level=level,
        classes=classes_score,
        taxonomy=taxonomy_score,
        attributes=attribute_scores,
        relations=relations_score,
        relations_allow_inverse=relations_inverse_score,
        class_match=class_match,
    )
