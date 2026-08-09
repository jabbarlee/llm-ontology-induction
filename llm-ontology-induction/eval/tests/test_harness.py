"""
Step 4 — Evaluation Harness: toy validation suite ("trust the ruler before
you use it"). See eval/PLAN.md §6 for the fixture table and expected values,
and eval/DECISIONS.md for the D1-D6 design decisions each test enforces.

Built incrementally alongside each build phase (PLAN.md §5: "each phase is
testable before the next exists") rather than written in one pass at the end.
Currently covers Phase 1 (schema_ir.py). T1-T9 land in Phase 6b once
matching.py and metrics.py exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.schema_ir import effective_attributes, load_gold_ttl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_TTL_PATH = REPO_ROOT / "schema" / "gold_schema.ttl"


@pytest.fixture(scope="session")
def gold():
    return load_gold_ttl(GOLD_TTL_PATH)


# ---------------------------------------------------------------------------
# Phase 1 — schema_ir.py acceptance checks
# ---------------------------------------------------------------------------

def test_gold_schema_counts(gold):
    """Loader-derived counts, not hardcoded literals (per the count
    discrepancy found while planning: schema_notes.md/PLAN.md's summary
    lines say "10 classes / 27 attributes" but the TTL itself declares 11
    classes / 30 attributes -- the loader is the source of truth, tests
    must never re-encode a stale doc number)."""
    n_classes = len(gold.classes)
    n_attrs = sum(len(c.declared_attributes) for c in gold.classes.values())
    n_relations = len(gold.relations)

    # Sanity floor: whatever the TTL says today, these must be internally
    # consistent and non-trivial -- catches a loader regression that
    # silently returns an empty/near-empty schema.
    assert n_classes > 0
    assert n_attrs > 0
    assert n_relations == 15, (
        "D1 flattening check: expected exactly 15 relation triples "
        f"(hasParty/represents sub-properties flattened to their parent "
        f"label), got {n_relations}"
    )

    # Known-good values for the current gold_schema.ttl, asserted directly
    # so a future silent schema edit is caught here rather than only
    # surfacing as a confusing downstream metric shift.
    assert n_classes == 11
    assert n_attrs == 30


def test_owner_effective_attributes_include_inherited(gold):
    """D2 check: Owner has no *declared* name/contactInfo (those live on
    Party), but effective_attributes must include them via inheritance."""
    declared = gold.classes["Owner"].declared_attributes
    assert "name" not in declared
    assert "contactInfo" not in declared

    effective = effective_attributes(gold, "Owner")
    assert {"name", "contactInfo", "ownerType", "taxId"} <= effective


def test_effective_attributes_tolerates_dangling_parent():
    """D2 walk must not raise on a class whose parent isn't itself declared
    (PLAN.md §2's own contract example: Building's parent "Asset" is never
    listed) -- it should just stop the walk there."""
    from eval.schema_ir import Schema, ClassDef

    schema = Schema(
        classes={
            "Building": ClassDef(
                name="Building", parent="Asset", declared_attributes=frozenset({"x"})
            )
        },
        relations=frozenset(),
    )
    assert effective_attributes(schema, "Building") == frozenset({"x"})


def test_induced_json_no_precleaning():
    """PLAN.md §2 hard rule: the harness normalizes, the pipeline never
    pre-cleans. parse_induced_schema must preserve messy names verbatim."""
    from eval.schema_ir import parse_induced_schema

    data = {
        "classes": [
            {
                "name": "MaintenanceRequest_TYPE!!",
                "parent": None,
                "attributes": ["Status Field"],
            }
        ],
        "relations": [],
    }
    schema = parse_induced_schema(data)
    assert "MaintenanceRequest_TYPE!!" in schema.classes
    assert "Status Field" in schema.classes["MaintenanceRequest_TYPE!!"].declared_attributes


# ---------------------------------------------------------------------------
# Phase 2 — matching.py M1 normalization
# ---------------------------------------------------------------------------

def test_m1_normalization_equivalences():
    """PLAN.md Phase 2 acceptance: camelCase, snake_case, and Title Case
    plural variants of the same underlying name all normalize identically."""
    from eval.matching import normalize

    variants = ["MaintenanceRequest", "maintenance_request", "Maintenance Requests"]
    normalized = {normalize(v) for v in variants}
    assert len(normalized) == 1, f"expected one normalized form, got {normalized}"


def test_m1_does_not_conflate_unrelated_singular_plural_forms():
    from eval.matching import normalize

    # A hand-rolled "-y -> -ie" singularizer would mangle this; inflect
    # should not (PLAN.md §8's named pitfall).
    assert normalize("Properties") != normalize("Premises")


# ---------------------------------------------------------------------------
# Phase 3 — matching.py M2 fuzzy + lexicon
# ---------------------------------------------------------------------------

CRE_CONFIG_PATH = REPO_ROOT / "eval" / "config" / "cre.yaml"


@pytest.fixture(scope="session")
def cre_config():
    from eval.matching import load_cre_config

    return load_cre_config(CRE_CONFIG_PATH)


def test_m2_lexicon_match_is_threshold_independent(cre_config):
    """A known synonym pair matches via the lexicon regardless of the
    (currently provisional, per PLAN.md §7) fuzzy threshold value -- the
    lexicon short-circuits to score 1.0."""
    from eval.matching import is_m2_match, m2_score

    assert m2_score("landlord", "owner", cre_config) == 1.0
    assert is_m2_match("landlord", "owner", cre_config)


def test_m2_does_not_match_unrelated_near_spelling(cre_config):
    """PLAN.md Phase 3 acceptance: "lease" must not match "least" despite
    scoring high on raw edit similarity (verified: both score 80/100 on
    token-set ratio, since it degenerates to plain Levenshtein-ratio for
    single-token strings). Exercises the *current provisional* threshold in
    config/cre.yaml (frozen: false) -- must be re-verified once §7's
    hand-labeling exercise freezes real threshold values."""
    from eval.matching import is_m2_match

    assert not is_m2_match("lease", "least", cre_config)


# ---------------------------------------------------------------------------
# Phase 4 — matching.py M3 semantic embeddings
# ---------------------------------------------------------------------------

EMBEDDING_CACHE_PATH = REPO_ROOT / "eval" / ".cache" / "embeddings.json"


@pytest.fixture(scope="session")
def embedding_cache():
    from eval.matching import EmbeddingCache

    cache = EmbeddingCache(EMBEDDING_CACHE_PATH)
    yield cache
    cache.flush()


def test_m3_semantic_ordering(cre_config, embedding_cache):
    """PLAN.md Phase 4 acceptance, restated as an ordering assertion rather
    than a hard threshold pass/fail -- m3_cosine_similarity in cre.yaml is
    still provisional (§7 not yet run). "premises"/"property" (a real
    synonym pair) must score higher than "vendor"/"tenant" (unrelated
    roles), regardless of where the eventual frozen threshold lands."""
    from eval.matching import m3_score

    related = m3_score("premises", "property", cre_config, embedding_cache)
    unrelated = m3_score("vendor", "tenant", cre_config, embedding_cache)
    assert related > unrelated


def test_m3_cache_roundtrip(tmp_path, cre_config):
    """A second score computed with a fresh EmbeddingCache instance pointed
    at the same (now-populated) file must not re-embed -- it should read
    straight from disk and return the same value."""
    from eval.matching import EmbeddingCache, m3_score, normalize

    cache_path = tmp_path / "embeddings.json"
    cache1 = EmbeddingCache(cache_path)
    score1 = m3_score("premises", "property", cre_config, cache1)
    cache1.flush()

    cache2 = EmbeddingCache(cache_path)
    normalized_premises = normalize("premises")
    assert cache2.get(cre_config.embedding.model_name, cre_config.embedding.model_version, normalized_premises) is not None
    score2 = m3_score("premises", "property", cre_config, cache2)
    assert score1 == pytest.approx(score2)


# ---------------------------------------------------------------------------
# Phase 5 — matching.py bipartite assignment
# ---------------------------------------------------------------------------

def test_match_sets_split_class_never_double_counts(cre_config):
    """T7 preview (full T7 fixture lands in Phase 6b): gold has one class,
    induced has two plausible matches for it (both are lexicon synonyms of
    the gold name, so both would score a lexicon-hit 1.0 individually).
    The optimal 1x2 assignment must produce exactly one match, never two --
    proves match_sets() doesn't double-count via greedy-style matching."""
    from eval.matching import match_sets

    result = match_sets(["Tenant"], ["Renter", "Lessee"], "M2", cre_config)
    assert len(result.matched) == 1
    assert len(result.unmatched_induced) == 1
    assert len(result.unmatched_gold) == 0


def test_match_sets_below_threshold_pairs_are_unmatched(cre_config):
    """An optimizer-chosen pair that doesn't clear the level's threshold
    must not be reported as matched -- it should fall through to both
    unmatched lists instead of being silently accepted as "best available"."""
    from eval.matching import match_sets

    result = match_sets(["Xyzzy123"], ["CompletelyUnrelatedTerm456"], "M2", cre_config)
    assert result.matched == ()
    assert result.unmatched_gold == ("Xyzzy123",)
    assert result.unmatched_induced == ("CompletelyUnrelatedTerm456",)


def test_match_sets_handles_empty_sets(cre_config):
    """D6: empty induced set must not crash -- everything falls into
    unmatched_gold, no exception."""
    from eval.matching import match_sets

    result = match_sets(["Tenant", "Owner"], [], "M1", cre_config)
    assert result.matched == ()
    assert set(result.unmatched_gold) == {"Tenant", "Owner"}
    assert result.unmatched_induced == ()


# ---------------------------------------------------------------------------
# Phase 6 — metrics.py
# ---------------------------------------------------------------------------

def test_score_schema_identity_is_all_ones(gold, cre_config, embedding_cache):
    """Sanity precursor to T1 (full T1 fixture lands in Phase 6b via the
    induced-JSON-contract path): induced == gold must score P=R=F1=1.0 on
    every layer at every level."""
    from eval.schema_ir import Schema
    from eval.metrics import score_schema

    induced = Schema(classes=dict(gold.classes), relations=frozenset(gold.relations))

    for level in ("M1", "M2", "M3"):
        result = score_schema(gold, induced, level, cre_config, embedding_cache, allow_inverse=True)
        assert result.classes.f1 == 1.0, level
        assert result.taxonomy.f1 == 1.0, level
        assert result.attributes.effective_micro.f1 == 1.0, level
        assert result.attributes.declared_micro.f1 == 1.0, level
        assert result.relations.f1 == 1.0, level
        assert result.relations_allow_inverse.f1 == 1.0, level


def test_relation_f1_not_coupled_to_class_f1(gold, cre_config, embedding_cache):
    """§8 named pitfall: 'if relation F1 mysteriously tracks class F1
    exactly, D4 conditioning may be misimplemented.' Drop one gold class
    entirely from the induced schema (tanking class F1) while leaving every
    relation *not touching that class* untouched -- relation F1 must not
    move in lockstep with class F1."""
    from eval.schema_ir import Schema
    from eval.metrics import score_schema, classes_layer

    # Drop a leaf class that participates in relatively few relations
    # (IndustrialProperty has none of its own -- only inherited "concerns"/
    # "covers"/etc. via Property -- so this specifically exercises the
    # "other relations should be unaffected" half of the claim).
    dropped = "IndustrialProperty"
    induced_classes = {name: c for name, c in gold.classes.items() if name != dropped}
    induced = Schema(classes=induced_classes, relations=frozenset(gold.relations))

    class_score, _ = classes_layer(gold, induced, "M1", cre_config, embedding_cache)
    result = score_schema(gold, induced, "M1", cre_config, embedding_cache)

    assert class_score.f1 < 1.0
    assert result.relations.f1 != class_score.f1


def test_attributes_layer_folds_in_unmatched_classes_as_fn(gold, cre_config, embedding_cache):
    """A gold class entirely absent from induced contributes its full
    attribute set as FN (not silently dropped from accounting)."""
    from eval.schema_ir import Schema
    from eval.metrics import attributes_layer, classes_layer
    from eval.schema_ir import effective_attributes

    dropped = "Vendor"
    induced_classes = {name: c for name, c in gold.classes.items() if name != dropped}
    induced = Schema(classes=induced_classes, relations=frozenset())

    _, class_match = classes_layer(gold, induced, "M1", cre_config, embedding_cache)
    scores = attributes_layer(gold, induced, class_match, "M1", cre_config, embedding_cache)

    dropped_attr_count = len(effective_attributes(gold, dropped))
    assert scores.effective_micro.fn >= dropped_attr_count


# ---------------------------------------------------------------------------
# Phase 6b — the toy validation suite proper (T1, T3-T9). Fixtures are
# generated from the loaded gold schema by eval/tests/toy_schemas/builders.py
# (T2 is hand-written directly, see toy_schemas/t2_empty.json).
#
# Expected values below are hand-computed against the ACTUAL gold schema (11
# classes / 30 attributes / 15 relations -- see test_gold_schema_counts),
# not against PLAN.md §6's illustrative "10/15" numbers. Two fixtures also
# deviate from PLAN.md's simplified table for reasons verified and recorded
# in eval/DECISIONS.md's 2026-08-07 addendum:
#   - T8 strict relation F1 is 1/15 (~0.067), not exactly 0.0, because
#     hasAmendment is a self-referential relation (source == target) and is
#     therefore a fixed point under direction-reversal -- it still matches
#     even in the "reversed" fixture.
#   - T9 is only asserted to recover at M3, not M2 (see the DECISIONS.md
#     addendum on the M2 token-containment fix and why it structurally
#     narrows what M2 can safely catch).
# ---------------------------------------------------------------------------

TOY_SCHEMAS_DIR = REPO_ROOT / "eval" / "tests" / "toy_schemas"


def _load_toy(stem: str):
    from eval.schema_ir import load_induced_json

    return load_induced_json(TOY_SCHEMAS_DIR / f"{stem}.json")


def test_t1_identity(gold, cre_config, embedding_cache):
    from eval.metrics import score_schema

    induced = _load_toy("t1_identity")
    for level in ("M1", "M2", "M3"):
        r = score_schema(gold, induced, level, cre_config, embedding_cache, allow_inverse=True)
        assert r.classes.f1 == 1.0, level
        assert r.taxonomy.f1 == 1.0, level
        assert r.attributes.effective_micro.f1 == 1.0, level
        assert r.attributes.declared_micro.f1 == 1.0, level
        assert r.relations.f1 == 1.0, level
        assert r.relations_allow_inverse.f1 == 1.0, level


def test_t2_empty(gold, cre_config, embedding_cache):
    from eval.metrics import score_schema

    induced = _load_toy("t2_empty")
    for level in ("M1", "M2", "M3"):
        r = score_schema(gold, induced, level, cre_config, embedding_cache)
        assert r.classes.precision == 0.0 and r.classes.recall == 0.0 and r.classes.f1 == 0.0
        assert r.relations.precision == 0.0 and r.relations.recall == 0.0 and r.relations.f1 == 0.0
        # D6: no exception, and no accidental n/a -- the GOLD side is non-empty,
        # only the induced side is empty, so this is the "score 0.0" case, not
        # the "skip, mark n/a" case (that's reserved for an empty *gold* layer).
        assert r.classes.n_a is False


def test_t3_perfect_rename(gold, cre_config, embedding_cache):
    """"This is the test that proves the levels are actually different from
    each other" (PLAN.md §6). M1 must be well below M2 (which recovers every
    renamed class deterministically via the lexicon); M3 must beat M1 too,
    demonstrating semantic recovery, though not asserted at a hard bound
    since its cosine threshold is still provisional (§7 not yet run) --
    ordering, not an absolute number, is what this fixture is actually
    proving."""
    from eval.metrics import score_schema

    induced = _load_toy("t3_perfect_rename")
    m1 = score_schema(gold, induced, "M1", cre_config, embedding_cache)
    m2 = score_schema(gold, induced, "M2", cre_config, embedding_cache)
    m3 = score_schema(gold, induced, "M3", cre_config, embedding_cache)

    assert m1.classes.f1 < m2.classes.f1
    assert m2.classes.f1 == 1.0  # deterministic: every renamed class is a lexicon hit
    assert m3.classes.f1 > m1.classes.f1


def test_t4_overgeneration(gold, cre_config, embedding_cache):
    """Recall = 1.0 (every gold class still present); precision =
    N_GOLD / (N_GOLD + 5) for the 5 invented, unrelated classes."""
    from eval.metrics import score_schema

    n_gold = len(gold.classes)
    induced = _load_toy("t4_overgeneration")
    for level in ("M1", "M2", "M3"):
        r = score_schema(gold, induced, level, cre_config, embedding_cache)
        assert r.classes.tp == n_gold
        assert r.classes.fp == 5
        assert r.classes.fn == 0
        assert r.classes.recall == 1.0
        assert r.classes.precision == pytest.approx(n_gold / (n_gold + 5))


def test_t5_undergeneration(gold, cre_config, embedding_cache):
    """Precision = 1.0 (every induced class is a real gold class); recall =
    kept/N_GOLD for the kept half."""
    from eval.metrics import score_schema

    n_gold = len(gold.classes)
    n_kept = n_gold // 2
    induced = _load_toy("t5_undergeneration")
    for level in ("M1", "M2", "M3"):
        r = score_schema(gold, induced, level, cre_config, embedding_cache)
        assert r.classes.tp == n_kept
        assert r.classes.fp == 0
        assert r.classes.fn == n_gold - n_kept
        assert r.classes.precision == 1.0
        assert r.classes.recall == pytest.approx(n_kept / n_gold)


def test_t6_flattened_taxonomy(gold, cre_config, embedding_cache):
    """Class F1 = 1.0 (every class still present); taxonomy F1 = 0.0 (no
    parent survives); attribute F1 "still high" under effective scoring
    (D2) -- hand-derived exactly: the induced schema keeps every class's
    own *declared* attributes (30 total, matching 1:1 -> 30 TP, 0 FP), but
    loses every *inherited* one Party/Property's 4+3 children no longer
    pick up (2 attrs x 4 Party children + 2 attrs x 3 Property children =
    14 FN)."""
    from eval.metrics import score_schema, prf1

    induced = _load_toy("t6_flattened_taxonomy")
    expected_attr_score = prf1(tp=30, fp=0, fn=14)

    for level in ("M1", "M2", "M3"):
        r = score_schema(gold, induced, level, cre_config, embedding_cache)
        assert r.classes.f1 == 1.0, level
        assert r.taxonomy.f1 == 0.0, level
        assert r.attributes.effective_micro.tp == expected_attr_score.tp, level
        assert r.attributes.effective_micro.fn == expected_attr_score.fn, level
        assert r.attributes.effective_micro.f1 == pytest.approx(expected_attr_score.f1), level


def test_t7_split_class(gold, cre_config, embedding_cache):
    """Phase 5 assignment check, at the metrics layer: at M2 (where the
    lexicon recognizes both split candidates as synonyms of the gold
    class), the optimal bipartite assignment produces exactly 1 TP + 1 FP
    for the split pair -- never 2 TP. (At M1, neither candidate
    exact-string-matches, which is also correct behavior -- just a
    different, not-double-counting-shaped outcome: 0 TP among the pair,
    both FP, gold class FN.)"""
    from eval.metrics import score_schema

    n_gold = len(gold.classes)
    induced = _load_toy("t7_split_class")

    m1 = score_schema(gold, induced, "M1", cre_config, embedding_cache)
    assert m1.classes.tp == n_gold - 1  # every other gold class still matches
    assert m1.classes.fn == 1  # the split gold class itself: neither candidate matched
    assert m1.classes.fp == 2  # both split candidates unmatched

    for level in ("M2", "M3"):
        r = score_schema(gold, induced, level, cre_config, embedding_cache)
        assert r.classes.tp == n_gold, level  # split gold class recovered via ONE candidate
        assert r.classes.fp == 1, level        # the OTHER candidate -- never both, never neither
        assert r.classes.fn == 0, level


def test_t8_reversed_relations(gold, cre_config, embedding_cache):
    """Relation F1 is near-zero strict (not exactly 0.0 -- see the
    DECISIONS.md addendum on the self-referential hasAmendment relation
    being a fixed point under reversal) and 1.0 with allow-inverse (D3)."""
    from eval.metrics import score_schema

    induced = _load_toy("t8_reversed_relations")
    for level in ("M1", "M2", "M3"):
        r = score_schema(gold, induced, level, cre_config, embedding_cache, allow_inverse=True)
        assert r.classes.f1 == 1.0, level  # reversing relations doesn't touch the class set
        assert r.relations.tp == 1, level  # hasAmendment only -- the self-loop fixed point
        assert r.relations.fp == 14, level
        assert r.relations.fn == 14, level
        assert r.relations_allow_inverse.f1 == 1.0, level


def test_t9_subproperty_literalism(gold, cre_config, embedding_cache):
    """Confirms D1 flattening matters: an induced schema emitting the
    pre-flattening OWL sub-property name verbatim fails to match gold's
    flattened label at M1, but IS recovered at M3 (semantic). Per the
    DECISIONS.md addendum, M2 recovery is deliberately NOT asserted here --
    it would require exactly the token-containment leniency that was fixed
    for the T4/Property-OfficeProperty bug, so relying on M2 for this case
    would silently reopen that bug."""
    from eval.metrics import score_schema

    induced = _load_toy("t9_subproperty_literalism")

    m1 = score_schema(gold, induced, "M1", cre_config, embedding_cache)
    assert m1.classes.f1 == 1.0
    assert m1.relations.fn == 4  # the 4 rewritten (hasParty x2, represents x2) edges
    assert m1.relations.fp == 4

    m3 = score_schema(gold, induced, "M3", cre_config, embedding_cache)
    assert m3.relations.tp > m1.relations.tp  # semantic matching recovers some/all of the 4


# ---------------------------------------------------------------------------
# Definition-of-done — domain leakage check (PLAN.md §9)
# ---------------------------------------------------------------------------

import re

# Case-sensitive, whole-word: matches the class names exactly as they'd
# appear if hardcoded (e.g. a literal `"Tenant"` or a comment naming the
# class). Deliberately NOT a lowercase substring check -- "property" as
# generic RDF/OWL vocabulary is unavoidable in schema_ir.py (DatatypeProperty,
# ObjectProperty, sub-property) regardless of domain, and a substring match
# would false-positive on that vocabulary forever. Whole-word + case-sensitive
# targets the actual leak (the domain class name as an identifier) without
# choking on OWL terminology every gold schema's loader legitimately uses.
FORBIDDEN_DOMAIN_STRINGS = ("Tenant", "Lease", "Owner", "Vendor", "Agent", "Property")
CORE_ENGINE_FILES = ("schema_ir.py", "matching.py", "metrics.py")


def test_no_domain_leakage():
    """Zero domain strings anywhere in the core engine (PLAN.md §9's grep
    check, made executable). If this file ever contains the string
    "Tenant", something has leaked from the domain-pack layer into the
    reusable engine layer (PLAN.md §0)."""
    eval_dir = Path(__file__).resolve().parent.parent
    offenders = []
    for filename in CORE_ENGINE_FILES:
        path = eval_dir / filename
        if not path.exists():
            continue  # not built yet -- later phases will add these
        text = path.read_text()
        for term in FORBIDDEN_DOMAIN_STRINGS:
            if re.search(rf"\b{term}\b", text):
                offenders.append((filename, term))
    assert not offenders, f"Domain strings leaked into core engine: {offenders}"
