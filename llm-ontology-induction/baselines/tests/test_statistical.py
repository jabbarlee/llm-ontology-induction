"""
B1 baseline validation suite.

Every expected value here is hand-computed from the formula in
baselines/DECISIONS.md D2, on a corpus small enough to count by hand, and the
arithmetic is written out in the assertion message. These are not
change-detector tests: if the C-value implementation drifts, the failure names
the exact quantity that moved.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from baselines import term_extraction as te

# ---------------------------------------------------------------------------
# T1 — nesting discount (DECISIONS.md D2)
# ---------------------------------------------------------------------------
#
# Corpus built so the counts are hand-countable:
#   3 sentences contain "maintenance request status"
#   2 sentences contain "maintenance request" with no following head noun
#
# The extractor emits every (ADJ|NOUN)* NOUN span, so each long sentence
# contributes an occurrence of the nested short term too:
#
#   freq("maintenance request")        = 3 (nested) + 2 (standalone) = 5
#   freq("maintenance request status") = 3
#
# "maintenance request" is nested in exactly one longer candidate, so:
#   C-value = length_factor(2) * (5 - mean_container_freq)
#           = log2(3)          * (5 - 3)
#           = 1.5849625007211562 * 2
#           = 3.1699250014423126
#
# Naive frequency scoring would instead give log2(3) * 5 = 7.924812503605781.
# The discount is the whole point of C-value: the short term stops looking
# important merely for being nested inside a longer, more specific one.

_NESTED_CORPUS = [
    "The maintenance request status was updated.",
    "A maintenance request status changed again.",
    "Every maintenance request status matters here.",
    "The maintenance request was filed.",
    "Another maintenance request arrived.",
]

_LF2 = math.log2(3)  # length_factor(2) = log2(2 + 1)


def test_length_factor_never_zeroes_single_token_terms():
    """D2a: log2(|a|) would be 0 for length 1, deleting a whole length class."""
    assert te.length_factor(1) == 1.0
    assert te.length_factor(2) == pytest.approx(1.5849625007211562)
    assert te.length_factor(3) == 2.0
    assert te.length_factor(4) == pytest.approx(2.321928094887362)
    # Monotonic in length -- C-value's multi-word reward is preserved.
    factors = [te.length_factor(n) for n in range(1, 5)]
    assert factors == sorted(factors)


def test_nested_term_frequencies_are_hand_countable():
    terms = te.extract_candidate_terms(_NESTED_CORPUS)
    assert terms["maintenance request"] == 5, (
        "3 nested occurrences + 2 standalone = 5; got "
        f"{terms.get('maintenance request')}"
    )
    assert terms["maintenance request status"] == 3, (
        f"3 long sentences = 3; got {terms.get('maintenance request status')}"
    )


def test_c_value_discounts_a_nested_term_below_its_naive_frequency_score():
    terms = te.extract_candidate_terms(_NESTED_CORPUS)
    nesting = te.build_nesting_map(terms)
    scores = te.c_value(terms, nesting)

    assert nesting["maintenance request"] == ["maintenance request status"]

    expected_discounted = _LF2 * (5 - 3)
    naive = _LF2 * terms["maintenance request"]

    assert scores["maintenance request"] == pytest.approx(expected_discounted), (
        f"C-value = log2(3) * (freq 5 - mean container freq 3) "
        f"= {expected_discounted!r}; got {scores['maintenance request']!r}"
    )
    assert expected_discounted == pytest.approx(3.1699250014423126)
    assert naive == pytest.approx(7.924812503605781)
    assert scores["maintenance request"] < naive, (
        "the nesting discount must pull the short term below its naive "
        f"frequency score ({expected_discounted!r} !< {naive!r})"
    )


def test_unnested_term_scores_are_plain_length_times_frequency():
    """The un-nested branch of the formula, verified on the longest term."""
    terms = te.extract_candidate_terms(_NESTED_CORPUS)
    scores = te.c_value(terms, te.build_nesting_map(terms))
    # length 3, freq 3, nested in nothing -> log2(4) * 3 = 2.0 * 3 = 6.0
    assert scores["maintenance request status"] == pytest.approx(6.0)


def test_nesting_map_uses_token_subsequences_not_substrings():
    """'rate' is a substring of 'corporate' but not nested in it."""
    terms = {"rate": 9, "corporate rate": 4}
    nesting = te.build_nesting_map(terms)
    assert nesting["rate"] == ["corporate rate"]
    assert nesting["corporate rate"] == []

    unrelated = {"rate": 9, "corporate office": 4}
    assert te.build_nesting_map(unrelated)["rate"] == []


# ---------------------------------------------------------------------------
# T2 — SVO extraction (DECISIONS.md D5)
# ---------------------------------------------------------------------------

def test_svo_extraction_on_an_unambiguous_sentence():
    triples = te.extract_svo_triples(
        ["The owner manages the property."], {"owner", "property"}
    )
    assert triples == [("owner", "manage", "property")], (
        f"expected exactly one (subject, verb_lemma, object) triple; got {triples}"
    )


def test_svo_requires_both_endpoints_to_be_class_candidates():
    """An endpoint outside the candidate set makes the triple disappear (D5)."""
    assert te.extract_svo_triples(["The owner manages the property."], {"owner"}) == []
    assert te.extract_svo_triples(["The owner manages the property."], set()) == []


def test_svo_skips_copulas():
    """'X is a Y' is a taxonomy claim, and B1 makes none (D5/D6)."""
    triples = te.extract_svo_triples(
        ["The owner is a party."], {"owner", "party"}
    )
    assert triples == [], f"copula must not yield a relation; got {triples}"


def test_svo_prefers_the_longest_multiword_candidate():
    triples = te.extract_svo_triples(
        ["The vendor resolves the maintenance request."],
        {"vendor", "request", "maintenance request"},
    )
    assert triples == [("vendor", "resolve", "maintenance request")], (
        f"expected the longer candidate to win over bare 'request'; got {triples}"
    )


# ---------------------------------------------------------------------------
# T3 — degenerate input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "corpus",
    [
        pytest.param([], id="empty-corpus"),
        pytest.param([""], id="empty-string"),
        pytest.param(["   "], id="whitespace"),
        pytest.param(["!!! ... ???", "123 456", "$$$"], id="no-nouns"),
    ],
)
def test_degenerate_corpora_return_empty_structures_without_crashing(corpus):
    terms = te.extract_candidate_terms(corpus)
    assert terms == {}
    assert te.build_nesting_map(terms) == {}
    assert te.c_value(terms, {}) == {}
    assert te.surface_forms(corpus) == {}
    assert te.extract_svo_triples(corpus, set()) == []
    assert te.extract_attribute_candidates(corpus, set(), set()) == {}


def test_c_value_of_an_empty_candidate_set_is_empty():
    assert te.c_value({}, {}) == {}


def test_terms_below_the_frequency_floor_are_dropped():
    """MIN_TERM_FREQ = 3 (D7): a term seen twice never reaches C-value."""
    corpus = ["The invoice arrived.", "The invoice arrived."]
    assert "invoice" not in te.extract_candidate_terms(corpus)


# ---------------------------------------------------------------------------
# T4 — D3a surface forms / D3b case-folded tagging
# ---------------------------------------------------------------------------

def test_case_variants_aggregate_into_one_term():
    """D3a: 'Tenant' and 'tenant' are one term, not two sub-threshold halves."""
    corpus = [
        "Tenant records were updated.",
        "The tenant called today.",
        "Our tenant signed it.",
    ]
    terms = te.extract_candidate_terms(corpus)
    assert terms.get("tenant") == 3, (
        f"3 occurrences across mixed casing should aggregate to 3; got {terms}"
    )


def test_surface_form_preserves_original_casing():
    """Rule 2 / D3a: the producer never lowercases its output."""
    corpus = [
        "Tenant records were updated.",
        "Tenant records were archived.",
        "Tenant records were reviewed.",
    ]
    forms = te.surface_forms(corpus)
    assert forms["tenant"] == "Tenant", (
        f"emitted form must be the observed capitalized span; got {forms.get('tenant')}"
    )


def test_title_cased_csv_headers_are_visible_as_terms():
    """D3b: without case-folded tagging these tag PROPN and vanish entirely."""
    row = "Lease Owner: Wagner & Associates, Commence Date: 04/01/2022"
    terms = te.extract_candidate_terms([row] * 3)
    assert "lease owner" in terms
    assert "commence date" in terms
    # Instance data stays out -- proper nouns are excluded (D3/D3b).
    assert not any("wagner" in t for t in terms)


# ---------------------------------------------------------------------------
# T5 — attribute patterns (DECISIONS.md D4)
# ---------------------------------------------------------------------------

_ATTR_TERMS = {"tenant", "name", "tenant name", "start", "date", "start date", "file"}


def test_descriptive_compound_pattern():
    """'the tenant name' -> name is an attribute of tenant (D4)."""
    corpus = ["The tenant name was recorded in the file."] * 3
    hits = te.extract_attribute_candidates(
        corpus, class_candidates={"tenant"}, all_terms=_ATTR_TERMS
    )
    assert hits["tenant"]["name"] == 3, f"expected 3 compound hits; got {dict(hits)}"


def test_possessive_pattern():
    """\"the tenant 's name\" -> name is an attribute of tenant (D4)."""
    corpus = ["The tenant 's name was recorded in the file."] * 3
    hits = te.extract_attribute_candidates(
        corpus, class_candidates={"tenant"}, all_terms=_ATTR_TERMS
    )
    assert hits["tenant"]["name"] == 3, f"expected 3 possessive hits; got {dict(hits)}"


def test_prepositional_pattern_steps_over_the_determiner():
    """'the start date OF THE lease' -- a determiner sits between 'of' and the
    class, so a bare doc[start-1] == 'of' check never fires (D4)."""
    corpus = ["The start date of the lease was moved."] * 3
    hits = te.extract_attribute_candidates(
        corpus,
        class_candidates={"lease"},
        all_terms={"lease", "start", "date", "start date"},
    )
    assert hits["lease"]["start date"] == 3, (
        f"expected 3 prepositional hits; got {dict(hits)}"
    )


def test_an_attribute_is_never_also_a_class_candidate():
    """D4: otherwise the top-ranked terms become one another's attributes."""
    corpus = ["The tenant name was recorded in the file."] * 3
    hits = te.extract_attribute_candidates(
        corpus, class_candidates={"tenant", "name"}, all_terms=_ATTR_TERMS
    )
    assert hits.get("tenant", Counter())["name"] == 0


def test_an_attribute_never_contains_its_own_class():
    """'tenant name' must not become an attribute of 'tenant' (D4)."""
    corpus = ["The tenant name was recorded in the file."] * 3
    hits = te.extract_attribute_candidates(
        corpus, class_candidates={"tenant"}, all_terms=_ATTR_TERMS
    )
    assert "tenant name" not in hits["tenant"]


# ---------------------------------------------------------------------------
# T6 — the anti-oracle tripwire (Critical Rule 1 / DECISIONS.md D3)
# ---------------------------------------------------------------------------

_MODULES = ["term_extraction.py", "statistical.py"]


def _gold_vocabulary() -> set[str]:
    """Every class, attribute and relation name in the gold schema, read from
    the TTL itself so this check cannot drift out of sync with it."""
    from eval.schema_ir import load_gold_ttl

    root = Path(__file__).resolve().parents[2]
    gold = load_gold_ttl(root / "schema" / "gold_schema.ttl")
    vocab: set[str] = set()
    for name, cls in gold.classes.items():
        vocab.add(name)
        vocab |= set(cls.declared_attributes)
    for rel in gold.relations:
        vocab |= {rel.source, rel.label, rel.target}
    return vocab


# Keys the induced-schema contract itself mandates (eval/PLAN.md §2 /
# schema_ir.parse_induced_schema). B1 has no choice about emitting these, and
# gold's `name` attribute happens to collide with the contract's `"name"` key.
# Exempted as literals, but pinned: the allowlist is asserted to be exactly the
# contract's keys, so it cannot be quietly grown into a leak-hiding escape
# hatch (test_contract_key_allowlist_matches_the_contract).
_CONTRACT_KEYS = frozenset(
    {
        "classes", "relations", "metadata",                  # top level
        "name", "parent", "attributes",                      # a class
        "source", "label", "target",                         # a relation
        "condition", "model", "run_id", "source_documents",  # metadata
    }
)


def _executable_vocabulary(source: str) -> set[str]:
    """Every string literal and self-defined identifier in `source`, normalized.

    Critical Rule 1 exists to stop gold vocabulary from *influencing B1's
    behavior* -- a seed list, a filter, a validation set. This function
    collects what could do that, and deliberately excludes three things that
    cannot:

    - **Comments and docstrings.** Prose explaining a methodological decision
      (D3b cites the corpus terms that motivated it) influences nothing, and
      banning it would push the reasoning out of the code implementing it.
    - **Attribute accesses** (`Path(...).resolve()`, `path.name`). That is
      third-party/stdlib API surface this module does not choose. A real seed
      list is string literals or a dict, both still collected.
    - **Contract keys** (`"name"`, `"source"`, ...), per _CONTRACT_KEYS above.

    Docstrings are skipped by walking the AST and dropping the leading
    expression statement of every module/class/function body.
    """
    import ast

    from eval.matching import normalize

    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    # Python builtins are exempt as IDENTIFIERS only: gold's relation label
    # `lists` normalizes to "list", which collides with the builtin used in
    # every type annotation. A bare `list` cannot act as domain vocabulary.
    # String literals get no such exemption -- a seed list `["lists", ...]`
    # is still caught.
    import builtins

    builtin_names = {normalize(n) for n in dir(builtins)}

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                if node.value not in _CONTRACT_KEYS:
                    found.add(normalize(node.value))
            continue
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.arg):
            identifier = node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifier = node.name
        else:
            continue
        norm = normalize(identifier)
        if norm not in builtin_names:
            found.add(norm)
    return {f for f in found if f}


def test_contract_key_allowlist_contains_no_gold_vocabulary():
    """The allowlist must never become a way to silence the tripwire.

    Anyone adding a domain term to _CONTRACT_KEYS -- deliberately or by
    copy-paste -- would make the leakage test go quiet for that term. This
    asserts the allowlist holds only structural keys, never gold vocabulary.

    `name` is the one genuine collision: gold declares an attribute called
    `name`, and the contract independently requires a `"name"` key on every
    class. It is exempted because B1 cannot emit valid output without it.
    """
    from eval.matching import normalize

    gold = {normalize(v) for v in _gold_vocabulary()}
    overlap = {k for k in _CONTRACT_KEYS if normalize(k) in gold}
    assert overlap == {"name"}, (
        f"allowlist overlaps gold vocabulary beyond the unavoidable 'name': {overlap}"
    )


def test_contract_key_allowlist_is_actually_the_contract():
    """Every allowlisted key must be one parse_induced_schema really reads."""
    import inspect

    from eval import schema_ir

    source = inspect.getsource(schema_ir.parse_induced_schema) + inspect.getsource(
        schema_ir.load_induced_metadata
    )
    structural = {"metadata", "condition", "model", "run_id", "source_documents"}
    for key in _CONTRACT_KEYS - structural:
        assert f'"{key}"' in source, (
            f"{key!r} is allowlisted but parse_induced_schema never reads it"
        )


def test_no_domain_vocabulary_leakage():
    """B1 must discover its terms from the corpus, never be handed them.

    Any gold class/attribute/relation name appearing as executable vocabulary
    in the extraction modules would make B1 an oracle and invalidate every
    downstream comparison in the paper (Critical Rule 1).

    Matching uses the harness's own `normalize`, so a leak is caught in any
    spelling: "MaintenanceRequest", "maintenance_request" and
    "maintenance requests" all normalize to the same string.

    A single-token gold term must equal a whole literal/identifier to count
    (otherwise the helper `model_name` would trip on gold's `name`); a
    multi-token gold term counts if it appears as a contiguous run of words.
    """
    from eval.matching import normalize

    root = Path(__file__).resolve().parents[1]
    vocab = {normalize(v) for v in _gold_vocabulary()}
    vocab.discard("")
    assert vocab, "gold vocabulary came back empty -- the check would be vacuous"

    leaks: list[str] = []
    for module in _MODULES:
        path = root / module
        if not path.exists():
            continue
        used = _executable_vocabulary(path.read_text())
        for term in sorted(vocab):
            needle = tuple(term.split())
            for item in used:
                words = tuple(item.split())
                hit = (
                    item == term
                    if len(needle) == 1
                    else te._contains_subsequence(words, needle)
                )
                if hit:
                    leaks.append(f"{module}: gold {term!r} appears as {item!r}")
    assert not leaks, (
        "gold-schema vocabulary found in B1 executable code (Critical Rule 1):\n"
        + "\n".join(sorted(set(leaks)))
    )


def test_leakage_check_actually_catches_a_planted_seed_list():
    """The tripwire must fail on real leakage, or it proves nothing.

    Guards against the check silently degrading into a no-op -- the failure
    mode that would let an oracle baseline through unnoticed.
    """
    from eval.matching import normalize

    planted = 'SEEDS = ["Owner", "Tenant", "MaintenanceRequest"]\n'
    used = _executable_vocabulary(planted)
    vocab = {normalize(v) for v in _gold_vocabulary()}
    assert used & vocab, (
        f"planted seed list was not detected; extracted {used!r}"
    )
