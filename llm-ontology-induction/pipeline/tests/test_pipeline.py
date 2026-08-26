"""
P1 pipeline validation suite (six-stage architecture, pipeline/DECISIONS.md).

Every test here runs offline -- no network call, and every checkpoint used is
either in-memory or under a pytest tmp_path, never the real
`results/raw/.checkpoints/`. Structure mirrors the six stages themselves:

  T0  Critical Rule 1 -- no gold vocabulary in any node module or any of the
      four new prompts (Stage 1's prompt is B3's own and already covered by
      baselines/tests/test_single_shot.py)
  T1  Stage 1 -- extraction (frozen-prompt-by-reference, skip-and-continue)
  T2  Stage 2 -- type consolidation (merge log, class_name_map, single-call-abort)
  T3  Stage 3 -- attribute consolidation
  T4  Stage 4 -- relation reconciliation (deterministic remap + one call)
  T5  Stage 5 -- taxonomy induction (conservatism, declared-supertype rule)
  T6  Stage 6 -- assemble + validate (contract compliance)
  T7  graph.py -- full wiring, checkpointing/resume, dry-run
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.b3_single_shot import single_shot as ss
from baselines.shared import model_clients as mc
from baselines.tests.test_single_shot import _completion, leaks_in_text
from baselines.tests.test_statistical import _CONTRACT_KEYS, _executable_vocabulary, _gold_vocabulary

from pipeline import state as pstate
from pipeline.nodes import (
    _common,
    assemble as assemble_mod,
    consolidate_attrs as attrs_mod,
    consolidate_types as types_mod,
    extract as extract_mod,
    induce_taxonomy as taxonomy_mod,
    reconcile_relations as relations_mod,
)

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _MODULE_ROOT.parent


def _docs(count: int) -> list[tuple[str, str]]:
    return [(f"dir/doc-{i:03d}.txt", f"body of document {i}") for i in range(count)]


# ---------------------------------------------------------------------------
# T0 -- Critical Rule 1, over every new module and every new prompt
# ---------------------------------------------------------------------------

_NODE_MODULES = [
    _MODULE_ROOT / "state.py",
    _MODULE_ROOT / "graph.py",
    _MODULE_ROOT / "nodes" / "_common.py",
    _MODULE_ROOT / "nodes" / "extract.py",
    _MODULE_ROOT / "nodes" / "consolidate_types.py",
    _MODULE_ROOT / "nodes" / "consolidate_attrs.py",
    _MODULE_ROOT / "nodes" / "reconcile_relations.py",
    _MODULE_ROOT / "nodes" / "induce_taxonomy.py",
    _MODULE_ROOT / "nodes" / "assemble.py",
]

_NEW_PROMPTS = [
    _MODULE_ROOT / "prompts" / "p1_consolidation_prompt.md",
    _MODULE_ROOT / "prompts" / "p1_attributes_prompt.md",
    _MODULE_ROOT / "prompts" / "p1_relations_prompt.md",
    _MODULE_ROOT / "prompts" / "p1_taxonomy_prompt.md",
]


def test_no_domain_vocabulary_leakage():
    """Critical Rule 1, over every module this six-stage rework added and
    every prompt it introduced. Stage 1's own prompt is B3's frozen file,
    already covered by baselines/tests/test_single_shot.py -- not rechecked
    here, since doing so would just be testing the same file twice."""
    leaks: list[str] = []

    for path in _NODE_MODULES:
        assert path.exists(), f"{path} not found -- the leakage guard is scanning nothing"
        used = _executable_vocabulary(path.read_text(encoding="utf-8"))
        for term in sorted(_gold_terms_local()):
            needle = tuple(term.split())
            for item in used:
                words = tuple(item.split())
                hit = item == term if len(needle) == 1 else _contains_subsequence_local(words, needle)
                if hit:
                    leaks.append(f"{path.name}: gold {term!r} appears as {item!r}")

    for path in _NEW_PROMPTS:
        assert path.exists(), f"{path} not found -- the leakage guard is scanning nothing"
        for term in leaks_in_text(path.read_text(encoding="utf-8")):
            leaks.append(f"{path.name}: gold {term!r} appears in the prompt text")

    assert not leaks, (
        "gold-schema vocabulary found in the P1 pipeline (Critical Rule 1):\n" + "\n".join(sorted(set(leaks)))
    )


def _gold_terms_local() -> set[str]:
    from eval.matching import normalize

    vocabulary = {normalize(v) for v in _gold_vocabulary()}
    vocabulary.discard("")
    assert vocabulary
    return vocabulary


def _contains_subsequence_local(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    span = len(needle)
    return any(haystack[i : i + span] == needle for i in range(len(haystack) - span + 1))


def test_the_leakage_check_actually_catches_a_planted_term():
    """Proves T0 is not vacuously passing."""
    planted = sorted(_gold_vocabulary())[0]
    assert leaks_in_text(f"Merge classes such as {planted} that mean the same thing.")
    assert not leaks_in_text("Merge classes that mean the same thing.")


def test_all_four_new_prompts_name_the_contract_keys_they_must():
    # p1_relations_prompt.md's own output shape is relations only
    # (source/label/target) -- it has no business mentioning name/parent/
    # attributes, since Stage 4 never touches classes.
    class_shaped = {
        _MODULE_ROOT / "prompts" / "p1_consolidation_prompt.md": ("classes", "name", "parent", "attributes"),
        _MODULE_ROOT / "prompts" / "p1_attributes_prompt.md": ("classes", "name", "parent", "attributes"),
        _MODULE_ROOT / "prompts" / "p1_taxonomy_prompt.md": ("classes", "name", "parent", "attributes"),
        _MODULE_ROOT / "prompts" / "p1_relations_prompt.md": ("relations", "source", "label", "target"),
    }
    for path, required_keys in class_shaped.items():
        text = path.read_text(encoding="utf-8")
        for key in required_keys:
            assert key in text, f"{path.name} never mentions required key {key!r}"


# ---------------------------------------------------------------------------
# T1 -- Stage 1, extraction
# ---------------------------------------------------------------------------

def test_stage_one_uses_the_frozen_b3_prompt_by_reference_not_a_copy():
    """P1-D2: the imported path object must literally be B3's own module
    constant -- not a pipeline/-local file that merely holds identical text
    today and could drift tomorrow."""
    assert extract_mod.B3_PROMPT_PATH is ss.PROMPT_PATH


def test_extract_makes_one_call_per_document(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_invoke(spec, prompt):
        calls.append(prompt)
        return _completion(json.dumps({"classes": [{"name": "Alpha", "parent": None, "attributes": ["a"]}], "relations": []}))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    documents = _docs(4)
    state = {"documents": documents, "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = extract_mod.extract(state)

    assert len(calls) == 4
    assert len(result["partial_schemas"]) == 4
    assert result["extraction_skipped"] == 0
    assert all(s["source_document"] == d[0] for s, d in zip(result["partial_schemas"], documents))
    assert len(log_path.read_text().splitlines()) == 4


def test_one_bad_extraction_call_is_skipped_not_fatal(monkeypatch, tmp_path):
    """192 independent calls -- one truncation, refusal, or unparseable
    response must not sink the run that already paid for the rest."""
    responses = iter(
        [
            _completion(json.dumps({"classes": [{"name": "Alpha", "parent": None, "attributes": []}], "relations": []})),
            _completion("sorry, I cannot help with that"),
            _completion(json.dumps({"classes": [{"name": "Beta", "parent": None, "attributes": []}], "relations": []})),
        ]
    )
    monkeypatch.setattr(mc, "invoke", lambda spec, prompt: next(responses))
    log_path = tmp_path / "calls.jsonl"
    state = {"documents": _docs(3), "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = extract_mod.extract(state)

    assert len(result["partial_schemas"]) == 2
    assert result["extraction_skipped"] == 1
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert "parse_error" in lines[1]


def test_a_truncated_extraction_call_is_recorded_and_skipped(monkeypatch, tmp_path):
    calls = iter(
        [
            mc.TruncatedResponseError("cap", '{"cla', 16000),
            _completion(json.dumps({"classes": [{"name": "Alpha", "parent": None, "attributes": []}], "relations": []})),
        ]
    )

    def fake_invoke(spec, prompt):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"documents": _docs(2), "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = extract_mod.extract(state)

    assert len(result["partial_schemas"]) == 1
    assert result["extraction_skipped"] == 1
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert lines[0]["stop_reason"] == "truncated"
    assert lines[0]["response"] == '{"cla'


# ---------------------------------------------------------------------------
# T2 -- Stage 2, type consolidation
# ---------------------------------------------------------------------------

_PARTIAL_A = {"source_document": "a.txt", "classes": [{"name": "Landlord", "parent": None, "attributes": ["x"]}], "relations": []}
_PARTIAL_B = {"source_document": "b.txt", "classes": [{"name": "Property Owner", "parent": None, "attributes": ["y"]}], "relations": []}


def test_consolidate_types_prompt_carries_the_pre_merged_flat_class_list():
    """render_prompt() takes the already-pre-merged flat class list (P1-D8),
    not the original nested per-document schemas -- occurrences included."""
    rendered = types_mod.render_prompt(
        types_mod.load_prompt_template(),
        [{"name": "Landlord", "parent": None, "attributes": ["x"], "occurrences": 3}],
    )
    assert "Landlord" in rendered
    assert '"occurrences": 3' in rendered


def test_pre_merge_literal_duplicates_collapses_the_same_name_and_counts_occurrences():
    """The actual fix (P1-D8): classes sharing the exact same name across many
    documents collapse into one entry with deduped attributes, not one entry
    per document -- this is what stopped the real repetition-degeneration
    failure documented in results/findings/P1-FINDINGS.md."""
    class_only_schemas = [
        {"source_document": f"doc-{i}.txt", "classes": [{"name": "Tenant", "parent": None, "attributes": ["name"]}]}
        for i in range(40)
    ]
    class_only_schemas.append(
        {"source_document": "doc-x.txt", "classes": [{"name": "Lessee", "parent": None, "attributes": ["name", "business name"]}]}
    )

    merged = types_mod._pre_merge_literal_duplicates(class_only_schemas)

    tenant = next(c for c in merged if c["name"] == "Tenant")
    assert tenant["attributes"] == ["name"]  # not 40 repeats
    assert tenant["occurrences"] == 40

    lessee = next(c for c in merged if c["name"] == "Lessee")
    assert lessee["occurrences"] == 1


def test_pre_merge_literal_duplicates_only_folds_exact_casing_not_different_spellings():
    """Never a substitute for Stage 2's own judgment -- 'Tenant' and 'tenant'
    fold together (a literal casing variant, the same standard clean_schema()
    already applies), but 'Tenant' and 'Lessee' never do, no matter how many
    times either appears."""
    class_only_schemas = [
        {"source_document": "a.txt", "classes": [{"name": "Tenant", "parent": None, "attributes": ["x"]}]},
        {"source_document": "b.txt", "classes": [{"name": "tenant", "parent": None, "attributes": ["y"]}]},
        {"source_document": "c.txt", "classes": [{"name": "Lessee", "parent": None, "attributes": ["z"]}]},
    ]
    merged = types_mod._pre_merge_literal_duplicates(class_only_schemas)
    names = {c["name"] for c in merged}
    assert "Lessee" in names
    assert sum(1 for c in merged if c["name"] in ("Tenant", "tenant")) == 1
    tenant = next(c for c in merged if c["name"] in ("Tenant", "tenant"))
    assert set(tenant["attributes"]) == {"x", "y"}
    assert tenant["occurrences"] == 2


def test_pre_merge_literal_duplicates_picks_the_majority_parent():
    class_only_schemas = [
        {"source_document": "a.txt", "classes": [{"name": "Vendor", "parent": "Party", "attributes": []}]},
        {"source_document": "b.txt", "classes": [{"name": "Vendor", "parent": "Party", "attributes": []}]},
        {"source_document": "c.txt", "classes": [{"name": "Vendor", "parent": "Organization", "attributes": []}]},
    ]
    merged = types_mod._pre_merge_literal_duplicates(class_only_schemas)
    assert merged == [{"name": "Vendor", "parent": "Party", "attributes": [], "occurrences": 3}]


def test_consolidation_prompt_asks_for_identity_only_but_still_shows_attributes():
    """P1-D9. Attributes stay in the *input* -- their overlap is real evidence
    for whether two differently named entries are the same kind of thing --
    but the response shape must not ask for them back. On the real corpus that
    echo was ~13,000 output tokens, 81% of haiku45's whole cap, which is the
    common cause behind all three recorded Stage 2 failures."""
    template = types_mod.load_prompt_template()
    rendered = types_mod.render_prompt(template, [{"name": "Landlord", "parent": None, "attributes": ["x"], "occurrences": 2}])
    assert '"attributes": [\n      "x"\n    ]' in rendered  # shown as input

    output_shape = template.split("## Output shape")[1].split("## Classes")[0]
    assert "merged_from" in output_shape
    assert "attributes" not in output_shape


def test_restore_attributes_rebuilds_the_union_of_exactly_the_merged_entries():
    """The bookkeeping the model used to be asked to do in its own output --
    now Python's, deterministically, off the merged_from names it reports."""
    pre_merged_by_name = {
        "landlord": {"name": "Landlord", "parent": None, "attributes": ["x", "shared"], "occurrences": 3},
        "property owner": {"name": "Property Owner", "parent": None, "attributes": ["SHARED", "y"], "occurrences": 1},
        "vendor": {"name": "Vendor", "parent": None, "attributes": ["z"], "occurrences": 1},
    }
    attributes, unresolved = types_mod._restore_attributes(["Landlord", "Property Owner"], pre_merged_by_name)

    # Union in merged_from order, first surface form of a literal repeat kept
    # (Critical Rule 5) -- "SHARED" never displaces the earlier "shared".
    assert attributes == ["x", "shared", "y"]
    assert unresolved == []
    # An entry that was not merged in contributes nothing.
    assert "z" not in attributes


def test_restore_attributes_reports_a_source_name_that_was_never_in_the_input():
    """Prompt rule 1 forbids inventing a name. If one shows up anyway, that
    class silently loses attributes it should have inherited -- so it is
    counted into the raw record rather than passing unnoticed."""
    attributes, unresolved = types_mod._restore_attributes(
        ["Landlord", "Invented Umbrella Term"],
        {"landlord": {"name": "Landlord", "parent": None, "attributes": ["x"], "occurrences": 1}},
    )
    assert attributes == ["x"]
    assert unresolved == ["Invented Umbrella Term"]


def test_consolidate_types_restores_attributes_the_model_never_sent_back(monkeypatch, tmp_path):
    """End-to-end P1-D9: the response carries no attributes at all, and the
    merged class still comes out with the full union from both source entries."""
    def fake_invoke(spec, prompt):
        payload = {"classes": [{"name": "Landlord", "parent": None, "merged_from": ["Landlord", "Property Owner"]}]}
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": [_PARTIAL_A, _PARTIAL_B], "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = types_mod.consolidate_types(state)

    assert result["merged_classes"] == [
        {"name": "Landlord", "parent": None, "attributes": ["x", "y"], "merged_from": ["Landlord", "Property Owner"]}
    ]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert record["input_entries_unaccounted_for"] == 0
    assert record["unresolved_source_names"] == []


def test_consolidate_types_counts_input_entries_the_model_dropped(monkeypatch, tmp_path):
    """Before P1-D9 a forgotten input entry still left its attributes visible
    in the response text; now it is simply absent, so the count is the only
    thing that makes it visible. Logged, never raised on -- how faithfully
    each model obeys the account-for-every-entry rule is itself a result."""
    def fake_invoke(spec, prompt):
        payload = {"classes": [{"name": "Landlord", "parent": None, "merged_from": ["Landlord"]}]}
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": [_PARTIAL_A, _PARTIAL_B], "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = types_mod.consolidate_types(state)

    assert [c["name"] for c in result["merged_classes"]] == ["Landlord"]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert record["input_entries_unaccounted_for"] == 1  # "Property Owner" was neither kept nor merged


def test_consolidate_types_defaults_merged_from_to_the_class_itself(monkeypatch, tmp_path):
    """merged_from is required by the prompt, but a model may omit it. That
    must fall back to the class's own name rather than losing its attributes."""
    def fake_invoke(spec, prompt):
        return _completion(json.dumps({"classes": [{"name": "Landlord", "parent": None}]}))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": [_PARTIAL_A], "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = types_mod.consolidate_types(state)
    assert result["merged_classes"] == [{"name": "Landlord", "parent": None, "attributes": ["x"], "merged_from": ["Landlord"]}]


def test_consolidate_types_provenance_still_traces_to_the_original_documents(monkeypatch, tmp_path):
    """Provenance (merge_log's source_documents) must reflect every real
    document a name came from, even though the *prompt* only sees the
    pre-merged, collapsed version -- the two are computed independently
    (P1-D8's own docstring)."""
    partial_schemas = [
        {"source_document": f"doc-{i}.txt", "classes": [{"name": "Tenant", "parent": None, "attributes": ["name"]}], "relations": []}
        for i in range(5)
    ]

    def fake_invoke(spec, prompt):
        # Confirms the model was actually shown the collapsed form.
        assert prompt.count('"name": "Tenant"') == 1
        payload = {"classes": [{"name": "Tenant", "parent": None, "attributes": ["name"], "merged_from": ["Tenant"]}]}
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": partial_schemas, "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = types_mod.consolidate_types(state)
    [entry] = result["merge_log"]
    assert entry["source_documents"] == [f"doc-{i}.txt" for i in range(5)]


def test_consolidate_types_builds_a_merge_log_and_class_name_map(monkeypatch, tmp_path):
    def fake_invoke(spec, prompt):
        payload = {
            "classes": [
                {
                    "name": "Landlord",
                    "parent": None,
                    "attributes": ["x", "y"],
                    "merged_from": ["Landlord", "Property Owner"],
                }
            ]
        }
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {
        "partial_schemas": [_PARTIAL_A, _PARTIAL_B],
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }

    result = types_mod.consolidate_types(state)

    assert result["merged_classes"] == [{"name": "Landlord", "parent": None, "attributes": ["x", "y"], "merged_from": ["Landlord", "Property Owner"]}]
    [entry] = result["merge_log"]
    assert entry["merged_name"] == "Landlord"
    assert entry["source_names"] == ["Landlord", "Property Owner"]
    assert entry["source_documents"] == ["a.txt", "b.txt"]
    assert result["class_name_map"]["landlord"] == "Landlord"
    assert result["class_name_map"]["property owner"] == "Landlord"


def test_a_class_merged_from_only_itself_still_gets_a_merge_log_entry(monkeypatch, tmp_path):
    def fake_invoke(spec, prompt):
        payload = {"classes": [{"name": "Landlord", "parent": None, "attributes": ["x"], "merged_from": ["Landlord"]}]}
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": [_PARTIAL_A], "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = types_mod.consolidate_types(state)
    assert result["merge_log"] == [{"merged_name": "Landlord", "source_names": ["Landlord"], "source_documents": ["a.txt"]}]


def test_an_unparseable_consolidate_types_response_aborts_and_is_logged(monkeypatch, tmp_path):
    """The one consolidation call has nothing else to fall back on if it
    fails -- mirrors B3's single-call-must-abort rule."""
    monkeypatch.setattr(mc, "invoke", lambda spec, prompt: _completion("no schema here"))
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": [_PARTIAL_A], "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    with pytest.raises(ss.ResponseParseError):
        types_mod.consolidate_types(state)

    [line] = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert line["response"] == "no schema here"


def test_a_refused_consolidate_types_call_aborts_with_a_clear_message(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mc,
        "invoke",
        lambda spec, prompt: (_ for _ in ()).throw(mc.RefusalError("refused", text="", completion_tokens=0)),
    )
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": [_PARTIAL_A], "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    with pytest.raises(RuntimeError, match="refused"):
        types_mod.consolidate_types(state)


# ---------------------------------------------------------------------------
# T3 -- Stage 3, attribute consolidation
# ---------------------------------------------------------------------------

def test_consolidate_attrs_dedupes_wordings_per_class(monkeypatch, tmp_path):
    merged_classes = [
        {"name": "Landlord", "parent": None, "attributes": ["monthly payment", "Monthly_Base_Rate"], "merged_from": ["Landlord"]},
        {"name": "Vendor", "parent": None, "attributes": ["handle"], "merged_from": ["Vendor"]},
    ]

    def fake_invoke(spec, prompt):
        payload = {
            "classes": [
                {"name": "Landlord", "parent": None, "attributes": ["monthly payment"]},
                {"name": "Vendor", "parent": None, "attributes": ["handle"]},
            ]
        }
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"merged_classes": merged_classes, "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = attrs_mod.consolidate_attrs(state)
    assert result["consolidated_attributes"] == {"Landlord": ["monthly payment"], "Vendor": ["handle"]}


def test_consolidate_attrs_prompt_excludes_merged_from():
    """merged_from is Stage 2's own provenance bookkeeping -- Stage 3's job
    is wording cleanup, and it has no bearing on that."""
    rendered = attrs_mod.render_prompt(
        attrs_mod.load_prompt_template(),
        [{"name": "Landlord", "parent": None, "attributes": ["x"], "merged_from": ["Landlord", "Property Owner"]}],
    )
    assert "merged_from" not in rendered
    assert "Property Owner" not in rendered


def test_an_unparseable_consolidate_attrs_response_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "invoke", lambda spec, prompt: _completion("not json"))
    log_path = tmp_path / "calls.jsonl"
    state = {
        "merged_classes": [{"name": "Landlord", "parent": None, "attributes": ["x"], "merged_from": ["Landlord"]}],
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }
    with pytest.raises(ss.ResponseParseError):
        attrs_mod.consolidate_attrs(state)


# ---------------------------------------------------------------------------
# T4 -- Stage 4, relation reconciliation
# ---------------------------------------------------------------------------

def test_remap_relation_endpoints_rewrites_onto_merged_names():
    partial_schemas = [
        {"source_document": "a.txt", "relations": [{"source": "Landlord", "label": "owns", "target": "Premises"}]},
        {"source_document": "b.txt", "relations": [{"source": "Property Owner", "label": "owns", "target": "Property"}]},
    ]
    name_map = {"landlord": "Owner", "property owner": "Owner", "premises": "Property", "property": "Property"}

    remapped, dropped = relations_mod.remap_relation_endpoints(partial_schemas, name_map)

    assert dropped == 0
    assert remapped == [{"source": "Owner", "label": "owns", "target": "Property"}]  # literal dup after remap collapsed


def test_remap_drops_relations_with_an_unmapped_endpoint():
    partial_schemas = [{"source_document": "a.txt", "relations": [{"source": "Ghost", "label": "haunts", "target": "House"}]}]
    remapped, dropped = relations_mod.remap_relation_endpoints(partial_schemas, {})
    assert remapped == []
    assert dropped == 1


def test_reconcile_relations_makes_no_call_when_nothing_survives_remap(monkeypatch, tmp_path):
    def fail_invoke(spec, prompt):
        raise AssertionError("must not call the model when there is nothing to reconcile")

    monkeypatch.setattr(mc, "invoke", fail_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {"partial_schemas": [], "class_name_map": {}, "model_key": "haiku45", "raw_log_path": str(log_path), "usage": []}

    result = relations_mod.reconcile_relations(state)
    assert result["reconciled_relations"] == []
    assert not log_path.exists() or log_path.read_text() == ""


def test_reconcile_relations_merges_close_wordings_after_remap(monkeypatch, tmp_path):
    partial_schemas = [
        {"source_document": "a.txt", "relations": [{"source": "Landlord", "label": "leases to", "target": "Tenant"}]},
        {"source_document": "b.txt", "relations": [{"source": "Property Owner", "label": "rents to", "target": "Renter"}]},
    ]
    name_map = {"landlord": "Owner", "property owner": "Owner", "tenant": "Renter Class", "renter": "Renter Class"}

    def fake_invoke(spec, prompt):
        payload = {"relations": [{"source": "Owner", "label": "leases to", "target": "Renter Class"}]}
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {
        "partial_schemas": partial_schemas,
        "class_name_map": name_map,
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }

    result = relations_mod.reconcile_relations(state)
    assert result["reconciled_relations"] == [{"source": "Owner", "label": "leases to", "target": "Renter Class"}]


def test_an_unparseable_reconcile_relations_response_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "invoke", lambda spec, prompt: _completion("not json"))
    log_path = tmp_path / "calls.jsonl"
    partial_schemas = [{"source_document": "a.txt", "relations": [{"source": "X", "label": "y", "target": "Z"}]}]
    state = {
        "partial_schemas": partial_schemas,
        "class_name_map": {"x": "X", "z": "Z"},
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }
    with pytest.raises(ss.ResponseParseError):
        relations_mod.reconcile_relations(state)


# ---------------------------------------------------------------------------
# T5 -- Stage 5, taxonomy induction
# ---------------------------------------------------------------------------

def test_induce_taxonomy_defaults_to_the_models_null_when_no_evidence_given(monkeypatch, tmp_path):
    merged_classes = [
        {"name": "Alpha", "parent": None, "attributes": []},
        {"name": "Beta", "parent": None, "attributes": []},
    ]

    def fake_invoke(spec, prompt):
        payload = {
            "classes": [
                {"name": "Alpha", "parent": None, "attributes": []},
                {"name": "Beta", "parent": None, "attributes": []},
            ]
        }
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {
        "merged_classes": merged_classes,
        "consolidated_attributes": {"Alpha": [], "Beta": []},
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }

    result = taxonomy_mod.induce_taxonomy(state)
    assert result["taxonomy_edges"] == {"Alpha": None, "Beta": None}
    assert result["induced_superclasses"] == []


def test_induce_taxonomy_declares_a_new_supertype_and_returns_it_separately(monkeypatch, tmp_path):
    merged_classes = [
        {"name": "Alpha", "parent": None, "attributes": []},
        {"name": "Beta", "parent": None, "attributes": []},
    ]

    def fake_invoke(spec, prompt):
        payload = {
            "classes": [
                {"name": "Alpha", "parent": "Root", "attributes": []},
                {"name": "Beta", "parent": "Root", "attributes": []},
                {"name": "Root", "parent": None, "attributes": []},
            ]
        }
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {
        "merged_classes": merged_classes,
        "consolidated_attributes": {"Alpha": [], "Beta": []},
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }

    result = taxonomy_mod.induce_taxonomy(state)
    assert result["taxonomy_edges"] == {"Alpha": "Root", "Beta": "Root"}
    assert [c["name"] for c in result["induced_superclasses"]] == ["Root"]


def test_induce_taxonomy_drops_an_undeclared_parent_reference_rather_than_dangle(monkeypatch, tmp_path):
    """The model names a new parent but never declares it as its own entry --
    the prompt requires this, but a model can still fail to follow it. Must
    not produce a schema whose `parent` points at nothing (P1-D6)."""
    merged_classes = [{"name": "Alpha", "parent": None, "attributes": []}]

    def fake_invoke(spec, prompt):
        payload = {"classes": [{"name": "Alpha", "parent": "Ghost", "attributes": []}]}
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {
        "merged_classes": merged_classes,
        "consolidated_attributes": {"Alpha": []},
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }

    result = taxonomy_mod.induce_taxonomy(state)
    assert result["taxonomy_edges"] == {"Alpha": None}
    assert result["induced_superclasses"] == []
    line = json.loads(log_path.read_text().splitlines()[0])
    assert line["undeclared_parents_dropped"] == 1


def test_induce_taxonomy_preserves_an_existing_parent_the_model_forgot_to_echo(monkeypatch, tmp_path):
    merged_classes = [{"name": "Alpha", "parent": "Beta", "attributes": []}]

    def fake_invoke(spec, prompt):
        return _completion(json.dumps({"classes": []}))  # model drops Alpha entirely

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    log_path = tmp_path / "calls.jsonl"
    state = {
        "merged_classes": merged_classes,
        "consolidated_attributes": {"Alpha": []},
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }

    result = taxonomy_mod.induce_taxonomy(state)
    assert result["taxonomy_edges"] == {"Alpha": "Beta"}


def test_an_unparseable_induce_taxonomy_response_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "invoke", lambda spec, prompt: _completion("not json"))
    log_path = tmp_path / "calls.jsonl"
    state = {
        "merged_classes": [{"name": "Alpha", "parent": None, "attributes": []}],
        "consolidated_attributes": {"Alpha": []},
        "model_key": "haiku45",
        "raw_log_path": str(log_path),
        "usage": [],
    }
    with pytest.raises(ss.ResponseParseError):
        taxonomy_mod.induce_taxonomy(state)


# ---------------------------------------------------------------------------
# T6 -- Stage 6, assemble + validate
# ---------------------------------------------------------------------------

def test_assemble_builds_a_contract_compliant_output():
    from eval.schema_ir import parse_induced_schema

    state = {
        "documents": [("a.txt", "..."), ("b.txt", "...")],
        "model_key": "haiku45",
        "run_id": "run-1",
        "merged_classes": [{"name": "Alpha", "parent": None, "attributes": ["raw"], "merged_from": ["Alpha"]}],
        "consolidated_attributes": {"Alpha": ["clean"]},
        "taxonomy_edges": {"Alpha": None},
        "induced_superclasses": [],
        "reconciled_relations": [],
        "merge_log": [{"merged_name": "Alpha", "source_names": ["Alpha"], "source_documents": ["a.txt"]}],
        "extraction_skipped": 0,
        "usage": [{"stage": "extract", "stop_reason": "end_turn", "completion_tokens": 5}],
    }

    output = assemble_mod.build_output(state)
    schema = parse_induced_schema(output)

    assert set(schema.classes) == {"Alpha"}
    assert schema.classes["Alpha"].declared_attributes == frozenset({"clean"})
    assert output["metadata"]["condition"] == "P1"
    assert output["metadata"]["model"] == mc.MODELS["haiku45"].model_id
    assert output["metadata"]["run_id"] == "run-1"


def test_assemble_includes_induced_superclasses_in_the_final_class_list():
    state = {
        "documents": [("a.txt", "...")],
        "model_key": "haiku45",
        "run_id": "run-1",
        "merged_classes": [{"name": "Alpha", "parent": "Root", "attributes": [], "merged_from": ["Alpha"]}],
        "consolidated_attributes": {"Alpha": []},
        "taxonomy_edges": {"Alpha": "Root"},
        "induced_superclasses": [{"name": "Root", "parent": None, "attributes": []}],
        "reconciled_relations": [],
        "merge_log": [],
        "extraction_skipped": 0,
        "usage": [],
    }
    output = assemble_mod.build_output(state)
    names = {c["name"] for c in output["classes"]}
    assert names == {"Alpha", "Root"}
    root = next(c for c in output["classes"] if c["name"] == "Root")
    assert root["parent"] is None


def test_assemble_node_raises_if_the_contract_is_not_actually_satisfied(monkeypatch):
    """A Stage 6 that produces something the harness cannot read is a failed
    run, not a success with a formatting quirk."""

    def broken_parse(data):
        raise ValueError("simulated contract violation")

    monkeypatch.setattr(assemble_mod, "parse_induced_schema", broken_parse)
    state = {
        "documents": [],
        "model_key": "haiku45",
        "run_id": "run-1",
        "merged_classes": [],
        "consolidated_attributes": {},
        "taxonomy_edges": {},
        "induced_superclasses": [],
        "reconciled_relations": [],
        "merge_log": [],
        "extraction_skipped": 0,
        "usage": [],
    }
    with pytest.raises(ValueError, match="simulated contract violation"):
        assemble_mod.assemble(state)


# ---------------------------------------------------------------------------
# T7 -- graph.py: wiring, checkpointing/resume, dry-run
# ---------------------------------------------------------------------------

@pytest.fixture
def three_doc_scenario(monkeypatch):
    """Wires a deterministic 7-call sequence through the full graph:
    3 extraction calls (two wordings of one class, one distinct second
    class), then the four consolidation-stage calls. Mirrors the same
    scenario manually exercised during development."""
    calls = {"n": 0}

    def fake_invoke(spec, prompt):
        calls["n"] += 1
        n = calls["n"]
        if n == 1:
            payload = {"classes": [{"name": "Alpha Corp", "parent": None, "attributes": ["yearly fee"]}], "relations": []}
        elif n == 2:
            payload = {
                "classes": [{"name": "AlphaCorp", "parent": None, "attributes": ["annual fee amount"]}],
                "relations": [{"source": "AlphaCorp", "label": "serves", "target": "Beta Unit"}],
            }
        elif n == 3:
            payload = {"classes": [{"name": "Beta Unit", "parent": None, "attributes": ["handle"]}], "relations": []}
        elif n == 4:
            payload = {
                "classes": [
                    {"name": "AlphaCorp", "parent": None, "attributes": ["yearly fee", "annual fee amount"], "merged_from": ["Alpha Corp", "AlphaCorp"]},
                    {"name": "Beta Unit", "parent": None, "attributes": ["handle"], "merged_from": ["Beta Unit"]},
                ]
            }
        elif n == 5:
            payload = {
                "classes": [
                    {"name": "AlphaCorp", "parent": None, "attributes": ["annual fee amount"]},
                    {"name": "Beta Unit", "parent": None, "attributes": ["handle"]},
                ]
            }
        elif n == 6:
            payload = {"relations": [{"source": "AlphaCorp", "label": "serves", "target": "Beta Unit"}]}
        elif n == 7:
            payload = {
                "classes": [
                    {"name": "AlphaCorp", "parent": "Org", "attributes": ["annual fee amount"]},
                    {"name": "Beta Unit", "parent": "Org", "attributes": ["handle"]},
                    {"name": "Org", "parent": None, "attributes": []},
                ]
            }
        else:
            raise AssertionError(f"unexpected call #{n}")
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    return calls


def test_full_graph_runs_all_six_stages_in_order(three_doc_scenario, tmp_path, monkeypatch):
    import pipeline.graph as g

    documents = [("doc-1.txt", "..."), ("doc-2.txt", "..."), ("doc-3.txt", "...")]
    monkeypatch.setattr(g, "load_documents", lambda corpus, limit=None: documents)

    output = g.run(
        model_key="haiku45",
        out_dir=tmp_path / "raw",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert three_doc_scenario["n"] == 7
    names = {c["name"] for c in output["classes"]}
    assert names == {"AlphaCorp", "Beta Unit", "Org"}
    assert output["relations"] == [{"source": "AlphaCorp", "label": "serves", "target": "Beta Unit"}]
    org = next(c for c in output["classes"] if c["name"] == "Org")
    assert org["parent"] is None
    alpha = next(c for c in output["classes"] if c["name"] == "AlphaCorp")
    assert alpha["parent"] == "Org"


def test_a_crash_mid_graph_can_be_resumed_without_repeating_completed_stages(tmp_path, monkeypatch):
    import pipeline.graph as g

    documents = [("d1.txt", "a"), ("d2.txt", "b"), ("d3.txt", "c")]
    monkeypatch.setattr(g, "load_documents", lambda corpus, limit=None: documents)

    calls = {"n": 0}

    def crash_after_extraction(spec, prompt):
        calls["n"] += 1
        n = calls["n"]
        if n <= 3:
            payload = {"classes": [{"name": f"Thing{n}", "parent": None, "attributes": ["x"]}], "relations": []}
            return _completion(json.dumps(payload))
        raise RuntimeError("simulated transient failure")

    monkeypatch.setattr(mc, "invoke", crash_after_extraction)

    out_dir = tmp_path / "raw"
    checkpoint_dir = tmp_path / "checkpoints"

    with pytest.raises(RuntimeError, match="simulated transient failure"):
        g.run(model_key="haiku45", out_dir=out_dir, checkpoint_dir=checkpoint_dir)

    assert calls["n"] == 4  # 3 extraction calls + the failing consolidate_types call
    [checkpoint_file] = list(checkpoint_dir.glob("*.sqlite"))
    run_id = checkpoint_file.stem
    log_path = out_dir / f"{run_id}_p1_haiku45_calls.jsonl"
    # 3 successful extraction calls, plus one record for the crashed Stage 2
    # attempt. That fourth line is the point: a transport-level failure (this
    # RuntimeError stands in for the real botocore ReadTimeoutError that killed
    # a full-corpus Haiku run) is not a ModelResponseError, so before the
    # transport branch in invoke_or_abort existed it propagated with nothing
    # logged -- leaving a log that read as though the pipeline had quietly
    # stopped after extraction rather than died in a specific, identifiable call.
    logged = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(logged) == 4
    assert [r["stage"] for r in logged] == ["extract"] * 3 + ["consolidate_types"]
    assert logged[-1]["stop_reason"] == "transport_error"
    assert "simulated transient failure" in logged[-1]["error"]

    def succeed_from_here(spec, prompt):
        calls["n"] += 1
        n = calls["n"]
        if n == 5:
            payload = {"classes": [{"name": "Thing1", "parent": None, "attributes": ["x"], "merged_from": ["Thing1", "Thing2", "Thing3"]}]}
        elif n == 6:
            payload = {"classes": [{"name": "Thing1", "parent": None, "attributes": ["x"]}]}
        elif n == 7:
            # Stage 4 (reconcile_relations) makes zero calls here -- none of
            # the three fake extraction responses carried a relation, so
            # there is nothing to remap or reconcile. This is call #7, not
            # #8: the taxonomy stage, not the relations stage.
            payload = {"classes": [{"name": "Thing1", "parent": None, "attributes": ["x"]}]}
        else:
            raise AssertionError(n)
        return _completion(json.dumps(payload))

    monkeypatch.setattr(mc, "invoke", succeed_from_here)

    output = g.run(model_key="haiku45", out_dir=out_dir, checkpoint_dir=checkpoint_dir, resume_run_id=run_id)

    assert output["classes"] == [{"name": "Thing1", "parent": None, "attributes": ["x"]}]
    # Extraction was never repeated -- calls 1-3 happened once, before the
    # crash. Total: 3 extraction + 1 failed consolidate_types attempt + 3
    # calls after resume (consolidate_types retried, consolidate_attrs,
    # induce_taxonomy -- reconcile_relations made none) = 7.
    assert calls["n"] == 7


def test_dry_run_reports_per_stage_call_counts_against_the_real_corpus(capsys):
    import pipeline.graph as g

    documents = ss.load_documents(g._CORPUS_ROOT, limit=2)
    g._dry_run(mc.MODELS["haiku45"], documents)
    out = capsys.readouterr().out

    assert "DRY RUN -- no API calls" in out
    assert f"{len(documents)} calls (one per document)" in out
    assert "consolidate_types" in out and "1 call" in out
    assert str(len(documents)) in out


def test_dry_run_makes_no_model_call(monkeypatch):
    import pipeline.graph as g

    def fail_invoke(spec, prompt):
        raise AssertionError("dry run must never call a model")

    monkeypatch.setattr(mc, "invoke", fail_invoke)
    documents = ss.load_documents(g._CORPUS_ROOT, limit=2)
    g._dry_run(mc.MODELS["opus5"], documents)  # raises if it ever calls mc.invoke


def test_build_graph_definition_has_the_six_stages_in_order():
    import pipeline.graph as g

    builder = g.build_graph_definition()
    graph = builder.compile()
    node_names = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert node_names == set(g.STAGE_ORDER)
