"""
P1 baseline validation suite.

Every test here runs offline. P1 reuses B3's document loading, prompt
rendering, response parsing, and malformed-element cleanup wholesale --
those are covered by test_single_shot.py and not re-tested here. This file
covers what is actually new: the two-stage orchestration (N extraction calls,
1 consolidation call), the consolidation prompt's own rendering and framing,
and the Critical Rule 1 leakage guard extended to the new prompt.

  Rule 1 (no gold vocabulary)  test_no_domain_vocabulary_leakage
  Rule 2 (frozen prompts)      test_consolidation_prompt_text_is_identical_regardless_of_input
  a bad extraction call is recorded and skipped, not fatal (192 calls, one bad
    one should not sink the run)   test_one_bad_extraction_call_is_skipped_not_fatal
  the one consolidation call has nothing to fall back on if it fails
    test_an_unparseable_consolidation_response_aborts
"""

from __future__ import annotations

import json

import pytest

from baselines.b3_single_shot import single_shot as ss
from baselines.p1_pipeline import pipeline as p1
from baselines.shared import model_clients as mc

from baselines.tests.test_statistical import _executable_vocabulary
from baselines.tests.test_single_shot import _gold_terms, _contains_subsequence, leaks_in_text, _completion


def _docs(count: int) -> list[tuple[str, str]]:
    return [(f"dir/doc-{i:03d}.txt", f"body of document {i}") for i in range(count)]


_SCHEMA_A = {"classes": [{"name": "Alpha", "parent": None, "attributes": ["a one"]}], "relations": []}
_SCHEMA_B = {"classes": [{"name": "Alpha", "parent": None, "attributes": ["a two"]}], "relations": []}
_MERGED = {"classes": [{"name": "Alpha", "parent": None, "attributes": ["a one", "a two"]}], "relations": []}


# ---------------------------------------------------------------------------
# T1 -- the consolidation prompt (Critical Rule 2, extended)
# ---------------------------------------------------------------------------

def test_consolidation_prompt_file_loads_and_declares_its_placeholder():
    template = p1.load_consolidation_prompt_template()
    assert template.count(p1.SCHEMAS_PLACEHOLDER) == 1


def test_load_consolidation_prompt_rejects_a_template_without_the_placeholder(tmp_path):
    path = tmp_path / "broken.md"
    path.write_text("no placeholder here", encoding="utf-8")
    with pytest.raises(ValueError):
        p1.load_consolidation_prompt_template(path)


def test_consolidation_prompt_text_is_identical_regardless_of_input():
    template = p1.load_consolidation_prompt_template()
    head, tail = template.split(p1.SCHEMAS_PLACEHOLDER)

    first = p1.render_consolidation_prompt(template, [_SCHEMA_A])
    second = p1.render_consolidation_prompt(template, [_SCHEMA_A, _SCHEMA_B])

    assert first.startswith(head) and first.endswith(tail)
    assert second.startswith(head) and second.endswith(tail)
    assert first != second, "the schemas block must actually be substituted"


def test_rendered_consolidation_prompt_carries_every_partial_schema():
    rendered = p1.render_consolidation_prompt(
        p1.load_consolidation_prompt_template(),
        [{"source_document": "dir/a.txt", **_SCHEMA_A}, {"source_document": "dir/b.txt", **_SCHEMA_B}],
    )
    assert "dir/a.txt" in rendered and "dir/b.txt" in rendered
    assert "a one" in rendered and "a two" in rendered


# ---------------------------------------------------------------------------
# T2 -- Stage 1, extraction (per-document, tolerant of individual failures)
# ---------------------------------------------------------------------------

def test_run_extraction_stage_makes_one_call_per_document(monkeypatch):
    calls: list[str] = []

    def fake_invoke(spec, prompt):
        calls.append(prompt)
        return _completion(json.dumps(_SCHEMA_A))

    monkeypatch.setattr(p1.mc, "invoke", fake_invoke)
    documents = _docs(5)
    records: list[dict] = []
    schemas = p1.run_extraction_stage(mc.MODELS["haiku45"], documents, ss.DOCUMENTS_PLACEHOLDER, records)

    assert len(calls) == 5
    assert len(schemas) == 5
    assert all(s["source_document"] == d[0] for s, d in zip(schemas, documents))
    assert [r["call"] for r in records] == [1, 2, 3, 4, 5]


def test_one_bad_extraction_call_is_skipped_not_fatal(monkeypatch):
    """Unlike B3's single call, P1 has N=192 independent extraction calls --
    one truncation, refusal, or unparseable response must not sink the run
    that has already paid for the other 191."""
    responses = iter([
        _completion(json.dumps(_SCHEMA_A)),
        _completion("sorry, I cannot help with that"),
        _completion(json.dumps(_SCHEMA_B)),
    ])
    monkeypatch.setattr(p1.mc, "invoke", lambda spec, prompt: next(responses))

    records: list[dict] = []
    schemas = p1.run_extraction_stage(mc.MODELS["haiku45"], _docs(3), ss.DOCUMENTS_PLACEHOLDER, records)

    assert len(schemas) == 2, "the one bad call is skipped, not fatal"
    assert "parse_error" in records[1]
    assert records[1]["response"] == "sorry, I cannot help with that"


def test_a_truncated_extraction_call_is_recorded_and_skipped(monkeypatch):
    calls = iter([
        mc.TruncatedResponseError("cap", '{"cla', 16000),
        _completion(json.dumps(_SCHEMA_A)),
    ])

    def fake_invoke(spec, prompt):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(p1.mc, "invoke", fake_invoke)
    records: list[dict] = []
    schemas = p1.run_extraction_stage(mc.MODELS["haiku45"], _docs(2), ss.DOCUMENTS_PLACEHOLDER, records)

    assert len(schemas) == 1
    assert records[0]["stop_reason"] == "truncated"
    assert records[0]["response"] == '{"cla'


# ---------------------------------------------------------------------------
# T3 -- Stage 2, consolidation (the one call with nothing to fall back on)
# ---------------------------------------------------------------------------

def test_run_consolidation_stage_makes_exactly_one_call(monkeypatch):
    calls: list[str] = []

    def fake_invoke(spec, prompt):
        calls.append(prompt)
        return _completion(json.dumps(_MERGED))

    monkeypatch.setattr(p1.mc, "invoke", fake_invoke)
    record: dict = {}
    result = p1.run_consolidation_stage(
        mc.MODELS["opus5"],
        [{"source_document": "a.txt", **_SCHEMA_A}, {"source_document": "b.txt", **_SCHEMA_B}],
        p1.load_consolidation_prompt_template(),
        record,
    )

    assert len(calls) == 1
    assert result == _MERGED
    assert record["input_schema_count"] == 2
    assert record["stop_reason"] == "end_turn"


def test_an_unparseable_consolidation_response_aborts(monkeypatch):
    """The one consolidation call has nothing else to fall back on if it fails
    -- mirrors B3's single-call-must-abort rule, for the same reason."""
    monkeypatch.setattr(p1.mc, "invoke", lambda spec, prompt: _completion("no schema here"))
    record: dict = {}
    with pytest.raises(ss.ResponseParseError):
        p1.run_consolidation_stage(
            mc.MODELS["opus5"], [{"source_document": "a.txt", **_SCHEMA_A}],
            p1.load_consolidation_prompt_template(), record,
        )
    assert record["response"] == "no schema here", "the paid-for response must still be captured"


def test_a_truncated_consolidation_response_is_captured_before_it_raises(monkeypatch):
    def fake_invoke(spec, prompt):
        raise mc.TruncatedResponseError("cap", '{"classes": [{"na', 32000)

    monkeypatch.setattr(p1.mc, "invoke", fake_invoke)
    record: dict = {}
    with pytest.raises(mc.TruncatedResponseError):
        p1.run_consolidation_stage(
            mc.MODELS["opus5"], [{"source_document": "a.txt", **_SCHEMA_A}],
            p1.load_consolidation_prompt_template(), record,
        )
    assert record["stop_reason"] == "truncated"
    assert record["response"] == '{"classes": [{"na'


# ---------------------------------------------------------------------------
# T4 -- output contract
# ---------------------------------------------------------------------------

def test_output_satisfies_the_induced_schema_contract():
    from eval.schema_ir import parse_induced_schema

    spec = mc.MODELS["haiku45"]
    output = p1.build_output(
        _MERGED, ["dir/a.txt", "dir/b.txt"], "run-1", spec,
        extraction_calls=2, extraction_skipped=0,
        consolidation_stop_reason="end_turn", consolidation_completion_tokens=30,
    )
    schema = parse_induced_schema(output)
    assert set(schema.classes) == {"Alpha"}
    assert schema.classes["Alpha"].declared_attributes == frozenset({"a one", "a two"})


def test_output_metadata_names_the_condition_and_records_both_stages():
    spec = mc.MODELS["opus5"]
    output = p1.build_output(
        {"classes": [], "relations": []}, ["a.txt"], "run-1", spec,
        extraction_calls=192, extraction_skipped=3,
        consolidation_stop_reason="end_turn", consolidation_completion_tokens=900,
    )
    assert output["metadata"] == {
        "condition": "P1",
        "model": spec.model_id,
        "run_id": "run-1",
        "source_documents": ["a.txt"],
        "extraction_calls": 192,
        "extraction_skipped": 3,
        "consolidation_stop_reason": "end_turn",
        "consolidation_completion_tokens": 900,
    }


def test_output_is_json_serializable():
    spec = mc.MODELS["haiku45"]
    output = p1.build_output(_MERGED, [], "run-1", spec, 1, 0, "end_turn", 10)
    assert json.loads(json.dumps(output)) == output


# ---------------------------------------------------------------------------
# T5 -- no gold vocabulary anywhere (Critical Rule 1)
# ---------------------------------------------------------------------------

def test_no_domain_vocabulary_leakage():
    """Critical Rule 1, over pipeline.py, the shared model-calling module (also
    checked by B3's suite -- redundant coverage on shared code is deliberate),
    and both prompts P1 actually uses."""
    modules = [
        p1._MODULE_ROOT / "pipeline.py",
        p1._REPO_ROOT / "baselines" / "shared" / "model_clients.py",
    ]

    leaks: list[str] = []
    for path in modules:
        assert path.exists(), f"{path} not found -- the leakage guard is scanning nothing"
        used = _executable_vocabulary(path.read_text(encoding="utf-8"))
        for term in sorted(_gold_terms()):
            needle = tuple(term.split())
            for item in used:
                words = tuple(item.split())
                hit = (
                    item == term if len(needle) == 1 else _contains_subsequence(words, needle)
                )
                if hit:
                    leaks.append(f"{path.name}: gold {term!r} appears as {item!r}")

    prompts = [
        ss._MODULE_ROOT / "prompts" / "b3_extraction_prompt.md",  # reused by P1 Stage 1
        p1.CONSOLIDATION_PROMPT_PATH,
    ]
    for prompt_path in prompts:
        for term in leaks_in_text(prompt_path.read_text(encoding="utf-8")):
            leaks.append(f"{prompt_path.name}: gold {term!r} appears in the prompt text")

    assert not leaks, (
        "gold-schema vocabulary found in P1 (Critical Rule 1):\n" + "\n".join(sorted(set(leaks)))
    )


def test_the_consolidation_prompt_still_names_the_contract_keys_it_must():
    prompt = p1.CONSOLIDATION_PROMPT_PATH.read_text(encoding="utf-8")
    for key in ("classes", "relations", "name", "parent", "attributes", "source", "label", "target"):
        assert key in prompt
