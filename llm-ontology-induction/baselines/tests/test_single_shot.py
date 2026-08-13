"""
B3 baseline validation suite.

Every test here runs offline: no Bedrock call, no Groq call, no network of any
kind. The backend layer is split so that the parts worth asserting on -- the
request body, the response parser, the consolidation -- are pure functions, and
the one test that exercises orchestration substitutes a fake `invoke`.

The four Critical Rules that make B3 a valid baseline each have a test that fails
if the rule is broken:

  Rule 1 (no gold vocabulary)     test_no_domain_vocabulary_leakage
  Rule 2 (one frozen prompt)      test_instruction_text_is_identical_across_batches
  Rule 3 (Luna at low effort)     test_luna_always_carries_low_reasoning_effort
  Rule 4 (naive consolidation)    test_distinct_wordings_are_never_merged
  Rule 7 (identical batching)     test_batching_cannot_depend_on_the_model
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from baselines import model_clients as mc
from baselines import single_shot as ss

# Reused rather than reimplemented: one source of truth for what counts as gold
# vocabulary and as executable vocabulary, shared with the B1 suite.
from baselines.tests.test_statistical import (
    _CONTRACT_KEYS,
    _executable_vocabulary,
    _gold_vocabulary,
)

_MODULES = ["single_shot.py", "model_clients.py"]
_PROMPT = "prompts/b3_extraction_prompt.md"


# ---------------------------------------------------------------------------
# Fixtures -- deliberately domain-free stand-ins
# ---------------------------------------------------------------------------

_SCHEMA = {
    "classes": [
        {"name": "Alpha", "parent": None, "attributes": ["a one", "a two"]},
        {"name": "Beta", "parent": "Alpha", "attributes": []},
    ],
    "relations": [{"source": "Alpha", "label": "touches", "target": "Beta"}],
}


def _docs(count: int) -> list[tuple[str, str]]:
    return [(f"dir/doc-{i:03d}.txt", f"body of document {i}") for i in range(count)]


@pytest.fixture
def corpus(tmp_path):
    """A miniature corpus mirroring the real subdirectory layout."""
    for subdir, names in (
        ("csv_exports", ["b.csv", "a.csv"]),
        ("lease_texts", ["t2.txt", "t1.txt"]),
        ("notes", ["n1.txt"]),
        ("messages", ["m1.txt"]),
    ):
        directory = tmp_path / subdir
        directory.mkdir()
        for filename in names:
            (directory / filename).write_text(
                f"col one,col two\nvalue in {filename}\n", encoding="utf-8"
            )
    (tmp_path / "notes" / "ignored.pdf").write_text("skip me", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# T1 -- corpus loading and batching (B3-D2, Critical Rule 7)
# ---------------------------------------------------------------------------

def test_batch_size_is_the_frozen_value():
    """B3-D2 (revised 2026-08-12, was 10). If this changes, every previously
    reported B3 number is stale."""
    assert ss.BATCH_SIZE == 7


def test_batching_cannot_depend_on_the_model():
    """Critical Rule 7, enforced structurally rather than by inspection.

    batch_documents takes no ModelSpec, so there is no parameter through which one
    model could receive different batches from another. A future signature change
    that adds one has to break this test first.
    """
    import inspect

    parameters = inspect.signature(ss.batch_documents).parameters
    assert list(parameters) == ["documents", "size"]


def test_every_document_lands_in_exactly_one_batch():
    documents = _docs(25)
    batches = ss.batch_documents(documents, 10)

    assert [len(b) for b in batches] == [10, 10, 5]
    flattened = [item for batch in batches for item in batch]
    assert flattened == documents, "batching must preserve order and lose nothing"


def test_batching_rejects_a_nonsense_size():
    with pytest.raises(ValueError):
        ss.batch_documents(_docs(3), 0)


def test_documents_are_read_whole_and_in_a_deterministic_order(corpus):
    """B3 hands raw files to the model -- no sentence splitting, no CSV flattening.

    That B1 preprocesses and B3 does not is a real difference between the two
    conditions, so the absence of preprocessing is asserted, not assumed.
    """
    documents = ss.load_documents(corpus)
    sources = [source for source, _text in documents]

    assert sources == [
        "csv_exports/a.csv",
        "lease_texts/t1.txt",
        "notes/n1.txt",
        "messages/m1.txt",
        "csv_exports/b.csv",
        "lease_texts/t2.txt",
    ], (
        "round-robin across subdirectories (revised B3-D2, 2026-08-12): one "
        "document per subdirectory per round, subdirectories in _SUBDIRS order, "
        "filenames sorted within each"
    )

    text = dict(documents)["csv_exports/a.csv"]
    assert text == "col one,col two\nvalue in a.csv\n", "raw bytes, untouched"


def test_load_documents_limit_caps_each_subdirectory(corpus):
    documents = ss.load_documents(corpus, limit=1)
    assert [s for s, _ in documents] == [
        "csv_exports/a.csv",
        "lease_texts/t1.txt",
        "notes/n1.txt",
        "messages/m1.txt",
    ]


# ---------------------------------------------------------------------------
# T2 -- the frozen prompt (Critical Rule 2)
# ---------------------------------------------------------------------------

def test_prompt_file_loads_and_declares_its_placeholder():
    template = ss.load_prompt_template()
    assert template.count(ss.DOCUMENTS_PLACEHOLDER) == 1


def test_load_prompt_template_rejects_a_template_without_the_placeholder(tmp_path):
    path = tmp_path / "broken.md"
    path.write_text("instructions with nowhere to put documents", encoding="utf-8")
    with pytest.raises(ValueError):
        ss.load_prompt_template(path)


def test_instruction_text_is_identical_across_batches():
    """Critical Rule 2: only the documents block varies, byte for byte."""
    template = ss.load_prompt_template()
    head, tail = template.split(ss.DOCUMENTS_PLACEHOLDER)

    first = ss.render_prompt(template, _docs(3))
    second = ss.render_prompt(template, _docs(7)[4:])

    assert first.startswith(head) and first.endswith(tail)
    assert second.startswith(head) and second.endswith(tail)
    assert first != second, "the documents block must actually be substituted"


def test_rendering_survives_json_braces_in_the_template():
    """Why render_prompt uses str.replace and not str.format.

    The frozen prompt shows a JSON output example. str.format would read those
    braces as replacement fields and raise, which is exactly the bug this guards.
    """
    template = '{"name": "..."} then ' + ss.DOCUMENTS_PLACEHOLDER

    rendered = ss.render_prompt(template, _docs(1))

    assert rendered.startswith('{"name": "..."} then ')
    with pytest.raises((KeyError, IndexError, ValueError)):
        template.format(documents="x")


def test_rendered_prompt_carries_every_document_and_its_source():
    batch = _docs(3)
    rendered = ss.render_prompt(ss.load_prompt_template(), batch)
    for source, text in batch:
        assert source in rendered
        assert text in rendered


# ---------------------------------------------------------------------------
# T3 -- response parsing
# ---------------------------------------------------------------------------

def test_parses_a_clean_response():
    assert ss.parse_response(json.dumps(_SCHEMA)) == _SCHEMA


def test_parses_through_markdown_fences():
    response = "```json\n" + json.dumps(_SCHEMA) + "\n```"
    assert ss.parse_response(response) == _SCHEMA


def test_parses_through_preamble_and_trailing_commentary():
    """The two habits models keep despite being told not to. Rejecting these
    batches would silently shrink the corpus this condition saw."""
    response = (
        "Sure! Here is the schema I extracted:\n\n"
        + json.dumps(_SCHEMA)
        + "\n\nLet me know if you would like me to expand any of these."
    )
    assert ss.parse_response(response) == _SCHEMA


def test_parses_through_a_reasoning_block_without_taking_its_contents():
    """Defensive, not model-specific: none of the five frozen models currently
    emits a <think> block, but the parser must not be fooled if one ever does.

    The <think> block here contains its own JSON object with a `classes` key --
    a parser that merely searched for the first brace would return the model's
    scratch work as the answer.
    """
    response = (
        "<think>\nFirst I will draft {\"classes\": [], \"relations\": []} and check it.\n"
        "</think>\n" + json.dumps(_SCHEMA)
    )
    assert ss.parse_response(response) == _SCHEMA


def test_parses_when_only_the_closing_reasoning_tag_is_present():
    response = (
        "considering the documents {and some braces} before answering\n"
        "</think>\n" + json.dumps(_SCHEMA)
    )
    assert ss.parse_response(response) == _SCHEMA


def test_a_brace_inside_a_string_value_does_not_end_the_scan():
    schema = {
        "classes": [{"name": "Alpha {braced}", "parent": None, "attributes": ["a }"]}],
        "relations": [],
    }
    response = "here you go:\n" + json.dumps(schema) + "\ndone"
    assert ss.parse_response(response) == schema


def test_nothing_is_dropped_when_the_response_needs_cleaning():
    """The point of tolerance: the recovered schema is whole, not partial."""
    response = "```json\n" + json.dumps(_SCHEMA) + "\n```\nHope that helps."
    parsed = ss.parse_response(response)
    assert len(parsed["classes"]) == len(_SCHEMA["classes"])
    assert len(parsed["relations"]) == len(_SCHEMA["relations"])


@pytest.mark.parametrize(
    "response",
    [
        "",
        "I could not find any structure in these documents.",
        '{"unrelated": true}',
        "```json\n{not json at all}\n```",
    ],
)
def test_unrecoverable_responses_raise_instead_of_returning_empty(response):
    """The one failure mode that would corrupt a reported number invisibly.

    An empty schema returned here is indistinguishable downstream from a batch
    that genuinely contained nothing, so it must raise.
    """
    with pytest.raises(ss.ResponseParseError):
        ss.parse_response(response)


# ---------------------------------------------------------------------------
# T4 -- consolidation (B3-D4, Critical Rules 4 and 5)
# ---------------------------------------------------------------------------

def test_case_and_whitespace_variants_merge():
    merged = ss.merge_batches(
        [
            {"classes": [{"name": "Owner", "parent": None, "attributes": []}]},
            {"classes": [{"name": "  owner ", "parent": None, "attributes": []}]},
        ]
    )
    assert [c["name"] for c in merged["classes"]] == ["Owner"], (
        "case/whitespace differences collapse, and the first surface form wins"
    )


def test_distinct_wordings_are_never_merged():
    """B3-D4 / Critical Rule 4 -- the load-bearing test of this file.

    A smarter method could obviously resolve these two to one entity. Doing so is
    P1's Stage 6 novelty; if B3 does it too, the B3-vs-P1 comparison measures
    nothing. Failing this test does not mean B3 got worse -- it means B3 stopped
    being the baseline the paper claims it is.
    """
    merged = ss.merge_batches(
        [
            {"classes": [{"name": "Owner", "parent": None, "attributes": []}]},
            {"classes": [{"name": "Landlord", "parent": None, "attributes": []}]},
        ]
    )
    assert sorted(c["name"] for c in merged["classes"]) == ["Landlord", "Owner"]


def test_plural_and_camel_case_variants_also_stay_separate():
    """_dedup_key is not eval.matching.normalize, and this is where that shows.

    The harness's normalizer would fold all three of these together (it
    singularizes and splits camelCase). Borrowing it here would let B3 merge
    wordings a naive method cannot, and make B3's output partly a function of the
    harness scoring it.
    """
    merged = ss.merge_batches(
        [
            {"classes": [{"name": "Invoice", "parent": None, "attributes": []}]},
            {"classes": [{"name": "Invoices", "parent": None, "attributes": []}]},
            {"classes": [{"name": "invoiceRecord", "parent": None, "attributes": []}]},
        ]
    )
    assert len(merged["classes"]) == 3


def test_attributes_union_across_batches_keeping_first_surface_form():
    merged = ss.merge_batches(
        [
            {"classes": [{"name": "Alpha", "parent": None, "attributes": ["Size", "Hue"]}]},
            {"classes": [{"name": "alpha", "parent": None, "attributes": ["size", "Mass"]}]},
        ]
    )
    assert merged["classes"][0]["attributes"] == ["Size", "Hue", "Mass"]


def test_names_are_emitted_exactly_as_the_model_returned_them():
    """Critical Rule 5: the harness normalizes, the producer never pre-cleans."""
    ugly = "  Maintenance   REQUESTS_v2  "
    merged = ss.merge_batches(
        [{"classes": [{"name": ugly, "parent": None, "attributes": [" Odd  Field "]}]}]
    )
    assert merged["classes"][0]["name"] == ugly
    assert merged["classes"][0]["attributes"] == [" Odd  Field "]


def test_parent_stays_null_unless_a_model_supplied_one():
    """Critical Rule 6 -- no taxonomy is inferred here, ever."""
    merged = ss.merge_batches(
        [
            {"classes": [{"name": "Alpha", "parent": None, "attributes": []}]},
            {"classes": [{"name": "Beta", "parent": "Alpha", "attributes": []}]},
        ]
    )
    by_name = {c["name"]: c for c in merged["classes"]}
    assert by_name["Alpha"]["parent"] is None
    assert by_name["Beta"]["parent"] == "Alpha"


def test_first_non_null_parent_wins_when_batches_disagree():
    merged = ss.merge_batches(
        [
            {"classes": [{"name": "Beta", "parent": None, "attributes": []}]},
            {"classes": [{"name": "Beta", "parent": "Alpha", "attributes": []}]},
            {"classes": [{"name": "Beta", "parent": "Gamma", "attributes": []}]},
        ]
    )
    assert merged["classes"][0]["parent"] == "Alpha"


def test_relations_deduplicate_on_the_whole_triple():
    merged = ss.merge_batches(
        [
            {"relations": [{"source": "Alpha", "label": "touches", "target": "Beta"}]},
            {"relations": [{"source": "alpha", "label": "Touches", "target": " beta"}]},
            {"relations": [{"source": "Alpha", "label": "avoids", "target": "Beta"}]},
        ]
    )
    assert len(merged["relations"]) == 2
    assert merged["relations"][0] == {
        "source": "Alpha",
        "label": "touches",
        "target": "Beta",
    }


def test_malformed_elements_are_dropped_not_repaired():
    """Repairing a model's malformed output would be the producer doing quality
    work this baseline is supposed to be measured without."""
    merged = ss.merge_batches(
        [
            {
                "classes": [
                    {"parent": None, "attributes": []},          # no name
                    {"name": "   ", "parent": None},             # blank name
                    {"name": "Alpha", "attributes": [None, 7, "ok"]},
                    "not even an object",
                ],
                "relations": [
                    {"source": "Alpha", "label": "touches"},     # no target
                    {"source": "Alpha", "label": "touches", "target": "Beta"},
                ],
            }
        ]
    )
    assert [c["name"] for c in merged["classes"]] == ["Alpha"]
    assert merged["classes"][0]["attributes"] == ["ok"]
    assert len(merged["relations"]) == 1


def test_merging_nothing_yields_an_empty_schema():
    assert ss.merge_batches([]) == {"classes": [], "relations": []}


# ---------------------------------------------------------------------------
# T5 -- output contract (B3-D5)
# ---------------------------------------------------------------------------

def test_output_satisfies_the_induced_schema_contract():
    """Asserted by round-tripping through the harness's own loader rather than by
    re-describing the shape here -- if the contract moves, this moves with it."""
    from eval.schema_ir import Relation, effective_attributes, parse_induced_schema

    spec = mc.MODELS["haiku45"]
    output = ss.build_output(
        ss.merge_batches([_SCHEMA]), ["dir/doc-000.txt"], "run-1", spec
    )

    schema = parse_induced_schema(output)
    assert set(schema.classes) == {"Alpha", "Beta"}
    assert schema.classes["Beta"].parent == "Alpha"
    assert schema.classes["Alpha"].declared_attributes == frozenset({"a one", "a two"})
    assert effective_attributes(schema, "Beta") == frozenset({"a one", "a two"})
    assert Relation(source="Alpha", label="touches", target="Beta") in schema.relations


def test_output_metadata_names_the_condition_and_the_exact_model():
    spec = mc.MODELS["luna"]
    output = ss.build_output({"classes": [], "relations": []}, ["a/b.txt"], "run-1", spec)

    assert set(output) == {"classes", "relations", "metadata"}
    assert output["metadata"] == {
        "condition": "B3",
        "model": spec.model_id,
        "run_id": "run-1",
        "source_documents": ["a/b.txt"],
    }


def test_output_is_json_serializable():
    spec = mc.MODELS["llama318b"]
    output = ss.build_output(ss.merge_batches([_SCHEMA]), [], "run-1", spec)
    assert json.loads(json.dumps(output)) == output


# ---------------------------------------------------------------------------
# T6 -- model registry and request bodies (B3-D1, B3-D1a, B3-D3)
# ---------------------------------------------------------------------------

def test_registry_holds_exactly_the_five_frozen_conditions():
    assert set(mc.MODELS) == {"fable5", "haiku45", "sol", "luna", "llama318b"}
    backends = [spec.backend for spec in mc.MODELS.values()]
    assert backends.count("bedrock") == 4
    assert backends.count("groq") == 1
    for key, spec in mc.MODELS.items():
        assert spec.key == key, "registry key must match the spec it points at"


def test_model_specs_are_immutable():
    """Critical Rule 3 cannot be honored by a spec that code can edit at runtime."""
    with pytest.raises(FrozenInstanceError):
        mc.MODELS["luna"].reasoning_effort = "high"


def test_luna_always_carries_low_reasoning_effort():
    """Critical Rule 3 / B3-D1a.

    "low" is not a default worth keeping -- it is the setting that makes Luna
    capability-matched to the other budget-tier model. Any other value silently
    turns this condition into a different experiment.
    """
    assert mc.MODELS["luna"].reasoning_effort == "low"
    body = mc.build_request_body(mc.MODELS["luna"], "prompt text")
    assert body["reasoning_effort"] == "low"


def test_no_other_model_sets_a_reasoning_effort():
    for key, spec in mc.MODELS.items():
        if key == "luna":
            continue
        assert spec.reasoning_effort is None
        assert "reasoning_effort" not in mc.build_request_body(spec, "prompt text")


def test_every_model_builds_a_body_carrying_the_prompt_and_frozen_sampling():
    """B3-D3: the sampling settings are identical across all five conditions."""
    for spec in mc.MODELS.values():
        body = mc.build_request_body(spec, "SENTINEL PROMPT")
        assert "SENTINEL PROMPT" in json.dumps(body)
        assert body["temperature"] == mc.TEMPERATURE == 0.0
        assert body["top_p"] == mc.TOP_P


def test_an_unknown_backend_is_an_error_not_a_default():
    spec = mc.ModelSpec(key="x", model_id="x", backend="nowhere", tier="budget")
    with pytest.raises(ValueError):
        mc.build_request_body(spec, "prompt text")
    with pytest.raises(ValueError):
        mc.invoke(spec, "prompt text")


def test_extract_text_refuses_an_unrecognized_response_shape():
    """An empty string here would be recorded as a batch that found nothing."""
    with pytest.raises(ValueError):
        mc.extract_text(mc.MODELS["sol"], {"choices": []})
    with pytest.raises(ValueError):
        mc.extract_text(mc.MODELS["haiku45"], {"content": [{"text": "   "}]})


def test_transient_errors_are_told_apart_from_real_ones():
    assert mc.is_transient(RuntimeError("ThrottlingException: slow down"))
    assert mc.is_transient(RuntimeError("HTTP 429 rate limit exceeded"))
    assert not mc.is_transient(ValueError("AccessDeniedException"))


# ---------------------------------------------------------------------------
# T7 -- orchestration, with the backend faked out (no network)
# ---------------------------------------------------------------------------

def test_run_batches_calls_the_model_once_per_batch(monkeypatch):
    calls: list[str] = []

    def fake_invoke(spec, prompt):
        calls.append(prompt)
        return json.dumps(_SCHEMA)

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    batches = ss.batch_documents(_docs(25), 10)
    schemas, records = ss.run_batches(mc.MODELS["haiku45"], batches, "P " + ss.DOCUMENTS_PLACEHOLDER)

    assert len(calls) == 3
    assert len(schemas) == 3
    assert [r["batch"] for r in records] == [1, 2, 3]
    assert records[0]["source_documents"] == [s for s, _ in batches[0]]


def test_one_unparseable_batch_is_recorded_and_skipped(monkeypatch):
    """A run that has already been paid for should not be thrown away by one bad
    response -- but the loss has to be visible, so it lands in the raw log."""
    responses = iter([json.dumps(_SCHEMA), "sorry, I cannot help with that"])
    monkeypatch.setattr(mc, "invoke", lambda spec, prompt: next(responses))

    batches = ss.batch_documents(_docs(20), 10)
    schemas, records = ss.run_batches(mc.MODELS["llama318b"], batches, ss.DOCUMENTS_PLACEHOLDER)

    assert len(schemas) == 1
    assert "parse_error" in records[1]
    assert records[1]["response"] == "sorry, I cannot help with that"


# ---------------------------------------------------------------------------
# T8 -- no gold vocabulary anywhere (Critical Rule 1)
# ---------------------------------------------------------------------------

def _gold_terms() -> set[str]:
    from eval.matching import normalize

    vocabulary = {normalize(v) for v in _gold_vocabulary()}
    vocabulary.discard("")
    assert vocabulary, "gold vocabulary came back empty -- the check would be vacuous"
    return vocabulary


def _contains_subsequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    span = len(needle)
    return any(
        haystack[i : i + span] == needle for i in range(len(haystack) - span + 1)
    )


def leaks_in_text(text: str) -> list[str]:
    """Gold terms appearing in free text -- used on the prompt.

    The module check inspects string literals and identifiers, because in code
    only those can steer behavior. In a prompt there is no such distinction: the
    prose *is* the instruction, so the whole file is scanned. Contract keys are
    exempt, since the prompt cannot ask for the required output shape without
    naming its keys.
    """
    from eval.matching import normalize

    exempt = {normalize(k) for k in _CONTRACT_KEYS}
    tokens = tuple(normalize(text).split())
    unexempt = set(tokens) - exempt

    found = []
    for term in sorted(_gold_terms()):
        needle = tuple(term.split())
        hit = term in unexempt if len(needle) == 1 else _contains_subsequence(tokens, needle)
        if hit:
            found.append(term)
    return found


def test_no_domain_vocabulary_leakage():
    """Critical Rule 1, over both modules and the prompt.

    A gold term reaching the prompt would make every model an oracle that was
    handed the answer key, and every B3-vs-P1 number in the paper meaningless.
    The prompt is the likeliest place for this to happen by accident -- writing
    "for example, a Lease" while drafting instructions is a natural thing to do.
    """
    root = ss._REPO_ROOT / "baselines"
    vocabulary = _gold_terms()

    leaks: list[str] = []
    for module in _MODULES:
        used = _executable_vocabulary((root / module).read_text(encoding="utf-8"))
        for term in sorted(vocabulary):
            needle = tuple(term.split())
            for item in used:
                words = tuple(item.split())
                hit = (
                    item == term
                    if len(needle) == 1
                    else _contains_subsequence(words, needle)
                )
                if hit:
                    leaks.append(f"{module}: gold {term!r} appears as {item!r}")

    for term in leaks_in_text((root / _PROMPT).read_text(encoding="utf-8")):
        leaks.append(f"{_PROMPT}: gold {term!r} appears in the prompt text")

    assert not leaks, (
        "gold-schema vocabulary found in B3 (Critical Rule 1):\n" + "\n".join(sorted(set(leaks)))
    )


def test_the_prompt_leakage_check_actually_catches_a_planted_term():
    """Proves the scan above is not vacuously passing.

    Without this, a bug in normalization or tokenization would make the guard
    quietly approve any prompt at all.

    Plants a *raw* gold-vocabulary term -- as it would actually appear if typed
    into a prompt -- rather than an already-normalized one. eval.matching.normalize
    is not a fixed point (normalize("address") == "addres", itself unequal to
    normalize("addres") == "addre"), and leaks_in_text normalizes its whole input
    exactly once; feeding it an already-normalized planted term would silently
    double-normalize just that one token and could miss a real hit for reasons
    that have nothing to do with the leakage check itself.
    """
    planted = sorted(_gold_vocabulary())[0]
    assert leaks_in_text(f"Extract entities such as {planted} from the documents.")
    assert not leaks_in_text("Extract the recurring kinds of entity you find.")


def test_the_prompt_still_names_the_contract_keys_it_must():
    """The exemption in leaks_in_text is only safe because these are genuinely
    required -- the model cannot emit valid output without being told them."""
    prompt = (ss._REPO_ROOT / "baselines" / _PROMPT).read_text(encoding="utf-8")
    for key in ("classes", "relations", "name", "parent", "attributes", "source", "label", "target"):
        assert key in prompt
