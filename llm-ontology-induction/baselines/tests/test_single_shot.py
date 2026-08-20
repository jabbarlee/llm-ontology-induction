"""
B3 baseline validation suite.

Every test here runs offline: no network call of any kind. The backend layer is
split so that the parts worth asserting on -- the request kwargs, the response
reader, the malformed-element cleanup -- are pure functions, and the tests that
exercise orchestration substitute a fake `invoke`.

The Critical Rules that make B3 a valid baseline each have a test that fails if
the rule is broken:

  Rule 1 (no gold vocabulary)   test_no_domain_vocabulary_leakage
  Rule 2 (one frozen prompt)    test_instruction_text_is_identical_regardless_of_documents
  Rule 5 (no pre-cleaning)      test_names_are_emitted_exactly_as_the_model_returned_them
  Rule 6 (parent stays null)    test_parent_stays_null_unless_a_model_supplied_one

Plus the properties that corrupt a reported number if broken rather than merely
break the code:

  a truncated response is never scored     test_a_capped_response_raises_instead_of_yielding_a_partial_schema
  a refusal is never scored                test_a_refusal_raises_a_distinct_error
  the one call aborts (nothing to skip to) test_an_unparseable_response_aborts_instead_of_emitting_nothing
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from baselines.b3_single_shot import single_shot as ss
from baselines.shared import model_clients as mc

# Reused rather than reimplemented: one source of truth for what counts as gold
# vocabulary and as executable vocabulary, shared with the B1 suite.
from baselines.tests.test_statistical import (
    _CONTRACT_KEYS,
    _executable_vocabulary,
    _gold_vocabulary,
)

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


# A minimal stand-in for the anthropic SDK's Message object -- just enough
# surface for extract_from_anthropic_message to read: .content (list of blocks with .type
# and, for text blocks, .text), .stop_reason, .usage.output_tokens, and
# (only on a refusal) .stop_details.category / .explanation.

class _Block:
    def __init__(self, type_: str, text: str = ""):
        self.type = type_
        self.text = text


class _Usage:
    def __init__(self, output_tokens: int | None):
        self.output_tokens = output_tokens


class _StopDetails:
    def __init__(self, category: str | None = None, explanation: str | None = None):
        self.category = category
        self.explanation = explanation


class _Message:
    def __init__(self, text: str, stop_reason: str, tokens: int | None = 100, stop_details=None):
        self.content = [_Block("text", text)] if text else []
        self.stop_reason = stop_reason
        self.usage = _Usage(tokens)
        self.stop_details = stop_details


# ---------------------------------------------------------------------------
# T1 -- corpus loading (unchanged from before the rework)
# ---------------------------------------------------------------------------

def test_documents_are_read_whole_and_in_a_deterministic_order(corpus):
    """B3 hands raw files to the model -- no sentence splitting, no CSV
    flattening. That B1 preprocesses and B3 does not is a real difference
    between the two conditions, so the absence of preprocessing is asserted."""
    documents = ss.load_documents(corpus)
    sources = [source for source, _text in documents]

    assert sources == [
        "csv_exports/a.csv",
        "lease_texts/t1.txt",
        "notes/n1.txt",
        "messages/m1.txt",
        "csv_exports/b.csv",
        "lease_texts/t2.txt",
    ], "round-robin across subdirectories, filenames sorted within each"

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


def test_instruction_text_is_identical_regardless_of_documents():
    """Critical Rule 2: only the documents block varies, byte for byte -- true
    whether the call carries all 192 documents (B3) or exactly 1 (P1's
    per-document extraction stage, which renders through this same function)."""
    template = ss.load_prompt_template()
    head, tail = template.split(ss.DOCUMENTS_PLACEHOLDER)

    first = ss.render_prompt(template, _docs(3))
    second = ss.render_prompt(template, _docs(1))

    assert first.startswith(head) and first.endswith(tail)
    assert second.startswith(head) and second.endswith(tail)
    assert first != second, "the documents block must actually be substituted"


def test_rendering_survives_json_braces_in_the_template():
    """Why render_prompt uses str.replace and not str.format. The frozen prompt
    shows a JSON output example; str.format would read those braces as fields."""
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
    response = (
        "Sure! Here is the schema I extracted:\n\n"
        + json.dumps(_SCHEMA)
        + "\n\nLet me know if you would like me to expand any of these."
    )
    assert ss.parse_response(response) == _SCHEMA


def test_parses_through_a_reasoning_block_without_taking_its_contents():
    """Defensive, not model-specific: Opus's thinking arrives as separate
    content blocks the SDK already splits out (extract_from_anthropic_message only
    assembles text-type blocks), so neither frozen condition is
    expected to emit inline <think> tags -- but the parser must not be fooled
    if the visible text ever contains one anyway."""
    response = (
        "<think>\nFirst I will draft {\"classes\": [], \"relations\": []} and check it.\n"
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
    """The one failure mode that would corrupt a reported number invisibly: an
    empty schema here is indistinguishable downstream from a call that
    genuinely found nothing, so it must raise."""
    with pytest.raises(ss.ResponseParseError):
        ss.parse_response(response)


# ---------------------------------------------------------------------------
# T4 -- output cleanup (B3-D4, revised -- single response only, no cross-call merge)
# ---------------------------------------------------------------------------

def test_case_and_whitespace_duplicates_within_one_response_collapse():
    cleaned = ss.clean_schema(
        {"classes": [
            {"name": "Owner", "parent": None, "attributes": []},
            {"name": "  owner ", "parent": None, "attributes": []},
        ]}
    )
    assert [c["name"] for c in cleaned["classes"]] == ["Owner"]


def test_distinct_wordings_are_never_merged():
    """The load-bearing test of this file. A smarter method could obviously
    resolve these two to one entity -- doing so is P1's job (its consolidation
    stage), not B3's. Failing this test does not mean B3 got worse; it means
    B3 stopped being the baseline the paper claims it is."""
    cleaned = ss.clean_schema(
        {"classes": [
            {"name": "Owner", "parent": None, "attributes": []},
            {"name": "Landlord", "parent": None, "attributes": []},
        ]}
    )
    assert sorted(c["name"] for c in cleaned["classes"]) == ["Landlord", "Owner"]


def test_attributes_dedup_within_one_class_keeping_first_surface_form():
    cleaned = ss.clean_schema(
        {"classes": [{"name": "Alpha", "parent": None, "attributes": ["Size", "size", "Hue"]}]}
    )
    assert cleaned["classes"][0]["attributes"] == ["Size", "Hue"]


def test_names_are_emitted_exactly_as_the_model_returned_them():
    """Critical Rule 5: the harness normalizes, the producer never pre-cleans."""
    ugly = "  Maintenance   REQUESTS_v2  "
    cleaned = ss.clean_schema(
        {"classes": [{"name": ugly, "parent": None, "attributes": [" Odd  Field "]}]}
    )
    assert cleaned["classes"][0]["name"] == ugly
    assert cleaned["classes"][0]["attributes"] == [" Odd  Field "]


def test_parent_stays_null_unless_a_model_supplied_one():
    """Critical Rule 6 -- no taxonomy is inferred here, ever."""
    cleaned = ss.clean_schema(
        {"classes": [
            {"name": "Alpha", "parent": None, "attributes": []},
            {"name": "Beta", "parent": "Alpha", "attributes": []},
        ]}
    )
    by_name = {c["name"]: c for c in cleaned["classes"]}
    assert by_name["Alpha"]["parent"] is None
    assert by_name["Beta"]["parent"] == "Alpha"


def test_relations_dedup_on_the_whole_triple():
    cleaned = ss.clean_schema(
        {"relations": [
            {"source": "Alpha", "label": "touches", "target": "Beta"},
            {"source": "alpha", "label": "Touches", "target": " beta"},
            {"source": "Alpha", "label": "avoids", "target": "Beta"},
        ]}
    )
    assert len(cleaned["relations"]) == 2
    assert cleaned["relations"][0] == {"source": "Alpha", "label": "touches", "target": "Beta"}


def test_malformed_elements_are_dropped_not_repaired():
    cleaned = ss.clean_schema({
        "classes": [
            {"parent": None, "attributes": []},          # no name
            {"name": "   ", "parent": None},              # blank name
            {"name": "Alpha", "attributes": [None, 7, "ok"]},
            "not even an object",
        ],
        "relations": [
            {"source": "Alpha", "label": "touches"},      # no target
            {"source": "Alpha", "label": "touches", "target": "Beta"},
        ],
    })
    assert [c["name"] for c in cleaned["classes"]] == ["Alpha"]
    assert cleaned["classes"][0]["attributes"] == ["ok"]
    assert len(cleaned["relations"]) == 1


def test_cleaning_nothing_yields_an_empty_schema():
    assert ss.clean_schema({}) == {"classes": [], "relations": []}


# ---------------------------------------------------------------------------
# T5 -- output contract (B3-D5)
# ---------------------------------------------------------------------------

def test_output_satisfies_the_induced_schema_contract():
    """Asserted by round-tripping through the harness's own loader rather than
    by re-describing the shape here -- if the contract moves, this moves with it."""
    from eval.schema_ir import Relation, effective_attributes, parse_induced_schema

    spec = mc.MODELS["haiku45"]
    output = ss.build_output(
        ss.clean_schema(_SCHEMA), ["dir/doc-000.txt"], "run-1", spec, "end_turn", 42
    )

    schema = parse_induced_schema(output)
    assert set(schema.classes) == {"Alpha", "Beta"}
    assert schema.classes["Beta"].parent == "Alpha"
    assert schema.classes["Alpha"].declared_attributes == frozenset({"a one", "a two"})
    assert effective_attributes(schema, "Beta") == frozenset({"a one", "a two"})
    assert Relation(source="Alpha", label="touches", target="Beta") in schema.relations


def test_output_metadata_names_the_condition_and_the_exact_model():
    spec = mc.MODELS["opus5"]
    output = ss.build_output(
        {"classes": [], "relations": []}, ["a/b.txt"], "run-1", spec, "end_turn", 17
    )

    assert set(output) == {"classes", "relations", "metadata"}
    assert output["metadata"] == {
        "condition": "B3",
        "model": spec.model_id,
        "run_id": "run-1",
        "source_documents": ["a/b.txt"],
        "stop_reason": "end_turn",
        "completion_tokens": 17,
    }


def test_output_is_json_serializable():
    spec = mc.MODELS["opus5"]
    output = ss.build_output(ss.clean_schema(_SCHEMA), [], "run-1", spec, "end_turn", 1)
    assert json.loads(json.dumps(output)) == output


# ---------------------------------------------------------------------------
# T6 -- model registry and request construction (B3-D1, B3-D3)
# ---------------------------------------------------------------------------

def test_registry_holds_exactly_the_two_frozen_conditions_on_their_own_backends():
    assert set(mc.MODELS) == {"haiku45", "opus5"}
    for key, spec in mc.MODELS.items():
        assert spec.key == key, "registry key must match the spec it points at"
    assert mc.MODELS["haiku45"].backend == "bedrock"
    assert mc.MODELS["haiku45"].model_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert mc.MODELS["opus5"].backend == "anthropic_api"
    assert mc.MODELS["opus5"].model_id == "claude-opus-5"


def test_model_specs_are_immutable():
    with pytest.raises(FrozenInstanceError):
        mc.MODELS["haiku45"].max_output_tokens = 999


def test_haiku_on_bedrock_keeps_temperature_but_drops_top_p():
    """Confirmed at a real call: Haiku 4.5 on Bedrock 400s if both temperature
    and top_p are present, regardless of value -- not either one alone.
    temperature=0.0 is kept (the load-bearing reproducibility setting); top_p is
    dropped, per B3-D3's own note that top_p=1.0 is a no-op at temperature 0."""
    spec = mc.MODELS["haiku45"]
    assert spec.supports_temperature is True
    assert spec.supports_top_p is False
    body = mc.build_request(spec, "prompt text")
    assert body["temperature"] == mc.TEMPERATURE == 0.0
    assert "top_p" not in body


def test_opus5_takes_neither_sampling_setting():
    """Opus 4.7+ rejects temperature/top_p outright -- a documented Anthropic
    API constraint, not a per-model preference."""
    spec = mc.MODELS["opus5"]
    assert spec.supports_temperature is False
    assert spec.supports_top_p is False
    kwargs = mc.build_request(spec, "prompt text")
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_only_opus_requests_thinking_and_effort():
    haiku_body = mc.build_request(mc.MODELS["haiku45"], "p")
    assert "thinking" not in haiku_body
    assert "output_config" not in haiku_body

    opus_kwargs = mc.build_request(mc.MODELS["opus5"], "p")
    assert opus_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert opus_kwargs["output_config"] == {"effort": "high"}


def test_bedrock_and_anthropic_api_bodies_have_the_right_shape_for_their_transport():
    """The two backends build genuinely different request shapes -- Bedrock's
    Anthropic-on-Bedrock body (`anthropic_version`, block-structured content, no
    `model` key -- the model ID goes on the invoke_model call, not the body) vs.
    the direct API's kwargs (`model` present, plain-string content)."""
    bedrock_body = mc.build_request(mc.MODELS["haiku45"], "SENTINEL")
    assert bedrock_body["anthropic_version"] == "bedrock-2023-05-31"
    assert bedrock_body["max_tokens"] == mc.MODELS["haiku45"].max_output_tokens
    assert bedrock_body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "SENTINEL"}]}]
    assert "model" not in bedrock_body

    api_kwargs = mc.build_request(mc.MODELS["opus5"], "SENTINEL")
    assert api_kwargs["model"] == "claude-opus-5"
    assert api_kwargs["max_tokens"] == mc.MODELS["opus5"].max_output_tokens
    assert api_kwargs["messages"] == [{"role": "user", "content": "SENTINEL"}]


def test_build_request_is_a_pure_function_of_spec_and_prompt():
    """No batching or call-shape argument exists to thread through -- the
    request for a given model is identical whether it is B3's one whole-corpus
    call or one of P1's per-document calls."""
    import inspect

    assert list(inspect.signature(mc.build_request).parameters) == ["spec", "prompt"]


def test_build_request_rejects_an_unknown_backend():
    spec = mc.ModelSpec(key="x", model_id="x", tier="budget", backend="nowhere", max_output_tokens=1)
    with pytest.raises(ValueError):
        mc.build_request(spec, "prompt text")
    with pytest.raises(ValueError):
        mc.invoke(spec, "prompt text")


# ---------------------------------------------------------------------------
# T7 -- response reading (truncation, refusal, empty), both backends
# ---------------------------------------------------------------------------

def _bedrock_payload(text: str, stop_reason: str, tokens: int | None = 100, stop_details=None) -> dict:
    payload = {
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "usage": {"output_tokens": tokens},
    }
    if stop_details is not None:
        payload["stop_details"] = stop_details
    return payload


def test_a_normal_bedrock_response_extracts_cleanly():
    completion = mc.extract_from_bedrock_payload(
        mc.MODELS["haiku45"], _bedrock_payload("hello", "end_turn", tokens=5)
    )
    assert completion.text == "hello"
    assert completion.stop_reason == "end_turn"
    assert completion.completion_tokens == 5


def test_a_normal_anthropic_api_response_extracts_cleanly():
    message = _Message("hello", "end_turn", tokens=5)
    completion = mc.extract_from_anthropic_message(mc.MODELS["opus5"], message)
    assert completion.text == "hello"
    assert completion.stop_reason == "end_turn"
    assert completion.completion_tokens == 5


def test_a_capped_bedrock_response_raises_instead_of_yielding_a_partial_schema():
    payload = _bedrock_payload('{"classes": [{"na', "max_tokens", tokens=16000)
    with pytest.raises(mc.TruncatedResponseError) as caught:
        mc.extract_from_bedrock_payload(mc.MODELS["haiku45"], payload)
    assert caught.value.text == '{"classes": [{"na'
    assert caught.value.completion_tokens == 16000


def test_a_capped_anthropic_api_response_raises_instead_of_yielding_a_partial_schema():
    message = _Message('{"classes": [{"na', "max_tokens", tokens=32000)
    with pytest.raises(mc.TruncatedResponseError) as caught:
        mc.extract_from_anthropic_message(mc.MODELS["opus5"], message)
    assert caught.value.text == '{"classes": [{"na'
    assert caught.value.completion_tokens == 32000


def test_a_bedrock_refusal_raises_a_distinct_error():
    payload = _bedrock_payload("", "refusal", tokens=0, stop_details={"category": "cyber", "explanation": "policy block"})
    with pytest.raises(mc.RefusalError) as caught:
        mc.extract_from_bedrock_payload(mc.MODELS["haiku45"], payload)
    assert "cyber" in str(caught.value)
    assert "policy block" in str(caught.value)
    assert not isinstance(caught.value, mc.TruncatedResponseError)


def test_an_anthropic_api_refusal_raises_a_distinct_error():
    message = _Message("", "refusal", tokens=0, stop_details=_StopDetails("cyber", "policy block"))
    with pytest.raises(mc.RefusalError) as caught:
        mc.extract_from_anthropic_message(mc.MODELS["opus5"], message)
    assert "cyber" in str(caught.value)
    assert "policy block" in str(caught.value)


def test_a_refusal_with_no_stop_details_still_raises_cleanly():
    """stop_details is only guaranteed populated on a refusal, and even then not
    every field is guaranteed present -- must not crash reading it, on either
    backend."""
    with pytest.raises(mc.RefusalError):
        mc.extract_from_bedrock_payload(mc.MODELS["haiku45"], _bedrock_payload("", "refusal", tokens=0))
    with pytest.raises(mc.RefusalError):
        mc.extract_from_anthropic_message(mc.MODELS["opus5"], _Message("", "refusal", tokens=0, stop_details=None))


def test_extract_refuses_an_empty_normal_response_on_either_backend():
    """An empty string on an ordinary end_turn would be recorded as a call that
    found nothing -- must raise, not return ""."""
    with pytest.raises(ValueError):
        mc.extract_from_bedrock_payload(mc.MODELS["haiku45"], _bedrock_payload("   ", "end_turn", tokens=0))
    with pytest.raises(ValueError):
        mc.extract_from_anthropic_message(mc.MODELS["opus5"], _Message("   ", "end_turn", tokens=0))


def test_response_errors_are_not_the_sdks_own_retryable_exception_types():
    """Both are deterministic at the settings this project freezes -- retrying
    would only spend money to fail the same way twice. There is no custom
    retry/is_transient layer to test here anymore (see model_clients.py's
    module docstring): retries are the SDK's own, gated on the typed exceptions
    it raises for 429/5xx/connection errors. TruncatedResponseError and
    RefusalError are raised from a successfully-completed call, never from a
    transport failure, so they must never collide with that hierarchy."""
    for cls in (mc.TruncatedResponseError, mc.RefusalError):
        assert issubclass(cls, mc.ModelResponseError)
        assert issubclass(cls, Exception)
        assert not issubclass(cls, (ConnectionError, TimeoutError))


# ---------------------------------------------------------------------------
# T8 -- orchestration, with the backend faked out (no network)
# ---------------------------------------------------------------------------

def _completion(text: str, stop_reason: str = "end_turn", tokens: int = 100):
    return mc.Completion(text=text, stop_reason=stop_reason, completion_tokens=tokens)


def test_run_whole_corpus_makes_exactly_one_call(monkeypatch):
    calls: list[str] = []

    def fake_invoke(spec, prompt):
        calls.append(prompt)
        return _completion(json.dumps(_SCHEMA))

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    documents = _docs(50)
    record: dict = {}
    schema = ss.run_whole_corpus(mc.MODELS["opus5"], documents, ss.DOCUMENTS_PLACEHOLDER, record)

    assert len(calls) == 1
    assert schema == _SCHEMA
    assert record["source_documents"] == [s for s, _ in documents]
    assert record["stop_reason"] == "end_turn"
    for source, _text in documents:
        assert source in calls[0], "every document must ride in the single call"


def test_an_unparseable_response_aborts_instead_of_emitting_nothing(monkeypatch):
    """With one call there is nothing to fall back to: skipping would merge
    zero schemas, write a well-formed file with no classes in it, and report a
    completed run -- indistinguishable downstream from a model that genuinely
    found no structure."""
    monkeypatch.setattr(mc, "invoke", lambda spec, prompt: _completion("sorry, I cannot help"))
    record: dict = {}
    with pytest.raises(ss.ResponseParseError):
        ss.run_whole_corpus(mc.MODELS["opus5"], _docs(5), ss.DOCUMENTS_PLACEHOLDER, record)
    assert record["response"] == "sorry, I cannot help", "the paid-for response must still be captured"


def test_a_truncated_call_is_captured_before_the_run_dies(monkeypatch):
    def fake_invoke(spec, prompt):
        raise mc.TruncatedResponseError("stopped at cap", '{"classes": [{"na', 32000)

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    record: dict = {}
    with pytest.raises(mc.TruncatedResponseError):
        ss.run_whole_corpus(mc.MODELS["opus5"], _docs(5), ss.DOCUMENTS_PLACEHOLDER, record)
    assert record["stop_reason"] == "truncated"
    assert record["completion_tokens"] == 32000
    assert record["response"] == '{"classes": [{"na'


def test_a_refused_call_is_captured_distinctly_from_a_truncated_one(monkeypatch):
    def fake_invoke(spec, prompt):
        raise mc.RefusalError("refused", "", None)

    monkeypatch.setattr(mc, "invoke", fake_invoke)
    record: dict = {}
    with pytest.raises(mc.RefusalError):
        ss.run_whole_corpus(mc.MODELS["opus5"], _docs(5), ss.DOCUMENTS_PLACEHOLDER, record)
    assert record["stop_reason"] == "refused"


# ---------------------------------------------------------------------------
# T9 -- no gold vocabulary anywhere (Critical Rule 1)
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
    """Gold terms appearing in free text -- used on the prompts. Contract keys
    are exempt, since a prompt cannot ask for the required output shape
    without naming its keys."""
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
    """Critical Rule 1, over the B3 module, the shared model-calling module, and
    the frozen prompt. A gold term reaching the prompt would make the model an
    oracle handed the answer key, and every B3-vs-P1 number in the paper
    meaningless."""
    modules = [
        ss._MODULE_ROOT / "single_shot.py",
        ss._REPO_ROOT / "baselines" / "shared" / "model_clients.py",
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
                    item == term
                    if len(needle) == 1
                    else _contains_subsequence(words, needle)
                )
                if hit:
                    leaks.append(f"{path.name}: gold {term!r} appears as {item!r}")

    prompt_path = ss._MODULE_ROOT / _PROMPT
    for term in leaks_in_text(prompt_path.read_text(encoding="utf-8")):
        leaks.append(f"{_PROMPT}: gold {term!r} appears in the prompt text")

    assert not leaks, (
        "gold-schema vocabulary found in B3 (Critical Rule 1):\n" + "\n".join(sorted(set(leaks)))
    )


def test_the_prompt_leakage_check_actually_catches_a_planted_term():
    """Proves the scan above is not vacuously passing. Plants a *raw* gold term
    -- as it would actually appear typed into a prompt -- rather than an
    already-normalized one, since eval.matching.normalize is not a fixed point."""
    planted = sorted(_gold_vocabulary())[0]
    assert leaks_in_text(f"Extract entities such as {planted} from the documents.")
    assert not leaks_in_text("Extract the recurring kinds of entity you find.")


def test_the_prompt_still_names_the_contract_keys_it_must():
    prompt = (ss._MODULE_ROOT / _PROMPT).read_text(encoding="utf-8")
    for key in ("classes", "relations", "name", "parent", "attributes", "source", "label", "target"):
        assert key in prompt
