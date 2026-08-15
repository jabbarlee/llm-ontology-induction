"""
B3 baseline -- model backends (baselines/DECISIONS.md B3-D1, B3-D1a, B3-D1c, B3-D3,
B3-D6).

Six frozen conditions behind one `invoke()` entry point, so single_shot.py never
branches on which model is running -- that branching is exactly what would let the
conditions drift apart, and B3's whole premise is that the task shape is identical
for all of them.

Two backends:
  - AWS Bedrock for the four cloud models (Fable 5, Haiku 4.5, Sol, Luna) and, since
    the 2026-08-14 rework, for the open-weight model as well
  - Groq's free tier, retained for the one historical run made there (B3-D6), so that
    result stays reproducible rather than becoming an artifact nothing can regenerate

Three request/response shapes live on Bedrock -- anthropic, openai and meta -- which
is why `family` exists separately from `backend`.

Sampling settings are frozen module constants (B3-D3), never per-call arguments -- a
per-call temperature is a knob, and a knob on a frozen baseline eventually gets turned.
The one exception is the output cap, which moved onto `ModelSpec` in the same rework:
it is a property of the model as served (Meta is capped at 4K where the Anthropic and
OpenAI families are not), and a single global was only ever right for one call shape.
It is deliberately *not* a property of the call shape -- whole-corpus and batched runs
of the same model send a byte-identical body, which is what makes them comparable.

HARD RULE (Critical Rule 1): zero gold-schema vocabulary in this module. Enforced by
baselines/tests/test_single_shot.py::test_no_domain_vocabulary_leakage.

boto3 and groq are imported lazily inside the invocation functions, so the pure
functions here (and therefore the whole test suite) import cleanly on a machine
where neither SDK is installed.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# --- Frozen sampling settings (B3-D3) -------------------------------------
TEMPERATURE = 0.0
TOP_P = 1.0

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; doubled per attempt with jitter

BEDROCK_REGION = os.environ.get("AWS_REGION") or "us-east-1"


class ModelResponseError(RuntimeError):
    """Base for a response that must never be scored, even though a call was paid
    for and completed without a transport-level error.

    Both subclasses below are deliberately fatal and deliberately *not* retried --
    at TEMPERATURE = 0.0, retrying an identical prompt reproduces the identical
    failure, so retrying only spends money to fail the same way twice. Both carry
    whatever text the provider did return rather than discarding it: fatal must not
    mean evidence-destroying, and how far a call got before it failed is itself
    something B3-D3's and B3-D1's decision rules turn into reported findings. The
    text is attached for the run log and never returned through a path that could
    parse it into a schema.

    Every message built from this base must contain none of `_TRANSIENT_MARKERS`,
    or `is_transient()` would classify a deterministic failure as a hiccup and burn
    MAX_RETRIES attempts on it -- pinned by
    test_a_truncated_response_is_never_treated_as_transient and
    test_a_refusal_is_never_treated_as_transient.
    """

    def __init__(self, message: str, text: str = "", completion_tokens: int | None = None):
        super().__init__(message)
        self.text = text
        self.completion_tokens = completion_tokens


class TruncatedResponseError(ModelResponseError):
    """The provider stopped generating because it hit the output cap.

    The only fixes are a bigger cap or a smaller call, and both are frozen values
    (B3-D2, B3-D3) that must not be mutated mid-run.
    """


class RefusalError(ModelResponseError):
    """The provider declined to answer on content-policy grounds
    (`stop_reason: "refusal"`), not a token-budget problem.

    Specific to the anthropic family so far -- Claude Fable 5's model card documents
    this explicitly and warns its refusal rate is "materially higher than on
    previous Claude models," instructing callers to "handle stop_reason: 'refusal'
    as a primary response path" rather than treat it as a rare edge case. Kept
    distinct from TruncatedResponseError rather than folded into one generic
    failure: a content-policy block and a token-cap cutoff are different findings
    (a modeling/prompt-content question vs. a serving-limit question), and
    collapsing them would blur which one a given run actually hit.
    """


@dataclass(frozen=True)
class Completion:
    """One model response plus the two facts needed to trust it.

    `stop_reason` and `completion_tokens` are carried out of the backend rather than
    discarded because a schema that was cut off mid-JSON and a schema the model
    genuinely finished are indistinguishable once the text is parsed -- and under
    whole-corpus prompting a truncation is not one bad batch out of 28, it is the
    entire run silently halved.
    """

    text: str
    stop_reason: str | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    """One frozen model condition.

    `model_id` is the exact provider identifier and is what lands in
    `metadata.model` of the emitted schema -- results must name the artifact that
    produced them, not a friendly alias that could later be repointed.
    """

    key: str  # CLI selector
    model_id: str  # exact provider identifier -> metadata.model
    backend: str  # "bedrock" | "groq"
    tier: str  # "frontier" | "budget" | "open-weight"
    max_output_tokens: int  # B3-D3, per model -- see the module docstring
    family: str | None = None  # request/response shape on Bedrock
    region: str | None = None  # overrides BEDROCK_REGION where a model demands one
    reasoning_effort: str | None = None  # Critical Rule 3 -- Luna only
    supports_temperature: bool = True
    # Independent from supports_temperature, not a duplicate of it: Claude 4.5+ on
    # Bedrock (confirmed here for Haiku 4.5, also reported for Sonnet 4.5 and Opus
    # 4.7/4.8) reject a request that specifies BOTH temperature and top_p -- not
    # either one alone. Dropping top_p rather than temperature is not a new judgment
    # call: B3-D3's own table already calls top_p=1.0 "Neutral -- with temperature at
    # 0 it does nothing," so this is applying existing reasoning, not inventing new.
    supports_top_p: bool = True


# TODO: Sol and Luna cannot be reached through this registry at all yet. Their model
# card shows `bedrock-runtime` unsupported entirely (Invoke and Converse both ✗) --
# they live only on `bedrock-mantle`'s Responses API, reached through the `openai` SDK
# (`client.responses.create(model=..., input=...)`), a different request/response
# shape than anything build_request_body/extract_text handle. Calling `invoke()` for
# either key raises ValueError from the unknown-backend branch rather than silently
# doing the wrong thing, but a real fourth backend has not been built -- do not run
# `--model sol` or `--model luna` expecting it to work.
MODELS: dict[str, ModelSpec] = {
    "fable5": ModelSpec(
        key="fable5",
        # The geo inference ID, not the bare `anthropic.claude-fable-5`. Same pattern
        # as Haiku 4.5 below: the model card's Programmatic Access table lists the
        # bare ID's In-Region endpoint URL as "N/A" for bedrock-runtime -- on-demand
        # invoke only works through a Geo/Global inference ID. Not yet runtime-
        # confirmed the way Haiku's was (that came from an actual ValidationException);
        # fixed proactively on the doc pattern. Smoke-test with --limit before trusting
        # this without question.
        model_id="us.anthropic.claude-fable-5",
        backend="bedrock",
        tier="frontier",
        max_output_tokens=16000,
        family="anthropic",
        # B3-D3's frozen values are not just rejected in combination (Haiku's
        # problem) -- they are individually out of range for this model. The model
        # card states: "temperature must be 1.0 or unset; top_p must be >= 0.99 and
        # < 1.0, or unset." TEMPERATURE=0.0 fails the first constraint outright;
        # TOP_P=1.0 fails the second (the range is exclusive at 1.0). Omitting both
        # ("unset") is the only frozen-adjacent option this model actually accepts --
        # not a value we chose, a constraint we have no room to negotiate inside.
        # This is a real, recorded loss, not a free substitution: B3-D3 freezes
        # temperature at (near-)zero specifically for run-to-run reproducibility,
        # and that guarantee does not hold for this condition. Recorded in
        # DECISIONS.md as a B3-D3 exception, not silently absorbed here.
        supports_temperature=False,
        supports_top_p=False,
    ),
    "haiku45": ModelSpec(
        key="haiku45",
        # Confirmed at the first real call: `anthropic.claude-haiku-4-5-20251001-v1:0`
        # (the bare ID) raised "Invocation of model ID ... with on-demand throughput
        # isn't supported. Retry your request with the ID or ARN of an inference
        # profile that contains this model." The model card explains why: its
        # bedrock-runtime row lists the bare ID's In-Region endpoint URL as "N/A" --
        # only the Geo/Global inference ID is invokable on demand.
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        backend="bedrock",
        tier="budget",
        # Haiku 4.5's real ceiling on Bedrock is 64,000. 16,000 is chosen for headroom
        # over the largest plausible whole-corpus schema, not because 64,000 is
        # unavailable.
        max_output_tokens=16000,
        family="anthropic",
        # Confirmed at the second real call: a body carrying both temperature and
        # top_p raised "`temperature` and `top_p` cannot both be specified for this
        # model. Please use only one." temperature=0.0 is kept (B3-D3's load-bearing
        # setting); top_p is the one dropped, per the reasoning on ModelSpec above.
        supports_top_p=False,
    ),
    "sol": ModelSpec(
        key="sol",
        model_id="openai.gpt-5.6-sol",
        backend="bedrock",
        tier="frontier",
        max_output_tokens=16000,
        family="openai",
        # This family is served from us-east-1/us-east-2 only (bedrock-mantle, per the
        # model card's regional table -- Geo/Global are both "Not supported" for this
        # model, so there is no cross-region fallback). Set `region` here if
        # AWS_REGION is anything else. Moot until the Responses-API backend above
        # exists; invoke() raises before this would ever matter.
    ),
    "luna": ModelSpec(
        key="luna",
        model_id="openai.gpt-5.6-luna",
        backend="bedrock",
        tier="budget",
        max_output_tokens=16000,
        family="openai",
        # Critical Rule 3 / B3-D1a: capability-matched to Haiku 4.5 at low effort.
        # Any other value makes this condition a different experiment.
        reasoning_effort="low",
    ),
    "llama318b_bedrock": ModelSpec(
        key="llama318b_bedrock",
        # The *geo* inference ID, not the bare `meta.` one. Read off the model card's
        # own regional table: us-east-1 and us-east-2 are In-Region NO / Geo YES, and
        # only us-west-2 serves the bare ID on demand. The geo ID is callable from all
        # three, which makes the region question moot for this condition instead of
        # pinning it to one region.
        # TODO: availability is not access -- confirm this model is enabled for the
        # account in the Bedrock console before the first run.
        model_id="us.meta.llama3-1-8b-instruct-v1:0",
        backend="bedrock",
        tier="open-weight",
        # 4096, not the 16000 the other Bedrock conditions get. The model card states
        # "Max output tokens: 4K" flat, against a 128K context window; that is a
        # property of the model as served, and Converse maps onto the same limit, so
        # there is no API surface that escapes it. Not a silent clamp -- see B3-D3,
        # which also predicts, before the run, that a whole-corpus answer may not fit.
        max_output_tokens=4096,
        family="meta",
    ),
    "llama318b_groq": ModelSpec(
        key="llama318b_groq",
        # Confirmed against Groq's catalog directly (not a Bedrock-console lookup,
        # so it carries no TODO): swapped in for qwen3:8b, which Groq does not serve
        # at that weight class (see B3-D1c).
        model_id="llama-3.1-8b-instant",
        backend="groq",
        tier="open-weight",
        # Held at the historical value on purpose (B3-D6). This condition exists to
        # keep the 2026-08-12 batched run reproducible; raising its cap would make a
        # re-run a different experiment from the one already reported.
        max_output_tokens=2048,
    ),
}


# ---------------------------------------------------------------------------
# Request bodies -- pure, no network (so the tests can assert on them)
# ---------------------------------------------------------------------------

# Meta's documented chat template. Applied here rather than in the prompt file
# because it is transport framing, not instruction text: the frozen prompt
# (Critical Rule 2) stays byte-identical across all six conditions, and this
# wrapper is what the Bedrock Meta family requires around any prompt at all.
_META_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
_META_SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def build_request_body(spec: ModelSpec, prompt: str) -> dict:
    """Provider payload for `spec` carrying `prompt` as the sole user turn.

    Split out from the invocation functions precisely so Critical Rule 3 is
    testable without a network call: the assertion that Luna always carries
    reasoning_effort="low" runs against this dict.

    Takes no batching argument, and must not gain one. The output cap comes from the
    spec alone, so a whole-corpus call and a batched call to the same model produce a
    byte-identical body -- pinned by
    test_the_body_is_identical_whatever_the_call_shape. If the cap could vary with the
    call shape, the two arms of B3-D6's comparison would differ in two ways at once
    and neither could be attributed.

    `supports_temperature` exists because reasoning-tuned models have historically
    rejected any non-default temperature. If a model refuses the frozen value, flip
    that flag on its spec rather than special-casing the sampling settings -- the
    omission is then visible in one place and recordable in DECISIONS.md.
    """
    if spec.backend == "groq":
        body = {
            "model": spec.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": spec.max_output_tokens,
        }
        if spec.supports_temperature:
            body["temperature"] = TEMPERATURE
            body["top_p"] = TOP_P
        if spec.reasoning_effort is not None:
            body["reasoning_effort"] = spec.reasoning_effort
        return body

    if spec.backend == "bedrock" and spec.family == "anthropic":
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": spec.max_output_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        # Gated independently, not together: some Claude 4.5+ Bedrock models 400 if
        # both are present at once, regardless of value (see supports_top_p on
        # ModelSpec). supports_temperature alone would have to drop both to fix that,
        # which is wrong -- temperature=0.0 is the load-bearing setting (B3-D3);
        # top_p=1.0 is the one already documented as redundant at that temperature.
        if spec.supports_temperature:
            body["temperature"] = TEMPERATURE
        if spec.supports_top_p:
            body["top_p"] = TOP_P
        return body

    if spec.backend == "bedrock" and spec.family == "openai":
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": spec.max_output_tokens,
        }
        # TODO: confirm that Bedrock passes `reasoning_effort` through at the top
        # level of the request body for this family. If it instead expects a nested
        # object, change it here -- but it must reach the model on every Luna call.
        if spec.reasoning_effort is not None:
            body["reasoning_effort"] = spec.reasoning_effort
        if spec.supports_temperature:
            body["temperature"] = TEMPERATURE
            body["top_p"] = TOP_P
        return body

    if spec.backend == "bedrock" and spec.family == "meta":
        # The native shape from the Meta parameter reference. AWS also publishes a
        # contradictory sample on this model's own card that posts `messages` +
        # `max_tokens` to the same endpoint, with no documented response shape. The
        # native shape is chosen because it is the one whose `stop_reason` semantics
        # are documented, and the truncation guard below depends on exactly that
        # field. If the smoke test rejects this body, the fallback is one edit here.
        #
        # Concatenated, never str.format: the frozen prompt contains a JSON output
        # example, and format() would read its braces as replacement fields --
        # the same bug single_shot.render_prompt avoids for the same reason.
        body = {
            "prompt": _META_PREFIX + prompt + _META_SUFFIX,
            "max_gen_len": spec.max_output_tokens,
        }
        if spec.supports_temperature:
            body["temperature"] = TEMPERATURE
            body["top_p"] = TOP_P
        return body

    raise ValueError(f"no request body defined for {spec.key!r} ({spec.backend})")


# ---------------------------------------------------------------------------
# Response reading
# ---------------------------------------------------------------------------

# Per family, the value of the stop field that means "cut off at the cap" rather
# than "finished". Anything else -- including a value not listed here -- is treated
# as a completed generation.
_TRUNCATION_MARKERS = {
    "anthropic": ("stop_reason", "max_tokens"),
    "openai": ("finish_reason", "length"),
    "meta": ("stop_reason", "length"),
}


def _check_not_truncated(
    spec: ModelSpec,
    field: str,
    value: object,
    text: str = "",
    completion_tokens: int | None = None,
) -> None:
    family = spec.family or "openai"
    expected = _TRUNCATION_MARKERS.get(family)
    if expected and (field, value) == expected:
        raise TruncatedResponseError(
            f"{spec.key!r} stopped at its output cap of {spec.max_output_tokens} "
            f"tokens ({field}={value!r}); the schema it returned is cut off and is "
            "not usable. Not retried: at temperature 0 the same prompt returns the "
            "same cut-off answer. See DECISIONS.md B3-D3.",
            text=text,
            completion_tokens=completion_tokens,
        )


def _check_not_refused(
    spec: ModelSpec, payload: dict, text: str, completion_tokens: int | None
) -> None:
    """Anthropic-family only: `stop_reason: "refusal"` is a content-policy block,
    not a token-budget problem, and must not be mistaken for either truncation or
    an ordinary empty response. See RefusalError's docstring for why it stays a
    distinct exception rather than folding into TruncatedResponseError."""
    if spec.family == "anthropic" and payload.get("stop_reason") == "refusal":
        details = payload.get("stop_details")
        raise RefusalError(
            f"{spec.key!r} refused to respond (stop_reason='refusal'"
            + (f", stop_details={details!r}" if details else "")
            + "). Not retried: a content-policy refusal is deterministic at "
            "temperature 0 and will not change on a retry.",
            text=text,
            completion_tokens=completion_tokens,
        )


def extract_text(spec: ModelSpec, payload: dict) -> str:
    """Pull the assistant's text out of a decoded provider response.

    Raises on a shape it does not recognize rather than returning "" -- an empty
    string would flow onward and be recorded as a batch that legitimately found
    nothing, which is a silent data loss this baseline cannot afford. Raises
    TruncatedResponseError on a capped generation for the same reason: truncated
    JSON that happens to still parse would be scored as if the model had finished.
    RefusalError is checked first on the anthropic family, before the truncation
    check, so a refusal is never misreported as either truncation or a bare "empty
    text" ValueError.
    """
    if spec.family == "meta":
        text = payload.get("generation") or ""
        _check_not_truncated(
            spec,
            "stop_reason",
            payload.get("stop_reason"),
            text,
            payload.get("generation_token_count"),
        )
    elif spec.family == "anthropic":
        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        completion_tokens = (payload.get("usage") or {}).get("output_tokens")
        _check_not_refused(spec, payload, text, completion_tokens)
        _check_not_truncated(
            spec, "stop_reason", payload.get("stop_reason"), text, completion_tokens
        )
    else:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"no choices in response from {spec.key!r}")
        text = (choices[0].get("message") or {}).get("content") or ""
        _check_not_truncated(
            spec,
            "finish_reason",
            choices[0].get("finish_reason"),
            text,
            (payload.get("usage") or {}).get("completion_tokens"),
        )

    if not text.strip():
        raise ValueError(f"empty text in response from {spec.key!r}")
    return text


def extract_completion(spec: ModelSpec, payload: dict) -> Completion:
    """`extract_text` plus the stop reason and token count, for the run log.

    Both are recorded on every call, not only on failures: a run that finished well
    under its cap is evidence the cap never bound, which is what B3-D3's decision
    rule needs in order to say the differing per-model caps had no effect.
    """
    text = extract_text(spec, payload)

    if spec.family == "meta":
        return Completion(
            text=text,
            stop_reason=payload.get("stop_reason"),
            completion_tokens=payload.get("generation_token_count"),
        )
    if spec.family == "anthropic":
        usage = payload.get("usage") or {}
        return Completion(
            text=text,
            stop_reason=payload.get("stop_reason"),
            completion_tokens=usage.get("output_tokens"),
        )
    choices = payload.get("choices") or []
    usage = payload.get("usage") or {}
    return Completion(
        text=text,
        stop_reason=choices[0].get("finish_reason") if choices else None,
        completion_tokens=usage.get("completion_tokens"),
    )


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------

_TRANSIENT_MARKERS = (
    "throttl",
    "too many requests",
    "rate limit",
    "429",
    "503",
    "service unavailable",
    "timeout",
    "timed out",
    "connection reset",
)


def is_transient(exc: BaseException) -> bool:
    """Matched on the message rather than on SDK exception classes, so this module
    keeps importing without boto3/groq installed."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _with_retries(call, *, what: str):
    last: BaseException | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- re-raised below unless transient
            if not is_transient(exc):
                raise
            last = exc
            if attempt == MAX_RETRIES - 1:
                break
            delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
            print(f"  {what}: transient error ({exc}); retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"{what}: giving up after {MAX_RETRIES} attempts") from last


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def invoke_bedrock(spec: ModelSpec, prompt: str) -> Completion:
    """One Bedrock call. Credentials come from the standard AWS chain."""
    import boto3  # lazy -- see module docstring

    client = boto3.client("bedrock-runtime", region_name=spec.region or BEDROCK_REGION)
    body = json.dumps(build_request_body(spec, prompt))

    def call() -> Completion:
        response = client.invoke_model(modelId=spec.model_id, body=body)
        payload = json.loads(response["body"].read())
        return extract_completion(spec, payload)

    return _with_retries(call, what=f"bedrock/{spec.key}")


def invoke_groq(spec: ModelSpec, prompt: str) -> Completion:
    """One Groq call (B3-D1c, retained under B3-D6). GROQ_API_KEY is read from .env,
    matching the GEMINI_API_KEY pattern in scripts/generate_synthetic_data.py."""
    from groq import Groq  # lazy -- see module docstring

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (add it to .env)")

    client = Groq(api_key=api_key)
    kwargs = build_request_body(spec, prompt)

    def call() -> Completion:
        # The SDK hands back objects, not dicts, so this reads the same three fields
        # extract_completion() reads rather than routing through it.
        completion = client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        _check_not_truncated(
            spec,
            "finish_reason",
            choice.finish_reason,
            text,
            getattr(usage, "completion_tokens", None),
        )
        if not text.strip():
            raise ValueError(f"empty text in response from {spec.key!r}")
        return Completion(
            text=text,
            stop_reason=choice.finish_reason,
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    return _with_retries(call, what=f"groq/{spec.key}")


def invoke(spec: ModelSpec, prompt: str) -> Completion:
    """The only backend entry point single_shot.py uses."""
    if spec.backend == "bedrock":
        return invoke_bedrock(spec, prompt)
    if spec.backend == "groq":
        return invoke_groq(spec, prompt)
    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.key!r}")
