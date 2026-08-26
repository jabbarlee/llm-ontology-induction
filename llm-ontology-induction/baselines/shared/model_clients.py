"""
Model backends for both LLM-driven baselines (B3-D1, B3-D3; P1-D1, P1-D2).

Shared between baselines/b3_single_shot/ and baselines/p1_pipeline/ because both
call the same two frozen conditions the same way -- the only thing that differs
between B3 and P1 is *how many calls* and *what each call is asked to do*
(single_shot.py vs pipeline.py), never how a call reaches a given model.

**Mixed transport, deliberately, as of the 2026-08-19 revision:** Haiku 4.5 runs
on AWS Bedrock; Opus 5 runs on the direct Anthropic API. This is not a return to
the old five-model Bedrock grid -- it is two conditions, one transport each,
chosen per model rather than uniformly. See DECISIONS.md B3-D1 (revised again)
for why: Haiku's Bedrock access and credentials were already working and paid
for; Opus 5 is not available on this account's Bedrock at all, and the direct
API is where it actually runs.

**A third condition, Llama 3.1 8B on Bedrock, restored 2026-08-21** (see
DECISIONS.md B3-D1, revised a third time): the same open-weight condition run
in the original five-model grid, on the same transport, using the native
Meta request/response shape (`prompt`/`max_gen_len` -> `generation`/
`stop_reason`) -- a third backend, `"bedrock_meta"`, distinct from `"bedrock"`
(Anthropic-on-Bedrock) because the two shapes share nothing but the
`invoke_model` call itself.

HARD RULE (Critical Rule 1): zero gold-schema vocabulary in this module. Enforced
by baselines/tests/test_single_shot.py::test_no_domain_vocabulary_leakage and
baselines/tests/test_p1_pipeline.py's equivalent.

boto3 and anthropic are both imported lazily inside their respective invoke
functions, so the pure functions here (and therefore the whole test suite)
import cleanly on a machine where neither SDK is installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# --- Frozen sampling settings (B3-D3 / P1-D2) ------------------------------
# Applied only where the model accepts them -- see ModelSpec.supports_temperature
# / supports_top_p below. Opus 5 rejects both parameters outright (a documented
# Anthropic API constraint on Opus 4.7+), so neither constant is ever sent to it.
TEMPERATURE = 0.0
TOP_P = 1.0

MAX_RETRIES = 5

BEDROCK_REGION = "us-east-1"


class ModelResponseError(RuntimeError):
    """Base for a response that must never be scored, even though the call
    completed without a transport-level error.

    Both subclasses are fatal and never retried by hand: a truncated or refused
    response at TEMPERATURE = 0.0 (or, for Opus 5's non-sampling behavior)
    reproduces identically on retry, so retrying only spends money to fail the
    same way twice. Both carry whatever text the model did return -- a paid-for
    call's partial output is not discarded just because the call as a whole
    failed. Identical in meaning on both backends; only how they get raised
    differs (Bedrock's raw JSON dict vs. the Anthropic SDK's typed Message).
    """

    def __init__(self, message: str, text: str = "", completion_tokens: int | None = None):
        super().__init__(message)
        self.text = text
        self.completion_tokens = completion_tokens


class TruncatedResponseError(ModelResponseError):
    """The model stopped at its output cap (`stop_reason == "max_tokens"`).

    On Opus 5 (thinking enabled) this cap is shared between thinking and the
    visible response -- see ModelSpec.thinking's docstring.
    """


class RefusalError(ModelResponseError):
    """The model declined on content-policy grounds (`stop_reason == "refusal"`).
    `stop_details` (`category`, `explanation`) is recorded on the raised error's
    message when the backend's response includes it."""


@dataclass(frozen=True)
class Completion:
    """One model response plus what's needed to trust it without re-parsing the
    raw payload: `stop_reason` and `completion_tokens` are what tell a truncated
    or refused schema apart from a genuinely finished one."""

    text: str
    stop_reason: str | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    """One frozen model condition.

    `model_id` is the exact provider identifier for whichever `backend` this
    spec uses, and is what lands in `metadata.model` of every emitted schema --
    results must name the artifact that produced them, not a friendly alias
    that could later be repointed.
    """

    key: str  # CLI selector
    model_id: str  # exact provider model ID for this backend -> metadata.model
    tier: str  # "frontier" | "budget" | "open-weight"
    backend: str  # "bedrock" | "bedrock_meta" | "anthropic_api"
    max_output_tokens: int  # B3-D3 / P1-D2, per model
    # Independent flags, not one -- a model that rejects only one of the pair
    # would otherwise force dropping both to fix the other.
    supports_temperature: bool = True
    supports_top_p: bool = True
    # Whether to request adaptive thinking (`thinking: {"type": "adaptive"}`) --
    # Anthropic-API backend only. On a thinking-enabled call, `max_output_tokens`
    # is a cap on thinking PLUS the visible response combined, not the response
    # alone -- see B3-D3 for the reasoning behind the chosen cap.
    thinking: bool = False
    # output_config.effort ("low".."max"); only meaningful when thinking=True.
    effort: str | None = None
    # Bedrock only. None means BEDROCK_REGION.
    region: str | None = None


MODELS: dict[str, ModelSpec] = {
    "haiku45": ModelSpec(
        key="haiku45",
        # The geo inference ID, not the bare `anthropic.claude-haiku-4-5-...`.
        # Confirmed at a real call: the bare ID raises "on-demand throughput
        # isn't supported" on this account's Bedrock. The model card explains
        # why -- its bedrock-runtime row lists the bare ID's In-Region endpoint
        # as "N/A"; only the Geo/Global inference ID is invokable on demand.
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tier="budget",
        backend="bedrock",
        max_output_tokens=16000,
        supports_temperature=True,
        # Confirmed at a real call: a body carrying both temperature and top_p
        # raised "`temperature` and `top_p` cannot both be specified for this
        # model. Please use only one." temperature=0.0 is kept (the
        # load-bearing setting for reproducibility); top_p is dropped -- B3-D3
        # already documents top_p=1.0 as "neutral, does nothing at temperature
        # 0," so nothing is lost by omitting it.
        supports_top_p=False,
    ),
    "opus5": ModelSpec(
        key="opus5",
        model_id="claude-opus-5",
        tier="frontier",
        backend="anthropic_api",
        max_output_tokens=32000,
        supports_temperature=False,
        supports_top_p=False,
        thinking=True,
        effort="high",
    ),
    "llama318b": ModelSpec(
        key="llama318b",
        # The *geo* inference ID, not the bare `meta.` one. Read off the model card's
        # own regional table: us-east-1 and us-east-2 are In-Region NO / Geo YES, and
        # only us-west-2 serves the bare ID on demand. The geo ID is callable from all
        # three, which makes the region question moot for this condition instead of
        # pinning it to one region.
        model_id="us.meta.llama3-1-8b-instruct-v1:0",
        tier="open-weight",
        backend="bedrock_meta",
        # 4096, not Haiku's 16000. The model card states "Max output tokens: 4K" flat,
        # against a 128K context window -- a property of the model as served, not a
        # silent clamp. The original 2026-08-14 whole-corpus run against this exact
        # condition truncated at exactly this cap mid-repetition (B3-FINDINGS.md) --
        # a known, already-documented risk this condition carries back with it, not
        # a new one.
        max_output_tokens=4096,
        supports_temperature=True,
        supports_top_p=True,
    ),
}


# ---------------------------------------------------------------------------
# Request construction -- pure, no network (so tests can assert on it)
# ---------------------------------------------------------------------------

def _build_bedrock_body(spec: ModelSpec, prompt: str) -> dict:
    """The JSON body for `bedrock-runtime`'s `invoke_model`, Anthropic-on-
    Bedrock shape (`anthropic_version` + block-structured `messages`)."""
    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": spec.max_output_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    if spec.supports_temperature:
        body["temperature"] = TEMPERATURE
    if spec.supports_top_p:
        body["top_p"] = TOP_P
    return body


_META_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
_META_SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def _build_bedrock_meta_body(spec: ModelSpec, prompt: str) -> dict:
    """The JSON body for `bedrock-runtime`'s `invoke_model`, Meta's native
    shape (`prompt`/`max_gen_len` -> `generation`/`stop_reason`) -- nothing
    like Anthropic-on-Bedrock's shape above, which is why this is a separate
    backend rather than a branch inside `_build_bedrock_body`.

    AWS also publishes a contradictory sample on this model's own card that
    posts `messages`/`max_tokens` to the same endpoint, with no documented
    response shape. The native shape is used here because it is the one
    whose `stop_reason` semantics are documented, and the truncation guard
    depends on exactly that field.

    Concatenated, never str.format: the frozen prompt contains a JSON output
    example, and format() would read its braces as replacement fields -- the
    same reasoning single_shot.render_prompt already follows for the same
    reason.
    """
    body: dict = {
        "prompt": _META_PREFIX + prompt + _META_SUFFIX,
        "max_gen_len": spec.max_output_tokens,
    }
    if spec.supports_temperature:
        body["temperature"] = TEMPERATURE
    if spec.supports_top_p:
        body["top_p"] = TOP_P
    return body


def _build_anthropic_api_kwargs(spec: ModelSpec, prompt: str) -> dict:
    """The kwargs `client.messages.stream(**kwargs)` is called with, direct
    Anthropic API shape (plain-string `content`, `thinking`/`output_config`)."""
    kwargs: dict = {
        "model": spec.model_id,
        "max_tokens": spec.max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if spec.supports_temperature:
        kwargs["temperature"] = TEMPERATURE
    if spec.supports_top_p:
        kwargs["top_p"] = TOP_P
    if spec.thinking:
        # display="summarized" costs nothing extra (billing is identical under
        # every display setting) and gives the raw run log something readable
        # instead of an empty thinking field, which is the default.
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    if spec.effort:
        kwargs["output_config"] = {"effort": spec.effort}
    return kwargs


def build_request(spec: ModelSpec, prompt: str) -> dict:
    """Dispatches to the right shape for `spec.backend`. Pure function of
    `spec` and `prompt` alone -- no batching or call-shape argument exists to
    thread through, so the request for a given model is identical whether it
    is B3's one whole-corpus call or one of P1's per-document calls."""
    if spec.backend == "bedrock":
        return _build_bedrock_body(spec, prompt)
    if spec.backend == "bedrock_meta":
        return _build_bedrock_meta_body(spec, prompt)
    if spec.backend == "anthropic_api":
        return _build_anthropic_api_kwargs(spec, prompt)
    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.key!r}")


# ---------------------------------------------------------------------------
# Response reading
# ---------------------------------------------------------------------------

def _check_not_refused(spec: ModelSpec, stop_reason, text: str, completion_tokens, stop_details=None) -> None:
    if stop_reason != "refusal":
        return
    category = None
    explanation = None
    if stop_details is not None:
        if isinstance(stop_details, dict):
            category, explanation = stop_details.get("category"), stop_details.get("explanation")
        else:
            category = getattr(stop_details, "category", None)
            explanation = getattr(stop_details, "explanation", None)
    raise RefusalError(
        f"{spec.key!r} refused to respond (stop_reason='refusal'"
        + (f", category={category!r}" if category else "")
        + (f": {explanation}" if explanation else "")
        + "). Not retried: a content-policy refusal is deterministic and "
        "will not change on a retry.",
        text=text,
        completion_tokens=completion_tokens,
    )


def _check_not_truncated(spec: ModelSpec, stop_reason, text: str, completion_tokens, marker: str = "max_tokens") -> None:
    """`marker` is the provider-specific value of `stop_reason` that means
    "cut off at the cap" -- Anthropic (direct API and Bedrock alike) uses
    `"max_tokens"`; Meta's native Bedrock shape uses `"length"` instead. One
    shared function, not one per family, since the meaning and the message
    are identical -- only the string being compared against differs."""
    if stop_reason != marker:
        return
    raise TruncatedResponseError(
        f"{spec.key!r} stopped at its output cap of {spec.max_output_tokens} "
        f"tokens (stop_reason={marker!r}); the schema it returned is cut off "
        "and is not usable. Not retried: the same prompt returns the same "
        "cut-off answer under this project's frozen, deterministic settings.",
        text=text,
        completion_tokens=completion_tokens,
    )


def extract_from_bedrock_payload(spec: ModelSpec, payload: dict) -> Completion:
    """Pull the assistant's text out of a decoded Bedrock `invoke_model`
    response body (a plain dict -- `json.loads(response["body"].read())`)."""
    blocks = payload.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
    stop_reason = payload.get("stop_reason")
    completion_tokens = (payload.get("usage") or {}).get("output_tokens")

    _check_not_refused(spec, stop_reason, text, completion_tokens, payload.get("stop_details"))
    _check_not_truncated(spec, stop_reason, text, completion_tokens)

    if not text.strip():
        raise ValueError(f"empty text in response from {spec.key!r} (stop_reason={stop_reason!r})")
    return Completion(text=text, stop_reason=stop_reason, completion_tokens=completion_tokens)


def extract_from_bedrock_meta_payload(spec: ModelSpec, payload: dict) -> Completion:
    """Pull the assistant's text out of a decoded Meta-native Bedrock
    `invoke_model` response body. No refusal check here: Meta's native shape
    documents no `stop_reason` value analogous to Anthropic's
    `"refusal"` -- only truncation (`stop_reason == "length"`) is a
    documented, checkable failure mode for this family."""
    text = payload.get("generation") or ""
    stop_reason = payload.get("stop_reason")
    completion_tokens = payload.get("generation_token_count")

    _check_not_truncated(spec, stop_reason, text, completion_tokens, marker="length")

    if not text.strip():
        raise ValueError(f"empty text in response from {spec.key!r} (stop_reason={stop_reason!r})")
    return Completion(text=text, stop_reason=stop_reason, completion_tokens=completion_tokens)


def extract_from_anthropic_message(spec: ModelSpec, message) -> Completion:
    """Pull the assistant's text out of a Messages API response object.

    `message` is the SDK's `Message` (from `.get_final_message()` after a
    stream) -- a typed object, not a dict: `message.content` is a list of
    content blocks distinguished by `.type`, `message.stop_reason` a string,
    `message.usage.output_tokens` an int, `message.stop_details` (only
    populated on a refusal) an object with `.category`/`.explanation`.
    """
    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    completion_tokens = getattr(message.usage, "output_tokens", None)
    stop_reason = message.stop_reason

    _check_not_refused(spec, stop_reason, text, completion_tokens, getattr(message, "stop_details", None))
    _check_not_truncated(spec, stop_reason, text, completion_tokens)

    if not text.strip():
        raise ValueError(f"empty text in response from {spec.key!r} (stop_reason={stop_reason!r})")
    return Completion(text=text, stop_reason=stop_reason, completion_tokens=completion_tokens)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

BEDROCK_READ_TIMEOUT = 300  # seconds -- see this function's own docstring


def _bedrock_client(spec: ModelSpec):
    """Shared by both Bedrock-hosted backends (`bedrock`, `bedrock_meta`) --
    the client construction is identical; only the body shape and response
    parsing differ per family, in their own functions.

    `read_timeout=300`, not boto3's 60-second default. `invoke_model` is a
    single blocking call that returns only once the *entire* completion has
    been generated server-side -- unlike the direct Anthropic API path below,
    Bedrock's non-streaming `invoke_model` has no equivalent to
    `.messages.stream()` for a synchronous SDK call. A request allowing up to
    16,000 output tokens (haiku45) can legitimately take longer than 60
    seconds to generate, especially under Stage 2's large consolidation
    prompts (pipeline/nodes/consolidate_types.py) -- confirmed at a real
    call: a full-corpus Haiku run raised `ReadTimeoutError` at the default
    60s on a call that was very plausibly still generating, not stuck. 300s
    is comfortable headroom under every current condition's cap without
    being so long that a genuinely hung request blocks the run indefinitely.
    """
    import boto3  # lazy -- see module docstring
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=spec.region or BEDROCK_REGION,
        config=Config(
            read_timeout=BEDROCK_READ_TIMEOUT,
            # boto3's own "standard" retry mode: retries throttling/5xx/connection
            # errors (including a read timeout) with backoff. Deliberately not a
            # hand-rolled second retry layer on top of this -- see the
            # Anthropic-API path below, which relies on the SDK's equivalent for
            # the same reason. Retrying a read timeout at the *default* 60s
            # would just reproduce the same failure MAX_RETRIES times in a row --
            # raising read_timeout first is what makes the retry meaningful.
            retries={"max_attempts": MAX_RETRIES, "mode": "standard"},
        ),
    )


def invoke_bedrock(spec: ModelSpec, prompt: str) -> Completion:
    """One Bedrock call, Anthropic-on-Bedrock shape. Credentials come from
    the standard AWS chain (`AWS_PROFILE`/`AWS_REGION` in .env, or the
    environment)."""
    client = _bedrock_client(spec)
    body = json.dumps(_build_bedrock_body(spec, prompt))
    response = client.invoke_model(modelId=spec.model_id, body=body)
    payload = json.loads(response["body"].read())
    return extract_from_bedrock_payload(spec, payload)


def invoke_bedrock_meta(spec: ModelSpec, prompt: str) -> Completion:
    """One Bedrock call, Meta's native shape. Same client, same credential
    chain as `invoke_bedrock` -- only the body and the response differ."""
    client = _bedrock_client(spec)
    body = json.dumps(_build_bedrock_meta_body(spec, prompt))
    response = client.invoke_model(modelId=spec.model_id, body=body)
    payload = json.loads(response["body"].read())
    return extract_from_bedrock_meta_payload(spec, payload)


def invoke_anthropic_api(spec: ModelSpec, prompt: str) -> Completion:
    """One direct Anthropic API call, streamed.

    Streaming (`client.messages.stream(...).get_final_message()`) rather than
    a plain `.create()`: Opus 5's thinking can run long before any visible
    text appears, which is exactly the case streaming exists for -- the SDK's
    own non-streaming guard would refuse some of these calls outright.

    Retries are the SDK's own (`max_retries=MAX_RETRIES` on the client
    construction), not a hand-rolled wrapper.
    """
    import anthropic  # lazy -- see module docstring

    client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    kwargs = _build_anthropic_api_kwargs(spec, prompt)
    with client.messages.stream(**kwargs) as stream:
        message = stream.get_final_message()
    return extract_from_anthropic_message(spec, message)


def invoke(spec: ModelSpec, prompt: str) -> Completion:
    """The only entry point single_shot.py and pipeline.py use."""
    if spec.backend == "bedrock":
        return invoke_bedrock(spec, prompt)
    if spec.backend == "bedrock_meta":
        return invoke_bedrock_meta(spec, prompt)
    if spec.backend == "anthropic_api":
        return invoke_anthropic_api(spec, prompt)
    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.key!r}")
