"""
B3 baseline -- model backends (baselines/DECISIONS.md B3-D1, B3-D1a, B3-D1c, B3-D3).

Five frozen models behind one `invoke()` entry point, so baselines/single_shot.py
never branches on which model is running -- that branching is exactly what would let
the five conditions drift apart, and B3's whole premise is that the task shape is
identical for all of them.

Two backends:
  - AWS Bedrock for the four cloud models (Fable 5, Haiku 4.5, Sol, Luna)
  - Groq's free tier for the open-weight model (B3-D1c: on-device Ollama was tested
    and proved infeasible on 8GB unified memory)

Sampling settings are frozen module constants (B3-D3), never per-call arguments --
a per-call temperature is a knob, and a knob on a frozen baseline eventually gets
turned.

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
MAX_OUTPUT_TOKENS = 8192

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; doubled per attempt with jitter

BEDROCK_REGION = os.environ.get("AWS_REGION") or "us-east-1"


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
    family: str | None = None  # request/response shape on Bedrock
    reasoning_effort: str | None = None  # Critical Rule 3 -- Luna only
    supports_temperature: bool = True


# TODO: confirm every exact model ID string below from the Bedrock console before
# the first run. These are placeholders, deliberately not plausible-looking guesses:
# a wrong-but-believable ID would fail at call time, while a wrong ID that happens to
# resolve would silently score the wrong model.
MODELS: dict[str, ModelSpec] = {
    "fable5": ModelSpec(
        key="fable5",
        model_id="anthropic.claude-fable-5",
        backend="bedrock",
        tier="frontier",
        family="anthropic",
    ),
    "haiku45": ModelSpec(
        key="haiku45",
        model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
        backend="bedrock",
        tier="budget",
        family="anthropic",
    ),
    "sol": ModelSpec(
        key="sol",
        model_id="openai.gpt-5.6-sol",
        backend="bedrock",
        tier="frontier",
        family="openai",
    ),
    "luna": ModelSpec(
        key="luna",
        model_id="openai.gpt-5.6-luna",
        backend="bedrock",
        tier="budget",
        family="openai",
        # Critical Rule 3 / B3-D1a: capability-matched to Haiku 4.5 at low effort.
        # Any other value makes this condition a different experiment.
        reasoning_effort="low",
    ),
    "llama318b": ModelSpec(
        key="llama318b",
        # Confirmed against Groq's catalog directly (not a Bedrock-console lookup,
        # so it carries no TODO): swapped in for qwen3:8b, which Groq does not serve
        # at that weight class (see B3-D1c).
        model_id="llama-3.1-8b-instant",
        backend="groq",
        tier="open-weight",
    ),
}


# ---------------------------------------------------------------------------
# Request bodies -- pure, no network (so the tests can assert on them)
# ---------------------------------------------------------------------------

def build_request_body(spec: ModelSpec, prompt: str) -> dict:
    """Provider payload for `spec` carrying `prompt` as the sole user turn.

    Split out from the invocation functions precisely so Critical Rule 3 is
    testable without a network call: the assertion that Luna always carries
    reasoning_effort="low" runs against this dict.

    `supports_temperature` exists because reasoning-tuned models have historically
    rejected any non-default temperature. If a model refuses the frozen value, flip
    that flag on its spec rather than special-casing the sampling settings -- the
    omission is then visible in one place and recordable in DECISIONS.md.
    """
    if spec.backend == "groq":
        body = {
            "model": spec.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_OUTPUT_TOKENS,
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
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        if spec.supports_temperature:
            body["temperature"] = TEMPERATURE
            body["top_p"] = TOP_P
        return body

    if spec.backend == "bedrock" and spec.family == "openai":
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
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

    raise ValueError(f"no request body defined for {spec.key!r} ({spec.backend})")


def extract_text(spec: ModelSpec, payload: dict) -> str:
    """Pull the assistant's text out of a decoded provider response.

    Raises on a shape it does not recognize rather than returning "" -- an empty
    string would flow onward and be recorded as a batch that legitimately found
    nothing, which is a silent data loss this baseline cannot afford.
    """
    if spec.family == "anthropic":
        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
    else:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"no choices in response from {spec.key!r}")
        text = (choices[0].get("message") or {}).get("content") or ""

    if not text.strip():
        raise ValueError(f"empty text in response from {spec.key!r}")
    return text


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

def invoke_bedrock(spec: ModelSpec, prompt: str) -> str:
    """One Bedrock call. Credentials come from the standard AWS chain."""
    import boto3  # lazy -- see module docstring

    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    body = json.dumps(build_request_body(spec, prompt))

    def call() -> str:
        response = client.invoke_model(modelId=spec.model_id, body=body)
        payload = json.loads(response["body"].read())
        return extract_text(spec, payload)

    return _with_retries(call, what=f"bedrock/{spec.key}")


def invoke_groq(spec: ModelSpec, prompt: str) -> str:
    """One Groq call (B3-D1c). GROQ_API_KEY is read from .env, matching the
    GEMINI_API_KEY pattern in scripts/generate_synthetic_data.py."""
    from groq import Groq  # lazy -- see module docstring

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (add it to .env)")

    client = Groq(api_key=api_key)
    kwargs = build_request_body(spec, prompt)

    def call() -> str:
        completion = client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""
        if not text.strip():
            raise ValueError(f"empty text in response from {spec.key!r}")
        return text

    return _with_retries(call, what=f"groq/{spec.key}")


def invoke(spec: ModelSpec, prompt: str) -> str:
    """The only backend entry point single_shot.py uses."""
    if spec.backend == "bedrock":
        return invoke_bedrock(spec, prompt)
    if spec.backend == "groq":
        return invoke_groq(spec, prompt)
    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.key!r}")
