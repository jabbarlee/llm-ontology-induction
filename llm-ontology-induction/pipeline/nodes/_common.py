"""
Shared helpers used by every stage node -- not a stage itself.

The per-call audit log (`raw_records`) is written incrementally, one line
appended per call, the moment that call resolves -- rather than accumulated
in `P1State` and flushed once at the end. Two reasons, both load-bearing:

  1. **A node must not mutate `P1State` in place.** LangGraph nodes return
     the keys they own; the runtime merges that into shared state. A node
     that also reaches in and appends to a list already in state fights that
     model and reintroduces exactly the "mutated in a closure" bug class
     `P1State`'s own docstring says every node avoids.
  2. **Every call already paid for must survive a mid-pipeline crash.**
     Stage 2/4/5 each make exactly one call with nothing to fall back on if
     it fails (P1-D1); if that call's response comes back and the *next*
     line of code then raises for an unrelated reason, the paid-for response
     must already be on disk, not sitting in a Python object that dies with
     the process. Appending immediately is strictly safer than batching.
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.shared import model_clients as mc


def append_raw_record(log_path: str | Path, record: dict) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def normalize_name(name: str) -> str:
    """Case/whitespace-folding key used everywhere a class or relation
    endpoint name needs a stable lookup key across stages (`class_name_map`
    in Stage 2, the remap-dedup key in Stage 4). Deliberately weaker than
    eval.matching.normalize, which is gold-vocabulary-aware and has no
    business being imported into pipeline code that must never see gold
    terms (Critical Rule 1) -- this only needs to tolerate the same literal
    casing/spacing variance baselines.b3_single_shot.single_shot's own
    `_dedup_key` already tolerates elsewhere in this codebase."""
    return " ".join(name.strip().lower().split())


def invoke_or_abort(spec, prompt: str, stage: str, log_path: str | Path, record: dict) -> mc.Completion:
    """Shared by every stage that makes exactly one call with nothing to
    fall back on if it fails (Stages 2, 4, 5 -- everything except the 192
    independent Stage 1 calls, which skip-and-continue instead).

    On a bad response: fills `record` with stop_reason/response/
    completion_tokens/error, appends it to the log immediately (the paid-for
    response must survive even though the pipeline is about to die), and
    raises. On a transport failure (no response to record at all): logs the
    attempt with `stop_reason: "transport_error"` and re-raises unchanged. On
    success: fills the same three fields and returns the Completion, but
    does **not** append the record yet -- the caller still has parsing and
    its own stage-specific stats (parsed_classes, merges_applied, ...) to add
    before the record is written once, complete, rather than twice.
    """
    try:
        completion = mc.invoke(spec, prompt)
    except mc.ModelResponseError as exc:
        label = "refused" if isinstance(exc, mc.RefusalError) else "truncated"
        record["response"] = exc.text
        record["stop_reason"] = label
        record["completion_tokens"] = exc.completion_tokens
        record["error"] = str(exc)
        append_raw_record(log_path, record)
        raise RuntimeError(f"Stage {stage!r} {label} -- aborting, nothing to fall back on") from exc
    except Exception as exc:
        # Anything that is not the model answering badly: a transport failure,
        # a timeout, an auth or throttling error from the SDK. Before this
        # branch existed these propagated with nothing appended, so a run
        # killed here left a log holding only Stage 1's records -- reading as
        # if the pipeline had quietly stopped after extraction rather than
        # died in a specific call. Real full-corpus runs were misdiagnosed
        # that way (`results/findings/P1-FINDINGS.md`), so the attempt is now
        # recorded before the exception continues on its way.
        #
        # A bare `raise`, not a re-wrap: unlike the ModelResponseError branch
        # above, there is no P1-D1 abort decision to express here -- the
        # original exception and its traceback are the most useful thing the
        # caller can be handed.
        record["stop_reason"] = "transport_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        append_raw_record(log_path, record)
        raise

    record["response"] = completion.text
    record["stop_reason"] = completion.stop_reason
    record["completion_tokens"] = completion.completion_tokens
    return completion
