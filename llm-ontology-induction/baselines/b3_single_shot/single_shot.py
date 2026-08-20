"""
B3 baseline -- single-shot LLM schema induction (baselines/DECISIONS.md, B3-D1..D5).

Sends the entire corpus to a model in **one call** under one frozen prompt, and
emits the resulting schema as JSON matching the contract in
eval/schema_ir.py::parse_induced_schema (eval/PLAN.md §2).

As of the 2026-08-19 rework, this is the only shape B3 has: two conditions
(haiku45, opus5), one whole-corpus call each, mixed transport -- Haiku 4.5 runs
on AWS Bedrock, Opus 5 runs on the direct Anthropic API (see
shared/model_clients.py's own docstring for the split). The legacy batched call
shape and the batched-vs-whole-corpus comparison (B3-D6) are retired, and so is
the third registry condition (opus48) briefly wired up in an earlier revision of
this rework. None of this is deleted from history: the prior runs, their
findings, and the decisions that governed them stay in git and in
DECISIONS.md/B3-FINDINGS.md, marked superseded rather than erased. See
DECISIONS.md B3-D1 (revised) for why.

One rule here is still load-bearing for the paper:

  * Nothing here cleans a model's output (Critical Rule 5). Names are emitted
    with the casing, spacing and pluralization the model returned. The harness
    normalizes at score time; the producer never pre-cleans.

HARD RULE (Critical Rule 1): zero gold-schema vocabulary in this module or in
the prompt. Enforced by
baselines/tests/test_single_shot.py::test_no_domain_vocabulary_leakage.

Usage:
    python -m baselines.b3_single_shot.single_shot --model haiku45 --dry-run
    python -m baselines.b3_single_shot.single_shot --model opus5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from baselines.shared import model_clients as mc
from baselines.shared.model_clients import MODELS, ModelSpec

CONDITION = "B3"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_ROOT = _REPO_ROOT / "data" / "documents"
_SUBDIRS = ("csv_exports", "lease_texts", "notes", "messages")
_READABLE_SUFFIXES = (".csv", ".txt")
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "raw"

_MODULE_ROOT = _REPO_ROOT / "baselines" / "b3_single_shot"
PROMPT_PATH = _MODULE_ROOT / "prompts" / "b3_extraction_prompt.md"
DOCUMENTS_PLACEHOLDER = "{{DOCUMENTS}}"


class ResponseParseError(ValueError):
    """No schema object could be recovered from a model response."""


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_documents(
    root: Path = _CORPUS_ROOT, limit: int | None = None
) -> list[tuple[str, str]]:
    """Every document as (repo-relative path, raw text), interleaved round-robin
    across subdirectories, in a deterministic order.

    Deliberately *not* B1's load_corpus(): that one sentence-splits prose and
    flattens CSV rows into pseudo-sentences, which is preprocessing B1's
    statistics need. B3's premise is that the model reads the mess exactly as it
    lies, so files are handed over verbatim.

    Round-robin (one document per subdirectory per round, subdirectories in
    _SUBDIRS order, filenames sorted within each) rather than draining one
    subdirectory before the next -- this is inherited from B3-D2's original
    Groq-era fix and kept because it still spreads the corpus's very uneven
    per-file density evenly through the single rendered prompt, which is a
    property worth having independent of why it was first adopted.

    Reused by P1's per-document extraction stage (baselines/p1_pipeline/), so
    both baselines see the corpus in the identical order.
    """
    per_subdir: list[list[tuple[str, str]]] = []
    for subdir in _SUBDIRS:
        directory = root / subdir
        if not directory.is_dir():
            per_subdir.append([])
            continue
        paths = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in _READABLE_SUFFIXES
        )
        # Filter before capping: `--limit 2` means two documents, not two
        # directory entries of which some unreadable stray may be one.
        capped = paths[:limit] if limit else paths
        per_subdir.append(
            [(f"{subdir}/{path.name}", path.read_text(encoding="utf-8")) for path in capped]
        )

    documents: list[tuple[str, str]] = []
    rounds = max((len(docs) for docs in per_subdir), default=0)
    for round_index in range(rounds):
        for docs in per_subdir:
            if round_index < len(docs):
                documents.append(docs[round_index])
    return documents


# ---------------------------------------------------------------------------
# Prompt rendering (Critical Rule 2)
# ---------------------------------------------------------------------------

def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    template = path.read_text(encoding="utf-8")
    if DOCUMENTS_PLACEHOLDER not in template:
        raise ValueError(f"{path} has no {DOCUMENTS_PLACEHOLDER} placeholder")
    return template


def format_documents(documents: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"--- document: {source} ---\n{text.strip()}" for source, text in documents
    )


def render_prompt(template: str, documents: list[tuple[str, str]]) -> str:
    """Substitute one call's documents into the frozen template.

    str.replace, not str.format: the template contains a JSON output example,
    and str.format would try to read its braces as fields. Interpolation
    touches the documents block and nothing else -- the instruction text is
    byte-identical whether `documents` holds all 192 files (B3) or exactly one
    (P1's per-document extraction stage, which reuses this function).
    """
    return template.replace(DOCUMENTS_PLACEHOLDER, format_documents(documents))


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's <think> block, if present.

    Neither frozen condition is expected to emit these -- Opus's
    thinking arrives as separate `thinking` content blocks the SDK already
    splits out (extract_from_anthropic_message in shared/model_clients.py only assembles
    `text`-type blocks), not inline `<think>` tags in the text itself. Kept
    defensive rather than removed: a no-op strip on plain text costs nothing,
    and this has been a stable guard since B3's very first version.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think" not in cleaned.lower():
        parts = _THINK_CLOSE.split(cleaned)
        cleaned = parts[-1]
    return cleaned


def strip_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged.

    Models wrap JSON in ```json fences despite being asked not to; that is a
    formatting habit, not missing data, so it is unwrapped rather than treated
    as a failure.
    """
    match = _FENCE.search(text)
    return match.group(1) if match else text


def _balanced_object(text: str, start: int) -> str | None:
    """The substring from `start` through its matching close brace, string-aware
    so a brace inside a JSON string value cannot end the scan early."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_response(text: str) -> dict:
    """Recover the schema object from a raw model response.

    Tolerant of the three deviations models actually produce -- a reasoning
    block, markdown fences, and commentary trailing the JSON -- because
    rejecting a response over formatting would quietly shrink the corpus this
    condition saw. Tolerant is not lenient: a response that yields no schema
    object raises ResponseParseError rather than returning {}, which would be
    recorded downstream as "genuinely found nothing" -- the one failure mode
    that would corrupt a reported number without leaving a trace.

    Shared with P1's per-document extraction stage, which parses the identical
    output shape from a single-document call.
    """
    candidate_text = strip_fences(strip_reasoning(text)).strip()

    attempts: list[str] = [candidate_text]
    for index, ch in enumerate(candidate_text):
        if ch == "{":
            found = _balanced_object(candidate_text, index)
            if found:
                attempts.append(found)

    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("classes" in parsed or "relations" in parsed):
            return parsed

    raise ResponseParseError(
        "no JSON object with classes/relations in response: "
        f"{text[:400]!r}{'...' if len(text) > 400 else ''}"
    )


# ---------------------------------------------------------------------------
# Output cleanup (B3-D4, revised -- see DECISIONS.md)
# ---------------------------------------------------------------------------

def _dedup_key(value: str) -> str:
    """Case- and whitespace-normalized key, used only to spot literal
    duplicates the model repeated within its own single response -- never to
    merge distinct wordings across sources, since a single whole-corpus call
    has no "across sources" left to merge (B3-D4's cross-batch consolidation is
    retired along with batching; see DECISIONS.md).

    Deliberately NOT eval.matching.normalize(), for the same reason as before:
    that function additionally splits camelCase, strips punctuation and
    singularizes -- lending it to the producer would let B3 merge wordings a
    naive method could not, and make B3's output partly a function of the
    harness that scores it.
    """
    return " ".join(value.split()).casefold()


def _usable(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def clean_schema(parsed: dict) -> dict:
    """Drop malformed elements from one parsed response; do not repair them
    (repairing would be the producer doing quality work this baseline is
    supposed to be measured without), and collapse only literal case/whitespace
    duplicates the model emitted within its own output.

    The first surface form seen for a repeated attribute is the one kept,
    unmodified (Critical Rule 5).
    """
    classes: dict[str, dict] = {}
    seen_attributes: dict[str, set[str]] = {}

    for raw in parsed.get("classes") or []:
        if not isinstance(raw, dict) or not _usable(raw.get("name")):
            continue
        class_name = raw["name"]
        key = _dedup_key(class_name)
        entry = classes.get(key)
        if entry is None:
            # Critical Rule 6: parent stays null unless the model volunteered one.
            parent = raw.get("parent")
            entry = {
                "name": class_name,
                "parent": parent if _usable(parent) else None,
                "attributes": [],
            }
            classes[key] = entry
            seen_attributes[key] = set()

        for attribute in raw.get("attributes") or []:
            if not _usable(attribute):
                continue
            attribute_key = _dedup_key(attribute)
            if attribute_key in seen_attributes[key]:
                continue
            seen_attributes[key].add(attribute_key)
            entry["attributes"].append(attribute)

    relations: dict[tuple[str, str, str], dict] = {}
    for raw in parsed.get("relations") or []:
        if not isinstance(raw, dict):
            continue
        source, label, target = raw.get("source"), raw.get("label"), raw.get("target")
        if not all(_usable(v) for v in (source, label, target)):
            continue
        triple = (_dedup_key(source), _dedup_key(label), _dedup_key(target))
        relations.setdefault(triple, {"source": source, "label": label, "target": target})

    return {"classes": list(classes.values()), "relations": list(relations.values())}


# ---------------------------------------------------------------------------
# Output assembly (B3-D5)
# ---------------------------------------------------------------------------

def build_output(
    cleaned: dict,
    sources: list[str],
    run_id: str,
    spec: ModelSpec,
    stop_reason: str | None = None,
    completion_tokens: int | None = None,
) -> dict:
    return {
        "classes": cleaned["classes"],
        "relations": cleaned["relations"],
        "metadata": {
            "condition": CONDITION,
            "model": spec.model_id,
            "run_id": run_id,
            "source_documents": sources,
            "stop_reason": stop_reason,
            "completion_tokens": completion_tokens,
        },
    }


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_whole_corpus(
    spec: ModelSpec, documents: list[tuple[str, str]], template: str, raw_record: dict
) -> dict:
    """Make the one call B3 now consists of. Fills `raw_record` in place so the
    caller still has it if this raises.

    Never skips a bad response the way the old batched path could afford to --
    with a single call there is nothing else to fall back on. A
    ResponseParseError, TruncatedResponseError, or RefusalError all propagate;
    `raw_record` is written to disk by the caller regardless.
    """
    prompt = render_prompt(template, documents)
    sources = [source for source, _text in documents]
    print(f"  whole corpus ({len(documents)} documents, {len(prompt)} chars)...")

    raw_record["source_documents"] = sources
    raw_record["prompt_chars"] = len(prompt)

    try:
        completion = mc.invoke(spec, prompt)
    except mc.ModelResponseError as exc:
        label = "refused" if isinstance(exc, mc.RefusalError) else "truncated"
        raw_record["response"] = exc.text
        raw_record["stop_reason"] = label
        raw_record["completion_tokens"] = exc.completion_tokens
        raw_record["error"] = str(exc)
        raise

    raw_record["response"] = completion.text
    raw_record["stop_reason"] = completion.stop_reason
    raw_record["completion_tokens"] = completion.completion_tokens
    print(
        f"    stop_reason={completion.stop_reason!r} "
        f"completion_tokens={completion.completion_tokens} "
        f"(cap {spec.max_output_tokens})"
    )

    schema = parse_response(completion.text)  # raises ResponseParseError, propagates
    raw_record["parsed_classes"] = len(schema.get("classes") or [])
    raw_record["parsed_relations"] = len(schema.get("relations") or [])
    return schema


def _dry_run(spec: ModelSpec, documents: list[tuple[str, str]], template: str) -> None:
    """Render the prompt and print a sample. Makes no API call, spends nothing."""
    prompt = render_prompt(template, documents)
    print(f"DRY RUN -- no API calls. model={spec.key} ({spec.model_id})")
    print(f"output cap: {spec.max_output_tokens} tokens")
    print(
        f"call: {len(documents)} documents, {len(prompt)} chars, "
        # ~3.5 chars/token, the ratio calibrated in earlier B3-D2 work against a
        # real provider token count. Labelled as an estimate deliberately.
        f"~{round(len(prompt) / 3.5)} est input tokens\n"
    )
    print(f"--- rendered prompt ({len(prompt)} chars) ---\n")
    if len(prompt) > 4000:
        print(prompt[:2000])
        print(f"\n[... {len(prompt) - 4000} chars elided ...]\n")
        print(prompt[-2000:])
    else:
        print(prompt)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the B3 single-shot LLM baseline.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--corpus", type=Path, default=_CORPUS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap files per subdirectory (smoke test only -- not for a reported run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the prompt and print it; make no API call and spend nothing.",
    )
    args = parser.parse_args(argv)

    spec = MODELS[args.model]
    template = load_prompt_template()
    documents = load_documents(args.corpus, limit=args.limit)
    if not documents:
        raise SystemExit(f"no documents under {args.corpus}")
    print(f"corpus: {len(documents)} documents -> 1 whole-corpus call")

    if args.dry_run:
        _dry_run(spec, documents, template)
        return

    run_id = _run_id()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"{run_id}_b3_{spec.key}_call.jsonl"
    raw_record: dict = {}

    try:
        schema = run_whole_corpus(spec, documents, template, raw_record)
    finally:
        # Written even when the call dies, so a truncated or refused response
        # is inspectable instead of lost with the process. No schema JSON is
        # written on that path -- a failed run must not leave behind a file
        # that looks like a result.
        raw_path.write_text(json.dumps(raw_record) + "\n", encoding="utf-8")
        print(f"wrote {os.path.relpath(raw_path, _REPO_ROOT)}")

    cleaned = clean_schema(schema)
    sources = [source for source, _text in documents]
    output = build_output(
        cleaned,
        sources,
        run_id,
        spec,
        stop_reason=raw_record.get("stop_reason"),
        completion_tokens=raw_record.get("completion_tokens"),
    )

    out_path = args.out_dir / f"{run_id}_b3_{spec.key}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    n_attributes = sum(len(c["attributes"]) for c in output["classes"])
    print(
        f"induced: {len(output['classes'])} classes, {n_attributes} attributes, "
        f"{len(output['relations'])} relations"
    )
    print(f"wrote {os.path.relpath(out_path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
