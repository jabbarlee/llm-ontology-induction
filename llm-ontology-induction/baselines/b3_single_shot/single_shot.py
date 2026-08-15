"""
B3 baseline -- single-shot LLM schema induction (baselines/DECISIONS.md, B3-D1..D6).

Reads the corpus as whole documents, sends **all of them in one call** to one model
under one frozen prompt (B3-D1, B3-D2), and emits a single induced-schema JSON
matching the contract in eval/schema_ir.py::parse_induced_schema (eval/PLAN.md §2).

Whole-corpus is the current shape as of the 2026-08-14 rework. Batching still exists,
labelled legacy throughout, for exactly one reason: the 2026-08-12 Groq run was made
at `--batch-size 7` and must stay reproducible (B3-D6). That old shape is what the
merge step below was written for -- under whole-corpus there is one schema to merge
and B3-D4 becomes a no-op, which relaxes nothing about the rule.

Two rules here are load-bearing for the paper rather than for the code:

  * Consolidation is deliberately naive (B3-D4). "Owner" and "owner" merge; "Owner"
    and "Landlord" do not, even though a smarter method plainly could. Cross-wording
    resolution is P1's Stage 6 novelty -- if B3 does that job too, the B3-vs-P1
    comparison stops measuring anything.
  * Nothing here cleans a model's output (Critical Rule 5). Names are emitted with
    the casing, spacing and pluralization the model returned. The harness
    normalizes at score time; the producer never pre-cleans.

HARD RULE (Critical Rule 1): zero gold-schema vocabulary in this module or in the
prompt. Enforced by
baselines/tests/test_single_shot.py::test_no_domain_vocabulary_leakage.

Usage:
    python -m baselines.b3_single_shot.single_shot --model haiku45 --dry-run
    python -m baselines.b3_single_shot.single_shot --model llama318b_bedrock
    python -m baselines.b3_single_shot.single_shot --model llama318b_groq --batch-size 7
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from baselines.b3_single_shot import model_clients as mc
from baselines.b3_single_shot.model_clients import MODELS, ModelSpec

# --- Legacy run setting (B3-D2, superseded 2026-08-14) --------------------
# NOT the current shape. Whole-corpus is (see the module docstring). This value
# survives only so `--batch-size 7` reproduces the 2026-08-12 Groq run exactly,
# because that run is now the batched arm of a same-model comparison (B3-D6)
# rather than a discarded first attempt.
#
# Its history, since the number looks arbitrary otherwise: it was 10, cut to 7 on
# 2026-08-12 after batch 1 under the old sequential document order (10 of 12
# csv_exports/ files back to back) requested ~6,566 input tokens against Groq's
# 6,000 TPM cap on llama-3.1-8b-instant -- over budget before a single output
# token was reserved. Both that cap and this number are Groq free-tier artifacts,
# which is why neither constrains the Bedrock conditions.
LEGACY_BATCH_SIZE = 7

CONDITION = "B3"

# The two call shapes, as recorded in metadata.batching.
WHOLE = "whole"
BATCHED = "batched"

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
# Corpus loading (B3-D2)
# ---------------------------------------------------------------------------

def load_documents(
    root: Path = _CORPUS_ROOT, limit: int | None = None
) -> list[tuple[str, str]]:
    """Every document as (repo-relative path, raw text), interleaved round-robin
    across subdirectories, in a deterministic order.

    Deliberately *not* B1's load_corpus(): that one sentence-splits prose and
    flattens CSV rows into pseudo-sentences, which is preprocessing B1's statistics
    need. B3's premise is that the model reads the mess exactly as it lies, so files
    are handed over verbatim.

    Order is fixed and deterministic, but round-robin rather than one subdirectory
    fully drained before the next (revised B3-D2, 2026-08-12): csv_exports/ runs
    far denser per file than lease_texts/notes/messages, and draining it first
    packed an entire subdirectory's worth of dense CSV into the earliest batches --
    which is exactly what blew batch 1 through Groq's free-tier TPM cap. One
    document per subdirectory per round (subdirectories still visited in _SUBDIRS
    order; filenames still sort within each) spreads that density out instead.
    Batch membership is derived from this order, and it must stay identical across
    every model (Critical Rule 7).
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


def batch_documents(
    documents: list[tuple[str, str]], size: int = LEGACY_BATCH_SIZE
) -> list[list[tuple[str, str]]]:
    """LEGACY (B3-D2, superseded 2026-08-14). Split into fixed-size batches.

    Reachable only via an explicit `--batch-size`, and kept for one purpose:
    reproducing the 2026-08-12 Groq run, which B3-D6 retains as the batched arm of a
    same-model batched-vs-whole-corpus comparison. New runs go through the
    whole-corpus path and never call this.

    Still a function of the document order alone -- it takes no ModelSpec, so it is
    structurally incapable of batching one model differently from another (Critical
    Rule 7). That guarantee is load-bearing for the legacy arm too: the comparison
    only works if the batched shape is the same for whoever runs it.
    """
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    return [documents[i : i + size] for i in range(0, len(documents), size)]


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

    str.replace, not str.format: the template contains a JSON output example, and
    str.format would try to read its braces as fields. Interpolation touches the
    documents block and nothing else -- the instruction text is byte-identical for
    every call, every call shape and every model (Critical Rule 2).
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

    None of the six frozen conditions is currently a reasoning model that emits these
    (llama-3.1-8b-instant, the open-weight condition, does not), but this stays
    defensive rather than model-specific: Groq and Bedrock both host reasoning
    models, a future swap could reintroduce one, and a no-op strip on plain text
    costs nothing. Both shapes are handled: a matched pair, and a bare closing tag
    when the opening one is suppressed by the server, so the tail after the last
    </think> is taken in that case.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think" not in cleaned.lower():
        parts = _THINK_CLOSE.split(cleaned)
        cleaned = parts[-1]
    return cleaned


def strip_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged.

    Models wrap JSON in ```json fences despite being asked not to; that is a
    formatting habit, not missing data, so it is unwrapped rather than treated as a
    failure.
    """
    match = _FENCE.search(text)
    return match.group(1) if match else text


def _balanced_object(text: str, start: int) -> str | None:
    """The substring from `start` through its matching close brace, string-aware so
    a brace inside a JSON string value cannot end the scan early."""
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

    Tolerant of the three deviations models actually produce -- a reasoning block,
    markdown fences, and commentary trailing the JSON -- because rejecting a batch
    over formatting would quietly shrink the corpus this condition saw, and a
    smaller corpus is a different experiment.

    Tolerant is not lenient: a response that yields no schema object raises
    ResponseParseError. Returning {} instead would be recorded downstream as a batch
    that genuinely found nothing, which is the one failure mode that would corrupt a
    reported number without leaving a trace.
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
# Consolidation (B3-D4)
# ---------------------------------------------------------------------------

def _dedup_key(value: str) -> str:
    """Case- and whitespace-normalized key for exact-string deduplication.

    Deliberately NOT eval.matching.normalize. That function additionally splits
    camelCase, strips punctuation and singularizes -- it is the scorer's matching
    logic, and lending it to the producer would (a) let B3 merge wordings a naive
    method could not and (b) make B3's output partly a function of the harness it is
    scored by. Casing and stray whitespace are the only differences collapsed here;
    every semantic difference survives to be counted.
    """
    return " ".join(value.split()).casefold()


def _usable(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def merge_batches(batch_schemas: list[dict]) -> dict:
    """Naive union of per-batch schemas (B3-D4).

    The first surface form seen for a key is the one emitted, unmodified (Critical
    Rule 5). Elements the model returned malformed (missing name, non-string field)
    are dropped rather than repaired -- repairing them would be the producer doing
    quality work the baseline is supposed to be measured without.
    """
    classes: dict[str, dict] = {}
    seen_attributes: dict[str, set[str]] = {}
    relations: dict[tuple[str, str, str], dict] = {}

    for schema in batch_schemas:
        for raw in schema.get("classes") or []:
            if not isinstance(raw, dict) or not _usable(raw.get("name")):
                continue
            # `class_name`, not `name`: a bare `name` identifier collides with a
            # gold attribute and trips the leakage guard, which exempts contract
            # keys as string literals only (test_statistical.py::
            # _executable_vocabulary).
            class_name = raw["name"]
            key = _dedup_key(class_name)
            entry = classes.get(key)
            if entry is None:
                # Critical Rule 6: parent stays null unless a model volunteered one.
                entry = {"name": class_name, "parent": None, "attributes": []}
                classes[key] = entry
                seen_attributes[key] = set()

            parent = raw.get("parent")
            if entry["parent"] is None and _usable(parent):
                entry["parent"] = parent

            for attribute in raw.get("attributes") or []:
                if not _usable(attribute):
                    continue
                attribute_key = _dedup_key(attribute)
                if attribute_key in seen_attributes[key]:
                    continue
                seen_attributes[key].add(attribute_key)
                entry["attributes"].append(attribute)

        for raw in schema.get("relations") or []:
            if not isinstance(raw, dict):
                continue
            source, label, target = raw.get("source"), raw.get("label"), raw.get("target")
            if not all(_usable(v) for v in (source, label, target)):
                continue
            triple = (_dedup_key(source), _dedup_key(label), _dedup_key(target))
            relations.setdefault(
                triple, {"source": source, "label": label, "target": target}
            )

    return {"classes": list(classes.values()), "relations": list(relations.values())}


# ---------------------------------------------------------------------------
# Output assembly (B3-D5)
# ---------------------------------------------------------------------------

def build_output(
    merged: dict,
    sources: list[str],
    run_id: str,
    spec: ModelSpec,
    batching: str = WHOLE,
    batch_size: int | None = None,
    usage: list[dict] | None = None,
) -> dict:
    """Assemble the emitted schema (B3-D5).

    `batching`, `batch_size` and `usage` were added 2026-08-14. Two runs of the same
    model in the two call shapes are otherwise indistinguishable from their output
    alone, and B3-D6 exists precisely to compare them -- a result file that cannot
    say which shape produced it is not usable evidence. `usage` carries the stop
    reason and completion-token count per call, which is what B3-D3's decision rule
    needs to say whether any run reached its output cap.

    Additive and contract-safe: parse_induced_schema ignores metadata entirely and
    load_induced_metadata is a bare `data.get("metadata", {})`.
    """
    return {
        "classes": merged["classes"],
        "relations": merged["relations"],
        "metadata": {
            "condition": CONDITION,
            "model": spec.model_id,
            "run_id": run_id,
            "source_documents": sources,
            "batching": batching,
            "batch_size": batch_size,
            "usage": usage or [],
        },
    }


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_calls(
    spec: ModelSpec,
    calls: list[list[tuple[str, str]]],
    template: str,
    raw_records: list[dict],
) -> list[dict]:
    """Call `spec` once per element of `calls`. Returns the parsed schemas.

    Under the current whole-corpus shape `calls` holds exactly one element; under the
    legacy batched shape it holds one per batch.

    `raw_records` is filled in place, and is owned by the caller on purpose: when this
    raises, the caller still has every response recorded so far and can write the log
    before the run dies. A response that was paid for must reach disk whether or not
    it turned out to be usable.

    An unparseable response is recorded and skipped **only when there is more than one
    call**, so one bad batch cannot throw away 27 good ones already paid for. With a
    single call there is nothing left to salvage: skipping would write out an empty
    schema and report it as a completed run, which is the one failure mode that
    corrupts a reported number without leaving a trace. So it raises instead.
    """
    schemas: list[dict] = []
    single = len(calls) == 1

    for index, documents in enumerate(calls, start=1):
        prompt = render_prompt(template, documents)
        sources = [source for source, _text in documents]
        label = "whole corpus" if single else f"batch {index}/{len(calls)}"
        print(f"  {label} ({len(documents)} documents, {len(prompt)} chars)...")

        record: dict = {
            "batch": index,
            "source_documents": sources,
            "prompt_chars": len(prompt),
        }
        try:
            completion = mc.invoke(spec, prompt)
        except mc.ModelResponseError as exc:
            # Fatal either way -- neither a cut-off schema nor a refusal is ever
            # merged -- but whatever text came back is kept so the run log shows
            # what actually happened, and the two causes are recorded as what they
            # are rather than collapsed into one generic failure.
            label = "refused" if isinstance(exc, mc.RefusalError) else "truncated"
            record["response"] = exc.text
            record["stop_reason"] = label
            record["completion_tokens"] = exc.completion_tokens
            record["error"] = str(exc)
            raw_records.append(record)
            raise

        record["response"] = completion.text
        record["stop_reason"] = completion.stop_reason
        record["completion_tokens"] = completion.completion_tokens
        print(
            f"    stop_reason={completion.stop_reason!r} "
            f"completion_tokens={completion.completion_tokens} "
            f"(cap {spec.max_output_tokens})"
        )

        try:
            schema = parse_response(completion.text)
        except ResponseParseError as exc:
            record["parse_error"] = str(exc)
            raw_records.append(record)
            if single:
                raise
            print("    UNPARSEABLE -- skipped; see the raw log")
            continue

        schemas.append(schema)
        record["parsed_classes"] = len(schema.get("classes") or [])
        record["parsed_relations"] = len(schema.get("relations") or [])
        raw_records.append(record)

    return schemas


def _dry_run(spec: ModelSpec, calls: list[list[tuple[str, str]]], template: str) -> None:
    """Render every prompt and print a sample. Makes no API call, spends nothing."""
    print(f"DRY RUN -- no API calls. model={spec.key} ({spec.model_id})")
    print(f"output cap: {spec.max_output_tokens} tokens\n")
    for index, documents in enumerate(calls, start=1):
        prompt = render_prompt(template, documents)
        # ~3.5 chars/token, the ratio calibrated in B3-D2 against a real provider
        # token count. An estimate, and labelled as one -- this model family does not
        # support Bedrock's count-tokens API, so there is no exact figure to be had
        # without spending a call.
        print(
            f"call {index}: {len(documents)} documents, {len(prompt)} chars, "
            f"~{round(len(prompt) / 3.5)} est input tokens"
        )

    prompt = render_prompt(template, calls[0])
    print(f"\n--- rendered prompt, call 1 ({len(prompt)} chars) ---\n")
    if len(prompt) > 4000:
        # The whole-corpus prompt is ~152,000 chars; printing it whole buries the
        # numbers above, which are the point of a dry run.
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
        "--batch-size",
        type=int,
        default=None,
        help=(
            "LEGACY. Omit for the current whole-corpus shape (one call). Pass 7 to "
            "reproduce the 2026-08-12 Groq run, which B3-D6 keeps as the batched arm "
            "of a same-model comparison. Any other value is neither shape."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap files per subdirectory (smoke test only -- not for a reported run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render prompts and print them; make no API call and spend nothing.",
    )
    args = parser.parse_args(argv)

    spec = MODELS[args.model]
    template = load_prompt_template()
    documents = load_documents(args.corpus, limit=args.limit)
    if not documents:
        raise SystemExit(f"no documents under {args.corpus}")

    # The one place the call shape is decided. Nothing downstream branches on it, and
    # nothing about it reaches build_request_body -- so the two shapes send the same
    # body for the same model, and the only difference between them is how many
    # documents ride in it (Critical Rule 7 / B3-D6).
    if args.batch_size is None:
        calls = [documents]
        batching = WHOLE
        print(f"corpus: {len(documents)} documents -> 1 whole-corpus call")
    else:
        calls = batch_documents(documents, args.batch_size)
        batching = BATCHED
        print(
            f"corpus: {len(documents)} documents -> {len(calls)} batches "
            f"of {args.batch_size} (LEGACY shape)"
        )

    if args.dry_run:
        _dry_run(spec, calls, template)
        return

    if batching == BATCHED and args.batch_size != LEGACY_BATCH_SIZE:
        print(
            f"WARNING: --batch-size {args.batch_size} is neither the current "
            f"whole-corpus shape nor the legacy value ({LEGACY_BATCH_SIZE}); this "
            "run is comparable to nothing already reported."
        )

    run_id = _run_id()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"{run_id}_b3_{spec.key}_batches.jsonl"
    raw_records: list[dict] = []

    try:
        schemas = run_calls(spec, calls, template, raw_records)
    finally:
        # Written even when the run dies, so a truncated or unparseable response is
        # inspectable instead of lost with the process. The schema JSON below is
        # deliberately NOT written on that path: a partial run must not leave behind
        # a file that looks like a result.
        raw_path.write_text(
            "".join(json.dumps(record) + "\n" for record in raw_records),
            encoding="utf-8",
        )
        if raw_records:
            print(f"wrote {os.path.relpath(raw_path, _REPO_ROOT)}")

    merged = merge_batches(schemas)
    sources = [source for _call in calls for source, _text in _call]
    usage = [
        {
            "batch": record["batch"],
            "stop_reason": record["stop_reason"],
            "completion_tokens": record["completion_tokens"],
        }
        for record in raw_records
    ]
    output = build_output(
        merged,
        sources,
        run_id,
        spec,
        batching=batching,
        batch_size=args.batch_size,
        usage=usage,
    )

    out_path = args.out_dir / f"{run_id}_b3_{spec.key}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    skipped = len(calls) - len(schemas)
    n_attributes = sum(len(c["attributes"]) for c in output["classes"])
    print(
        f"induced: {len(output['classes'])} classes, {n_attributes} attributes, "
        f"{len(output['relations'])} relations"
        + (f" ({skipped} of {len(calls)} calls unparseable)" if skipped else "")
    )
    print(f"wrote {os.path.relpath(out_path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
