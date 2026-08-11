"""
B3 baseline -- single-shot LLM schema induction (baselines/DECISIONS.md, B3-D1..D5).

Reads the corpus as whole documents, groups them into fixed-size batches identical
for every model (B3-D2), sends each batch to one model under one frozen prompt
(B3-D1), merges the per-batch schemas by exact-string deduplication only (B3-D4),
and emits a single induced-schema JSON matching the contract in
eval/schema_ir.py::parse_induced_schema (eval/PLAN.md §2).

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
    python -m baselines.single_shot --model haiku45 --dry-run
    python -m baselines.single_shot --model qwen3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from baselines import model_clients as mc
from baselines.model_clients import MODELS, ModelSpec

# --- Frozen run settings (B3-D2) -----------------------------------------
# Uniform across all five models, including the open-weight one: giving the
# cloud models bigger batches because their context windows allow it would make
# each model solve a differently-shaped task, which is the one thing this
# comparison cannot tolerate (Critical Rule 7).
BATCH_SIZE = 10

CONDITION = "B3"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_ROOT = _REPO_ROOT / "data" / "documents"
_SUBDIRS = ("csv_exports", "lease_texts", "notes", "messages")
_READABLE_SUFFIXES = (".csv", ".txt")
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "raw"

PROMPT_PATH = _REPO_ROOT / "baselines" / "prompts" / "b3_extraction_prompt.md"
DOCUMENTS_PLACEHOLDER = "{{DOCUMENTS}}"


class ResponseParseError(ValueError):
    """No schema object could be recovered from a model response."""


# ---------------------------------------------------------------------------
# Corpus loading (B3-D2)
# ---------------------------------------------------------------------------

def load_documents(
    root: Path = _CORPUS_ROOT, limit: int | None = None
) -> list[tuple[str, str]]:
    """Every document as (repo-relative path, raw text), in a deterministic order.

    Deliberately *not* B1's load_corpus(): that one sentence-splits prose and
    flattens CSV rows into pseudo-sentences, which is preprocessing B1's statistics
    need. B3's premise is that the model reads the mess exactly as it lies, so files
    are handed over verbatim.

    Order is fixed (subdirectories in _SUBDIRS order, filenames sorted within each)
    because batch membership is derived from it, and batches must be identical
    across all five models (Critical Rule 7).
    """
    documents: list[tuple[str, str]] = []
    for subdir in _SUBDIRS:
        directory = root / subdir
        if not directory.is_dir():
            continue
        paths = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in _READABLE_SUFFIXES
        )
        # Filter before capping: `--limit 2` means two documents, not two
        # directory entries of which some unreadable stray may be one.
        for path in (paths[:limit] if limit else paths):
            documents.append((f"{subdir}/{path.name}", path.read_text(encoding="utf-8")))
    return documents


def batch_documents(
    documents: list[tuple[str, str]], size: int = BATCH_SIZE
) -> list[list[tuple[str, str]]]:
    """Split into fixed-size batches. A function of the document order alone -- it
    takes no ModelSpec, so it is structurally incapable of batching one model
    differently from another (Critical Rule 7)."""
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


def format_documents(batch: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"--- document: {source} ---\n{text.strip()}" for source, text in batch
    )


def render_prompt(template: str, batch: list[tuple[str, str]]) -> str:
    """Substitute the batch's documents into the frozen template.

    str.replace, not str.format: the template contains a JSON output example, and
    str.format would try to read its braces as fields. Interpolation touches the
    documents block and nothing else -- the instruction text is byte-identical for
    every batch and every model (Critical Rule 2).
    """
    return template.replace(DOCUMENTS_PLACEHOLDER, format_documents(batch))


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's <think> block.

    The open-weight condition is a reasoning model and emits these inline. Both
    shapes occur in practice: a matched pair, and a bare closing tag when the
    opening one is suppressed by the server, so the tail after the last </think>
    is taken in that case.
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
    merged: dict, sources: list[str], run_id: str, spec: ModelSpec
) -> dict:
    return {
        "classes": merged["classes"],
        "relations": merged["relations"],
        "metadata": {
            "condition": CONDITION,
            "model": spec.model_id,
            "run_id": run_id,
            "source_documents": sources,
        },
    }


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_batches(
    spec: ModelSpec, batches: list[list[tuple[str, str]]], template: str
) -> tuple[list[dict], list[dict]]:
    """Call `spec` once per batch. Returns (per-batch schemas, raw records).

    A batch that fails to parse is recorded in the raw log with its error and
    skipped, so one unusable response cannot abort a run that has already been paid
    for -- but the count of skipped batches is surfaced by main() and belongs in any
    reported result.
    """
    schemas: list[dict] = []
    raw_records: list[dict] = []

    for index, batch in enumerate(batches, start=1):
        prompt = render_prompt(template, batch)
        sources = [source for source, _text in batch]
        print(f"  batch {index}/{len(batches)} ({len(batch)} documents)...")

        response = mc.invoke(spec, prompt)
        record = {
            "batch": index,
            "source_documents": sources,
            "prompt_chars": len(prompt),
            "response": response,
        }
        try:
            schema = parse_response(response)
        except ResponseParseError as exc:
            record["parse_error"] = str(exc)
            print(f"    UNPARSEABLE: {exc}")
        else:
            schemas.append(schema)
            record["parsed_classes"] = len(schema.get("classes") or [])
            record["parsed_relations"] = len(schema.get("relations") or [])
        raw_records.append(record)

    return schemas, raw_records


def _dry_run(spec: ModelSpec, batches: list[list[tuple[str, str]]], template: str) -> None:
    """Render every prompt and print the first one. Makes no API call."""
    print(f"DRY RUN -- no API calls. model={spec.key} ({spec.model_id})\n")
    for index, batch in enumerate(batches, start=1):
        prompt = render_prompt(template, batch)
        sources = ", ".join(source for source, _text in batch)
        print(f"batch {index}: {len(batch)} documents, {len(prompt)} chars -- {sources}")
    print("\n--- rendered prompt, batch 1 ---\n")
    print(render_prompt(template, batches[0]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the B3 single-shot LLM baseline.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--corpus", type=Path, default=_CORPUS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
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
    batches = batch_documents(documents, args.batch_size)
    print(
        f"corpus: {len(documents)} documents -> {len(batches)} batches "
        f"of {args.batch_size}"
    )

    if args.dry_run:
        _dry_run(spec, batches, template)
        return

    if args.batch_size != BATCH_SIZE:
        print(
            f"WARNING: --batch-size {args.batch_size} differs from the frozen "
            f"B3-D2 value ({BATCH_SIZE}); this run is not comparable to the others."
        )

    run_id = _run_id()
    schemas, raw_records = run_batches(spec, batches, template)
    merged = merge_batches(schemas)
    sources = [source for _batch in batches for source, _text in _batch]
    output = build_output(merged, sources, run_id, spec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{run_id}_b3_{spec.key}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    raw_path = args.out_dir / f"{run_id}_b3_{spec.key}_batches.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records), encoding="utf-8"
    )

    skipped = len(batches) - len(schemas)
    n_attributes = sum(len(c["attributes"]) for c in output["classes"])
    print(
        f"induced: {len(output['classes'])} classes, {n_attributes} attributes, "
        f"{len(output['relations'])} relations"
        + (f" ({skipped} batches unparseable)" if skipped else "")
    )
    print(f"wrote {os.path.relpath(out_path, _REPO_ROOT)}")
    print(f"wrote {os.path.relpath(raw_path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
