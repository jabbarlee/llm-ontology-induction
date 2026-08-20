"""
P1 baseline -- staged pipeline: N per-document extraction calls, then one
consolidation call (baselines/DECISIONS.md P1-D1..D3).

This is the pipeline B1 (statistical, zero-LLM) and B3 (single whole-corpus
call, naive exact-string cleanup) both exist to give something to beat. Where
B3 asks a model to read the entire corpus in one breath and merges by exact
string match, P1 asks the model to read one document at a time -- so each
extraction call sees a manageable, uniform amount of context -- and then asks
a *second* model call to reconcile the N results into one schema, resolving
cross-wording the way B3-D4 deliberately refuses to. That reconciliation is
the actual thing P1 exists to test the value of: whether an LLM doing the
merge, instead of an exact-string key, recovers more of the gold schema
without inventing structure that was not in any of the per-document reads.

Reuses B3's document loading, prompt rendering, response parsing, and
malformed-element cleanup wholesale (baselines.b3_single_shot.single_shot) --
the per-document extraction stage is doing exactly what a B3 call does, just
with `documents` always of length 1, and the frozen extraction prompt (Critical
Rule 2) is the same one, unmodified. Only the consolidation prompt and stage
are new.

HARD RULE (Critical Rule 1): zero gold-schema vocabulary in this module or in
either prompt. Enforced by
baselines/tests/test_p1_pipeline.py::test_no_domain_vocabulary_leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from baselines.b3_single_shot.single_shot import (
    ResponseParseError,
    clean_schema,
    load_documents,
    load_prompt_template,
    render_prompt,
    parse_response,
)
from baselines.b3_single_shot.single_shot import PROMPT_PATH as B3_PROMPT_PATH
from baselines.b3_single_shot.single_shot import (
    _CORPUS_ROOT,
    _READABLE_SUFFIXES,  # noqa: F401 -- re-exported for test parity with B3's fixture
    _SUBDIRS,  # noqa: F401
)
from baselines.shared import model_clients as mc
from baselines.shared.model_clients import MODELS, ModelSpec

CONDITION = "P1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "raw"

_MODULE_ROOT = _REPO_ROOT / "baselines" / "p1_pipeline"
CONSOLIDATION_PROMPT_PATH = _MODULE_ROOT / "prompts" / "p1_consolidation_prompt.md"
SCHEMAS_PLACEHOLDER = "{{SCHEMAS}}"


# ---------------------------------------------------------------------------
# Consolidation prompt rendering (Critical Rule 2, extended to a second prompt)
# ---------------------------------------------------------------------------

def load_consolidation_prompt_template(path: Path = CONSOLIDATION_PROMPT_PATH) -> str:
    template = path.read_text(encoding="utf-8")
    if SCHEMAS_PLACEHOLDER not in template:
        raise ValueError(f"{path} has no {SCHEMAS_PLACEHOLDER} placeholder")
    return template


def render_consolidation_prompt(template: str, per_document_schemas: list[dict]) -> str:
    """Substitute the extraction stage's output into the frozen consolidation
    template. str.replace, not str.format, for the same reason as B3's
    render_prompt: the template's own JSON output example would otherwise be
    misread as format fields.
    """
    payload = json.dumps(per_document_schemas, indent=2)
    return template.replace(SCHEMAS_PLACEHOLDER, payload)


# ---------------------------------------------------------------------------
# Stage 1 -- per-document extraction
# ---------------------------------------------------------------------------

def run_extraction_stage(
    spec: ModelSpec,
    documents: list[tuple[str, str]],
    template: str,
    raw_records: list[dict],
) -> list[dict]:
    """One call per document. Returns the per-document schemas that parsed and
    completed cleanly, each tagged with the document it came from.

    Unlike B3's single whole-corpus call, a failure here -- a truncation, a
    refusal, an unparseable response -- is recorded and skipped rather than
    fatal: with N=192 independent calls, one bad document does not invalidate
    the run the way it would if there were only one call to begin with. The
    count of skipped documents is returned to the caller and belongs in any
    reported result.
    """
    schemas: list[dict] = []
    for index, document in enumerate(documents, start=1):
        source, _text = document
        prompt = render_prompt(template, [document])
        record: dict = {"call": index, "source_document": source, "prompt_chars": len(prompt)}
        print(f"  extraction {index}/{len(documents)}: {source}...")

        try:
            completion = mc.invoke(spec, prompt)
        except mc.ModelResponseError as exc:
            label = "refused" if isinstance(exc, mc.RefusalError) else "truncated"
            record["response"] = exc.text
            record["stop_reason"] = label
            record["completion_tokens"] = exc.completion_tokens
            record["error"] = str(exc)
            raw_records.append(record)
            print(f"    {label} -- skipped")
            continue

        record["response"] = completion.text
        record["stop_reason"] = completion.stop_reason
        record["completion_tokens"] = completion.completion_tokens

        try:
            schema = parse_response(completion.text)
        except ResponseParseError as exc:
            record["parse_error"] = str(exc)
            raw_records.append(record)
            print(f"    UNPARSEABLE -- skipped")
            continue

        cleaned = clean_schema(schema)
        record["parsed_classes"] = len(cleaned["classes"])
        record["parsed_relations"] = len(cleaned["relations"])
        raw_records.append(record)
        schemas.append({"source_document": source, **cleaned})

    return schemas


# ---------------------------------------------------------------------------
# Stage 2 -- consolidation
# ---------------------------------------------------------------------------

def run_consolidation_stage(
    spec: ModelSpec,
    per_document_schemas: list[dict],
    template: str,
    raw_record: dict,
) -> dict:
    """The one call that reconciles every per-document schema into one final
    schema. Fills `raw_record` in place so the caller still has it if this
    raises. Never skips on failure -- unlike Stage 1, there is exactly one
    consolidation call and nothing else to fall back on if it fails.
    """
    prompt = render_consolidation_prompt(template, per_document_schemas)
    print(f"  consolidation ({len(per_document_schemas)} partial schemas, {len(prompt)} chars)...")

    raw_record["input_schema_count"] = len(per_document_schemas)
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
    cleaned = clean_schema(schema)
    raw_record["parsed_classes"] = len(cleaned["classes"])
    raw_record["parsed_relations"] = len(cleaned["relations"])
    return cleaned


# ---------------------------------------------------------------------------
# Output assembly (same contract as B3 -- eval/schema_ir.py::parse_induced_schema)
# ---------------------------------------------------------------------------

def build_output(
    cleaned: dict,
    sources: list[str],
    run_id: str,
    spec: ModelSpec,
    extraction_calls: int,
    extraction_skipped: int,
    consolidation_stop_reason: str | None,
    consolidation_completion_tokens: int | None,
) -> dict:
    return {
        "classes": cleaned["classes"],
        "relations": cleaned["relations"],
        "metadata": {
            "condition": CONDITION,
            "model": spec.model_id,
            "run_id": run_id,
            "source_documents": sources,
            "extraction_calls": extraction_calls,
            "extraction_skipped": extraction_skipped,
            "consolidation_stop_reason": consolidation_stop_reason,
            "consolidation_completion_tokens": consolidation_completion_tokens,
        },
    }


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


def _dry_run(spec: ModelSpec, documents: list[tuple[str, str]], extraction_template: str) -> None:
    """Render the first document's extraction prompt and print sizing for the
    whole run. Makes no API call, spends nothing."""
    print(f"DRY RUN -- no API calls. model={spec.key} ({spec.model_id})")
    print(f"output cap: {spec.max_output_tokens} tokens")
    print(f"{len(documents)} documents -> {len(documents)} extraction calls + 1 consolidation call\n")
    sample = render_prompt(extraction_template, [documents[0]])
    print(f"--- rendered extraction prompt, document 1 ({len(sample)} chars) ---\n")
    print(sample)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the P1 staged-pipeline baseline.")
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
        help="Render the first extraction prompt and print sizing; make no API call.",
    )
    args = parser.parse_args(argv)

    spec = MODELS[args.model]
    extraction_template = load_prompt_template(B3_PROMPT_PATH)
    consolidation_template = load_consolidation_prompt_template()
    documents = load_documents(args.corpus, limit=args.limit)
    if not documents:
        raise SystemExit(f"no documents under {args.corpus}")
    print(f"corpus: {len(documents)} documents -> {len(documents)} extraction calls + 1 consolidation call")

    if args.dry_run:
        _dry_run(spec, documents, extraction_template)
        return

    run_id = _run_id()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"{run_id}_p1_{spec.key}_calls.jsonl"
    raw_records: list[dict] = []
    consolidation_record: dict = {}

    try:
        per_document_schemas = run_extraction_stage(spec, documents, extraction_template, raw_records)
        if not per_document_schemas:
            raise SystemExit("every extraction call failed or was unparseable -- nothing to consolidate")
        cleaned = run_consolidation_stage(
            spec, per_document_schemas, consolidation_template, consolidation_record
        )
    finally:
        # Written even if the pipeline dies mid-run, so every extraction call
        # already paid for is inspectable, plus the consolidation attempt.
        raw_records.append({"stage": "consolidation", **consolidation_record})
        raw_path.write_text(
            "".join(json.dumps(r) + "\n" for r in raw_records), encoding="utf-8"
        )
        print(f"wrote {os.path.relpath(raw_path, _REPO_ROOT)}")

    sources = [source for source, _text in documents]
    output = build_output(
        cleaned,
        sources,
        run_id,
        spec,
        extraction_calls=len(documents),
        extraction_skipped=len(documents) - len(per_document_schemas),
        consolidation_stop_reason=consolidation_record.get("stop_reason"),
        consolidation_completion_tokens=consolidation_record.get("completion_tokens"),
    )

    out_path = args.out_dir / f"{run_id}_p1_{spec.key}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    n_attributes = sum(len(c["attributes"]) for c in output["classes"])
    skipped = output["metadata"]["extraction_skipped"]
    print(
        f"induced: {len(output['classes'])} classes, {n_attributes} attributes, "
        f"{len(output['relations'])} relations"
        + (f" ({skipped} of {len(documents)} extraction calls unparseable)" if skipped else "")
    )
    print(f"wrote {os.path.relpath(out_path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
