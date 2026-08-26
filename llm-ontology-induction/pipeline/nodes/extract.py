"""
Stage 1 -- per-document extraction (architecture plan §2, Stage 1).

Capability (a): full attention per document. 192 calls, one document each,
under the identical frozen `b3_extraction_prompt.md` B3 uses -- imported by
path, never copied (P1-D2). If P1's extraction prompt differed from B3's even
slightly, a difference in the final schema could be attributed to either the
prompting or the pipeline shape, and the comparison this whole baseline exists
to make would stop meaning what the paper claims it means.

Fault tolerance: with N=192 independent calls, one bad document (a
truncation, a refusal, an unparseable response) is recorded and skipped
rather than fatal -- unlike every later stage, which has nothing else to
fall back on if its one call fails. Identical to the two-stage pipeline's own
Stage 1 (baselines/p1_pipeline/pipeline.py, prior revision) -- this stage's
job did not change when the consolidation stage split into four.
"""

from __future__ import annotations

from baselines.b3_single_shot.single_shot import (
    PROMPT_PATH as B3_PROMPT_PATH,
)
from baselines.b3_single_shot.single_shot import (
    ResponseParseError,
    clean_schema,
    load_prompt_template,
    parse_response,
    render_prompt,
)
from baselines.shared import model_clients as mc

from pipeline.nodes._common import append_raw_record
from pipeline.state import P1State


def extract(state: P1State) -> dict:
    """LangGraph node: one call per document in `state["documents"]`.

    Returns only the keys this node owns (`partial_schemas`,
    `extraction_skipped`, plus this stage's contribution to `raw_records` and
    `usage`) -- LangGraph merges a node's return dict into the shared state,
    it does not require the node to pass the rest of the state through.
    """
    spec = mc.MODELS[state["model_key"]]
    documents = state["documents"]
    log_path = state["raw_log_path"]
    template = load_prompt_template(B3_PROMPT_PATH)

    schemas: list[dict] = []
    usage: list[dict] = []
    skipped = 0

    for index, document in enumerate(documents, start=1):
        source, _text = document
        prompt = render_prompt(template, [document])
        record: dict = {"stage": "extract", "call": index, "source_document": source, "prompt_chars": len(prompt)}
        print(f"  [1/6 extract] {index}/{len(documents)}: {source}...")

        try:
            completion = mc.invoke(spec, prompt)
        except mc.ModelResponseError as exc:
            label = "refused" if isinstance(exc, mc.RefusalError) else "truncated"
            record["response"] = exc.text
            record["stop_reason"] = label
            record["completion_tokens"] = exc.completion_tokens
            record["error"] = str(exc)
            append_raw_record(log_path, record)
            usage.append({"stage": "extract", "stop_reason": label, "completion_tokens": exc.completion_tokens})
            skipped += 1
            print(f"    {label} -- skipped")
            continue

        record["response"] = completion.text
        record["stop_reason"] = completion.stop_reason
        record["completion_tokens"] = completion.completion_tokens
        usage.append(
            {"stage": "extract", "stop_reason": completion.stop_reason, "completion_tokens": completion.completion_tokens}
        )

        try:
            schema = parse_response(completion.text)
        except ResponseParseError as exc:
            record["parse_error"] = str(exc)
            append_raw_record(log_path, record)
            skipped += 1
            print("    UNPARSEABLE -- skipped")
            continue

        cleaned = clean_schema(schema)
        record["parsed_classes"] = len(cleaned["classes"])
        record["parsed_relations"] = len(cleaned["relations"])
        append_raw_record(log_path, record)
        schemas.append({"source_document": source, **cleaned})

    return {
        "partial_schemas": schemas,
        "extraction_skipped": skipped,
        "usage": state.get("usage", []) + usage,
    }
