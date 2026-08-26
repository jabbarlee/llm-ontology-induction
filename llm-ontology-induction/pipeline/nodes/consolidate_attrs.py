"""
Stage 3 -- attribute consolidation (architecture plan §2, Stage 3).

Splitting attribute-wording cleanup from Stage 2's class-identity merge is
justified by real evidence, not tidiness: Llama's batched B3 run produced a
`Tenant` class with 58 attributes because `monthly payment`,
`Monthly_Base_Rate`, `monthly_payment`, `payment amount`, and `rent` never
merged (results/findings/B3-FINDINGS.md). Class duplication and
attribute-wording duplication are different failure modes; conflating their
cleanup into one call (the old single-consolidation-prompt design) meant a
failure in one was indistinguishable from a failure in the other.

**Granularity decision (plan §8, open question 2): one call for all classes,
not one call per merged class.** Per-class calls would multiply the call
count by roughly the number of merged classes (~10, per the Opus 5 B3 run) on
top of Stage 1's 192 -- real money for a stage whose job is wording cleanup,
not judgment calls hard enough to need isolated attention the way per-document
extraction does. Recorded as P1-D7; revisit if per-class attribute lists turn
out too large for one call's output cap to hold together (the same class of
risk B3-D3 already tracks for Llama's 4,096-token Bedrock ceiling).
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.b3_single_shot.single_shot import clean_schema, parse_response
from baselines.shared import model_clients as mc

from pipeline.nodes._common import append_raw_record, invoke_or_abort
from pipeline.state import P1State

_MODULE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = _MODULE_ROOT / "prompts" / "p1_attributes_prompt.md"
CLASSES_PLACEHOLDER = "{{CLASSES}}"


def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    template = path.read_text(encoding="utf-8")
    if CLASSES_PLACEHOLDER not in template:
        raise ValueError(f"{path} has no {CLASSES_PLACEHOLDER} placeholder")
    return template


def render_prompt(template: str, merged_classes: list[dict]) -> str:
    # Only name/parent/attributes travel into this call -- merged_from is
    # Stage 2's own provenance bookkeeping and has no bearing on wording
    # cleanup, so it is not given to a call whose job is narrower than the
    # class list it's handed.
    payload = [{"name": c["name"], "parent": c["parent"], "attributes": c["attributes"]} for c in merged_classes]
    return template.replace(CLASSES_PLACEHOLDER, json.dumps(payload, indent=2))


def consolidate_attrs(state: P1State) -> dict:
    """LangGraph node: exactly one call, over every merged class's raw
    attribute union. Single-call-must-abort, same as Stage 2."""
    spec = mc.MODELS[state["model_key"]]
    merged_classes = state["merged_classes"]
    log_path = state["raw_log_path"]
    template = load_prompt_template()

    prompt = render_prompt(template, merged_classes)
    print(f"  [3/6 consolidate_attrs] {len(merged_classes)} merged classes, {len(prompt)} chars...")

    record: dict = {"stage": "consolidate_attrs", "input_class_count": len(merged_classes), "prompt_chars": len(prompt)}
    completion = invoke_or_abort(spec, prompt, "consolidate_attrs", log_path, record)
    print(f"    stop_reason={completion.stop_reason!r} completion_tokens={completion.completion_tokens}")

    try:
        parsed = parse_response(completion.text)
    except Exception:
        append_raw_record(log_path, record)
        raise

    cleaned = clean_schema({"classes": parsed.get("classes", []), "relations": []})
    consolidated_attributes = {c["name"]: c["attributes"] for c in cleaned["classes"]}

    record["parsed_classes"] = len(cleaned["classes"])
    append_raw_record(log_path, record)

    return {
        "consolidated_attributes": consolidated_attributes,
        "usage": state["usage"]
        + [{"stage": "consolidate_attrs", "stop_reason": completion.stop_reason, "completion_tokens": completion.completion_tokens}],
    }
