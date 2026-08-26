"""
Stage 6 -- assemble + validate (architecture plan §2, Stage 6). Deterministic;
no model call. Pure Python, joining every earlier stage's output into the
induced-schema contract, then loading the result back through
`eval/schema_ir.py`'s real `parse_induced_schema()` before declaring success --
the same contract verification used for B1 and B3, not a P1-specific check
that could drift from what the harness actually expects.

What gets joined, and from where:

  * classes -- Stage 2's `merged_classes` (name, merged `parent` baseline)
    plus Stage 5's `induced_superclasses` (any newly declared supertype),
    with `attributes` taken from Stage 3's `consolidated_attributes` and
    `parent` taken from Stage 5's `taxonomy_edges` -- the final word on
    `parent` always belongs to Stage 5, even for a class whose parent never
    changed there, since `taxonomy_edges` already carries every class's
    Stage-2 value forward untouched when Stage 5 found no new evidence.
  * relations -- Stage 4's `reconciled_relations`, unchanged.

No cleanup happens here beyond assembling these into the contract shape --
every stage upstream already did its own cleaning (`clean_schema()`), so
Stage 6 doing more would be silently re-deciding something an earlier stage
already settled.
"""

from __future__ import annotations

from eval.schema_ir import parse_induced_schema

from baselines.shared import model_clients as mc
from pipeline.state import P1State


def build_output(state: P1State) -> dict:
    merged_classes = state["merged_classes"]
    consolidated_attributes = state["consolidated_attributes"]
    taxonomy_edges = state["taxonomy_edges"]
    induced_superclasses = state.get("induced_superclasses") or []

    classes = [
        {
            "name": c["name"],
            "parent": taxonomy_edges.get(c["name"], c["parent"]),
            "attributes": consolidated_attributes.get(c["name"], c["attributes"]),
        }
        for c in merged_classes
    ]
    classes.extend(
        {"name": c["name"], "parent": c.get("parent"), "attributes": c.get("attributes", [])}
        for c in induced_superclasses
    )

    usage = state["usage"]
    stage_stop_reasons = {}
    stage_completion_tokens = {}
    for entry in usage:
        stage = entry["stage"]
        stage_stop_reasons.setdefault(stage, []).append(entry["stop_reason"])
        stage_completion_tokens[stage] = stage_completion_tokens.get(stage, 0) + (entry["completion_tokens"] or 0)

    return {
        "classes": classes,
        "relations": state["reconciled_relations"],
        "metadata": {
            "condition": "P1",
            "model": mc.MODELS[state["model_key"]].model_id,
            "run_id": state["run_id"],
            "source_documents": [source for source, _text in state["documents"]],
            "extraction_calls": len(state["documents"]),
            "extraction_skipped": state["extraction_skipped"],
            "merge_log": state["merge_log"],
            "induced_superclasses": [c["name"] for c in induced_superclasses],
            "stage_stop_reasons": stage_stop_reasons,
            "stage_completion_tokens": stage_completion_tokens,
        },
    }


def assemble(state: P1State) -> dict:
    """LangGraph node: build the contract dict, then load it back through the
    real scoring-harness parser before returning -- a Stage 6 that produced
    something the harness cannot read is a failed run, not a successful one
    with a formatting quirk, and must be caught here rather than surfacing as
    a mysterious eval.report crash later."""
    output = build_output(state)
    parse_induced_schema(output)  # raises if the contract isn't actually satisfied
    print(
        f"  [6/6 assemble] {len(output['classes'])} classes, "
        f"{sum(len(c['attributes']) for c in output['classes'])} attributes, "
        f"{len(output['relations'])} relations -- contract verified"
    )
    return {"output": output}
