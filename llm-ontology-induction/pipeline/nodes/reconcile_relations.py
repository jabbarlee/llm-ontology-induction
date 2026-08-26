"""
Stage 4 -- relation reconciliation (architecture plan §2, Stage 4).

Non-optional per the plan: a relation extracted as (say) two documents'
"Landlord ... Premises" and "Property Owner ... Property" becomes dangling --
referencing class names that no longer exist -- the moment Stage 2 merges
those two classes into one canonical name. Left unfixed, this is the same
endpoint-conditioning failure that gives B1 a flat 0.0 on relations
(eval.metrics._relations_match buckets by class-level match; an endpoint with
no valid class match makes the whole relation unscoreable, see
results/findings/B3-FINDINGS.md's Opus 5 section for a worked example of
exactly this on real data).

**Split deliberately into a mechanical half and a judgment half:**

  * Endpoint remapping is a dict lookup against Stage 2's `class_name_map` --
    entirely deterministic, done in plain Python before any call. Asking a
    model to both rename endpoints *and* judge which relations are now
    duplicates in the same pass invites it to garble the mechanical part
    while doing the judgment part, for no accuracy benefit: the rename has
    exactly one correct answer per name, already computed.
  * What genuinely needs a model: after remapping, two relations may now
    share the same endpoints under different label wordings, and deciding
    "same link, reworded" from "same endpoints, different real link" is a
    judgment call, not a lookup. That is `reconcile_relations()`'s one call.

Any relation whose source or target has no entry in `class_name_map` is
dropped before the call, not sent in -- Stage 2 only ever adds entries for
class names it actually saw, so a miss here means the relation referenced a
class Stage 1 emitted but Stage 2's cleanup discarded as malformed
(baselines.b3_single_shot.single_shot.clean_schema already drops relations
with empty/malformed endpoints for the same reason). Counted and reported,
not silently absorbed.
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.b3_single_shot.single_shot import clean_schema, parse_response
from baselines.shared import model_clients as mc

from pipeline.nodes._common import append_raw_record, invoke_or_abort, normalize_name
from pipeline.state import P1State

_MODULE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = _MODULE_ROOT / "prompts" / "p1_relations_prompt.md"
RELATIONS_PLACEHOLDER = "{{RELATIONS}}"


def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    template = path.read_text(encoding="utf-8")
    if RELATIONS_PLACEHOLDER not in template:
        raise ValueError(f"{path} has no {RELATIONS_PLACEHOLDER} placeholder")
    return template


def render_prompt(template: str, relations: list[dict]) -> str:
    payload = [{"source": r["source"], "label": r["label"], "target": r["target"]} for r in relations]
    return template.replace(RELATIONS_PLACEHOLDER, json.dumps(payload, indent=2))


def remap_relation_endpoints(
    partial_schemas: list[dict], class_name_map: dict[str, str]
) -> tuple[list[dict], int]:
    """Pure Python: gather every relation across all 192 partial schemas,
    rewrite `source`/`target` onto Stage 2's merged class names, and drop the
    (exact, case/whitespace-folded) literal duplicates this remap
    mechanically produces -- e.g. two documents both saying "X manages Y"
    about entities that only later turned out to be the same X and the same
    Y. This is the same naive, exact-string dedup B3-D4 established for a
    single response; applying it here, after remapping, only collapses
    remap-created literal duplicates, not near-wording duplicates -- those
    are `reconcile_relations()`'s call's job, not this function's.

    Returns (remapped_relations, dropped_count) -- `dropped_count` is every
    relation whose source or target had no entry in `class_name_map`.
    """
    seen: set[tuple[str, str, str]] = set()
    remapped: list[dict] = []
    dropped = 0

    for entry in partial_schemas:
        for rel in entry.get("relations", []):
            source_key = normalize_name(rel.get("source", ""))
            target_key = normalize_name(rel.get("target", ""))
            label = rel.get("label")
            if source_key not in class_name_map or target_key not in class_name_map or not isinstance(label, str):
                dropped += 1
                continue
            mapped = {
                "source": class_name_map[source_key],
                "label": label,
                "target": class_name_map[target_key],
            }
            dedup_key = (mapped["source"], normalize_name(mapped["label"]), mapped["target"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            remapped.append(mapped)

    return remapped, dropped


def reconcile_relations(state: P1State) -> dict:
    """LangGraph node: deterministic remap, then exactly one call to merge
    the remaining near-wording duplicates. Single-call-must-abort, same as
    Stages 2 and 3, for the call half -- the remap half cannot itself fail in
    a way worth aborting over (a bad lookup just means a dropped relation,
    already counted)."""
    spec = mc.MODELS[state["model_key"]]
    partial_schemas = state["partial_schemas"]
    class_name_map = state["class_name_map"]
    log_path = state["raw_log_path"]

    remapped, dropped = remap_relation_endpoints(partial_schemas, class_name_map)
    print(f"  [4/6 reconcile_relations] {len(remapped)} remapped relations ({dropped} dropped, unmapped endpoint)...")

    if not remapped:
        # Nothing to reconcile -- an empty relation set is a legitimate
        # outcome (Rule 6-style: absence of evidence is not a call failure),
        # not a reason to make a call with nothing in it.
        return {"reconciled_relations": [], "usage": state["usage"]}

    template = load_prompt_template()
    prompt = render_prompt(template, remapped)
    print(f"    {len(prompt)} chars...")

    record: dict = {"stage": "reconcile_relations", "input_relation_count": len(remapped), "dropped_unmapped": dropped, "prompt_chars": len(prompt)}
    completion = invoke_or_abort(spec, prompt, "reconcile_relations", log_path, record)
    print(f"    stop_reason={completion.stop_reason!r} completion_tokens={completion.completion_tokens}")

    try:
        parsed = parse_response(completion.text)
    except Exception:
        append_raw_record(log_path, record)
        raise

    cleaned = clean_schema({"classes": [], "relations": parsed.get("relations", [])})
    reconciled = cleaned["relations"]

    record["parsed_relations"] = len(reconciled)
    append_raw_record(log_path, record)

    return {
        "reconciled_relations": reconciled,
        "usage": state["usage"]
        + [{"stage": "reconcile_relations", "stop_reason": completion.stop_reason, "completion_tokens": completion.completion_tokens}],
    }
