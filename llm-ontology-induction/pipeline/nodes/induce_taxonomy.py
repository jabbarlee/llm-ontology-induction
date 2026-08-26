"""
Stage 5 -- taxonomy induction (architecture plan §2, Stage 5; capability (b)).

The one stage with the clearest B3-impossibility argument, and the only stage
in this pipeline allowed to introduce a class name that appears nowhere in
any partial schema -- every other stage (2, 3, 4) is bound by "only reconcile
what the extractions already contain." That is a deliberate, narrow exception
to Critical Rule 1's spirit, not a loosening of it: Rule 1 forbids the gold
schema's own vocabulary from reaching a prompt or the code that builds one,
which this stage's prompt is independently verified against
(test_no_domain_vocabulary_leakage) exactly like every other prompt in this
project. What's different here is that the *output* is allowed to contain a
class name absent from the *input* -- a capability B3 structurally lacks,
since B3 never has a "step back and look at the finished answer" moment.

**Caution the plan states up front, carried into the prompt itself
(P1-D6):** this is the stage most likely to invent structure that is not
there. `prompts/p1_taxonomy_prompt.md` requires concrete support (a naming
pattern or real attribute overlap across at least two classes) and defaults
to `null`, mirroring the extraction prompt's own "when in doubt, use null"
conservatism. A stage that hallucinates a tidy hierarchy would be worse than
one that finds nothing, and the prompt is written to make the cheap answer
(null) the default rather than something the model has to choose against its
grain.

Only `name` and `parent` are read back from this stage's response --
`attributes` are not, even though the prompt echoes them for the model's own
context. Stage 3 already finalized attributes; trusting Stage 5's echo of
them back would reopen a wording-drift risk for no reason, when the only new
fact this stage can contribute is a `parent` value.

**A newly proposed supertype must be declared, not just referenced.** The
prompt requires the model to add any brand-new category as its own entry in
the response, not only as a string written into some other class's `parent`
field -- a `parent` naming nothing in the schema's own `classes` list would
be exactly the dangling-endpoint failure Stage 4's docstring describes for
relations (see results/findings/B3-FINDINGS.md's Opus 5 section for a worked
real-data example), just one layer up, in taxonomy instead of relations. If
the model still names a parent it never declares, that reference is dropped
back to `null` rather than assembled into a schema pointing at nothing --
counted in the raw log as `undeclared_parents_dropped`, never silently
absorbed. Declared new supertypes are returned separately as
`induced_superclasses`, since they are not among the classes Stage 2
produced and Stage 6 needs to know to add them to the final class list.
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.b3_single_shot.single_shot import clean_schema, parse_response
from baselines.shared import model_clients as mc

from pipeline.nodes._common import append_raw_record, invoke_or_abort
from pipeline.state import P1State

_MODULE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = _MODULE_ROOT / "prompts" / "p1_taxonomy_prompt.md"
CLASSES_PLACEHOLDER = "{{CLASSES}}"


def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    template = path.read_text(encoding="utf-8")
    if CLASSES_PLACEHOLDER not in template:
        raise ValueError(f"{path} has no {CLASSES_PLACEHOLDER} placeholder")
    return template


def render_prompt(template: str, classes: list[dict]) -> str:
    return template.replace(CLASSES_PLACEHOLDER, json.dumps(classes, indent=2))


def induce_taxonomy(state: P1State) -> dict:
    """LangGraph node: exactly one call, over the final class list with
    Stage 3's consolidated attributes. Single-call-must-abort, same as
    Stages 2 and 4."""
    spec = mc.MODELS[state["model_key"]]
    merged_classes = state["merged_classes"]
    consolidated_attributes = state["consolidated_attributes"]
    log_path = state["raw_log_path"]

    classes_in = [
        {
            "name": c["name"],
            "parent": c["parent"],
            "attributes": consolidated_attributes.get(c["name"], c["attributes"]),
        }
        for c in merged_classes
    ]

    template = load_prompt_template()
    prompt = render_prompt(template, classes_in)
    print(f"  [5/6 induce_taxonomy] {len(classes_in)} classes, {len(prompt)} chars...")

    record: dict = {"stage": "induce_taxonomy", "input_class_count": len(classes_in), "prompt_chars": len(prompt)}
    completion = invoke_or_abort(spec, prompt, "induce_taxonomy", log_path, record)
    print(f"    stop_reason={completion.stop_reason!r} completion_tokens={completion.completion_tokens}")

    try:
        parsed = parse_response(completion.text)
    except Exception:
        append_raw_record(log_path, record)
        raise

    cleaned = clean_schema({"classes": parsed.get("classes", []), "relations": []})
    returned_by_name = {c["name"]: c for c in cleaned["classes"]}
    input_names = {c["name"] for c in merged_classes}
    original_parent = {c["name"]: c["parent"] for c in merged_classes}

    # Every input class keeps a taxonomy_edges entry regardless of whether the
    # model's response happened to include it -- a class this stage dropped
    # is not evidence its parent should now be unknown, it just means this
    # call's own bookkeeping fell through for that one name. Fall back to
    # whatever parent it carried in (its Stage 2 value), which is exactly
    # what leaving it untouched would have produced anyway.
    taxonomy_edges: dict[str, str | None] = {}
    dropped_by_model = 0
    for c in merged_classes:
        name = c["name"]
        if name in returned_by_name:
            taxonomy_edges[name] = returned_by_name[name]["parent"]
        else:
            taxonomy_edges[name] = c["parent"]
            dropped_by_model += 1

    # The "must be declared" rule applies only to a parent this stage itself
    # newly asserted -- a value already present before Stage 5 ran (carried
    # through from Stage 2 untouched, whether or not it happens to name a
    # declared class) is not a claim this stage is making, and is not this
    # stage's job to retroactively validate. Only a name that is both new
    # relative to that class's own prior parent *and* absent from the input
    # class set counts as a newly proposed supertype needing its own entry.
    newly_asserted = {
        name: parent
        for name, parent in taxonomy_edges.items()
        if parent is not None and parent != original_parent.get(name) and parent not in input_names
    }
    induced_superclasses: list[dict] = []
    undeclared_parents = 0
    declared_new_names: set[str] = set()
    for name, proposed_parent in newly_asserted.items():
        if proposed_parent in returned_by_name:
            if proposed_parent not in declared_new_names:
                induced_superclasses.append(returned_by_name[proposed_parent])
                declared_new_names.add(proposed_parent)
        else:
            # A class named as a `parent` here but never given its own entry
            # in the response is a newly proposed supertype the prompt
            # requires the model to also declare (p1_taxonomy_prompt.md) --
            # but a model can still fail to follow that; treated the same way
            # every other stage treats a class dangling with no declared
            # entry: drop the reference rather than assemble a schema with a
            # parent that names nothing.
            undeclared_parents += 1
            taxonomy_edges[name] = None

    record["parsed_classes"] = len(cleaned["classes"])
    record["dropped_by_model"] = dropped_by_model
    record["induced_superclasses"] = [c["name"] for c in induced_superclasses]
    record["undeclared_parents_dropped"] = undeclared_parents
    new_parents_assigned = sum(
        1 for c in merged_classes if c["parent"] is None and taxonomy_edges[c["name"]] is not None
    )
    record["new_parents_assigned"] = new_parents_assigned
    append_raw_record(log_path, record)

    return {
        "taxonomy_edges": taxonomy_edges,
        "induced_superclasses": induced_superclasses,
        "usage": state["usage"]
        + [{"stage": "induce_taxonomy", "stop_reason": completion.stop_reason, "completion_tokens": completion.completion_tokens}],
    }
