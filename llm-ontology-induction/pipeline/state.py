"""
P1 pipeline state (architecture plan §3).

One `TypedDict` threaded through every LangGraph node. Each node reads the
fields it needs and writes the one or two fields that are its own stage's
output -- nothing hidden in closures, nothing mutated that isn't declared
here.

Fields beyond the plan's own list (`class_name_map`, `run_id`,
`raw_log_path`, `extraction_skipped`) are bookkeeping the six-stage split
turned out to need in practice, not scope creep:

  * `class_name_map` -- Stage 4 (relation reconciliation) must rewrite every
    relation's endpoints from a document's original class name onto Stage 2's
    merged name. That map is exactly what Stage 2's own merge log already
    contains (P1-D5) -- `merge_log` records which original names collapsed
    into which merged name, so `class_name_map` is a deterministic Python
    inversion of it, not a second LLM call asked to re-derive the same fact.
  * `extraction_skipped` -- carried over from the two-stage pipeline's own
    bookkeeping (skip-and-continue fault tolerance for Stage 1). Still
    needed; six stages doesn't remove the need to know which of the 192
    extraction calls failed.
  * `run_id` -- identifies the run; `assemble` needs it for the output
    contract's metadata.
  * `raw_log_path` -- where every call's full record (prompt size, response,
    stop reason, completion tokens) is appended as it happens, by
    `pipeline.nodes._common.append_raw_record()`. Deliberately **not** a list
    field on `P1State` -- see that module's docstring for why appending to a
    state list from inside a node is the wrong shape here. `usage` below is
    the small, checkpoint-safe summary a node *does* return through state.

**`model_key` is stored, not the resolved `ModelSpec` object.** Every node
that needs the spec re-derives it via `MODELS[state["model_key"]]` rather
than reading a `spec` field. This isn't a style preference: `P1State` is
serialized to sqlite on every checkpoint (that's the whole point of using one
-- see `graph.py`), and `ModelSpec` is a custom dataclass with no registered
codec in LangGraph's checkpoint serializer. Storing it directly triggers "will
be blocked in a future version" on every resume today, and would hard-fail a
resume outright the moment that block ships. `model_key` is a plain string --
trivially serializable, and `MODELS` is a static registry already imported
everywhere a spec is needed, so re-deriving it costs nothing.
"""

from __future__ import annotations

from typing import TypedDict


class P1State(TypedDict):
    # Input
    documents: list[tuple[str, str]]  # (source_path, text), Stage 1's input
    model_key: str
    run_id: str
    raw_log_path: str

    # Stage 1 -- per-document extraction
    partial_schemas: list[dict]  # [{"source_document": str, "classes": [...], "relations": [...]}]
    extraction_skipped: int

    # Stage 2 -- type consolidation
    merged_classes: list[dict]  # [{"name", "parent", "attributes" (raw union, undeduped)}]
    merge_log: list[dict]  # [{"merged_name", "source_names": [...], "source_documents": [...]}]
    class_name_map: dict[str, str]  # normalized original name -> merged canonical name

    # Stage 3 -- attribute consolidation
    consolidated_attributes: dict[str, list[str]]  # merged class name -> deduped attribute list

    # Stage 4 -- relation reconciliation
    reconciled_relations: list[dict]  # [{"source", "label", "target"}], endpoints already merged names

    # Stage 5 -- taxonomy induction
    taxonomy_edges: dict[str, str | None]  # merged class name -> parent name, or None
    induced_superclasses: list[dict]  # [{"name", "parent", "attributes"}] -- new classes this stage
    # introduced that were named as a `parent` in taxonomy_edges but never appeared as their own
    # entry anywhere upstream. See pipeline/nodes/induce_taxonomy.py's docstring for why this
    # stage alone is allowed to add a class name absent from every earlier stage's output.

    # Stage 6 -- assemble + validate
    output: dict  # the final induced-schema contract dict

    # Cross-cutting bookkeeping
    usage: list[dict]  # [{"stage": str, "stop_reason": str | None, "completion_tokens": int | None}]
