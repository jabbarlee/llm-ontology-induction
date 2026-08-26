"""
Stage 2 -- type consolidation (architecture plan §2, Stage 2). The novelty
stage: capability (c), an explicit, instrumentable merge you can point at in
the paper rather than assert.

Takes every class from all 192 partial schemas and asks, in one call, which
ones are the same real-world kind of thing. Scoped to class *identity* only --
attribute wording (Stage 3) and relation wording (Stage 4) are deliberately
not this stage's job, so a failure in one is not conflated with a failure in
another (the whole reason the six-stage split exists over the old one-call
consolidation).

Produces two things beyond the merged class list itself, both required by
later stages and both instrumentation the roadmap asked for (P1-D5):

  * `merge_log` -- one entry per merged class, naming every original class
    name folded into it and which source documents they came from. This is
    the artifact that makes the novelty claim inspectable rather than
    asserted.
  * `class_name_map` -- a deterministic Python inversion of `merge_log`
    (normalized original name -> merged canonical name). Stage 4 needs this
    to rewrite relation endpoints; it costs nothing extra to compute since
    the model already reports `merged_from` per class.

**Literal-duplicate pre-merge, added 2026-08-21 (P1-D8), fixing a real
repetition-degeneration failure.** Before this fix, the prompt sent all 192
partial schemas through untouched, one object per document, and asked the
model to concatenate their classes' attributes "as-is." A class like `Tenant`
that appears identically named across dozens of documents produced a raw
attribute union that was mostly literal repeats of the same word (`"name"`
recorded once per document). On a real full-corpus Haiku 4.5 run
(`results/findings/P1-FINDINGS.md`), asking the model to faithfully reproduce
that union sent it into an unbounded `"name", "business name", "name", ...`
loop that consumed its entire output cap without finishing even the `Tenant`
entry, let alone the rest of the corpus's classes.

`_pre_merge_literal_duplicates()` now collapses classes sharing the exact
same name (case/whitespace-folded -- not a judgment call, the same
literal-only standard `baselines.b3_single_shot.single_shot.clean_schema()`
already applies everywhere else in this project) and their exact-duplicate
attributes, in Python, before the call. This does not do any part of Stage
2's real job: it never merges two *differently*-spelled names into one, and
it changes nothing about what the model is asked to decide. It only removes
the specific, mechanical busywork -- reproducing a list that is mostly exact
repeats -- that triggered the loop. Each pre-merged entry carries an
`occurrences` count so the model can still tell which of the genuinely
different spellings was more common, which its own "choose the most frequent
wording" instruction depends on.

**Identity-only response, added 2026-08-24 (P1-D9).** P1-D8 shrank what the
model is *shown*; it did nothing about what the model is asked to *write
back*. The prompt still required every merged class to echo its full
attribute array, so a perfect, zero-merge response over the real 192-document
corpus measured ~13,000 output tokens -- 81% of haiku45's 16,000 cap before a
single merge's bookkeeping is added, and three times llama318b's entire 4,096
cap. That single number is the common cause behind all three observed Stage 2
failures (`results/findings/P1-FINDINGS.md`): the pre-P1-D8 run needed ~40,000
tokens and looped; the post-P1-D8 run needed ~13,000 and spent long enough
generating them to trip a 60-second transport read timeout; Llama could not
have fit its answer at any prompt size and wrote a Python script instead.

The attributes were never this stage's judgment -- the module docstring above
already says Stage 2 is scoped to class identity only -- so the model no
longer returns them. It returns `name`, `parent` and `merged_from`, and
`_restore_attributes()` rebuilds each merged class's attribute union in
Python from exactly the pre-merged entries the model named in `merged_from`.
That is the same deterministic-bookkeeping-over-Stage-1's-own-output move
`_index_names_by_document()` already makes for provenance, and it is
byte-identical to the concatenation rule the prompt used to spell out for the
model ("drop a literal repeat, same wording ignoring case and whitespace").
Measured on the same real corpus, the required response drops from ~13,000
tokens to ~5,950 -- 37% of haiku45's cap. Attributes are still *shown* in the
prompt, because their overlap is real evidence for whether two differently
named entries are the same kind of thing; only the echo is removed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from baselines.b3_single_shot.single_shot import clean_schema, parse_response
from baselines.shared import model_clients as mc

from pipeline.nodes._common import append_raw_record, invoke_or_abort, normalize_name
from pipeline.state import P1State

_MODULE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = _MODULE_ROOT / "prompts" / "p1_consolidation_prompt.md"
CLASSES_PLACEHOLDER = "{{CLASSES}}"


def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    template = path.read_text(encoding="utf-8")
    if CLASSES_PLACEHOLDER not in template:
        raise ValueError(f"{path} has no {CLASSES_PLACEHOLDER} placeholder")
    return template


def render_prompt(template: str, classes: list[dict]) -> str:
    """str.replace, not str.format -- the template's own JSON output example
    would otherwise be misread as format fields (same reasoning as B3's
    render_prompt and the prior single-call consolidation prompt)."""
    payload = json.dumps(classes, indent=2)
    return template.replace(CLASSES_PLACEHOLDER, payload)


def _pre_merge_literal_duplicates(class_only_schemas: list[dict]) -> list[dict]:
    """Collapses every class sharing the exact same name (case/whitespace-
    folded) across all documents into one entry, with attributes deduped the
    same way and an `occurrences` count recording how many (document, class)
    pairs fed into it. See this module's own docstring for why this is safe
    -- it is bookkeeping over an already-unambiguous literal match, not a
    merge decision between different spellings, which stays entirely the
    model's job.

    Order is preserved by first-appearance, not sorted -- so a smoke test
    against a small corpus and a full run differ only in which entries exist,
    never in a reordering that would make a diff between two runs' prompts
    harder to read than it has to be.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []

    for entry in class_only_schemas:
        for cls in entry.get("classes", []):
            if not isinstance(cls, dict):
                continue
            name = cls.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            key = normalize_name(name)
            if key not in groups:
                groups[key] = {"name": name, "parents": Counter(), "attributes": [], "_seen_attrs": set(), "occurrences": 0}
                order.append(key)
            group = groups[key]
            group["occurrences"] += 1
            parent = cls.get("parent")
            if isinstance(parent, str) and parent.strip():
                group["parents"][parent] += 1
            for attribute in cls.get("attributes") or []:
                if not isinstance(attribute, str) or not attribute.strip():
                    continue
                attr_key = normalize_name(attribute)
                if attr_key in group["_seen_attrs"]:
                    continue
                group["_seen_attrs"].add(attr_key)
                group["attributes"].append(attribute)

    merged = []
    for key in order:
        group = groups[key]
        parent = group["parents"].most_common(1)[0][0] if group["parents"] else None
        merged.append(
            {
                "name": group["name"],
                "parent": parent,
                "attributes": group["attributes"],
                "occurrences": group["occurrences"],
            }
        )
    return merged


def _restore_attributes(
    source_names: list, pre_merged_by_name: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """The attribute union of exactly the pre-merged entries named in one
    merged class's `merged_from`, in the order those names were reported,
    with literal (case/whitespace-folded) repeats dropped -- see this
    module's docstring (P1-D9) for why the model no longer writes this out
    itself.

    Also returns the source names that matched no pre-merged entry. Rule 1 of
    the prompt forbids inventing a name, so a non-empty second element means
    the model either coined a name or misspelled one it was given; either way
    that class silently loses the attributes it should have inherited. Counted
    into the raw record rather than raised on, because a single unmatched name
    among many is a quality signal about the merge, not a reason to throw away
    a call that has already been paid for.
    """
    attributes: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    for source in source_names:
        if not isinstance(source, str) or not source.strip():
            continue
        entry = pre_merged_by_name.get(normalize_name(source))
        if entry is None:
            unresolved.append(source)
            continue
        for attribute in entry["attributes"]:
            key = normalize_name(attribute)
            if key in seen:
                continue
            seen.add(key)
            attributes.append(attribute)
    return attributes, unresolved


def _index_names_by_document(class_only_schemas: list[dict]) -> dict[str, list[str]]:
    """Every raw class name, keyed by its case/whitespace-folded form, mapped
    to every source document it appeared under. Deterministic Python
    bookkeeping over Stage 1's own output -- the consolidation call is never
    asked to report this itself, since it already has to report which names
    it merged and asking it to also recall per-name provenance across a
    192-document input is exactly the kind of bookkeeping a model is more
    likely to garble than a dict built directly from the input we already
    have in hand.
    """
    index: dict[str, list[str]] = {}
    for entry in class_only_schemas:
        source_document = entry["source_document"]
        for cls in entry.get("classes", []):
            name = cls.get("name") if isinstance(cls, dict) else None
            if not isinstance(name, str) or not name.strip():
                continue
            index.setdefault(normalize_name(name), []).append(source_document)
    return index


def _build_merge_log_and_map(
    classes: list[dict], name_to_documents: dict[str, list[str]]
) -> tuple[list[dict], dict[str, str]]:
    merge_log: list[dict] = []
    name_map: dict[str, str] = {}
    for cls in classes:
        merged_name = cls["name"]
        sources = cls.get("merged_from") or [merged_name]
        source_documents = sorted({doc for name in sources for doc in name_to_documents.get(normalize_name(name), [])})
        merge_log.append({"merged_name": merged_name, "source_names": sources, "source_documents": source_documents})
        for original in sources:
            name_map[normalize_name(original)] = merged_name
        # The merged name itself must also resolve to itself, in case a later
        # stage looks up a relation endpoint already written in merged form.
        name_map[normalize_name(merged_name)] = merged_name
    return merge_log, name_map


def consolidate_types(state: P1State) -> dict:
    """LangGraph node: exactly one call, over every partial schema's classes.

    Unlike Stage 1, there is nothing else to fall back on if this call fails
    -- it raises rather than skips, the same single-call-must-abort rule B3's
    one whole-corpus call and the old pipeline's one consolidation call both
    followed.
    """
    spec = mc.MODELS[state["model_key"]]
    partial_schemas = state["partial_schemas"]
    log_path = state["raw_log_path"]
    template = load_prompt_template()

    # Stage 2 only needs classes -- relations ride along to later stages via
    # `state["partial_schemas"]` directly, not through this call.
    class_only_schemas = [
        {"source_document": entry["source_document"], "classes": entry["classes"]} for entry in partial_schemas
    ]
    # Provenance (`_index_names_by_document`, below) is computed from the
    # original, un-pre-merged `class_only_schemas` -- it needs every literal
    # (name, document) pair, which the pre-merge below deliberately collapses
    # for the *prompt* only. The two are independent by design: what the
    # model is shown and what Python already knows about where each name
    # came from are not required to be the same list.
    pre_merged_classes = _pre_merge_literal_duplicates(class_only_schemas)
    prompt = render_prompt(template, pre_merged_classes)
    print(
        f"  [2/6 consolidate_types] {len(class_only_schemas)} partial class lists "
        f"pre-merged to {len(pre_merged_classes)} literal-distinct classes, {len(prompt)} chars..."
    )

    record: dict = {
        "stage": "consolidate_types",
        "input_schema_count": len(class_only_schemas),
        "pre_merged_class_count": len(pre_merged_classes),
        "prompt_chars": len(prompt),
    }
    completion = invoke_or_abort(spec, prompt, "consolidate_types", log_path, record)
    print(f"    stop_reason={completion.stop_reason!r} completion_tokens={completion.completion_tokens}")

    try:
        parsed = parse_response(completion.text)
    except Exception:
        append_raw_record(log_path, record)
        raise

    # P1-D9: the response carries identity only. Rebuild each merged class's
    # attributes here, from the pre-merged entries it says it merged, before
    # clean_schema sees it -- so clean_schema still applies exactly the same
    # malformed-element and literal-duplicate rules to the same shape it
    # always did, and every stage downstream keeps receiving classes with
    # populated attributes as it always has.
    pre_merged_by_name = {normalize_name(c["name"]): c for c in pre_merged_classes}
    restored: list[dict] = []
    unresolved_source_names: list[str] = []
    for raw in parsed.get("classes", []):
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        sources = raw.get("merged_from")
        if not isinstance(sources, list) or not sources:
            sources = [name]
        attributes, unresolved = _restore_attributes(sources, pre_merged_by_name)
        unresolved_source_names.extend(unresolved)
        restored.append({"name": name, "parent": raw.get("parent"), "attributes": attributes, "merged_from": sources})

    cleaned = clean_schema({"classes": restored, "relations": []})
    merged_classes = cleaned["classes"]
    # clean_schema() doesn't know about merged_from -- re-attach it from the
    # pre-clean classes by matching on the cleaned name, since clean_schema
    # only drops malformed entries and dedupes attributes, it never renames a
    # class that survived.
    by_name = {c["name"]: c["merged_from"] for c in restored}
    for cls in merged_classes:
        cls["merged_from"] = by_name.get(cls["name"]) or [cls["name"]]

    name_to_documents = _index_names_by_document(class_only_schemas)
    merge_log, class_name_map = _build_merge_log_and_map(merged_classes, name_to_documents)
    record["parsed_classes"] = len(merged_classes)
    # Two coverage checks the identity-only response (P1-D9) makes worth
    # logging: before the fix, an input entry the model forgot still left its
    # attributes visible in the response text; now a forgotten entry is simply
    # absent, so it has to be counted here or it is invisible. Logged, not
    # raised on -- Stage 2 aborting on an imperfect-but-usable merge would
    # throw away a paid call, and how faithfully each model obeys the
    # account-for-every-entry rule is itself a result this project reports.
    accounted = {normalize_name(n) for cls in merged_classes for n in cls["merged_from"]}
    record["input_entries_unaccounted_for"] = sum(1 for c in pre_merged_classes if normalize_name(c["name"]) not in accounted)
    record["unresolved_source_names"] = sorted(set(unresolved_source_names))
    append_raw_record(log_path, record)

    return {
        "merged_classes": merged_classes,
        "merge_log": merge_log,
        "class_name_map": class_name_map,
        "usage": state["usage"] + [{"stage": "consolidate_types", "stop_reason": completion.stop_reason, "completion_tokens": completion.completion_tokens}],
    }
