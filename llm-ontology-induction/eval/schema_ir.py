"""
Step 4 — Evaluation Harness: Canonical Intermediate Representation + loaders.

Defines the abstract Schema IR that every downstream module (matching.py,
metrics.py, report.py) operates on. Both the gold TTL and any induced-schema
JSON (the contract in eval/PLAN.md §2) are parsed into this same shape, so
scoring code never needs to know where a Schema came from.

Domain-agnostic per eval/PLAN.md §0 — this file must contain zero
domain-specific class/attribute/relation names from any concrete gold
schema. See eval/tests/test_harness.py::test_no_domain_leakage for the
enforced check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS
from rdflib.term import URIRef


# ---------------------------------------------------------------------------
# Canonical IR (eval/PLAN.md §4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassDef:
    name: str
    parent: str | None
    declared_attributes: frozenset[str]


@dataclass(frozen=True)
class Relation:
    source: str
    label: str
    target: str


@dataclass(frozen=True)
class Schema:
    classes: dict[str, ClassDef]
    relations: frozenset[Relation]


def effective_attributes(schema: Schema, class_name: str) -> frozenset[str]:
    """Attributes declared on `class_name` plus everything inherited by
    walking the parent chain (D2 — effective vs. declared attributes).

    A free function rather than a ClassDef method: it needs to run
    identically over both gold and induced schemas, and keeping it at the
    Schema level (not the ClassDef level) makes that symmetry explicit
    instead of duplicated.

    Tolerant of a dangling/unresolvable `parent` reference (a class whose
    `parent` string doesn't appear as a key in `schema.classes` — a real
    case, not hypothetical: PLAN.md's own §2 JSON contract example has a
    class whose declared parent is never itself declared as a class) and of
    parent cycles — both simply stop the walk rather than raising.
    """
    attrs: set[str] = set()
    seen: set[str] = set()
    current = class_name
    while current is not None and current not in seen:
        seen.add(current)
        cls = schema.classes.get(current)
        if cls is None:
            break
        attrs |= cls.declared_attributes
        current = cls.parent
    return frozenset(attrs)


# ---------------------------------------------------------------------------
# Gold TTL loader
# ---------------------------------------------------------------------------

def _local_name(uri: URIRef) -> str:
    """Strip the namespace off a URIRef, returning just the fragment/local
    name (e.g. http://example.org/onto#Foo -> "Foo")."""
    s = str(uri)
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def load_gold_ttl(path: str | Path) -> Schema:
    """Parse a gold-standard OWL/Turtle schema into the canonical IR.

    Implements D1 (sub-property flattening): a role-specific sub-property
    (rdfs:subPropertyOf some general object property) is emitted under its
    *parent's* label with the sub-property's own concrete range, not under
    its own literal name — no model reading messy documents will ever
    produce an OWL implementation artifact like a role-specific
    sub-property name verbatim. See eval/DECISIONS.md D1 for the concrete
    domain example this generalizes.
    """
    g = rdflib.Graph()
    g.parse(str(path), format="turtle")

    # --- classes + rdfs:subClassOf -----------------------------------
    class_names: set[str] = set()
    parent_of: dict[str, str | None] = {}
    for cls_uri in g.subjects(RDF.type, OWL.Class):
        name = _local_name(cls_uri)
        class_names.add(name)
        parent_uri = next(g.objects(cls_uri, RDFS.subClassOf), None)
        parent_of[name] = _local_name(parent_uri) if parent_uri is not None else None

    # --- datatype properties -> declared attributes per class --------
    declared_attrs: dict[str, set[str]] = {name: set() for name in class_names}
    for prop_uri in g.subjects(RDF.type, OWL.DatatypeProperty):
        prop_name = _local_name(prop_uri)
        domain_uri = next(g.objects(prop_uri, RDFS.domain), None)
        if domain_uri is None:
            continue
        domain_name = _local_name(domain_uri)
        declared_attrs.setdefault(domain_name, set()).add(prop_name)

    classes = {
        name: ClassDef(
            name=name,
            parent=parent_of.get(name),
            declared_attributes=frozenset(declared_attrs.get(name, set())),
        )
        for name in class_names
    }

    # --- object properties -> relations, with D1 flattening ----------
    subproperty_of: dict[str, str] = {}
    for child_uri, parent_uri in g.subject_objects(RDFS.subPropertyOf):
        subproperty_of[_local_name(child_uri)] = _local_name(parent_uri)

    # The set of values in subproperty_of are the generic super-properties
    # (e.g. "hasParty", "represents") — they carry no concrete range of
    # their own worth emitting a triple for; only their sub-properties do.
    generic_superproperties = set(subproperty_of.values())

    relations: set[Relation] = set()
    for prop_uri in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_name = _local_name(prop_uri)
        if prop_name in generic_superproperties:
            continue
        domain_uri = next(g.objects(prop_uri, RDFS.domain), None)
        range_uri = next(g.objects(prop_uri, RDFS.range), None)
        if domain_uri is None or range_uri is None:
            continue
        label = subproperty_of.get(prop_name, prop_name)
        relations.add(
            Relation(
                source=_local_name(domain_uri),
                label=label,
                target=_local_name(range_uri),
            )
        )

    schema = Schema(classes=classes, relations=frozenset(relations))

    # D1 tripwire — hard sanity check at load time, not just in a test.
    if len(schema.relations) != 15:
        raise ValueError(
            f"D1 flattening sanity check failed: expected 15 relation "
            f"triples after sub-property flattening, got {len(schema.relations)}. "
            f"Relations: {sorted((r.source, r.label, r.target) for r in schema.relations)}"
        )

    return schema


# ---------------------------------------------------------------------------
# Induced-schema JSON loader (eval/PLAN.md §2 contract)
# ---------------------------------------------------------------------------

def parse_induced_schema(data: dict) -> Schema:
    """Pure parse of an already-loaded induced-schema dict into the IR.

    Split from load_induced_json() so it's directly unit-testable without
    touching disk, and reusable by the toy-fixture generator.

    Hard rule (PLAN.md §2): the harness normalizes, the pipeline never
    pre-cleans. This function must NOT lowercase, strip, or otherwise
    normalize class/attribute/relation names — that would hide exactly the
    messiness the harness is meant to measure. Normalization happens only
    at match time, in matching.py.
    """
    classes: dict[str, ClassDef] = {}
    for c in data.get("classes", []):
        name = c["name"]
        parent = c.get("parent")  # already None for JSON null
        attrs = frozenset(c.get("attributes", []))
        classes[name] = ClassDef(name=name, parent=parent, declared_attributes=attrs)

    relations = frozenset(
        Relation(source=r["source"], label=r["label"], target=r["target"])
        for r in data.get("relations", [])
    )

    return Schema(classes=classes, relations=relations)


def load_induced_json(path: str | Path) -> Schema:
    """Load an induced-schema JSON file (the Step 5-7 output contract) into
    the canonical IR."""
    data = json.loads(Path(path).read_text())
    return parse_induced_schema(data)


def load_induced_metadata(path: str | Path) -> dict:
    """Load just the `metadata` block of an induced-schema JSON file.

    Kept separate from Schema/parse_induced_schema on purpose: metadata
    (condition, model, run_id, source_documents) is carried through to
    results by report.py but never scored, so Schema and every scoring
    function in metrics.py stay metadata-agnostic.
    """
    data = json.loads(Path(path).read_text())
    return data.get("metadata", {})
