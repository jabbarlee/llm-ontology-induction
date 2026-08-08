"""
Toy validation suite fixture builders (eval/PLAN.md §6, T1/T3-T9). Each
function mechanically derives an induced-JSON-contract-shaped dict (the
eval/PLAN.md §2 format) from the *loaded gold Schema*, rather than being
hand-authored -- avoids hand-typing ~30-attribute JSON across several files
(error-prone, drifts silently if the gold schema is ever edited) while still
producing static, browsable fixture files on disk (run this module's
`generate_all()` to write them).

This file is explicitly domain-specific test-fixture code, not core engine
code -- it is NOT subject to eval/tests/test_harness.py::test_no_domain_leakage
(that check only covers schema_ir.py/matching.py/metrics.py).

T2 (empty induced schema) has no gold-derived content and is hand-written
directly as a static JSON file instead of via a builder here.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.schema_ir import Schema, load_gold_ttl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GOLD_TTL_PATH = REPO_ROOT / "schema" / "gold_schema.ttl"
FIXTURES_DIR = Path(__file__).resolve().parent


def _metadata(condition: str) -> dict:
    return {
        "condition": condition,
        "model": "toy-fixture",
        "run_id": f"toy-{condition}",
        "source_documents": [],
    }


def schema_to_induced_dict(schema: Schema, condition: str) -> dict:
    """The inverse of parse_induced_schema() -- serializes a Schema back
    into the induced-JSON-contract shape, for fixtures built by mutating a
    loaded gold Schema rather than typed by hand."""
    classes = [
        {
            "name": name,
            "parent": c.parent,
            "attributes": sorted(c.declared_attributes),
        }
        for name, c in schema.classes.items()
    ]
    relations = [
        {"source": r.source, "label": r.label, "target": r.target} for r in schema.relations
    ]
    return {"classes": classes, "relations": relations, "metadata": _metadata(condition)}


# ---------------------------------------------------------------------------
# T1 — identity
# ---------------------------------------------------------------------------

def build_t1_identity(gold: Schema) -> dict:
    induced = Schema(classes=dict(gold.classes), relations=frozenset(gold.relations))
    return schema_to_induced_dict(induced, "T1")


# ---------------------------------------------------------------------------
# T3 — perfect rename (every class with a lexicon synonym gets renamed to
# it; classes without one -- Party and the three Property subtypes, which
# have no single-token CRE synonym -- are left identical). Must stay in
# sync with the `synonyms` block in eval/config/cre.yaml.
# ---------------------------------------------------------------------------

T3_RENAME_MAP = {
    "Owner": "Landlord",
    "Tenant": "Lessee",
    "Property": "Premises",
    "Agent": "Broker",
    "Vendor": "Contractor",
    "Lease": "Rental Agreement",
    "MaintenanceRequest": "Work Order",
}


def build_t3_perfect_rename(gold: Schema) -> dict:
    def renamed(name: str | None) -> str | None:
        if name is None:
            return None
        return T3_RENAME_MAP.get(name, name)

    classes = [
        {
            "name": renamed(name),
            "parent": renamed(c.parent),
            "attributes": sorted(c.declared_attributes),
        }
        for name, c in gold.classes.items()
    ]
    relations = [
        {"source": renamed(r.source), "label": r.label, "target": renamed(r.target)}
        for r in gold.relations
    ]
    return {"classes": classes, "relations": relations, "metadata": _metadata("T3")}


# ---------------------------------------------------------------------------
# T4 — over-generation: gold + 5 invented classes with no synonym relation
# to anything in gold (verified distinct at M1/M2/M3 by the acceptance test).
# ---------------------------------------------------------------------------

T4_INVENTED_CLASSES = [
    "Insurance Policy",
    "Zoning Permit",
    "Utility Account",
    "Parking Permit",
    "Signage Placard",
]


def build_t4_overgeneration(gold: Schema) -> dict:
    base = schema_to_induced_dict(
        Schema(classes=dict(gold.classes), relations=frozenset(gold.relations)), "T4"
    )
    for name in T4_INVENTED_CLASSES:
        base["classes"].append({"name": name, "parent": None, "attributes": []})
    return base


# ---------------------------------------------------------------------------
# T5 — under-generation: half the gold classes (rounded down), sorted for
# determinism. Relations are filtered to only those whose endpoints both
# survive, so the fixture stays internally sensible.
# ---------------------------------------------------------------------------

def build_t5_undergeneration(gold: Schema) -> dict:
    kept_names = sorted(gold.classes)[: len(gold.classes) // 2]
    kept = {name: gold.classes[name] for name in kept_names}
    relations = frozenset(
        r for r in gold.relations if r.source in kept and r.target in kept
    )
    induced = Schema(classes=kept, relations=relations)
    return schema_to_induced_dict(induced, "T5")


# ---------------------------------------------------------------------------
# T6 — flattened taxonomy: every class present, every `parent` set to null.
# ---------------------------------------------------------------------------

def build_t6_flattened_taxonomy(gold: Schema) -> dict:
    classes = [
        {"name": name, "parent": None, "attributes": sorted(c.declared_attributes)}
        for name, c in gold.classes.items()
    ]
    relations = [
        {"source": r.source, "label": r.label, "target": r.target} for r in gold.relations
    ]
    return {"classes": classes, "relations": relations, "metadata": _metadata("T6")}


# ---------------------------------------------------------------------------
# T7 — split class: gold's Tenant class becomes two induced classes,
# "Renter" and "Lessee" (both lexicon synonyms of "tenant" per cre.yaml),
# each carrying Tenant's own declared attributes.
# ---------------------------------------------------------------------------

T7_SPLIT_SOURCE = "Tenant"
T7_SPLIT_INTO = ["Renter", "Lessee"]


def build_t7_split_class(gold: Schema) -> dict:
    split_attrs = sorted(gold.classes[T7_SPLIT_SOURCE].declared_attributes)
    split_parent = gold.classes[T7_SPLIT_SOURCE].parent

    classes = [
        {"name": name, "parent": c.parent, "attributes": sorted(c.declared_attributes)}
        for name, c in gold.classes.items()
        if name != T7_SPLIT_SOURCE
    ]
    for new_name in T7_SPLIT_INTO:
        classes.append({"name": new_name, "parent": split_parent, "attributes": split_attrs})

    relations = [
        {"source": r.source, "label": r.label, "target": r.target} for r in gold.relations
    ]
    return {"classes": classes, "relations": relations, "metadata": _metadata("T7")}


# ---------------------------------------------------------------------------
# T8 — reversed relations: every triple's source/target swapped, label kept.
# ---------------------------------------------------------------------------

def build_t8_reversed_relations(gold: Schema) -> dict:
    classes = [
        {"name": name, "parent": c.parent, "attributes": sorted(c.declared_attributes)}
        for name, c in gold.classes.items()
    ]
    relations = [
        {"source": r.target, "label": r.label, "target": r.source} for r in gold.relations
    ]
    return {"classes": classes, "relations": relations, "metadata": _metadata("T8")}


# ---------------------------------------------------------------------------
# T9 — sub-property literalism: the D1-flattened "hasParty"/"represents"
# edges are emitted under their pre-flattening OWL sub-property names
# (verbatim, as they appear in schema/gold_schema.ttl) instead of gold's
# flattened parent label.
# ---------------------------------------------------------------------------

_T9_LITERAL_LABELS = {
    ("hasParty", "Owner"): "hasOwnerParty",
    ("hasParty", "Tenant"): "hasTenantParty",
    ("represents", "Owner"): "representsOwner",
    ("represents", "Tenant"): "representsTenant",
}


def build_t9_subproperty_literalism(gold: Schema) -> dict:
    classes = [
        {"name": name, "parent": c.parent, "attributes": sorted(c.declared_attributes)}
        for name, c in gold.classes.items()
    ]
    relations = []
    for r in gold.relations:
        literal_label = _T9_LITERAL_LABELS.get((r.label, r.target), r.label)
        relations.append({"source": r.source, "label": literal_label, "target": r.target})
    return {"classes": classes, "relations": relations, "metadata": _metadata("T9")}


# ---------------------------------------------------------------------------
# Generate + write all fixtures
# ---------------------------------------------------------------------------

BUILDERS = {
    "t1_identity": build_t1_identity,
    "t3_perfect_rename": build_t3_perfect_rename,
    "t4_overgeneration": build_t4_overgeneration,
    "t5_undergeneration": build_t5_undergeneration,
    "t6_flattened_taxonomy": build_t6_flattened_taxonomy,
    "t7_split_class": build_t7_split_class,
    "t8_reversed_relations": build_t8_reversed_relations,
    "t9_subproperty_literalism": build_t9_subproperty_literalism,
}


def generate_all() -> None:
    gold = load_gold_ttl(GOLD_TTL_PATH)
    for filename_stem, builder in BUILDERS.items():
        data = builder(gold)
        out_path = FIXTURES_DIR / f"{filename_stem}.json"
        out_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    generate_all()
