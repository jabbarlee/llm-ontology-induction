# Evaluation Design Decisions

## D1 — Sub-property flattening (affects what "the gold relation set" even is)

The TTL declares `hasParty` with sub-properties `hasOwnerParty` / `hasTenantParty`, and `represents` with `representsOwner` / `representsTenant`. Those sub-property names are an OWL implementation artifact — no model reading messy documents will ever produce the token `hasOwnerParty`.

**Decision:** flatten sub-properties to their parent label with the concrete range. The loader emits `(Lease, hasParty, Owner)` and `(Lease, hasParty, Tenant)`, not `hasOwnerParty`. Same for `represents`.

**Sanity check:** flattening yields exactly 15 relation triples, matching the 15 relations in `schema_notes.md`. If the loader produces 18 or 12, it's wrong.

## D2 — Effective vs. declared attributes (prevents a double penalty)

Gold puts `name`/`contactInfo` on `Party`; `Owner` inherits them. Suppose a model flattens the hierarchy: no `Party` class, but `name` and `contactInfo` declared directly on `Owner`, `Tenant`, `Agent`, `Vendor`.

- Scoring **declared** attributes: the model loses on taxonomy *and* loses `name`/`contactInfo` on all four classes. Two penalties, one mistake — the attribute score becomes confounded with the taxonomy score.
- Scoring **effective** (inheritance-resolved) attributes: taxonomy F1 measures hierarchy recovery, attribute F1 measures attribute recovery, independently.

**Decision:** effective attributes are the primary metric; declared is reported as a secondary column. This keeps RQ2's layer separation actually separate, which is the whole point of scoring layers apart.

## D3 — Relation direction

`(Owner, owns, Property)` vs. `(Property, ownedBy, Owner)` express the same fact with reversed direction.

**Decision:** strict direction is primary. Add an `--allow-inverse` mode reported as a robustness variant. Don't make inverse-tolerant the headline number — but having both lets us say something concrete in Error Analysis about how often direction reversal is the failure, which is a genuinely interesting result.

## D4 — Endpoint conditioning for relations

**Decision:** a relation triple matches only if both endpoints matched at the same strictness level, AND the label matches at that level. Relations whose endpoints never matched are automatically FP/FN.

This means relation scores are bounded above by class scores — expected, and should be stated in the paper rather than discovered by a reviewer.

## D5 — M3 semantic matching: embeddings, not LLM-as-judge

The roadmap suggests Claude Sonnet as an LLM judge. Real conflict: Claude Sonnet is also one of the three tested models (Step 5 B3, Step 6 P1). A judge scoring its own outputs alongside competitors' is a bias risk a reviewer will catch immediately, and it's the kind of thing that sinks credibility on an otherwise clean design.

**Decision:** sentence-transformers (`all-MiniLM-L6-v2`) locally as primary M3.

- Deterministic and reproducible — critical for an evaluation instrument
- Free, no API, preserves the $128 AWS credits for the actual experiment
- No model-family bias toward any contestant

If an LLM-judge sensitivity check is wanted, use a model outside the tested set, run it on a sample, and report it as a robustness appendix — never as the primary metric. Keep the roadmap's "manually check ~100 judgments" discipline either way: sample 100 M3 match decisions and hand-verify them, reporting the agreement rate as a harness-validity number in the paper.

## D6 — Degenerate-case conventions

- Empty induced schema → precision = 0.0 (not NaN), recall = 0.0, F1 = 0.0
- Empty gold layer (shouldn't happen, but) → skip layer, mark n/a
- Report the raw TP/FP/FN counts alongside every P/R/F1 so any convention choice is auditable rather than baked invisibly into a number

## Addendum (2026-08-07) — class/attribute count correction

`schema/schema_notes.md`'s "Totals" line and this project's own execution
plan (`eval/PLAN.md` §5/§9/§6) stated "10 classes, 27 attributes" — stale
relative to `schema/gold_schema.ttl` itself, which declares 11 classes and
30 attributes (and always did; `schema_notes.md`'s own attribute table
always listed 30 rows). Caught while building `eval/schema_ir.py`'s
loader and cross-checking its output against the documented totals.
Resolution: the harness derives class/attribute/relation counts directly
from `load_gold_ttl()`, never from hardcoded literals, so this discrepancy
doesn't affect scoring correctness — but the stale prose numbers in
`schema_notes.md` and `eval/PLAN.md` are corrected to 11/30 for
consistency. The relation count (15, after D1 flattening) was already
correct and needed no change.

## Addendum (2026-08-07) — M2 token-containment bug found while building the T4/T9 toy fixtures

While building the toy validation suite against the actual gold schema, `rapidfuzz.fuzz.token_set_ratio` alone was found to score a **perfect** 100/100 for pure token-containment pairs — e.g. a class name that's a strict token-superset of another's (this gold schema's own Property/OfficeProperty/RetailProperty/IndustrialProperty taxonomy is exactly this shape). That tied a superclass's self-match against its own subclass's match at M2, letting the T4 fixture's bipartite class assignment legally swap them — class-layer P/R/F1 looked fine in aggregate, but the *specific* pairing corrupted the taxonomy/attribute/relation layers downstream. Fixed in `matching.py::m2_score` by scoring `min(token_set_ratio, token_sort_ratio)` instead of `token_set_ratio` alone — `token_sort_ratio` doesn't inflate on pure containment, so the tie is broken, while genuine multi-word synonyms/reorderings (e.g. `landlord`/`owner`) are unaffected.

**Consequence for D1 / T9 (sub-property literalism):** the fix narrows what M2 can be relied on to catch. A literal OWL sub-property name (e.g. the un-flattened form of `hasParty`) is *also* a token-superset of its parent label — structurally the identical shape as the Property/OfficeProperty bug just fixed. Verified numerically: no single M2 threshold can simultaneously (a) reject the `lease`/`least` near-spelling false positive (raw score 80) and (b) accept all four T9 sub-property-literal pairs (raw scores 72–77) — the ranges don't overlap. Recovering D1-flattened sub-property literalism at M1/M2/M3 is therefore **guaranteed only at M3 (semantic embeddings)**, not M2 — M2 is pure surface-string similarity and cannot safely distinguish "this is a specialization of that" from "this is a different-but-similarly-named class" using string shape alone; that distinction requires either the domain lexicon (which doesn't cover OWL implementation artifacts) or actual semantic understanding. `eval/tests/test_harness.py::test_t9_subproperty_literalism` asserts M1 failure and M3 recovery accordingly, and does not assert M2 recovery.
