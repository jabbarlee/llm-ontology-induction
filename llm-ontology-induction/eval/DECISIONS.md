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
