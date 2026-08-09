# Step 4 — Evaluation Harness: Execution Plan

This is the complete plan for the chapter. It answers the architecture question,
locks the design decisions, defines the interfaces other steps depend on, and
gives the phased build order with a validation suite that has hand-computable
expected values.

---

## 0. The core question: reusable or per-domain?

**The harness is domain-agnostic. The domain lives in a small config file and
the gold schema itself.** Three layers:

| Layer | What's in it | Domain-specific? | Approx. share of code |
|---|---|---|---|
| **1. Core engine** | Schema IR, TTL/JSON loaders, matchers, bipartite assignment, metrics, reporting, CLI | **No** — operates on abstract classes/attributes/relations | ~95% |
| **2. Domain pack** | Synonym lexicon, stopwords, frozen thresholds | **Yes** — one YAML file | ~5% |
| **3. Domain data** | `gold_schema.ttl`, toy validation schemas | **Yes** — inputs, not code | n/a |

### Why this matters beyond tidiness

- **It's a paper contribution.** "We release a domain-agnostic ontology-induction
  evaluation harness" is a stronger artifact claim than "we wrote scoring code
  for our experiment." It's also what makes the Zenodo release worth citing.
- **It's how the future-work Corpus A path stays open.** Your execution plan
  descoped the real-legal-corpus (Leivaditi/CUAD) experiment to future work. If
  the harness is domain-agnostic, that future work needs a new gold schema and a
  new synonym file — not a rewrite of the evaluation.
- **It forces the right abstraction now.** If the core engine ever contains the
  string `"Tenant"`, something has leaked into the wrong layer. That's a useful
  tripwire while building.

The one genuinely domain-bound thing is the **synonym lexicon** for M2 (fuzzy
matching): `landlord ≈ owner`, `lessee ≈ tenant`, `premises ≈ property`,
`work order ≈ maintenance request`. These can't be derived generically, and
hardcoding them in `matching.py` would be the leak described above.

---

## 1. Design decisions to lock before writing code

Each of these has a recommendation. Lock them, write them into
`eval/DECISIONS.md`, and don't revisit after experiments start.

### D1 — Sub-property flattening (affects what "the gold relation set" even is)

Your TTL declares `hasParty` with sub-properties `hasOwnerParty` /
`hasTenantParty`, and `represents` with `representsOwner` / `representsTenant`.
Those sub-property names are an OWL implementation artifact — no model reading
messy documents will ever produce the token `hasOwnerParty`.

**Decision: flatten sub-properties to their parent label with the concrete
range.** The loader emits `(Lease, hasParty, Owner)` and
`(Lease, hasParty, Tenant)`, not `hasOwnerParty`. Same for `represents`.

Sanity check that this is right: flattening yields exactly **15 relation
triples**, matching the 15 relations in `schema_notes.md`. If your loader
produces 18 or 12, it's wrong.

### D2 — Effective vs. declared attributes (prevents a double penalty)

Gold puts `name`/`contactInfo` on `Party`; `Owner` inherits them. Suppose a
model flattens the hierarchy: no `Party` class, but `name` and `contactInfo`
declared directly on `Owner`, `Tenant`, `Agent`, `Vendor`.

- Scoring **declared** attributes: the model loses on taxonomy *and* loses
  `name`/`contactInfo` on all four classes. Two penalties, one mistake — the
  attribute score becomes confounded with the taxonomy score.
- Scoring **effective** (inheritance-resolved) attributes: taxonomy F1 measures
  hierarchy recovery, attribute F1 measures attribute recovery, independently.

**Decision: effective attributes are the primary metric; declared is reported
as a secondary column.** This keeps RQ2's layer separation actually separate,
which is the whole point of scoring layers apart.

### D3 — Relation direction

`(Owner, owns, Property)` vs. `(Property, ownedBy, Owner)` express the same
fact with reversed direction.

**Decision: strict direction is primary. Add an `--allow-inverse` mode reported
as a robustness variant.** Don't make inverse-tolerant the headline number — but
having both lets you say something concrete in Error Analysis about how often
direction reversal is the failure, which is a genuinely interesting result.

### D4 — Endpoint conditioning for relations

**Decision: a relation triple matches only if both endpoints matched at the same
strictness level, AND the label matches at that level.** Relations whose
endpoints never matched are automatically FP/FN. Document this — it means
relation scores are bounded above by class scores, which is expected and should
be stated in the paper rather than discovered by a reviewer.

### D5 — M3 semantic matching: embeddings, not LLM-as-judge

Your roadmap suggests Claude Sonnet as an LLM judge. **Flagging a real conflict:
Claude Sonnet is also one of your three tested models** (Step 5 B3, Step 6 P1).
A judge scoring its own outputs alongside competitors' is a bias risk a reviewer
will catch immediately, and it's the kind of thing that sinks credibility on an
otherwise clean design.

**Decision: `sentence-transformers` (`all-MiniLM-L6-v2`) locally as primary M3.**
- Deterministic and reproducible — critical for an evaluation instrument
- Free, no API, preserves your $128 AWS credits for the actual experiment
- No model-family bias toward any contestant

If you want an LLM-judge sensitivity check, use a model **outside the tested
set**, run it on a sample, and report it as a robustness appendix — never as the
primary metric. Keep the roadmap's "manually check ~100 judgments" discipline
either way: sample 100 M3 match decisions and hand-verify them, reporting the
agreement rate as a harness-validity number in the paper.

### D6 — Degenerate-case conventions

- Empty induced schema → precision = 0.0 (not NaN), recall = 0.0, F1 = 0.0
- Empty gold layer (shouldn't happen, but) → skip layer, mark `n/a`
- Report the raw TP/FP/FN counts alongside every P/R/F1 so any convention
  choice is auditable rather than baked invisibly into a number

---

## 2. The induced-schema contract (unblocks Steps 5–7)

Defining this now is the highest-leverage thing in this chapter: it's the
interface every baseline and pipeline stage must emit. Without it, Step 5/6 code
gets written against an imaginary format and you get a painful adapter layer.

```json
{
  "classes": [
    {
      "name": "Building",
      "parent": "Asset",
      "attributes": ["street address", "size_sqft", "building type"]
    }
  ],
  "relations": [
    { "source": "Landlord", "label": "owns", "target": "Building" }
  ],
  "metadata": {
    "condition": "P1",
    "model": "claude-sonnet-4-6",
    "run_id": "2026-08-07T14:22:11Z-a3f9",
    "source_documents": ["notes/mr-001.txt", "csv_exports/leases_export_01.csv"]
  }
}
```

Rules: `parent` is `null` for root classes. Names are free-text as the model
produced them — **the harness normalizes, the pipeline never pre-cleans**
(pre-cleaning would hide exactly the messiness you're measuring). `metadata` is
carried through to results but never scored.

---

## 3. File layout

```
eval/
├── DECISIONS.md              # §1 decisions, frozen, dated
├── schema_ir.py              # canonical IR + TTL loader + induced-JSON loader
├── matching.py               # M1/M2/M3 matchers + bipartite assignment
├── metrics.py                # P/R/F1 per layer, aggregation
├── report.py                 # CLI entry, table/CSV output to results/
├── error_analysis.py         # Step 8 — stub now, filled later
├── config/
│   └── cre.yaml              # THE domain pack: synonyms, stopwords, thresholds
└── tests/
    ├── toy_schemas/          # §6 fixtures
    └── test_harness.py       # assertions with hand-computed expected values
```

This adds `schema_ir.py`, `report.py`, `config/`, and `tests/` beyond the
roadmap's `matching.py` / `metrics.py` / `error_analysis.py`. The deviation is
deliberate: the IR is what keeps loaders separate from scoring, and the tests
directory *is* the roadmap's "run it against toy schemas" check, made
executable instead of manual.

---

## 4. Canonical IR

```python
@dataclass(frozen=True)
class ClassDef:
    name: str
    parent: str | None
    declared_attributes: frozenset[str]
    # effective_attributes computed by walking parent chain (D2)

@dataclass(frozen=True)
class Relation:
    source: str
    label: str
    target: str

@dataclass(frozen=True)
class Schema:
    classes: dict[str, ClassDef]
    relations: frozenset[Relation]
```

Both `load_gold_ttl()` and `load_induced_json()` return this. Every downstream
function takes `Schema` and knows nothing about where it came from — that's the
property that makes the engine reusable.

---

## 5. Phased build order

Build in this order; each phase is testable before the next exists.

**Phase 1 — IR + loaders.** `schema_ir.py`. Acceptance: loading
`gold_schema.ttl` yields 11 classes, 2 taxonomies, 30 attributes, **15
relations** (D1 check), and `Owner.effective_attributes` includes inherited
`name`/`contactInfo` (D2 check). *(Corrected 2026-08-07 from a stale "10
classes, 27 attributes" — see eval/DECISIONS.md's addendum. The harness
itself never hardcodes these counts; it derives them from the loader.)*

**Phase 2 — Normalization + M1.** camelCase/snake_case splitting, lowercasing,
punctuation strip, singularization (use `inflect`, not a hand-rolled rule — 
`Property → propertie` is the classic bug). Acceptance:
`MaintenanceRequest` ≡ `maintenance_request` ≡ `Maintenance Requests`.

**Phase 3 — M2 fuzzy.** Token-set similarity (`rapidfuzz`) + synonym lexicon
lookup from `config/cre.yaml`. Acceptance: `landlord` matches `owner` via
lexicon; `lease` does **not** match `least` despite high edit similarity
(threshold + token-set, not raw Levenshtein).

**Phase 4 — M3 semantic.** Embedding cosine via `sentence-transformers`,
cached to disk by string so repeated runs are instant and deterministic.
Acceptance: `premises` ↔ `property` scores above threshold; `vendor` ↔ `tenant`
scores below it.

**Phase 5 — Bipartite assignment.** *Do not greedily match.* If induced has both
`Renter` and `Lessee`, both plausibly match gold `Tenant`; greedy matching
double-counts and inflates recall. Use
`scipy.optimize.linear_sum_assignment` over the similarity matrix for a
one-to-one optimal assignment, per strictness level. Acceptance: the split-class
toy case (§6) scores exactly 1 TP + 1 FP, never 2 TP.

**Phase 6 — Metrics.** Four layers scored separately (RQ2):
1. **Classes** — P/R/F1 on the class set
2. **Taxonomy** — P/R/F1 on `(child, parent)` edges, conditioned on both
   endpoints matching
3. **Attributes** — per matched class, then **both** micro (pooled
   `(class, attr)` pairs) and macro (mean of per-class F1). Report both: your
   `Lease` has 5 attributes and `Party` has 2, so the two will diverge and the
   divergence is informative.
4. **Relations** — P/R/F1 on triples, per D4

**Phase 7 — Reporting/CLI.**
`python -m eval.report --gold schema/gold_schema.ttl --induced results/raw/<run>.json --level all`
→ writes a tidy row per (run, level, layer) into `results/tables_figures/`.
Long-format output, one row per metric — makes the paper's tables a groupby
rather than a reshaping exercise.

**Phase 8 — Freeze.** Tag the commit. Write `DECISIONS.md`. After this point,
harness changes require a documented reason and a re-run of everything already
scored.

---

## 6. Toy validation suite — "trust the ruler before you use it"

Each fixture has a hand-computable expected score. These are unit-test
assertions, not eyeballing.

| # | Fixture | Expected |
|---|---|---|
| T1 | **Identity** — induced == gold | All layers P=R=F1=1.0 at every level |
| T2 | **Empty** — `{"classes":[],"relations":[]}` | All 0.0, no crash (D6) |
| T3 | **Perfect rename** — every class replaced by a lexicon synonym (`Owner→Landlord`, `Tenant→Lessee`, `Property→Premises`) | M1 near 0, M2 and M3 near 1.0. *This is the test that proves the levels are actually different from each other.* |
| T4 | **Over-generation** — gold + 5 invented classes | Recall = 1.0; precision = 11/16 = 0.6875 |
| T5 | **Under-generation** — 5 of 11 gold classes (half, rounded down) | Precision = 1.0; recall = 5/11 = 0.4545; F1 = 0.625 |
| T6 | **Flattened taxonomy** — all 11 classes, zero `parent` set | Class F1 = 1.0, **taxonomy F1 = 0.0**, attribute F1 still high under effective scoring (D2): hand-derived TP=30/FP=0/FN=14 → F1 ≈ 0.811 |
| T7 | **Split class** — gold `Tenant` → induced `Renter` + `Lessee` | At M2/M3 (where the lexicon recognizes both as synonyms): 1 TP + 1 FP, never 2 TP (Phase 5 assignment check). At M1 the pair simply doesn't exact-match at all (0 TP, 2 FP, gold class FN) — also correct, just a different shape. |
| T8 | **Reversed relations** — all 15 triples direction-flipped | Relation F1 = 1/15 ≈ 0.067 strict (not exactly 0.0 — `hasAmendment` is self-referential, source==target, so it's a fixed point under reversal and still matches); 1.0 with `--allow-inverse` (D3) |
| T9 | **Sub-property literalism** — induced emits `hasOwnerParty` verbatim | Confirms D1 flattening: fails to match gold's `(Lease, hasParty, Owner)` at M1; recovers at **M3**. NOT guaranteed at M2 — see eval/DECISIONS.md's M2 token-containment addendum: the same fix required to stop M2 from conflating a class with its own compound-named subtype (e.g. a `Foo`/`OfficeFoo` pair) also caps how much containment-leniency M2 can safely apply to a sub-property label being a superset of its parent label. |

T3, T6, T7, and T8 are the load-bearing ones — they're the cases where a
plausible-looking harness silently gives wrong numbers. Building T4 against
the real gold schema also surfaced a real M2 scoring bug (fixed) — see
eval/DECISIONS.md's 2026-08-07 addenda for both that fix and the class/
attribute count correction below.

---

## 7. Threshold freeze protocol (the p-hacking guard)

M2 and M3 need numeric thresholds. Tuning them after seeing pipeline results is
p-hacking, even unintentionally — and it's the single most likely
methodological criticism of this chapter.

1. Hand-label ~60–80 term pairs from the domain as match / no-match
   (`landlord`/`owner` = match; `vendor`/`tenant` = no-match). Do this from the
   **gold schema vocabulary and plausible synonyms only** — not from any model
   output.
2. Pick the threshold maximizing agreement with your labels.
3. Write the value into `config/cre.yaml` with the date and the labeled set
   committed alongside.
4. **Never change it again.** If you must, re-run every previously scored result
   and say so in Limitations.

Do this *before* Step 5 produces its first baseline output.

---

## 8. Known pitfalls

- **Greedy matching** (Phase 5) — inflates recall, hardest bug to notice because
  the numbers look plausible.
- **Singularization** — `inflect`, not string-slicing. `Property`/`Properties`
  and `Premises` (already plural) both break naive rules.
- **Case-only differences in the assignment matrix** — normalize once, up front,
  and match on normalized forms throughout; don't normalize inside each matcher.
- **Silent layer coupling** — if relation F1 mysteriously tracks class F1
  exactly, D4 conditioning may be misimplemented.
- **Embedding cache invalidation** — key the cache on
  `(model_name, model_version, string)`. A silently upgraded
  `sentence-transformers` model would change scores mid-experiment.
- **Scoring your own gold against itself as the only test** — T1 passing proves
  almost nothing on its own; T3/T6/T7/T8 are where real bugs surface.

---

## 9. Definition of done

- [x] `DECISIONS.md` written and dated, D1–D6 recorded (plus two 2026-08-07
      addenda: the class/attribute count correction and the M2
      token-containment fix, both found while building the toy suite)
- [x] Loading `gold_schema.ttl` yields 11 classes / 30 attributes / 15
      relations (corrected from a stale "10/27" — see the addenda above;
      the harness itself derives these from the loader, never hardcodes them)
- [x] All 9 toy fixtures pass with hand-computed expected values
- [ ] Thresholds frozen in `config/cre.yaml`, labeled pair-set committed
      (still provisional/placeholder — §7's hand-labeling exercise is
      separate, out-of-band future work, not done as part of this build)
- [ ] 100 sampled M3 decisions hand-verified, agreement rate recorded
      (deferred to Step 8 / `error_analysis.py`, currently a stub)
- [x] CLI produces long-format output into `results/tables_figures/`
- [x] Zero domain strings (`Tenant`, `Lease`, …) anywhere in `matching.py`,
      `metrics.py`, `schema_ir.py` — grep for them as a literal check
      (enforced automatically by `test_no_domain_leakage`)
- [ ] Harness commit tagged; Dr. Oncu has seen the toy-suite results

Only then start Step 5. The harness is the instrument every number in the paper
depends on — a bug found here costs an afternoon, the same bug found in Step 7
costs every result you've generated.