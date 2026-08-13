## B3 — Single-shot LLM baseline: Llama 3.1 8B (via Groq)

**Run ID:** `2026-08-12T17:53:58Z-83cb`
**Scored:** full corpus, 28 batches of 7 documents each (uniform batch size —
see note below on batching, since this specific run predates the
decoupled-batching decision)
**Verified:** loaded cleanly through `eval/schema_ir.py`'s real
`load_induced_json()`/`parse_induced_schema()` before scoring — contract
genuinely satisfied, not just visually inspected.

### Scores

| Level | Classes F1 | Taxonomy F1 | Attributes F1 (eff.) | Relations F1 |
|---|---|---|---|---|
| M1 (exact) | 0.179 | 0.077 | 0.089 | 0.000 |
| M2 (fuzzy+lexicon) | 0.205 | 0.000 | 0.073 | 0.027 |
| M3 (semantic) | 0.231 | 0.077 | 0.179 | 0.009 |

Class-level detail: TP 7→8→9, FP 60→59→58, FN 4→3→2 across M1→M3, against
11 true gold classes. 67 induced classes total.

### Headline, surprising result: B1 beats B3-Llama on class F1, at every level

| | B1 (no AI) | B3-Llama |
|---|---|---|
| Classes F1, M1 | **0.323** | 0.179 |
| Classes F1, M3 | **0.387** | 0.231 |
| Taxonomy F1 (any level) | 0.000 (guaranteed, no mechanism) | **0.077** |
| Attributes F1, M3 (effective) | 0.094 | **0.179** |

**Verified.** Pure word-frequency counting currently outscores an actual LLM
on raw class-level F1. Not because B1 "understood" more — it's the direct
cost of over-generation: 67 induced classes vs. B1's much smaller candidate
set means precision collapses regardless of how many real concepts Llama
correctly identified. But the result is genuinely mixed, not a clean win for
either baseline — Llama clearly wins on taxonomy (structurally impossible
for B1) and roughly doubles B1's attribute F1 at M3, exactly where real
language understanding should show up. Worth stating both halves plainly in
Results — a reader who only sees "B1 > B3 on classes" without the taxonomy/
attribute context would draw the wrong conclusion about which method is
actually better overall.

### What's driving the class explosion — verified via direct computation, not estimated

Checked every induced class name against the real 11 gold classes
(normalized for case/spacing/camelCase — what M1 does). **8 raw string
matches, but `MaintenanceRequest` and `Maintenance Request` both target the
same gold class** — bipartite one-to-one assignment credits only one as TP,
confirming the harness's split-class handling (same mechanism as the T7 toy
fixture) is working correctly on real data, not just synthetic tests.

The dominant driver of the 67-class count: **incident-as-class confusion.**
`Ant Problem`, `Mouse Issue`, `HVAC Repair`, `Lighting Issue`,
`Plumbing Fix`, `Irrigation Blowout`, `storefront`, `window`, `pipe`,
`irrigation`, `Plumbing`, `Light`, `Carpet`, `Tree`, `Outlet` — the model
mints a new class for every individual maintenance incident type
encountered, rather than recognizing them as instances of one
`MaintenanceRequest` with a category attribute. None of these have a gold
analog — pure false positives. **Hypothesis** on root cause (not fully
confirmed): the prompt's own class criterion — "several distinct instances
... beyond a bare label" — is a reasonable rule that the model likely isn't
reliably applying, since most of these probably come from a single
document each. Worth checking whether this is Llama-8B-specific
(weaker instruction-following) or persists across Haiku/Sol/Luna/Fable 5 —
a real, plannable comparison once those run.

### Non-monotonic scores across M1→M2→M3 — a real, explainable harness behavior, not a bug

Taxonomy F1 goes 0.077 → 0.000 → 0.077 across M1/M2/M3; relations goes
0.000 → 0.027 → 0.009. **Verified mechanism**, traced through the actual
per-level match tables: bipartite assignment is solved independently at
each level, and the *globally optimal* class pairing can shift between
levels as new candidate matches open up (e.g. `Office`↔`OfficeProperty`
only becomes available at M3). When the winning pairing for `Property` or
`Agent` changes between levels, a taxonomy/relation edge that depended on
the old pairing can stop counting even though the underlying data didn't
change. This is a correct consequence of solving assignment per-level, not
a scoring error — worth stating explicitly in the paper's methodology
section as a known property of the harness design, since a reviewer could
otherwise mistake it for a bug. Possibly worth a dedicated toy fixture
later to lock this behavior in as intended.

### Other confirmed issues

- **Attribute bloat**: `Tenant` (58 attributes), `Lease` (48), `Landlord`
  (46) — largely near-duplicate facts reworded per batch (`monthly payment`
  / `Monthly_Base_Rate` / `monthly_payment` / `payment amount` / `rent` for
  one underlying fact). Direct, expected consequence of B3-D4's naive,
  exact-string-only consolidation across ~28 stateless batch calls — no
  batch has visibility into another batch's naming choices. Working as
  designed, not a bug; this is the cost B3 is supposed to make visible.
- **Self-loop relations** (6 confirmed): `Plumbing fixed Plumbing`,
  `Irrigation busted Irrigation`, `Window cleaned Window`,
  `Light flickering Light`, plus two `Person`-to-`Person` generic role
  collapses. Likely passive-voice source sentences forced into an SVO shape
  with nothing else to fill the object slot.
- **Instance/class conflation**: `Hoffman Services resolves Mouse Issue`,
  `Hurst Co. resolves Lighting Issue` — real vendor company names (instance
  data from the corpus) appearing as relation sources instead of the
  generic `Vendor` class.
- **Good, correctly-recovered structure**: `Broker represents Tenant/
  Landlord` (real match to gold's `Agent-represents-Owner/Tenant`),
  `Landlord owns Property` (exact conceptual match to gold's
  `Owner-owns-Property`). `Warehouse`/`Retail Space`/`Industrial Property`
  all got `parent: Property` — a real echo of gold's actual
  Office/Retail/Industrial-under-Property structure, unprompted.

### Batching note — this run predates a mid-project design change

This run used `BATCH_SIZE=7` uniformly across all conditions, inherited
from Groq's free-tier 6,000 TPM ceiling for Llama specifically. Decided
afterward (before running Haiku/Sol/Luna/Fable 5): **decouple batch size —
Llama/Groq stays at 7 (its genuine constraint), the four paid Bedrock
models get a much larger batch size**, since none of them are anywhere
near that ceiling and forcing Llama's limitation onto all five was testing
everyone under the weakest condition's hosting constraint, not a fair
like-for-like task shape. See `baselines/DECISIONS.md` for the full
reasoning. Important for interpreting this Llama run specifically: its
28-batch fragmentation is a genuine, real property of the free-tier
condition — worth citing directly as "the cost of free" rather than
treating it as noise to control away.

### Open questions / Limitations material

- Does the incident-as-class confusion persist across Haiku/Sol/Luna/
  Fable 5, or is it specific to an 8B model's instruction-following
  reliability? Directly testable once those runs complete.
- The prompt's class criterion ("several distinct instances... beyond a
  bare label") is reasonable as written — worth checking whether stronger
  models simply follow it more reliably, which would itself be a finding
  about model capability rather than prompt design.
- Non-monotonic M1→M2→M3 scoring (documented above) should be explained
  in the paper's methodology section proactively, not left for a reviewer
  to flag as unexpected.