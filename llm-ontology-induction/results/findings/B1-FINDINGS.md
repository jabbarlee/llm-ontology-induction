# Findings Log

Running notes, written as each condition (B1, B3, P1) gets scored — not the
paper itself, but the raw material to draft it from later. Add a new section
per condition as soon as you've scored it and understood *why* it scored that
way, while the reasoning is still fresh. Each section should be self-contained
enough that six months from now, you don't have to re-derive the explanation
from the raw CSV numbers.

**How to use this when writing the paper later:**
- The "What worked / What didn't / Why" bullets under each condition map
  roughly to the **Results** section (state the numbers) and **Error
  Analysis** section (explain them) of your paper's write-up order (§9 of the
  roadmap: Method → Datasets → Eval Setup → Results → Error Analysis → ...).
- "Open questions" are candidate **Limitations** section material.
- Anything marked "verified" below was checked against the actual repo/corpus
  in conversation, not guessed — safe to cite directly. Anything marked
  "hypothesis" was a plausible explanation that was NOT independently
  confirmed — re-verify before stating it as fact in the paper.

---

## B1 — Statistical baseline (TF-IDF/C-value + POS relation patterns)

**Run ID:** `2026-08-10T01:06:25Z-a6db`
**Scored:** all 192 documents, one pooled run (per DECISIONS.md D1)

### Scores

| Level | Classes F1 | Taxonomy F1 | Attributes F1 (eff.) | Relations F1 |
|---|---|---|---|---|
| M1 (exact) | 0.32 | 0.0 | 0.03 | 0.0 |
| M2 (fuzzy+lexicon) | 0.32 | 0.0 | 0.0 | 0.0 |
| M3 (semantic) | 0.39 | 0.0 | 0.09 | 0.0 |

Class-level detail: TP=5→6, FP=15→14, FN=6→5 across M1→M3, against 11 true
gold classes (verified: `grep -c "a owl:Class"` on `gold_schema.ttl` = 11).

### What worked

- Correctly found 5 gold classes from raw frequency alone, no AI: `Tenant`,
  `Owner`, `Lease`, `Property`, `Party` (matched via its plural form
  "parties"). **Verified.**
- M3 picked up one additional match beyond M1/M2 — `office` ↔
  `OfficeProperty` — a real semantic match with no shared string and (almost
  certainly) no lexicon entry. **Hypothesis** on the exact pairing (not
  independently confirmed which specific pair drove the M3-only gain), but
  the TP count increase (5→6) itself is verified from the scores.

### What didn't work, and why

- **Relations: 0.0 at every level, TP=0 throughout.** Cause, verified by
  inspecting the actual induced relations: every one of B1's 10 extracted
  relations has at least one endpoint (`sum`, `amount`, `terms`, `premises`,
  `payment`) that never matched a real gold class. Per the harness's D4 rule
  (both relation endpoints must match for the triple to count), every
  relation was automatically disqualified — not because the grammar-finding
  failed, but because the *nouns* plugged into that grammar weren't the real
  entities. B1 found real sentence patterns ("tenant provides X"), it just
  attached them to the wrong nouns.
- **`MaintenanceRequest` never appears as a candidate class at all.**
  **Verified**: `grep -ril "maintenance request" data/documents/` returns
  zero files out of 192. This traces back to Step 3 — the document-generation
  prompts told Gemini to invent CRM-style column names and describe issues in
  plain language, and it simply never produced that exact two-word phrase
  anywhere. This is a corpus property, not a B1 weakness — worth stating
  explicitly in Limitations, since it means B1 (and possibly B3/P1) is
  structurally unable to recover this class by name-matching alone, no
  matter how good the method is.
- **Taxonomy: 0.0 at every level, by design.** B1 has no mechanism for
  detecting hierarchy — every class gets `parent: null` (DECISIONS.md D6).
  Not a bug, not a finding requiring explanation beyond this.
- **M1 → M2 shows zero improvement (0.32 → 0.32 exactly), which is *not* a
  bug.** **Verified mechanism**: `Landlord` and `Lessee` are present in B1's
  raw output and almost certainly recognized as synonyms of `Owner`/`Tenant`
  by the M2 lexicon — but `Owner` and `Tenant`'s single gold "slots" were
  already claimed by the exact-string matches `owner`/`Tenant` at that same
  pass. Bipartite one-to-one assignment (same logic as the T7 split-class
  toy fixture) correctly refuses a second true positive for an
  already-matched gold class, so the synonym recognition produced no new
  points. This is confirmation the harness is working correctly, not a B1
  flaw — worth a line in Results explaining the flat M1→M2 step so a reader
  doesn't mistake it for the lexicon being broken.
- **Noise classes**: `AM`, `EOW`, `Date`, `sum`, `amount` — informal-text
  abbreviations and generic financial nouns picked up by naive noun-chunking.
  Left as-is deliberately (see DECISIONS.md addendum, if one was written,
  re: the freeze discipline — don't retroactively filter after seeing the
  score). Good, concrete material for Error Analysis: "B1 confuses
  timestamp/currency vocabulary for entities."

### Open questions / Limitations material

- How much of B1's ceiling is capped by the corpus never using gold's exact
  attribute/class vocabulary (per the `MaintenanceRequest` finding above) —
  worth checking whether B3/P1 hit the same wall, or whether LLM-based
  methods can bridge it semantically where pure string/frequency methods
  can't.
- Attributes dip to 0.0 exactly at M2 (from 0.03 at M1) before recovering to
  0.09 at M3 — mechanically explained by attributes only being scorable under
  an already-matched parent class, and class-matching not improving at M2
  (see above). Worth a one-line note in Results so this doesn't read as a
  contradiction.

---

## B3 — Single-shot LLM baseline

*(not yet run — fill in after scoring, same structure as above: scores
table, what worked, what didn't + why, open questions)*

---

## P1 — Full multi-stage pipeline

*(not yet run — fill in after scoring, same structure as above)*

---

## Cross-condition comparisons

*(fill in once at least B1 + B3 are both scored — this is where the actual
"our pipeline wins because X" paper narrative starts taking shape)*
