## At a glance

Every number below is reproduced from the per-condition tables further down this
file — the charts are a visual index into the same evidence, not a separate claim.
Where the two disagree, the tables are the source of truth.

**The two charts immediately below predate the Opus 5 run (added 2026-08-19) and
show only B1/Llama/Haiku** — rebuilding their pixel geometry for a fourth series
is left for a follow-up pass rather than done inline here. **The headline number
they're missing:** Opus 5 scores 0.857 classes F1 at M3, against Haiku's 0.429 and
B1's 0.387 — see the dedicated Opus 5 section below for the full picture, which is
not a small step up but the first condition to look like it is actually reading
the corpus rather than pattern-matching a slice of it.


Haiku is the first B3 condition to beat B1 on Classes F1, at every matching level —
and, per the chart below, by the opposite failure mode from Llama: precision instead
of coverage.

Llama's batched run over-generates: 67 induced classes buy high recall (0.82 at M3)
at the cost of precision (0.13 — 58 of 67 induced classes are false positives).
Haiku's whole-corpus run does the opposite: 3 classes, all correct (precision 1.00),
at the cost of recall (0.27 — 8 of 11 gold classes are missed entirely). Neither
number is "better" in isolation; see each condition's own section for what drives it.

---

## B3 — Single-shot LLM baseline: Llama 3.1 8B (open-weight, two conditions)

Two runs of the same open weights, on two hosts, in the two B3 call shapes (B3-D6).
The **batched/Groq** run is scored and analyzed in full below. The **whole-corpus/
Bedrock** run truncated and produced no scoreable output — its own section, with the
precise evidence for why, follows at the end.

### Condition: batched (Groq, `llama318b_groq`) — scored

**Run ID:** `2026-08-12T17:53:58Z-83cb`
**Scored:** full corpus, 28 batches of 7 documents each (uniform batch size).
**Status:** the **batched arm** of B3-D6's same-model comparison. Reproduce with
`--model llama318b_groq --batch-size 7`. The current B3 shape is whole-corpus
single-shot — see the batching note at the end for what changed and why this run
is retained rather than superseded.
**Verified:** loaded cleanly through `eval/schema_ir.py`'s real
`load_induced_json()`/`parse_induced_schema()` before scoring — contract
genuinely satisfied, not just visually inspected.

#### Scores

| Level | Classes F1 | Taxonomy F1 | Attributes F1 (eff.) | Relations F1 |
|---|---|---|---|---|
| M1 (exact) | 0.179 | 0.077 | 0.089 | 0.000 |
| M2 (fuzzy+lexicon) | 0.205 | 0.000 | 0.073 | 0.027 |
| M3 (semantic) | 0.231 | 0.077 | 0.179 | 0.009 |

Class-level detail: TP 7→8→9, FP 60→59→58, FN 4→3→2 across M1→M3, against
11 true gold classes. 67 induced classes total.

#### Headline, surprising result: B1 beats B3-Llama on class F1, at every level

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

#### What's driving the class explosion — verified via direct computation, not estimated

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

#### Non-monotonic scores across M1→M2→M3 — a real, explainable harness behavior, not a bug

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

#### Other confirmed issues

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

#### Batching note — this run is now the batched arm of a controlled comparison

*Rewritten 2026-08-14. The previous version of this section described a
"decouple batch size" decision — Llama at 7, the paid models larger — and
cited `baselines/DECISIONS.md` for its reasoning. **No such decision was ever
made.** It is not in DECISIONS.md, not in the code, and not in git history;
the citation pointed at nothing. It is replaced here with the real history
rather than left standing.*

This run used `BATCH_SIZE=7` uniformly across all conditions, inherited from
Groq's free-tier 6,000 TPM ceiling for Llama specifically (B3-D2, as it stood
on 2026-08-12). Critical Rule 7 then propagated that one host's limit to all
five models — uniformity applied at the wrong level, so every condition was
being tested under the weakest-provisioned one's hosting constraint rather
than a fair like-for-like task shape.

**What actually happened next (2026-08-14):** the constraint was found to be
an artifact, not a fact about the task. Llama 3.1 8B is served on Bedrock with
a 128K context window and no per-minute cap, and the corpus rendered through
the frozen prompt is ~43,400 input tokens — inside every model's window. B3-D2
was revised to make **whole-corpus single-shot the current shape for all
models**, and batching survives only as an explicit legacy flag.

**How that changes the standing of this run:** it is no longer a superseded
first attempt. B3-D6 retains it as the **batched arm of a same-model
batched-vs-whole-corpus comparison** — same weights, same frozen prompt, same
document order, same naive consolidation, differing in how many documents ride
in a call. The 67-class explosion analyzed above is the measurement of what
cross-batch fragmentation costs when nothing intelligent stitches the pieces
back together, which is precisely the gap P1's Stage 6 exists to close.

Two caveats to carry into any write-up of that comparison. The arms differ in
**host** (Groq free tier vs. Bedrock) and **output cap** (2048 vs. 4096) as
well as in call shape — real confounds that must be named, not smoothed over.
And this run remains the only genuinely zero-cost cell in the grid: the
Bedrock move reframed the open-weight condition from "zero-cost" to
"open-weight model, paid hosting" (B3-D1c, revised).

#### Open questions / Limitations material

- Does the incident-as-class confusion persist across Haiku/Sol/Luna/
  Fable 5, or is it specific to an 8B model's instruction-following
  reliability? **Partly answered 2026-08-14: it did not transfer to Haiku 4.5**,
  whose whole-corpus run carried zero false positives of any kind (see the
  Haiku section below). That supports the 8B-instruction-following
  hypothesis, but it is one model at one tier, and the two runs also differ
  in call shape — not yet settled.
- The prompt's class criterion ("several distinct instances... beyond a
  bare label") is reasonable as written — worth checking whether stronger
  models simply follow it more reliably, which would itself be a finding
  about model capability rather than prompt design.
- Non-monotonic M1→M2→M3 scoring (documented above) should be explained
  in the paper's methodology section proactively, not left for a reviewer
  to flag as unexpected.
- Does the repetition-degeneration loop documented in the whole-corpus
  condition below appear in the four cloud models under whole-corpus, or is
  it specific to an 8B model's ability to sustain coherent long-context
  generation? Directly testable once those runs complete.

### Condition: whole-corpus (Bedrock, `llama318b_bedrock`) — truncated, not scored

**Run ID:** `2026-08-14T20:29:28Z-f800`
**Attempted:** full corpus, one call, 192 documents, 151,831 prompt chars (~43,400
est. input tokens).
**Result:** `stop_reason: "length"` at exactly **4,096 completion tokens** — the
documented Bedrock ceiling for this model (B3-D3). The truncation guard fired as
designed: the run aborted, **no schema JSON was written**, and the cut-off response
is preserved in
`results/raw/2026-08-14T20:29:28Z-f800_b3_llama318b_bedrock_batches.jsonl`.
**Not scoreable.** Per B3-D3's decision rule, recorded here as a finding about the
Bedrock Meta serving limit, not a limitation to fix by lowering the other four
models' caps.

#### What actually happened before the cap bit: a repetition-degeneration loop, not legitimate over-generation

The response is not 4,096 tokens of genuinely distinct content cut off mid-thought.
It is exactly 50 `classes` objects, and the pattern is precise and mechanical
(extracted directly from the raw response, not summarized from memory):

- **Classes 1–7 are real, distinct extractions**, with sensible taxonomy:
  `Lease` (`parent: null`), `Tenant` (`parent: Lease`), `Landlord` (`parent: Lease`),
  `Property` (`parent: null`), `Maintenance Request` (`parent: null`), `Vendor`
  (`parent: null`), `Contractor` (`parent: Vendor`).
- **Classes 9–12** — `Retail Property`, `Industrial Property`, `Office Property`,
  `Warehouse`, all `parent: Property` — echo the real Office/Retail/Industrial-under-
  Property structure also seen in the batched Groq run above: a genuine, if
  attribute-templated, recovery.
- **From class 13 the model stops extracting and starts templating.**
  `Lease Agreement` (13, `parent: Lease`) carries the *exact same nine-attribute
  list* as `Lease` (1), with its own name spliced in as an extra attribute. The
  identical copy-with-a-new-name pattern repeats through class 25 —
  `Tenant Business`, `Landlord Business`, `Property Owner`, `Tenant Representative`,
  `Landlord Representative`, `Leaseholder`, `Contractor Representative`,
  `Vendor Representative`, `Tenant Operations`, `Landlord Operations`,
  `Property Maintenance` — each one cloning class 1, 2, 3, or 7's attribute list
  verbatim under an invented name, all `parent: Lease/Tenant/Landlord/Contractor`.
- **From class 26 it goes fully combinatorial.** The model generates the Cartesian
  product of `Lease` × {`Renewal`, `Termination`, `Extension`, `Amendment`,
  `Cancellation`, `Modification`} (classes 26–31), then that same set × `Notice`
  (32–37), then × `Agreement` (38–43), then × `Notice Agreement` (44–49) — 24
  classes from one mechanical pattern, every single one `parent: Lease`, every
  single one carrying class 1's identical nine-attribute list. **At class 50 the
  pattern completes a full cycle and repeats class 38
  (`Lease Termination Agreement`) verbatim** — the response was cut off by the
  token cap mid-loop, not mid-thought.

**Interpretation, stated as interpretation, not a proven mechanism:** this reads as
the small (8B) model losing coherent control of a long, structurally repetitive
generation task once ~43,400 tokens of dense, heterogeneous input (192 documents
across four different formats) filled its working context — a known small-model
failure mode under long-context, high-output generation, distinct from genuinely
having 100+ real classes to name. The batched Groq run never showed this pattern:
no single batch there ever asked the model to hold more than 7 documents in view.

**What this means for B3-D6's comparison, stated plainly:** whole-corpus produced
**zero usable output** for the open-weight model, while batched produced a usable
(if precision-poor, over-fragmented) schema. Fragmentation cost precision — 67
induced classes, mostly real incidents minted as classes — but the model never lost
coherence entirely. Whole-corpus removed the fragmentation but broke down into
outright repetition before finishing. That is a capability-tier-dependent tradeoff,
not evidence that either shape is unconditionally better, and should be stated that
way rather than as a simple "whole-corpus wins."

---

## B3 — Single-shot LLM baseline: Claude Haiku 4.5 (budget tier, whole-corpus)

**Run ID:** `2026-08-14T21:34:50Z-6ee0`
**Scored:** full corpus, 192 documents, **one whole-corpus call** (151,831 prompt
chars). Reproduce with `--model haiku45`.
**Completed cleanly:** `stop_reason: "end_turn"` at **417 completion tokens** against
a 16,000 cap — the cap never came close to binding. This is not truncation; the model
chose to be this terse.
**Verified:** loaded through `eval/schema_ir.py`'s real
`load_induced_json()`/`parse_induced_schema()` before scoring.
**Note on B3-D4:** with one call the naive merge runs over a single schema and is a
no-op. Nothing about the rule was relaxed — it simply has nothing to do here.

### Scores

| Level | Classes F1 | Taxonomy F1 | Attributes F1 (eff.) | Relations F1 |
|---|---|---|---|---|
| M1 (exact) | 0.429 | 0.000 | 0.000 | 0.111 |
| M2 (fuzzy+lexicon) | 0.429 | 0.000 | 0.000 | 0.111 |
| M3 (semantic) | 0.429 | 0.000 | 0.139 | 0.111 |

Class-level detail: **TP 3, FP 0, FN 8** — identical at all three levels, against 11
gold classes. Precision **1.000**, recall **0.273**. Only 3 induced classes total.

### Headline: the first condition to beat B1 — and it wins on precision, not coverage

| Classes F1 | M1 | M3 |
|---|---|---|
| B1 (no AI) | 0.323 | 0.387 |
| B3-Llama (batched, Groq) | 0.179 | 0.231 |
| **B3-Haiku (whole-corpus)** | **0.429** | **0.429** |

**Verified.** Haiku is the first B3 condition to beat the zero-LLM baseline, and it
does so through the exact opposite failure mode from Llama. Llama emitted 67 classes
and carried 60 false positives; Haiku emitted **3 and carried none**
(`classes_unmatched_induced` is empty in the match log). Every class it named was
real: `Lease`↔`Lease`, `Property`↔`Property`,
`Maintenance Request`↔`MaintenanceRequest`, all at score 1.0.

Worth stating plainly in Results, because the single F1 number hides it: these two
conditions are not "one better than the other by 0.2 F1." They are opposite errors —
uncontrolled over-generation versus severe under-generation — that happen to land on
comparable-looking scores. A reader given only the F1 column would draw the wrong
conclusion about what a budget cloud model actually does.

### The mechanism: 12 of 192 documents produced 100% of the output

**Verified by direct comparison against the corpus, not inferred.** Haiku's three
classes are exactly the three CSV export types, and their attribute lists are the CSV
column headers **verbatim and in order** — including the inconsistent casing *between*
files, which is the tell:

| Induced class | Attributes emitted | Source |
|---|---|---|
| `Lease` | `Lease Owner`, `Tenant Name`, `Agent Name`, `Monthly Base Rate`, … | header row of `csv_exports/leases_export_*.csv` |
| `Property` | `Unit Location`, `Asset Class`, `Net Rentable Area`, `Level Count`, … | header row of `csv_exports/properties_export_*.csv` |
| `Maintenance Request` | `CurrentState`, `Urgency`, `EntryDate`, `IssueSummary`, … | header row of `csv_exports/maintenance_requests_export_*.csv` |

The corpus is 12 `csv_exports/`, 50 `lease_texts/`, 65 `notes/`, 65 `messages/`.
**The 12 CSV files — 6% of the documents — account for the entire output. The 180
prose documents left no detectable trace.** Haiku did not synthesize a schema from
192 heterogeneous sources; it transcribed three CSV headers.

This is the single most important caveat on the whole-corpus condition and must not be
buried in the write-up. It is a threat to that condition's validity: if the model is
effectively reading 6% of the corpus, "whole-corpus" describes what was *sent*, not
what was *used*.

**What it is not:** it is not simply an artifact of length. The `--limit 2` smoke test
(8 documents, 2 CSV files) also came back terse — 2 classes, `Lease` and
`Maintenance Request`. The terseness is present at both scales; the CSV anchoring is
what becomes stark at 192.

### The 8 missed classes split into two groups, with different causes

| Missed gold class | Why |
|---|---|
| `Tenant`, `Owner`, `Agent`, `Vendor` | **Not missing from the evidence — demoted.** Haiku saw these as CSV columns and modeled them as *attributes of Lease* (`Lease Owner`, `Tenant Name`, `Agent Name`) rather than promoting them to classes. A class/attribute boundary error, the same judgment call B1's D4 heuristic struggles with. |
| `OfficeProperty`, `RetailProperty`, `IndustrialProperty` | **Genuinely unseen.** These distinctions live in the prose documents, which contributed nothing. Note the contrast: the Llama batched run *did* recover all three with `parent: Property`, unprompted. |
| `Party` | Gold's abstract superclass. No condition has recovered it; it requires positing an unnamed generalization. |

The first row is the more interesting finding: the failure is not perception but
*modeling*. The information reached the model and was assigned to the wrong layer.

### Taxonomy 0.000 — and this is a regression against Llama

Every `parent` Haiku emitted is `null` (TP 0, FP 0, FN 7). Critical Rule 6 means
nothing here infers hierarchy, so a nonzero score requires the model to volunteer it —
Llama did (0.077, via `Warehouse`/`Retail Space`/`Industrial Property` under
`Property`), and **Haiku did not at all**. Any claim that the stronger model produced
better structure is false as stated: it produced cleaner classes and *no* structure.

### Attributes: 0.000 at M1/M2, 0.139 at M3 — a direct consequence of the CSV transcription

TP 0, FP 28, FN 44 at M1 and M2; TP 5, FP 23, FN 39 at M3. Emitting raw CSV headers
means `Net Rentable Area` never string-matches gold's `squareFootage` and
`Monthly Base Rate` never matches `rentAmount`, so exact and fuzzy matching both score
zero. Only M3's semantic matching recovers anything. This is the clearest single
demonstration in the results so far of *why* the harness has three matching levels.

### Relations: 1 of 3 emitted matched, and the near-miss is instructive

TP 1, FP 2, FN 14 (P 0.333, R 0.067, F1 0.111), unchanged across levels.

- ✅ `Maintenance Request —concerns→ Property` — **exact match** to gold's `:concerns`
  (domain `MaintenanceRequest`, range `Property`), verified against the TTL.
- ❌ `Lease —occupies→ Property` — **right endpoints, wrong verb.** Gold has
  `:covers` (domain `Lease`, range `Property`). Conceptually the same edge; it fails
  at M1/M2 on the label and does not recover at M3 either. Worth citing as a concrete
  case where the relation layer is stricter than the class layer.
- ❌ `Maintenance Request —involves→ Lease` — no gold analog; gold asserts no
  MaintenanceRequest→Lease edge.

### Open questions / Limitations material

- **Does the CSV anchoring persist in the frontier tier?** This is now the most
  important open question for the whole-corpus condition. If Fable 5 and Sol also
  transcribe the three CSV headers and ignore 180 prose files, the whole-corpus shape
  has a structural problem worth reporting in its own right. If they don't, this is a
  budget-tier capability finding.
- **The incident-as-class confusion did *not* transfer.** The Llama batched run's
  dominant error (minting a class per maintenance incident) is entirely absent here —
  zero false positives. That supports the hypothesis recorded in the Llama section:
  it looks like an 8B instruction-following limitation, not something inherent to
  single-shot prompting. One data point, one model — not yet settled.
- **Under-generation may be the budget tier's characteristic failure**, as
  over-generation was the open-weight tier's. Two conditions is not enough to claim
  this; the remaining three cells decide it.
- The class/attribute demotion of `Tenant`/`Owner`/`Agent`/`Vendor` is a **precision-
  preserving** error, which flatters F1. A method that named them as classes and got
  some wrong would score worse while arguably modeling better. Worth a sentence in the
  paper's discussion of what F1 does and does not reward.

### Discussion: why the CSV anchoring might happen, and what to expect from Fable 5

*Everything in this subsection is HYPOTHESIS, not verified — recorded 2026-08-15,
before Fable 5 has run, so it can be checked against the outcome rather than
rationalized after it. Contrast with the mechanism section above, which is
measurement (the attribute lists are byte-identical to real CSV headers, checked
directly against the corpus files).*

**Not hallucination.** Worth ruling out explicitly: hallucination means confidently
stating something false. Haiku's three classes had zero false positives — everything
it said was true and grounded. This is under-generation / selective extraction, a
different failure mode from fabrication, and the two should not be conflated in the
paper's error taxonomy.

**Two candidate mechanisms for the CSV anchoring, not mutually exclusive:**

1. **Economy of effort.** A CSV header row is free structure — `Lease Owner,Tenant
   Name,Monthly Base Rate` requires no inference, the schema is handed to the model
   pre-labeled. A prose document (*"Sarah from Unit 4B called about a leaking
   faucet, Contractor Mike came out Tuesday"*) requires synthesizing structure that
   is never stated explicitly. A model tuned toward speed and efficiency — which is
   literally Haiku 4.5's stated design point — may be disproportionately drawn to
   the cheap, high-confidence source over the expensive, inferential one. This is
   also consistent with the incident-as-class question above: a more conservative,
   effort-economizing model requires stronger evidence before committing to a
   class, which is exactly what suppressed Llama's over-eager pattern here.
2. **Long-context positional effects.** CSV documents are the *first* file type in
   every early round of the interleaved document order (`_SUBDIRS` visits
   `csv_exports` first each round), and there are only 12 of them against 180 prose
   documents that follow with no further structured anchor. This is consistent with
   the widely-reported "lost in the middle" effect in long-context LLMs generally —
   not isolated experimentally here, but a known phenomenon that predicts exactly
   this pattern: early, structured content dominates a single long, unstructured
   generation task.

**Confound not yet resolved:** these results cannot yet separate "whole-corpus
shape causes this" from "Haiku specifically causes this," because there is no
batched-Haiku run to compare against. A `--batch-size 7` run of `haiku45` would
settle it directly — if it also anchors disproportionately on CSV batches, the
effect is model-level; if it spreads evenly, whole-corpus is doing something
Haiku's batched condition would not.

**Prediction for Fable 5, stated before it runs so it can be checked:** Fable 5 is
architecturally close to the opposite of Haiku — frontier tier, 1M context,
"sustained autonomous operation across multi-day tasks... plans across stages...
self-verifies its work," adaptive thinking always on. That disposition argues
*against* the CSV-anchoring pattern repeating: a model built to systematically
canvas complex input before answering is less likely to lock onto the first easy
structure and stop. **Expect higher recall than Haiku** — more of the 180 prose
documents actually contributing to the schema.

Precision is the genuinely open question, and could go either way for reasons that
are both plausible:
- *Self-verification could act as a brake*, holding precision near Haiku's (1.000)
  while raising recall — the best-case outcome for this condition.
- *Or the same thoroughness could re-introduce something like Llama's
  over-generation* — not via cross-batch fragmentation (there is no batching in one
  call) but via a single, coherent, very thorough pass deciding that every distinct
  incident, every named party, every CSV row deserves its own class, aided by a
  128K output cap (32× Haiku's) that leaves ample room to do so.

Do not treat "frontier tier" as a reason to assume better results by default — that
is the assumption this note exists to resist. Record the actual outcome here once
Fable 5 has run, and note explicitly whether precision moved toward the first
scenario or the second.

**Superseded 2026-08-19, not deleted.** Fable 5 was dropped from the active B3
registry when the grid narrowed to Haiku 4.5 (Bedrock) + Opus 5 (direct API) —
see `baselines/DECISIONS.md` B3-D1 (revised again). No Fable 5 run exists or will
exist under this baseline's current shape, so the prediction above is never
checked against the model it names. It stays as written because the *reasoning*
about frontier-tier disposition and the two candidate precision outcomes is still
a real hypothesis worth having on record — it is simply now read against Opus 5,
the frontier condition that actually ran, in the section immediately below rather
than against Fable 5.

---

## B3 — Single-shot LLM baseline: Claude Opus 5 (frontier tier, whole-corpus, direct Anthropic API)

**Run ID:** `2026-08-19T20:38:13Z-4776`
**Scored:** full corpus, 192 documents, **one whole-corpus call** (151,831 prompt
chars, ~43,380 est. input tokens). Reproduce with `--model opus5`.
**Completed cleanly:** `stop_reason: "end_turn"` at **4,311 completion tokens**
against a 32,000 cap shared between thinking and visible text (B3-D3, revised
2026-08-19) — nowhere near binding. Not truncation; adaptive thinking plus a
terse final answer.
**Verified:** loaded through `eval/schema_ir.py`'s real
`load_induced_json()`/`parse_induced_schema()` before scoring, then scored via
`eval.report` against `schema/gold_schema.ttl` at all three levels.
**Note on B3-D4:** one call, naive merge is a no-op — same as every other B3
condition.

### Scores

| Level | Classes F1 | Taxonomy F1 | Attributes F1 (eff.) | Relations F1 |
|---|---|---|---|---|
| M1 (exact) | 0.571 | 0.000 | 0.000 | 0.065 |
| M2 (fuzzy+lexicon) | 0.667 | 0.000 | 0.016 | 0.129 |
| M3 (semantic) | **0.857** | **0.600** | 0.244 | 0.065 |

Class-level detail at M3: **TP 9, FP 1, FN 2** against 11 gold classes — precision
**0.900**, recall **0.818**. 10 induced classes total, closest to gold's 11 of any
B3 condition so far.

### Headline: the first condition to clear 0.8 on class F1 — and the first to score on taxonomy at all

| Classes F1 | M1 | M3 |
|---|---|---|
| B1 (no AI) | 0.323 | 0.387 |
| B3-Llama (batched, Groq) | 0.179 | 0.231 |
| B3-Haiku (whole-corpus, Bedrock) | 0.429 | 0.429 |
| **B3-Opus 5 (whole-corpus, direct API)** | **0.571** | **0.857** |

| Taxonomy F1 (best level) | value |
|---|---|
| B1 | 0.000 (guaranteed — no mechanism) |
| B3-Llama | 0.077 |
| B3-Haiku | 0.000 |
| **B3-Opus 5** | **0.600** (M3; precision 1.000, recall 0.429) |

**Verified.** Opus 5 more than doubles Haiku's class F1 at M3 and is the only B3
condition to date whose taxonomy F1 is not effectively zero. Every taxonomy edge
it got right is exactly correct (precision 1.000, TP 3, FP 0) — it never guessed
wrong, it just didn't guess often enough to catch everything (FN 4).

### The mechanism: Opus 5 reads the prose documents — this is the opposite of Haiku's CSV anchoring

**Verified by direct inspection of the induced attribute lists against the
corpus**, the same check that established Haiku's CSV-only behavior. Several
Opus 5 attributes have no CSV column of any kind behind them and can only have
come from `lease_texts/`, `notes/`, or `messages/` prose:

- `Lease`: `renews into a new term starting`, `term (period of years)`,
  `square feet of space`, `use of premises (office-professional, retail,
  restaurant-food service, warehouse-logistics, medical)`
- `Tenant`: `doing business as`, `business (Huff Group, Higgins & Associates,
  Parker & Associates, etc.)`, `industry / use (office-professional, retail,
  …)`
- `Vendor`: `type of work (pest, HVAC, plumbing, electrical, landscaping,
  cleaning, drywall)`, `active insurance coverage on file`

None of these strings appear in any `csv_exports/*.csv` header — they are
free-text categories and clauses synthesized from the lease texts and message
threads. This directly answers the open question the Haiku section left hanging
("does the CSV anchoring persist in the frontier tier?"): **no, not for Opus 5.**
The taxonomy recovery (`office`/`retail`/`industrial` under `Property`, matching
gold's `OfficeProperty`/`RetailProperty`/`IndustrialProperty` exactly at M3) is
itself only extractable from prose — no CSV column names a property's use type —
so the taxonomy score above is direct evidence of the same mechanism, not a
separate finding.

This also resolves the two-scenario prediction recorded in the Haiku section's
discussion for the (now-superseded) Fable 5 run: outcome landed close to the
"self-verification acts as a brake" branch — recall rose sharply (0.273 → 0.818
at M3) while precision stayed high (1.000 → 0.900), not the over-generation
branch. One data point, one model family — stated as observation, not as a
general claim about "frontier tier" behavior.

### Familiar attribute bloat, present but far less severe than Llama's

`Lease` carries 23 attributes, several clearly the same B3-D4 naive-consolidation
pattern seen in the Llama batched run — near-duplicate facts under different
per-source wording, un-merged because nothing here does semantic consolidation:
`Monthly Payment`/`Monthly Base Rate`, `Escalation Method`/`Escalation
Structure`, `Held Deposit`/`Holding Amount`/`Sec_Dep` (three variants of one
fact), `Lease Start`/`Commence Date`, `Expiration Date`/`Expiry Date`,
`Notes`/`Contract Notes`. This is expected and working as designed (B3-D4) —
flagged here because it is the same mechanism as Llama's 58-attribute `Tenant`
bloat, just far less extreme with only 12 CSV sources instead of 28 stateless
batches feeding it. Attributes F1 (effective, M3) is **0.244** — TP 21, FP 107,
FN 23 — the best of any B3 condition so far, but still low in absolute terms;
the bloat is the direct cost.

### Relations: only 1 of 15 induced relations matched at M3 — and the reason is verified, not the label wording

**Verified by calling `eval.metrics._relations_match()` directly** against the
real gold and induced schemas (not estimated from the aggregate table): the sole
M3 true positive is `Lease --covers--> Property`, an exact label and direction
match to gold's `Lease :covers Property`.

**The more interesting, fully verified finding is why so many conceptually
correct relations still miss.** Opus 5 named the maintenance-request-approval
party and the property-owning party `Landlord`; gold's class is `Owner`. At M2,
lexicon-based class matching pairs `Owner`↔`Landlord` and a second relation
becomes scoreable: `Landlord --owns--> Property` matches gold's `Owner :owns
Property` exactly (TP rises to 2 at M2). **At M3, that same class pair does not
match** — `Owner` and `Landlord` both show up in `classes_unmatched_*` at M3,
the identical non-monotonic-matching mechanism the Llama section documented
(a semantic-embedding threshold miss where a lexicon-list hit had succeeded).
Because relation matching buckets by *already-matched* class pairs (M2's
`relations_layer` docstring, D4), every relation touching `Landlord` becomes
unscoreable the moment its class match disappears — **6 of the 15 M3 false
positives** (`Landlord --owns--> Property`, `Landlord --leases to--> Tenant`,
`Lease --entered into by--> Landlord`, `Maintenance Request --Owner
Approval--> Landlord`, `Tenant --shall pay--> Landlord`, `Agent --waiting on
the owner to sign off--> Landlord`) are relations whose *content* is
substantially correct but whose *scoreability* was lost entirely to a class-level
matching decision one layer up, not to anything wrong with the relation itself.
`Landlord --owns--> Property` in particular is a verbatim verb match to gold's
`Owner :owns Property` — it would have scored as a clean TP at M3 under the
exact same matching logic that accepted it at M2.

Worth stating explicitly in the paper's methodology section, reinforcing the
non-monotonicity note from the Llama section with a second, independent example:
relations-layer recall is bounded above by classes-layer matching at the *same*
level, and a real semantic-content match can be invisible to the relations score
purely because of where the class-matching threshold happened to fall.

### Open questions / Limitations material

- **Does the Owner/Landlord non-monotonicity generalize, or is it a one-off
  threshold miss?** Two independent conditions (Llama, Opus 5) now show a class
  pair matching at one level and not another, each time costing downstream
  relation/taxonomy score. Worth a dedicated toy fixture (flagged in the Llama
  section too) to lock in and stress-test this harness behavior directly rather
  than only observing it incidentally in real runs.
- **CSV anchoring looks budget-tier-specific, not whole-corpus-shape-specific**,
  now with two data points (Haiku anchors, Opus 5 does not) instead of one
  hypothesis. The confound the Haiku section named — no batched-Haiku run to
  separate "shape" from "model" — is still open; a `--batch-size 7` run of
  either model would settle it directly.
- **The attribute-bloat mechanism is confirmed present at the frontier tier**,
  just proportionally smaller. Worth checking whether P1's consolidation stage
  (which exists specifically to merge cross-wording) removes most of this bloat
  for Opus 5 specifically, since a smaller starting mess is an easier case than
  Llama's 58-attribute `Tenant`.
- The at-a-glance charts at the top of this file do not yet include Opus 5 —
  noted there, repeated here so it isn't missed by a reader who starts mid-file.