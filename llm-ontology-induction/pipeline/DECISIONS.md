# P1 — Decomposed Pipeline: Design Decisions

Frozen decisions for the six-stage P1 pipeline (`pipeline/graph.py` +
`pipeline/nodes/`). P1 is not a baseline the way B1 and B3 are — it is the
paper's contribution, which is also why it lives at the repo's top level
(`pipeline/`) rather than under `baselines/`, moved there on 2026-08-21 per
the decision recorded in P1-D1 below.

**This supersedes, not replaces, `baselines/DECISIONS.md`'s own P1-D1 through
P1-D5.** Those recorded the two-stage design (N extraction calls + 1
consolidation call) that shipped first and is still a legitimate, working
baseline shape — its code and its scored runs are not deleted. This file
records the six-stage decomposition built on top of it. See
`baselines/DECISIONS.md` for a dated note pointing here.

---

## P1-D1 — Six stages, not two: the architecture plan's own case, accepted

**Decision:** decompose the old single consolidation call into four stages —
type consolidation, attribute consolidation, relation reconciliation, and
taxonomy induction — each its own call, each independently
fault-isolated and independently instrumentable, plus the unchanged
extraction stage (1) and a deterministic assembly stage (6) with no model
call at all.

| # | Stage | LLM? | Calls |
|---|---|---|---|
| 1 | Per-document extraction | Yes | 192 |
| 2 | Type consolidation | Yes | 1 |
| 3 | Attribute consolidation | Yes | 1 |
| 4 | Relation reconciliation | Yes | 1 (0 if no relations survive endpoint remapping) |
| 5 | Taxonomy induction | Yes | 1 |
| 6 | Assemble + validate | No | 0 |

**Why not stay at two stages, given B3 with Opus 5 already scores ~0.86
classes F1 at M3 and near-1.0 precision** (results/findings/B3-FINDINGS.md) —
the honest problem this whole decision has to answer, stated in the plan's
own §0: if P1 is "B3 with more steps," it costs 5-10x more and risks scoring
the same or worse, which would undercut the paper's central comparison rather
than support it. The two-stage design's single consolidation call conflated
three genuinely different failure modes into one response: getting class
identity wrong, getting attribute wording wrong, and getting relation
endpoints wrong are three different things to get right, and a single call
asked to do all three at once cannot fail at one without the failure being
indistinguishable, in the output alone, from a failure at another. Splitting
them means a wrong merge in Stage 2 and a wrong wording choice in Stage 3 are
now two different log entries, not one conflated response — this is what
makes capability (c) (explicit, instrumentable consolidation) actually
inspectable rather than asserted, which is the whole roadmap-stated point of
building P1's consolidation as a distinct thing from B3's naive merge in the
first place (B3-D4).

**Stage 5 (taxonomy induction) is the one addition that is not a split of
existing work — it is new capability.** No prior P1 shape (two-stage or the
old single-consolidation-call design) had any mechanism for proposing a class
name absent from every partial schema. Every run so far — B1, the Llama
batched run, and Opus 5's B3 run — missed the gold schema's abstract
superclass entirely, for the structural reason that none of them ever "step
back and look at the finished answer." B3 cannot do this by construction (one
call, one pass, no second look). Stage 5 is P1's answer to that gap, and it
is the stage with the clearest case that decomposition offers something a
single call cannot, rather than the same thing done more expensively.

**The weak-model hypothesis (P1-D4, below) is what makes six stages a bet
worth recording rather than "more stages because more must be better."**
Decomposition is not expected to help a model that already nearly saturates
the task; it is expected to help the model whose single-shot output visibly
fragmented (67 classes against 11 gold). That prediction, not "P1 beats B3,"
is the finding this architecture exists to produce.

## P1-D2 — Stage 1 still uses the frozen B3 extraction prompt by reference

**Unchanged from the prior two-stage design.** `pipeline/nodes/extract.py`
imports `PROMPT_PATH` from `baselines.b3_single_shot.single_shot` and reads
the file at that path directly — never a copy under `pipeline/prompts/`. If
Stage 1's prompt differed from B3's even slightly, a difference in the final
schema could be attributed to either the prompting or the pipeline structure,
and the B3-vs-P1 comparison would stop meaning what the paper claims it
means. Verified by
`pipeline/tests/test_pipeline.py::test_stage_one_uses_the_frozen_b3_prompt_by_reference`,
which asserts the imported path object is literally B3's own module
constant, not a `pipeline/`-local file that merely happens to hold identical
text today.

## P1-D3 — Model matrix: ~~two conditions, not three~~ **three conditions, matching the plan's original count — for a different reason than the plan gave** — a discrepancy with the architecture plan, resolved in favor of the frozen registry

**The architecture plan that drove this build refers throughout to "your
three-model matrix" (Opus 5, Haiku 4.5, and Llama on Bedrock) and a
cost table totaling "~590-615 calls... across three models."** The actual
registry as of this pipeline's build date (`baselines/shared/model_clients.py`)
holds exactly two conditions: `haiku45` (AWS Bedrock) and `opus5` (direct
Anthropic API). Llama was retired from the active grid entirely in an earlier
revision (`baselines/DECISIONS.md`, B3-D1 revised) — the mixed-transport
correction that put Haiku back on Bedrock and narrowed the frontier condition
to Opus 5 alone never reintroduced an open-weight condition.

**Resolution: build against the two conditions the registry actually has, not
the three the plan assumed.** `pipeline/graph.py` reads `MODELS` from
`baselines.shared.model_clients` directly (`--model {haiku45,opus5}`), so it
automatically tracks whatever the registry holds rather than hardcoding
either count. This is stated here explicitly, rather than silently building
to two and letting a reader infer the plan's "three-model" language was
approximate — it was written against a matrix state that had already changed
by the time this architecture was built.

**Sampling, thinking/effort, and per-model output caps are identical to B3's
own (B3-D3, revised) for every condition** — `pipeline/nodes/` never sets its
own values; every stage's model call goes through the same
`baselines.shared.model_clients.invoke(spec, prompt)` entry point B3 and the
old two-stage P1 both use, with `spec` resolved from the same `MODELS`
registry via `state["model_key"]` (see `pipeline/state.py`'s own docstring
for why the resolved `ModelSpec` object itself is never stored in checkpointed
state — a serialization concern, not a modeling one).

### Revised 2026-08-21: Llama restored — three conditions again, matching the plan's original assumption after all

The correction above was resolved in the registry's favor at the time it was
written; the registry then changed again. Asked to give dry-run scripts for
"Llama, Haiku, and Opus," the user confirmed explicitly (via `AskUserQuestion`)
that Llama should come back on AWS Bedrock, the same condition and transport used
for B3's original run — see `baselines/DECISIONS.md`, B3-D1's third revision, for
the restoration itself (a third backend, `"bedrock_meta"`, alongside the existing
`"bedrock"` and `"anthropic_api"` ones).

**Nothing in this file's account of *why* two conditions were built is retracted
— it is superseded by events, not shown to have been wrong.** The registry
genuinely held two conditions at build time, for the reason stated above; the
user's own later instruction is what changed the count, not a flaw in the
original resolution. `pipeline/graph.py` needed zero code changes for this —
`--model {haiku45,opus5,llama318b}` is exactly what "reads `MODELS` directly
rather than hardcoding a count" (P1-D3's own design) was for.

**One condition-specific risk worth restating here, not just in the B3 record:**
`llama318b`'s 4,096-token output cap applies to every one of P1's five LLM
stages, not only Stage 1 — and Stage 2 (type consolidation) in particular takes
every partial schema's classes at once, which is the stage most likely to
produce an output large enough to test that ceiling. If it truncates, that is a
finding about this condition under six-stage decomposition specifically, following
B3-D3's own standing rule: never grounds to lower another condition's cap.

## P1-D4 — The weak-model hypothesis, written before any run

**Prediction, recorded 2026-08-21, before either model has been run through
this pipeline:** decomposition should help a weak/fragmenting model
substantially more than it helps a model that already nearly saturates the
task under B3's single-shot shape.

Concretely: Opus 5 scored 0.857 classes F1 at M3 under B3 with near-1.0
precision (10 classes against 11 gold) — there is very little headroom left
for six stages of additional machinery to recover. Haiku 4.5 under B3 scored
0.429 at every level, driven by severe *under*-generation (3 classes, CSV-only
— see B3-FINDINGS.md) rather than the fragmentation the plan's own §0
discusses for open-weight models, so the specific mechanism P1 might fix for
Haiku is different from the one it would fix for a fragmenting model like
Llama: Stage 1's per-document extraction, giving each of the 180 prose
documents full individual attention instead of competing for space in one
43,000-token prompt, is the capability most directly aimed at Haiku's
specific failure (missing the 180 non-CSV documents entirely), separate from
Stage 2's consolidation, which targets fragmentation specifically.

**The claim worth stating in the paper, if this holds:** *"pipeline
structure substitutes for model capability"* is a stronger, more falsifiable
finding than *"our pipeline beats one prompt."* It is directly testable
across the two-condition matrix this repo actually has (P1-D3): if Opus 5
gains little or nothing from decomposition while Haiku gains substantially,
that supports the claim; if both gain similarly, or Opus 5 gains more, that
is a real, reportable finding against it, not a result to explain away.

**Not yet checked against real results — no model has been run through this
pipeline as of this writing.** `results/FINDINGS.md`'s P1 section is required
to check this prediction explicitly against whatever the real runs show,
per the plan's own Definition of Done.

## P1-D5 — Merge-log format (Stage 2 instrumentation)

**Decision:** Stage 2's prompt (`pipeline/prompts/p1_consolidation_prompt.md`)
requires every merged class to report `merged_from`: the exact list of
original per-document class names folded into it. `pipeline/nodes/
consolidate_types.py` combines this with a deterministic Python index (every
raw class name, keyed to every source document it appeared under, built
directly from Stage 1's own output — never asked of the model a second time)
to produce two artifacts:

- `merge_log`: `[{"merged_name", "source_names": [...], "source_documents":
  [...]}]` — one entry per merged class, in the final output's own
  `metadata.merge_log`, so a reader of a scored run's JSON can inspect
  exactly which document(s) contributed which original wording to which
  final class, without re-running anything.
- `class_name_map`: a normalized-original-name -> merged-name dict, used
  internally by Stage 4 to rewrite relation endpoints (P1-D1's table,
  Stage 4's own justification). Not written to the final output — it is
  working state, not a claim about the schema.

**Why the source-document provenance is computed in Python, not asked of the
model:** Stage 2 already has to report which original names it merged;
asking it to also recall which of the (up to) 192 documents each of those
names came from, across a single call handling every partial schema at once,
is exactly the kind of bookkeeping a model is more likely to garble than a
dict already buildable, with certainty, from the input the code already has
in hand before the call is even made.

## P1-D6 — Taxonomy stage conservatism: require support, default to null, and require the new class to be declared

**Decision:** `pipeline/prompts/p1_taxonomy_prompt.md` requires concrete
support — a naming pattern or a real, shared-attribute overlap across **at
least two** of the given classes — before proposing any parent, new or
existing, and defaults to `null` otherwise, with the prompt stating plainly
that a missed generalization costs less than an invented one. This mirrors
the extraction prompt's own "when in doubt, use `null`" rule (Critical Rule
6) rather than inventing a separate standard for this one stage.

**A second, narrower rule found necessary while building, not anticipated by
the plan's own text: a newly proposed supertype must be declared as its own
class entry in the same response, not only referenced as some other class's
`parent` string.** Without this, a `parent` value naming a class that exists
nowhere in the schema's own `classes` list is exactly the dangling-endpoint
failure `pipeline/nodes/reconcile_relations.py`'s docstring describes for
relations (and which `results/findings/B3-FINDINGS.md`'s Opus 5 section
documents concretely on real data: 6 of 15 M3 false-positive relations were
conceptually correct but unscoreable purely because an endpoint's class match
didn't survive to that level) — the identical failure mode one layer up, in
taxonomy instead of relations. `pipeline/nodes/induce_taxonomy.py` treats a
model's failure to follow this defensively: any `parent` value that names a
class absent from both the input and the model's own declared output is
dropped back to `null` (counted as `undeclared_parents_dropped` in the raw
log) rather than assembled into a schema pointing at nothing.

**The illustrative example inside the prompt is deliberately unrelated to the
corpus's real domain** (vehicles, not real estate/leasing) — an example drawn
from the actual domain risks semantically priming the model toward the gold
schema's real superclass even without repeating its exact vocabulary, which
this project's Critical Rule 1 (no gold-schema vocabulary) is written to
prevent in spirit as well as in the literal, computationally-checked sense
`test_no_domain_vocabulary_leakage` enforces. The prompt is independently
verified leak-free the same way every other prompt in this project is —
computed against the real `gold_schema.ttl` vocabulary, not eyeballed.

## P1-D7 — Attribute consolidation granularity: one call for all classes, not one per class

**Decision:** Stage 3 makes exactly one call carrying every merged class's
raw attribute union, rather than one call per merged class (the architecture
plan's own open question 2, §8).

**Why:** per-class calls would multiply Stage 3's call count by roughly the
number of merged classes (~10, based on Opus 5's B3 run) on top of Stage 1's
192 — real, recurring cost for a stage whose job is wording cleanup within an
already-bounded, already-merged class list, not a judgment call hard enough
to need the isolated full attention Stage 1's per-document design exists to
give. A single call handling every class's attributes at once is not
expected to face the same "many documents compete for attention" problem
Stage 1 was built to solve (capability (a)), since ten-or-so classes' worth
of attribute wordings is a far smaller, far more uniform input than 192
heterogeneous raw documents.

**Revisit condition, stated before any run so it is checked rather than
assumed:** if a real run's merged-class list turns out large enough that one
class's attribute array alone risks the output cap (the same class of risk
B3-D3 already tracks for Llama's 4,096-token Bedrock ceiling, though neither
active condition in this repo's registry carries that specific limit), that
is grounds to reopen this decision for the affected condition specifically —
not to lower every condition's granularity by default.

## P1-D8 — Stage 2's input is pre-merged for literal-exact duplicates before the call, fixing a real repetition-degeneration failure

**The failure, first.** A real full-corpus Haiku 4.5 run
(`results/findings/P1-FINDINGS.md`, run `2026-08-21T21:32:24Z-ed15`) truncated
at Stage 2's 16,000-token output cap, not on a clean cutoff but mid a
mechanical repetition loop: after correctly starting to merge classes, the
model's `Tenant` attribute array became an unbroken `"name", "email address",
"name", "representative", "name", "operating as", "name", "business name",
...` cycle that consumed the rest of its budget without ever finishing that
one class. Direct measurement against the real data showed why: `Tenant` was
extracted, under that exact literal name, from 48 of the 192 documents, and
the prior prompt design concatenated attributes "as-is" — the raw union
Stage 2 had to hold and reproduce was built almost entirely of the word
`"name"` recorded once per document. Asking a model to faithfully reproduce a
long, already-repetitive list is close to handing it a template for the loop
it fell into.

**Decision:** before Stage 2's prompt is built, collapse every class sharing
the exact same name — case- and whitespace-folded, nothing more — across all
documents into one entry, deduping exact-repeat attributes the same way, and
record an `occurrences` count on each collapsed entry
(`pipeline/nodes/consolidate_types._pre_merge_literal_duplicates()`).

**Why this is not a step toward doing Stage 2's own job in Python, and stays
strictly separate from it.** Stage 2 exists to decide which *differently*
-spelled classes are the same real thing — that is a judgment call, and this
fix makes none. Collapsing two occurrences of the identical literal string
`"Tenant"` requires no judgment at all; it is the same literal-only standard
`baselines.b3_single_shot.single_shot.clean_schema()`'s `_dedup_key()` already
applies everywhere else in this project (B1, B3, and Stage 1's own per-document
cleanup), just applied here across documents rather than within one response.
Nothing about which *distinct* wordings exist, or which of them the model
should judge to be the same real-world kind, changes.

**Why `occurrences` is carried into the prompt rather than dropped.** Stage
2's prompt instructs the model to "choose the name that appears most often"
when merging genuinely different spellings. Collapsing `Tenant`'s 48
literal-identical occurrences into one entry would otherwise erase that
frequency signal — the model would see `Tenant` (appearing once, post
-collapse) and `Lessee` (appearing once, if never repeated) as equally
common, when the real corpus evidence overwhelmingly favors `Tenant`. The
`occurrences` field preserves exactly the count needed for that one
instruction, and the prompt explicitly tells the model not to echo it back.

**Measured effect, against the real failing data, not a synthetic case:**
re-running the same 192 real extraction responses through the new
pre-merge shows 755 raw class entries collapsing to 180 literal-distinct
ones, and the prompt payload shrinking from 131,151 to 39,435 characters —
`Tenant` itself goes from a raw, repetition-heavy union down to 34 deduped
attributes with `occurrences: 48` attached. **This is a measurement of the
fix against the data that caused the failure, not a claim that the failure
is fully resolved** — a live re-run is what actually confirms whether a
smaller, non-repetitive input prevents the loop outright; see
`results/findings/P1-FINDINGS.md`'s own open-questions note.

**What did not change:** the placeholder is renamed `{{CLASSES}}` (was
`{{SCHEMAS}}`) since the prompt's input is now genuinely a flat class list,
not nested per-document schemas — an honest rename, not cosmetic. `merge_log`
and `class_name_map`'s provenance (P1-D5) are computed from the *original*,
un-pre-merged per-document data, independently of what the prompt contains —
the pre-merge only changes what the model is shown, never what Python already
knows about where each name came from.

---

## P1-D9 — Stage 2 returns class identity only; attributes are re-attached in Python

**Decided 2026-08-24, after a third real full-corpus Stage 2 failure. Supersedes
the attribute-concatenation half of the Stage 2 prompt; P1-D8's input pre-merge
stays exactly as it was.**

P1-D8 shrank what Stage 2 is *shown*. It did nothing about what Stage 2 is asked
to *write back*, and that turned out to be the binding constraint. The prompt
still required every merged class to echo its full attribute array, so the size
of a correct response was fixed by the corpus, not by how much merging the model
actually decided to do.

**Measured on the real 192-document run** (`2026-08-21T21:50:44Z-484f`'s
checkpoint, replayed offline — 180 literal-distinct classes carrying 820
attributes):

| Stage 2 response shape | required output | share of `haiku45`'s 16,000 cap |
|---|---|---|
| attributes echoed (before this decision) | ~13,000 tokens | **81%**, best case |
| identity only (after) | ~5,950 tokens | 37% |

The 81% figure is the *best possible* case — zero merges, every entry echoed
once, no `merged_from` bookkeeping beyond the minimum. There was no margin at
all, and this single number is the common cause behind all three recorded Stage
2 failures (`results/findings/P1-FINDINGS.md`), which had until now been treated
as three unrelated incidents:

- The pre-P1-D8 Haiku run needed ~40,000 output tokens against a 16,000 cap. It
  was structurally impossible, and the repetition loop was the model failing at
  an unachievable transcription job rather than degenerating spontaneously.
- The post-P1-D8 Haiku run needed ~13,000 tokens, which takes long enough to
  generate that it tripped a 60-second transport read timeout (B3-D3, revised a
  fourth time). Raising `read_timeout` to 300s was necessary but treats the
  symptom.
- Llama 3.1 8B's cap is 4,096 tokens. It could not have fitted a correct answer
  at *any* prompt size, and answering with a Python script that would compute
  the result is a more reasonable response to that situation than it first
  appeared. **This materially revises the reading of the Llama finding.**

**The decision.** The model returns `name`, `parent` and `merged_from` and
nothing else. `_restore_attributes()` rebuilds each merged class's attribute
union in Python from exactly the pre-merged entries named in `merged_from`,
deduping literal repeats first-surface-form-wins (Critical Rule 5).

**Why this is not the model's job being taken away.** Stage 2's scope has been
"class identity only" since it was written — attribute wording is Stage 3's, and
the module docstring says so. The concatenation rule the prompt used to spell
out (*"drop a literal repeat, same wording ignoring case and surrounding
whitespace"*) is byte-identical to what `_pre_merge_literal_duplicates()` and
`clean_schema()` already do in Python everywhere else in this project. No merge
judgment moved out of the model: which differently-spelled entries are the same
kind of thing is still entirely its decision, and that decision is still what
`merge_log` records.

**Attributes are still shown in the prompt.** Their overlap is genuine evidence
for whether two differently named entries denote the same kind of thing, and
input tokens are not the constrained resource here. Only the echo is removed.

**New instrumentation, forced by this change.** Before P1-D9, an input entry the
model silently forgot still left its attributes visible in the response text.
Now a forgotten entry is simply absent, so two counts are written into the raw
record: `input_entries_unaccounted_for` (entries neither kept nor named in any
`merged_from`) and `unresolved_source_names` (names in a `merged_from` that were
never in the input, which prompt rule 1 forbids). Both are logged, never raised
on — aborting on an imperfect-but-usable merge would discard a paid call, and
how faithfully each model obeys the account-for-every-entry rule is itself a
result this project reports.

**Not yet confirmed by a live run.** The table above is an offline replay against
the real corpus. Whether Haiku actually completes Stage 2 under the new contract
is what the next run answers.

**What this does not fix:** ~5,950 tokens still exceeds `llama318b`'s 4,096-token
cap. P1-D9 does not rescue the open-weight condition at Stage 2, and no claim is
made that it does.
