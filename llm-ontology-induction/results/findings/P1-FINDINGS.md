## At a glance

P1's six-stage pipeline (`pipeline/DECISIONS.md` P1-D1) has three real
full-corpus runs behind it so far, all aborted at Stage 2, none scored — the
same convention `B3-FINDINGS.md` uses for B3's own truncated whole-corpus Llama
attempt. Stage 1 is not implicated in any of them: extraction has completed
essentially perfectly every time (192/192 for Haiku, 191/192 for Llama).

**The three failures were originally recorded as three unrelated incidents.
A measurement taken 2026-08-24 shows they share one cause** — see the section
below. Both of the earlier sections are left as originally written, with their
readings corrected in place rather than rewritten, so the sequence in which this
was actually understood stays legible.

---

## All three Stage 2 failures have one cause: the response was nearly as large as the corpus

**Established 2026-08-24**, by replaying the third run's checkpoint
(`2026-08-21T21:50:44Z-484f`) offline and measuring what a *correct* Stage 2
response would have had to contain. Not inferred from the failures — measured
independently of them, then compared.

Until P1-D9, Stage 2's prompt required every merged class to echo its full
attribute array. Over the real corpus — 180 literal-distinct classes carrying
820 attributes — that fixes the size of a correct answer at **~13,000 output
tokens**, regardless of how much merging the model actually performs:

| condition | output cap | required Stage 2 output | outcome |
|---|---|---|---|
| Haiku 4.5, pre-P1-D8 | 16,000 | ~40,000 tokens | impossible → repetition loop |
| Haiku 4.5, post-P1-D8 | 16,000 | ~13,000 tokens (**81% of cap**) | ~2–3 min generation → 60s transport timeout |
| Llama 3.1 8B | 4,096 | ~13,000 tokens | impossible at any prompt size → wrote code |

This reframes each of the three sections below:

- **The Haiku repetition loop was not spontaneous degeneration.** The model was
  asked for roughly 40,000 tokens through a 16,000-token cap. No response could
  have succeeded. The loop is what failing at an impossible transcription job
  looked like.
- **The `ReadTimeoutError` was not an unrelated infrastructure problem.** It was
  the direct consequence of asking for 13,000 tokens in one blocking,
  non-streaming call. Raising `read_timeout` to 300s (B3-D3, revised a fourth
  time) was necessary and is retained, but it treats the symptom.
- **Llama's "task-framing failure" reads differently now.** Its cap is 4,096
  tokens against a ~13,000-token requirement. It could not have fitted a correct
  answer under any prompt phrasing, and responding with a script that would
  *compute* the answer is a more defensible reaction to that than the original
  write-up allowed. The open question below asking whether rephrasing the prompt
  would fix it is answered: no, and rephrasing was never the relevant variable.

**The fix (P1-D9).** Stage 2 now returns `name`, `parent` and `merged_from`
only; attributes are re-attached in Python from exactly the entries named in
`merged_from`. Required output drops from ~13,000 to ~5,950 tokens — 37% of
Haiku's cap. Attributes are still *shown* in the prompt, since their overlap is
evidence for the merge judgment; only the echo is removed. No merge decision
moved out of the model.

**Two caveats, stated plainly.** ~5,950 tokens still exceeds Llama's 4,096-token
cap, so P1-D9 does not rescue the open-weight condition at Stage 2. And the table
above is an offline replay against real data, not a completed run — whether Haiku
now finishes Stage 2 is what the next run answers, and nothing here should be
read as it having already done so.

**A related observability defect, fixed at the same time.** The third run's log
contained only its 192 extraction records and nothing else, because
`invoke_or_abort()` caught only `ModelResponseError` — a transport-level
exception propagated with nothing written. The run therefore *looked* like it had
quietly stopped after Stage 1 rather than died in one identifiable call, and was
initially misdiagnosed that way. Transport failures are now logged with
`stop_reason: "transport_error"` before the exception continues.

---

## P1 — Decomposed pipeline: Llama 3.1 8B (open-weight, AWS Bedrock) — aborted at Stage 2, not scored

**Run ID:** `2026-08-21T21:21:58Z-082a`
**Attempted:** full corpus, 192 documents. Reproduce with `--model llama318b`.
**Result:** Stage 1 (extraction) completed in full — **191 of 192 calls succeeded**,
one truncated at the model's 4,096-token Bedrock cap (`csv_exports/
leases_export_03.csv`, `stop_reason: "length"`, skipped per Stage 1's
skip-and-continue design, exactly as intended). Stage 2 (type consolidation)
then made its one call over all 191 partial schemas and **aborted**: the
response was not a truncation, a refusal, or malformed JSON — it was prose
introducing a Python script, with no `classes`/`relations` object anywhere in
it. `parse_response()` correctly found nothing to parse and raised
`ResponseParseError`; the pipeline aborted per P1-D1's single-call-must-abort
rule rather than assembling anything from it. No output JSON exists for this
run — Stage 6 never ran.
**Verified**, not inferred: every number above is read directly from
`results/raw/2026-08-21T21:21:58Z-082a_p1_llama318b_calls.jsonl`, not
estimated.

### The failure, verbatim

Stage 2's call: 191 partial schemas, a 149,298-character prompt, 461
completion tokens, `stop_reason: "stop"` — the model finished normally, it did
not hit its own output cap. What it returned:

> *"Here is the Python code to solve the problem:*
> ```python
> import json
> from collections import defaultdict
> # Load the data
> data = []
> with open('data.json') as f:
>     data = json.load(f)
> # Create a dictionary to store the merged classes
> merged_classes = defaultdict(lambda: {'name': '', 'parent': None, 'attributes': [], 'merged_from': []})
> ...
> ```
> *This code first loads the data from the JSON file, then iterates over the
> partial schemas and their classes..."*

The script it wrote is a genuinely reasonable *implementation* of what Stage 2
asks for — case-insensitive-ish merge by name, attribute concatenation,
`merged_from` tracking — which makes this more interesting than a bare
refusal or a garbled response: **the model understood the task well enough to
describe a correct algorithm for it, but answered with a description of how
to do the task rather than doing it.** This is a task-framing failure, not a
capability-ceiling one in the usual sense.

### This is a different failure mode from B3's Llama failure, not the same one recurring

B3's whole-corpus Llama run (`B3-FINDINGS.md`) failed by **generation
degeneration**: a real start (7 correctly-extracted classes) collapsing into a
mechanical, combinatorial repetition loop that ran until the 4,096-token cap
cut it off mid-cycle. That is a long-context-generation coherence failure.

This P1 Stage 2 failure has **no repetition, no combinatorial loop, and did
not hit the output cap** (`stop_reason: "stop"`, 461 of 4,096 tokens used).
The model produced a short, coherent, well-formed response — it was simply
the wrong kind of response. Two distinct manifestations of the same
underlying model's difficulty with a large, structured-consolidation task,
worth reporting as two separate data points rather than folded into one
"Llama struggles at scale" sentence.

### Not the truncation risk P1-D3 already flagged — a different, unanticipated one

`pipeline/DECISIONS.md` (P1-D3, revised 2026-08-21) predicted Stage 2 was "the
stage most likely to test" Llama's 4,096-token ceiling, via truncation. That
prediction was written before this run and is **not what happened** — Stage 2
finished well under its cap. The actual failure (answering with code instead
of a direct result) is a new, unanticipated failure mode, not a confirmation
of the one already on record. Stated explicitly so a reader doesn't conflate
the two: this run neither confirms nor refutes the truncation prediction, it
surfaces a different problem that arrived first.

### What this means for P1-D4's weak-model hypothesis — complicated, not confirmed

P1-D4 predicts decomposition should help a weak/fragmenting model more than a
capable one. This result is a genuine complication, not a confirmation:
**decomposition's own Stage 2 is where Llama broke**, on a task shape (hold
191 documents' worth of extracted structure and consolidate it in one call)
that B3's single whole-corpus call never asks Llama to do at all — B3 asks it
to read 192 raw documents and extract directly, once. Six-stage decomposition
gave Llama's Stage 1 exactly the capability-(a) benefit it was designed to
give (191 of 192 extraction calls succeeded cleanly, each with full attention
on one document), and then lost that gain at the very next stage, on a job
novel to this architecture. Whether this generalizes or is specific to this
one prompt's phrasing is not yet answerable from one run.

### Open questions / next steps

> **Corrected 2026-08-24.** The first question below is answered, and not in the
> direction it assumed. Llama's cap is 4,096 tokens; the correct Stage 2 answer
> required ~13,000. No phrasing of the prompt could have made a correct response
> fit, so prompt wording was never the operative variable — see the
> one-cause section at the top of this file. The remaining questions stand.

- ~~**Does rephrasing Stage 2's prompt for weaker models fix this, or does the
  failure recur under any sufficiently large consolidation prompt?** Not yet
  tested. Retrying the identical prompt will not help (`TEMPERATURE = 0.0`
  reproduces the identical response) — this needs a deliberate prompt change
  to investigate, not a retry.~~ Answered: the required output exceeded the
  model's entire output cap by ~3×. Even under P1-D9's much smaller
  ~5,950-token requirement it still does not fit.
- **Does Haiku 4.5 hit the same wall at Stage 2, a different one, or none at
  all?** Directly testable, not yet run for real (only a `--limit 8` smoke
  test exists so far, which completed all six stages cleanly — see the
  pipeline architecture discussion this file's companion analysis covers).
  This is the comparison P1-D4 actually needs: two models, identical
  architecture, to see whether decomposition's benefit is capability-tier
  dependent the way the hypothesis predicts.
- **Is 191 documents' worth of consolidation input simply too large a single
  ask for an 8B model**, independent of how the prompt is worded? If Haiku
  handles the identical-sized Stage 2 input without difficulty, that would
  point at model capability rather than prompt design as the driver — worth
  checking directly rather than assuming either way.

---

## P1 — Decomposed pipeline: Haiku 4.5 (budget tier, AWS Bedrock) — aborted at Stage 2, a different mechanism than Llama's

**Run ID:** `2026-08-21T21:32:24Z-ed15`
**Attempted:** full corpus, 192 documents. Reproduce with `--model haiku45`.
**Result:** Stage 1 completed **perfectly** — all 192 extraction calls
succeeded, every one `stop_reason: "end_turn"`, zero truncations, zero parse
errors. Stage 2 (type consolidation) made its one call over all 192 partial
schemas (a 133,965-character prompt) and **truncated at its own 16,000-token
output cap** (`stop_reason: "truncated"`, `completion_tokens: 16000`). The
pipeline aborted per P1-D1's single-call-must-abort rule. No output JSON
exists for this run either.
**Verified** directly from `results/raw/2026-08-21T21:32:24Z-ed15_p1_haiku45_calls.jsonl`.

### Not a clean cutoff — a repetition-degeneration loop, the same failure category as B3's Llama run, happening to a different model at a different stage

The response was not 16,000 tokens of genuinely distinct content interrupted
mid-thought. It correctly started merging real classes — `Lease`, then
`Commercial Lease Agreement` (itself never merged with `Lease`, the same
near-duplicate-not-merged pattern documented for Opus 5's B3 run and for this
same model's own `--limit 8` smoke test) — then began `Tenant`'s attribute
array and entered an unbroken alternating loop for the rest of its output
budget, extracted verbatim from the raw response:

> `"name", "email address", "name", "representative", "name", "operating as",
> "name", "business name", "name", "entity represented", "name", "business
> name", "name", "business name", "name", "entity represented", ...`

`Tenant` never finished. None of the other classes 192 documents should
produce were ever reached.

**Why, verified by direct measurement, not guessed:** re-parsing all 192 raw
extraction responses from this exact run and rebuilding what Stage 2's prompt
actually contained shows `Tenant` was extracted, under that literal name, from
**48 separate documents** — the prior consolidation prompt design
concatenated attributes "as-is" per Stage 2's own instruction, so the raw
union Stage 2 had to hold and reproduce for `Tenant` alone was built almost
entirely of the word `"name"` recorded once per document. Asking the model to
faithfully reproduce a long, already-mechanically-repetitive list is close to
handing it a template for exactly the loop it fell into.

### How this differs from Llama's Stage 2 failure, precisely

Llama (this file's first section) answered the wrong task — task-framing
failure, no truncation, no repetition, a short and coherent (if useless)
response. Haiku understood the task correctly, executed it competently on the
first two classes, and then degenerated into mechanical repetition on the
third — a generation-coherence failure, not a comprehension one. It is the
same *category* of failure B3's whole-corpus Llama run showed (a real start
collapsing into cyclical repetition, cut off by the output cap mid-loop,
`B3-FINDINGS.md`), but happening to the *stronger* model this time, at a
*different* stage, triggered by a *different* mechanism (a naturally
repetitive attribute union, not long-context narrative generation). Three
data points now exist for "repetition under a large generation task," each
from a different concrete cause — worth treating as a pattern across this
project's models and stages, not three unrelated incidents.

### The fix: collapse literal duplicates in Python before the call (P1-D8)

Implemented and validated against this exact run's real data (not a
synthetic case): `pipeline/nodes/consolidate_types._pre_merge_literal_duplicates()`
now collapses every class sharing the exact same name (case/whitespace-folded
— a literal match, not a merge judgment; Stage 2's actual job, deciding
*differently*-spelled duplicates, is completely untouched) before the prompt
is built. Re-running this exact failing prompt construction offline against
the same 192 real extraction responses:

| | Before the fix | After the fix |
|---|---|---|
| Classes fed into the prompt | 755 (one entry per document per class) | 180 (one per literal-distinct name) |
| Prompt payload size | 131,151 chars | 39,435 chars |
| `Tenant`'s attribute count | ~48 raw entries, mostly repeats of `"name"` | 34 deduped attributes, `occurrences: 48` preserved |

The `occurrences` count is carried into the prompt specifically so Stage 2's
own "choose the most frequent wording" instruction still has the frequency
signal it needs when later deciding between *different* spellings — the fix
removes the trivial, mechanical part of concatenation, not the judgment part.
**Not yet re-run for real** — this table is a measurement of the fix against
the actual data that caused the failure, not a new completed run. The next
full Haiku run is what confirms whether it actually prevents the loop rather
than only shrinking the input.

### Open questions / next steps

> **Corrected 2026-08-24.** The first question below was the right one to ask,
> and the answer was "only delay it." The P1-D8 input pre-merge did not prevent
> the loop, because input size was never the binding constraint: the *response*
> was still fixed at ~13,000 tokens, 81% of this model's entire cap. The next
> run after this section was written failed again, at the same stage, on a
> transport timeout caused by exactly that generation length. See the one-cause
> section at the top of this file, and P1-D9.

- ~~**Does the fix actually prevent the loop, or only delay it?** The measured
  70% size reduction is real, but repetition degeneration in small-to-mid
  models is not always simply a function of input size — worth confirming
  with the actual next run rather than assuming the offline measurement
  settles it.~~ Answered: only delayed it. Input size was the wrong variable;
  the constraint was output size, addressed in P1-D9.
- **Does Llama's Stage 2 failure improve under the same fix?** Untested.
  Llama's failure was task-framing (wrote code), not repetition — the fix
  targets the mechanism behind *this* run's failure, and there is no
  evidence yet that a smaller, less repetitive prompt changes Llama's
  behavior at all.
- **Three repetition-category failures across two models and two stages** (B3
  whole-corpus Llama, P1 Stage 2 Haiku, and structurally the same risk
  flagged but not yet observed for P1 Stage 2 Llama) suggests this may be a
  property of "ask a small-to-mid model to hold and reproduce a large,
  structurally repetitive block" in general, not specific to any one prompt.
  Worth a deliberate, dedicated test once more runs exist, rather than
  treating each occurrence as independent.
