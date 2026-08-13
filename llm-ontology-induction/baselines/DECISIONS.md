# B1 Baseline — Design Decisions

Frozen decisions for **B1**, the zero-LLM statistical schema-induction baseline
(`baselines/term_extraction.py` + `baselines/statistical.py`).

B1 uses term-frequency statistics (C-value) and dependency parsing only. Its
scientific purpose is to establish what a purely distributional method recovers
from a messy corpus **without any semantic knowledge of the domain**. That
purpose is destroyed if gold-schema vocabulary leaks into the extraction logic,
so the rules below are constraints on the method, not just notes about it.

Everything in this file was fixed **before B1 was ever scored against the gold
schema**. See D7.

---

## D1 — One pooled run over the whole corpus, not per-document

Schema induction needs cross-document evidence — no single messy document
contains the full picture, and a per-document run would have almost no
frequency signal to work with.

**Decision:** B1 reads all **192** documents under `data/documents/`
(`csv_exports/` 12, `lease_texts/` 50, `notes/` 65, `messages/` 65) as one
pooled corpus and emits exactly **one** output schema.

This also gives the statistical method its best possible shot — more text means
a more reliable frequency signal — which matters for B1 to be a fair,
non-strawman baseline rather than a rigged comparison.

## D2 — Term scoring: C-value, not raw frequency

Raw frequency favors short, generic words ("the", "date", "amount"). C-value
(Frantzi et al.) is built for exactly this problem: it rewards multi-word terms
and subtracts out the frequency a short term only has because it is nested
inside longer, more specific terms.

**Decision:** implement C-value directly rather than add a terminology-extraction
dependency — it is a short, well-defined formula, and implementing it keeps full
control over the candidate patterns feeding it.

For a candidate term `a` of length `|a|` tokens:

```
if a is NOT nested in any longer candidate term:
    C-value(a) = length_factor(|a|) * freq(a)
else:
    Ta = set of longer candidate terms containing a
    C-value(a) = length_factor(|a|) * ( freq(a) - (1/|Ta|) * sum(freq(b) for b in Ta) )
```

### D2a — The `log2(1) = 0` edge case (decided on formula hygiene, not on scores)

The literature's length factor is `log2(|a|)`, which is **0 for every
single-token term** — it would zero out an entire length class by construction,
regardless of how dominant those terms are in the corpus. That is an artifact of
the formula, not a finding about the corpus.

**Decision:** `length_factor(n) = log2(n + 1)`.

This preserves C-value's intent (monotonically increasing in term length:
1.000, 1.585, 2.000, 2.322 for lengths 1–4) while never annihilating single-token
candidates a priori. It is applied uniformly to all lengths so relative ranking
across lengths stays meaningful.

This choice was made from formula reasoning alone, before any scoring run, and
is deliberately *not* revisited afterwards — see D7. It is a real methodological
choice and is reported as such rather than buried.

## D3 — Candidate terms are POS patterns, never a vocabulary list

Candidates are contiguous token spans matching `(ADJ|NOUN)* NOUN`, 1–4 tokens,
extracted with spaCy POS tags. Determiners are excluded (spaCy's raw
`noun_chunks` include them: "The owner"), so the pattern is applied over tokens
directly rather than taking chunk text verbatim.

**Hard rule: no hardcoded domain vocabulary anywhere in `term_extraction.py` or
`statistical.py`** — not to seed extraction, not to filter it, not to validate
it. Writing `["owner", "tenant", "lease", ...]` into the code would make B1 an
oracle that already knows the answer key, and every downstream comparison in the
paper would be invalid. B1 must discover candidates from the documents alone and
only meet the gold schema at scoring time.

Verified by grep over both modules for every gold class, attribute, and relation
name (Definition of Done, Phase 5).

### D3b — Tag case-folded text; proper nouns stay excluded

`PROPN` is excluded from the candidate pattern, because proper nouns in this
corpus are instance data (people, companies, street addresses), not schema
vocabulary. That exclusion is only defensible in combination with the
following, and the two must be read together.

**Observed problem.** Capitalization in this corpus is a *formatting
convention*, not a linguistic fact. Contracts capitalize defined terms
mid-sentence ("the Tenant", "the Lessee") and CSV headers are title-cased
("Lease Owner", "Commence Date"). spaCy therefore tags precisely the
schema-level terms as `PROPN`, while their lowercase synonyms in the same
corpus ("landlord", "lease") stay `NOUN`. Measured on `lease-001.txt`:
`Tenant` (PROPN ×3) and `Lessee` (PROPN ×2) vs. `landlord` (NOUN ×2). CSV
pseudo-sentences yielded **zero** candidate terms. An arbitrary orthographic
convention was deciding which terms B1 could see at all.

**Decision:** run the POS tagger over a **lowercased copy** of each sentence,
and recover the emitted surface form from the original text by character
offset.

Lowercasing is generic text preprocessing — no vocabulary, no domain
knowledge, applied uniformly to every sentence. It is **not** output cleaning:
names are sliced from the original text and keep their real casing, spacing and
pluralization, per Rule 2 / D3a. Verified effect: `tenant`, `lessee`,
`lease owner`, `commence date` become visible, while `Philip Cannon` and
`Wagner & Associates` remain `PROPN` and stay excluded.

**Rejected alternative:** filtering instance data with spaCy's NER instead. On
this corpus NER labels `Tenant` as PRODUCT and `Lessee` as PERSON — it deletes
the exact terms B1 needs, and on CSV rows it swallows `Lease Owner: Wagner &
Associates` as a single ORG span. Recorded here because "just filter named
entities" is the obvious reviewer question, and the answer is that it was tried
and measured, not overlooked.

### D3a — Surface-form aggregation

Frequency counts are aggregated on a **case-folded key** (so "Tenant" at
sentence start and "tenant" mid-sentence do not split one term's count into two
sub-threshold halves and vanish), but the emitted name is the **most frequent
surface form actually observed** for that key.

This is form *selection*, not normalization: casing, spacing, and pluralization
are emitted exactly as they appear in the corpus. Per `parse_induced_schema()`'s
contract — *the harness normalizes, the producer never pre-cleans* — B1 does no
lowercasing, singularizing, or tidying of output names.

## D4 — Class/attribute split: a frequency-and-position heuristic

A purely statistical method cannot reliably tell a class ("Owner") from an
attribute ("tax id"). This split is a heuristic and is reported as one.

**Decision:**

- The **top-N C-value terms become class candidates** (N = 20, D7).
- For each class candidate `C`, an **attribute** is a shorter term `T`
  co-occurring with `C` inside the same sentence (for CSV rows: the same
  flattened pseudo-sentence) in one of these generic grammatical patterns:

  | Pattern | Form | Example shape |
  |---|---|---|
  | Possessive | `C 's T` | `X's start date` |
  | Prepositional | `T of C` | `end date of X` |
  | Colon | `C : T` | CSV/notes `X: value` |
  | Descriptive compound | noun chunk `C T` | header `X Name` → attr `Name` |

  All four are grammar, not vocabulary. The compound pattern matters most for
  CSV headers, which is precisely where tabular attribute names live.

- Kept if co-occurrence count ≥ 3, capped at the 10 highest-count attributes per
  class (D7).

### D4a — Two corrections found while building, before any scoring run

1. **"Shorter than the class" was degenerate.** As first written, an attribute
   had to be strictly shorter in tokens than its class. Most class candidates in
   a real corpus are single-token, so that rule makes attributes *structurally
   impossible* for them — nothing can be shorter than one token. Replaced with
   what the constraint was actually for: an attribute may not be a class
   candidate, may not equal its class, and may not contain its class as a token
   subsequence (so `tenant name` never becomes an attribute of `tenant`).

2. **The prepositional pattern never fired.** Checking `doc[start-1] == "of"`
   misses the determiner that normally sits between: in "the start date **of
   the** lease", the token before `lease` is `the`. The matcher now steps back
   over determiners.

Both are correctness fixes to a rule that could not fire as specified, found by
unit test and fixed **before B1 was scored against gold even once** — not
adjustments made in response to a score (D7).

### D4b — Known limitation: the colon pattern inverts on tabular data

In a CSV pseudo-sentence `Lease Owner: Wagner & Associates`, the *attribute
name* is the header and the *value* is what follows the colon — but the `C : T`
pattern reads it the other way round and treats the value as the attribute.
Most values escape anyway (proper nouns and numbers are not candidate terms),
but common-noun values (`retail`, `standalone`, `medium`, `open`) can be
attached as attributes.

Kept as specified rather than special-cased. It is a genuine limitation of a
grammar-only heuristic applied to tabular data, and it belongs in the paper as
evidence for D4's larger point — that separating entities from properties is
exactly what a statistical method cannot do reliably — rather than being
engineered away.

Be upfront in the paper about this limitation rather than engineering around it.
A statistical method genuinely struggling to separate entities from properties
**is itself a finding**, not a flaw to hide — it is exactly what motivates the
multi-stage pipeline (P1) having separate Stage 3 (type induction) and Stage 4
(attribute induction) instead of doing both at once.

## D5 — Relations: dependency-parse SVO triples, frequency-thresholded

For each sentence (prose leases, notes, message turns) and each CSV row
(flattened to `"{col}: {val}, {col}: {val}, ..."`), run spaCy's dependency
parser and extract `(subject, verb_lemma, object)` triples where:

- subject and object are **both class candidates** from D4, and
- the verb is **not a copula** (`be`/is/are/was/were — a copular link asserts
  type identity, not a relation), and
- the triple occurs **≥ 3 times** across the pooled corpus (D7).

No hardcoded verb list (`owns`, `leases`, …) — same oracle concern as D3.
Endpoints are emitted as the class's D3a surface form so relation endpoints
resolve against the emitted class names (the harness's D4 endpoint conditioning
requires both endpoints to match for a relation to count).

## D6 — Every class's `parent` is `null` — no taxonomy detection

B1 has no mechanism to infer `Owner isSubclassOf Party` from term statistics
alone, and inventing one would misrepresent what the method is.

**Decision:** every emitted class has `"parent": null`.

**Expect and report a taxonomy-layer F1 of exactly 0.0 for B1 in every table.**
This is not a bug to patch. It is the actual, correct, informative answer to
"what does hierarchy induction cost you if you skip it?" — a nonzero taxonomy
score here would indicate a bug, not a success.

## D7 — Frozen hyperparameters

Chosen from term-extraction convention and a shape sanity check (does the output
look like a plausible schema — not zero classes, not 400?), **never** by running
the harness and adjusting until the score improved. Tuning against gold would
silently collapse B1 into a second copy of the harness's own thresholds and
defeat the purpose of an independent baseline.

| Parameter | Value | Basis |
|---|---|---|
| `MAX_TERM_TOKENS` | 4 | Standard C-value candidate ceiling; longer spans are phrases, not terms |
| `MIN_TERM_FREQ` | 3 | Drops hapax/near-hapax noise before C-value, per Frantzi et al.'s frequency-threshold step |
| `LENGTH_FACTOR` | `log2(n+1)` | D2a — avoids zeroing single-token terms |
| `TOP_N_CLASSES` | 20 | Plausible schema size; deliberately loose (~2× a typical hand-authored schema) so B1 is not starved of recall |
| `MIN_ATTR_COOC` | 3 | Same near-hapax logic as `MIN_TERM_FREQ`, applied to the pattern hit count |
| `MAX_ATTRS_PER_CLASS` | 10 | Generic plausibility bound — an attribute list beyond ~10 is not a hand-authored schema shape |
| `ATTR_WINDOW` | same sentence | Standard noun-co-occurrence window; CSV row = one pseudo-sentence |
| `MIN_RELATION_FREQ` | 3 | Requires cross-document corroboration, not one parse fluke |

**These values are frozen as of this file.** They were set before B1's first
scoring run and must not be changed in response to a score. If they ever change,
every previously reported B1 number must be re-run and the change recorded here.

---

## Expected result shape (a prediction, not a target)

Recorded here **before** the first scoring run so the outcome can be checked
against the prediction rather than rationalized after the fact:

- **Taxonomy F1 = exactly 0.0** — by construction (D6).
- **Relation F1 low** — SVO triples over messy informal text, further bounded
  above by class F1 via the harness's endpoint conditioning.
- **Class F1 rising noticeably M1 → M3** — B1 emits corpus surface forms
  ("landlord", "lessee") that only fuzzy/semantic matching can connect to gold
  labels.

If the actual result departs sharply from this shape — nonzero taxonomy, or
class F1 unexpectedly high already at M1 — that is a **bug signal** (most
likely vocabulary leakage violating D3), not a win, and must be investigated
before the number is reported.

---

## Outcome of the first scoring run

Run `2026-08-09T22:20:37Z-54e6`, 192 documents → 1762 sentences → 20 classes,
42 attributes, 10 relations. F1 (TP/FP/FN):

| Layer | M1 | M2 | M3 |
|---|---|---|---|
| classes | 0.323 (5/15/6) | 0.323 (5/15/6) | **0.387** (6/14/5) |
| taxonomy | **0.000** (0/0/7) | **0.000** (0/0/7) | **0.000** (0/0/7) |
| attributes (effective, micro) | 0.033 (1/16/43) | 0.000 (0/18/44) | 0.094 (3/17/41) |
| attributes (declared, micro) | 0.043 (1/16/29) | 0.000 (0/18/30) | 0.120 (3/17/27) |
| relations | **0.000** (0/10/15) | **0.000** (0/10/15) | **0.000** (0/10/15) |

**The prediction held.** Taxonomy is exactly 0.0 at every level (D6). Class F1
rises M1 → M3. Relation F1 is not merely low but exactly zero. No result
suggests leakage: class precision is 0.25–0.30, and the classes B1 recovers are
corpus surface forms it had to discover (`Tenant`, `Landlord`, `Lessee`,
`lease`, `owner`, `property`, `premises`, `parties`) alongside honest noise it
could not filter (`AM` — from message timestamps `[09:12 AM]` — plus `EOW`,
`Date`, `sum`, `amount`).

### Three findings worth carrying into the paper

1. **M2 buys nothing at the class layer (0.323 → 0.323), for a structural
   reason.** B1 emits *both* the gold term and its synonym (`owner` **and**
   `Landlord`; `Tenant` **and** `Lessee`). Bipartite assignment is one-to-one,
   so gold `Owner` was already consumed by the exact match at M1; the synonym
   cannot also match and stays a false positive. A synonym lexicon only helps a
   method that emits the synonym *instead of* the gold term, not one that emits
   both.

2. **Attribute F1 *drops* at M2 (0.033 → 0.000), and the cause is in the
   harness, not in B1.** At M1 the assignment picks `Tenant`→`Tenant` and
   `Property`→`property`; at M2 the lexicon makes `Lessee` and `premises` score
   1.0 as well, the tie is broken toward the synonym, and `Tenant`→`Lessee` /
   `Property`→`premises` win instead. Class TP is unchanged at 5, but the
   *newly matched* induced classes carry different attribute lists, so the
   attribute layer loses its one true positive. **Recommendation: break M2 ties
   in favor of the stricter (exact) match.** Until then, cross-level attribute
   comparisons are confounded by assignment tie-breaking rather than by
   matching quality.

3. **Relations are 0.0 because the verbs are wrong, not the endpoints.** B1
   recovers plausible SVO shapes (`Tenant provide sum`, `parties agree terms`)
   but never the gold labels (`owns`, `covers`, `reports`). The corpus states
   relations in prose that does not put two schema-level terms in a direct
   subject-verb-object configuration, so a parse-only method reaches for
   whatever noun is nearest. This is the strongest single argument in the B1
   results for a semantically-aware induction stage.

Instance data also leaked into the *attribute* layer — `Megan Mcclain`,
`Robinson Partners`, `Lindsey Roman` attach as attributes of `Landlord`/`owner`.
This is the predictable cost of D3b: lowercasing for tagging also makes some
person and company names tag `NOUN` rather than `PROPN`. It is reported, not
patched, per D4.

---
---

# B3 Baseline — Design Decisions

Frozen decisions for **B3**, the single-shot LLM schema-induction baseline
(`baselines/single_shot.py` + `baselines/model_clients.py` +
`baselines/prompts/b3_extraction_prompt.md`).

B3 gives a model batches of raw documents under one frozen prompt, asks once for a
schema, and merges the per-batch answers with nothing cleverer than exact-string
deduplication. Its scientific purpose is to measure **what an LLM recovers with no
staged decomposition and no intelligent consolidation** — the thing P1's extra
machinery has to beat to be worth having. That purpose is destroyed if gold-schema
vocabulary reaches the models, or if B3 quietly does P1's consolidation work, so the
rules below are constraints on the method, not notes about it.

**Decision IDs are `B3-` prefixed** so that bare `D1`–`D7` references in B1's code
and docstrings keep pointing unambiguously at B1's decisions above.

Everything here was fixed **before B3 was ever run against a model**. No B3 model
call of any kind — not a smoke test, not a dummy prompt — was made before these
decisions were recorded.

---

## B3-D1 — Five frozen models: two tiers × two vendors, plus open weights

One model cannot separate "LLMs do this well" from "*this* LLM does this well". The
grid varies vendor and capability tier independently, so a difference in the results
can be attributed to one or the other.

| | Anthropic | OpenAI |
|---|---|---|
| **Frontier** | Claude Fable 5 | GPT-5.6 Sol |
| **Budget** | Claude Haiku 4.5 | GPT-5.6 Luna (`reasoning_effort="low"`) |

Plus **llama-3.1-8b-instant**, open-weight, as a fifth condition — the only one a
reader can reproduce without a vendor account, and the floor the paid tiers must
justify themselves against.

**Decision:** these five, frozen. Four run through AWS Bedrock (two request-body
shapes, one invocation path); the open-weight model runs through Groq (B3-D1c). The
exact provider identifier for each is what lands in `metadata.model` — results name
the artifact that produced them, never a friendly alias that could later be
repointed.

### B3-D1a — Luna runs at `reasoning_effort="low"`, always

The budget tier is only a controlled comparison if both budget models are actually
comparable. On the Artificial Analysis Intelligence Index, Luna at low reasoning
effort scores **33** against Haiku 4.5's **30** — the closest available match.

**Decision:** every Luna call carries `reasoning_effort="low"`. Default or higher
effort settings are *not* capability-matched: they would put a stronger model in the
budget cell and turn a vendor comparison into a reasoning-budget comparison.

Enforced, not just documented: `ModelSpec` is a frozen dataclass, and
`test_single_shot.py::test_luna_always_carries_low_reasoning_effort` asserts on the
actual request body.

### B3-D1b — Budget estimate (order of magnitude, before the first call)

**Corrected 2026-08-12:** corpus is ≈ **40,600 tokens** (≈142,000 chars across 192
documents), not the ≈35,500 first estimated — measured directly by reading every file
in `data/documents/` rather than approximated, once the Groq TPM failure below made it
worth checking precisely. The original figure was a same-order-of-magnitude
approximation, ~14% low; corrected here rather than left standing as a known-wrong
number.

At the now-revised `BATCH_SIZE = 7` (B3-D2), the corpus splits into **28 batches per
model, 140 calls across the five conditions**.

| Quantity | Per batch | Per model (28 batches) |
|---|---|---|
| Input (documents + prompt) | ≈ 1,525 docs + 645 fixed prompt ≈ **2,170** avg (ranges ~1,950–3,300 across the 27 full batches, see B3-D2's table) | ≈ **60,800** (0.061 MTok) |
| Output (schema JSON) | ≈ 800 | ≈ **22,400** (0.022 MTok) |

Luna is the only reasoning-budgeted condition and bills hidden reasoning tokens not
counted above. The open-weight condition (llama-3.1-8b-instant) is not a reasoning
model, so it carries none of that hidden cost, and Groq's free tier makes its share
zero regardless.

Across the plausible price range for the tiers involved, a full five-model sweep is
**low single-digit dollars**, and a `--limit 2` smoke test is fractions of a cent.
`# TODO: confirm exact per-model $/MTok before quoting a figure in the paper` —
the token arithmetic above is real; the dollar conversion is not yet.

### B3-D1c — Local/open-weight model moved from on-device Ollama to Groq's free tier,
and from qwen3:8b to llama-3.1-8b-instant

Tested qwen3:8b via Ollama directly on an M3 MacBook, 8GB unified memory —
confirmed infeasible (severe memory pressure). Pivoted to Groq's free,
no-credit-card API tier.

Groq's own catalog does not serve qwen3 at the 8B weight class tested locally
(only a 27B variant was available), so the open-weight condition itself changed:
it now runs **llama-3.1-8b-instant** (Groq's exact catalog ID:
`llama-3.1-8b-instant`) — genuinely 8B, open-weight, and free on Groq's tier, same
as the model this condition was originally scoped around.
`# TODO: record the fuller rationale for the qwen3 -> llama-3.1-8b-instant swap
(closest available size match vs. only 8B open-weight option on Groq) in the
paper's methods section.`

Preserves the open-weight + zero-cost properties this condition exists to test;
"physically local" was already dropped by the Groq pivot, and the model identity
changed alongside it. One consequence worth naming: unlike qwen3, llama-3.1-8b-instant
is not a reasoning model — it never emits `<think>` blocks and carries no hidden
reasoning-token cost (see the note in B3-D1b). Corpus size (~40,600 tokens total, per
B3-D1b's corrected figure) is trivially within Groq's free-tier *daily* limits; the
*per-minute* limit is a separate, tighter constraint that did bind — see B3-D2 and
B3-D3.

## B3-D2 — Fixed batch size of 7, documents interleaved round-robin across subdirectories, uniform across all five models

*Revised 2026-08-12. Was `BATCH_SIZE = 10`, documents ordered by draining one
subdirectory fully before the next. Both changed together — see the failure and the
measurements below.*

The corpus is too small to need batching for context reasons and too large to fit one
call comfortably at every tier. The number matters less than its uniformity.

**What broke the original decision:** the first real call (`llama318b`, `--limit 2`)
hit Groq's free-tier rate limit for `llama-3.1-8b-instant` — a hard **6,000
tokens/minute** cap, the tightest of the five conditions' limits by a wide margin.
Two compounding causes:

1. `csv_exports/` documents run far denser per file (~1,930 chars avg) than
   `lease_texts/`/`notes/`/`messages/` (~1,413 / ~184 / ~559 chars avg). Draining
   `csv_exports/` before moving on packed 10 of its 12 files into batch 1 alone:
   22,980 chars, ≈6,566 estimated input tokens — over the entire 6,000 TPM ceiling
   before a single output token was even reserved. No amount of output-budget tuning
   (B3-D3) rescues a batch whose *input alone* exceeds the cap.
2. At `BATCH_SIZE = 10`, batches 2–6 (still CSV/lease-heavy under the old order) were
   only marginally under the cap even after B3-D3's output-budget fix — not enough
   margin to trust against a schema-JSON response bigger than the ~800-token estimate.

**Decision:** two changes together, because interleaving alone did not leave enough
margin on its own (measured — see below):

- **Interleave, not drain-then-advance.** `load_documents()` takes one document per
  subdirectory per round (subdirectories still visited in `_SUBDIRS` order; filenames
  still sort within each subdirectory) instead of exhausting one subdirectory before
  starting the next. This spreads the dense `csv_exports/` files across many batches
  instead of clustering them in the first one or two.
- **`BATCH_SIZE = 7`**, down from 10, applied identically to all five models — extra
  margin on top of interleaving, not instead of it.

Together these give **28 batches** (was 20) at 7 documents each (final batch 3, since
192 is not a multiple of 7). Measured via dry-run (no API calls) against the tightest
of the five conditions' rate limits:

| Batch | Docs | Chars | Est. input tokens | + 2,048 reserved output | Margin vs. 6,000 TPM |
|---|---|---|---|---|---|
| 1 | 7 | 9,889 | 2,827 | 4,875 | 1,125 |
| 2 | 7 | 8,764 | 2,505 | 4,553 | 1,447 |
| **3** | 7 | 11,528 | 3,295 | 5,343 | **657 — tightest of the 28** |
| 4 | 7 | 9,587 | 2,740 | 4,788 | 1,212 |
| 5 | 7 | 10,424 | 2,979 | 5,027 | 973 |
| 6 | 7 | 9,485 | 2,711 | 4,759 | 1,241 |
| 7 | 7 | 8,847 | 2,529 | 4,577 | 1,423 |
| 8–23 | 7 | 6,807–8,441 | 1,946–2,413 | 3,994–4,461 | 1,539–2,006 |
| 24–27 | 7 | 4,825–5,565 | 1,379–1,591 | 3,427–3,639 | 2,361–2,573 |
| 28 | 3 | 3,621 | 1,035 | 3,083 | 2,917 |

Every batch clears the 6,000 TPM cap. Batch 3 (`notes/mr-004`, `messages/mr-004`,
`csv_exports/maintenance_requests_export_01`, `lease_texts/lease-005`,
`notes/mr-005`, `messages/mr-005`, `csv_exports/maintenance_requests_export_02`) is
the tightest at **657 tokens of margin (≈11%)** — it happens to catch two of the
densest individual CSV files in one round. That margin is real but not large, and it
rests on a single calibration point (chars-per-token measured from one actual Groq
error response, not a real tokenizer run against every batch); if a live run 413s on
batch 3 specifically, that is the one to look at first, and the fallback is dropping
`BATCH_SIZE` further rather than re-deriving the ratio from more guesswork.

Cloud models have context windows that would swallow the whole corpus at once. Giving
them larger batches for that reason is the tempting mistake: it would hand every cloud
model more cross-document evidence per call than the open-weight model ever sees, and
any resulting difference would be unattributable — better model, or better view of the
corpus? Uniform batches keep the task shape identical, which is the entire point of
the comparison. That reasoning is why the fix is a *smaller, uniform* batch size
rather than a Groq-only exception.

Enforced structurally: `batch_documents()` takes no `ModelSpec`, so there is no
parameter through which one model could receive different batches from another
(`test_batching_cannot_depend_on_the_model`).

## B3-D3 — Frozen sampling settings

*`MAX_OUTPUT_TOKENS` revised 2026-08-12. Was `8192`.*

| Setting | Value | Basis |
|---|---|---|
| `TEMPERATURE` | `0.0` | Schema induction has no use for sampling diversity; near-zero makes a re-run reproducible enough to debug |
| `TOP_P` | `1.0` | Neutral — with temperature at 0 it does nothing, and leaving it explicit stops a provider default from varying between backends |
| `MAX_OUTPUT_TOKENS` | `2048` | Was `8192`, chosen only to sit comfortably above the largest plausible schema. That reasoning missed a real cost: Groq's free tier appears to reserve the full `max_completion_tokens` against its per-minute cap regardless of what the model actually generates, not just count the tokens produced. The first real call confirmed this exactly — a request reporting `Requested 11186` decomposed as ≈2,994 measured input tokens + 8,192 reserved output, to the token. Against a 6,000 TPM ceiling (`llama-3.1-8b-instant`, the tightest of the five conditions' limits), that left less input budget than the batch needed even before B3-D2's batching fix. `2048` is still ≈2.5x the ~800-token estimate for a batch's schema JSON (B3-D1b) — real headroom against truncation, not a bare minimum — while freeing ~6,100 tokens of previously-wasted reserved capacity on every single call. |
| `MAX_RETRIES` | `5`, exponential backoff with jitter | Throttling is transient; a schema that silently omits a throttled batch is not |
| `reasoning_effort` | unset except Luna | B3-D1a |

Identical for all five conditions and defined once as module constants, never as
per-call arguments — a per-call temperature is a knob, and a knob on a frozen baseline
eventually gets turned.

**Known open item:** reasoning-tuned models have historically rejected any non-default
temperature. `ModelSpec.supports_temperature` exists so a model that refuses `0.0` can
omit it in one visible place rather than by special-casing the sampling settings; if
that flag has to be flipped for Sol or Luna, the change gets recorded here and every
affected number re-run.

## B3-D4 — Consolidation across batches is naive, deliberately

Each batch is answered independently, so the same entity arrives under several
wordings: `Owner` from one batch, `owner` from another, `Landlord` from a third. A
smarter merge could obviously resolve all three.

**Decision:** merge on an **exact string key, case- and whitespace-normalized, and
nothing else**. `Owner` and `owner` merge. `Owner` and `Landlord` do not. No LLM call,
no fuzzy matching, no embedding similarity, no synonym lexicon.

**Why, since a better method is right there:** cross-wording consolidation is P1's
Stage 6 — the dedicated novelty stage the pipeline exists to justify. If B3 does that
job too, then B3-vs-P1 no longer measures what staged consolidation buys; it measures
two implementations of the same idea, and the paper's central comparison collapses.
Failing `test_distinct_wordings_are_never_merged` would not mean B3 got worse — it
would mean B3 stopped being the baseline the paper claims it is.

The merge key is specifically **not** `eval.matching.normalize()`. The harness's
normalizer also singularizes and splits camelCase, which would fold `Invoice`,
`Invoices` and `invoiceRecord` together — smarter matching, and worse, it would make
B3's *output* partly a function of the harness that *scores* it. The producer does not
borrow the scorer's brain.

Expect this to cost B3 precision: the same entity emitted under three wordings is one
true positive and two false positives. That cost is the measurement, not a defect.

## B3-D5 — Output contract, identical to B1's

Same JSON contract as B1 (`eval/PLAN.md` §2, `eval/schema_ir.py::parse_induced_schema`):
`classes` (`name`, `parent`, `attributes`), `relations` (`source`, `label`, `target`),
`metadata` (`condition`, `model`, `run_id`, `source_documents`), with
`metadata.condition = "B3"` and `metadata.model` set to the exact frozen provider
identifier of whichever model produced the file.

Two rules inherited verbatim from B1 and worth restating because a model makes them
easier to break than a statistical method does:

- **No pre-cleaning** (Critical Rule 5). Names are emitted with the casing, spacing and
  pluralization the model returned. The harness normalizes at score time; the producer
  never tidies. Case/whitespace collapsing happens *only* inside the dedup key, never
  on an emitted string.
- **`parent` is `null`** unless a model volunteered hierarchy in its own response
  (Critical Rule 6). No taxonomy-inference logic exists in B3. Unlike B1 (D6), a
  nonzero taxonomy score here is *not* a bug — it means a model produced hierarchy
  unprompted, which is itself a finding.

One further rule is specific to B3: **no gold-schema vocabulary in the modules or in
the prompt** (Critical Rule 1). A gold term in the prompt would make all five models
oracles handed the answer key. The prompt is the likeliest place for this to slip in
by accident, so the leakage guard scans its full text, not just its code — and because
`eval.matching.normalize()` singularizes, ordinary prompt English collides with gold
relation labels: the prompt cannot say "list" (gold `lists`), "properties" (gold
`Property`), or "covers"/"reports"/"concerns". It says "array", "attributes",
"links" instead. Enforced by `test_no_domain_vocabulary_leakage`, which is itself
checked against a planted term so it cannot pass vacuously.

---

## Expected result shape (a prediction, not a target)

Recorded **before** the first B3 call so the outcome can be checked against the
prediction rather than rationalized after it:

- **Class F1 well above B1's 0.323–0.387**, at every matching level. If an LLM reading
  the documents does not beat C-value term frequency, something is wrong with the
  harness or the prompt, not with the finding.
- **Precision limited by B3-D4**, visibly: the same entity under multiple wordings,
  one match and the rest false positives. The clearest single argument for a
  consolidation stage.
- **Taxonomy F1 low but plausibly nonzero** — unlike B1's structural zero, a model may
  volunteer an is-a link the prompt permits but never requests.
- **Relation F1 above B1's 0.000**, but bounded by the harness's endpoint
  conditioning: a relation only counts if both endpoints matched.
- **Frontier > budget within each vendor**, and **llama-3.1-8b-instant at or below
  the budget tier**. A budget model beating a frontier model from the same vendor is
  a bug signal worth investigating before it is reported.

If class F1 comes back near-perfect at M1, suspect vocabulary leakage into the prompt
(Critical Rule 1) before celebrating.
