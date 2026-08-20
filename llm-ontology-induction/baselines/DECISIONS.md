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

---

### Revised 2026-08-19: three Anthropic conditions, direct API, Bedrock/Groq/open-weight all retired. Was the five-model grid above.

**The brief changed, not just the models.** The user's own words: *"we will use these
models from Anthropic only,"* replacing the two-vendor × two-tier grid with:

| Condition | Models | Provider | Calls |
|---|---|---|---|
| B3 | Haiku 4.5, Opus 4.8, Opus 5 | Anthropic (direct API) | 1 whole-corpus call per model |
| P1 | Haiku 4.5, Opus 4.8, Opus 5 | Anthropic (direct API) | N per-doc extraction calls + 1 consolidation call per model |

**What's dropped, and why the old reasoning was still right about the wrong
constraint.** The two-vendor grid above existed to separate "LLMs do this well"
from "*this* LLM does this well" — a real, still-valid concern in the abstract.
What actually consumed the sessions between that decision and this one was not the
grid's logic but its *transport*: Fable 5 turned out to need an AWS account
entitlement this account was never granted (a genuine "contact AWS Sales" wall, not
a missed click — see the resolved-and-then-abandoned Finding 4 below); Sol and Luna
were never reachable through `bedrock-runtime` at all, only through a
`bedrock-mantle` Responses API this codebase never built a client for; and the
open-weight condition's whole reason for existing on Groq — reproducibility without
a vendor account — was already compromised the moment it moved to Bedrock (see
B3-D1c below: *"open-weight model, paid hosting,"* not *"zero-cost"*). Three
separate transport problems, one after another, each real and each independently
blocking. Moving to direct Anthropic API calls removes every one of them at once,
because none of the three was actually about *which model* — all three were about
*how a call reaches it*.

**What's lost, stated plainly rather than left for a reader to notice on their
own:** this is no longer a two-vendor comparison. OpenAI's Sol/Luna cell and the
open-weight/Llama cell are both gone from the active grid. The vendor-independence
argument this entry opened with — "one model cannot separate general LLM capability
from this LLM's capability" — is weaker with three same-vendor conditions than it
was with a 2×2 grid, and that is a real limitation to state in the paper, not paper
over. What is preserved: two capability tiers within the surviving vendor (Haiku
4.5 budget; Opus 4.8 and Opus 5 frontier, one generation apart from each other),
which still lets the paper separate "frontier vs. budget" from "which Opus
generation" as two different axes.

**"Opus 4.8/5" was given as the frontier pick, ambiguously.** Rather than guess
which one was meant, both are real, runnable conditions (`opus48`, `opus5`) — see
B3-D1 (revised) — so the choice is visible in the results rather than made
silently on the user's behalf.

**Nothing above is deleted.** The five-model grid, B3-D1a (Luna's reasoning
effort), B3-D1b (the Groq-era budget estimate), and B3-D1c (the Ollama → Groq →
Bedrock chain) all stay as written below — they are the accurate record of what was
actually tried, in order, and why each step was abandoned. The Bedrock and Groq
runs already completed and scored (Llama batched and whole-corpus, Haiku
whole-corpus) are not deleted from `results/raw/` or from `B3-FINDINGS.md` either —
they are real, paid-for, verified evidence about how single-shot prompting behaves
at different model scales, and that evidence does not stop being true because the
transport that produced it is no longer the active one. `baselines/b3_single_shot/
model_clients.py`'s Bedrock-era content and `baselines/p1_pipeline/`'s absence
before this date are both visible in git history for anyone who wants the prior
shape back.

### Revised again, 2026-08-19: two conditions, mixed transport — Haiku 4.5 back on AWS Bedrock, only Opus 5 on the direct API. Was the three-Anthropic-condition, direct-API-only grid immediately above.

**The correction, in the user's own words:** *"We're actually gonna use Haiku from
AWS Bedrock. We're only gonna use Opus 5 from our Anthropic Keys... The rule of
thumb is Haiku from AWS Bedrock and Opus will be used from API."* This lands within
the same session as the revision immediately above — the direct-API-only grid was
never run against, just reworked again before a single live call was made.

| Condition | Models | Transport | Calls |
|---|---|---|---|
| B3 | Haiku 4.5, Opus 5 | Haiku: AWS Bedrock. Opus 5: Anthropic (direct API). | 1 whole-corpus call per model |
| P1 | Haiku 4.5, Opus 5 | Same split as B3. | N per-doc extraction calls + 1 consolidation call per model |

**Two changes, not one.** First, `opus48` is dropped: the prior revision's "given
ambiguously, so wire up both" hedge is resolved by the user naming Opus 5 alone,
repeatedly, across both the interrupted message and its full follow-up. `opus48`'s
code and registry entry are removed; it is not deleted from *this file's* history
above, which stays as the accurate record of why both were briefly wired up.
Second, Haiku moves back to AWS Bedrock while Opus 5 stays on the direct API — a
single model, calling through the transport it actually needs credentials for,
rather than the whole grid pinned to one transport. This is smaller than it looks:
the immediately-preceding revision's stated reason for leaving Bedrock entirely was
Fable 5's entitlement wall and Sol/Luna's unreachability through `bedrock-runtime`
— neither of which applies to Haiku 4.5, which was already working on Bedrock
before this session's pivot to direct-API-only (see B3-D1c below). Haiku going back
to Bedrock is not undoing that finding; it's noting the finding was never about
Haiku in the first place.

**This resolves a flagged discrepancy rather than leaving it open.** B3-D3
(revised, 2026-08-19) flagged that Haiku's `supports_top_p=False` constraint had
been *empirically confirmed only on Bedrock*, and that carrying it into a
direct-API-only Haiku condition was "carried forward as a prediction, not
re-verified" against a transport where it had never actually been tested. Moving
Haiku back to Bedrock removes the open question rather than answering it under a
new transport: the constraint is reinstated exactly as originally measured, on the
exact transport it was measured on.

**What this means for `model_clients.py`:** the module now carries two backends,
selected per `ModelSpec.backend` (`"bedrock"` | `"anthropic_api"`), with the
request-building and response-reading logic split into backend-specific pure
functions (`_build_bedrock_body`/`_build_anthropic_api_kwargs`,
`extract_from_bedrock_payload`/`extract_from_anthropic_message`) behind two
transport-agnostic entry points, `build_request(spec, prompt)` and
`invoke(spec, prompt)`. Both `single_shot.py` and `pipeline.py` call only
`invoke(spec, prompt)` and never branch on transport themselves — B3's and P1's own
code needed zero changes for this revision, which is exactly what "mixed
transport, one shared registry" is supposed to buy.

**`requirements.txt` and `.env` follow the transport, not the vendor list:**
`boto3` is back (Haiku only), `anthropic` stays (Opus 5 only), `AWS_PROFILE`/
`AWS_REGION` are restored as live values, `ANTHROPIC_API_KEY` stays. `GROQ_API_KEY`
remains present-but-unused, as it was after the prior revision — still not deleted,
still the user's credential rather than the repo's concern.

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

### B3-D1c — Local/open-weight model moved from on-device Ollama to Groq's free tier, and from qwen3:8b to llama-3.1-8b-instant — then from Groq to Bedrock

*Revised 2026-08-14: the open-weight condition now runs on **AWS Bedrock**
(`us.meta.llama3-1-8b-instruct-v1:0`), not Groq. **This changes what the condition
measures** — see the reframe at the end of this entry. The Groq entry is retained as a
separate condition (B3-D6), not deleted. Everything below is the original record.*


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

---

#### Groq → Bedrock, and the reframe it forces (revision of 2026-08-14)

The same weights are served on Bedrock as `meta.llama3-1-8b-instruct-v1:0` with a **128K
context window** and no per-minute token cap. Moving there is what dissolved the
constraint behind B3-D2 and B3-D3, so the open-weight condition stops being the one that
drags every other condition's task shape down to its hosting limit.

**Model ID: the *geo* inference ID `us.meta.llama3-1-8b-instruct-v1:0`, not the bare
`meta.` one.** Read off the model card's own regional table rather than guessed:
us-east-1 and us-east-2 are In-Region ✗ / Geo ✓, and only us-west-2 serves the bare ID
on demand. The geo ID is callable from all three US regions, which makes the region
question moot for this condition instead of pinning it to one region. `ModelSpec` also
gained a `region` field, because Sol and Luna have a genuine regional constraint that
the default cannot express.

**The reframe, stated explicitly rather than left to be inferred from a model ID.** This
cell was scoped to answer *"what does an open-weight model get you at zero cost?"* — it
was the only condition a reader could reproduce without a vendor account. **It now
answers a different question: "what does an open-weight model get you, on paid
hosting?"** Two of the three properties survive the move and one does not:

| Property | Before (Groq free tier) | After (Bedrock) |
|---|---|---|
| Open weights | ✓ | ✓ |
| Reproducible without a vendor account | ✓ | ✗ — Bedrock needs an AWS account and billing |
| Zero cost | ✓ | ✗ — priced per token like the rest |

The paid tiers still have a floor to justify themselves against, and it is still an 8B
open-weight model. But "the only condition a reader can reproduce for free" is no longer
true of this cell, and any sentence in the paper that leans on that claim has to be
rewritten. `# TODO: check the Results and Discussion drafts for a "zero-cost" or "free"
framing of the open-weight condition before submission.`

This is precisely why the Groq condition is kept alive rather than deleted (B3-D6): it
is the only thing still holding the zero-cost end of that comparison.

## B3-D2 — ~~Fixed batch size of 7~~ **Whole corpus in one call**, documents interleaved round-robin across subdirectories, uniform across all models

*Revised 2026-08-14. Was `BATCH_SIZE = 7`, applied to every model. Batching is now
reachable only through an explicit `--batch-size` and exists only to reproduce the one
run made under it (B3-D6). The interleaved document order is unchanged. **See "What
made batching wrong" at the end of this entry — the original reasoning below is kept
because it was correct about the constraint it was solving, and the constraint is what
turned out to be an artifact.***

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

---

### What made batching wrong (revision of 2026-08-14)

Everything above is a correct response to a 6,000 TPM cap on Groq's free tier. What it
missed is that the cap was **a property of one host's free tier, not of the task, and
not of any model in the grid.** Critical Rule 7 then did its job faithfully and
propagated that one host's limit to all five conditions — so every model was being
tested under the weakest-provisioned one's hosting constraint rather than a fair
like-for-like task shape. Uniformity was the right instinct; it was applied at the
wrong level.

**Two things forced the re-examination.**

1. **The measured cost of fragmentation.** The one completed batched run produced
   **67 induced classes against a gold schema of 11** (`results/findings/B3-FINDINGS.md`).
   The dominant driver was cross-batch fragmentation: 28 stateless calls, none of which
   could see what any other had already named. `Tenant` accumulated 58 attributes,
   largely the same underlying facts reworded per batch. That is the cost of the
   *hosting workaround*, and reporting it as the cost of single-shot prompting would
   have made B3 a strawman.
2. **The constraint dissolved.** Llama 3.1 8B is served on Bedrock with a **128K
   context window** (B3-D1c, revised). The corpus rendered through the frozen prompt is
   **151,831 chars ≈ 43,400 input tokens** — measured, not estimated, by rendering it.
   Every model in the grid now clears the whole corpus in one call with room to spare.

**Decision:** the corpus goes to the model in **one call**. `--batch-size` still exists,
labelled legacy in the code and the help text, reachable only when passed explicitly,
and used for exactly one thing: reproducing the 2026-08-12 Groq run (B3-D6).

**What this does *not* change:**

- **B3-D4 is not relaxed.** With one call there is one schema to merge, so the naive
  exact-string consolidation becomes a no-op. The rule is not weakened — it stops
  biting. No fuzzy matching, no embeddings, no LLM call has been added, and
  `test_distinct_wordings_are_never_merged` still passes unmodified.
- **Critical Rule 7 still holds structurally.** `batch_documents()` still takes no
  `ModelSpec`, and the call shape is decided in exactly one place in `main()` and never
  reaches `build_request_body()` — so the same model sends a byte-identical body in
  either shape (`test_the_body_is_identical_whatever_the_call_shape`). Without that,
  B3-D6's two arms would differ in two ways at once and neither result could be
  attributed.
- **Document order is untouched.** The round-robin interleaving above stays, so the
  whole-corpus prompt is literally the batched sequence concatenated. Ordering does not
  become a silent new variable between the two arms.

**What it costs, stated plainly:** whole-corpus removes fragmentation as a failure mode
*structurally*, which is a real advantage handed to B3 over the version of B3 already
run. That is the point — B3 should be the strongest honest version of "one prompt, no
staged decomposition", because P1 has to beat it. It is not prompt tuning: the prompt
(Critical Rule 2) is byte-identical and was not touched.

## B3-D3 — Frozen sampling settings

*`MAX_OUTPUT_TOKENS` revised 2026-08-14: the single global constant is gone, replaced
by a per-model `ModelSpec.max_output_tokens`. See "Why the cap had to become per-model"
below.*

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

---

### `supports_top_p`, a second per-model sampling exception (revision of 2026-08-14)

Confirmed at the first real Haiku 4.5 call, after the model-ID fix above got past the
first error: a body carrying both `temperature` and `top_p` raised **"`temperature`
and `top_p` cannot both be specified for this model. Please use only one."** This is
not a Haiku-specific quirk — it is documented behavior of Claude 4.5+ models on
Bedrock generally (also reported for Sonnet 4.5 and Opus 4.7/4.8): they reject the
combination outright, independent of what either value is.

`supports_temperature` alone cannot express this: it is all-or-nothing, and dropping
*both* settings to fix a conflict over one of them would be wrong. **Decision:**
`ModelSpec` gains a second, independent flag, `supports_top_p`. `haiku45` sets it
`False`; `temperature=0.0` is kept, `top_p` is omitted entirely from its request body.

**This is not a new judgment call about which value to sacrifice** — B3-D3's own row
above already documents `top_p=1.0` as *"Neutral — with temperature at 0 it does
nothing"*. Omitting a setting already on record as a no-op costs nothing; the
load-bearing setting (`temperature=0.0`, kept for reproducibility) is untouched.

Enforced on the request body, not just the flag:
`test_haiku_omits_top_p_but_keeps_temperature` asserts `top_p` is absent from the
actual body sent, and `test_supports_temperature_and_supports_top_p_are_independent_
flags` pins that the two settings can be dropped independently rather than coupled.

**Watch for this on Fable 5 too when its blocker (Finding 4, sampling constraints
incompatible with `TEMPERATURE=0.0`/`TOP_P=1.0` entirely) gets resolved** — it is a
different, harder problem than Haiku's (Fable 5 rejects the frozen *values*, not just
the *combination*), but the same `supports_temperature`/`supports_top_p` machinery is
where that decision will need to attach.

---

### Fable 5's blocker resolved (2026-08-15): `supports_temperature`/`supports_top_p` set `False`, a real reproducibility loss recorded rather than absorbed

Fable 5's model card: *"temperature must be 1.0 or unset; top_p must be ≥ 0.99 and
< 1.0, or unset."* This is not Haiku's problem (reject the *combination*) — it is
each frozen value individually out of range: `TEMPERATURE=0.0` fails "must be 1.0 or
unset" outright, and `TOP_P=1.0` fails "< 1.0" (the range is exclusive at 1.0, so
1.0 itself is excluded). There is no in-range substitute this record can pick without
inventing a new hyperparameter.

**Decision:** both flags set `False`. Both settings are omitted entirely — the one
option the model card actually permits ("or unset") — rather than any body carrying
an out-of-range value.

**This must be stated as a real loss, not a free substitution.** B3-D3 freezes
`temperature` near zero specifically so a re-run is reproducible enough to debug.
That guarantee **does not hold for Fable 5**: omitted, the model uses its own
(unknown, undocumented) default temperature, and Fable 5's adaptive thinking being
always-on (cannot be disabled, per its model card) compounds this — reasoning
traces on extended-thinking models are known to vary run-to-run even when the
sampling temperature is pinned elsewhere. **Fable 5 is the one B3 condition where a
repeated run is not expected to reproduce the same schema**, and that has to be
stated plainly wherever B3-D3's reproducibility claim is cited in the paper, not
left to be discovered by a reviewer re-running the experiment.

**A second, unrelated finding folded in here because it surfaced from the same
model card while resolving this:** Fable 5's card documents `stop_reason:
"refusal"` for content-policy blocks, with refusal rates "materially higher than
on previous Claude models," and instructs callers to treat it as "a primary
response path." The existing truncation guard could not have told a refusal apart
from an ordinary empty response — it would have surfaced as a bare `ValueError`
indistinguishable from a model that legitimately found nothing. **New:**
`RefusalError`, structurally parallel to `TruncatedResponseError` (both now share a
`ModelResponseError` base) — fatal, not retried (a content-policy block is
deterministic at temperature 0, same as a cap cutoff), carries whatever partial
text came back, and is logged in the run's raw JSONL as `stop_reason: "refused"`,
distinct from `"truncated"`, so the two failure modes are never conflated by
whoever reads the log. Scoped to the anthropic family only — the constraint is
model-documented, not assumed to generalize to Meta or OpenAI-family stop reasons.

**Still open, not resolved here — verified but not run:** the model card's separate
data-retention requirement — *"To use this model, you must opt in to provider data
sharing by setting your data retention mode to `provider_data_share` via the Data
Retention API."* Confirmed against AWS's own abuse-detection page and the API
reference (not guessed): Anthropic requires this specifically for Fable 5 —
*"inputs and outputs will be retained for up to 30 days... you must opt in to
sharing retained traffic with Anthropic for abuse detection and potential human
review."* The real operation is **`PutAccountDataRetention`** — a **`bedrock`**
control-plane call (not `bedrock-runtime`, and not something `model_clients.py`'s
request body can express), one-time and account-level:

```python
import boto3
boto3.client("bedrock", region_name="us-east-1").put_account_data_retention(
    mode="provider_data_share"
)
```

Requires IAM permission `bedrock:PutAccountDataRetention`. **This account's IAM
user could not even list its own attached policies or call
`bedrock:ListFoundationModels`** (confirmed while debugging the earlier Haiku
Marketplace-subscription block) — it is narrowly scoped to `bedrock:InvokeModel`
and almost certainly lacks this permission too. Expect either an `AccessDenied`
here or, if skipped, an access/validation error on the first Fable 5 `invoke_model`
call — a different failure from anything this revision fixes, and one this session
cannot resolve without broader IAM access than the current user has.

---

### Why the cap had to become per-model (revision of 2026-08-14)

`2048` was a Groq free-tier artifact and nothing else: that tier reserves the full
requested output against its per-minute cap regardless of what the model generates, so
a large cap starved the input. On Bedrock nothing reserves anything, and the number was
left describing a constraint no longer present.

The obvious fix — one larger global — does not survive contact with the grid, because
**the ceiling is not the same for every model.**

| Condition | `max_output_tokens` | Basis |
|---|---|---|
| `fable5`, `haiku45`, `sol`, `luna` | `16000` | Headroom over the largest plausible whole-corpus schema. Haiku 4.5's real Bedrock ceiling is 64,000; 16,000 is chosen for headroom, not because more is unavailable. |
| `llama318b_bedrock` | **`4096`** | The Llama 3.1 8B Instruct model card states **"Max output tokens: 4K"** flat, against its 128K context window. Verified this is a property of the model as served and not of the API surface: Converse is supported and maps `maxTokens` onto the same limit, so there is no way around it. |
| `llama318b_groq` | `2048` | Held at the historical value on purpose (B3-D6). Raising it would make a re-run a different experiment from the one already reported. |

**Decision:** `max_output_tokens` moves onto `ModelSpec`. It is a property of the model
as served — **never** of the call shape. Whole-corpus and batched runs of the same model
send a byte-identical request body; enforced structurally, since `build_request_body()`
takes no batching argument and reads the cap only from the spec
(`test_the_output_cap_comes_from_the_spec_and_differs_per_model`,
`test_the_body_is_identical_whatever_the_call_shape`).

**A truncation guard, failing loud, with no retry.** `TruncatedResponseError` is raised
whenever the provider reports stopping at the cap — meta `stop_reason == "length"`,
anthropic `"max_tokens"`, openai `finish_reason == "length"`. A truncated schema is
never merged and never written to a result file. It is deliberately *not* retried: at
`TEMPERATURE = 0.0` an identical prompt returns an identical cut-off response, so
retrying spends money to fail the same way, and the only fixes would be mutating a
frozen value mid-run. The error's wording is itself load-bearing — `is_transient()`
matches on message text, so a stray "rate limit" or "timeout" in the phrasing would burn
five retries on a perfectly deterministic failure
(`test_a_truncated_response_is_never_treated_as_transient`). The cut-off text *is* kept
and written to the run log: fatal must not mean evidence-destroying.

Every call records its stop reason and completion-token count, into the per-call JSONL
and into `metadata.usage`. A run that finished well under its cap is the evidence that
the differing caps had no effect, and that claim cannot be made if nothing recorded it.

**A prediction, recorded before the run** (measured, not guessed): the merged schema
from the 28-batch Groq run is **25,094 chars ≈ 7,170 tokens** — and that is the
*deduplicated union*, the smallest honest estimate of what one whole-corpus answer must
carry. That is **~1.75× the 4,096 ceiling**. Whole-corpus should over-generate less than
28 fragmented calls did, so 7,170 is an upper bound rather than a forecast, but the
margin is not comfortable and the open-weight whole-corpus run may well truncate.

**Decision rule, fixed now so the outcome cannot be rationalized afterward:**

> If no run in the matrix reaches its output cap, the differing caps had no effect and
> are noted in Limitations only. If the open-weight whole-corpus run truncates at 4096,
> that is reported as a finding about the **Bedrock Meta serving limit** — not a
> limitation of Llama 3.1 8B itself, which has no such ceiling when served elsewhere —
> and the other four models are **NOT** lowered to 4096 in response.

Lowering the others would be the tempting move, by analogy with B3-D2's original
uniformity argument. It is wrong here for the same reason B3-D2 was: it would propagate
one host's serving limit to models that do not have it, and make every condition worse
in order to make one number match.

**Outcome, applied 2026-08-14.** The open-weight whole-corpus run did truncate
(`2026-08-14T20:29:28Z-f800`, `stop_reason: "length"` at exactly 4,096 tokens). The
decision rule above therefore applies as written: this is reported as a finding about
the Bedrock Meta serving limit, and the other four models' caps are unchanged.

One thing the rule didn't anticipate, so it is recorded rather than folded silently
into "it truncated": the response was not 4,096 tokens of genuine content cut off
mid-thought. It is 7 real classes followed by a mechanical repetition-degeneration
loop — the model cloning earlier classes' attribute lists under invented combinatorial
names (`Lease Renewal`, `Lease Termination Notice`, `Lease Cancellation Notice
Agreement`, …) rather than extracting anything new, and the cap caught it mid-loop.
Full evidence and analysis in `results/findings/B3-FINDINGS.md`, "Whole-corpus
condition (Bedrock)". This does not change the decision rule's application — the cap
is still the reason the run has no usable output — but it changes what should be said
about *why* an 8B model in particular hits it, and that distinction belongs in
Limitations rather than being silently absorbed into "the model had too much to say."

---

### Revised 2026-08-19: direct-API sampling, thinking, effort, and per-model caps for the three surviving conditions. The Bedrock-era caps table, the truncation-guard/`is_transient` retry apparatus, and the Fable-5/Llama-specific material above are superseded, not deleted.

With Bedrock and Groq gone (B3-D1, revised) the whole shape of this decision
changes: there is no per-provider request-body branching left (Meta, OpenAI, and
Anthropic-on-Bedrock shapes all disappear along with the providers that needed
them), and the sampling story is now about what the **direct Anthropic Messages
API** accepts per model, not about what one specific hosting surface enforced.

**Sampling: `haiku45` keeps the frozen values; both Opus conditions reject them
outright, confirmed against Anthropic's own current API documentation.**
`temperature`, `top_p`, and `top_k` are removed entirely on Opus 4.7 and later —
sending any of them is a 400, independent of value. This is not the Haiku-specific
"can't combine both" problem found on Bedrock (the entry above) — it is a harder,
flat rejection of the parameters existing at all. `opus48` and `opus5` therefore
set both `supports_temperature=False` and `supports_top_p=False`; `TEMPERATURE`
and `TOP_P` are never sent to either, and there is no fallback value to pick,
because none exists — the API does not take one.

**What replaces sampling control on the two Opus conditions: adaptive thinking at
a fixed effort level, not a substitute for reproducibility so much as an
acknowledgment that reproducibility in the old sense does not survive the move.**
`thinking={"type": "adaptive"}` plus `output_config={"effort": "high"}`, applied
identically to `opus48` and `opus5`. `effort` is the closest analog Opus offers to
a frozen dial — fixed, documented, and applied uniformly — but it does not pin
sampling the way `TEMPERATURE = 0.0` did; adaptive thinking is explicitly
non-deterministic in depth and content run to run. **State this plainly wherever
B3-D3's original reproducibility claim is cited**: it holds for `haiku45` exactly
as before, and does not hold for `opus48`/`opus5` in the way it used to hold for
every condition when everything ran through Bedrock.

**A flag worth carrying into the paper's methods section, not resolved here:**
Anthropic's own documentation groups Haiku 4.5 with "older models" that still
accept `temperature`/`top_p`/`top_k` — which is what `haiku45`'s spec assumes. The
Bedrock-era entry above found the *opposite* for Haiku 4.5 specifically (`top_p`
had to be dropped to avoid a 400) empirically, on that transport. Whether that
was a Bedrock-specific enforcement quirk that does not carry over to the direct
API, or documentation that is itself stale, is not yet confirmed against a real
call — the first real Haiku run under this rework settles it, the same discipline
applied throughout this project: verify against a real response before trusting
either the docs or the prior transport's behavior.

**Per-model output caps, chosen with thinking's token-sharing in mind:**

| Condition | `max_output_tokens` | Basis |
|---|---|---|
| `haiku45` | `16000` | No thinking, no effort — this cap bounds the visible response alone. Generous headroom over the largest plausible whole-corpus schema; Haiku 4.5's real ceiling is 64,000. |
| `opus48`, `opus5` | `32000` | **Not directly comparable to Haiku's 16,000** — on a thinking-enabled call this cap is shared between the model's thinking and its visible response combined, not the response alone. A non-thinking-equivalent cap would starve reasoning before any schema text is written. 32,000 is generous headroom under both models' real 128,000-token ceiling, chosen without a completed run to calibrate against — flagged as a prediction, the same way the Llama 4,096-token cap was predicted-then-checked in the entry above, and to be revisited once a real Opus run shows how much of the cap thinking alone consumes. |

**The custom retry apparatus (`is_transient()`, `_TRANSIENT_MARKERS`,
`_with_retries()`) is removed, not ported.** It existed to hand-roll exactly what
the official `anthropic` SDK already does natively — retry 429/5xx/connection
errors with backoff — via message-substring matching, which is exactly the kind
of thing a stray word in an error message could quietly break (as the truncation
guard's own docstring above warned about its own wording). `baselines/shared/
model_clients.py` now configures the SDK client with `max_retries=5` (the same
frozen value) and lets the SDK's own typed exceptions do this correctly. This is
a genuine simplification the move enables, not a loss: `TruncatedResponseError`
and `RefusalError` remain fatal-not-retried, but that is now enforced by their
type (never raised from a transport failure, so never routed into the SDK's own
retry path) rather than by keeping their wording clear of a hand-maintained
substring list.

**The truncation and refusal guards themselves carry over unchanged in spirit,
generalized to one body shape instead of three.** `TruncatedResponseError` on
`stop_reason == "max_tokens"`, `RefusalError` on `stop_reason == "refusal"` — both
now checked unconditionally (there is only the Anthropic family left, so the
"scoped to the anthropic family only" caveat two entries up no longer needs
scoping). Both still carry whatever partial text came back rather than
discarding it, and neither is ever merged or scored.

### Revised again, 2026-08-19: Haiku moves back to AWS Bedrock (B3-D1, revised again); `opus48` is dropped. This resolves the flag immediately above rather than leaving it open.

The flag two paragraphs up asked whether Haiku's Bedrock-only `top_p` rejection was
a transport quirk or stale documentation, "not yet confirmed against a real call."
That question is moot now, not answered: Haiku goes back to the exact transport the
constraint was originally measured on, per the user's correction recorded in B3-D1
(revised again, above) — *"Haiku from AWS Bedrock and Opus will be used from API."*
`haiku45.supports_top_p` reverts to `False`, `supports_temperature` stays `True`,
exactly as first measured, with no new claim about how Haiku behaves on the direct
API left dangling.

Two backends now coexist in `model_clients.py`, one per model, and the two paragraphs
above describing "the direct Anthropic Messages API" apply to `opus5` only:

| Condition | Transport | `max_output_tokens` | Sampling | Thinking/effort |
|---|---|---|---|---|
| `haiku45` | AWS Bedrock | `16000` | `temperature=0.0` only — Bedrock 400s if `top_p` is also present (empirically confirmed on this exact transport, this exact model) | none |
| `opus5` | Anthropic (direct API) | `32000` | none — rejected outright by the API on Opus 4.7+ | `thinking={"type": "adaptive"}`, `output_config={"effort": "high"}` |

`opus48`'s row is removed along with the condition itself (B3-D1, revised again) —
its cap and sampling rule were identical to `opus5`'s and are not lost information,
just no longer a live row.

**Request/response handling is now genuinely two functions per direction, not one
generalized shape.** `build_request` dispatches to `_build_bedrock_body` (Bedrock's
`anthropic_version`/block-structured `content`, model ID passed to `invoke_model`
separately, not in the body) or `_build_anthropic_api_kwargs` (plain-string
`content`, `model` in the kwargs) by `spec.backend`; response reading dispatches
similarly to `extract_from_bedrock_payload` or `extract_from_anthropic_message`.
The truncation/refusal checks (`_check_not_truncated`/`_check_not_refused`) stay
backend-agnostic, called from both extraction functions — the guards described
above did not need to change, only where they're called from.

**The SDK-native retry story splits the same way, rather than only applying to
Anthropic's SDK as the paragraph above assumed.** `invoke_anthropic_api` keeps
`anthropic.Anthropic(max_retries=5)` exactly as described above; `invoke_bedrock`
gets its own equivalent, `boto3.client("bedrock-runtime", config=Config(retries=
{"max_attempts": 5, "mode": "standard"}))` — boto3's built-in standard retry mode,
the same "let the SDK do it, don't hand-roll message-substring matching" reasoning
applied to the transport that needed reintroducing.

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

*Revised 2026-08-19: "P1's Stage 6" above refers to a staged design P1 never ended up
being — the actual P1 (`baselines/p1_pipeline/`, see P1-D1 below) is two stages, N
per-document extractions plus one consolidation call, and the consolidation call is
what does the cross-wording resolution this entry describes B3 as deliberately not
doing. The reasoning is unchanged; only the stage number was aspirational and is now
wrong. Also worth stating precisely rather than left implicit: with B3 now always a
single whole-corpus call (B3-D2, revised), this merge key structurally never sees more
than one source schema — B3-D4 governs `clean_schema()`'s within-response cleanup
(dropping malformed elements, collapsing literal duplicates the model repeated in its
own output), not a cross-call merge, because there is no second call left to merge
against. The rule is not weakened by this — it was never a live cross-call mechanism to
begin with once B3-D2 moved to one call — but it is worth being exact that "naive
exact-string consolidation" now describes a property of `clean_schema()`'s single-
source cleanup, not an active multi-source merge step.*

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

*Revised 2026-08-14: `metadata` gains `batching` (`"whole"` | `"batched"`),
`batch_size`, and `usage` (per-call stop reason and completion-token count). Additive
and contract-safe — `parse_induced_schema` ignores metadata entirely and
`load_induced_metadata` is a bare `data.get("metadata", {})`. Required by B3-D6: two
runs of one model in the two call shapes are otherwise indistinguishable from their
output alone, and comparing them is the entire point of that entry.*

*Revised again 2026-08-19: `batching` and `batch_size` are dropped — with only one
call shape left (B3-D2, revised), a key that would be constant across every future run
carries no information. `usage` collapses from a per-call list to two direct scalar
fields, `stop_reason` and `completion_tokens`, since there is only ever one call to
report on now. Still additive/contract-safe for the same reason as before. B3-D6's
comparison (which needed `batching` to tell its two arms apart) is itself retired
along with the Groq transport it compared against — see B3-D1 (revised) and B3-D6
below.*

One further rule is specific to B3: **no gold-schema vocabulary in the modules or in
the prompt** (Critical Rule 1). A gold term in the prompt would make all five models
oracles handed the answer key. The prompt is the likeliest place for this to slip in
by accident, so the leakage guard scans its full text, not just its code — and because
`eval.matching.normalize()` singularizes, ordinary prompt English collides with gold
relation labels: the prompt cannot say "list" (gold `lists`), "properties" (gold
`Property`), or "covers"/"reports"/"concerns". It says "array", "attributes",
"links" instead. Enforced by `test_no_domain_vocabulary_leakage`, which is itself
checked against a planted term so it cannot pass vacuously.

## B3-D6 — The Groq batched run is retained as its own condition, not superseded

*New 2026-08-14, alongside the B3-D2 and B3-D1c revisions above.*

Moving the open-weight model to Bedrock and to whole-corpus prompting would ordinarily
retire the 2026-08-12 Groq run as a superseded first attempt. **Keeping it instead turns
it into evidence.**

The registry therefore holds **six conditions, not five**: `llama318b_bedrock` and
`llama318b_groq` are the same open weights on two hosts, with distinct `model_id`s so
`metadata.model` and the output filenames tell them apart. `--batch-size 7` reproduces
the original run exactly — verified by dry run, per-batch char counts matching B3-D2's
table to the character (9,889 / 8,764 / 11,528 / …).

**Why this is worth the extra condition.** The difference between the two arms is a
direct measurement of **what cross-batch fragmentation costs when nothing intelligent
stitches the pieces back together** — same weights, same frozen prompt, same document
order, same naive consolidation, differing only in how many documents ride in a call.
That is exactly the gap P1's Stage 6 exists to close, so it is not a housekeeping detail
but a load-bearing number for the paper's central argument. Discarding the batched run
would have destroyed the only evidence for why the rework was needed at all.

**What it is not.** It is not a clean two-factor experiment: the two arms differ in host
(Groq vs. Bedrock) and output cap (2048 vs. 4096) as well as in call shape. Those are
confounds and must be named as such wherever the comparison is reported. The reason they
are not eliminated — by re-running the batched arm on Bedrock — is that doing so would
spend money to weaken the *other* thing the Groq run uniquely holds, which is the only
genuinely zero-cost cell in the grid (B3-D1c's reframe). `# TODO: decide before the
Results section whether a batched Bedrock run is worth buying to de-confound this.`

**What is held constant, structurally:** `batch_documents()` still takes no `ModelSpec`,
the call shape is decided in one place in `main()` and never reaches
`build_request_body()`, and the document order is byte-identical between the arms — so
the whole-corpus prompt is literally the batched sequence concatenated.

*Revised 2026-08-19: this comparison is retired going forward, not superseded in the
sense of being wrong — the open-weight/Llama condition itself is gone from the active
grid (B3-D1, revised), so there is no `llama318b_bedrock` to compare the retained
`llama318b_groq` run against anymore, and no future run will add a second data point to
either arm. The `# TODO` above about buying a de-confounding Bedrock batched run is
moot for the same reason. **The comparison's actual finding does not stop being true**:
fragmentation cost precision without breaking coherence, whole-corpus removed
fragmentation but broke an 8B model's coherence outright (see the repetition-
degeneration finding recorded earlier in B3-D3) — that remains real, reportable evidence
about scale-dependent single-shot behavior, just no longer an axis this project keeps
collecting data on. Both runs and this section stay in the record for that reason.*

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

---
---

# P1 Baseline — Design Decisions

Frozen decisions for **P1**, the staged-pipeline baseline
(`baselines/p1_pipeline/pipeline.py` + `baselines/p1_pipeline/prompts/
p1_consolidation_prompt.md` + the extraction prompt and calling code it shares with
B3). Built 2026-08-19, alongside B3's move to direct-API-only calling — first
decisions recorded here **before any P1 call of any kind**, the same discipline B1
and B3 were held to.

P1 is what B3 is measured against. Where B3 asks a model to read the whole corpus
in one breath and merges the answer by exact string match (B3-D4), P1 splits the
same corpus into one call per document and asks a **second** model call to
reconcile the N results into one schema — resolving cross-wording the way B3-D4
deliberately refuses to. That reconciliation is the entire thing this baseline
exists to measure the value of; everything else about P1 is held as close to B3 as
possible so that the comparison isolates it.

## P1-D1 — Two stages: N per-document extraction calls, then one consolidation call

**Decision:** for a corpus of N documents, make N calls — one per document, each
asking for that single document's partial schema — then one further call that is
handed all N partial schemas and asked to produce one final, reconciled schema.
Both calls go through the identical model for a given condition; there is no
cheaper model doing extraction and a stronger one doing consolidation.

**Per-document, not per-batch.** A batch-of-K design (K documents per extraction
call, like B3's retired legacy shape) would introduce a second free variable —
batch size — on top of the one this baseline is meant to isolate. Per-document
extraction removes that variable entirely: every extraction call sees exactly one
document's worth of context, uniformly, regardless of corpus size or document
density. The N/1 split is also the plainest possible operationalization of
"staged" — each stage does the smallest coherent unit of work the next stage can
build on.

**Why the extraction stage reuses B3's frozen prompt verbatim, rather than P1
getting its own.** `baselines/p1_pipeline/pipeline.py` imports
`load_prompt_template`, `render_prompt`, and `parse_response` straight from
`baselines.b3_single_shot.single_shot` and calls them unmodified, pointed at the
same `b3_extraction_prompt.md` file (`baselines/b3_single_shot/prompts/`, not
copied). This is deliberate, not laziness: if P1's extraction prompt differed from
B3's even slightly, a difference in the final result could be attributed to either
the prompting or the pipeline shape, and the comparison would stop meaning what the
paper claims it means. Holding the extraction prompt byte-identical makes the
consolidation call — the one genuinely new thing P1 does — the *only* variable
between the two baselines' schemas at the point where they diverge.

**Why the same document order.** `load_documents()` is imported from
`baselines.b3_single_shot.single_shot` too, unmodified — the corpus is walked in
the identical round-robin interleaving B3 uses. Document 1 of P1's extraction stage
sees the identical file B3's whole-corpus call sees first. Order cannot introduce a
difference either baseline's result could be attributed to.

## P1-D2 — Model calling is shared with B3, not duplicated

`baselines/shared/model_clients.py` — the registry, the request-building
functions, the response readers, the truncation/refusal guards — is the same
module B3 imports, with no P1-specific branch anywhere in it. The two frozen
conditions (`haiku45` on AWS Bedrock, `opus5` on the direct Anthropic API), their
output caps, their sampling/thinking/effort rules, are exactly B3-D3 (revised
2026-08-19, twice) unchanged. A P1 extraction call and a B3 whole-corpus call for
the same model go through the identical `invoke(spec, prompt)` entry point and the
identical per-backend request-building function, differing only in which prompt
and how many documents ride in it.

**Decision, stated explicitly so it isn't discovered by omission:** there is no
independent P1-D3-equivalent sampling decision to make. Sharing the calling module
is itself the decision — it is what keeps "how a call reaches the model" identical
across both baselines, so that whatever differs in the results is attributable to
the staging and the consolidation call, not to two different implementations of the
same transport plumbing quietly drifting apart.

## P1-D3 — The consolidation prompt: merge by wording, never by invention

**Decision:** a second, separately frozen prompt
(`baselines/p1_pipeline/prompts/p1_consolidation_prompt.md`, `{{SCHEMAS}}`
placeholder), given the full list of N partial schemas from Stage 1, serialized as
JSON and tagged with the source document each came from. It is instructed to:

- merge class (and relation) entries that plausibly name the same real-world kind
  across partial schemas — different casing, spacing, abbreviation, or genuine
  synonym — choosing the most frequently occurring wording as the merged name,
  the same "most frequent wording wins" rule B3's own extraction prompt already
  uses for a different purpose (picking a wording *within* one document rather than
  *across* many);
- keep genuinely different kinds separate even when their names or attribute
  arrays overlap, told explicitly that a missed merge costs less here than a wrong
  one — this is the one place P1's design deliberately biases against
  over-consolidation, since the alternative failure mode (silently collapsing two
  real gold classes into one) would be invisible in the emitted schema and only
  show up as a harder-to-diagnose scoring loss;
- work only from what the partial schemas already contain — explicitly forbidden
  from adding a class, attribute, or relation that is not present, in some wording,
  in at least one partial schema. This is a merge of existing extractions, not a
  second reading of the source documents; the consolidation call never sees the raw
  corpus text, only Stage 1's output.

**Same output contract as the extraction prompt** (`classes`/`relations`,
`name`/`parent`/`attributes`, `source`/`label`/`target`) — deliberately, so
`parse_response` and `clean_schema` (imported from B3, unmodified) apply
identically to a consolidation response as to an extraction one. Critical Rule 6
(`parent` stays `null` unless a model volunteers one) is unchanged and applies at
the consolidation stage exactly as it does everywhere else: nothing in
`clean_schema()` infers hierarchy; a non-null `parent` in the final output means
the consolidation call itself decided two merged classes stood in an is-a
relationship, which is a real finding about that call, not a rule violation.

**Critical Rule 1, extended to a second prompt.** Verified computationally, not
assumed, before this prompt was ever used: every identifier and literal it
introduces was checked against the real gold vocabulary loaded from
`gold_schema.ttl`, the same way every prior prompt and module in this project has
been. Three real collisions were caught and fixed this way, not hypothetically —
"list" (from "a list of partial schemas" and "attribute lists"), "resolves" (from
"resolves this"), and "own" (from "the partial schemas' own vocabulary") all
normalize to gold relation-label terms (`lists`, `resolves`, `owns`) and were
rewritten to "set", "reconciles", and "existing" respectively before the leakage
guard passed clean. Enforced ongoing by
`baselines/tests/test_p1_pipeline.py::test_no_domain_vocabulary_leakage`.

## P1-D4 — Fault tolerance is asymmetric between the two stages, and that asymmetry is deliberate

**Stage 1 (extraction) tolerates a single call's failure; Stage 2 (consolidation)
does not.** A truncated, refused, or unparseable extraction call is recorded in the
raw JSONL and skipped — the run continues with N-1 partial schemas rather than
aborting — because with N independent calls (192, at full corpus size), one bad
document does not invalidate the other 191 that already succeeded and were paid
for. This is the same reasoning the old batched B3 shape used when it tolerated one
bad batch out of 28 (see the retired B3-D2 material above); it applies here for an
identical reason.

**The consolidation call has no such tolerance — a truncation, refusal, or parse
failure there aborts the run.** There is exactly one consolidation call; unlike a
skipped extraction, there is nothing left to fall back on if it fails, and
skipping it would mean emitting whatever Stage 1 produced with no reconciliation
at all — silently changing what condition actually ran. This mirrors B3's own
single-whole-corpus-call rule (B3-D2, revised) precisely: when a call is the only
source, its failure is fatal by construction, not a judgment call made per run.

**What gets written to disk either way:** the raw JSONL log captures every
extraction call attempted (success or failure) plus the consolidation attempt,
appended even when the pipeline dies mid-run — a paid-for response is never lost
just because a later stage failed. No final schema JSON is written unless
consolidation actually succeeds; a partial run must not leave behind a file that
looks like a completed result.

## P1-D5 — Output contract

Same JSON contract as B1 and B3 (`eval/PLAN.md` §2,
`eval/schema_ir.py::parse_induced_schema`): `classes` (`name`, `parent`,
`attributes`), `relations` (`source`, `label`, `target`), with
`metadata.condition = "P1"` and `metadata.model` set to the exact frozen model ID
that produced the file — identical to what B3 records for the same model, so a
reader can compare `metadata.model` values across both baselines without needing
to know which pipeline produced which file.

`metadata` additionally carries `extraction_calls` (always the document count),
`extraction_skipped` (how many Stage 1 calls failed — P1-D4), and
`consolidation_stop_reason` / `consolidation_completion_tokens` (the one Stage 2
call's outcome). No per-extraction-call detail lives in the metadata itself — that
level of detail belongs in the raw JSONL log, matching the split B3 already uses
between its lean `metadata` and its verbose raw call record.

Both Critical Rules inherited from B1/B3 apply unchanged: **no pre-cleaning**
(Rule 5 — names are emitted exactly as returned; `clean_schema()`'s case/whitespace
collapsing happens only inside its dedup key, never on an emitted string) and
**`parent` stays `null` unless volunteered** (Rule 6).

---

## Expected result shape for P1 (a prediction, not a target)

Recorded before the first P1 call, the same discipline B1 and B3 were held to:

- **Higher recall than B3 at the same model**, specifically on the classes B3's
  naive merge fragments across wordings (B3-D4's known cost). The consolidation
  call exists to buy back exactly this; if it does not, the staged design is not
  earning its extra 192 calls' worth of cost over B3's 1.
- **A real risk of over-consolidation** — the consolidation prompt's own bias
  against merging genuinely different classes (P1-D3) is a mitigation, not a
  guarantee. Watch specifically for two gold-distinct classes (e.g. `Owner` and
  `Agent`, both `Party` subclasses with overlapping attribute vocabulary in the
  corpus) collapsing into one merged entry — this would be a **P1-specific**
  failure mode with no B3 analog, since B3 structurally cannot over-merge (B3-D4
  never merges distinct wordings at all).
- **Substantially higher cost than B3** — 193 calls per model versus B3's 1 — is
  the direct, known price of the staged design, not a defect to explain away. Any
  quality gain has to be weighed against this explicitly in the paper, not
  reported as a free improvement.
- **The extraction stage should look like a degenerate, single-document version of
  B3's own class-explosion problem** — with only one document's worth of evidence
  per call, "several distinct instances beyond a bare label" (the extraction
  prompt's own class criterion) is nearly impossible for the model to satisfy
  honestly within one document, so Stage 1 output is expected to be sparse and
  fragmented by design. This is not a bug in Stage 1; the whole point of Stage 2
  is to do the job Stage 1 structurally cannot.

If P1's class F1 does not exceed B3's at the same model and the same matching
level, that is real, reportable evidence that staged consolidation added by an LLM
call does not clearly outperform B3-D4's naive one for this corpus — not a result
to quietly re-run until it looks better.
