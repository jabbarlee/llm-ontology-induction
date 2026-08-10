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
