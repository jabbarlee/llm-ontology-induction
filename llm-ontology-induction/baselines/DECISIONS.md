# B1 Baseline — Design Decisions

Frozen decisions for the B1 statistical baseline, recorded before B1 is ever run against the harness.

## D1 — One combined run over the whole corpus, not per-document

Schema induction needs cross-document evidence — no single messy document contains the full picture.

**Decision:** B1 reads all 192 documents as one pooled corpus and produces exactly one output schema. This also gives the statistical methods their best possible shot (more text = more reliable frequency signal), which matters for B1 to be a fair, non-strawman baseline.

## D2 — Term extraction: C-value, not raw frequency

Raw frequency favors short, generic words ("the," "property," "date"). The roadmap specifically names C-value (Frantzi et al.) — it's built for exactly this problem: it rewards multi-word terms and subtracts out the frequency they only have because they're nested inside longer, more specific terms.

```
C-value(term) = log2(|term|) × ( freq(term) − mean freq of longer terms containing it )
```

So if "maintenance request" appears 40 times and "maintenance request status" appears 12 times as a sub-occurrence, the shorter term's score gets discounted by that overlap — it stops looking artificially important just for being a substring of something else.

**Decision:** implement C-value directly (it's a well-defined formula, not a heavy dependency) rather than pull in a third-party terminology-extraction library — one less unmaintained dependency, and full control over the candidate-term patterns feeding into it.

## D3 — Candidate term patterns, POS-based, not hardcoded vocabulary

Candidates come from noun-phrase patterns over POS tags — `(ADJ|NOUN)* NOUN`, 1 to 4 tokens — extracted via spaCy, not a hand-typed list of expected CRE terms.

Writing `["owner", "tenant", "lease", ...]` into the extraction code would make B1 an oracle, not a baseline — it would already "know" the answer key. The whole point is that B1 discovers candidates from the documents alone and only afterward gets compared against gold.

## D4 — Class vs. attribute split: a frequency-and-position heuristic

Statistical methods can't reliably tell "Owner" (a class) from "tax id" (an attribute) the way a schema-aware method could.

**Decision**, stated as a heuristic, not a guarantee:

- Take the top-N C-value terms (N configurable, default 20) as class candidates.
- For each class candidate, attribute candidates are shorter terms that co-occur with it inside a small token window (default: same sentence) and follow a possessive/descriptive pattern (`X's Y`, `Y of X`, `X: Y` in the CSVs). These become that class's attribute list.

Be upfront in the paper about this limitation rather than engineering around it — a statistical method genuinely struggling to separate entities from properties is itself a finding, not a flaw to hide. It's exactly the kind of result that motivates why the multi-stage pipeline (P1) has separate Stages 3 and 4 for type induction and attribute induction rather than doing both at once.

## D5 — Relations: dependency-parse SVO triples, frequency-thresholded

For each sentence (prose lease texts, notes, messages) and each CSV row (treated as a flattened pseudo-sentence: `"{col}: {val}, {col}: {val}..."`), run spaCy's dependency parser and extract `(subject, verb_lemma, object)` triples where the subject and object are both class-candidate terms (from D4) and the verb isn't a copula (is/are/was).

Keep triples occurring above a minimum frequency (default: 3+ times across the corpus) as relation candidates. No hardcoded verb list (owns, leases, etc.) — same oracle concern as D3.

## D6 — No taxonomy detection, and that's the honest result

B1 has no mechanism to infer `Owner isSubclassOf Party` from term statistics alone.

**Decision:** every class gets `parent: null`. Expect and report a taxonomy-layer F1 of exactly 0.0 for B1 in every table — this is not a bug to patch, it's the actual, correct, informative result of "what does hierarchy induction cost you, if you skip it."

## D7 — Hyperparameters frozen before ever looking at a score

Top-N class count, minimum relation frequency, attribute co-occurrence window — pick these from term-extraction literature conventions (C-value's original paper, standard noun-chunk windows) or a quick sanity pass on the shape of the output (does 20 classes look like a plausible-sized list, not zero, not 400), never by tuning until B1's score against gold looks good. That would silently turn B1 into a second copy of the harness's threshold, defeating the purpose of having an independent baseline.

Frozen in this file (`baselines/DECISIONS.md`) alongside D1–D6, before running B1 against the real harness even once.
