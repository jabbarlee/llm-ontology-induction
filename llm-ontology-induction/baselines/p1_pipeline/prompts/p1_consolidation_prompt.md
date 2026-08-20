# Schema Consolidation from Independent Extractions

You are given a set of partial schemas. Each one was extracted independently
from a single source document, by a separate call that could not see any other
document in the set. Because of that, the same real-world kind of thing may
appear more than once under different wordings, and the same real-world link
between two kinds may appear more than once under different labels.

Produce one consolidated schema that reconciles this, and return it as a single
JSON object.

## What to do

**Merge classes that refer to the same real-world kind of thing**, even when the
partial schemas name them differently -- different casing, spacing,
abbreviation, singular/plural, or a genuine synonym. When you merge two or more
class entries into one:

- Choose the name that appears most often across the partial schemas as the
  merged class's `name`. Do not coin a new umbrella term that none of the
  partial schemas used.
- Combine their `attributes` into one array, keeping each attribute's original
  wording and dropping only literal repeats (same wording, ignoring case and
  surrounding whitespace).
- Keep a non-null `parent` if any of the merged entries had one; if more than
  one disagrees, keep the one that appears most often.

**Do not merge classes that are genuinely different kinds of thing**, even if
their names are similar or their attribute arrays overlap. When in doubt, keep
them separate -- a missed merge costs less here than a wrong one.

**Merge relations the same way**: two relation entries refer to the same link
if their endpoints map to the same merged classes and their `label`s are
close wordings of the same relationship (a synonym or a casing/spacing
difference, not a different relationship). Keep the most frequent wording of
the label.

## Rules

1. Work only from what the partial schemas already contain. Do not add a class,
   attribute, or relation that is not present, in some wording, in at least one
   of them -- this is a merge of existing extractions, not a fresh reading of
   the source documents.
2. Use the partial schemas' existing vocabulary. Do not tidy, standardize, expand,
   or translate a term into a cleaner equivalent beyond the wording-selection
   rule above.
3. Every field shown in the output shape below is required on every element.
   Use `null` for `parent` when there is no broader class; never omit the key.
4. Return the JSON object and nothing else -- no preamble, no explanation
   afterwards, no markdown fences.

## Output shape

```json
{
  "classes": [
    { "name": "...", "parent": null, "attributes": ["...", "..."] }
  ],
  "relations": [
    { "source": "...", "label": "...", "target": "..." }
  ]
}
```

## Partial schemas

{{SCHEMAS}}
