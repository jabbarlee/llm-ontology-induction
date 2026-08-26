# Relation Reconciliation

You are given a set of relations. Each one's `source` and `target` have
already been rewritten onto a final, merged set of classes -- that part is done
and is not your job. Because the relations were originally extracted from
many separate documents, the same real-world link between two classes may
still appear more than once here, under different wordings, now that its
endpoints match.

Produce one reconciled set of relations, and return it as a single JSON object.

## What to do

**Merge relation entries that describe the same link** -- same `source` and
`target`, and a `label` that is a close wording of the same relationship (a
synonym, or a casing/spacing/punctuation difference, not a different
relationship). When you merge two or more entries into one:

- Among the recorded wordings for that link, prefer whichever one reads as
  an action or a relationship between the two classes, if one of them does.
  Do not invent a wording that is not already recorded for that link.
- If none of the recorded wordings reads as an action, or more than one
  does, choose whichever wording appears most often.

**Do not merge relation entries that describe genuinely different links**,
even if they share the same source and target. Two classes can be connected
more than one real way. When in doubt, keep them separate -- a missed merge
costs less here than a wrong one.

## Rules

1. Work only from the relations already given below. Do not add a relation,
   and do not change a `source` or `target` -- endpoints are already final.
2. Use the given relations' existing vocabulary for the chosen `label`. Do
   not tidy, standardize, expand, or translate a term into a cleaner
   equivalent beyond the wording-selection rule above.
3. Every field shown in the output shape below is required on every element;
   never omit a key.
4. Return the JSON object and nothing else -- no preamble, no explanation
   afterwards, no markdown fences.

## Output shape

```json
{
  "relations": [
    { "source": "...", "label": "...", "target": "..." }
  ]
}
```

## Relations

{{RELATIONS}}
