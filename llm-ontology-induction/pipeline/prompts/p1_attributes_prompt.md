# Attribute Consolidation

You are given a set of classes. Each class's `attributes` array is a raw
concatenation of every attribute name recorded for it across many separate
partial extractions, before any cleanup. Because of that, the same field may
appear several times under different wordings.

Your only job here is reconciling attribute wording, one class at a time. Do
not rename a class, change its `parent`, add a class, drop a class, or move
an attribute from one class to another -- that is settled already and is not
this pass's job.

Produce the same classes with each one's `attributes` array reconciled, and
return it as a single JSON object.

## What to do

**Merge attribute entries that name the same underlying field**, even when
worded differently -- different casing, spacing, punctuation, an
abbreviation, or a genuine synonym (`monthly payment` / `Monthly_Base_Rate` /
`payment amount` are one field, not three). When you merge two or more
entries into one:

- Choose the wording that appears most often within that same class's
  attribute array. Do not coin a cleaner-sounding replacement that never
  appeared.
- If every wording for a merged entry appears exactly once, choose the
  shortest one -- shortest is the least likely to be a one-off, oddly
  specific restatement of a field that is named more plainly elsewhere in
  the same array.

**Do not merge attribute entries that name genuinely different fields**, even
if their wording overlaps. When in doubt, keep them separate -- a missed
merge costs less here than a wrong one.

## Rules

1. Work only from the attribute wordings already present in that class's
   attribute array. Do not add an attribute that is not present, in some
   wording, in that same array.
2. Use the array's existing vocabulary for the chosen wording. Do not tidy,
   standardize, expand, or translate a term into a cleaner equivalent beyond
   the wording-selection rule above.
3. Preserve every class's `name` and `parent` exactly as given. Every field
   shown in the output shape below is required on every element; use `null`
   for `parent` when it was `null` going in, never omit the key.
4. Return the JSON object and nothing else -- no preamble, no explanation
   afterwards, no markdown fences.

## Output shape

```json
{
  "classes": [
    { "name": "...", "parent": null, "attributes": ["...", "..."] }
  ]
}
```

## Classes

{{CLASSES}}
