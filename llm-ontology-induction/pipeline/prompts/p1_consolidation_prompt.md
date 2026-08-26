# Type Consolidation from Independent Extractions

You are given a set of classes gathered from many independent extractions, each
done separately, with no extraction able to see any other. Because of that,
the same real-world kind of thing may appear more than once under different
wordings.

Classes that were already spelled and cased identically across extractions
have already been combined for you, one entry per distinct spelling, before
you ever see them -- that part required no judgment. Each entry's
`occurrences` count says how many original extractions used exactly that
spelling; it is a frequency signal for the choice below, not something you
merge, change, or repeat back.

Your only job here is deciding which of the *remaining, differently-spelled*
entries are the same kind of thing. Attribute wording and relation wording are
handled by a later pass -- do not try to clean those up now.

Produce one consolidated set of classes, and return it as a single JSON
object.

## What to do

**Merge entries that refer to the same real-world kind of thing**, even when
they are named differently -- different casing, spacing, abbreviation,
singular/plural, or a genuine synonym. Each entry's `attributes` are shown to
you as evidence for that judgment: two entries describing the same kind of
thing usually record overlapping attributes. Read them, weigh them, and then
leave them alone -- see the note below.

When you merge two or more entries into one:

- Choose the name with the highest total `occurrences` among the entries you
  are merging as the merged class's `name`. Do not coin a new umbrella term
  that none of the entries used.
- Keep a non-null `parent` if any of the merged entries had one; if more than
  one disagrees, keep the one from the entry with the highest `occurrences`.
- Record every original name that fed into this merged entry in
  `merged_from`, spelled exactly as it appears in the entries below. If
  nothing was merged (the entry stood alone), `merged_from` still records
  that one name.

**Do not merge entries that are genuinely different kinds of thing**, even if
their names are similar or their attribute arrays overlap. When in doubt,
keep them separate -- a missed merge costs less here than a wrong one.

## Do not repeat the attributes back

Your answer must not contain any `attributes` field. The attributes of a
merged class are recovered afterwards from the `merged_from` names you give,
by taking the attributes of exactly those entries -- so nothing is lost by
leaving them out, and copying them back would only risk changing wording this
pass is not allowed to change.

## Rules

1. Work only from the names already present in the entries below. Do not add
   a class that is not present, in some wording, in at least one of them --
   this is a merge of existing extractions, not a fresh reading of the source
   documents.
2. Every entry below must appear in your answer exactly once: either by
   itself, or inside the `merged_from` of the entry it was merged into. Do
   not leave an entry out, and do not place one under two different merged
   classes.
3. Use the existing vocabulary for the merged `name`. Do not tidy,
   standardize, expand, or translate a term into a cleaner equivalent beyond
   the wording-selection rule above.
4. Every field shown in the output shape below is required on every element.
   Use `null` for `parent` when there is no broader class; never omit the
   key. `merged_from` is never empty. Do not include `occurrences` or
   `attributes` in your output -- both are input-only context.
5. Return the JSON object and nothing else -- no preamble, no explanation
   afterwards, no markdown fences.

## Output shape

```json
{
  "classes": [
    {
      "name": "...",
      "parent": null,
      "merged_from": ["...", "..."]
    }
  ]
}
```

## Classes

{{CLASSES}}
