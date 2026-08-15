# Schema Induction from Unstructured Documents

You are given a batch of documents drawn from one business domain. They are messy and
inconsistent: the same idea is worded differently from document to document, terms are
abbreviated, and information is often left implicit.

Infer the data schema that lies underneath these documents, and return it as a single
JSON object.

## What to extract

**classes** — the recurring kinds of entity the documents talk about. A kind belongs
here when the documents describe several distinct instances of it and it carries
information beyond a bare label.

For each class, give:

- `name` — the term the documents themselves use for that kind.
- `parent` — a broader class, also present in this batch, that this one is a
  specialization of; otherwise `null`. Use a non-null `parent` only where the
  documents genuinely support an is-a generalization. When in doubt, use `null`.
- `attributes` — an array of the fields, values, and descriptors the documents attach
  to that kind.

**relations** — directed links between two classes, either stated outright or clearly
implied. Each has a `source` class, a `label` (the verb or phrase the documents use
for the link), and a `target` class. Both endpoints must be classes you emit above.

## Rules

1. Extract only what these documents support. Do not add kinds, fields, or links from
   general knowledge of the domain.
2. Use the documents' vocabulary verbatim. Do not tidy, standardize, expand, or
   translate a term into a cleaner equivalent — emit it with the spelling, casing, and
   spacing it carries in the text.
3. Where several wordings appear for what seems to be one idea, choose the wording
   that appears most often. Do not coin a new umbrella term that no document uses.
4. Every field shown in the output shape below is required on every element. Use
   `null` for `parent` when there is no broader class; never omit the key.
5. Return the JSON object and nothing else — no preamble, no explanation afterwards,
   no markdown fences.

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

## Documents

{{DOCUMENTS}}
