# Taxonomy Induction

You are given a final, consolidated set of classes, each with its name,
current `parent` (which may already be set, or `null`), and its attributes.

Your job is different from every earlier pass: instead of reconciling
wordings that already exist, look at the set as a whole and ask whether
several of these classes are, in substance, specific kinds of one broader
category -- one that none of them may have been individually labeled as, but
that their names and attributes themselves support.

## When a broader category is supported

Propose one only when **at least two** of the given classes would belong
under it, and the case for it rests on one or both of:

- **A naming pattern.** Several class names read as specific varieties of
  one broader kind (e.g. -- purely as an illustration, not a hint about this
  particular set -- `Sedan`, `Pickup Truck`, and `Motorcycle` all read as
  kinds of a broader `Vehicle`, because the broader term is what their names
  are specific *versions of*).
- **Substantial shared attributes.** Several classes carry attributes that
  plainly belong to one broader concept rather than to each class
  individually (continuing the same illustration: `passenger capacity` and
  `fuel type` are attributes that belong to a `Vehicle` in general, not attributes specific
  to being a `Sedan` versus a `Pickup Truck`).

Name the broader category using a term the evidence itself supports -- a
word implied by the shared naming pattern or the shared attributes, not a
term borrowed from outside knowledge about the domain in general.

**If the broader category is not already one of the given classes, add it as
a new entry in the output**, with that name, `parent: null` (it is
unusual for a newly proposed category to itself need a still-broader parent;
only set one if the same evidence standard supports that too), and an empty
`attributes` array unless one or more of the shared attributes you identified
plainly belongs to the broader category itself rather than to any one of its
specific kinds. Writing it as a separate entry is required, not optional --
every class named as a `parent` anywhere in your output must also appear as
an entry in `classes`.

## When it is not supported -- the default

If you cannot point to a naming pattern or a real attribute overlap across at
least two classes, output `null` for that relationship. **A missed
generalization costs less here than an invented one that is not actually
supported by these classes' names and attributes themselves.** Do not propose a
broader category merely because it seems plausible in general, and do not
draw on anything you know about this kind of domain beyond what is written
in the class names and attributes given to you below.

A class that already has a non-null `parent` keeps it unless you have
concrete evidence, by the same standard above, that it belongs under a
*different* broader category instead. When in doubt, leave an existing
`parent` exactly as given.

## Rules

1. Every class from the input must appear in the output, with its `name` and
   `attributes` unchanged. Only `parent` may change. A newly proposed
   category is the one exception -- it is new, per the rule above.
2. Do not merge, rename, split, or drop any *given* class -- that was already
   settled in earlier passes and is not this one's job.
3. Every field shown in the output shape below is required on every element;
   use `null` for `parent` when there is no supported broader category,
   never omit the key.
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
