"""
B1 baseline — term extraction primitives (baselines/DECISIONS.md D2, D3, D5).

Pure functions over already-loaded text. No file I/O, no JSON assembly, no
corpus paths — everything here is unit-testable in isolation against a
hand-written list of sentences.

HARD RULE (DECISIONS.md D3): this module must contain zero domain vocabulary.
No gold-schema class, attribute, or relation name may appear as a literal
anywhere in it — not to seed extraction, not to filter it, not to validate it.
Candidates come from POS patterns and dependency structure alone. Enforced by
baselines/tests/test_statistical.py::test_no_domain_vocabulary_leakage.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc

# --- Frozen hyperparameters (DECISIONS.md D7) ----------------------------
MAX_TERM_TOKENS = 4
MIN_TERM_FREQ = 3

_SPACY_MODEL = "en_core_web_sm"

# POS tags admissible inside a candidate term. The pattern is (ADJ|NOUN)* NOUN:
# modifiers may lead, but a candidate must END on a noun (D3). PROPN is
# excluded -- proper nouns are instance data (people, companies, addresses),
# not schema-level terms. See D3b for why this exclusion is only safe in
# combination with case-folded tagging.
_MODIFIER_POS = frozenset({"ADJ", "NOUN"})
_HEAD_POS = frozenset({"NOUN"})

# Copular lemmas excluded from relation verbs (D5): a copula asserts type
# identity ("X is a Y"), which is a taxonomy claim, not a relation -- and B1
# makes no taxonomy claims (D6).
_COPULA_LEMMAS = frozenset({"be"})

_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass"})
_OBJECT_DEPS = frozenset({"dobj", "pobj", "attr", "obj"})


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Load the spaCy pipeline once per process.

    Cached because loading is ~1s and the corpus is parsed in several passes;
    lru_cache rather than a module-level global so importing this module stays
    side-effect free (and tests can run without paying for a load they don't
    need).
    """
    return spacy.load(_SPACY_MODEL)


# ---------------------------------------------------------------------------
# Parsing (D3b — case-folded tagging)
# ---------------------------------------------------------------------------
#
# D3b: the tagger is run over a LOWERCASED copy of each sentence, and surface
# forms are recovered from the original by character offset.
#
# Why: in this corpus capitalization is a formatting convention, not a
# linguistic fact. Contracts capitalize defined terms mid-sentence ("the
# Tenant", "the Lessee") and CSV headers are title-cased ("Lease Owner",
# "Commence Date"), so the tagger labels exactly the schema-level terms PROPN
# while leaving their lowercase synonyms ("landlord", "lease") as NOUN. With
# PROPN excluded, an arbitrary orthographic convention would decide which
# terms B1 can see at all.
#
# Lowercasing is generic text preprocessing -- no vocabulary, no domain
# knowledge, applied uniformly to every sentence. It is NOT output cleaning:
# emitted names are sliced from the ORIGINAL text and keep their real casing,
# spacing and pluralization, per Rule 2 / D3a.
#
# (spaCy's NER was evaluated as the alternative filter for instance data and
# rejected: on this corpus it labels "Tenant" PRODUCT and "Lessee" PERSON,
# i.e. it deletes the very terms B1 needs.)


def _analyze(sentences: list[str]) -> list[tuple[Doc, str]]:
    """Parse each sentence lowercased, paired with its original text."""
    lowered = [s.lower() for s in sentences]
    nlp = get_nlp()
    return list(zip(nlp.pipe(lowered, batch_size=64), sentences))


def _surface(doc: Doc, start: int, end: int, original: str) -> str:
    """The span's text as it appears in the ORIGINAL, un-lowercased sentence.

    str.lower() is length-preserving for this corpus, so token character
    offsets carry over; the length guard falls back to the lowercased span for
    the rare Unicode case where it is not (e.g. 'İ' -> 'i̇').
    """
    if len(original) != len(doc.text):
        return doc[start : end + 1].text
    first, last = doc[start], doc[end]
    return original[first.idx : last.idx + len(last.text)]


def _key(term: str) -> str:
    """Internal case/whitespace-folded aggregation key (D3a).

    Never used as an output name -- only to count "Tenant" and "tenant" as one
    term and to test membership of a span in the candidate set.
    """
    return " ".join(term.split()).casefold()


# ---------------------------------------------------------------------------
# Candidate terms (D3)
# ---------------------------------------------------------------------------

def _indexed_term_spans(doc: Doc) -> list[tuple[int, int]]:
    """Every contiguous (ADJ|NOUN)* NOUN span of 1..MAX_TERM_TOKENS tokens, as
    (start, end_inclusive) token indices.

    All lengths are emitted, not just the maximal span: C-value's nesting
    discount (D2) is only meaningful if the shorter nested candidates are in
    the candidate set to be discounted in the first place.
    """
    out: list[tuple[int, int]] = []
    n = len(doc)
    for start in range(n):
        if doc[start].pos_ not in _MODIFIER_POS:
            continue
        for end in range(start, min(start + MAX_TERM_TOKENS, n)):
            tok = doc[end]
            if tok.pos_ not in _MODIFIER_POS:
                break
            if tok.pos_ in _HEAD_POS:
                out.append((start, end))
    return out


def extract_candidate_terms(sentences: list[str]) -> dict[str, int]:
    """POS-based candidate terms -> raw corpus frequency.

    Keys are the case-folded aggregation key (D3a): "Tenant" at sentence start
    and "tenant" mid-sentence are one term, not two sub-threshold halves. The
    original surface forms are recovered separately via surface_forms() so the
    emitted schema can carry a real observed form.

    Terms below MIN_TERM_FREQ (D7) are dropped as near-hapax noise before
    C-value ever sees them.
    """
    counts: Counter[str] = Counter()
    for doc, _original in _analyze(sentences):
        for start, end in _indexed_term_spans(doc):
            counts[_key(doc[start : end + 1].text)] += 1
    return {term: freq for term, freq in counts.items() if freq >= MIN_TERM_FREQ}


def surface_forms(sentences: list[str]) -> dict[str, str]:
    """Aggregation key -> most frequent surface form observed for it (D3a).

    This is form *selection*, not normalization: the returned string is a
    verbatim slice of the original corpus text, with its casing, spacing and
    pluralization intact. Ties break on the lexicographically first form so a
    run is deterministic.
    """
    observed: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for doc, original in _analyze(sentences):
        for start, end in _indexed_term_spans(doc):
            span_text = doc[start : end + 1].text
            observed[_key(span_text)][_surface(doc, start, end, original)] += 1
    return {
        key: min(forms.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for key, forms in observed.items()
    }


# ---------------------------------------------------------------------------
# C-value (D2)
# ---------------------------------------------------------------------------

def build_nesting_map(terms: dict[str, int]) -> dict[str, list[str]]:
    """term -> the LONGER candidate terms that contain it as a token subsequence.

    Contiguous token-subsequence containment, not substring containment:
    substring matching would report "rate" as nested in "corporate", which is
    not a nesting relationship at all.
    """
    tokenized = {term: tuple(term.split()) for term in terms}
    by_length: defaultdict[int, list[str]] = defaultdict(list)
    for term, toks in tokenized.items():
        by_length[len(toks)].append(term)

    nesting: dict[str, list[str]] = {}
    for term, toks in tokenized.items():
        containers: list[str] = []
        for longer_len in range(len(toks) + 1, MAX_TERM_TOKENS + 1):
            for candidate in by_length.get(longer_len, ()):
                if _contains_subsequence(tokenized[candidate], toks):
                    containers.append(candidate)
        nesting[term] = sorted(containers)
    return nesting


def _contains_subsequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    span = len(needle)
    return any(
        haystack[i : i + span] == needle for i in range(len(haystack) - span + 1)
    )


def length_factor(n_tokens: int) -> float:
    """C-value's term-length weight, as log2(n + 1) rather than log2(n).

    DECISIONS.md D2a: the literature's log2(|a|) is exactly 0 for single-token
    terms, which zeroes out an entire length class by construction regardless
    of corpus evidence -- an artifact of the formula, not a finding. log2(n+1)
    keeps the monotonic length reward (1.000, 1.585, 2.000, 2.322 for lengths
    1-4) without annihilating length-1 candidates a priori.
    """
    return math.log2(n_tokens + 1)


def c_value(terms: dict[str, int], nesting: dict[str, list[str]]) -> dict[str, float]:
    """C-value score per term (D2).

        not nested:  length_factor(|a|) * freq(a)
        nested:      length_factor(|a|) * ( freq(a) - mean freq of its containers )

    The subtraction is what stops a short term from looking important merely
    for being nested inside something longer and more specific.

    Scores may be negative (a term occurring ONLY inside longer terms), which
    is meaningful and left unclamped -- such a term should rank below every
    independently-attested one.
    """
    scores: dict[str, float] = {}
    for term, freq in terms.items():
        containers = [c for c in nesting.get(term, ()) if c in terms]
        factor = length_factor(len(term.split()))
        if not containers:
            scores[term] = factor * freq
        else:
            mean_container_freq = sum(terms[c] for c in containers) / len(containers)
            scores[term] = factor * (freq - mean_container_freq)
    return scores


# ---------------------------------------------------------------------------
# Attributes (D4)
# ---------------------------------------------------------------------------

def extract_attribute_candidates(
    sentences: list[str],
    class_candidates: set[str],
    all_terms: set[str],
) -> dict[str, Counter[str]]:
    """class candidate -> Counter of attribute candidates co-occurring with it.

    Implements D4's four generic grammatical patterns, all applied within a
    single sentence (for CSV input, one flattened row is one pseudo-sentence):

        possessive    C 's T
        colon         C : T
        descriptive   noun chunk beginning with C and continuing into T
        prepositional T of C

    Every pattern is grammar, never vocabulary (D3). The descriptive-compound
    pattern is the one that fires on tabular headers, which is where attribute
    names actually live in the CSV half of this corpus.

    An attribute must be a known candidate term, must not itself be a class
    candidate, and must not contain the class as a token subsequence --
    otherwise the top-ranked terms would all appear as one another's
    attributes, and "tenant name" would become an attribute of "tenant".

    (D4 originally said "shorter than the class". That rule is degenerate:
    most class candidates in a real corpus are single-token, so a strict
    length comparison makes attributes structurally impossible for them. The
    containment rule above is what that constraint was actually for. See D4.)

    Keys and values are aggregation keys (D3a); callers map them to surface
    forms for output.
    """
    hits: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for doc, _original in _analyze(sentences):
        keyed = [
            (_key(doc[s : e + 1].text), s, e) for s, e in _indexed_term_spans(doc)
        ]
        chunks = [(c.start, c.end) for c in doc.noun_chunks]

        def record(cls_key: str, attr_key: str) -> None:
            if attr_key not in all_terms or attr_key in class_candidates:
                return
            if attr_key == cls_key:
                return
            if _contains_subsequence(
                tuple(attr_key.split()), tuple(cls_key.split())
            ):
                return
            hits[cls_key][attr_key] += 1

        for cls_key, start, end in keyed:
            if cls_key not in class_candidates:
                continue
            nxt = end + 1
            # "T of C" -- step back over any determiner ("the start date OF THE
            # lease"), otherwise the pattern only ever fires on bare noun pairs.
            prep = start - 1
            while prep >= 0 and doc[prep].pos_ == "DET":
                prep -= 1
            has_of = prep >= 1 and doc[prep].text == "of"

            for attr_key, a_start, a_end in keyed:
                # possessive / colon: C 's T, C : T
                if (
                    nxt < len(doc)
                    and doc[nxt].text in ("'s", "’s", ":")
                    and a_start == nxt + 1
                ):
                    record(cls_key, attr_key)
                # descriptive compound: C T inside one noun chunk
                elif a_start == nxt and any(
                    cs <= start and a_end < ce for cs, ce in chunks
                ):
                    record(cls_key, attr_key)
                # prepositional: T of C
                elif has_of and a_end == prep - 1:
                    record(cls_key, attr_key)

    return dict(hits)


# ---------------------------------------------------------------------------
# Relations (D5)
# ---------------------------------------------------------------------------

def extract_svo_triples(
    sentences: list[str], class_candidates: set[str]
) -> list[tuple[str, str, str]]:
    """(subject, verb_lemma, object) triples where both endpoints are class
    candidates and the verb is not a copula (D5).

    `class_candidates` holds aggregation keys (D3a); returned subjects and
    objects are those same keys, so callers can map them onto emitted class
    names. The verb is spaCy's lemma, returned as produced -- no cleaning.

    An endpoint token is resolved to the longest candidate term ending at that
    token, so a multi-word class candidate is recognised from the single head
    token the parser hands back.
    """
    triples: list[tuple[str, str, str]] = []
    for doc, _original in _analyze(sentences):
        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            if token.lemma_.casefold() in _COPULA_LEMMAS:
                continue
            subjects = [c for c in token.children if c.dep_ in _SUBJECT_DEPS]
            objects = [c for c in token.children if c.dep_ in _OBJECT_DEPS]
            # An object of a preposition hangs off the preposition, not the verb.
            for child in token.children:
                if child.dep_ == "prep":
                    objects.extend(g for g in child.children if g.dep_ in _OBJECT_DEPS)
            for subj in subjects:
                subj_term = _resolve_candidate(doc, subj, class_candidates)
                if subj_term is None:
                    continue
                for obj in objects:
                    obj_term = _resolve_candidate(doc, obj, class_candidates)
                    if obj_term is None or obj_term == subj_term:
                        continue
                    triples.append((subj_term, token.lemma_, obj_term))
    return triples


def _resolve_candidate(doc: Doc, token, class_candidates: set[str]) -> str | None:
    """Longest candidate term ending at `token`, or None.

    Longest-first so a multi-word candidate wins over its own bare head noun
    when both are in the candidate set.
    """
    for length in range(MAX_TERM_TOKENS, 0, -1):
        start = token.i - length + 1
        if start < 0:
            continue
        key = _key(doc[start : token.i + 1].text)
        if key in class_candidates:
            return key
    return None
